import sys
import os
import glob
import json
from bisect import bisect_right
from html import escape
import ctypes
import math
import sysconfig
import subprocess
import re
import shutil
import collections
import queue
import threading
import time
import numpy as np
import cv2


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
MODELS_DIR = os.path.join(BASE_DIR, "animejanai")
BG_IMAGE_PATH = os.path.join(BASE_DIR, "sfondo.png")
RIFE_MODEL_PATH = os.path.join(MODELS_DIR, "rife49_ensemble_True_scale_1_sim.onnx")
RIFE_SCENE_CUT = 0.10   # differenza media oltre la quale non si interpola (cambio scena)
EARS_IMAGE_PATH = os.path.join(BASE_DIR, "ears.png")
SHADERS_DIR = os.path.join(BASE_DIR, "shaders")   # fragment shader GLSL, attivabili (anche più di uno) dal menu col tasto destro
SHADER_EXTS = (".glsl", ".frag", ".fs")
os.makedirs(MODELS_DIR, exist_ok=True)

AUDIO_SYNC = True
DECODE_QUEUE_SIZE = 4
USE_TENSORRT = True
TRT_FORCE_FP16 = True
TRT_CACHE_DIR = os.path.join(BASE_DIR, "trt_cache")
TORCH_FP16 = True
USE_NATIVE_FILE_DIALOG = False
LOOP_VIDEO = True
HW_DECODE = True                 # decodifica hardware (NVDEC) tramite ffmpeg, se disponibile
HW_DECODE_METHODS = ("cuda",)    # metodi hwaccel di ffmpeg da provare, in ordine
HW_SKIP_MAX = 90                 # seek in avanti entro N frame: si scartano i frame senza riavviare ffmpeg

larghezza_occhi = 500            # larghezza massima (px, da un estremo all'altro dell'arco) di ciascun occhio: seguono la larghezza della finestra fino a questo limite
FRAME_COLOR = "#0f0c12"
MINIMAL_CLOCK = True              # in minimal, senza video e con il mouse lontano dai controlli: menu nascosto e orologio al suo posto
FRAME_BORDER_WIDTH = 10          # spessore del bordo nero della finestra ellittica
ELLIPSE_BAND = 0                 # fascia a tinta unita (colore tema) tra bordo nero e video; 0 = nessuna
EARS_X_FRACTION = 0.27           # posizione orizzontale del centro delle orecchie (frazione della larghezza)
EARS_ROTATION_REF_HEIGHT = 520.0  # altezza finestra (px) a cui le orecchie ruotano per allinearsi del tutto all'ellisse; sotto, la rotazione scala in proporzione all'altezza
HEAD_SHAKE_AMPLITUDE = 6.0       # vibrazione della testa quando si clicca un baffo: ampiezza massima (px)
HEAD_SHAKE_DURATION = 0.45       # durata (s)
HEAD_SHAKE_FREQ = 13.0           # oscillazioni al secondo
WHISKER_PULL_MAX_STRETCH = 1.5   # tenendo premuto e tirando un baffo si allunga fino a questa frazione della sua lunghezza (1.5 = 150%); oltre, la finestra inizia a seguire il cursore
# Le dimensioni salvate dalle versioni a 5 finestre erano quelle del solo rettangolo centrale: gli archi
# lo ingrandivano di questo fattore (più il bordo) su ogni asse. Serve a convertirle alla prima apertura.
LEGACY_SIZE_GROWTH = 1.0 + 0.7 * (math.sqrt(2.0) - 1.0)
MOUTH_STEM_REACHES_NOSE = True   # la linea verticale della bocca sale fino al bordo del naso
WHISKERS_REF_WIDTH = 740   # larghezza finestra a cui i baffi hanno lunghezza 100%
WHISKERS_REF_HEIGHT = 520  # altezza finestra a cui distanza tra i baffi e spessore sono al 100%
# (i due riferimenti valgono per la vecchia finestra rettangolare; la finestra ellittica viene convertita)
WHISKERS_LENGTH_SCALE = 0.6  # i baffi ora partono dal bordo e non più dal bordo del video: accorciati di conseguenza
WHISKERS_MIN_LENGTH_SCALE = 0.30     # minimo della scala di lunghezza (1.0 = riferimento): a finestra molto stretta i baffi non si accorciano oltre
WHISKERS_MIN_SPACING_SCALE = 0.45    # minimo della scala verticale (distanza tra i baffi): a finestra molto bassa non si avvicinano oltre
WHISKERS_MIN_THICKNESS_SCALE = 0.60  # minimo dello spessore (1.0 = LINE_W/GLOW_W/DOT_R): a finestra molto bassa le linee non si assottigliano oltre

UI_FONT_FAMILY = "Nunito"
UI_FONT_FALLBACKS = ("Nunito", "Nunito Sans", "Quicksand", "Comfortaa", "sans-serif")

FREE_WINDOW_MOVE = True


def _config_bool_precoce(chiave, default):
    """Legge un booleano dal config prima della creazione dei widget (le impostazioni vere si caricano piu' tardi)."""
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f: return bool(json.load(f).get(chiave, default))
    except Exception: return default


# Sempre in primo piano (spunta nel menu): su Linux la finestra senza window manager (override-redirect) sta
# nel livello piu' alto e copre le altre app. Togliendo la spunta diventa una finestra normale, gestita dal WM.
ALWAYS_ON_TOP = _config_bool_precoce("always_on_top", True)
BYPASS_WM = FREE_WINDOW_MOVE and sys.platform.startswith("linux") and ALWAYS_ON_TOP


def _imposta_bypass_overlay(w, on):
    """Orecchie e baffi (finestre separate): con o senza flag di bypass del window manager."""
    flags = Qt.FramelessWindowHint | Qt.WindowDoesNotAcceptFocus | Qt.Tool
    if on: flags |= Qt.X11BypassWindowManagerHint
    visibile = w.isVisible()
    w.setWindowFlags(flags)
    if visibile: w.show()

if USE_NATIVE_FILE_DIALOG:
    os.environ["QT_QPA_PLATFORMTHEME"] = "xdgdesktopportal"

site_packages = sysconfig.get_path('purelib')
nvidia_lib_dirs = glob.glob(os.path.join(site_packages, 'nvidia', '*', 'lib'))
if USE_TENSORRT:
    nvidia_lib_dirs += glob.glob(os.path.join(site_packages, 'tensorrt_libs'))

for lib_dir in nvidia_lib_dirs:
    for so_file in glob.glob(os.path.join(lib_dir, '*.so*')):
        try:
            ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass

import onnxruntime as ort
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QCheckBox, QStatusBar, QSlider, QStyle, QMessageBox,
    QFileIconProvider, QTreeView, QToolButton, QSplitter, QColorDialog, QSizePolicy
)
from PyQt5.QtGui import (
    QImage, QPixmap, QKeyEvent, QMouseEvent, QWheelEvent,
    QPainter, QColor, QPen, QBrush, QTransform, QRegion, QPainterPath,
    QPainterPathStroker, QIcon, QFont, QCursor, QFontMetrics, QLinearGradient, QPolygon,
    QTextDocument, QTextOption
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QUrl, QPoint, QPointF, QRect, QRectF, QSize, QEvent,
    QFileInfo, QStandardPaths, QStorageInfo, QTranslator, QLocale, QLibraryInfo, QMimeDatabase, QTime
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
try:
    from PyQt5.QtWidgets import QOpenGLWidget
    from PyQt5.QtGui import QSurfaceFormat
except Exception:
    QOpenGLWidget = None
    QSurfaceFormat = None

TORCH_AVAILABLE = False
TORCH_CUDA = False
try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_CUDA = torch.cuda.is_available()
    if TORCH_CUDA:
        torch.backends.cudnn.benchmark = True
except ImportError:
    pass

SPANDREL_AVAILABLE = False
try:
    from spandrel import ModelLoader
    SPANDREL_AVAILABLE = True
except ImportError:
    pass

MODERNGL_AVAILABLE = False
try:
    import moderngl
    MODERNGL_AVAILABLE = True
except ImportError:
    pass

def format_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0: return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

_UI_FONT_RESOLVED = None
def resolve_ui_font_family() -> str:
    global _UI_FONT_RESOLVED
    if _UI_FONT_RESOLVED is None: _UI_FONT_RESOLVED = UI_FONT_FAMILY
    return _UI_FONT_RESOLVED

def build_font_stack() -> str:
    seen, out = set(), []
    for fam in (UI_FONT_FAMILY,) + UI_FONT_FALLBACKS:
        key = fam.lower()
        if key not in seen:
            seen.add(key)
            out.append(fam)
    return ", ".join(out)

def _bean_qss_body(name: str, font_stack: str, colors: dict, size_rule: str, press_rule: str) -> str:
    base = (
        f"QPushButton#{name} {{ "
        f"font-family: {font_stack}; font-size: 13px; font-weight: bold; "
        f"color: {colors['text']}; background-color: qlineargradient("
        f"x1:0, y1:0, x2:0, y2:1, {colors['bg']}); "
        f"border: 2px solid {colors['border']}; " + size_rule
    )
    hover = (
        f"QPushButton#{name}:hover {{ "
        f"background-color: qlineargradient("
        f"x1:0, y1:0, x2:0, y2:1, {colors['hover']}); "
        f"border: 2px solid {colors['border_hover']}; }} "
    )
    pressed = (
        f"QPushButton#{name}:pressed {{ "
        f"background-color: qlineargradient("
        f"x1:0, y1:0, x2:0, y2:1, {colors['pressed']}); "
        + press_rule + f" color: {colors['text_pressed']}; }} "
    )
    disabled = (
        f"QPushButton#{name}:disabled {{ "
        f"background-color: {colors['disabled_bg']}; color: {colors['disabled_text']}; "
        f"border: 2px solid {colors['disabled_border']}; }} "
    )
    return base + hover + pressed + disabled

PINK_BEAN_COLORS = {
    "text": "#40213a",
    "bg": "stop:0 #ffd6ef, stop:0.55 #f5c2e7, stop:1 #eaa9d8",
    "border": "#11111b",
    "hover": "stop:0 #ffe4f6, stop:0.55 #fbd0ee, stop:1 #f2b6e2",
    "border_hover": "#2a1020",
    "pressed": "stop:0 #dba7cd, stop:1 #f0bfe4",
    "text_pressed": "#2b1426",
    "disabled_bg": "#3a3140",
    "disabled_text": "#6c7086",
    "disabled_border": "#241d2b",
}

DARK_BEAN_COLORS = {
    "text": "#cdd6f4",
    "bg": "stop:0 #2b3a67, stop:0.55 #1f2b4d, stop:1 #16203a",
    "border": "#000000",
    "hover": "stop:0 #38507f, stop:0.55 #2b3a67, stop:1 #1f2b4d",
    "border_hover": "#1a1a1a",
    "pressed": "stop:0 #16203a, stop:1 #2b3a67",
    "text_pressed": "#a5b4fc",
    "disabled_bg": "#3a3140",
    "disabled_text": "#6c7086",
    "disabled_border": "#241d2b",
}

def make_bean_qss(name: str, font_stack: str, compact: bool = False) -> str:
    if compact:
        size_rule = ("border-radius: 14px; padding: 2px; "
                      "min-width: 28px; max-width: 34px; min-height: 28px; } ")
    else:
        size_rule = ("border-radius: 19px; "
                      "padding: 6px 10px; min-width: 40px; max-width: 170px; min-height: 22px; } ")
    press_rule = ("border-radius: 16px; padding-top: 4px; padding-bottom: 0px; "
                  if compact else
                  "border-radius: 22px; padding-top: 8px; padding-bottom: 4px; ")
    return _bean_qss_body(name, font_stack, PINK_BEAN_COLORS, size_rule, press_rule)

def make_dark_bean_qss(name: str, font_stack: str) -> str:
    size_rule = ("border-radius: 19px; padding: 6px 10px; "
                  "min-width: 40px; max-width: 170px; min-height: 22px; } ")
    press_rule = "border-radius: 22px; padding-top: 8px; padding-bottom: 4px; "
    body = _bean_qss_body(name, font_stack, DARK_BEAN_COLORS, size_rule, press_rule)
    return (body.replace("border: 2px solid #000000;", "border: 10px solid #000000;")
                .replace("border: 2px solid #1a1a1a;", "border: 10px solid #1a1a1a;"))

FLAT_ICON_QSS = (
    "QPushButton#flatIcon {"
    " background-color: rgba(49, 50, 68, 0.75); color: #cdd6f4;"
    " border: 1px solid #45475a; border-radius: 8px;"
    " font-size: 15px; padding: 0px; }"
    "QPushButton#flatIcon:hover { background-color: #313244; border-color: #fab387; }"
    "QPushButton#flatIcon:pressed { background-color: #45475a; }"
)

def make_main_style(bg_hex: str) -> str:
    font_stack = build_font_stack()
    return (
        f"QMainWindow {{ background-color: {bg_hex}; color: #cdd6f4; "
        f"font-family: {font_stack}; }} "
        f"QWidget {{ font-family: {font_stack}; }} "
        "QWidget#controlsPanel { background-color: transparent; border: none; } "
        "QLabel { color: #cdd6f4; font-size: 12px; } "
        "QPushButton { background-color: rgba(49, 50, 68, 0.85); color: #cdd6f4; border: 1px solid #45475a; border-radius: 6px; padding: 5px 8px; font-weight: bold; font-size: 12px; } "
        "QPushButton:hover { background-color: #fab387; color: #11111b; } "
        + make_bean_qss("pinkBean", font_stack)
        + make_bean_qss("pinkBeanCompact", font_stack, compact=True)
        + make_dark_bean_qss("darkBean", font_stack)
        + FLAT_ICON_QSS +
        "QCheckBox { color: #cdd6f4; font-weight: bold; font-size: 12px; } "
        f"QStatusBar {{ background-color: {bg_hex}; color: #a6adc8; border-top: 1px solid #25202b; }} "
        "QSlider#volumeSlider::groove:horizontal { height: 5px; background: rgba(49, 50, 68, 0.80); border-radius: 2px; } "
        "QSlider#volumeSlider::sub-page:horizontal { background: #a6e3a1; border-radius: 2px; } "
        "QSlider#volumeSlider::handle:horizontal { background: #cdd6f4; border: 1px solid #a6e3a1; width: 12px; margin-top: -4px; margin-bottom: -4px; border-radius: 6px; } "
        "QSlider#volumeSlider::handle:horizontal:hover { background: #ffffff; } "
    )

def make_osd_style(r: int, g: int, b: int) -> str:
    return (
        f"background-color: rgba({r}, {g}, {b}, 0.94); "
        "color: #fab387; "
        "border: 1px solid #fab387; "
        "border-radius: 6px; "
        "padding: 10px 14px; "
        "font-family: monospace; "
        "font-weight: bold; "
        "font-size: 13px; "
    )

VIDEO_EXTS = {"mp4", "mkv", "avi", "webm", "mov", "m4v", "ts", "flv", "wmv", "mpg", "mpeg"}
AUDIO_EXTS = {"mp3", "wav", "flac", "m4a", "aac", "ogg", "oga", "opus", "wma", "mka", "aif", "aiff"}
MODEL_EXTS = {"onnx", "pt", "pth"}
VIDEO_FILTERS = [
    "Video e audio (" + "  ".join("*." + e for e in sorted(VIDEO_EXTS | AUDIO_EXTS)) + ")",
    "Video (" + "  ".join("*." + e for e in sorted(VIDEO_EXTS)) + ")",
    "Audio (" + "  ".join("*." + e for e in sorted(AUDIO_EXTS)) + ")",
    "Tutti i file (*)",
]
MODEL_FILTERS = [
    "Tutti i Modelli (*.onnx *.pt *.pth)",
    "Modelli ONNX (*.onnx)",
    "Modelli PyTorch (*.pt *.pth)",
    "Tutti i file (*)",
]

STYLE_DIALOG = (
    "QDialog { background-color: #0f0c12; } "
    "QWidget { color: #cdd6f4; font-size: 13px; } "
    "QLabel { color: #a6adc8; } "
    "QLineEdit, QComboBox { background-color: #211c28; color: #cdd6f4; border: 1px solid #45475a; border-radius: 6px; padding: 5px 8px; "
    "selection-background-color: #fab387; selection-color: #11111b; } "
    "QLineEdit:focus, QComboBox:focus { border: 1px solid #fab387; } "
    "QComboBox QAbstractItemView { background-color: #211c28; color: #cdd6f4; border: 1px solid #45475a; "
    "selection-background-color: #fab387; selection-color: #11111b; outline: 0; } "
    "QPushButton { background-color: rgba(49, 50, 68, 0.85); color: #cdd6f4; border: 1px solid #45475a; border-radius: 6px; padding: 6px 18px; font-weight: bold; } "
    "QPushButton:hover, QPushButton:default:hover { background-color: #fab387; color: #11111b; } "
    "QPushButton:default { border: 1px solid #fab387; } "
    "QPushButton:disabled { color: #6c7086; border: 1px solid #313244; } "
    "QToolButton { background-color: transparent; border: 1px solid transparent; border-radius: 6px; padding: 2px 8px; "
    "font-size: 17px; font-weight: bold; } "
    "QToolButton:hover { background-color: #45475a; } "
    "QToolButton:checked { background-color: #313244; border: 1px solid #fab387; } "
    "QToolButton:disabled { background-color: transparent; } "
    "QListView, QTreeView { background-color: #0b090e; alternate-background-color: #131018; color: #cdd6f4; "
    "border: 1px solid #313244; border-radius: 6px; outline: 0; } "
    "QListView::item { padding: 3px 4px; border-radius: 4px; } "
    "QTreeView::item { padding: 3px 0; } "
    "QListView::item:hover, QTreeView::item:hover { background-color: #25202c; } "
    "QListView::item:selected, QTreeView::item:selected { background-color: #fab387; color: #11111b; } "
    "QHeaderView { background-color: #0f0c12; border: none; } "
    "QHeaderView::section { background-color: #0f0c12; color: #a6adc8; border: none; border-bottom: 1px solid #45475a; "
    "padding: 5px 8px; font-weight: bold; } "
    "QScrollBar:vertical { background: transparent; width: 10px; margin: 0; } "
    "QScrollBar::handle:vertical { background: #45475a; border-radius: 5px; min-height: 30px; } "
    "QScrollBar::handle:vertical:hover { background: #fab387; } "
    "QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; } "
    "QScrollBar::handle:horizontal { background: #45475a; border-radius: 5px; min-width: 30px; } "
    "QScrollBar::handle:horizontal:hover { background: #fab387; } "
    "QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; } "
    "QScrollBar::add-page, QScrollBar::sub-page { background: transparent; } "
    "QSplitter::handle { background-color: #313244; } "
    "QMenu { background-color: #211c28; color: #cdd6f4; border: 1px solid #45475a; padding: 4px; } "
    "QMenu::item { padding: 6px 18px; border-radius: 4px; } "
    "QMenu::item:selected { background-color: #fab387; color: #11111b; } "
    + make_bean_qss("pinkBean", build_font_stack())
    + make_bean_qss("pinkBeanCompact", build_font_stack(), compact=True)
    + make_dark_bean_qss("darkBean", build_font_stack())
    + FLAT_ICON_QSS
)

def apply_system_icon_theme():
    if QIcon.hasThemeIcon("folder"): return
    for schema in ("org.cinnamon.desktop.interface", "org.gnome.desktop.interface"):
        try:
            out = subprocess.run(
                ["gsettings", "get", schema, "icon-theme"], capture_output=True, text=True, timeout=1.5
            ).stdout.strip().strip("'\"")
        except Exception: continue
        if out:
            QIcon.setThemeName(out)
            if QIcon.hasThemeIcon("folder"): return

class NekoIconProvider(QFileIconProvider):
    def __init__(self):
        super().__init__()
        style = QApplication.style()
        def themed(names, fallback):
            for name in names:
                icon = QIcon.fromTheme(name)
                if not icon.isNull(): return icon
            return style.standardIcon(fallback)
        self._dir = themed(["folder"], QStyle.SP_DirIcon)
        self._drive = themed(["drive-harddisk"], QStyle.SP_DriveHDIcon)
        self._video = themed(["video-x-generic"], QStyle.SP_FileIcon)
        self._audio = themed(["audio-x-generic"], QStyle.SP_FileIcon)
        self._model = themed(["package-x-generic", "application-x-executable"], QStyle.SP_FileIcon)
        self._file = themed(["text-x-generic", "application-x-executable"], QStyle.SP_FileIcon)
    def icon(self, arg):
        if isinstance(arg, QFileInfo):
            if arg.isDir(): return self._dir
            ext = arg.suffix().lower()
            if ext in VIDEO_EXTS: return self._video
            if ext in AUDIO_EXTS: return self._audio
            if ext in MODEL_EXTS: return self._model
            return self._file
        if arg == QFileIconProvider.Folder: return self._dir
        if arg == QFileIconProvider.Drive: return self._drive
        if arg == QFileIconProvider.File: return self._file
        return super().icon(arg)

_TOOL_GLYPHS = {
    "backButton": "\u2190", "forwardButton": "\u2192", "toParentButton": "\u2191",
    "newFolderButton": "\uff0b", "listModeButton": "\u2630", "detailModeButton": "\u2637",
}

def open_file_dialog(parent, title, start_dir, filters, places, icons, select=None):
    dlg = QFileDialog(parent, title, start_dir)
    dlg.setOption(QFileDialog.DontUseNativeDialog, True)
    dlg.setAcceptMode(QFileDialog.AcceptOpen)
    dlg.setFileMode(QFileDialog.ExistingFile)
    dlg.setViewMode(QFileDialog.Detail)
    dlg.setNameFilters(filters)
    dlg.setSidebarUrls([QUrl.fromLocalFile(p) for p in places])
    dlg.setIconProvider(icons)
    dlg.setStyleSheet(STYLE_DIALOG)
    dlg.resize(1000, 620)
    if BYPASS_WM:
        dlg.setWindowFlags(dlg.windowFlags() | Qt.X11BypassWindowManagerHint)
        dlg.setStyleSheet(STYLE_DIALOG + " QFileDialog { border: 1px solid #45475a; }")
        screen = QApplication.screenAt(parent.geometry().center()) or QApplication.primaryScreen()
        dlg.move(screen.availableGeometry().center() - QPoint(500, 310))
        def to_front():
            dlg.raise_()
            dlg.activateWindow()
        QTimer.singleShot(50, to_front)
    if select and os.path.isfile(select) and os.path.dirname(os.path.abspath(select)) == os.path.abspath(start_dir):
        dlg.selectFile(select)
    def tune():
        tree = dlg.findChild(QTreeView, "treeView")
        if tree is not None:
            tree.setAlternatingRowColors(True)
            tree.header().setStretchLastSection(True)
            for col, width in ((0, 380), (1, 100), (2, 110), (3, 170)):
                tree.setColumnWidth(col, width)
        split = dlg.findChild(QSplitter, "splitter")
        if split is not None: split.setSizes([200, 800])
        for name, glyph in _TOOL_GLYPHS.items():
            btn = dlg.findChild(QToolButton, name)
            if btn is not None:
                btn.setIcon(QIcon())
                btn.setText(glyph)
    QTimer.singleShot(0, tune)
    path = ""
    if dlg.exec_():
        files = dlg.selectedFiles()
        path = files[0] if files else ""
    dlg.deleteLater()
    return path


class CatEar(QWidget):
    BASE_WIDTH = 340
    BASE_HEIGHT = 310
    BOX_WIDTH = 560     # finestra rettangolare: larga e con margine sotto l'ancora, cosi' la base ruotata non viene tagliata
    BOX_HEIGHT = 500
    def __init__(self, is_left=True, parent=None):
        flags = Qt.FramelessWindowHint | Qt.WindowDoesNotAcceptFocus | Qt.Tool
        if BYPASS_WM: flags |= Qt.X11BypassWindowManagerHint
        super().__init__(parent, flags)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.is_left = is_left
        self.setFixedSize(self.BOX_WIDTH, self.BOX_HEIGHT)
        self.flare = 0.0
        self.squash = 0.0
        self.ear_w = float(self.BASE_WIDTH)
        self.ear_h = float(self.BASE_HEIGHT)
        self.angle = 0.0        # inclinazione dell'asse verticale dell'orecchio (gradi)
        self.base_angle = 0.0   # inclinazione della base, uguale alla corda sull'ellisse (gradi)
        self.anchor_x = self.BOX_WIDTH / 2.0
        self.anchor_y = 380.0
        self.texture_pixmap = None
        if os.path.isfile(EARS_IMAGE_PATH):
            raw_pixmap = QPixmap(EARS_IMAGE_PATH)
            if not raw_pixmap.isNull():
                if self.is_left: self.texture_pixmap = raw_pixmap
                else: self.texture_pixmap = raw_pixmap.transformed(QTransform().scale(-1, 1))
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._on_anim_tick)
        self.anim_step = 0
        self.anim_total_steps = 18
        self._unit_pts = None
    def _opaque_unit_points(self):
        # Pixel opachi della texture in coordinate normalizzate (0..1), calcolati una sola volta
        if self._unit_pts is None:
            self._unit_pts = False
            pm = self.texture_pixmap
            if pm is not None and not pm.isNull():
                if pm.width() > 256: pm = pm.scaledToWidth(256, Qt.SmoothTransformation)
                img = pm.toImage().convertToFormat(QImage.Format_ARGB32)
                w, h = img.width(), img.height()
                ptr = img.constBits()
                ptr.setsize(img.byteCount())
                arr = np.frombuffer(ptr, np.uint8).reshape(h, img.bytesPerLine() // 4, 4)[:, :w]
                ys, xs = np.nonzero(arr[:, :, 3] > 24)
                if len(xs):
                    self._unit_pts = ((xs + 0.5) / w).astype(np.float32), ((ys + 0.5) / h).astype(np.float32)
        return self._unit_pts or None
    def base_corners_unit(self):
        # Angoli VISIBILI della base in coordinate normalizzate (u1, u2, vb): estremi sinistro/destro delle righe
        # opache più in basso della texture (già specchiata per l'orecchio destro). Senza texture: angoli del rettangolo.
        pts = self._opaque_unit_points()
        if pts is None: return 0.0, 1.0, 1.0
        u, v = pts
        vb = float(v.max())
        sel = v >= vb - 0.02
        return float(u[sel].min()), float(u[sel].max()), vb
    def top_reach(self, angle: float, ear_w: float, ear_h: float, base_angle: float = None) -> float:
        # Distanza in px tra l'ancora e il punto più alto dell'orecchio (stessa trasformazione del paintEvent:
        # asse x lungo la base inclinata di base_angle, asse y inclinato di angle)
        if base_angle is None: base_angle = angle
        sp = math.sin(math.radians(base_angle))
        ca = math.cos(math.radians(angle))
        pts = self._opaque_unit_points()
        if pts is None:
            return ear_h * ca + (ear_w / 2.0) * abs(sp)
        u, v = pts
        x = (u - 0.5) * ear_w
        y = (v - 1.0) * ear_h
        return float(-(x * sp + y * ca).min())
    def trigger_twitch(self):
        self.anim_step = 0
        self.anim_timer.start(16)
    def _on_anim_tick(self):
        self.anim_step += 1
        progress = self.anim_step / self.anim_total_steps
        if progress >= 1.0:
            self.flare = 0.0
            self.anim_timer.stop()
        else:
            if progress < 0.28: self.flare = progress / 0.28
            else: self.flare = 1.0 - ((progress - 0.28) / 0.72)
        self.update()
    def set_ear_config(self, angle: float, width: float, height: float, base_angle: float = None):
        self.angle = angle
        self.base_angle = angle if base_angle is None else base_angle
        self.ear_w = width
        self.ear_h = height
        self.update()
    def paintEvent(self, event):
        if self.ear_h <= 5: return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.translate(self.anchor_x, self.anchor_y)
        # la base segue la corda sull'ellisse (base_angle) mentre l'asse verticale dell'orecchio si inclina solo di angle
        pa, aa = math.radians(self.base_angle), math.radians(self.angle)
        painter.setTransform(QTransform(math.cos(pa), math.sin(pa), -math.sin(aa), math.cos(aa), 0.0, 0.0), True)
        tilt = (0.24 * self.squash + 0.18 * self.flare)
        if tilt > 0.001:
            pivot_x = (self.ear_w / 2.0) if self.is_left else (-self.ear_w / 2.0)
            painter.translate(pivot_x, 0)
            shear_factor = -tilt if self.is_left else tilt
            painter.shear(shear_factor, 0.0)
            painter.translate(-pivot_x, 0)
        draw_rect = QRectF(-self.ear_w / 2.0, -self.ear_h, self.ear_w, self.ear_h)
        if self.texture_pixmap and not self.texture_pixmap.isNull():
            painter.drawPixmap(draw_rect.toRect(), self.texture_pixmap)
        else:
            painter.setPen(QPen(QColor("#543b68"), 2))
            painter.setBrush(QBrush(QColor(15, 12, 18, 220)))
            painter.drawRoundedRect(draw_rect, 16, 16)

class CatWhiskers(QWidget):
    WIDTH = 340
    HEIGHT = 180
    clicked = pyqtSignal()
    pull_window_moved = pyqtSignal(QPoint)   # spostamento cumulativo (px) che la finestra deve seguire mentre si tira un baffo oltre il limite
    pull_ended = pyqtSignal()
    LINE_W = 7.0    # spessore linea nera (px a finestra di riferimento)
    GLOW_W = 10.0   # spessore alone bianco
    DOT_R = 5.0     # raggio puntino alla base
    def __init__(self, is_left=True, parent=None):
        flags = Qt.FramelessWindowHint | Qt.WindowDoesNotAcceptFocus | Qt.Tool
        if BYPASS_WM: flags |= Qt.X11BypassWindowManagerHint
        super().__init__(parent, flags)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.is_left = is_left
        self._sx = 1.0   # scala orizzontale (lunghezza): proporzionale alla larghezza della finestra
        self._sy = 1.0   # scala verticale (distanza tra i baffi, spessore): proporzionale all'altezza della finestra
        self._pad = 0                       # larghezza extra (px) sul lato esterno, solo mentre si tira un baffo
        self._stretch = [1.0, 1.0, 1.0]     # fattore di lunghezza di ciascun baffo (1.0 = a riposo)
        self._pull_idx = None               # baffo afferrato (None = nessuna trazione)
        self._pull_press = None
        self._pull_w = QPointF(0.0, 0.0)
        self._pull_rest = 1.0
        self.setFixedSize(self.WIDTH, self.HEIGHT)
        self.anim_timer = QTimer(self)
        self.anim_timer.setInterval(16)
        self.anim_timer.timeout.connect(self._on_anim_tick)
        self.anim_time = 0.0
        self.is_wiggling = False
        self._update_mask()
    def set_scale(self, sx: float, sy: float):
        sx = max(0.1, float(sx))
        sy = max(0.1, float(sy))
        if abs(sx - self._sx) < 1e-4 and abs(sy - self._sy) < 1e-4: return
        self._sx, self._sy = sx, sy
        self.setFixedSize(self.base_size().width() + self._pad, self.base_size().height())
        self._update_mask()
        self.update()
    def base_size(self) -> QSize:
        # dimensioni a riposo (senza il margine extra della trazione)
        return QSize(max(1, int(round(self.WIDTH * self._sx))), max(1, int(round(self.HEIGHT * self._sy))))
    def origin_offset(self) -> QPoint:
        # di quanto il widget sporge a sinistra rispetto alla sua posizione a riposo (solo baffo sinistro, in trazione)
        return QPoint(self._pad if self.is_left else 0, 0)
    def _set_pad(self, pad: int):
        pad = max(0, int(pad))
        if pad == self._pad: return
        d = pad - self._pad
        self._pad = pad
        self.setFixedSize(self.base_size().width() + pad, self.base_size().height())
        if self.is_left: self.move(self.x() - d, self.y())   # la base dei baffi resta ferma
        self._update_mask()
        self.update()
    def _thickness_scale(self) -> float:
        # lo spessore segue l'altezza della finestra ma non scende sotto il minimo
        return max(self._sy, WHISKERS_MIN_THICKNESS_SCALE)
    def trigger_wiggle(self):
        self.anim_time = 0.0
        self.is_wiggling = True
        self.anim_timer.start()
    def _on_anim_tick(self):
        self.anim_time += 0.075
        amplitude = math.exp(-3.0 * self.anim_time)
        if amplitude < 0.008 or self.anim_time > 1.8:
            self.is_wiggling = False
            self.anim_time = 0.0
            self.anim_timer.stop()
        self._update_mask()
        self.update()
    def _shapes(self):
        # tracciati dei baffi (coordinate del widget) nello stato di animazione corrente
        sx, sy = self._sx, self._sy
        # Le forme sono definite in coordinate "logiche" 340x180; x scala con la larghezza, y con l'altezza.
        # Si converte ogni punto (senza scalare il painter) così lo spessore delle linee resta uniforme.
        ox = float(self._pad) if self.is_left else 0.0
        def P(x, y): return QPointF(x * sx + ox, y * sy)
        base_x = float(self.WIDTH - 15) if self.is_left else 15.0
        center_y = self.HEIGHT / 2.0
        whisker_configs = [
            {"base_y": center_y - 28, "angle": -12.0, "len": 280.0, "phase": 0.0},
            {"base_y": center_y + 2,  "angle": 2.0,   "len": 305.0, "phase": 0.40},
            {"base_y": center_y + 32, "angle": 17.0,  "len": 275.0, "phase": 0.80},
        ]
        direction = -1.0 if self.is_left else 1.0
        paths = []
        for i, cfg in enumerate(whisker_configs):
            by = cfg["base_y"]
            base_angle = cfg["angle"]
            length = cfg["len"] * self._stretch[i]
            wiggle_deg = 0.0
            if self.is_wiggling:
                damp = math.exp(-3.0 * self.anim_time)
                wiggle_deg = 22.0 * damp * math.sin(20.0 * self.anim_time - cfg["phase"])
            total_deg = base_angle + wiggle_deg
            rad = math.radians(total_deg)
            tip_x = base_x + direction * length * math.cos(rad)
            tip_y = by + length * math.sin(rad) + 24.0
            ctrl1_x = base_x + direction * (length * 0.35) * math.cos(rad)
            ctrl1_y = by + (length * 0.12) * math.sin(rad) - 2.0
            ctrl2_x = base_x + direction * (length * 0.70) * math.cos(rad)
            ctrl2_y = by + (length * 0.55) * math.sin(rad) + 12.0
            path = QPainterPath()
            path.moveTo(P(base_x, by))
            path.cubicTo(P(ctrl1_x, ctrl1_y), P(ctrl2_x, ctrl2_y), P(tip_x, tip_y))
            paths.append(path)
        return paths, whisker_configs, base_x, P
    def _update_mask(self):
        # solo i baffi (con un po' di margine) ricevono il mouse: il resto della finestra resta trasparente ai clic
        paths, cfgs, base_x, P = self._shapes()
        st = QPainterPathStroker()
        st.setWidth(self.GLOW_W * self._thickness_scale() + 8.0)
        st.setCapStyle(Qt.RoundCap)
        st.setJoinStyle(Qt.RoundJoin)
        region = QRegion()
        for pth in paths:
            for poly in st.createStroke(pth).toFillPolygons():
                region = region.united(QRegion(poly.toPolygon(), Qt.WindingFill))
        if region.isEmpty(): region = QRegion(0, 0, 1, 1)
        self.setMask(region)
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # baffo afferrato = quello più vicino al punto premuto
            paths = self._shapes()[0]
            pos = QPointF(event.pos())
            best, idx = None, 0
            for i, pth in enumerate(paths):
                for k in range(41):
                    pt = pth.pointAtPercent(k / 40.0)
                    d = (pt.x() - pos.x()) ** 2 + (pt.y() - pos.y()) ** 2
                    if best is None or d < best: best, idx = d, i
            self._pull_idx = idx
            self._pull_rest = max(1.0, paths[idx].length())
            self._pull_press = QPoint(event.globalPos())
            self._pull_w = QPointF(0.0, 0.0)
            self.clicked.emit()
            event.accept()
        else: event.ignore()
    def mouseMoveEvent(self, event):
        if self._pull_idx is None or not (event.buttons() & Qt.LeftButton):
            event.ignore()
            return
        # il baffo si allunga con la distanza del cursore dal punto afferrato, fino a WHISKER_PULL_MAX_STRETCH;
        # oltre il limite la finestra segue il cursore (guinzaglio: non torna indietro se il cursore rientra)
        limit = (WHISKER_PULL_MAX_STRETCH - 1.0) * self._pull_rest
        cur = event.globalPos()
        rel = QPointF(cur.x() - self._pull_press.x() - self._pull_w.x(),
                      cur.y() - self._pull_press.y() - self._pull_w.y())
        r = math.hypot(rel.x(), rel.y())
        if r > limit and r > 0.0:
            k = (r - limit) / r
            self._pull_w = QPointF(self._pull_w.x() + rel.x() * k, self._pull_w.y() + rel.y() * k)
            r = limit
        if r > 2.0 and self._pad == 0:
            # il widget si allarga sul lato esterno per contenere il baffo allungato
            ext = max(0.0, 305.0 * WHISKER_PULL_MAX_STRETCH - (self.WIDTH - 15))
            self._set_pad(int(math.ceil(ext * self._sx)) + 8)
        self._stretch[self._pull_idx] = 1.0 + r / self._pull_rest
        self._update_mask()
        self.update()
        w = QPoint(int(round(self._pull_w.x())), int(round(self._pull_w.y())))
        if not w.isNull(): self.pull_window_moved.emit(w)
        event.accept()
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._pull_idx is not None:
            self._pull_idx = None
            self._stretch = [1.0, 1.0, 1.0]
            self._set_pad(0)
            self._update_mask()
            self.update()
            self.trigger_wiggle()   # scatto elastico al rilascio
            self.pull_ended.emit()
            event.accept()
        else: event.ignore()
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        sy = self._thickness_scale()
        paths, whisker_configs, base_x, P = self._shapes()
        pen_glow = QPen(QColor(255, 255, 255, 120), self.GLOW_W * sy, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen_glow)
        for p in paths: painter.drawPath(p)
        pen_black = QPen(QColor(10, 10, 14, 255), self.LINE_W * sy, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen_black)
        for p in paths: painter.drawPath(p)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(10, 10, 14, 255)))
        for cfg in whisker_configs:
            painter.drawEllipse(P(base_x, cfg["base_y"]), self.DOT_R * sy, self.DOT_R * sy)

def legacy_to_ellipse_size(w: int, h: int):
    """Dimensioni del vecchio rettangolo centrale -> dimensioni della finestra ellittica con la stessa sagoma."""
    return (int(round(w * LEGACY_SIZE_GROWTH + 2 * FRAME_BORDER_WIDTH)),
            int(round(h * LEGACY_SIZE_GROWTH + 2 * FRAME_BORDER_WIDTH)))

def ellipse_region(w: int, h: int, inset: int = 0) -> QRegion:
    """Regione ellittica inscritta nel rettangolo w x h, ridotta di `inset` px per lato."""
    r = QRect(inset, inset, int(w) - 2 * inset, int(h) - 2 * inset)
    if r.width() <= 0 or r.height() <= 0: return QRegion()
    return QRegion(r, QRegion.Ellipse)

def ellipse_half_width(w: float, h: float, y: float, inset: float = 0.0) -> float:
    """Semi-larghezza dell'ellisse (rettangolo w x h, ridotto di `inset`) all'ordinata y."""
    a, b = w / 2.0 - inset, h / 2.0 - inset
    if a <= 0 or b <= 0: return 0.0
    t = (y - h / 2.0) / b
    if abs(t) >= 1.0: return 0.0
    return a * math.sqrt(1.0 - t * t)

class EllipseBorder(QWidget):
    """Bordo della finestra ellittica, disegnato sopra il contenuto (trasparente al mouse):
    anello nero esterno + fascia a tinta unita col colore tema. Il bordo interno è antialiasing."""
    def __init__(self, parent, color=FRAME_COLOR):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.color = QColor(color)
    def set_color(self, hex_color: str):
        self.color = QColor(hex_color)
        self.update()
    def paintEvent(self, event):
        w, h = self.width(), self.height()
        bw, band = float(FRAME_BORDER_WIDTH), float(ELLIPSE_BAND)
        if w <= 2 * (bw + band) + 2 or h <= 2 * (bw + band) + 2: return
        def ell(inset):
            p = QPainterPath()
            p.addEllipse(QRectF(inset, inset, w - 2 * inset, h - 2 * inset))
            return p
        outer = QPainterPath()
        outer.addEllipse(QRectF(-1.0, -1.0, w + 2.0, h + 2.0))   # 1 px oltre: la maschera della finestra taglia il bordo esterno
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        # prima colore (bordo + fascia), poi il nero sopra: tra i due non resta nessuna cucitura
        painter.setBrush(QBrush(self.color))
        painter.drawPath(outer.subtracted(ell(bw + band)))
        if bw > 0:
            painter.setBrush(QBrush(QColor("#000000")))
            painter.drawPath(outer.subtracted(ell(bw)))

class ClickableSlider(QSlider):
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            val = QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), event.x(), self.width())
            self.setValue(val)
            self.sliderMoved.emit(val)
        super().mousePressEvent(event)

class MouthVolumeSlider(ClickableSlider):
    PAD = 8
    JUNCTION_Y = 9.0   # quota (px dal bordo alto del widget) del punto centrale della bocca: sopra resta solo il posto per la maniglia
    OFF_COLOR = "#585b70"
    ON_COLOR = "#a6e3a1"
    OFF_ALPHA_REST = 255
    OFF_ALPHA_HOVER = 255

    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.setAttribute(Qt.WA_Hover, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setCursor(Qt.PointingHandCursor)
        self._side_margin = 0
        self.set_target_width(192)

    def set_side_margin(self, m):
        m = max(0, int(m))
        if m != self._side_margin:
            self._side_margin = m
            self.set_target_width(self._mouth_w)

    def _apply_size(self):
        depth = self._depth_for_width(self._mouth_w)
        # il widget parte appena sopra il punto centrale della bocca (prima aveva sopra uno spazio vuoto alto quanto la bocca)
        height = int(round(self.JUNCTION_Y + depth + 10))
        total = self._mouth_w + 2 * self._side_margin
        if self.width() != total or self.height() != height:
            self.setFixedSize(total, height)
            self.updateGeometry()
            self._update_mask()

    @classmethod
    def _depth_for_width(cls, width):
        a = (width - 2 * cls.PAD) / 4.0
        return max(14.0, min(40.0, a * 0.4))

    def set_target_width(self, width):
        self._mouth_w = int(width)
        self._apply_size()

    def _geom(self):
        m = float(self._side_margin)
        x0, x1 = m + float(self.PAD), m + float(self.width() - self._side_margin * 2 - self.PAD)
        cx = (x0 + x1) / 2.0
        a = (x1 - x0) / 4.0
        depth = self._depth_for_width(self._mouth_w)
        y_j = self.JUNCTION_Y
        y_top = y_j - depth
        return x0, x1, cx, a, y_top, y_j, depth

    def _mouth_path(self):
        x0, x1, cx, a, y_top, y_j, depth = self._geom()
        path = QPainterPath()
        path.moveTo(cx, y_j)
        path.arcTo(QRectF(cx, y_j - depth, 2 * a, 2 * depth), 180, 180)
        path.moveTo(cx, y_j)
        path.arcTo(QRectF(x0, y_j - depth, 2 * a, 2 * depth), 0, -180)
        return path

    def _update_mask(self):
        self.clearMask()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_mask()

    def _handle_pos(self):
        x0, x1, cx, a, y_top, y_j, depth = self._geom()
        span = max(1, self.maximum() - self.minimum())
        t = (self.value() - self.minimum()) / span
        hx = x0 + t * (x1 - x0)
        ctr = (cx + a) if hx >= cx else (x0 + a)
        u = (hx - ctr) / a
        hy = y_j + depth * math.sqrt(max(0.0, 1.0 - u * u))
        return hx, hy

    def stem_info(self):
        x0, x1, cx, a, y_top, y_j, depth = self._geom()
        hx, _ = self._handle_pos()
        hover = self.underMouse() or getattr(self, "_dragging", False)
        if hx > cx:
            col = QColor(self.ON_COLOR)
        else:
            col = QColor(self.OFF_COLOR)
            col.setAlpha(self.OFF_ALPHA_HOVER if hover else self.OFF_ALPHA_REST)
        return cx, y_j, col, y_top

    def _value_from_x(self, x):
        x0, x1 = self._geom()[0], self._geom()[1]
        x = max(x0, min(x1, int(x)))
        return QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), int(x - x0), int(x1 - x0))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._set_from_mouse(event.x())
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if getattr(self, "_dragging", False):
            self._set_from_mouse(event.x())
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False
            event.accept()

    def _set_from_mouse(self, x):
        val = self._value_from_x(x)
        if val != self.value():
            self.setValue(val)
            self.sliderMoved.emit(val)

    def enterEvent(self, event):
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        path = self._mouth_path()
        hx, hy = self._handle_pos()
        hover = self.underMouse() or getattr(self, "_dragging", False)
        off = QColor(self.OFF_COLOR)
        off.setAlpha(self.OFF_ALPHA_HOVER if hover else self.OFF_ALPHA_REST)
        pen = QPen(off, 4.4)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        if off.alpha() > 0:
            p.drawPath(path)
        p.save()
        p.setClipRect(QRectF(0, 0, hx, self.height()))
        pen.setColor(QColor(self.ON_COLOR))
        p.setPen(pen)
        p.drawPath(path)
        p.restore()
        p.setPen(QPen(QColor(self.ON_COLOR), 2))
        p.setBrush(QColor("#ffffff" if hover else "#cdd6f4"))
        p.drawEllipse(QPointF(hx, hy), 7.5, 7.5)
        p.end()

class EyeSlider(QSlider):
    """Barra di avanzamento a forma di collina (arco verso l'alto): affiancandone due sembrano due occhi chiusi sorridenti ^ ^."""
    PAD = 8           # margine laterale (spazio per la maniglia): gli estremi dell'arco stanno a PAD dal bordo del widget
    TOP = 9.0         # spazio sopra il culmine dell'arco (per la maniglia)
    BOTTOM = 9.0      # spazio sotto la base dell'arco
    MIN_DEPTH, MAX_DEPTH, DEPTH_RATIO = 20.0, 170.0, 0.35  # altezza della collina = 35% della larghezza dell'arco, limitata tra 20 e 170 px
    GAP_RATIO, GAP_MIN = 0.20, 24                          # distanza tra i due occhi: 10% della larghezza della riga sotto, almeno 24 px
    OFF_COLOR = "#585b70"
    ON_COLOR = "#fab387"
    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.setAttribute(Qt.WA_Hover, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setCursor(Qt.PointingHandCursor)
        self._depth = self.MIN_DEPTH
        self.setFixedHeight(int(round(self.TOP + self._depth + self.BOTTOM)))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._dragging = False
    @classmethod
    def depth_for_width(cls, arc_width):
        """Altezza della collina per un arco largo arc_width px (stessa regola della bocca)."""
        return max(cls.MIN_DEPTH, min(cls.MAX_DEPTH, cls.DEPTH_RATIO * float(arc_width)))
    def set_arc_depth(self, depth):
        """Imposta l'altezza della collina e quindi del widget; True se è cambiata."""
        h = int(round(self.TOP + depth + self.BOTTOM))
        self._depth = float(depth)
        if h == self.height(): return False
        self.setFixedHeight(h)
        self.update()
        return True
    def sizeHint(self): return QSize(110, self.height())
    def minimumSizeHint(self): return QSize(40, self.height())
    def _geom(self):
        x0, x1 = float(self.PAD), float(self.width() - self.PAD)
        a = max(1.0, (x1 - x0) / 2.0)
        y_b = float(self.height()) - self.BOTTOM       # quota della base dell'arco (estremi)
        depth = max(4.0, min(y_b - self.TOP, a * 0.8))  # altezza della collina
        return x0, x1, a, y_b, depth
    def _arch_path(self):
        x0, x1, a, y_b, depth = self._geom()
        path = QPainterPath()
        path.moveTo(x0, y_b)
        path.arcTo(QRectF(x0, y_b - depth, 2 * a, 2 * depth), 180, -180)   # semiellisse superiore, da sinistra a destra
        return path
    def _handle_pos(self):
        x0, x1, a, y_b, depth = self._geom()
        span = max(1, self.maximum() - self.minimum())
        t = (self.value() - self.minimum()) / span
        hx = x0 + t * (x1 - x0)
        u = (hx - (x0 + a)) / a
        return hx, y_b - depth * math.sqrt(max(0.0, 1.0 - u * u))
    def _value_from_x(self, x):
        x0, x1 = self._geom()[0], self._geom()[1]
        x = max(x0, min(x1, float(x)))
        return QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), int(x - x0), max(1, int(x1 - x0)))
    def _set_from_mouse(self, x):
        val = self._value_from_x(x)
        self.setValue(val)
        self.sliderMoved.emit(val)
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self.setSliderDown(True)          # emette sliderPressed
            self._set_from_mouse(event.x())
            event.accept()
        else: event.ignore()
    def mouseMoveEvent(self, event):
        if self._dragging:
            self._set_from_mouse(event.x())
            event.accept()
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self.setSliderDown(False)         # emette sliderReleased
            event.accept()
    def enterEvent(self, event):
        self.update()
        super().enterEvent(event)
    def leaveEvent(self, event):
        self.update()
        super().leaveEvent(event)
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        path = self._arch_path()
        hx, hy = self._handle_pos()
        hover = self.underMouse() or self._dragging
        pen = QPen(QColor(self.OFF_COLOR), 15.0)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)
        p.save()
        p.setClipRect(QRectF(0, 0, hx, self.height()))
        pen.setColor(QColor(self.ON_COLOR))
        p.setPen(pen)
        p.drawPath(path)
        p.restore()
        p.setPen(QPen(QColor(self.ON_COLOR), 2))
        p.setBrush(QColor("#0aa633" if hover else "#0aa633"))
        p.drawEllipse(QPointF(hx, hy), 11.0, 11.0)
        p.end()

class MouthStemOverlay(QWidget):
    """Linea verticale centrale della bocca, ridisegnata da un livello trasparente SEMPRE sopra tutti gli altri widget del
    pannello comandi (e non ritagliato dai contenitori della bocca): niente può più nasconderla."""
    def __init__(self, panel, slider, nose):
        super().__init__(panel)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self._slider, self._nose = slider, nose
        self._key = None
        self._line = None
        self.hide()
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self.sync)
        self._timer.start()
    def sync(self):
        panel, sl, nose = self.parentWidget(), self._slider, self._nose
        if panel is None or not (panel.isVisible() and sl.isVisible()):
            if self.isVisible(): self.hide()
            return
        cx, y_j, col, y_top = sl.stem_info()
        o = sl.mapTo(panel, QPoint(0, 0))
        x = o.x() + cx
        y_junc = o.y() + y_j
        top = o.y() + y_top
        if MOUTH_STEM_REACHES_NOSE and nose.isVisible():
            no = nose.mapTo(panel, QPoint(0, 0))
            rx, ry = (nose.width() - 3.0) / 2.0, (nose.height() - 3.0) / 2.0   # stessa ellisse disegnata dal naso
            dx = x - (no.x() + nose.width() / 2.0)
            if abs(dx) < rx:
                edge = no.y() + nose.height() / 2.0 + ry * math.sqrt(max(0.0, 1.0 - (dx / rx) ** 2))
                if edge < y_junc - 2.0: top = edge + 0.8
        key = (round(x, 1), round(top, 1), round(y_junc, 1), col.rgba())
        if key != self._key:
            self._key = key
            self._line = (x, top, y_junc, col)
            self.setGeometry(int(x) - 6, int(top) - 6, 12, int(math.ceil(y_junc - top)) + 12)
            self.update()
        if not self.isVisible(): self.show()
        kids = [c for c in panel.children() if isinstance(c, QWidget)]
        if kids and kids[-1] is not self: self.raise_()
    def paintEvent(self, event):
        if not self._line: return
        x, y0, y1, col = self._line
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.translate(-self.x(), -self.y())
        pen = QPen(col, 3.2)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawLine(QPointF(x, y0), QPointF(x, y1))
        p.end()

class NosePlayButton(QPushButton):
    FILL_ALPHA = 255      # opacità (0-255) dell'ellisse: abbassala per renderla più trasparente
    def __init__(self, parent=None):
        super().__init__(parent)
        self._playing = True
        self.setFixedSize(64, 40)
        self.setMask(QRegion(0, 0, 64, 40, QRegion.Ellipse))   # fuori dall'ellisse il widget non esiste
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.NoFocus)
        self.setAttribute(Qt.WA_Hover, True)
        self.setToolTip("Play / Pausa (Spazio)")
    def sizeHint(self): return QSize(64, 40)
    def minimumSizeHint(self): return QSize(64, 40)
    def set_playing(self, playing):
        self._playing = bool(playing)
        self.update()
    def hitButton(self, pos):
        r = QRectF(self.rect())
        nx = (pos.x() - r.center().x()) / (r.width() / 2.0)
        ny = (pos.y() - r.center().y()) / (r.height() / 2.0)
        return nx * nx + ny * ny <= 1.0
    def enterEvent(self, event):
        self.update()
        super().enterEvent(event)
    def leaveEvent(self, event):
        self.update()
        super().leaveEvent(event)
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        if not self.isEnabled():
            fill, glyph = QColor("#313244"), QColor("#6c7086")
        elif self.isDown():
            if self._playing: fill, glyph = QColor("#16233f"), QColor("#c3d4ff")
            else: fill, glyph = QColor("#e0a07a"), QColor("#11111b")
        elif self.underMouse():
            if self._playing: fill, glyph = QColor("#2f4d8f"), QColor("#ffffff")
            else: fill, glyph = QColor("#fab387"), QColor("#11111b")
        else:
            if self._playing: fill, glyph = QColor("#1e3a6e"), QColor("#dbe4ff")
            else: fill, glyph = QColor("#f5c2e7"), QColor("#11111b")
        fill.setAlpha(self.FILL_ALPHA)
        p.setPen(Qt.NoPen)            # nessun bordo
        p.setBrush(QBrush(fill))
        p.drawEllipse(rect)
        if self.isEnabled():
            shine = QRectF(rect.left() + rect.width() * 0.20, rect.top() + rect.height() * 0.14,
                           rect.width() * 0.26, rect.height() * 0.18)
            p.setBrush(QColor(255, 255, 255, 95))
            p.drawEllipse(shine)
        cx, cy = rect.center().x(), rect.center().y() + 1
        p.setBrush(QBrush(glyph))
        if self._playing:
            bw, bh, gap = 4.5, 14.0, 4.5
            p.drawRoundedRect(QRectF(cx - gap / 2 - bw, cy - bh / 2, bw, bh), 1.5, 1.5)
            p.drawRoundedRect(QRectF(cx + gap / 2, cy - bh / 2, bw, bh), 1.5, 1.5)
        else:
            path = QPainterPath()
            path.moveTo(cx - 4.5, cy - 8)
            path.lineTo(cx - 4.5, cy + 8)
            path.lineTo(cx + 8.5, cy)
            path.closeSubpath()
            p.drawPath(path)
        p.end()

class CenteredRow(QWidget):
    def __init__(self, left, center, right, spacing=10, clip_right=False, parent=None):
        super().__init__(parent)
        self._l, self._c, self._r, self._sp = left, center, right, spacing
        # clip_right: il gruppo di destra sta sempre a destra del centro; se manca spazio viene tagliato
        # (il testo sparisce) invece di finire sotto/sopra il widget centrale
        self._clip_right = clip_right
        self._l_extra = 0
        for w in (left, center, right): w.setParent(self)
    def _hint(self, minimum):
        get = (lambda w: w.minimumSizeHint()) if minimum else (lambda w: w.sizeHint())
        l, c, r = get(self._l), get(self._c), get(self._r)
        l.setWidth(max(0, l.width() - self._l_extra))   # il margine extra anti-taglio non conta nelle dimensioni
        if minimum: side = max(l.width(), r.width()) // 2
        else: side = max(l.width(), r.width())
        return QSize(2 * side + c.width() + 2 * self._sp, max(l.height(), c.height(), r.height()))
    def sizeHint(self): return self._hint(False)
    def minimumSizeHint(self): return self._hint(True)
    def _place(self):
        w, h = self.width(), self.height()
        c = self._c.sizeHint()
        cw = min(c.width(), max(0, w))
        self._c.setGeometry((w - cw) // 2, (h - c.height()) // 2, cw, c.height())
        side = max(0, (w - cw) // 2 - self._sp)
        rh = self._r.sizeHint().height()
        # disposizione originale: sinistra nello spazio a sinistra del centro (se manca spazio i pulsanti
        # si accavallano), destra a filo del bordo destro
        self._place_left(side, h)
        rw = min(self._r.sizeHint().width(), w)
        if self._clip_right:
            rx = max(w - rw, (w - cw) // 2 + cw + self._sp)   # mai a sinistra del bordo destro del centro
            self._r.setGeometry(rx, (h - rh) // 2, max(0, w - rx), rh)
        else:
            self._r.setGeometry(w - rw, (h - rh) // 2, rw, rh)
        # il centro (play / volume) sta sopra, ma ha una maschera: fuori dalla sua forma
        # si vede e si clicca ciò che sta sotto
        self._c.raise_()
    def _place_left(self, side, h):
        """Il gruppo sinistro si dispone come se avesse larghezza `side` (pulsanti accavallati se stretto),
        ma il suo riquadro si allarga quanto serve perché nessun pulsante venga tagliato."""
        box = self._l
        lay = box.layout()
        bh = box.sizeHint().height()
        y = (h - bh) // 2
        def apply(extra):
            if extra != self._l_extra:
                lay.setContentsMargins(0, 0, extra, 0)
                self._l_extra = extra
            box.setGeometry(0, y, side + extra, bh)
            lay.setGeometry(box.rect())
            lay.activate()
        apply(0)
        kids = [c for c in box.children() if isinstance(c, QWidget) and not c.isHidden()]
        reach = max((c.geometry().right() + 1 for c in kids), default=0)
        if reach > side: apply(reach - side)
    def resizeEvent(self, event):
        self._place()
        super().resizeEvent(event)
    def showEvent(self, event):
        self._place()
        super().showEvent(event)

class VideoDisplayLabel(QLabel):
    double_clicked = pyqtSignal()
    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.double_clicked.emit()
            event.accept()
        else: super().mouseDoubleClickEvent(event)
    def wheelEvent(self, event: QWheelEvent): event.ignore()

if QOpenGLWidget is not None:
    class VideoDisplayGL(QOpenGLWidget):
        """Stessa interfaccia di VideoDisplayLabel, ma il frame viene composto in una superficie OpenGL con
        swap sincronizzato al refresh del monitor (V-Sync): ogni frame compare per intero, senza strappi.
        Un QLabel passa invece dal backing store raster, che scrive i pixel in qualunque momento."""
        double_clicked = pyqtSignal()

        def __init__(self, parent=None):
            super().__init__(parent)
            fmt = QSurfaceFormat(QSurfaceFormat.defaultFormat())
            fmt.setSwapInterval(1)          # 1 = attendi il refresh verticale prima di mostrare il frame
            fmt.setAlphaBufferSize(8)
            self.setFormat(fmt)
            self._pm = None
            self._align = Qt.AlignCenter
            self._bg = QColor(15, 12, 18, 217)
            self._radius = 0.0

        def setAlignment(self, a): self._align = a; self.update()
        def pixmap(self): return self._pm
        def setPixmap(self, pm): self._pm = pm; self.update()
        def clear(self): self._pm = None; self.update()

        def setStyleSheet(self, css):
            # un QOpenGLWidget ignora il foglio di stile: ne leggiamo solo sfondo e raggio e li disegniamo noi
            m = re.search(r"background-color:\s*([^;]+);?", css)
            if m:
                v = m.group(1).strip()
                if v.lower().startswith("rgba"):
                    n = [x.strip() for x in v[v.index("(") + 1:v.rindex(")")].split(",")]
                    self._bg = QColor(int(float(n[0])), int(float(n[1])), int(float(n[2])),
                                      int(round(float(n[3]) * 255)))
                else:
                    self._bg = QColor(v)
            m = re.search(r"border-radius:\s*([\d.]+)", css)
            self._radius = float(m.group(1)) if m else 0.0
            self.update()

        def paintGL(self):
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            p.setCompositionMode(QPainter.CompositionMode_Source)
            p.fillRect(self.rect(), Qt.transparent)
            path = QPainterPath()
            path.addRoundedRect(QRectF(self.rect()), self._radius, self._radius)
            p.fillPath(path, self._bg)
            p.setCompositionMode(QPainter.CompositionMode_SourceOver)
            pm = self._pm
            if pm is not None and not pm.isNull():
                dpr = pm.devicePixelRatio() or 1.0
                pw, ph = pm.width() / dpr, pm.height() / dpr
                x = (self.width() - pw) / 2.0
                y = (self.height() - ph) / 2.0
                p.setClipPath(path)
                p.drawPixmap(QPointF(x, y), pm)
            p.end()

        def mouseDoubleClickEvent(self, event):
            if event.button() == Qt.LeftButton:
                self.double_clicked.emit()
                event.accept()
            else: super().mouseDoubleClickEvent(event)

        def wheelEvent(self, event): event.ignore()

# ---------------------------------------------------------------------------
# File solo audio: schermo nero con i sottotitoli karaoke e una sinusoide la cui ampiezza
# segue il livello in decibel nel tempo (inviluppo calcolato da ffmpeg in un thread).
# ---------------------------------------------------------------------------
ENV_RATE = 50            # campioni di inviluppo al secondo (uno ogni 20 ms)
ENV_SR = 8000            # frequenza di campionamento a cui ffmpeg decodifica per l'analisi
WAVE_FLOOR_DB = -60.0    # sotto questo livello la sinusoide e' piatta
WAVE_SPAN = 8.0          # secondi visibili nella finestra (la testina e' al centro)
WAVE_CYCLES = 18         # oscillazioni della sinusoide visibili in tutta la finestra


def compute_audio_db(path, should_stop=None):
    """Livello RMS in dB (rispetto al fondo scala) ogni 1/ENV_RATE s, o None se ffmpeg fallisce."""
    ff = shutil.which("ffmpeg")
    if not ff: return None
    hop = ENV_SR // ENV_RATE
    proc = subprocess.Popen(
        [ff, "-v", "error", "-nostdin", "-i", path, "-vn", "-ac", "1", "-ar", str(ENV_SR),
         "-f", "s16le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    out, rest = [], b""
    try:
        while True:
            if should_stop and should_stop(): return None
            data = proc.stdout.read(hop * 2 * ENV_RATE * 10)
            if not data: break
            data = rest + data
            n = (len(data) // (2 * hop)) * (2 * hop)
            rest = data[n:]
            if n:
                x = np.frombuffer(data[:n], dtype="<i2").astype(np.float32) / 32768.0
                rms = np.sqrt(np.mean(x.reshape(-1, hop) ** 2, axis=1))
                out.append(20.0 * np.log10(np.maximum(rms, 1e-6)))
    finally:
        if proc.poll() is None: proc.kill()
        try: proc.stdout.close()
        except Exception: pass
        proc.wait()
    if not out: return None
    return np.concatenate(out).astype(np.float32)


class AudioEnvelopeThread(QThread):
    ready = pyqtSignal(str, object)
    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path
        self._stop = False
    def stop(self): self._stop = True
    def run(self):
        try: db = compute_audio_db(self.path, lambda: self._stop)
        except Exception: db = None
        if db is not None and not self._stop:
            self.ready.emit(self.path, db)


class AudioWaveOverlay(QWidget):
    """Sfondo nero con una sinusoide che scorre: ampiezza = livello in dB al tempo corrente.
    A sinistra della testina (gia' riprodotto) e' arancione-rossa, a destra azzurra e piu' tenue."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.NoFocus)
        self._db = None
        self._amp = None
        self._times = None
        self._t = 0.0
        parent.installEventFilter(self)
        self.setGeometry(parent.rect())
        self.hide()

    def eventFilter(self, obj, ev):
        if obj is self.parent() and ev.type() == QEvent.Resize:
            self.setGeometry(self.parent().rect())
        return False

    def set_envelope(self, db):
        if db is None or len(db) == 0:
            self._db = self._amp = self._times = None
            self.update()
            return
        db = np.asarray(db, dtype=np.float32)
        hi = max(float(np.percentile(db, 99.5)), WAVE_FLOOR_DB + 12.0)
        amp = np.clip((db - WAVE_FLOOR_DB) / (hi - WAVE_FLOOR_DB), 0.0, 1.0)
        amp = np.convolve(amp, np.array([0.25, 0.5, 0.25], dtype=np.float32), mode="same")
        self._db = db
        self._amp = amp.astype(np.float32)
        self._times = np.arange(len(db), dtype=np.float32) / float(ENV_RATE)
        self.update()

    def set_time(self, t):
        self._t = float(t)
        if self.isVisible(): self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0))
        w, h = self.width(), self.height()
        if w < 40 or h < 40:
            return
        cx, cy = w / 2.0, h * 0.40
        amp_px = h * 0.20
        lw = max(1.5, h * 0.006)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(255, 255, 255, 28), 1))
        p.drawLine(QPointF(w * 0.04, cy), QPointF(w * 0.96, cy))
        if self._amp is None:
            p.end()
            return

        t = self._t
        step = max(2, int(w / 360))
        xs = np.arange(0, w + 1, step, dtype=np.float64)
        u = (xs - cx) / (w * 0.47)
        taper = np.sqrt(np.clip(1.0 - u * u, 0.0, 1.0))     # niente picchi fuori dall'ellisse
        tx = t + (xs - cx) / w * WAVE_SPAN
        a = np.interp(tx, self._times, self._amp, left=0.0, right=0.0)
        period = WAVE_SPAN / WAVE_CYCLES
        ys = cy - a * amp_px * taper * np.sin(2.0 * np.pi * tx / period)

        path = QPainterPath()
        path.moveTo(float(xs[0]), float(ys[0]))
        for x, y in zip(xs[1:], ys[1:]):
            path.lineTo(float(x), float(y))

        past = QLinearGradient(cx - w * 0.40, 0, cx, 0)
        past.setColorAt(0.0, QColor(255, 180, 84, 40))
        past.setColorAt(0.6, QColor(SUB_SPOKEN))
        past.setColorAt(1.0, QColor(SUB_CURRENT))
        fut = QLinearGradient(cx, 0, w, 0)
        fut.setColorAt(0.0, QColor(130, 170, 220, 170))
        fut.setColorAt(1.0, QColor(130, 170, 220, 30))
        for clip, grad in ((QRectF(0, 0, cx, h), past), (QRectF(cx, 0, w - cx, h), fut)):
            p.save()
            p.setClipRect(clip)
            glow = QPen(QBrush(grad), lw * 4.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            c = QColor(255, 255, 255)
            p.setOpacity(0.18)
            p.setPen(glow)
            p.drawPath(path)
            p.setOpacity(1.0)
            p.setPen(QPen(QBrush(grad), lw, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawPath(path)
            p.restore()

        # testina: linea verticale, punto sulla sinusoide e livello in dB
        a0 = float(np.interp(t, self._times, self._amp, left=0.0, right=0.0))
        y0 = cy - a0 * amp_px * math.sin(2.0 * math.pi * t / period)
        p.setPen(QPen(QColor(255, 255, 255, 70), 1))
        p.drawLine(QPointF(cx, cy - amp_px * 1.1), QPointF(cx, cy + amp_px * 1.1))
        r = max(3.0, h * 0.014)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 42, 42, 90))
        p.drawEllipse(QPointF(cx, y0), r * 1.9, r * 1.9)
        p.setBrush(QColor(SUB_CURRENT))
        p.drawEllipse(QPointF(cx, y0), r, r)

        i = int(min(len(self._db) - 1, max(0, round(t * ENV_RATE))))
        val = float(self._db[i])
        txt = "-inf dB" if val < -80 else "%d dB" % round(val)
        f = QFont(self.font())
        f.setPixelSize(max(10, int(h * 0.034)))
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(255, 255, 255, 150))
        p.drawText(QRectF(cx - 70, cy - amp_px * 1.1 - h * 0.06, 140, h * 0.05), Qt.AlignCenter, txt)
        p.end()


class FFmpegCapture:
    """Sostituto minimale di cv2.VideoCapture: ffmpeg decodifica (NVDEC) e manda frame BGR24 su una pipe."""
    _FAIL_RE = re.compile(r"failed setup|initialisation returned error|cannot load|device creation failed|no device", re.I)

    def __init__(self, path, meta_cap, methods=HW_DECODE_METHODS):
        self.path = path
        self.ffmpeg = shutil.which("ffmpeg")
        if not self.ffmpeg: raise RuntimeError("ffmpeg non trovato nel PATH")
        self._fps = meta_cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._count = meta_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self._w, self._h = self._probe_size(meta_cap)
        if self._w <= 0 or self._h <= 0: raise RuntimeError("dimensioni video non valide")
        self._frame_bytes = self._w * self._h * 3
        self._scratch = np.empty(self._frame_bytes, dtype=np.uint8)
        self._max_skip = max(0, min(HW_SKIP_MAX, int(150e6 // self._frame_bytes)))
        self._proc = None
        self._pos = 0          # posizione pubblica (prossimo frame restituito da read)
        self._dec_pos = 0      # frame già consumati dalla pipe
        self._pending = None   # seek che richiede il riavvio di ffmpeg
        self._prefetch = None
        self._hw_failed = False
        self._msgs = 0
        self._open = False
        self.method = None
        err = "nessun metodo hwaccel"
        for m in methods:
            try:
                self.method = m
                self._start(0)
                first = np.empty((self._h, self._w, 3), dtype=np.uint8)
                ok = self._read_into(first.reshape(-1))
                if ok: time.sleep(0.05)   # lascia arrivare gli avvisi di ffmpeg su stderr
                if ok and not self._hw_failed:
                    self._prefetch = first
                    self._dec_pos = 1
                    self._open = True
                    return
                err = "hwaccel non supportato per questo video" if self._hw_failed else "ffmpeg non ha prodotto frame"
            except Exception as e:
                err = str(e)
            self._kill()
        self.method = None
        raise RuntimeError(err)

    def _probe_size(self, meta_cap):
        w = int(meta_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(meta_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            try:
                out = subprocess.run(
                    [ffprobe, "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=width,height:stream_tags=rotate:stream_side_data=rotation",
                     "-of", "json", self.path],
                    capture_output=True, text=True, timeout=10).stdout
                st = json.loads(out)["streams"][0]
                pw, ph = int(st["width"]), int(st["height"])
                rot = (st.get("tags") or {}).get("rotate")
                for sd in st.get("side_data_list") or []:
                    if "rotation" in sd: rot = sd["rotation"]
                rot = int(float(rot)) if rot is not None else 0
                if abs(rot) % 180 == 90: pw, ph = ph, pw
                if pw > 0 and ph > 0: return pw, ph
            except Exception: pass
        return w, h

    def _start(self, frame_no):
        self._kill()
        cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "warning", "-nostdin"]
        if self.method: cmd += ["-hwaccel", self.method]
        if frame_no > 0: cmd += ["-ss", f"{max(0.0, (frame_no - 0.5) / self._fps):.6f}"]
        cmd += ["-i", self.path, "-map", "0:v:0", "-an", "-sn", "-dn",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        self._hw_failed = False
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=self._frame_bytes * 2)
        self._proc = proc
        self._dec_pos = frame_no
        threading.Thread(target=self._drain_stderr, args=(proc,), daemon=True).start()

    def _drain_stderr(self, proc):
        try:
            for raw in proc.stderr:
                line = raw.decode("utf-8", "replace").strip()
                if not line: continue
                if self._FAIL_RE.search(line) and proc is self._proc: self._hw_failed = True
                if self._msgs < 5:
                    self._msgs += 1
                    print(f"[ffmpeg] {line}")
        except Exception: pass

    def _kill(self):
        p, self._proc = self._proc, None
        if p is None: return
        try: p.kill()
        except Exception: pass
        try: p.stdout.close()
        except Exception: pass
        try: p.wait(timeout=2)
        except Exception: pass

    def _read_into(self, flat):
        p = self._proc
        if p is None or p.stdout is None: return False
        try: return p.stdout.readinto(flat) == flat.size
        except (ValueError, OSError): return False

    def isOpened(self): return self._open

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_FRAMES: return float(self._pos)
        if prop == cv2.CAP_PROP_FPS: return float(self._fps)
        if prop == cv2.CAP_PROP_FRAME_COUNT: return float(self._count)
        if prop == cv2.CAP_PROP_FRAME_WIDTH: return float(self._w)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT: return float(self._h)
        return 0.0

    def set(self, prop, value):
        if prop != cv2.CAP_PROP_POS_FRAMES: return False
        target = max(0, int(value))
        self._pos = target
        if self._pending is not None:
            self._pending = target
        elif self._proc is not None and 0 <= target - self._dec_pos <= self._max_skip:
            pass   # seek breve in avanti: read() scarta i frame intermedi
        else:
            self._pending = target
        return True

    def read(self):
        if not self._open: return False, None
        if self._pending is not None:
            self._start(self._pending)
            self._pending = None
            self._prefetch = None
        if self._prefetch is not None:
            f, self._prefetch = self._prefetch, None
            if self._pos == 0:
                self._pos = 1
                return True, f
        if self._proc is None: return False, None
        while self._dec_pos < self._pos:
            if not self._read_into(self._scratch): return False, None
            self._dec_pos += 1
        frame = np.empty((self._h, self._w, 3), dtype=np.uint8)
        if not self._read_into(frame.reshape(-1)): return False, None
        self._dec_pos += 1
        self._pos += 1
        return True, frame

    def release(self):
        self._open = False
        self._prefetch = None
        self._kill()


def open_video_capture(path):
    """Restituisce (capture, modalità): ffmpeg con decodifica hardware se possibile, altrimenti OpenCV su CPU."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened() or not HW_DECODE:
        return cap, "CPU"
    try:
        hw = FFmpegCapture(path, cap)
    except Exception as e:
        print(f"[Neko Player] Decodifica hardware non disponibile ({e}): uso la CPU.")
        return cap, "CPU"
    cap.release()
    label = {"cuda": "NVDEC"}.get(hw.method, str(hw.method))
    print(f"[Neko Player] Decodifica hardware attiva: {label}")
    return hw, f"GPU ({label})"


_GLSL_COMMENTS = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
_GLSL_VERSION = re.compile(r"^[ \t]*#version[^\n]*\n?", re.M)
_GLSL_SAMPLER = re.compile(r"\buniform\s+(?:(?:lowp|mediump|highp)\s+)?sampler2D\s+(\w+)\s*;")
_GLSL_VARYING = re.compile(r"^[ \t]*(?:in|varying)\s+(?:(?:lowp|mediump|highp)\s+)?(float|vec2|vec3|vec4)\s+(\w+)\s*;", re.M)
# nomi di uniform riconosciuti negli shader "classici" (se dichiarati vengono valorizzati a ogni frame)
_SU_RES = ("iResolution", "u_resolution", "uResolution", "resolution", "u_res", "u_size", "uSize",
           "inputSize", "outputSize", "screenSize", "texSize", "u_texSize")
_SU_TEXEL = ("texelSize", "u_texelSize", "uTexelSize", "u_texel", "pixelSize", "u_pixelSize", "onePixel", "invResolution")
_SU_TIME = ("iTime", "iGlobalTime", "u_time", "uTime", "time", "Time")
_SU_FRAME = ("iFrame", "u_frame", "uFrame", "frame", "frameCount", "FrameCount")
# shader mpv: i punti di aggancio che per noi coincidono col frame RGB corrente (LUMA/CHROMA non sono supportati)
_MPV_HOOKS = {"MAIN", "RGB", "NATIVE", "PREKERNEL", "POSTKERNEL", "SCALED", "OUTPUT"}
_VS_POS = ("#version 330\nin vec2 in_pos;\nout vec2 _pos;\n"
           "void main() { _pos = in_pos * 0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }\n")
_FS_BLIT = ("#version 330\nin vec2 _pos;\nout vec4 _c;\nuniform sampler2D tex;\nuniform float flip;\n"
            "void main() { vec2 p = _pos; if (flip > 0.5) p.y = 1.0 - p.y; _c = texture(tex, p); }\n")
_MPV_BIND_TMPL = """uniform sampler2D @N@_raw;
#define @N@_pos _pos
#define @N@_size vec2(textureSize(@N@_raw, 0))
#define @N@_pt (vec2(1.0) / vec2(textureSize(@N@_raw, 0)))
#define @N@_off vec2(0.0)
#define @N@_mul 1.0
#define @N@_rot mat2(1.0, 0.0, 0.0, 1.0)
#define @N@_map(p) (p)
vec4 @N@_tex(vec2 pos) { return texture(@N@_raw, pos); }
vec4 @N@_texOff(vec2 off) { return @N@_tex(_pos + (vec2(1.0) / vec2(textureSize(@N@_raw, 0))) * off); }
"""
_MPV_GATHER_TMPL = """vec4 @N@_gather(vec2 pos, int c) {
    ivec2 s = textureSize(@N@_raw, 0);
    ivec2 b = ivec2(floor(pos * vec2(s) - 0.5));
    ivec2 lo = clamp(b, ivec2(0), s - 1);
    ivec2 hi = clamp(b + 1, ivec2(0), s - 1);
    return vec4(texelFetch(@N@_raw, ivec2(lo.x, hi.y), 0)[c], texelFetch(@N@_raw, hi, 0)[c],
                texelFetch(@N@_raw, ivec2(hi.x, lo.y), 0)[c], texelFetch(@N@_raw, lo, 0)[c]);
}
"""

def _mpv_rpn(expr, sizes):
    """Valuta un'espressione RPN dei metadati mpv (//!WHEN, //!WIDTH, //!HEIGHT), es. 'MAIN.w 2 *'."""
    ops = {
        "+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
        "/": lambda a, b: a / b if b else 0.0, "%": lambda a, b: math.fmod(a, b) if b else 0.0,
        ">": lambda a, b: float(a > b), "<": lambda a, b: float(a < b),
        ">=": lambda a, b: float(a >= b), "<=": lambda a, b: float(a <= b),
        "==": lambda a, b: float(abs(a - b) < 1e-6), "!=": lambda a, b: float(abs(a - b) >= 1e-6),
        "&&": lambda a, b: float(bool(a) and bool(b)), "||": lambda a, b: float(bool(a) or bool(b)),
    }
    st = []
    def pop():
        if not st: raise ValueError(f"espressione non valida: '{expr}'")
        return st.pop()
    for tok in expr.split():
        if tok in ops:
            b = pop()
            a = pop()
            st.append(ops[tok](a, b))
        elif tok == "!": st.append(float(not pop()))
        elif tok == "?:":
            c = pop()
            b = pop()
            a = pop()
            st.append(b if a else c)
        else:
            name, _, axis = tok.rpartition(".")
            if name in sizes and axis in ("w", "h"): st.append(float(sizes[name][0 if axis == "w" else 1]))
            else:
                try: st.append(float(tok))
                except ValueError: raise ValueError(f"token '{tok}' sconosciuto in '{expr}'")
    if len(st) != 1: raise ValueError(f"espressione non valida: '{expr}'")
    return st[0]

def _parse_mpv_shader(src):
    """Divide un file mpv in pass: [{hooks, binds, save, width, height, when, components, body}]."""
    passes, cur, body = [], None, []
    def flush():
        if cur is not None:
            cur["body"] = "\n".join(body)
            passes.append(cur)
    for line in src.splitlines():
        if line.startswith("//!"):
            if cur is None or any(l.strip() for l in body):
                flush()
                cur = {"hooks": [], "binds": [], "save": None, "width": None, "height": None,
                       "when": None, "components": 4}
                body = []
            parts = line[3:].strip().split(None, 1)
            if not parts: continue
            key, val = parts[0].upper(), (parts[1].strip() if len(parts) > 1 else "")
            if key == "HOOK": cur["hooks"].append(val)
            elif key == "BIND": cur["binds"].append(val)
            elif key == "SAVE": cur["save"] = val
            elif key == "WIDTH": cur["width"] = val
            elif key == "HEIGHT": cur["height"] = val
            elif key == "WHEN": cur["when"] = val
            elif key == "COMPONENTS":
                try: cur["components"] = max(1, min(4, int(val)))
                except ValueError: pass
            elif key in ("TEXTURE", "BUFFER", "PARAM"):
                raise ValueError(f"direttiva //!{key} non supportata")
        elif cur is not None: body.append(line)
    flush()
    return passes

class _GLTex:
    __slots__ = ("tex", "fbo", "w", "h")
    def __init__(self, tex, fbo, w, h):
        self.tex, self.fbo, self.w, self.h = tex, fbo, w, h

class ShaderChain:
    """Applica in sequenza una lista di file shader GLSL a un frame RGB uint8 (OpenGL via moderngl).
    Va creata e usata sempre dallo stesso thread (il contesto OpenGL è legato al thread che lo crea).
    Formati accettati:
      - shader mpv ("user shader", //!HOOK ...), anche multi-pass, come Anime4K: supportati HOOK
        MAIN/RGB/NATIVE/..., BIND, SAVE, WIDTH, HEIGHT, WHEN, COMPONENTS. Non supportati: HOOK LUMA/CHROMA
        e le direttive TEXTURE/BUFFER/PARAM;
      - fragment shader classico: `uniform sampler2D <nome>;` (il primo sampler riceve il frame), eventuale
        `in/varying vec2 <uv>;`, uscita `out vec4 ...` oppure `gl_FragColor`;
      - stile Shadertoy: `mainImage(out vec4, in vec2)` con iChannel0 / iResolution / iTime / iFrame.
    Gli shader mpv lavorano con l'origine in alto a sinistra, quelli classici con l'origine in basso a sinistra:
    il capovolgimento necessario viene fatto sulla GPU solo quando si passa da un tipo all'altro."""
    QUAD = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")

    def __init__(self):
        if not MODERNGL_AVAILABLE: raise RuntimeError("modulo 'moderngl' non installato (pip install moderngl)")
        self.ctx = None
        last = None
        for kw in ({"backend": "egl"}, {}):   # EGL: nessuna dipendenza dal server grafico
            try:
                self.ctx = moderngl.create_standalone_context(**kw)
                break
            except Exception as e: last = e
        if self.ctx is None: raise RuntimeError(f"contesto OpenGL non disponibile: {last}")
        self.vbo = self.ctx.buffer(self.QUAD.tobytes())
        self._blit_prog = self.ctx.program(vertex_shader=_VS_POS, fragment_shader=_FS_BLIT)
        self._blit_prog["tex"].value = 0
        self._blit_vao = self.ctx.vertex_array(self._blit_prog, [(self.vbo, "2f", "in_pos")])
        self.passes = []
        self._sig = None
        self.in_tex = None
        self._out = None
        self._free = {}
        self._used = []
        self._saved = {}
        self._main = None
        self._flip = False

    # ------------------------------------------------------------------ compilazione
    @staticmethod
    def _rel(obj):
        try:
            if obj is not None: obj.release()
        except Exception: pass

    @staticmethod
    def _build_sources(src):
        src = _GLSL_VERSION.sub("", src, count=1)
        clean = _GLSL_COMMENTS.sub("", src)
        toy = "mainImage" in clean and not re.search(r"\bvoid\s+main\s*\(", clean)
        head = ["#version 330", "#define texture2D texture", "#define varying in"]
        tail = ""
        if toy:
            for decl, name in (("uniform vec3 iResolution;", "iResolution"), ("uniform float iTime;", "iTime"),
                               ("uniform int iFrame;", "iFrame"), ("uniform vec4 iMouse;", "iMouse"),
                               ("uniform sampler2D iChannel0;", "iChannel0")):
                if not re.search(r"\buniform\b[^;]*\b" + name + r"\b", clean): head.append(decl)
            head.append("out vec4 _fragColor;")
            tail = "\nvoid main() { vec4 c = vec4(0.0, 0.0, 0.0, 1.0); mainImage(c, gl_FragCoord.xy); _fragColor = c; }\n"
        elif "gl_FragColor" in clean and not re.search(r"\bout\s+vec4\b", clean):
            head += ["out vec4 _fragColor;", "#define gl_FragColor _fragColor"]
        init = {"float": "1.0", "vec2": "in_pos * 0.5 + 0.5", "vec3": "vec3(1.0)", "vec4": "vec4(1.0)"}
        vs = ["#version 330", "in vec2 in_pos;"]
        body = ["gl_Position = vec4(in_pos, 0.0, 1.0);"]
        for typ, name in _GLSL_VARYING.findall(clean):
            vs.append(f"out {typ} {name};")
            body.append(f"{name} = {init[typ]};")
        vs.append("void main() { " + " ".join(body) + " }")
        fs = "\n".join(head) + "\n#line 1\n" + src + tail
        return "\n".join(vs), fs

    @staticmethod
    def _uniform(prog, name):
        try: return prog[name]
        except KeyError: return None

    def _compile(self, path):
        """Compila un file; restituisce la lista dei pass (uno solo per gli shader classici)."""
        with open(path, "r", encoding="utf-8", errors="replace") as f: src = f.read()
        if "//!HOOK" in src: return self._compile_mpv(path, src)
        return [self._compile_plain(path, src)]

    def _compile_plain(self, path, src):
        vs, fs = self._build_sources(src)
        prog = self.ctx.program(vertex_shader=vs, fragment_shader=fs)
        vao = self.ctx.vertex_array(prog, [(self.vbo, "2f", "in_pos")])
        for name in _GLSL_SAMPLER.findall(fs):   # tutti i sampler leggono il frame in ingresso (unità 0)
            try: prog[name].value = 0
            except Exception: pass
        def find(names, dims):
            out = []
            for n in names:
                u = self._uniform(prog, n)
                if u is not None and getattr(u, "dimension", 0) in dims and getattr(u, "array_length", 1) == 1: out.append(u)
            return out
        def is_float(u):
            try:
                u.value = 0.5
                return True
            except Exception: return False
        return {"kind": "plain", "path": path, "prog": prog, "vao": vao,
                "res": find(_SU_RES, (2, 3)), "texel": find(_SU_TEXEL, (2,)), "time": find(_SU_TIME, (1,)),
                "frame": [(u, is_float(u)) for u in find(_SU_FRAME, (1,))]}

    def _compile_mpv(self, path, src):
        parsed = [p for p in _parse_mpv_shader(src) if p["hooks"]]
        out, skipped = [], set()
        for p in parsed:
            if not (set(p["hooks"]) & _MPV_HOOKS):
                skipped.update(p["hooks"])
                continue
            out.append(self._compile_mpv_pass(path, p))
        if not out:
            raise ValueError("nessun pass supportato: gli HOOK trovati (" + ", ".join(sorted(skipped)) +
                             ") non sono gestiti, sono supportati solo " + "/".join(sorted(_MPV_HOOKS)))
        if skipped: print(f"[Shader] {os.path.basename(path)}: pass con HOOK {', '.join(sorted(skipped))} ignorati")
        return out

    def _compile_mpv_pass(self, path, p):
        binds = list(dict.fromkeys(p["binds"]))
        # in mpv la texture agganciata è raggiungibile come HOOKED e anche con il nome dell'hook (es. MAIN_texOff),
        # anche senza un //!BIND esplicito: la rendo disponibile se il codice la usa
        for n in ["HOOKED"] + [h for h in p["hooks"] if h in _MPV_HOOKS]:
            if n not in binds and re.search(r"\b" + n + r"_", p["body"]): binds.append(n)
        decl = ""
        for n in binds:
            decl += _MPV_BIND_TMPL.replace("@N@", n)
            if f"{n}_gather" in p["body"]: decl += _MPV_GATHER_TMPL.replace("@N@", n)
        comps = {1: "vec4(r.x, 0.0, 0.0, 1.0)", 2: "vec4(r.xy, 0.0, 1.0)", 3: "vec4(r.xyz, 1.0)"}.get(p["components"], "r")
        fs = ("#version 330\nin vec2 _pos;\nout vec4 _fragColor;\n"
              "uniform float random;\nuniform int frame;\nuniform vec2 input_size;\nuniform vec2 target_size;\n"
              "uniform vec2 tex_offset;\n" + decl + "#line 1\n" + p["body"] +
              "\nvoid main() { vec4 r = hook(); _fragColor = " + comps + "; }\n")
        prog = self.ctx.program(vertex_shader=_VS_POS, fragment_shader=fs)
        vao = self.ctx.vertex_array(prog, [(self.vbo, "2f", "in_pos")])
        for i, n in enumerate(binds):
            u = self._uniform(prog, n + "_raw")
            if u is not None: u.value = i
        return {"kind": "mpv", "path": path, "prog": prog, "vao": vao, "binds": binds, "hooks": p["hooks"],
                "save": p["save"], "width": p["width"], "height": p["height"], "when": p["when"],
                "u": {k: self._uniform(prog, k) for k in ("random", "frame", "input_size", "target_size")}}

    def set_shaders(self, paths):
        """Compila i file indicati. Restituisce (percorsi compilati, [(percorso, messaggio errore)])."""
        for p in self.passes:
            self._rel(p["vao"])
            self._rel(p["prog"])
        self.passes = []
        ok, errors = [], []
        for path in paths:
            try:
                self.passes.extend(self._compile(path))
                ok.append(path)
            except Exception as e: errors.append((path, str(e)))
        return ok, errors

    # ------------------------------------------------------------------ risorse GPU
    def _new_target(self, w, h, dtype="f2"):
        tex = self.ctx.texture((w, h), 4, dtype=dtype)
        tex.repeat_x = tex.repeat_y = False
        return _GLTex(tex, self.ctx.framebuffer(color_attachments=[tex]), w, h)

    def _alloc(self, w, h):
        key = (w, h)
        lst = self._free.setdefault(key, [])
        tg = lst.pop() if lst else self._new_target(w, h)
        self._used.append((key, tg))
        return tg

    def _free_targets(self):
        pool = [tg for lst in self._free.values() for tg in lst] + [tg for _k, tg in self._used]
        if self._out is not None: pool.append(self._out)
        for tg in pool:
            self._rel(tg.fbo)
            self._rel(tg.tex)
        self._rel(self.in_tex)
        self._free, self._used, self._out, self.in_tex, self._sig = {}, [], None, None, None

    def _blit(self, tex, fbo, flip):
        fbo.use()
        tex.use(0)
        self._blit_prog["flip"].value = 1.0 if flip else 0.0
        self._blit_vao.render(moderngl.TRIANGLE_STRIP)

    def _orient(self, flipped):
        """Porta il frame corrente nell'orientamento richiesto (False: riga 0 in alto, come mpv; True: in basso)."""
        if self._flip == flipped: return
        src = self._main
        dst = self._alloc(src.w, src.h)
        self._blit(src.tex, dst.fbo, True)
        self._main, self._flip = dst, flipped

    # ------------------------------------------------------------------ esecuzione
    def _run_plain(self, p, t, frame_no):
        self._orient(True)
        src = self._main
        w, h = src.w, src.h
        dst = self._alloc(w, h)
        dst.fbo.use()
        src.tex.use(0)
        for u in p["res"]: u.value = (float(w), float(h), 1.0)[:u.dimension]
        for u in p["texel"]: u.value = (1.0 / w, 1.0 / h)
        for u in p["time"]: u.value = float(t)
        for u, as_float in p["frame"]: u.value = float(frame_no) if as_float else int(frame_no)
        p["vao"].render(moderngl.TRIANGLE_STRIP)
        self._main = dst

    def _run_mpv(self, p, out_size, frame_no, force_when=False):
        main = self._main
        sizes = {nm: (main.w, main.h) for nm in _MPV_HOOKS}
        sizes["HOOKED"] = (main.w, main.h)
        sizes["OUTPUT"] = sizes["SCALED"] = out_size
        sizes.update({k: (v.w, v.h) for k, v in self._saved.items()})
        if p["when"] is not None and not force_when and not _mpv_rpn(p["when"], sizes): return
        if not all(b == "HOOKED" or b in _MPV_HOOKS or b in self._saved for b in p["binds"]): return
        self._orient(False)
        main = self._main
        ow = max(1, int(round(_mpv_rpn(p["width"], sizes)))) if p["width"] else main.w
        oh = max(1, int(round(_mpv_rpn(p["height"], sizes)))) if p["height"] else main.h
        dst = self._alloc(ow, oh)
        dst.fbo.use()
        for i, b in enumerate(p["binds"]):
            (main if (b == "HOOKED" or b in _MPV_HOOKS) else self._saved[b]).tex.use(i)
        u = p["u"]
        if u["random"] is not None: u["random"].value = float(np.random.random())
        if u["frame"] is not None:
            try: u["frame"].value = int(frame_no)
            except Exception: pass
        if u["input_size"] is not None: u["input_size"].value = (float(main.w), float(main.h))
        if u["target_size"] is not None: u["target_size"].value = (float(out_size[0]), float(out_size[1]))
        p["vao"].render(moderngl.TRIANGLE_STRIP)
        if p["save"] and p["save"] != "HOOKED" and p["save"] not in _MPV_HOOKS: self._saved[p["save"]] = dst
        else: self._main = dst

    def run(self, frame, t, frame_no, out_size=None, force_when=False):
        """frame: RGB uint8 (h, w, 3). out_size: dimensioni di visualizzazione (per le condizioni //!WHEN OUTPUT.*)."""
        h, w = frame.shape[:2]
        out_size = (int(out_size[0]), int(out_size[1])) if out_size and out_size[0] > 0 and out_size[1] > 0 else (w, h)
        sig = (w, h, out_size)
        if sig != self._sig:
            self._free_targets()
            self._sig = sig
            self.in_tex = self.ctx.texture((w, h), 3)
            self.in_tex.repeat_x = self.in_tex.repeat_y = False
        self.in_tex.write(np.ascontiguousarray(frame))
        inp = _GLTex(self.in_tex, None, w, h)
        self._main, self._flip, self._saved, self._used = inp, False, {}, []
        try:
            for p in self.passes:
                if p["kind"] == "plain": self._run_plain(p, t, frame_no)
                else: self._run_mpv(p, out_size, frame_no, force_when)
            main = self._main
            if main is inp and not self._flip: return frame   # nessun pass è stato eseguito
            if self._out is None or (self._out.w, self._out.h) != (main.w, main.h):
                if self._out is not None:
                    self._rel(self._out.fbo)
                    self._rel(self._out.tex)
                self._out = self._new_target(main.w, main.h, dtype="f1")
            self._blit(main.tex, self._out.fbo, self._flip)
            raw = self._out.fbo.read(components=3, alignment=1)
            return np.frombuffer(raw, dtype=np.uint8).reshape(main.h, main.w, 3).copy()
        finally:
            for key, tg in self._used: self._free.setdefault(key, []).append(tg)
            self._used, self._saved, self._main = [], {}, None

    def release(self):
        self.set_shaders([])
        self._free_targets()
        self._rel(self._blit_vao)
        self._rel(self._blit_prog)
        self._rel(self.vbo)
        self._rel(self.ctx)


_Frame = collections.namedtuple("_Frame", "gen pos img is_seek")
_EOF = object()
_LOOP = object()

class VideoProcessorThread(QThread):
    frame_ready = pyqtSignal(object, float, float, int, int)
    position_changed = pyqtSignal(int, int, float, float)
    playback_finished = pyqtSignal()
    loop_restarted = pyqtSignal()
    status_message = pyqtSignal(str)
    shader_failed = pyqtSignal(object, str)   # (lista percorsi shader, messaggio): compilazione o esecuzione fallita
    POS_EMIT_INTERVAL = 0.1
    SYNC_HARD = 0.12

    def __init__(self):
        super().__init__()
        self.video_path = None

        # Modelli AI (Ora gestiti come dizionari per supportare pipeline multiple)
        self.upscaler = None
        self.upscaler_model_path = None
        self.denoiser = None
        self.denoise_model_path = None
        self.rife = None
        self._rife_prev = None
        self.decode_mode = "CPU"

        self.device = "cuda" if TORCH_CUDA else "cpu"
        self._model_lock = threading.Lock()

        self.running = False
        self.paused = False
        self.upscale_enabled = True
        self.denoise_enabled = False
        self.rife_enabled = False
        self.loop_enabled = LOOP_VIDEO

        self.seek_target_frame = None
        self.seek_delta_sec = 0.0
        self.last_frame_time = 0.0
        self.real_fps = 0.0
        self.video_fps = 25.0
        self._fps_ready = False        # True quando video_fps e' quello reale del file aperto
        self._pending_seek_sec = None  # salto in secondi richiesto prima che l'fps reale sia noto
        self.audio_clock = None
        self._seek_gen = 0
        self._shown_pos = 0
        self.display_size = None

        # Shader GLSL di post-processing (applicati ai frame dopo la pipeline AI, nello stesso thread)
        self.shader_paths = ()
        self.shader_force_when = False
        self._shader_chain = None
        self._shader_applied = ()
        self._shader_dirty = False
        self._last_raw = None   # (frame senza shader, w, h): serve a ridisegnarlo quando si cambiano gli shader in pausa

    def _load_model_internal(self, model_path):
        """Carica un modello e restituisce un dizionario con le sue proprietà."""
        ext = os.path.splitext(model_path)[1].lower()
        info = {
            "path": model_path, "type": "onnx" if ext == ".onnx" else "torch",
            "session": None, "torch_model": None,
            "input_name": None, "out_name": None,
            "np_dtype": np.float32, "t_dtype": None, "t_out_dtype": None,
            "onnx_gpu": False, "onnx_out_shapes": {}, "onnx_out_buf": None,
        }
        if ext == ".onnx":
            available = ort.get_available_providers()
            providers = []
            if USE_TENSORRT and 'TensorrtExecutionProvider' in available:
                os.makedirs(TRT_CACHE_DIR, exist_ok=True)
                providers.append(('TensorrtExecutionProvider', {
                    "trt_engine_cache_enable": True, "trt_engine_cache_path": TRT_CACHE_DIR,
                    "trt_timing_cache_enable": True, "trt_timing_cache_path": TRT_CACHE_DIR,
                    "trt_fp16_enable": TRT_FORCE_FP16,
                }))
            if 'CUDAExecutionProvider' in available: providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
            session = ort.InferenceSession(model_path, providers=providers)
            input_meta = session.get_inputs()[0]
            output_meta = session.get_outputs()[0]
            in_fp16 = "float16" in input_meta.type
            out_fp16 = "float16" in output_meta.type
            np_dtype = np.float16 if in_fp16 else np.float32
            active_provider = session.get_providers()[0]
            use_gpu = TORCH_AVAILABLE and TORCH_CUDA and (active_provider in ('CUDAExecutionProvider', 'TensorrtExecutionProvider'))

            info["session"] = session
            info["input_name"] = input_meta.name
            info["out_name"] = output_meta.name
            info["np_dtype"] = np_dtype
            info["t_dtype"] = (torch.float16 if in_fp16 else torch.float32) if use_gpu else None
            info["t_out_dtype"] = (torch.float16 if out_fp16 else torch.float32) if use_gpu else None
            info["onnx_gpu"] = use_gpu

        elif ext in (".pt", ".pth"):
            if not TORCH_AVAILABLE or not SPANDREL_AVAILABLE:
                raise RuntimeError("PyTorch o Spandrel non installati.")
            loader = ModelLoader()
            descriptor = loader.load_from_file(model_path)
            device = "cuda" if TORCH_CUDA else "cpu"
            torch_model = descriptor.model.to(device).eval()
            t_dtype = torch.float32
            if device == "cuda":
                torch_model = torch_model.to(memory_format=torch.channels_last)
                if TORCH_FP16 and getattr(descriptor, "supports_half", False):
                    torch_model = torch_model.half()
                    t_dtype = torch.float16
                    if not self._fp16_is_stable(torch_model):
                        torch_model = torch_model.float()
                        t_dtype = torch.float32
            info["torch_model"] = torch_model
            info["t_dtype"] = t_dtype
            info["np_dtype"] = np.float16 if t_dtype == torch.float16 else np.float32
        else:
            raise ValueError(f"Formato non supportato: {ext}")
        return info

    def load_model(self, model_path):
        try:
            info = self._load_model_internal(model_path)
            with self._model_lock:
                self.upscaler = info
                self.upscaler_model_path = model_path
            print(f"[Neko Player] Upscaler caricato: {os.path.basename(model_path)}")
        except Exception as e:
            print(f"[Neko Player] Errore Upscaler: {e}")
            raise

    def load_denoise_model(self, model_path):
        if not model_path:
            with self._model_lock:
                self.denoiser = None
                self.denoise_model_path = None
            return
        try:
            info = self._load_model_internal(model_path)
            with self._model_lock:
                self.denoiser = info
                self.denoise_model_path = model_path
            print(f"[Neko Player] Denoise caricato: {os.path.basename(model_path)}")
        except Exception as e:
            print(f"[Neko Player] Errore Denoise: {e}")
            raise

    def load_rife(self):
        """Carica il modello RIFE fisso (ONNX). Restituisce True se pronto."""
        if self.rife is not None: return True
        if not os.path.isfile(RIFE_MODEL_PATH):
            print(f"[Neko Player] Modello RIFE non trovato: {RIFE_MODEL_PATH}")
            return False
        try:
            # Solo CUDA (niente TensorRT): le dimensioni cambiano con la finestra
            available = ort.get_available_providers()
            providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in available else []
            providers.append('CPUExecutionProvider')
            session = ort.InferenceSession(RIFE_MODEL_PATH, providers=providers)
            inputs = session.get_inputs()
            if len(inputs) < 2: raise RuntimeError("il modello RIFE deve avere almeno 2 ingressi (img0, img1)")
            use_gpu = TORCH_AVAILABLE and TORCH_CUDA and session.get_providers()[0] == 'CUDAExecutionProvider'
            info = {
                "session": session,
                "img_names": [inputs[0].name, inputs[1].name],
                "ts_name": inputs[2].name if len(inputs) > 2 else None,
                "ts_rank": len(inputs[2].shape) if len(inputs) > 2 else 0,
                "out_name": session.get_outputs()[0].name,
                "onnx_gpu": use_gpu, "out_buf": None,
            }
            with self._model_lock: self.rife = info
            print(f"[Neko Player] RIFE caricato ({'GPU' if use_gpu else 'CPU'}): {os.path.basename(RIFE_MODEL_PATH)}")
            return True
        except Exception as e:
            print(f"[Neko Player] Errore RIFE: {e}")
            return False

    def set_rife_enabled(self, enabled):
        self.rife_enabled = bool(enabled)
        self._rife_prev = None

    def set_shader_force(self, on):
        self.shader_force_when = bool(on)
        self._shader_dirty = True

    def set_shaders(self, paths):
        """Imposta la catena di shader (percorsi, nell'ordine di applicazione). La compilazione avviene nel thread video."""
        self.shader_paths = tuple(paths)
        self._shader_dirty = True

    def _rife_active(self):
        return bool(self.rife_enabled and self.rife)

    def _ai_active(self):
        return bool((self.denoise_enabled and self.denoiser) or (self.upscale_enabled and self.upscaler) or self._rife_active())

    def _fp16_is_stable(self, model):
        try:
            with torch.inference_mode():
                probe = torch.rand(1, 3, 64, 64, device=self.device, dtype=torch.float16)
                probe = probe.contiguous(memory_format=torch.channels_last)
                return bool(torch.isfinite(model(probe)).all().item())
        except Exception: return False

    def set_video(self, video_path):
        self.video_path = video_path
        self._fps_ready = False
        self._pending_seek_sec = None
    def request_seek_frame(self, frame_no: int): self.seek_target_frame = frame_no
    def request_seek_time(self, sec: float):
        """Salto a un istante assoluto: il frame si calcola con l'fps reale del file (non con quello di default)."""
        if self._fps_ready: self.seek_target_frame = int(max(0.0, sec) * self.video_fps)
        else: self._pending_seek_sec = sec
    def request_seek_seconds(self, delta_seconds: float): self.seek_delta_sec = delta_seconds

    def run(self):
        if not self.video_path: return
        cap, self.decode_mode = open_video_capture(self.video_path)
        if not cap.isOpened(): return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if not video_fps or video_fps <= 0 or video_fps > 120: video_fps = 25.0
        self.video_fps = video_fps
        self._fps_ready = True
        if self._pending_seek_sec is not None:
            self.seek_target_frame = int(max(0.0, self._pending_seek_sec) * video_fps)
            self._pending_seek_sec = None
        frame_interval = 1.0 / video_fps
        duration_sec = total_frames / video_fps
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.running = True
        self._shown_pos = 0
        self._rife_prev = None
        self._last_raw = None
        self._shader_dirty = False
        self._warmup(vid_w, vid_h)
        if AUDIO_SYNC:
            t_wait = time.perf_counter()
            while self.running and self.audio_clock is None and time.perf_counter() - t_wait < 0.5: time.sleep(0.01)
        frames_q = queue.Queue(maxsize=DECODE_QUEUE_SIZE)
        stop_evt = threading.Event()
        decoder = threading.Thread(target=self._decode_loop, args=(cap, frames_q, stop_evt, total_frames), daemon=True)
        decoder.start()
        now = time.perf_counter()
        next_t = now
        self.last_frame_time = now
        self.real_fps = video_fps
        last_pos_emit = 0.0
        proc_est = 0.0
        catching_up = False
        sync_events = []
        sync_disabled = False
        ended = False
        while self.running:
            if self._shader_dirty:
                self._shader_dirty = False
                if self.paused: self._reemit_shaded()   # in pausa il frame non cambia: lo si ridisegna con i nuovi shader
            head = self._peek(frames_q)
            if head is None:
                if not decoder.is_alive():
                    ended = True
                    break
                time.sleep(0.002)
                continue
            is_marker = head is _EOF or head is _LOOP
            if not is_marker and head.gen != self._seek_gen:
                self._pop(frames_q)
                continue
            if self.paused and (is_marker or not head.is_seek):
                time.sleep(0.02)
                next_t = time.perf_counter()
                self.last_frame_time = next_t
                continue
            item = self._pop(frames_q)
            if item is None: continue
            if item is _EOF:
                ended = True
                break
            if item is _LOOP:
                self._rife_prev = None
                self.loop_restarted.emit()
                continue
            if item.gen != self._seek_gen: continue
            frame_pos = item.pos
            self._shown_pos = frame_pos
            t_frame = (frame_pos - 1) / video_fps
            if item.is_seek:
                self.position_changed.emit(frame_pos, total_frames, frame_pos / video_fps, duration_sec)
                self._rife_prev = None
                self._process_and_emit(item.img, 0.0)
                now = time.perf_counter()
                next_t = now + frame_interval
                self.last_frame_time = now
                last_pos_emit = now
                catching_up = False
                continue
            delay = next_t - time.perf_counter()
            if delay > 0: time.sleep(delay)
            elif delay < -0.25: next_t = time.perf_counter()
            drift = None if sync_disabled else self._audio_drift(t_frame, proc_est)
            if drift is not None:
                hard_event = False
                if drift > self.SYNC_HARD or (catching_up and drift > 0.03):
                    if not catching_up: hard_event = True
                    catching_up = True
                    self._rife_prev = None
                    next_t = time.perf_counter()
                    if hard_event: sync_events.append(next_t)
                    sync_disabled = self._check_sync_reliability(sync_events, proc_est, frame_interval) or sync_disabled
                    continue
                catching_up = False
                if drift < -self.SYNC_HARD:
                    while self.running and not self.paused and item.gen == self._seek_gen:
                        d = self._audio_drift(t_frame, proc_est)
                        if d is None or d > -0.02: break
                        time.sleep(min(-d - 0.02, 0.05))
                    next_t = time.perf_counter()
                    sync_events.append(next_t)
                    sync_disabled = self._check_sync_reliability(sync_events, proc_est, frame_interval) or sync_disabled
                else:
                    limit = 0.25 * frame_interval
                    next_t -= max(-limit, min(limit, 0.05 * drift))
            now = time.perf_counter()
            if now - last_pos_emit >= self.POS_EMIT_INTERVAL:
                last_pos_emit = now
                self.position_changed.emit(frame_pos, total_frames, frame_pos / video_fps, duration_sec)
            dt = now - self.last_frame_time
            self.last_frame_time = now
            if dt > 0: self.real_fps = 0.85 * self.real_fps + 0.15 * (1.0 / dt)
            self._process_and_emit(item.img, self.real_fps, interp=True)
            proc = time.perf_counter() - now
            proc_est = proc if proc_est == 0.0 else 0.9 * proc_est + 0.1 * proc
            next_t += frame_interval
        stop_evt.set()
        decoder.join()
        cap.release()
        self._release_shaders()   # il contesto OpenGL va rilasciato dal thread che l'ha creato
        self.running = False
        if ended: self.playback_finished.emit()

    def _decode_loop(self, cap, q, stop_evt, total_frames):
        frame_pos = 0
        got_frames = False
        try:
            while not stop_evt.is_set():
                if self.seek_target_frame is not None or self.seek_delta_sec != 0.0:
                    if self.seek_target_frame is not None:
                        target = max(0, min(total_frames - 1, self.seek_target_frame))
                        self.seek_target_frame = None
                    else:
                        target = max(0, min(total_frames - 1, int(self._shown_pos + self.seek_delta_sec * self.video_fps)))
                        self.seek_delta_sec = 0.0
                    self._seek_gen += 1
                    self._flush(q)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                    ret, frame = cap.read()
                    if ret:
                        got_frames = True
                        frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        self._put(q, stop_evt, _Frame(self._seek_gen, frame_pos, frame, True), force=True)
                    else: frame_pos = target
                    continue
                ret, frame = cap.read()
                if not ret:
                    if self.loop_enabled and got_frames:
                        if not self._put(q, stop_evt, _LOOP): continue
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_pos = 0
                        got_frames = False
                        continue
                    if self._put(q, stop_evt, _EOF): return
                    if stop_evt.is_set(): return
                    continue
                got_frames = True
                frame_pos += 1
                self._put(q, stop_evt, _Frame(self._seek_gen, frame_pos, frame, False))
        except Exception as e: print(f"[Decoder] Errore: {e}")

    def _put(self, q, stop_evt, item, force=False):
        while not stop_evt.is_set():
            if not force and (self.seek_target_frame is not None or self.seek_delta_sec != 0.0): return False
            try:
                q.put(item, timeout=0.05)
                return True
            except queue.Full: continue
        return False

    @staticmethod
    def _flush(q):
        try:
            while True: q.get_nowait()
        except queue.Empty: pass

    @staticmethod
    def _peek(q):
        with q.mutex: return q.queue[0] if q.queue else None

    @staticmethod
    def _pop(q):
        try: return q.get_nowait()
        except queue.Empty: return None

    def _audio_drift(self, t_frame, lead=0.0):
        ac = self.audio_clock
        if not AUDIO_SYNC or ac is None: return None
        audio_t = ac[0] + (time.perf_counter() - ac[1]) + lead
        d = audio_t - t_frame
        if abs(d) > 2.0: return None
        return d

    def _check_sync_reliability(self, events, proc_est, frame_interval):
        now = time.perf_counter()
        events[:] = [t for t in events if now - t < 10.0]
        if len(events) > 4 and proc_est < 0.9 * frame_interval:
            print("[Neko Player] Clock audio instabile: sincronizzazione disattivata.")
            return True
        return False

    def _warmup(self, w, h):
        if w <= 0 or h <= 0: return
        with self._model_lock:
            if not self._ai_active(): return
            self.status_message.emit("🧠 Preparazione Pipeline AI...")
            try:
                dummy = np.zeros((h, w, 3), dtype=np.uint8)
                self._rife_prev = None
                self._run_pipeline(dummy)
                if self._rife_active(): self._run_pipeline(dummy)
            except Exception as e: print(f"[Neko Player] Warmup non riuscito: {e}")
            finally: self._rife_prev = None

    def _process_and_emit(self, frame, fps_val, interp=False):
        gpu_time_ms = 0.0
        display_frame = None
        mid_frame = None
        full_w = full_h = 0
        with self._model_lock:
            if self._ai_active():
                t0 = time.perf_counter()
                try:
                    display_frame, full_w, full_h, mid_frame = self._run_pipeline(frame)
                    gpu_time_ms = (time.perf_counter() - t0) * 1000.0
                except Exception as e:
                    print(f"[Errore Pipeline]: {e}")
                    display_frame = None
                    mid_frame = None
                    self._rife_prev = None
        if display_frame is None:
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            full_h, full_w = display_frame.shape[:2]
        h, w = display_frame.shape[:2]
        size = self._reduced_size(w, h)
        if size: display_frame = cv2.resize(display_frame, size, interpolation=cv2.INTER_AREA)
        self._last_raw = (display_frame, full_w, full_h)
        src_wh = (display_frame.shape[1], display_frame.shape[0])
        display_frame, shaded = self._shade_and_fit(display_frame)
        full_w, full_h = self._scaled_full(full_w, full_h, src_wh, shaded)
        if mid_frame is not None: mid_frame = self._shade_and_fit(mid_frame)[0]
        if mid_frame is not None and interp:
            # frame intermedio subito, frame reale mezzo intervallo dopo
            self.frame_ready.emit(np.ascontiguousarray(mid_frame), fps_val * 2.0, gpu_time_ms, full_w, full_h)
            time.sleep(0.5 / max(self.video_fps, 1.0))
            fps_val *= 2.0
        self.frame_ready.emit(np.ascontiguousarray(display_frame), fps_val, gpu_time_ms, full_w, full_h)

    def _shade_and_fit(self, frame):
        """Applica gli shader; se hanno ingrandito il frame (es. upscale 2x) lo riporta alla dimensione di visualizzazione."""
        out = self._apply_shaders(frame)
        shaded = (out.shape[1], out.shape[0])   # dimensione prodotta dagli shader, prima del ridimensionamento
        if out is not frame:
            h, w = out.shape[:2]
            size = self._reduced_size(w, h)
            if size: out = cv2.resize(out, size, interpolation=cv2.INTER_AREA)
        return out, shaded

    @staticmethod
    def _scaled_full(full_w, full_h, src_wh, shaded):
        """Se gli shader hanno cambiato la dimensione del frame (es. upscale 2x) la risoluzione riportata nell'OSD segue lo stesso fattore."""
        if shaded != src_wh and src_wh[0] > 0 and src_wh[1] > 0:
            return int(round(full_w * shaded[0] / src_wh[0])), int(round(full_h * shaded[1] / src_wh[1]))
        return full_w, full_h

    def _apply_shaders(self, frame):
        """Passa il frame RGB nella catena di shader attivi. In caso di errore disattiva gli shader e avvisa la GUI."""
        paths = self.shader_paths
        if not paths and self._shader_chain is None: return frame
        try:
            chain = self._shader_chain
            if chain is None:
                chain = self._shader_chain = ShaderChain()
            if paths != self._shader_applied:
                self._shader_applied = paths
                _ok, errors = chain.set_shaders(paths)
                for p, msg in errors:
                    print(f"[Shader] {os.path.basename(p)}: {msg}")
                    self.shader_failed.emit([p], msg)
            if not chain.passes: return frame
            return chain.run(frame, self._shown_pos / max(self.video_fps, 1.0), self._shown_pos, self.display_size,
                             self.shader_force_when)
        except Exception as e:
            print(f"[Shader] Errore: {e}")
            failed = list(paths)
            self.shader_paths = ()
            self._shader_applied = ()
            if self._shader_chain is not None:
                try: self._shader_chain.set_shaders([])
                except Exception: pass
            self.shader_failed.emit(failed, str(e))
            return frame

    def _reemit_shaded(self):
        raw = self._last_raw
        if raw is None: return
        frame, full_w, full_h = raw
        out, shaded = self._shade_and_fit(frame)
        full_w, full_h = self._scaled_full(full_w, full_h, (frame.shape[1], frame.shape[0]), shaded)
        self.frame_ready.emit(np.ascontiguousarray(out), self.real_fps, 0.0, full_w, full_h)

    def _release_shaders(self):
        chain, self._shader_chain = self._shader_chain, None
        self._shader_applied = ()
        if chain is not None:
            try: chain.release()
            except Exception: pass

    def _reduced_size(self, w, h):
        ds = self.display_size
        if not ds: return None
        dw, dh = ds
        if dw <= 0 or dh <= 0 or h <= dh: return None
        s = dh / h
        return max(1, int(round(w * s))), dh

    def _run_pipeline(self, frame_bgr_u8):
        """Catena: Input -> Denoise -> Upscale -> riduzione -> RIFE. Restituisce (frame, w, h, frame_intermedio|None)."""
        t = torch.from_numpy(frame_bgr_u8).to(self.device)
        x = t.flip(-1).permute(2, 0, 1).unsqueeze(0).float().mul_(1.0 / 255.0)

        if self.denoise_enabled and self.denoiser:
            x = self._infer_internal(self.denoiser, x)
        if self.upscale_enabled and self.upscaler:
            x = self._infer_internal(self.upscaler, x)

        x = x.clamp_(0.0, 1.0)
        full_h, full_w = x.shape[-2:]
        size = self._reduced_size(full_w, full_h)
        if size:
            x = torch.nn.functional.interpolate(x, size=(size[1], size[0]), mode="area")

        mid = None
        if self._rife_active():
            prev = self._rife_prev
            self._rife_prev = x.clone()   # copia: x può puntare a un buffer riutilizzato dall'upscaler
            if prev is not None and prev.shape == x.shape and not self._rife_scene_cut(prev, x):
                mid = self._rife_interpolate(prev, x)
        else:
            self._rife_prev = None

        out = x if mid is None else torch.cat([mid, x], dim=0)
        out_u8 = (out * 255.0).round_().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
        if mid is None: return out_u8[0], int(full_w), int(full_h), None
        return out_u8[1], int(full_w), int(full_h), out_u8[0]

    @staticmethod
    def _rife_scene_cut(a, b):
        return float((a[..., ::8, ::8] - b[..., ::8, ::8]).abs().mean().item()) > RIFE_SCENE_CUT

    def _rife_interpolate(self, a, b):
        """Frame a metà tra a e b (tensori 1x3xHxW in [0,1]) con RIFE ONNX."""
        info = self.rife
        h, w = a.shape[-2:]
        ph, pw = ((h - 1) // 64 + 1) * 64, ((w - 1) // 64 + 1) * 64
        if (ph, pw) != (h, w):
            pad = (0, pw - w, 0, ph - h)
            a = torch.nn.functional.pad(a, pad, mode="replicate")
            b = torch.nn.functional.pad(b, pad, mode="replicate")
        a = a.contiguous()
        b = b.contiguous()
        session = info["session"]
        n0, n1 = info["img_names"]
        if info["ts_rank"] == 0: ts = np.array(0.5, dtype=np.float32)
        elif info["ts_rank"] == 1: ts = np.array([0.5], dtype=np.float32)
        else: ts = np.full((1, 1, ph, pw), 0.5, dtype=np.float32)
        if info["onnx_gpu"]:
            shape = (1, 3, ph, pw)
            buf = info["out_buf"]
            if buf is None or tuple(buf.shape) != shape:
                buf = torch.empty(shape, dtype=torch.float32, device=self.device)
                info["out_buf"] = buf
            dev_id = torch.cuda.current_device()
            bd = session.io_binding()
            bd.bind_input(n0, 'cuda', dev_id, np.float32, shape, a.data_ptr())
            bd.bind_input(n1, 'cuda', dev_id, np.float32, shape, b.data_ptr())
            if info["ts_name"]: bd.bind_cpu_input(info["ts_name"], ts)
            bd.bind_output(info["out_name"], 'cuda', dev_id, np.float32, shape, buf.data_ptr())
            torch.cuda.current_stream().synchronize()
            session.run_with_iobinding(bd)
            sync = getattr(bd, "synchronize_outputs", None)
            if sync: sync()
            out = buf
        else:
            feeds = {n0: a.cpu().numpy(), n1: b.cpu().numpy()}
            if info["ts_name"]: feeds[info["ts_name"]] = ts
            out = torch.from_numpy(session.run(None, feeds)[0]).to(self.device)
        return out[..., :h, :w].clamp(0.0, 1.0)

    def _infer_internal(self, info, x):
        """Smista l'inferenza mantenendo i tensori in VRAM."""
        if info["type"] == "torch":
            x_in = x.to(self.device, dtype=info["t_dtype"])
            if self.device == "cuda": x_in = x_in.contiguous(memory_format=torch.channels_last)
            with torch.inference_mode():
                out = info["torch_model"](x_in)
            return out.float()
        elif info["type"] == "onnx":
            if info["onnx_gpu"]: return self._infer_onnx_gpu(info, x)
            else: return self._infer_onnx_cpu(info, x)

    def _infer_onnx_gpu(self, info, x):
        x_in = x.to(info["t_dtype"]).contiguous()
        h, w = x_in.shape[2], x_in.shape[3]
        shape = info["onnx_out_shapes"].get((h, w))
        if shape is None:
            out_np = info["session"].run(None, {info["input_name"]: x_in.cpu().numpy()})[0]
            info["onnx_out_shapes"][(h, w)] = tuple(out_np.shape)
            return torch.from_numpy(out_np).to(self.device).float()

        if info["onnx_out_buf"] is None or tuple(info["onnx_out_buf"].shape) != shape:
            info["onnx_out_buf"] = torch.empty(shape, dtype=info["t_out_dtype"] or torch.float32, device=self.device)
        out = info["onnx_out_buf"]
        dev_id = torch.cuda.current_device()

        b = info["session"].io_binding()
        b.bind_input(info["input_name"], 'cuda', dev_id, np.dtype(np.float16 if info["t_dtype"] == torch.float16 else np.float32), tuple(x_in.shape), x_in.data_ptr())

        b.bind_output(info["out_name"], 'cuda', dev_id, np.dtype(np.float16 if info["t_out_dtype"] == torch.float16 else np.float32), shape, out.data_ptr())
        torch.cuda.current_stream().synchronize()
        info["session"].run_with_iobinding(b)
        sync = getattr(b, "synchronize_outputs", None)
        if sync: sync()
        return out.float()

    def _infer_onnx_cpu(self, info, x):
        x_np = x.cpu().numpy()
        if info["np_dtype"] == np.float16: x_np = x_np.astype(np.float16)
        out = info["session"].run(None, {info["input_name"]: x_np})[0]
        return torch.from_numpy(out).to(self.device).float()

    def stop(self):
        self.running = False
        self.wait()

class BubbleMenu(QWidget):
    """Menu contestuale a forma di nuvoletta di pensiero: al posto della codina, due cerchi (uno grande e uno piccolo) che
    scendono verso il cursore, come se il gatto stesse pensando il menu."""
    TAIL_H = 24          # altezza riservata ai due cerchi del pensiero
    DOT_R1, DOT_R2 = 5.5, 3.5   # raggio del cerchio vicino al fumetto e di quello vicino al cursore
    DOT_GAP = 3.0        # distanza tra i cerchi e tra il cerchio grande e il fumetto
    PAD = 8
    ROW_H = 30
    RADIUS = 16
    MARGIN = 3
    def __init__(self, items, parent=None, keep_open=False, refresh=None):
        # items: lista di (testo, callback, abilitata) oppure (testo, callback, abilitata, checked) per i toggle
        # keep_open: i toggle non chiudono il menu; il loro callback riceve il nuovo stato (bool)
        # refresh: funzione che restituisce le voci aggiornate (stesso numero), richiamata dopo ogni toggle
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setMouseTracking(True)
        self.items = [tuple(it[:3]) + tuple(it[3:]) if len(it) > 3 else tuple(it[:3]) for it in items]
        self._hover = -1
        self._tail_top = True
        self._tail_x = 24
        self._from_center = None   # (dx, dy): posizione del cursore rispetto al centro della finestra del gatto
        self._font = QFont(resolve_ui_font_family(), 10)
        self._font.setBold(True)
        fm = QFontMetrics(self._font)
        # le voci possono avere 3 elementi (testo, callback, abilitata) o 4 (toggle con checked)
        self.keep_open = keep_open
        self._refresh = refresh
        num_w = fm.horizontalAdvance("00. ") if keep_open else 0   # spazio per la numerazione
        self._body_w = max(fm.horizontalAdvance(it[0]) + (50 if len(it) > 3 else 28) for it in items) + num_w + 2 * self.PAD
        # troppe voci per lo schermo: se ne mostra solo una parte e si scorre con la rotella
        scr = QApplication.primaryScreen()
        self._max_rows = max(4, ((scr.geometry().height() if scr else 800) - 160) // self.ROW_H)
        self._rows = min(len(self.items), self._max_rows)
        self._scroll = 0
        self._body_h = self._rows * self.ROW_H + 2 * self.PAD
        self.resize(self._body_w + 2 * self.MARGIN, self._body_h + self.TAIL_H + 2 * self.MARGIN)
    def popup_at(self, gpos, center=None):
        # center: centro (globale) della finestra del gatto. I cerchi del pensiero sono orientati verso di esso: partono
        # dal fumetto, che sta dal lato del cursore opposto al centro, e puntano al centro della finestra.
        self._from_center = (gpos.x() - center.x(), gpos.y() - center.y()) if center is not None else None
        screen = QApplication.screenAt(gpos) or QApplication.primaryScreen()
        sg = screen.geometry()
        W, H, M = self.width(), self.height(), self.MARGIN
        inset = M + self.RADIUS + 8
        x = gpos.x() - inset
        if x + W > sg.right() + 1: x = gpos.x() - (W - inset)
        x = max(sg.left(), min(x, sg.right() + 1 - W))
        self._tail_x = max(M + self.RADIUS, min(gpos.x() - x, W - M - self.RADIUS))
        below = True if self._from_center is None else self._from_center[1] >= 0   # fumetto sotto il cursore se questo sta nella metà bassa
        fits_below = gpos.y() + H <= sg.bottom() + 1
        fits_above = gpos.y() - H + M >= sg.top()
        if below and not fits_below: below = False
        elif not below and not fits_above and fits_below: below = True
        self._tail_top = below
        y = (gpos.y() - M) if below else (gpos.y() - H + M)
        self.move(x, max(sg.top(), y))
        self.show()
    def _body_rect(self):
        top = self.MARGIN + (self.TAIL_H if self._tail_top else 0)
        return QRectF(self.MARGIN, top, self._body_w, self._body_h)
    def _row_rect(self, i):
        b = self._body_rect()
        return QRectF(b.x() + self.PAD, b.y() + self.PAD + (i - self._scroll) * self.ROW_H, b.width() - 2 * self.PAD, self.ROW_H)
    def _row_at(self, pos):
        for i in range(self._scroll, self._scroll + self._rows):
            if self._row_rect(i).contains(QPointF(pos)): return i
        return -1
    def _bubble_path(self):
        body = self._body_rect()
        path = QPainterPath()
        path.addRoundedRect(body, self.RADIUS, self.RADIUS)
        tx = float(self._tail_x)
        d = 1.0 if self._tail_top else -1.0                  # verso del pensiero: dal cursore al fumetto
        tip_y = float(self.MARGIN) if self._tail_top else self.height() - float(self.MARGIN)
        r1, r2, g = self.DOT_R1, self.DOT_R2, self.DOT_GAP
        step = r2 + g + r1                                   # distanza verticale tra i centri dei due cerchi
        if self._from_center is not None:
            # la fila di cerchi giace sulla retta tra il centro della finestra e il cursore: il cerchio piccolo è il più vicino al centro
            dx, dy = self._from_center
            off = max(-12.0, min(12.0, dx / max(abs(dy), 1.0) * step))
        else:
            off = (7.0 if tx < self.width() / 2.0 else -7.0)
        c2 = QPointF(tx, tip_y + d * r2)
        c1 = QPointF(tx + off, tip_y + d * (2 * r2 + g + r1))
        path.addEllipse(c1, r1, r1)
        path.addEllipse(c2, r2, r2)
        return path
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        path = self._bubble_path()
        grad = QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0.0, QColor("#ffd6ef"))
        grad.setColorAt(0.55, QColor("#f5c2e7"))
        grad.setColorAt(1.0, QColor("#eaa9d8"))
        pen = QPen(QColor("#11111b"), 2.5)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(QBrush(grad))
        p.drawPath(path)
        p.setFont(self._font)
        check_font = QFont(self._font)
        check_font.setPointSize(max(8, self._font.pointSize()))
        for i in range(self._scroll, self._scroll + self._rows):
            item = self.items[i]
            text, _cb, enabled = item[0], item[1], item[2]
            checked = bool(item[3]) if len(item) > 3 else None
            r = self._row_rect(i)
            if enabled and i == self._hover:
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(255, 255, 255, 150))
                p.drawRoundedRect(r, 10, 10)
            p.setPen(QColor("#40213a") if enabled else QColor(64, 33, 58, 110))
            if checked is not None:
                # Voce "toggle": mostra una casellina con spunta a sinistra della voce
                box = QRectF(r.x() + 14, r.center().y() - 7, 14, 14)
                p.save()
                p.setRenderHint(QPainter.Antialiasing)
                pen = QPen(QColor("#40213a") if enabled else QColor(64, 33, 58, 110), 1.6)
                p.setPen(pen)
                p.setBrush(QColor(255, 255, 255, 200))
                p.drawRoundedRect(box, 3, 3)
                if checked:
                    p.setPen(QPen(QColor("#40213a"), 2.4))
                    p.drawLine(QPointF(box.x() + 3, box.y() + 7.5),
                               QPointF(box.x() + 5.8, box.y() + 11))
                    p.drawLine(QPointF(box.x() + 5.8, box.y() + 11),
                               QPointF(box.x() + 11.5, box.y() + 3.5))
                p.restore()
                p.setFont(check_font)
                p.drawText(r.adjusted(36, 0, -10, 0), Qt.AlignVCenter | Qt.AlignLeft, text)
                p.setFont(self._font)
            else:
                p.drawText(r.adjusted(14, 0, -10, 0), Qt.AlignVCenter | Qt.AlignLeft, text)
        # frecce: ci sono altre voci sopra/sotto (si scorre con la rotella)
        b = self._body_rect()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#40213a"))
        def tri(cy, up):
            d = 4 if up else -4
            t = QPainterPath()
            t.moveTo(b.center().x() - 5, cy + d)
            t.lineTo(b.center().x() + 5, cy + d)
            t.lineTo(b.center().x(), cy - d)
            t.closeSubpath()
            p.drawPath(t)
        if self._scroll > 0: tri(b.top() + 4, True)
        if self._scroll + self._rows < len(self.items): tri(b.bottom() - 4, False)
        p.end()
    def wheelEvent(self, event):
        extra = len(self.items) - self._rows
        if extra > 0:
            step = -1 if event.angleDelta().y() > 0 else 1
            new = max(0, min(extra, self._scroll + step))
            if new != self._scroll:
                self._scroll = new
                i = self._row_at(self.mapFromGlobal(QCursor.pos()))
                self._hover = i if (i >= 0 and self.items[i][2]) else -1
                self.update()
        event.accept()
    def mouseMoveEvent(self, event):
        i = self._row_at(event.pos())
        if i >= 0 and not self.items[i][2]: i = -1
        if i != self._hover:
            self._hover = i
            self.setCursor(Qt.PointingHandCursor if i >= 0 else Qt.ArrowCursor)
            self.update()
    def leaveEvent(self, event):
        self._hover = -1
        self.update()
    def mouseReleaseEvent(self, event):
        i = self._row_at(event.pos())
        if i >= 0 and self.items[i][2] and event.button() in (Qt.LeftButton, Qt.RightButton):
            item = self.items[i]
            cb = item[1]
            if self.keep_open and len(item) > 3:
                new = not bool(item[3])
                self.items[i] = (item[0], cb, item[2], new)
                cb(new)
                if self._refresh is not None and len(self.items) == len(self._refresh()):
                    self.items = [tuple(it[:3]) + tuple(it[3:]) if len(it) > 3 else tuple(it[:3]) for it in self._refresh()]
                self.update()
            else:
                self.close()
                QTimer.singleShot(0, cb)
        event.accept()

class BackgroundContainer(QWidget):
    def __init__(self, bg_pixmap, bg_color=None, parent=None):
        super().__init__(parent)
        self.bg_pixmap = bg_pixmap
        self.show_texture = True
        self.bg_color = QColor(bg_color or FRAME_COLOR)
        self._scaled_bg = None
        self._scaled_bg_size = None
    def set_show_texture(self, enabled: bool):
        self.show_texture = enabled
        self.update()
    def set_color(self, hex_color: str):
        self.bg_color = QColor(hex_color)
        self.update()
    def paintEvent(self, event):
        painter = QPainter(self)
        if self.show_texture and self.bg_pixmap and not self.bg_pixmap.isNull():
            if self._scaled_bg is None or self._scaled_bg_size != self.size():
                self._scaled_bg = self.bg_pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                self._scaled_bg_size = self.size()
            scaled = self._scaled_bg
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            painter.fillRect(self.rect(), self.bg_color)
        super().paintEvent(event)

def grip_sector_region(edges: str, w: int, h: int) -> QRegion:
    """Spicchio angolare dell'ellisse (angoli nelle coordinate normalizzate: 0 = destra, y verso il basso)."""
    lo, hi = ResizeGrip.SECTORS[edges]
    a, b = w / 2.0, h / 2.0
    pts = [QPoint(int(round(a)), int(round(b)))]
    steps = 4
    for i in range(steps + 1):
        th = math.radians(lo + (hi - lo) * i / steps)
        pts.append(QPoint(int(round(a + 3.0 * a * math.cos(th))), int(round(b + 3.0 * b * math.sin(th)))))
    return QRegion(QPolygon(pts))

class ResizeGrip(QWidget):
    """Maniglia di ridimensionamento: occupa l'anello del bordo nero, limitata a uno spicchio dell'ellisse
    (lati: 60 gradi, angoli: 30 gradi). Larghezza e altezza si modificano quindi in modo indipendente."""
    THICKNESS = FRAME_BORDER_WIDTH
    SECTORS = {
        "r": (-30, 30), "br": (30, 60), "b": (60, 120), "bl": (120, 150),
        "l": (150, 210), "tl": (210, 240), "t": (240, 300), "tr": (300, 330),
    }
    _CURSORS = {
        "l": Qt.SizeHorCursor, "r": Qt.SizeHorCursor,
        "t": Qt.SizeVerCursor, "b": Qt.SizeVerCursor,
        "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
        "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor,
    }
    resize_started = pyqtSignal(str, QPoint)
    resize_moved = pyqtSignal(QPoint)
    resize_ended = pyqtSignal()
    def __init__(self, edges, parent):
        super().__init__(parent)
        self.edges = edges
        self.setCursor(self._CURSORS[edges])
        self._active = False
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._active = True
            self.resize_started.emit(self.edges, event.globalPos())
            event.accept()
        else: event.ignore()
    def mouseMoveEvent(self, event):
        if self._active:
            self.resize_moved.emit(event.globalPos())
            event.accept()
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._active:
            self._active = False
            self.resize_ended.emit()
            event.accept()
    def hideEvent(self, event):
        if self._active:
            self._active = False
            self.resize_ended.emit()
        super().hideEvent(event)

class RestoreStub(QWidget):
    restore_requested = pyqtSignal()
    def __init__(self):
        super().__init__(None, Qt.Window)
        self.setWindowTitle("🐾 Neko Player")
        self.resize(220, 60)
        lay = QVBoxLayout(self)
        lbl = QLabel("Neko Player è minimizzato")
        lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(lbl)
        self._armed = False
    def minimize(self):
        self._armed = False
        self.showMinimized()
        QTimer.singleShot(400, lambda: setattr(self, "_armed", True))
    def changeEvent(self, event):
        if (event.type() == QEvent.WindowStateChange and self._armed
                and self.isVisible() and not self.isMinimized()):
            self.restore_requested.emit()
        super().changeEvent(event)

# ---------------------------------------------------------------------------
# Sottotitoli karaoke: legge il <video>_karaoke.json prodotto da hira.py (trascrivi)
# e mostra sul video kanji, hiragana e traduzione, illuminati man mano che si parla.
# ---------------------------------------------------------------------------
SUB_SPOKEN = "#ffb454"   # parte gia' pronunciata
SUB_UNSPOKEN = "#ffffff"  # parte non ancora pronunciata
SUB_CURRENT = "#ff2a2a"   # picco di rosso della lettera/token pronunciato in questo istante
SUB_TRAD_CURRENT = "#00b0ff"   # azzurro intenso: parte della traduzione che corrisponde al token rosso
SUB_TRAD = "#bfe3ff"     # traduzione non ancora illuminata


_RE_TS_SUB = re.compile(
    r"^\s*\[\s*\d+\s*[|\uff5c:/]\s*[\d:.,]+\s*\]\s*"
    r"|^\s*\[\s*[\d:.,]+\s*(?:->|-->|\u2192)\s*\[?\s*[\d:.,]+\s*\]\s*")


def mix_colore(c1, c2, t):
    """Interpola linearmente due colori '#rrggbb' (t=0 -> c1, t=1 -> c2)."""
    t = min(1.0, max(0.0, t))
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def l16(s):
    """Lunghezza in unita' UTF-16 (le posizioni di QTextCursor usano queste)."""
    return len(s.encode("utf-16-le")) // 2


def allinea_traduzione(t0, t1, trad):
    """Riallinea i tempi delle righe tradotte a quelli dei token giapponesi.

    Le righe nascono da un secondo passaggio di whisper con tempi propri (di solito
    in anticipo rispetto alle parole giapponesi). Ogni token giapponese viene
    assegnato alla riga con cui si sovrappone di piu' (o alla piu' vicina) e la riga
    appare quando parte il suo primo token e scompare quando finisce l'ultimo.
    Righe senza token vicini: tempi originali. Ripetibile senza effetti collaterali."""
    if not trad or not trad.get("righe") or not t0:
        return trad
    if trad.get("fissa"):      # tempi decisi dall'utente in hira (timestamp nel testo): non toccarli
        return trad
    ta, tb = list(trad["t0"]), list(trad["t1"])
    n = len(ta)
    if n == 0 or not (len(tb) == n == len(trad["righe"])):
        return trad

    assegn = [[] for _ in range(n)]
    for j in range(len(t0)):
        a, b = t0[j], t1[j]
        m = (a + b) / 2.0
        k = bisect_right(ta, m) - 1
        cand = range(max(0, k - 2), min(n, k + 3))
        best, best_ov = None, 0.0
        for i in cand:
            ov = min(b, tb[i]) - max(a, ta[i])
            if ov > best_ov:
                best, best_ov = i, ov
        if best is None:
            best = min(cand, key=lambda i: 0.0 if ta[i] <= m <= tb[i] else min(abs(m - ta[i]), abs(m - tb[i])))
        assegn[best].append(j)

    na, nb = list(ta), list(tb)
    for i in range(n):
        if assegn[i]:
            na[i] = t0[assegn[i][0]]
            nb[i] = t1[assegn[i][-1]]
    prev = 0.0
    for i in range(n):
        na[i] = max(na[i], prev)
        nb[i] = max(nb[i], na[i])
        prev = na[i]

    out = dict(trad)
    out["t0"], out["t1"] = na, nb
    return out


def karaoke_espandi(d):
    """Da formato v2 a v1 (quello usato internamente). I dati v1 passano invariati."""
    if d.get("versione", 1) < 2:
        return d
    if "s" not in d:
        raise ValueError("File karaoke non valido: manca 's'.")

    orig, hira = [], []
    span_o, span_h, t0, t1 = [], [], [], []
    tok_o, tok_h, tok_seg = [], [], []
    po = ph = 0

    for n, (pre, toks) in enumerate(d["s"]):
        orig.append(pre)
        hira.append(pre)
        po += l16(pre)
        ph += l16(pre)
        for e in toks:
            if len(e) == 3:
                o, a, b = e
                h = o
            else:
                o, h, a, b = e
            orig.append(o)
            hira.append(h)
            span_o.append((po, po + l16(o)))
            span_h.append((ph, ph + l16(h)))
            po += l16(o)
            ph += l16(h)
            t0.append(a)
            t1.append(b)
            tok_o.append(o)
            tok_h.append(h)
            tok_seg.append(n)

    out = {k: v for k, v in d.items() if k not in ("s", "tr", "trf", "trn")}
    if d.get("tr"):
        out["trad"] = dict(
            righe=[e[0] for e in d["tr"]],
            t0=[e[1] for e in d["tr"]],
            t1=[e[2] for e in d["tr"]],
        )
        if d.get("trf"):
            out["trad"]["fissa"] = True
        if d.get("trn"):
            out["trad"]["num"] = True
    if out.get("trad"):
        out["trad"] = allinea_traduzione(t0, t1, out["trad"])
    out.update(
        orig="".join(orig), hira="".join(hira),
        span_o=span_o, span_h=span_h, t0=t0, t1=t1,
        tok=dict(o=tok_o, h=tok_h, seg=tok_seg),
    )
    return out


class KaraokeSubs:
    """Dati del karaoke: token giapponesi (kanji + hiragana) con tempi, e righe tradotte."""

    def __init__(self, d):
        for k in ("t0", "t1", "tok"):
            if k not in d:
                raise ValueError("manca '{}'".format(k))
        self.t0 = [float(x) for x in d["t0"]]
        self.t1 = [float(x) for x in d["t1"]]
        tok = d["tok"]
        self.o, self.h, self.seg = list(tok["o"]), list(tok["h"]), list(tok["seg"])
        n = len(self.t0)
        if not (n and len(self.t1) == len(self.o) == len(self.h) == len(self.seg) == n):
            raise ValueError("dati incoerenti")

        self.seg_range = {}
        for i, sg in enumerate(self.seg):
            r = self.seg_range.setdefault(sg, [i, i + 1])
            r[1] = i + 1

        # peso cumulato del giapponese: serve ad allineare l'illuminazione della traduzione
        self.cw = [0.0]
        for h in self.h:
            self.cw.append(self.cw[-1] + (float(len(h)) if any(c.isalnum() for c in h) else 0.3))

        tr = d.get("trad") or {}
        self.tr = list(tr.get("righe", []))
        self.t0_t = [float(x) for x in tr.get("t0", [])]
        self.t1_t = [float(x) for x in tr.get("t1", [])]
        if not (len(self.tr) == len(self.t0_t) == len(self.t1_t)):
            self.tr, self.t0_t, self.t1_t = [], [], []
        # il testo mostrato non deve contenere timestamp ([N|tempo] o [inizio -> fine])
        self.tr = [_RE_TS_SUB.sub("", t).strip() for t in self.tr]
        # la ricerca per tempo (bisect) richiede inizi non decrescenti
        prev = 0.0
        for i in range(len(self.t0_t)):
            self.t0_t[i] = max(self.t0_t[i], prev)
            self.t1_t[i] = max(self.t1_t[i], self.t0_t[i])
            prev = self.t0_t[i]

    @classmethod
    def from_file(cls, path):
        with open(path, encoding="utf-8") as fh:
            return cls(karaoke_espandi(json.load(fh)))

    def _prog_jp(self, t):
        i = bisect_right(self.t0, t) - 1
        if i < 0:
            return 0.0
        d = self.t1[i] - self.t0[i]
        if t >= self.t1[i] or d <= 0:
            return self.cw[i + 1]
        return self.cw[i] + (self.cw[i + 1] - self.cw[i]) * (t - self.t0[i]) / d

    def _frazione_trad(self, i, t):
        """Frazione della riga tradotta i da illuminare: stessa proporzione di giapponese
        illuminato nell'intervallo della riga (a tempo se non c'e' giapponese)."""
        a, b = self.t0_t[i], self.t1_t[i]
        if t <= a:
            return 0.0
        if t >= b or b <= a:
            return 1.0
        ca, cb = self._prog_jp(a), self._prog_jp(b)
        f = (self._prog_jp(t) - ca) / (cb - ca) if cb - ca > 1e-9 else (t - a) / (b - a)
        return min(1.0, max(0.0, f))

    def _intervallo_trad(self, i, idx):
        """Intervallo [c0, c1) di caratteri della riga tradotta i che corrisponde al token giapponese
        idx (stessa proporzione usata per l'illuminazione); esteso a parole intere se il testo ha spazi."""
        testo = self.tr[i]
        n = len(testo)
        a, b = self.t0_t[i], self.t1_t[i]
        ca, cb = self._prog_jp(a), self._prog_jp(b)
        if cb - ca > 1e-9:
            f0 = (self.cw[idx] - ca) / (cb - ca)
            f1 = (self.cw[idx + 1] - ca) / (cb - ca)
        elif b > a:
            f0 = (self.t0[idx] - a) / (b - a)
            f1 = (self.t1[idx] - a) / (b - a)
        else:
            return 0, 0
        c0 = int(round(n * min(1.0, max(0.0, f0))))
        c1 = int(round(n * min(1.0, max(0.0, f1))))
        if c1 <= c0:
            return c0, c0
        if " " in testo:
            while 0 < c0 < n and not testo[c0 - 1].isspace() and not testo[c0].isspace():
                c0 -= 1
            while 0 < c1 < n and not testo[c1 - 1].isspace() and not testo[c1].isspace():
                c1 += 1
            if c1 > c0 and testo[c1 - 1].isspace():
                c1 -= 1                   # niente spazio finale evidenziato
        return c0, c1

    ZOOM_PASSI = 16      # a quanti passi si divide la durata di un token (ridisegni di zoom e colore)
    ZOOM_AMPIEZZA = 0.25  # ingrandimento massimo: +25%

    @staticmethod
    def _liscia(x):
        x = min(1.0, max(0.0, x))
        return x * x * (3.0 - 2.0 * x)

    def _env(self, passo):
        """Intensita' 0..1 al passo 'passo': sale in fretta, tiene, poi scende."""
        u = (passo + 0.5) / self.ZOOM_PASSI
        return self._liscia(u / 0.25) if u < 0.5 else self._liscia((1.0 - u) / 0.25)

    def _scala_zoom(self, passo):
        """Scala della lettera al passo 'passo': sale in fretta, tiene, poi torna normale."""
        return 1.0 + self.ZOOM_AMPIEZZA * self._env(passo)

    def _colore_traduzione(self, passo):
        """Azzurro intenso del tratto di traduzione in corso: stessa curva del rosso dei kanji."""
        u = (passo + 0.5) / self.ZOOM_PASSI
        base = SUB_TRAD if u < 0.5 else SUB_SPOKEN
        return mix_colore(base, SUB_TRAD_CURRENT, self._env(passo))

    def _colore_corrente(self, passo):
        """Colore del token in riproduzione: dal bianco sale al rosso, poi si raffredda
        verso l'arancione dei token gia' pronunciati (nessun salto di colore agli estremi)."""
        u = (passo + 0.5) / self.ZOOM_PASSI
        base = SUB_UNSPOKEN if u < 0.5 else SUB_SPOKEN
        return mix_colore(base, SUB_CURRENT, self._env(passo))

    def chiave(self, t, zoom=False):
        """Stato dei sottotitoli al tempo t (serve a ridisegnare solo quando cambia)."""
        k_jp = k_tr = k_z = k_c = None
        idx = bisect_right(self.t0, t) - 1
        if idx >= 0:
            in_corso = t <= self.t1[idx]
            if in_corso or t - self.t1[idx] <= 1.5:
                k_jp = (idx, in_corso)
            if in_corso and self.o[idx].strip():
                d = self.t1[idx] - self.t0[idx]
                u = (t - self.t0[idx]) / d if d > 0 else 0.5
                k_c = (idx, min(self.ZOOM_PASSI - 1, max(0, int(u * self.ZOOM_PASSI))))
                if zoom:
                    k_z = k_c
        if self.tr:
            it = bisect_right(self.t0_t, t) - 1
            if it >= 0 and t <= self.t1_t[it] + 1.5:
                c0 = c1 = 0
                if k_c is not None:
                    c0, c1 = self._intervallo_trad(it, k_c[0])
                if c1 > c0:
                    k_tr = (it, c0, c1)       # tratto azzurro = dove i kanji sono rossi
                else:
                    k_tr = (it, int(round(len(self.tr[it]) * self._frazione_trad(it, t))), None)
        return (k_jp, k_tr, k_z, k_c)

    def _riga(self, tok, a, b, idx, nascondi=None, corrente=None):
        out = []
        for i in range(a, b):
            col = SUB_SPOKEN if i <= idx else SUB_UNSPOKEN
            if i == idx and corrente:
                col = corrente           # token in riproduzione: sfumatura verso il rosso
            if i == nascondi and tok[i].strip():
                col = "transparent"      # la lettera in zoom la disegna l'overlay
            out.append('<span style="color:%s;">%s</span>' % (col, escape(tok[i])))
        return '<div style="white-space: pre-wrap;">%s</div>' % "".join(out)

    def html(self, k):
        k_jp, k_tr, k_z, k_c = k
        nascondi = k_z[0] if k_z else None
        corrente = self._colore_corrente(k_c[1]) if k_c else None
        righe = []
        if k_jp is not None:
            idx = k_jp[0]
            a, b = self.seg_range[self.seg[idx]]
            righe.append(self._riga(self.o, a, b, idx, nascondi, corrente))
            if self.h[a:b] != self.o[a:b]:   # niente riga hiragana se identica al testo
                righe.append(self._riga(self.h, a, b, idx, nascondi, corrente))
        if k_tr is not None:
            testo = self.tr[k_tr[0]]
            out = ""
            if k_tr[2] is not None and k_c:
                c0, c1 = k_tr[1], k_tr[2]
                parti = ((testo[:c0], SUB_SPOKEN),
                         (testo[c0:c1], self._colore_traduzione(k_c[1])),
                         (testo[c1:], SUB_TRAD))
            else:
                cut = k_tr[1]
                parti = ((testo[:cut], SUB_SPOKEN), (testo[cut:], SUB_TRAD))
            for frag, col in parti:
                if frag:
                    out += '<span style="color:%s;">%s</span>' % (col, escape(frag))
            righe.append('<div style="white-space: pre-wrap;">%s</div>' % out)
        return "".join(righe)

    def zoom_tokens(self, k):
        """Lettere da disegnare ingrandite: [(blocco, posizione UTF-16 nel blocco, testo, scala)].
        I blocchi sono le righe del html(): kanji, poi (se diversa) hiragana."""
        k_z = k[2]
        if not k_z or k[0] is None:
            return ()
        idx, passo = k_z
        a, b = self.seg_range[self.seg[idx]]
        sc = self._scala_zoom(passo)
        col = self._colore_corrente(passo)
        out = [(0, l16("".join(self.o[a:idx])), self.o[idx], sc, col)]
        if self.h[a:b] != self.o[a:b]:
            out.append((1, l16("".join(self.h[a:idx])), self.h[idx], sc, col))
        return tuple(out)


class SubtitleOverlay(QWidget):
    """Strato trasparente sopra il video (ignora il mouse) che disegna i sottotitoli con contorno nero."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.NoFocus)
        self._html = ""
        self._zoom = ()
        self._cache = None
        parent.installEventFilter(self)
        self.setGeometry(parent.rect())
        self.hide()

    def eventFilter(self, obj, ev):
        if obj is self.parent() and ev.type() == QEvent.Resize:
            self.setGeometry(self.parent().rect())
        return False

    def set_html(self, html, zoom=()):
        if html == self._html and zoom == self._zoom:
            return
        if html != self._html:
            self._cache = None
        self._html = html
        self._zoom = tuple(zoom)
        self.setVisible(bool(html))
        if html:
            self.raise_()
        self.update()

    def _doc(self, html, larg, px):
        d = QTextDocument()
        f = QFont(self.font())
        f.setPixelSize(px)
        f.setBold(True)
        d.setDefaultFont(f)
        o = QTextOption(Qt.AlignHCenter)
        o.setWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        d.setDefaultTextOption(o)
        d.setTextWidth(larg)
        d.setHtml(html)
        return d

    def paintEvent(self, event):
        if not self._html:
            return
        w, h = self.width(), self.height()
        win = self.window()
        rect_mode = win.isFullScreen() or win.isMaximized()
        px = max(13, min(int(h * 0.05), 46))
        y_bottom = h * (0.95 if rect_mode else 0.86)
        # nella finestra ellittica il testo sta dentro la corda dell'ellisse all'altezza dell'ultima riga
        larg = int(w * 0.92) if rect_mode else int(2 * ellipse_half_width(w, h, y_bottom) * 0.9)
        if larg < 40:
            return
        chiave = (self._html, larg, px)
        if self._cache is None or self._cache[0] != chiave:
            nero = re.sub(r"color:\s*#[0-9a-fA-F]{6}", "color:#000000", self._html)
            self._cache = (chiave, self._doc(self._html, larg, px), self._doc(nero, larg, px))
        _, doc, doc_nero = self._cache
        x = (w - larg) / 2.0
        y = y_bottom - doc.size().height()
        r = max(2, px // 12)
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing)
        for dx in (-r, 0, r):
            for dy in (-r, 0, r):
                if dx or dy:
                    p.save()
                    p.translate(x + dx, y + dy)
                    doc_nero.drawContents(p)
                    p.restore()
        p.translate(x, y)
        doc.drawContents(p)
        p.translate(-x, -y)
        if self._zoom:
            self._disegna_zoom(p, doc, x, y, px, r)
        p.end()

    def _disegna_zoom(self, p, doc, x, y, px, r):
        """Disegna ingrandita (con contorno nero) la lettera in riproduzione: nel testo e' trasparente."""
        f = QFont(self.font())
        f.setPixelSize(px)
        f.setBold(True)
        p.setRenderHint(QPainter.Antialiasing)
        for blocco, off, testo, sc, col in self._zoom:
            b = doc.findBlockByNumber(blocco)
            lay = b.layout() if b.isValid() else None
            if lay is None or lay.lineCount() == 0 or not testo.strip():
                continue
            pos = min(off, max(0, b.length() - 1))
            riga = lay.lineForTextPosition(pos)
            if not riga.isValid():
                continue
            fine = min(pos + l16(testo), riga.textStart() + riga.textLength())
            x0 = riga.cursorToX(pos)[0]
            x1 = riga.cursorToX(fine)[0]
            base = QPointF(x + lay.position().x() + x0, y + lay.position().y() + riga.y() + riga.ascent())
            cx = base.x() + (x1 - x0) / 2.0
            cy = base.y() - riga.ascent() * 0.4
            path = QPainterPath()
            path.addText(QPointF(0, 0), f, testo)
            p.save()
            p.translate(cx, cy)
            p.scale(sc, sc)
            p.translate(-cx, -cy)
            p.translate(base)
            p.strokePath(path, QPen(QColor("#000000"), 2 * r + 1, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.fillPath(path, QColor(col))
            p.restore()


class ClockOverlay(QWidget):
    """Orologio (ora digitale con quadrante e lancette) mostrato sul video in modalita' minimal quando non c'e' nessun video e il mouse e' lontano
    dai controlli. Ignora il mouse; si aggiorna ogni secondo solo mentre e' visibile."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.NoFocus)
        self.active = False
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self.update)
        parent.installEventFilter(self)
        self.setGeometry(parent.rect())
        self.hide()

    def eventFilter(self, obj, ev):
        if obj is self.parent() and ev.type() == QEvent.Resize:
            self.setGeometry(self.parent().rect())
        return False

    def set_active(self, on):
        self.active = bool(on)
        if self.active:
            self.setGeometry(self.parent().rect())
            self.show()
            self.raise_()
            self._timer.start()
        else:
            self._timer.stop()
            self.hide()

    def paintEvent(self, event):
        w, h = self.width(), self.height()
        if w < 40 or h < 30: return
        adesso = QTime.currentTime()
        ora = adesso.toString("HH:mm")
        px = max(14, int(min(w * 0.28, h * 0.34)))
        f = QFont(resolve_ui_font_family())
        f.setBold(True)
        f.setPixelSize(px)
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing)
        p.setRenderHint(QPainter.Antialiasing)
        # quadrante: tacche delle ore e lancette (ore, minuti, secondi) che girano con l'ora; le tacche stanno dietro le cifre, le lancette sopra
        cx, cy = w / 2.0, h / 2.0
        R = min(w, h) / 2.0 * 0.86
        for i in range(12):
            a = math.radians(i * 30)
            dx, dy = math.sin(a), -math.cos(a)
            r1 = R * (0.86 if i % 3 == 0 else 0.92)
            p.setPen(QPen(QColor(255, 255, 255, 120 if i % 3 == 0 else 70), max(1.2, R * (0.028 if i % 3 == 0 else 0.018)),
                          Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(cx + dx * r1, cy + dy * r1), QPointF(cx + dx * R, cy + dy * R))
        sec = adesso.second()
        ang_s = sec * 6.0
        ang_m = (adesso.minute() + sec / 60.0) * 6.0
        ang_h = ((adesso.hour() % 12) + adesso.minute() / 60.0) * 30.0

        def lancetta(ang, lung, spess, col, coda=0.0):
            a = math.radians(ang)
            dx, dy = math.sin(a), -math.cos(a)
            a0 = QPointF(cx - dx * coda, cy - dy * coda)
            a1 = QPointF(cx + dx * lung, cy + dy * lung)
            o = max(1.0, R * 0.03)
            p.setPen(QPen(QColor(0, 0, 0, 110), spess, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(a0 + QPointF(o, o), a1 + QPointF(o, o))
            p.setPen(QPen(col, spess, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(a0, a1)

        p.setFont(f)
        # centratura sull'inchiostro reale delle cifre (non sul riquadro del font, che ha spazio sopra e sotto)
        tr = QFontMetrics(f).tightBoundingRect(ora)
        x = w / 2.0 - (tr.left() + tr.right()) / 2.0
        y = h / 2.0 - (tr.top() + tr.bottom()) / 2.0      # y della linea di base
        off = max(1, px // 22)
        p.setPen(QColor(0, 0, 0, 150))
        p.drawText(QPointF(x + off, y + off), ora)
        p.setPen(QColor(255, 255, 255, 235))
        p.drawText(QPointF(x, y), ora)
        # le lancette stanno sopra le cifre
        lancetta(ang_h, R * 0.50, max(2.0, R * 0.075), QColor(0, 0, 255, 185))
        lancetta(ang_m, R * 0.76, max(1.6, R * 0.05), QColor(0, 0, 255, 185))
        lancetta(ang_s, R * 0.84, max(1.0, R * 0.02), QColor(255, 42, 42, 230), coda=R * 0.15)
        p.end()


class NekoPlayer(QMainWindow):
    _fs = False
    _normal_geo = None
    _min = False
    def __init__(self, initial_video=None):
        super().__init__()
        flags = Qt.Window | Qt.FramelessWindowHint
        if BYPASS_WM: flags |= Qt.X11BypassWindowManagerHint
        self.setWindowFlags(flags)
        self.setWindowTitle("🐾 Neko Player - Anime AI Upscaler")
        self.resize(*legacy_to_ellipse_size(740, 520))
        self.is_slider_dragged = False
        self.total_duration_sec = 0.0
        self.is_muted = False
        self.current_volume = 80
        self.show_osd = False
        self.subs = None            # KaraokeSubs del video corrente (se esiste il _karaoke.json)
        self.subs_enabled = True    # spunta nel menu contestuale
        self.sub_zoom = True        # zoom leggero sulla lettera in riproduzione (spunta nel menu)
        self.always_on_top = ALWAYS_ON_TOP     # spunta nel menu: finestra sopra tutte le altre
        self.vsync_enabled = self._leggi_config_bool("vsync_display", True)   # anti-tearing (spunta nel menu, al riavvio)
        self._audio_only = False    # file solo audio: schermo nero + onda dei dB + karaoke
        self._audio_path = None
        self._env_threads = []
        self._wave_anchor = None
        self._wave_last_t = 0.0
        self._sub_key = None
        self.show_bg_texture = True
        self._last_hud_update = 0.0
        self._audio_last_pos = None
        self._audio_last_change = 0.0
        self._ears_key = None
        self._ears_force_raise = False
        self._whiskers_key = None
        self.last_video_dir = ""
        self.last_model_dir = ""
        self.active_shaders = []   # nomi dei file in SHADERS_DIR attivi, nell'ordine in cui sono stati attivati (= ordine di applicazione)
        self.shader_force_when = False   # ignora le condizioni //!WHEN degli shader mpv
        self._file_icons = None
        self.theme_color = FRAME_COLOR
        self.bg_pixmap = None
        if os.path.isfile(BG_IMAGE_PATH): self.bg_pixmap = QPixmap(BG_IMAGE_PATH)
        self.container = BackgroundContainer(self.bg_pixmap, self.theme_color)
        self.setCentralWidget(self.container)
        self._video_aspect = None
        self._video_fill_mode = "height"  # "height": adatta il video all'altezza; "width": alla larghezza; "contain": angoli dentro l'ellisse
        self._last_display_frame = None   # ultima cornice mostrata, per ridisegnarla al volo
        # Modalità "minimal": dimensioni della finestra separate, salvate in config come window_minimal_*
        self._minimal_on = False              # stato del toggle (spunta nel menu col tasto destro)
        self._minimal_size = None             # (w, h) minimal caricate/salvate da config
        self._normal_size_before_minimal = None  # dimensioni originali da ripristinare al toggle off
        self._normal_pos_before_minimal = None   # posizione originale da ripristinare al toggle off
        # Timestamp di ripresa per video: {percorso_video: millisecondi} (persistito in config)
        self.resume_positions = {}
        self._controls_visible = True
        self._controls_hover = False
        self._overlay_hide_timer = QTimer(self)
        self._overlay_hide_timer.setSingleShot(True)
        self._overlay_hide_timer.setInterval(1400)
        self._overlay_hide_timer.timeout.connect(self._hide_overlay)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)
        self._save_timer.timeout.connect(self._write_settings)
        self.setFocusPolicy(Qt.StrongFocus)
        self._drag_active = False
        self._drag_cursor0 = None
        self._drag_pos0 = None
        self._drag_fg0 = None
        self._drag_fg = None
        self._resize_active = False
        self._rs_edges = ""
        self._rs_cursor0 = None
        self._rs_geo0 = None
        self._shape_key = None     # (larghezza, altezza, schermo intero) dell'ultima forma applicata
        self._shape_full = None
        self._grips_key = None
        self.ear_left = CatEar(is_left=True)
        self.ear_right = CatEar(is_left=False)
        self._ear_rel_l = QPoint(0, 0)
        self._ear_rel_r = QPoint(0, 0)
        self.whiskers_left = CatWhiskers(is_left=True)
        self.whiskers_right = CatWhiskers(is_left=False)
        self.whiskers_left.clicked.connect(self.on_whiskers_clicked)
        self.whiskers_right.clicked.connect(self.on_whiskers_clicked)
        for _wh in (self.whiskers_left, self.whiskers_right):
            _wh.pull_window_moved.connect(self.on_whisker_pull_moved)
            _wh.pull_ended.connect(self.on_whisker_pull_ended)
        self._shaking = False
        self._shake_origin = None
        self._shake_t0 = 0.0
        self._shake_ph0 = 0.0          # istante d'inizio della vibrazione (la fase continua senza salti)
        self._shake_hold = False       # baffo tenuto premuto: la vibrazione resta a piena ampiezza finché non si rilascia
        self._shake_last_off = QPoint(0, 0)
        self._shake_timer = QTimer(self)
        self._shake_timer.setInterval(8)
        self._shake_timer.timeout.connect(self._shake_tick)
        # Trascina e rilascia: finestra principale + overlay (sono finestre separate e intercetterebbero il drop)
        self.setAcceptDrops(True)
        for w in (self.ear_left, self.ear_right):
            w.setAcceptDrops(True)
        self.ear_timer = QTimer(self)
        self.ear_timer.setInterval(8)
        self.ear_timer.timeout.connect(self.sync_ears_position)
        self.ear_timer.start()
        self.audio_player = QMediaPlayer(self)
        self.audio_player.mediaStatusChanged.connect(self._on_audio_status)
        self.worker = VideoProcessorThread()
        self.worker.frame_ready.connect(self.update_video_frame)
        self.worker.position_changed.connect(self.update_timeline)
        self.worker.playback_finished.connect(self.on_video_finished)
        self.worker.loop_restarted.connect(self.on_loop_restarted)
        self.worker.status_message.connect(lambda m: self.status.showMessage(m))
        self.worker.shader_failed.connect(self._on_shader_failed)
        self.audio_clock_timer = QTimer(self)
        self.audio_clock_timer.setInterval(15)
        self.audio_clock_timer.timeout.connect(self._update_audio_clock)
        self.audio_clock_timer.start()
        self.init_ui()
        self.apply_theme_color(self.theme_color)
        self.load_settings()
        self._mouse_poll_timer = QTimer(self)
        self._mouse_poll_timer.setInterval(250)
        self._mouse_poll_timer.timeout.connect(self._update_overlay_visibility)
        self._mouse_poll_timer.start()
        QApplication.instance().installEventFilter(self)
        self._restore_stub = RestoreStub()     # serve solo senza window manager, ma esiste sempre (la spunta cambia a runtime)
        self._restore_stub.restore_requested.connect(self._restore_from_stub)
        if BYPASS_WM:
            area = QApplication.primaryScreen().availableGeometry()
            self.move(area.center() - QPoint(self.width() // 2, self.height() // 2))
        if initial_video and os.path.isfile(initial_video):
            self.play_video_file(initial_video)

    def init_ui(self):
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        # V-Sync anti-tearing: il valore salvato decide quale display creare (si applica all'avvio)
        self._vsync_active = False
        if self.vsync_enabled and QOpenGLWidget is not None:
            try:
                self.video_display = VideoDisplayGL(self.container)
                self._vsync_active = True
            except Exception as e:
                print(f"[Neko Player] V-Sync non disponibile, uso il display classico: {e}")
        if not self._vsync_active:
            self.video_display = VideoDisplayLabel(self.container)
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        # Nessun minimo fisso sul video: la finestra deve potersi rimpicciolire liberamente
        # (anche sotto i 320x200). In modalità minimal il limite minimo della finestra viene
        # abbassato al volo (vedi toggle_minimal_mode); a minimal spento vale il minimo classico.
        self._VIDEO_MIN_NORMAL = QSize(320, 200)
        self._VIDEO_MIN_MINIMAL = QSize(40, 30)
        self.video_display.setMinimumSize(self._VIDEO_MIN_NORMAL)
        self.video_display.double_clicked.connect(self.toggle_fullscreen)
        self.video_display.setStyleSheet("background-color: rgba(15, 12, 18, 0.85); border: none; border-radius: 0px;")
        self.show_default_background()
        self.main_layout.addWidget(self.video_display, stretch=1)
        self.osd_label = QLabel(self.video_display)
        self.osd_label.hide()
        self.wave_overlay = AudioWaveOverlay(self.video_display)
        self._wave_timer = QTimer(self)
        self._wave_timer.setInterval(33)
        self._wave_timer.timeout.connect(self._tick_audio_mode)
        self.sub_overlay = SubtitleOverlay(self.video_display)
        self.clock_overlay = ClockOverlay(self.video_display)
        self._sub_timer = QTimer(self)
        self._sub_timer.setInterval(40)
        self._sub_timer.timeout.connect(self._update_subtitles)
        self.timeline_widget = QWidget()
        timeline_layout = QHBoxLayout()
        timeline_layout.setContentsMargins(0, 0, 0, 0)
        # due barre di avanzamento a collina affiancate (due occhi chiusi sorridenti), sincronizzate: muoverne una sposta anche l'altra
        timeline_layout.setSpacing(0)
        self.slider = EyeSlider()
        self.slider_b = EyeSlider()
        for sl in (self.slider, self.slider_b):
            sl.setRange(0, 1000)
            sl.setValue(0)
            sl.setFocusPolicy(Qt.NoFocus)
            sl.sliderPressed.connect(self.on_slider_pressed)
            sl.sliderReleased.connect(self.on_slider_released)
        def _link(src, dst):
            def f(v):
                if dst.value() != v:
                    dst.blockSignals(True)
                    dst.setValue(v)
                    dst.blockSignals(False)
                    dst.update()
            src.valueChanged.connect(f)
        _link(self.slider, self.slider_b)
        _link(self.slider_b, self.slider)
        timeline_layout.addWidget(self.slider, 1)
        timeline_layout.addWidget(self.slider_b, 1)
        self.timeline_widget.setLayout(timeline_layout)
        self.controls_widget = QWidget()
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)
        bean_font = QFont(resolve_ui_font_family())
        bean_font.setBold(True)
        bean_font.setPointSize(10)

        self.btn_load_video = QPushButton("Video")
        self.btn_load_video.setObjectName("pinkBean")
        self.btn_load_video.setToolTip("Carica File Video")
        self.btn_load_video.setFocusPolicy(Qt.NoFocus)
        self.btn_load_video.setCursor(Qt.PointingHandCursor)
        self.btn_load_video.setFont(bean_font)
        self.btn_load_video.clicked.connect(self.select_video)

        # Nuovo pulsante Denoise
        self.btn_load_denoise = QPushButton("Denoise")
        self.btn_load_denoise.setObjectName("pinkBean")
        self.btn_load_denoise.setToolTip("Carica Modello AI Denoise")
        self.btn_load_denoise.setFocusPolicy(Qt.NoFocus)
        self.btn_load_denoise.setCursor(Qt.PointingHandCursor)
        self.btn_load_denoise.setFont(bean_font)
        self.btn_load_denoise.clicked.connect(self.select_denoise_model)

        self.btn_load_model = QPushButton("Modello")
        self.btn_load_model.setObjectName("pinkBean")
        self.btn_load_model.setToolTip("Carica Modello AI Upscale")
        self.btn_load_model.setFocusPolicy(Qt.NoFocus)
        self.btn_load_model.setCursor(Qt.PointingHandCursor)
        self.btn_load_model.setFont(bean_font)
        self.btn_load_model.clicked.connect(self.select_model)

        # Nuova Checkbox Denoise
        self.chk_denoise = QCheckBox("DN")
        self.chk_denoise.setChecked(False)
        self.chk_denoise.setToolTip("DN = Denoise AI: attiva/disattiva")
        self.chk_denoise.setFocusPolicy(Qt.NoFocus)
        self.chk_denoise.stateChanged.connect(self.toggle_denoise)

        self.chk_upscale = QCheckBox("2x")
        self.chk_upscale.setChecked(True)
        self.chk_upscale.setToolTip("2x = Upscaling AI 2x: attiva/disattiva")
        self.chk_upscale.setFocusPolicy(Qt.NoFocus)
        self.chk_upscale.stateChanged.connect(self.toggle_upscale)

        self.chk_rife = QCheckBox("RIFE")
        self.chk_rife.setChecked(False)
        self.chk_rife.setToolTip("Attiva/disattiva interpolazione fotogrammi RIFE")
        self.chk_rife.setFocusPolicy(Qt.NoFocus)
        self.chk_rife.stateChanged.connect(self.toggle_rife)

        self.chk_texture = QCheckBox("BG")
        self.chk_texture.setChecked(self.show_bg_texture)
        self.chk_texture.setToolTip("BG = Sfondo: attiva/disattiva la texture sfondo.png")
        self.chk_texture.setFocusPolicy(Qt.NoFocus)
        self.chk_texture.stateChanged.connect(self.toggle_bg_texture)

        self.btn_color = QPushButton("🎨")
        self.btn_color.setObjectName("flatIcon")
        self.btn_color.setFixedSize(32, 30)
        self.btn_color.setToolTip("Cambia Colore Tema")
        self.btn_color.setCursor(Qt.PointingHandCursor)
        self.btn_color.setFont(bean_font)
        self.btn_color.setFocusPolicy(Qt.NoFocus)
        self.btn_color.clicked.connect(self.choose_theme_color)

        self.btn_play_pause = NosePlayButton()
        self.btn_play_pause.setEnabled(False)
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)

        self.btn_mute = QPushButton("🔊")
        self.btn_mute.setFixedWidth(34)
        self.btn_mute.setFocusPolicy(Qt.NoFocus)
        self.btn_mute.clicked.connect(self.toggle_mute)

        self.slider_volume = MouthVolumeSlider()
        self.slider_volume.setObjectName("volumeSlider")
        self.slider_volume.setRange(0, 100)
        self.slider_volume.setValue(self.current_volume)
        self.slider_volume.setFocusPolicy(Qt.NoFocus)
        self.slider_volume.valueChanged.connect(self.on_volume_changed)

        self.lbl_volume = QLabel(f"{self.current_volume}%")
        self.lbl_volume.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.lbl_volume.setFixedWidth(34)
        self.lbl_volume.setAlignment(Qt.AlignCenter)

        self.btn_fullscreen = QPushButton("⛶")
        self.btn_fullscreen.setObjectName("pinkBeanCompact")
        self.btn_fullscreen.setToolTip("Schermo intero")
        self.btn_fullscreen.setFixedSize(34, 32)
        self.btn_fullscreen.setCursor(Qt.PointingHandCursor)
        self.btn_fullscreen.setFont(bean_font)
        self.btn_fullscreen.setFocusPolicy(Qt.NoFocus)
        self.btn_fullscreen.clicked.connect(self.toggle_fullscreen)

        self.btn_minimize = QPushButton("🗕")
        self.btn_minimize.setObjectName("pinkBeanCompact")
        self.btn_minimize.setToolTip("Minimizza")
        self.btn_minimize.setFixedSize(34, 32)
        self.btn_minimize.setCursor(Qt.PointingHandCursor)
        self.btn_minimize.setFont(bean_font)
        self.btn_minimize.setFocusPolicy(Qt.NoFocus)
        self.btn_minimize.clicked.connect(self.minimize_window)

        self.btn_close = QPushButton("✕")
        self.btn_close.setObjectName("pinkBeanCompact")
        self.btn_close.setToolTip("Chiudi")
        self.btn_close.setFixedSize(34, 32)
        self.btn_close.setCursor(Qt.PointingHandCursor)
        self.btn_close.setFont(bean_font)
        self.btn_close.setFocusPolicy(Qt.NoFocus)
        self.btn_close.clicked.connect(self.close)

        left_box = QWidget()
        left_lay = QHBoxLayout(left_box)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(6)
        left_lay.addWidget(self.btn_load_video)
        left_lay.addWidget(self.btn_load_denoise) # Aggiunto
        left_lay.addWidget(self.btn_load_model)
        left_lay.addStretch()

        right_box = QWidget()
        right_lay = QHBoxLayout(right_box)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(6)
        right_lay.addStretch()
        for b in (self.btn_color, self.btn_fullscreen, self.btn_minimize, self.btn_close):
            right_lay.addWidget(b)

        self.nose_row = nose_row = CenteredRow(left_box, self.btn_play_pause, right_box, spacing=10)

        mouth_box = QWidget()
        mouth_lay = QHBoxLayout(mouth_box)
        mouth_lay.setContentsMargins(0, 0, 0, 0)
        mouth_lay.setSpacing(6)
        mouth_lay.addWidget(self.btn_mute)
        mouth_lay.addWidget(self.slider_volume)
        mouth_lay.addWidget(self.lbl_volume)

        ai_box = QWidget()
        ai_lay = QHBoxLayout(ai_box)
        ai_lay.setContentsMargins(0, 0, 0, 0)
        ai_lay.setSpacing(4)
        ai_lay.addStretch()
        ai_lay.addWidget(self.chk_denoise) # Aggiunto
        ai_lay.addWidget(self.chk_upscale)

        bg_box = QWidget()
        bg_lay = QHBoxLayout(bg_box)
        bg_lay.setContentsMargins(0, 0, 0, 0)
        bg_lay.setSpacing(8)
        bg_lay.addWidget(self.chk_rife)
        bg_lay.addWidget(self.chk_texture)
        bg_lay.addStretch()

        self.mouth_row = CenteredRow(ai_box, mouth_box, bg_box, spacing=8, clip_right=True)

        self.controls_panel = QWidget(self.container)
        self.controls_panel.setObjectName("controlsPanel")
        self.controls_panel.setAttribute(Qt.WA_StyledBackground, True)
        panel_lay = QVBoxLayout(self.controls_panel)
        panel_lay.setContentsMargins(10, 6, 10, 8)
        panel_lay.setSpacing(100)
        controls_layout.setSpacing(4)
        # le righe si centrano (non si stirano ai bordi) e vengono limitate alla corda dell'ellisse in _fit_rows_to_ellipse
        controls_layout.addWidget(nose_row, 0, Qt.AlignHCenter)
        controls_layout.addWidget(self.mouth_row, 0, Qt.AlignHCenter)
        self.controls_widget.setLayout(controls_layout)
        panel_lay.addWidget(self.timeline_widget, 0, Qt.AlignHCenter)   # larghezza fissata da _fit_timeline_eyes
        panel_lay.addWidget(self.controls_widget)
        self.container.setLayout(self.main_layout)
        #self.mouth_stem = MouthStemOverlay(self.controls_panel, self.slider_volume, self.btn_play_pause)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.setSizeGripEnabled(False)
        self.status.hide()

        self.ellipse_border = EllipseBorder(self, self.theme_color)

        self.grips = []
        for edges in ("t", "b", "l", "r", "tl", "tr", "bl", "br"):
            grip = ResizeGrip(edges, self)
            grip.resize_started.connect(self._resize_start)
            grip.resize_moved.connect(self._resize_move)
            grip.resize_ended.connect(self._resize_end)
            self.grips.append(grip)
        self._layout_grips()
        self.controls_panel.raise_()
        self._update_shape()

    def _overlay_zone_rect(self):
        if not hasattr(self, "controls_panel"): return QRect()
        cw, ch = self.container.width(), self.container.height()
        if getattr(self, "_minimal_on", False):
            top = max(0, ch // 2 - 30)   # in minimal il pannello parte da metà finestra
            return QRect(0, top, cw, ch - top)
        if self.isFullScreen() or self.isMaximized():
            # a tutto schermo il pannello e' stretto e centrato: la zona sensibile segue la sua posizione reale
            # (prima era una fascia larga quanto lo schermo, quindi il pannello non spariva finche' il mouse
            # restava nella parte bassa, anche lontano da esso)
            g = self.controls_panel.geometry()
            if g.width() > 0 and g.height() > 0:
                left = max(0, g.left() - 80)
                right = min(cw, g.right() + 80)
                top = max(0, g.top() - 40)
                return QRect(left, top, max(1, right - left + 1), max(1, ch - top))
        ph = self.controls_panel.sizeHint().height()
        band = max(120, ph + 46)
        top = max(0, ch - band)
        top = min(top, max(0, ch // 2 - 100))   # il pannello parte da metà finestra (occhi sopra la metà)
        return QRect(0, top, cw, ch - top)

    def _has_video(self):
        try:
            if getattr(self.worker, "running", False): return True
            if self.audio_player.state() == QMediaPlayer.PlayingState: return True
            if self.btn_play_pause.isEnabled(): return True
            return False
        except Exception: return False

    def _set_minimal_compact(self, on: bool):
        """In modalità minimal il pannello in basso mostra solo Play e la barra del volume:
        nasconde timeline, file/ai/sfondo, fullscreen/minimizza/chiudi, mute ed etichetta %.
        Il pannello resta comunque visibile quando il cursore è sopra (vedi overlay)."""
        for w in (self.timeline_widget, self.btn_load_video, self.btn_load_denoise,
                  self.btn_load_model, self.btn_color, self.chk_denoise, self.chk_upscale,
                  self.chk_rife, self.chk_texture, self.btn_fullscreen, self.btn_minimize,
                  self.btn_close, self.btn_mute, self.lbl_volume):
            try:
                if on: w.hide()
                else:  w.show()
            except Exception:
                pass

    def _set_clock(self, on):
        co = getattr(self, "clock_overlay", None)
        if co is not None and co.active != bool(on): co.set_active(on)

    def _update_overlay_visibility(self):
        if not hasattr(self, "controls_panel"): return
        self._set_clock(False)      # l'orologio compare solo nel ramo minimal senza video, piu' sotto
        # In modalità minimal: pannello compatto (solo play + volume), mostrato
        # all'hover come di consueto; senza video resta visibile finche' il mouse e' vicino
        # (se MINIMAL_CLOCK: lontano dal mouse il pannello sparisce e compare l'orologio)
        if getattr(self, "_minimal_on", False):
            self._set_minimal_compact(True)
            if not self._has_video():
                near = True
                if MINIMAL_CLOCK and not (self._drag_active or self._resize_active):
                    try:
                        local = self.container.mapFromGlobal(QCursor.pos())
                        zone = self._overlay_zone_rect()
                        near = (zone.contains(local) or
                                self.controls_panel.geometry().adjusted(-30, -30, 30, 30).contains(local))
                    except Exception: near = True
                if near:
                    self._controls_hover = True
                    if not self._controls_visible or not self.controls_panel.isVisible():
                        self._show_overlay()
                else:
                    self._controls_hover = False
                    if self._controls_visible or self.controls_panel.isVisible():
                        self._controls_visible = False
                        self.controls_panel.hide()
                        self.video_display.update()
                    self._set_clock(True)
                self._overlay_hide_timer.stop()
                return
            gp = QCursor.pos()
            inside = False
            try:
                local = self.container.mapFromGlobal(gp)
                zone = self._overlay_zone_rect()
                inside = (zone.contains(local) or self.controls_panel.geometry().adjusted(-30, -30, 30, 30).contains(local))
            except Exception: inside = True
            if inside:
                self._controls_hover = True
                if not self._controls_visible: self._show_overlay()
                self._overlay_hide_timer.start()
            else:
                self._controls_hover = False
                if self._controls_visible and not self._overlay_hide_timer.isActive():
                    self._overlay_hide_timer.start(500)
            return
        self._set_minimal_compact(False)
        if not self._has_video():
            self._controls_hover = True
            if not self._controls_visible or not self.controls_panel.isVisible(): self._show_overlay()
            self._overlay_hide_timer.stop()
            return
        gp = QCursor.pos()
        inside = False
        try:
            local = self.container.mapFromGlobal(gp)
            zone = self._overlay_zone_rect()
            inside = (zone.contains(local) or self.controls_panel.geometry().adjusted(-30, -30, 30, 30).contains(local))
        except Exception: inside = True
        if inside:
            self._controls_hover = True
            if not self._controls_visible: self._show_overlay()
            self._overlay_hide_timer.start()
        else:
            self._controls_hover = False
            if self._controls_visible and not self._overlay_hide_timer.isActive():
                self._overlay_hide_timer.start(500)

    def _show_overlay(self):
        self._controls_visible = True
        self.controls_panel.show()
        self._layout_overlay()
        self.controls_panel.raise_()
        self.video_display.update()

    def _layout_overlay(self):
        if not hasattr(self, "controls_panel"): return
        lay = self.controls_panel.layout()
        if lay is not None:
            lay.invalidate()
            lay.activate()
        cw, ch = self.container.width(), self.container.height()
        # in minimal il naso ha posizione fissa (parte da metà finestra) e la bocca sta a metà strada tra il centro del naso e il bordo inferiore
        fixed_y = self._minimal_face_layout(ch)
        hint = self.controls_panel.sizeHint()
        h = hint.height()
        if self.isFullScreen() or self.isMaximized():
            w = max(240, min(cw - 16, hint.width()))
            x = (cw - w) // 2
            y = max(8, ch - h - 8)
        else:
            margin_bottom = FRAME_BORDER_WIDTH + ELLIPSE_BAND + 8
            max_w = max(240, cw - 2 * (FRAME_BORDER_WIDTH + ELLIPSE_BAND + 12))
            w = min(max_w, max(240, hint.width()))
            x = (cw - w) // 2
            y = max(8, ch - h - margin_bottom)
            if fixed_y is None:
                y = max(8, y - self._overlay_lift(y, h))   # se la corda dell'ellisse è troppo stretta, il pannello sale un poco
        if fixed_y is None:
            # la base degli occhi parte da metà finestra e i bottoni stanno subito sotto; se non c'è abbastanza spazio sotto,
            # il pannello resta dove sarebbe stato (ancorato in basso)
            mid_y = self._eyes_mid_y(ch)
            if mid_y is not None: y = max(8, min(y, mid_y))
        if fixed_y is not None:
            w = max(1, min(cw, hint.width()))
            x = (cw - w) // 2
            self.controls_panel.setGeometry(x, fixed_y, w, h)
            self.controls_panel.raise_()
            self._fit_rows_to_ellipse(fixed_y, h)
            return
        self.controls_panel.setGeometry(x, y, w, min(h, max(0, ch - 16)))
        self.controls_panel.raise_()
        self._fit_rows_to_ellipse(y, min(h, max(0, ch - 16)))

    MINIMAL_REF_W, MINIMAL_REF_H = 203, 179   # dimensioni minimal fino a cui il layout sta nella finestra; sotto, naso e bocca traslano verso l'alto

    def _minimal_numbers(self, ch):
        """Posizioni del pannello in minimal per una finestra alta ch, senza traslazione:
        (y del pannello, spaziatura tra naso e bocca, y alta della riga bocca, altezza riga bocca)."""
        pl = self.controls_panel.layout()
        hn = self.nose_row.sizeHint().height()
        hm = self.mouth_row.sizeHint().height()
        nose_h = self.btn_play_pause.height()
        sl = self.slider_volume
        mt = pl.contentsMargins().top()
        nose_top_in_panel = mt + (hn - nose_h) // 2
        # il "centro" della bocca: a metà tra il punto centrale (giunzione) e il punto più basso degli archi
        depth = sl._depth_for_width(sl._mouth_w)
        mouth_c_in_row = (hm - sl.height()) // 2 + MouthVolumeSlider.JUNCTION_Y + depth / 2.0
        panel_y = ch // 2 - nose_top_in_panel                 # bordo alto del naso = metà finestra
        nose_center = ch / 2.0 + nose_h / 2.0
        mouth_center = (nose_center + ch) / 2.0               # metà strada tra centro naso e bordo inferiore
        nose_row_bottom = panel_y + mt + hn
        spacing = max(0, int(round(mouth_center - mouth_c_in_row - nose_row_bottom)))
        return panel_y, spacing, nose_row_bottom + spacing, hm

    def _minimal_clearance(self, W, H, y_eval):
        """Margine (px) tra la corda dell'ellisse alla quota della bocca e la larghezza minima della riga bocca."""
        inset = FRAME_BORDER_WIDTH + ELLIPSE_BAND
        chord = int(2 * ellipse_half_width(W, H, y_eval, inset)) - 20
        return chord - (self._mouth_row_base() + 96)

    def _minimal_face_layout(self, ch):
        """Modalità minimal: il naso (pulsante play) ha posizione fissa, il suo bordo alto sta a metà finestra e scende da lì;
        la bocca (barra del volume) è centrata a metà strada tra il centro del naso e il bordo inferiore della finestra.
        Sotto MINIMAL_REF_W x MINIMAL_REF_H, naso e bocca (spaziatura invariata) traslano verso l'alto quanto basta per
        mantenere lo stesso margine dal bordo dell'ellisse che c'è a quelle dimensioni, così restano nella finestra.
        Imposta la spaziatura tra le due righe e restituisce la y del pannello (None se minimal è spento)."""
        cl = self.controls_widget.layout()
        pl = self.controls_panel.layout()
        if cl is None or pl is None: return None
        if not getattr(self, "_minimal_on", False):
            if cl.spacing() != 4:
                cl.setSpacing(4)
                pl.invalidate()
                pl.activate()
            return None
        cl.setSpacing(4)
        pl.invalidate()
        pl.activate()
        panel_y, spacing, mouth_top, hm = self._minimal_numbers(ch)
        # margine di riferimento: quello che c'è alla dimensione minimal di riferimento
        _, _, ref_top, _ = self._minimal_numbers(self.MINIMAL_REF_H)
        ref_clear = self._minimal_clearance(self.MINIMAL_REF_W, self.MINIMAL_REF_H, ref_top + hm - self._mouth_eval_h(hm))
        y_eval = mouth_top + hm - self._mouth_eval_h(hm)
        W, H = self.width(), self.height()
        lift = 0
        for lift in range(0, max(0, panel_y) + 1):   # il pannello non esce dal bordo alto
            if self._minimal_clearance(W, H, y_eval - lift) >= ref_clear: break
        cl.setSpacing(spacing)
        pl.invalidate()
        pl.activate()
        return panel_y - int(lift * 0.6)   # la salita è il 60% di quella calcolata

    def _overlay_lift(self, y, h):
        """Di quanti px alzare il pannello perché la riga inferiore (con lo slider volume al minimo) stia nella corda dell'ellisse."""
        if not hasattr(self, "mouth_row") or not hasattr(self, "slider_volume"): return 0
        hm = self.mouth_row.sizeHint().height()
        need = self._mouth_row_base() + self._mouth_target()
        ch = self.container.height()
        # il pannello deve restare nella metà inferiore della finestra, e non salire più di un quinto dell'altezza
        max_lift = max(0, min(y - ch // 2, int(ch * 0.20)))
        best = 0
        for lift in range(0, max_lift + 1, 2):
            best = lift
            if self._ellipse_chord(y - lift + h - 8 - self._mouth_eval_h(hm)) >= need: break
        return best

    def _mouth_target(self):
        """Larghezza desiderata della bocca (doppia rispetto alla versione precedente)."""
        return max(120, min(640, int(self.width() * 0.44)))

    def _mouth_row_base(self):
        """Larghezza della riga bocca senza lo slider (i due gruppi di checkbox + spaziature)."""
        return self.mouth_row.sizeHint().width() - self.slider_volume.width()

    @staticmethod
    def _mouth_eval_h(row_h):
        """Come _row_eval_h, per la riga bocca: il punto centrale della bocca sta in alto nel widget."""
        return max(0.3 * row_h, row_h - 21.0)

    @staticmethod
    def _row_eval_h(row_h):
        """Altezza sopra il bordo inferiore della riga a cui si misura la corda dell'ellisse."""
        return max(0.3 * row_h, 0.5 * row_h - 8)

    def _ellipse_chord(self, y, margin=10):
        """Larghezza utile (in px) dell'ellisse visibile alla quota y (coordinate finestra), meno un margine per lato."""
        inset = FRAME_BORDER_WIDTH + ELLIPSE_BAND
        return int(2 * ellipse_half_width(self.width(), self.height(), y, inset)) - 2 * margin

    def _fit_timeline_eyes(self):
        """Gli occhi (barre a collina) si allargano seguendo la larghezza della finestra fino al limite `larghezza_occhi`
        (per ciascun occhio), con una distanza tra loro; più sono larghi, più alta è la collina.
        Ritorna True se larghezza, distanza o altezza sono cambiate (il pannello va ridisposto)."""
        if getattr(self, "_minimal_on", False) or not hasattr(self, "timeline_widget"): return False
        pl = self.controls_panel.layout()
        m = pl.contentsMargins()
        pad = EyeSlider.PAD
        inner = self.controls_panel.width() - m.left() - m.right()
        # larghezza disponibile: corda dell'ellisse alla quota della base degli occhi (metà finestra), nel pannello
        chord = self._ellipse_chord(self.height() / 2.0)
        avail = max(60, min(inner, chord))
        R = EyeSlider.GAP_RATIO
        e1 = (avail - 4 * pad) / (2.0 * (1.0 + R))                 # larghezza d'arco libera con distanza proporzionale
        if 2 * R * e1 < EyeSlider.GAP_MIN: e1 = (avail - 4 * pad - EyeSlider.GAP_MIN) / 2.0
        eye_arc = max(10.0, min(float(larghezza_occhi), e1))       # larghezza di ciascun arco, limitata a larghezza_occhi
        gap = int(max(EyeSlider.GAP_MIN, 2 * R * eye_arc))         # distanza tra i due occhi
        ew = int(round(eye_arc + 2 * pad))                         # larghezza del widget di un occhio
        tw = 2 * ew + gap
        depth = EyeSlider.depth_for_width(eye_arc)
        # gli occhi salgono dalla metà finestra: la collina non deve uscire dal bordo alto
        room = (self.container.height() - 2 * (FRAME_BORDER_WIDTH + ELLIPSE_BAND) - self.controls_widget.sizeHint().height()
                - m.top() - m.bottom() - pl.spacing() - EyeSlider.TOP - EyeSlider.BOTTOM)
        depth = max(8.0, min(depth, room))
        changed = False
        if self.timeline_widget.width() != tw or self.timeline_widget.minimumWidth() != tw:
            self.timeline_widget.setFixedWidth(tw)
            changed = True
        tl = self.timeline_widget.layout()
        if tl.spacing() != gap:
            tl.setSpacing(gap)
            changed = True
        for sl in (self.slider, self.slider_b):
            if sl.minimumWidth() != ew or sl.maximumWidth() != ew:
                sl.setFixedWidth(ew)
                changed = True
            if sl.set_arc_depth(depth): changed = True
        return changed

    def _eyes_mid_y(self, ch):
        """y del pannello per cui la base degli occhi sta a metà finestra (i bottoni della riga sotto restano subito sotto di loro);
        None se la timeline non è visibile (minimal)."""
        if getattr(self, "_minimal_on", False) or self.timeline_widget.isHidden(): return None
        mt = self.controls_panel.layout().contentsMargins().top()
        return int(round(ch // 2 +40 -(mt + self.slider.height() - EyeSlider.BOTTOM)))   # int: setGeometry non accetta float

    def _fit_rows_to_ellipse(self, py, ph):
        """Limita le due righe di bottoni alla corda dell'ellisse alla loro quota, così non escono dalla forma
        quando la finestra cambia proporzioni: restano centrate e lo slider del volume si accorcia."""
        if not hasattr(self, "mouth_row") or not hasattr(self, "nose_row"): return
        if self.isFullScreen() or self.isMaximized():
            cap_m = None
            self.mouth_row.setMaximumWidth(16777215)
            self.nose_row.setMaximumWidth(16777215)
        else:
            hm = self.mouth_row.sizeHint().height()
            hn = self.nose_row.sizeHint().height()
            yb_m = py + ph - 8          # margine inferiore interno del pannello
            yb_n = yb_m - hm - 4        # spaziatura tra le due righe
            cap_m = self._ellipse_chord(yb_m - self._mouth_eval_h(hm))
            cap_n = self._ellipse_chord(yb_n - self._row_eval_h(hn))
            self.mouth_row.setMaximumWidth(max(120, cap_m))
            self.nose_row.setMaximumWidth(max(120, cap_n))
        tl_changed = self._fit_timeline_eyes()
        changed = cap_m != getattr(self, "_mouth_row_cap", None)
        self._mouth_row_cap = cap_m
        if (changed or tl_changed) and not getattr(self, "_fitting", False):
            self._fitting = True
            try:
                self._scale_mouth()
                self._layout_overlay()   # la larghezza naturale del pannello è cambiata
            finally:
                self._fitting = False


    def _hide_overlay(self):
        if not hasattr(self, "controls_panel"): return
        if not self._has_video():
            if MINIMAL_CLOCK and getattr(self, "_minimal_on", False) and not self._controls_hover:
                return      # minimal senza video e mouse lontano: i controlli restano nascosti (c'e' l'orologio)
            if not self._controls_visible: self._show_overlay()
            return
        if self._drag_active or self._resize_active:
            self._overlay_hide_timer.start()
            return
        if getattr(self, "is_slider_dragged", False):
            self._overlay_hide_timer.start()
            return
        if self.slider.isSliderDown() or self.slider_b.isSliderDown() or getattr(self.slider_volume, "_dragging", False):
            self._overlay_hide_timer.start()
            return
        if self._controls_hover:
            self._overlay_hide_timer.start()
            return
        under = QApplication.widgetAt(QCursor.pos())
        if under is not None and (under is self.controls_panel or self.controls_panel.isAncestorOf(under)):
            self._overlay_hide_timer.start()
            return
        self._controls_visible = False
        self.controls_panel.hide()
        self.video_display.update()

    def toggle_bg_texture(self, state):
        self.show_bg_texture = (state == Qt.Checked)
        self.container.set_show_texture(self.show_bg_texture)
        self.save_settings()

    def choose_theme_color(self):
        dlg = QColorDialog(QColor(self.theme_color), self)
        dlg.setWindowTitle("Seleziona Colore Tema")
        dlg.setOption(QColorDialog.DontUseNativeDialog, True)
        dlg.setStyleSheet(STYLE_DIALOG)
        if BYPASS_WM:
            dlg.setWindowFlags(dlg.windowFlags() | Qt.X11BypassWindowManagerHint)
            dlg.setStyleSheet(STYLE_DIALOG + " QColorDialog { border: 1px solid #45475a; }")
            screen = QApplication.screenAt(self.geometry().center()) or QApplication.primaryScreen()
            dlg.move(screen.availableGeometry().center() - QPoint(dlg.sizeHint().width() // 2, dlg.sizeHint().height() // 2))
            QTimer.singleShot(50, lambda: (dlg.raise_(), dlg.activateWindow()))
        if dlg.exec_():
            col = dlg.selectedColor()
            if col.isValid():
                self.apply_theme_color(col.name())
                self.save_settings()
        if BYPASS_WM: self._grab_focus()

    def apply_theme_color(self, hex_color: str):
        self.theme_color = hex_color
        c = QColor(hex_color)
        r, g, b = c.red(), c.green(), c.blue()
        self.setStyleSheet(make_main_style(hex_color))
        if hasattr(self, "ellipse_border"): self.ellipse_border.set_color(hex_color)
        self._ears_force_raise = True
        self.sync_ears_position()
        if hasattr(self, "container"): self.container.set_color(hex_color)
        if hasattr(self, "video_display") and not self.isFullScreen():
            self.video_display.setStyleSheet(
                f"background-color: rgba({r}, {g}, {b}, 0.85); border: none; border-radius: 8px;"
            )
        if hasattr(self, "osd_label"): self.osd_label.setStyleSheet(make_osd_style(r, g, b))

    def trigger_whiskers_wiggle(self):
        self.whiskers_left.trigger_wiggle()
        self.whiskers_right.trigger_wiggle()

    def on_whiskers_clicked(self):
        self.trigger_whiskers_wiggle()
        self._shake_hold = True   # la testa vibra finché il baffo è tenuto premuto
        self.shake_head()

    def on_whisker_pull_moved(self, w: QPoint):
        """Baffo tirato oltre il 150%: la finestra (con orecchie e baffi) segue il cursore di `w` px.
        Se la testa sta vibrando, la vibrazione continua attorno alla nuova posizione."""
        if self.isFullScreen() or self.isMaximized() or self.isMinimized(): return
        if not self._drag_active:
            if self._resize_active: return
            keep = self._shaking
            origin = QPoint(self._shake_origin) if keep else None
            self._arc_drag_start(QCursor.pos(), keep_shake=keep)
            if not self._drag_active: return
            if keep:
                # posizione di partenza = quella senza lo scarto della vibrazione
                self._drag_fg0 = self._drag_fg0.translated(origin - self.pos())
                self._drag_pos0 = origin
                self._drag_fg = QRect(self._drag_fg0)
        if self._shaking:
            self._shake_origin = self._drag_pos0 + w
            self._apply_shake_pos(self._shake_last_off)
        else:
            self.move(self._drag_pos0 + w)
            self._drag_fg = self._drag_fg0.translated(w)
        self.sync_ears_position()
        QApplication.flush()

    def on_whisker_pull_ended(self):
        self._shake_hold = False
        if self._shaking: self._shake_t0 = time.monotonic()   # da qui la vibrazione si smorza
        if self._drag_active: self._arc_drag_end()
        self.trigger_whiskers_wiggle()

    def shake_head(self):
        """Fa vibrare la testa (finestra + orecchie + baffi, che la seguono) per un istante."""
        if self.isFullScreen() or self.isMaximized() or self.isMinimized() or not self.isVisible(): return
        if self._drag_active or self._resize_active: return
        now = time.monotonic()
        if not self._shaking:
            self._shake_origin = QPoint(self.pos())
            self._shaking = True
            self._shake_ph0 = now
        self._shake_t0 = now
        if not self._shake_timer.isActive(): self._shake_timer.start()

    def _stop_shake(self, restore=True):
        if not self._shaking: return
        self._shake_timer.stop()
        self._shaking = False
        origin, self._shake_origin = self._shake_origin, None
        if restore and origin is not None and not (self.isFullScreen() or self.isMaximized()):
            self.move(origin)

    def _apply_shake_pos(self, off: QPoint):
        self._shake_last_off = QPoint(off)
        target = self._shake_origin + off
        self.move(target)
        if self._drag_active:
            self._drag_fg = self._drag_fg0.translated(target - self._drag_pos0)   # orecchie e baffi seguono la vibrazione

    def _shake_tick(self):
        if not self._shaking: return self._shake_timer.stop()
        if self.isFullScreen() or self.isMaximized() or self.isMinimized():
            return self._stop_shake(restore=False)
        now = time.monotonic()
        if self._shake_hold:
            damp = 1.0   # baffo tenuto premuto: ampiezza costante
        else:
            t = now - self._shake_t0
            if t >= HEAD_SHAKE_DURATION: return self._stop_shake()
            damp = (1.0 - t / HEAD_SHAKE_DURATION) ** 2
        ph = 2.0 * math.pi * HEAD_SHAKE_FREQ * (now - self._shake_ph0)
        dx = HEAD_SHAKE_AMPLITUDE * damp * math.sin(ph)
        dy = 0.5 * HEAD_SHAKE_AMPLITUDE * damp * math.sin(ph * 1.3 + 1.0)
        self._apply_shake_pos(QPoint(int(round(dx)), int(round(dy))))

    def isFullScreen(self): return self._fs if BYPASS_WM else super().isFullScreen()

    def _set_fullscreen(self, on):
        if not BYPASS_WM:
            self.showFullScreen() if on else self.showNormal()
            return
        if on:
            self._normal_geo = QRect(self.geometry())
            screen = QApplication.screenAt(self.geometry().center()) or QApplication.primaryScreen()
            self._fs = True
            self.setGeometry(screen.geometry())
            self._grab_focus()
        else:
            self._fs = False
            if self._normal_geo is not None: self.setGeometry(self._normal_geo)

    def minimize_window(self):
        if not BYPASS_WM:
            self.showMinimized()
            return
        if self._min or self.isFullScreen(): return
        self._min = True
        self._restore_stub.minimize()
        self.hide()
        self.ear_left.hide()
        self.ear_right.hide()
        self.whiskers_left.hide()
        self.whiskers_right.hide()

    def _restore_from_stub(self):
        if not self._min: return
        self._min = False
        self._restore_stub.hide()
        self.show()
        if not BYPASS_WM: self.lower_ears_behind_window()
        self._ears_key = None
        self._whiskers_key = None
        self.sync_ears_position()
        self._grab_focus()

    def lower_ears_behind_window(self):
        try:
            self.ear_left.lower()
            self.ear_right.lower()
        except Exception: pass

    def _ensure_ears_behind(self):
        if BYPASS_WM: return
        if getattr(self, "_min", False) or self.isFullScreen(): return
        self.lower_ears_behind_window()
        self._ears_force_raise = True
        QTimer.singleShot(0, lambda: self.sync_ears_position())

    def _grab_focus(self):
        self.raise_()
        self.activateWindow()
        self._ears_force_raise = True
        self.lower_ears_behind_window()

    def _is_overlay_window(self, obj):
        return obj in (self.ear_left, self.ear_right)

    @staticmethod
    def _dropped_video_path(mime):
        if not mime.hasUrls(): return None
        db = QMimeDatabase()
        for url in mime.urls():
            if not url.isLocalFile(): continue
            p = url.toLocalFile()
            if not os.path.isfile(p): continue
            ext = os.path.splitext(p)[1].lower().lstrip(".")
            if (ext in VIDEO_EXTS or ext in AUDIO_EXTS
                    or db.mimeTypeForFile(p).name().startswith(("video/", "audio/"))):
                return p
        return None

    def _handle_drag_event(self, event):
        path = self._dropped_video_path(event.mimeData())
        if not path:
            event.ignore()
            return
        event.acceptProposedAction()
        if event.type() == QEvent.Drop:
            # apre il video dopo il ritorno dal drop, per non bloccare chi trascina
            QTimer.singleShot(0, lambda p=path: self._open_dropped_video(p))

    def _open_dropped_video(self, path):
        self.last_video_dir = os.path.dirname(path)
        self.save_settings()
        self.play_video_file(path)
        if BYPASS_WM: self._grab_focus()
        else: self.activateWindow()

    def dragEnterEvent(self, event): self._handle_drag_event(event)
    def dragMoveEvent(self, event): self._handle_drag_event(event)
    def dropEvent(self, event): self._handle_drag_event(event)

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.DragEnter, QEvent.DragMove, QEvent.Drop) and self._is_overlay_window(obj):
            self._handle_drag_event(event)
            return True
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
            if isinstance(obj, QWidget) and self._is_our_widget(obj):
                self.show_context_bubble(event.globalPos())
                return True
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            if isinstance(obj, QWidget) and self._is_our_widget(obj):
                self.ear_left.trigger_twitch()
                self.ear_right.trigger_twitch()
                self.trigger_whiskers_wiggle()
                if BYPASS_WM and not self.isActiveWindow(): self._grab_focus()
        return super().eventFilter(obj, event)

    def _is_our_widget(self, obj):
        return (
            obj.window() is self or obj is self
            or obj in (self.ear_left, self.ear_right)
        )

    def _current_frame_geometry(self):
        return self._drag_fg if self._drag_fg is not None else self.frameGeometry()

    def _arc_drag_start(self, gpos, keep_shake=False):
        if not keep_shake: self._stop_shake()
        if self.isFullScreen() or self.isMaximized() or self.isMinimized(): return
        self._drag_cursor0 = QPoint(gpos)
        self._drag_pos0 = self.pos()
        self._drag_fg0 = self.frameGeometry()
        self._drag_fg = QRect(self._drag_fg0)
        self._drag_active = True
        self.raise_()
        self.activateWindow()
        self._ears_force_raise = True

    def _arc_drag_move(self, gpos):
        if not self._drag_active: return
        delta = gpos - self._drag_cursor0
        self.move(self._drag_pos0 + delta)
        self._drag_fg = self._drag_fg0.translated(delta)
        self.sync_ears_position()
        QApplication.flush()

    def mousePressEvent(self, event: QMouseEvent):
        # Gli eventi arrivano qui solo se i widget figli (video, pannello comandi vuoto) non li hanno gestiti:
        # pulsanti, slider e maniglie di ridimensionamento mantengono quindi il loro comportamento
        if event.button() == Qt.LeftButton:
            self._arc_drag_start(event.globalPos())
            if self._drag_active:
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_active:
            self._arc_drag_move(event.globalPos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self._drag_active:
            self._arc_drag_end()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _arc_drag_end(self):
        if not self._drag_active: return
        self._drag_active = False
        self._drag_fg = None
        self._ears_key = None
        self._whiskers_key = None
        # A minimal spento, la nuova posizione diventa quella "normale" da ripristinare
        if not getattr(self, "_minimal_on", False):
            self._normal_pos_before_minimal = QPoint(self._shake_origin) if self._shaking else QPoint(self.pos())
        self.sync_ears_position()
        self.save_settings()

    def _update_shape(self):
        """Finestra unica ellittica: maschera sulla finestra, margini del contenuto e bordo.
        A schermo intero / massimizzata la forma torna rettangolare, senza bordo."""
        if not hasattr(self, "ellipse_border"): return
        full = self.isFullScreen() or self.isMaximized()
        w, h = self.width(), self.height()
        key = (w, h, full)
        if key == self._shape_key: return
        self._shape_key = key
        if full:
            self.clearMask()
            self.ellipse_border.hide()
        else:
            self.setMask(ellipse_region(w, h))
            self.ellipse_border.setGeometry(0, 0, w, h)
            self.ellipse_border.show()
            self.ellipse_border.raise_()
        if full != self._shape_full:
            self._shape_full = full
            m = 0 if full else FRAME_BORDER_WIDTH + ELLIPSE_BAND
            self.main_layout.setContentsMargins(m, m, m, m)
            self.main_layout.setSpacing(0)

    def _layout_grips(self):
        grips = getattr(self, "grips", None)
        if not grips: return
        w, h = self.width(), self.height()
        # durante il ridimensionamento la maniglia premuta tiene il mouse: le maschere si rifanno alla fine
        if (w, h) != self._grips_key and not self._resize_active:
            self._grips_key = (w, h)
            ring = ellipse_region(w, h).subtracted(ellipse_region(w, h, ResizeGrip.THICKNESS))
            for g in grips:
                g.setGeometry(0, 0, w, h)
                g.setMask(ring.intersected(grip_sector_region(g.edges, w, h)))
        for g in grips: g.raise_()

    def _update_window_chrome(self):
        if not getattr(self, "grips", None): return
        resizable = not (self.isMaximized() or self.isFullScreen())
        for g in self.grips: g.setVisible(resizable)
        if resizable: self._layout_grips()

    def _resize_start(self, edges, gpos):
        self._stop_shake()
        if self.isFullScreen() or self.isMaximized() or self.isMinimized(): return
        self._rs_edges = edges
        self._rs_cursor0 = QPoint(gpos)
        self._rs_geo0 = QRect(self.geometry())
        self._drag_fg = QRect(self._rs_geo0)
        self._resize_active = True
        self.raise_()
        self.activateWindow()

    def _resize_move(self, gpos):
        if not self._resize_active: return
        d = gpos - self._rs_cursor0
        g = self._rs_geo0
        left, top, right, bottom = g.left(), g.top(), g.right(), g.bottom()
        min_size = self.minimumSizeHint().expandedTo(self.minimumSize())
        min_w, min_h = min_size.width(), min_size.height()
        e = self._rs_edges
        if "l" in e: left = min(left + d.x(), right - min_w + 1)
        if "r" in e: right = max(right + d.x(), left + min_w - 1)
        if "t" in e: top = min(top + d.y(), bottom - min_h + 1)
        if "b" in e: bottom = max(bottom + d.y(), top + min_h - 1)
        rect = QRect(QPoint(left, top), QPoint(right, bottom))
        self._drag_fg = rect
        self.setGeometry(rect)
        self.sync_ears_position()
        QApplication.flush()

    def _resize_end(self):
        if not self._resize_active: return
        self._resize_active = False
        self._layout_grips()
        self._drag_fg = None
        self._ears_key = None
        self._whiskers_key = None
        # Con minimal attivo, le nuove dimensioni diventano quelle "minimal" salvate in config
        if getattr(self, "_minimal_on", False):
            self._minimal_size = self._normal_window_size()
        else:
            # A minimal spento, la posizione corrente diventa quella "normale"
            # da ripristinare alla prossima disattivazione di minimal
            self._normal_pos_before_minimal = QPoint(self.pos())
        self.sync_ears_position()
        self.save_settings()

    def _ears_layout(self, w: float, h: float):
        # Calcolo puro (nessun effetto collaterale). Ogni orecchio ha i DUE ANGOLI DELLA BASE esattamente sull'ellisse
        # della finestra (semiassi w/2 e h/2; y = 0 è il vertice superiore): la base è la corda che li unisce, lunga
        # quanto l'orecchio; l'orecchio ruota sull'angolo di quella corda e ha per ancora il suo punto medio. La
        # parte di base interna all'ellisse resta nascosta dietro la finestra.
        a = max(1.0, w / 2.0)
        b = max(1.0, h / 2.0)
        def top_y(x):
            t = max(-1.0, min(1.0, (x - a) / a))
            return b - b * math.sqrt(max(0.0, 1.0 - t * t))
        def chord_x(xc, ew):
            # ascisse (x1, x2) dei due angoli: corda lunga `ew` sull'ellisse con punto medio (in x) = xc
            def length(hx): return math.hypot(2.0 * hx, top_y(xc + hx) - top_y(xc - hx))
            hx_max = max(0.0, min(xc, w - xc))
            if length(hx_max) <= ew: return xc - hx_max, xc + hx_max   # non ci sta: la corda più lunga possibile
            lo, hi = 0.0, hx_max
            for _ in range(40):
                mid = (lo + hi) / 2.0
                if length(mid) < ew: lo = mid
                else: hi = mid
            hx = (lo + hi) / 2.0
            return xc - hx, xc + hx
        ear_w = min(float(CatEar.BASE_WIDTH), max(260.0, w * 0.36), w * 0.48)
        bl = self.ear_left.base_corners_unit()
        br = self.ear_right.base_corners_unit()
        frac_l = max(0.05, bl[1] - bl[0])    # frazione della larghezza dell'immagine occupata dalla base visibile
        frac_r = max(0.05, br[1] - br[0])
        x_min = w * EARS_X_FRACTION
        xc_l = x_min
        for _ in range(6):   # l'angolo esterno non deve uscire dalla finestra
            x1, x2 = chord_x(xc_l, ear_w * frac_l)
            xc_l = max(x_min, (x2 - x1) / 2.0 + 2.0)
        corners_l = chord_x(xc_l, ear_w * frac_l)
        corners_r = chord_x(w - xc_l, ear_w * frac_r)
        def corners_len(c): return math.hypot(c[1] - c[0], top_y(c[1]) - top_y(c[0]))
        # finestra minuscola: orecchio più corto
        ear_w = min(ear_w, corners_len(corners_l) / frac_l, corners_len(corners_r) / frac_r)
        ear_h = ear_w * (float(CatEar.BASE_HEIGHT) / float(CatEar.BASE_WIDTH))
        anchor_x = self.ear_left.anchor_x
        anchor_y = self.ear_left.anchor_y
        def place(c, bc):
            x1, x2 = c
            y1, y2 = top_y(x1), top_y(x2)
            # i due angoli della base stanno ESATTAMENTE sull'ellisse: la base segue la corda (phi); l'asse verticale
            # dell'orecchio ruota invece solo in proporzione all'altezza della finestra: a finestra bassa l'orecchio
            # resta dritto (la base si inclina senza ruotarlo), da EARS_ROTATION_REF_HEIGHT in su ruota del tutto
            phi = math.atan2(y2 - y1, x2 - x1)
            k = max(0.0, min(1.0, h / EARS_ROTATION_REF_HEIGHT))
            ang = math.degrees(phi) * k
            # punto medio della base visibile nel sistema locale dell'orecchio (origine = ancora)
            mx = ((bc[0] + bc[1]) / 2.0 - 0.5) * ear_w
            my = (bc[2] - 1.0) * ear_h
            sa, ca = math.sin(math.radians(ang)), math.cos(math.radians(ang))
            ax = (x1 + x2) / 2.0 - (mx * math.cos(phi) - my * sa)
            ay = (y1 + y2) / 2.0 - (mx * math.sin(phi) + my * ca)
            return ang, math.degrees(phi), ax - anchor_x, ay - anchor_y
        angle_l, base_l, rel_x_l, rel_y_l = place(corners_l, bl)
        angle_r, base_r, rel_x_r, rel_y_r = place(corners_r, br)
        rel_x_l, rel_y_l, rel_x_r, rel_y_r = (int(round(v)) for v in (rel_x_l, rel_y_l, rel_x_r, rel_y_r))
        # Quota (rispetto al bordo superiore della finestra) della punta più alta delle orecchie
        tip_l = rel_y_l + anchor_y - self.ear_left.top_reach(angle_l, ear_w, ear_h, base_l)
        tip_r = rel_y_r + anchor_y - self.ear_right.top_reach(angle_r, ear_w, ear_h, base_r)
        return {
            "angle_l": angle_l, "angle_r": angle_r, "base_l": base_l, "base_r": base_r, "ear_w": ear_w, "ear_h": ear_h,
            "rel_l": QPoint(rel_x_l, rel_y_l), "rel_r": QPoint(rel_x_r, rel_y_r),
            "tip_top": min(tip_l, tip_r),
        }

    def _recompute_ears_geometry(self, w: float, h: float):
        lay = self._ears_layout(w, h)
        self.ear_left.set_ear_config(lay["angle_l"], lay["ear_w"], lay["ear_h"], lay["base_l"])
        self.ear_right.set_ear_config(lay["angle_r"], lay["ear_w"], lay["ear_h"], lay["base_r"])
        self._ear_rel_l = lay["rel_l"]
        self._ear_rel_r = lay["rel_r"]

    def _vertical_reach(self, w: int, h: int):
        # (px sopra la finestra occupati dalle orecchie, px sotto la finestra): l'ellisse non sporge sotto
        above = max(0.0, -self._ears_layout(float(w), float(h))["tip_top"])
        return int(math.ceil(above)), 0

    def _side_reach(self, w: int, h: int) -> int:
        # px che sporgono a sinistra (= a destra) della finestra: nessuno, il profilo è la finestra stessa
        return 0

    def _solve_height(self, w: int, avail: int) -> int:
        min_h = max(self.minimumSizeHint().height(), self.minimumHeight(), 1)
        def total(h): return sum(self._vertical_reach(w, h)) + h
        if total(min_h) > avail: return min_h
        lo, hi = min_h, max(min_h, avail)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if total(mid) <= avail: lo = mid
            else: hi = mid - 1
        return lo

    def _solve_width(self, h: int, avail: int) -> int:
        min_w = max(self.minimumSizeHint().width(), self.minimumWidth(), 1)
        def total(w): return w + 2 * self._side_reach(w, h)
        if total(min_w) > avail: return min_w
        lo, hi = min_w, max(min_w, avail)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if total(mid) <= avail: lo = mid
            else: hi = mid - 1
        return lo

    def _fits_screen(self, w: int, h: int, sg) -> bool:
        above, below = self._vertical_reach(w, h)
        return above + h + below <= sg.height() and w + 2 * self._side_reach(w, h) <= sg.width()

    def fit_to_screen(self, fit_w: bool, fit_h: bool):
        """Adatta la finestra centrale allo schermo.
        Altezza: punte delle orecchie sul bordo superiore, arco inferiore sul bordo inferiore.
        Larghezza: archi laterali sui bordi sinistro e destro (i baffi non contano).
        Ciò che non viene adattato resta invariato (dimensione e posizione sull'asse corrispondente)."""
        if self.isFullScreen() or self.isMaximized() or self.isMinimized() or not self.isVisible(): return
        screen = QApplication.screenAt(self.geometry().center()) or QApplication.primaryScreen()
        sg = screen.geometry()
        w, h = self.width(), self.height()
        if fit_w and fit_h:
            # Larghezza e altezza dipendono l'una dall'altra (curvatura degli archi, dimensione delle orecchie):
            # si alternano i due calcoli finché raggiungono un punto stabile
            for _ in range(30):
                nh = self._solve_height(w, sg.height())
                nw = self._solve_width(nh, sg.width())
                if (nw, nh) == (w, h): break
                w, h = nw, nh
            else:
                min_w = max(self.minimumSizeHint().width(), self.minimumWidth(), 1)
                min_h = max(self.minimumSizeHint().height(), self.minimumHeight(), 1)
                for _ in range(400):
                    if self._fits_screen(w, h, sg) or (w <= min_w and h <= min_h): break
                    w, h = max(min_w, w - 1), max(min_h, h - 1)
        elif fit_h: h = self._solve_height(w, sg.height())
        elif fit_w: w = self._solve_width(h, sg.width())
        x = sg.left() + self._side_reach(w, h) if fit_w else self.x()
        y = sg.top() + self._vertical_reach(w, h)[0] if fit_h else self.y()
        self._ears_key = None
        self._whiskers_key = None
        self.setGeometry(x, y, w, h)
        self._ears_force_raise = True
        self.sync_ears_position()
        self.save_settings()

    def fit_height_to_screen(self): self.fit_to_screen(False, True)
    def fit_width_to_screen(self): self.fit_to_screen(True, False)
    def fit_all_to_screen(self): self.fit_to_screen(True, True)

    def _video_native_ratio(self):
        """Aspect ratio nativo (larghezza/altezza) del video corrente, o None."""
        frame = self._last_display_frame
        vw, vh = (frame.shape[1], frame.shape[0]) if frame is not None else (0, 0)
        if vw <= 0 or vh <= 0:
            pm = self.video_display.pixmap()
            if pm is not None and not pm.isNull(): vw, vh = pm.width(), pm.height()
        if vw <= 0 or vh <= 0: return None
        return vw / float(vh)

    def show_context_bubble(self, gpos):
        if getattr(self, "_bubble", None) is not None:
            try: self._bubble.close()
            except RuntimeError: pass
        can_fit = not (self.isFullScreen() or self.isMaximized() or self.isMinimized())
        has_video = self._has_video()
        self._bubble = BubbleMenu([
            ("Apri file…", self.select_video, True),
            ("Adatta altezza", self.fit_height_to_screen, can_fit),
            ("Adatta larghezza", self.fit_width_to_screen, can_fit),
            ("Adatta tutto", self.fit_all_to_screen, can_fit),
            ("Adatta video", lambda: self.set_video_fill_mode("contain"), has_video),
            ("Adatta video L", lambda: self.set_video_fill_mode("width"), has_video),
            ("Adatta video H", lambda: self.set_video_fill_mode("height"), has_video),
            # Toggle: dimensioni separate "minimal" (salvate in config come window_minimal_*)
            ("Minimal", self.toggle_minimal_mode, can_fit, bool(self._minimal_on)),
            ("Sottotitoli" if self.subs is not None else "Sottotitoli (nessun karaoke)",
             self.toggle_subtitles, self.subs is not None, bool(self.subs_enabled and self.subs is not None)),
            ("Zoom sulla lettera", self.toggle_sub_zoom, self.subs is not None, bool(self.sub_zoom)),
            ("Sempre in primo piano", self.toggle_always_on_top,
             can_fit and FREE_WINDOW_MOVE and sys.platform.startswith("linux"), bool(self.always_on_top)),
            ("V-Sync anti-tearing" + (" (al riavvio)" if self.vsync_enabled != self._vsync_active else ""),
             self.toggle_vsync, QOpenGLWidget is not None, bool(self.vsync_enabled)),
            self._shaders_menu_item(gpos),
        ])
        self._bubble.popup_at(gpos, self.frameGeometry().center())

    # ------------------------------------------------------------ sottotitoli karaoke
    def _load_subtitles(self, video_path):
        """Cerca <video>_karaoke.json accanto al video e, se c'e', lo carica."""
        self.subs = None
        self._sub_key = None
        pj = os.path.splitext(video_path)[0] + "_karaoke.json"
        if os.path.isfile(pj):
            try:
                self.subs = KaraokeSubs.from_file(pj)
            except Exception as e:
                print(f"[Sottotitoli] {os.path.basename(pj)} non utilizzabile: {e}")
        self._sync_sub_timer()
        return self.subs is not None

    def _sync_sub_timer(self):
        if self.subs is not None and self.subs_enabled:
            self._sub_timer.start()
        else:
            self._sub_timer.stop()
            self._sub_key = None
            self.sub_overlay.set_html("")

    def _update_subtitles(self):
        if (self.subs is None or not self.subs_enabled
                or self.audio_player.state() == QMediaPlayer.StoppedState):
            if self._sub_key is not None:
                self._sub_key = None
                self.sub_overlay.set_html("")
            return
        k = self.subs.chiave(self.audio_player.position() / 1000.0, self.sub_zoom)
        if k == self._sub_key: return
        self._sub_key = k
        if k == (None, None, None, None):
            self.sub_overlay.set_html("")
        else:
            self.sub_overlay.set_html(self.subs.html(k), self.subs.zoom_tokens(k) if self.sub_zoom else ())

    @staticmethod
    def _leggi_config_bool(chiave, default):
        """Legge un booleano dal config prima che le impostazioni vengano caricate (serve alla creazione dei widget)."""
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f: return bool(json.load(f).get(chiave, default))
        except Exception: return default

    def toggle_always_on_top(self):
        """Con la spunta la finestra e' senza window manager e resta sopra le altre; senza, e' una finestra normale."""
        global BYPASS_WM
        if self.isFullScreen() or self.isMaximized() or self.isMinimized() or getattr(self, "_min", False):
            self.status.showMessage("Esci da schermo intero/riduzione a icona per cambiare 'Sempre in primo piano'")
            return
        self.always_on_top = not self.always_on_top
        BYPASS_WM = FREE_WINDOW_MOVE and sys.platform.startswith("linux") and self.always_on_top
        geo = QRect(self.geometry())
        flags = Qt.Window | Qt.FramelessWindowHint
        if BYPASS_WM: flags |= Qt.X11BypassWindowManagerHint
        self.setWindowFlags(flags)            # ricrea la finestra nativa (la nasconde)
        self.setGeometry(geo)
        self.show()
        for w in (self.ear_left, self.ear_right, self.whiskers_left, self.whiskers_right):
            _imposta_bypass_overlay(w, BYPASS_WM)
        self._ears_key = None
        self._whiskers_key = None
        self._ears_force_raise = True
        self._last_size_key = None
        self._update_shape()
        self.sync_ears_position()
        self.activateWindow()
        self.save_settings()

    def toggle_vsync(self):
        self.vsync_enabled = not self.vsync_enabled
        self.save_settings()
        self.status.showMessage("V-Sync anti-tearing " + ("attivo" if self.vsync_enabled else "disattivato")
                                + ": si applica al prossimo avvio di Neko Player")

    def toggle_sub_zoom(self):
        self.sub_zoom = not self.sub_zoom
        self._sub_key = None
        self.save_settings()

    def toggle_subtitles(self):
        self.subs_enabled = not self.subs_enabled
        self._sync_sub_timer()
        self.save_settings()

    def _shader_files(self):
        try:
            names = [n for n in os.listdir(SHADERS_DIR)
                     if os.path.splitext(n)[1].lower() in SHADER_EXTS and os.path.isfile(os.path.join(SHADERS_DIR, n))]
        except OSError: return []
        return sorted(names, key=str.lower)

    def _shaders_menu_item(self, gpos):
        files = self._shader_files()
        if not files: return ("Shaders (nessuno)", lambda: None, False)
        n_on = sum(1 for n in files if n in self.active_shaders)
        return (f"Shaders… ({n_on})", lambda: self.show_shaders_bubble(gpos), True)

    def show_shaders_bubble(self, gpos):
        """Sotto-menu con un toggle per ogni shader: resta aperto così se ne possono attivare più di uno di seguito."""
        if getattr(self, "_bubble", None) is not None:
            try: self._bubble.close()
            except RuntimeError: pass
        files = self._shader_files()
        if not files: return
        self._bubble = BubbleMenu(self._shader_menu_items(), keep_open=True, refresh=self._shader_menu_items)
        self._bubble.popup_at(gpos, self.frameGeometry().center())

    def _shader_menu_items(self):
        items = [("Disattiva tutti", self.clear_shaders, bool(self.active_shaders))]
        for name in self._shader_files():
            label = os.path.splitext(name)[0]
            if name in self.active_shaders: label = f"{self.active_shaders.index(name) + 1}. {label}"   # posizione nella catena
            items.append((label, lambda on, n=name: self.set_shader_enabled(n, on), True, name in self.active_shaders))
        items.append(("Forza upscale (ignora WHEN)", self.set_shader_force, True, self.shader_force_when))
        return items

    def set_shader_force(self, on):
        self.shader_force_when = bool(on)
        self.worker.set_shader_force(self.shader_force_when)
        self.save_settings()

    def _push_shaders(self):
        files = self._shader_files()
        self.active_shaders = [n for n in self.active_shaders if n in files]
        self.worker.set_shaders([os.path.join(SHADERS_DIR, n) for n in self.active_shaders])

    def set_shader_enabled(self, name, on):
        if on and name not in self.active_shaders: self.active_shaders.append(name)
        elif not on and name in self.active_shaders: self.active_shaders.remove(name)
        self._push_shaders()
        self.save_settings()

    def clear_shaders(self):
        self.active_shaders.clear()
        self._push_shaders()
        self.save_settings()

    def _on_shader_failed(self, paths, msg):
        names = [os.path.basename(p) for p in paths]
        self.active_shaders = [n for n in self.active_shaders if n not in names]
        self._push_shaders()
        self.save_settings()
        print(f"[Shader] disattivato: {', '.join(names)}\n{msg}")
        self.show_toast(f"Shader disattivato: {', '.join(names)}\n{msg.strip()[:500]}")

    def show_toast(self, text, ms=8000):
        """Avviso temporaneo sopra il video (non modale: con la finestra senza cornice un QMessageBox poteva restare nascosto e bloccare l'app)."""
        lab = getattr(self, "_toast", None)
        if lab is None:
            lab = self._toast = QLabel(self.container)
            lab.setWordWrap(True)
            lab.setAlignment(Qt.AlignCenter)
            c = QColor(self.theme_color)
            lab.setStyleSheet(make_osd_style(c.red(), c.green(), c.blue()))
            self._toast_timer = QTimer(self)
            self._toast_timer.setSingleShot(True)
            self._toast_timer.timeout.connect(lab.hide)
        cw, ch = self.container.width(), self.container.height()
        lab.setText(text)
        lab.setMaximumWidth(max(200, int(cw * 0.7)))
        lab.adjustSize()
        lab.move(max(0, (cw - lab.width()) // 2), int(ch * 0.12))
        lab.show()
        lab.raise_()
        self._toast_timer.start(ms)

    def _apply_minimal_size_limits(self, on: bool):
        """Abbassa/ripristina il limite minimo della finestra in base allo stato di minimal.
        Con minimal attivo la finestra può essere ridotta fino a _VIDEO_MIN_MINIMAL;
        con minimal spento vale il minimo classico (video_display 320x200)."""
        self.video_display.setMinimumSize(
            self._VIDEO_MIN_MINIMAL if on else self._VIDEO_MIN_NORMAL)

    def toggle_minimal_mode(self):
        """Attiva/disattiva la modalità 'minimal': dimensioni della finestra separate,
        salvate in config (window_minimal_w/h) e modificabili ridimensionando la finestra
        quando minimal è attiva. Disattivandola, la finestra ritorna alle dimensioni e
        alla posizione originali."""
        if self.isFullScreen() or self.isMaximized() or self.isMinimized(): return
        if not self._minimal_on:
            # ON: ricorda geometria originale (dimensioni + posizione) e passa a quelle minimal
            self._normal_size_before_minimal = self._normal_window_size()
            self._normal_pos_before_minimal = QPoint(self.pos())
            if self._minimal_size is None:
                # Prima volta: usa le dimensioni correnti come punto di partenza
                self._minimal_size = self._normal_size_before_minimal
            self._minimal_on = True
            # Consente di rimpicciolire la finestra ben sotto il minimo normale
            self._apply_minimal_size_limits(True)
            self._set_minimal_compact(True)   # in basso restano solo Play + barra volume
            old_geo = QRect(self.geometry())
            self.resize(*self._minimal_size)
            new_geo = QRect(self.geometry())
            # Mantieni fisso il centro della finestra durante il passaggio a minimal
            center = old_geo.center() - QPoint(new_geo.width() // 2, new_geo.height() // 2)
            self.move(center)
        else:
            # OFF: salva le dimensioni attuali come minimal e ripristina dimensione+posizione originali
            self._minimal_on = False
            self._minimal_size = self._normal_window_size()
            restore_size = self._normal_size_before_minimal
            restore_pos = self._normal_pos_before_minimal
            if restore_size is None:
                restore_size = (self._VIDEO_MIN_NORMAL.width(), self._VIDEO_MIN_NORMAL.height())
            # Prima di rimettere il minimo normale, assicurarsi che la finestra
            # rientri nei limiti (evita che Qt la "congelhi" sopra il minimo)
            if (self.width() < self._VIDEO_MIN_NORMAL.width() or
                    self.height() < self._VIDEO_MIN_NORMAL.height()):
                self.resize(max(self.width(), self._VIDEO_MIN_NORMAL.width()),
                            max(self.height(), self._VIDEO_MIN_NORMAL.height()))
            self._apply_minimal_size_limits(False)
            self._set_minimal_compact(False)  # ripristina tutti i controlli nel pannello
            self.resize(*restore_size)
            if restore_pos is not None:
                self.move(restore_pos)
            self._normal_pos_before_minimal = None
        self._scale_mouth()
        self._layout_overlay()
        self.save_settings()

    def sync_ears_position(self):
        self._sync_ears()
        self._sync_whiskers()
        if self._ears_force_raise:
            self.ear_left.lower()
            self.ear_right.lower()
            if not BYPASS_WM: self.raise_()
            self.whiskers_left.raise_()
            self.whiskers_right.raise_()
            self._ears_force_raise = False

    def _sync_ears(self):
        if self.isFullScreen() or self.isMinimized() or not self.isVisible():
            self._ears_key = None
            if self.ear_left.isVisible():
                self.ear_left.hide()
                self.ear_right.hide()
            return
        fg = self._current_frame_geometry()
        w = fg.width()
        h = fg.height()
        top_center = QPoint(fg.x() + w // 2, fg.y())
        screen = None
        if hasattr(QApplication, "screenAt"): screen = QApplication.screenAt(top_center)
        if not screen and hasattr(self, "screen"): screen = self.screen()
        if not screen: screen = QApplication.primaryScreen()
        screen_top = screen.geometry().top() if screen else 0
        size_key = (w, h)
        if size_key != getattr(self, "_last_size_key", None):
            self._recompute_ears_geometry(float(w), float(h))
            self._last_size_key = size_key
        key = (fg.x(), fg.y(), w, h, screen_top)
        if key == self._ears_key and not self._ears_force_raise: return
        self._ears_key = key
        if fg.y() <= screen_top + 15:
            if self.ear_left.isVisible():
                self.ear_left.hide()
                self.ear_right.hide()
            return
        if not self.ear_left.isVisible():
            self.ear_left.show()
            self.ear_right.show()
        self.ear_left.move(fg.x() + self._ear_rel_l.x(), fg.y() + self._ear_rel_l.y())
        self.ear_right.move(fg.x() + self._ear_rel_r.x(), fg.y() + self._ear_rel_r.y())

    def _sync_whiskers(self):
        if self.isFullScreen() or self.isMinimized() or not self.isVisible():
            self._whiskers_key = None
            if self.whiskers_left.isVisible():
                self.whiskers_left.hide()
                self.whiskers_right.hide()
            return
        fg = self._current_frame_geometry()
        key = (fg.x(), fg.y(), fg.width(), fg.height())
        if key == self._whiskers_key and not self._ears_force_raise: return
        self._whiskers_key = key
        # lunghezza e spaziatura seguono la dimensione "equivalente" della vecchia finestra rettangolare
        eq_w = max(1.0, (fg.width() - 2 * FRAME_BORDER_WIDTH) / LEGACY_SIZE_GROWTH)
        eq_h = max(1.0, (fg.height() - 2 * FRAME_BORDER_WIDTH) / LEGACY_SIZE_GROWTH)
        sx = max(WHISKERS_MIN_LENGTH_SCALE, eq_w / float(WHISKERS_REF_WIDTH) * WHISKERS_LENGTH_SCALE)   # lunghezza: segue la larghezza (con minimo)
        sy = max(WHISKERS_MIN_SPACING_SCALE, eq_h / float(WHISKERS_REF_HEIGHT))                          # distanza tra i baffi: segue l'altezza (con minimo); lo spessore ha il suo minimo
        self.whiskers_left.set_scale(sx, sy)
        self.whiskers_right.set_scale(sx, sy)
        w_w = self.whiskers_left.base_size().width()
        w_h = self.whiskers_left.base_size().height()
        inset = int(round(15 * sx))
        rel_y = int(fg.height() * 0.58)
        whisker_y = fg.y() + rel_y - (w_h // 2)
        # la base dei baffi poggia sul profilo dell'ellisse all'altezza dei baffi, appena dentro il bordo nero
        edge = fg.width() / 2.0 - ellipse_half_width(fg.width(), fg.height(), rel_y)
        base_in = FRAME_BORDER_WIDTH + ELLIPSE_BAND / 2.0
        left_x = int(round(fg.x() + edge + base_in)) - (w_w - inset)
        right_x = int(round(fg.x() + fg.width() - edge - base_in)) - inset
        newly_shown = not self.whiskers_left.isVisible()
        if newly_shown:
            self.whiskers_left.show()
            self.whiskers_right.show()
        pos_l = QPoint(left_x, whisker_y) - self.whiskers_left.origin_offset()    # in trazione il widget sporge verso l'esterno
        pos_r = QPoint(right_x, whisker_y) - self.whiskers_right.origin_offset()
        if self.whiskers_left.pos() != pos_l: self.whiskers_left.move(pos_l)
        if self.whiskers_right.pos() != pos_r: self.whiskers_right.move(pos_r)
        if newly_shown or self._ears_force_raise:
            self.whiskers_left.raise_()
            self.whiskers_right.raise_()

    def show_default_background(self):
        if self.show_bg_texture and self.bg_pixmap and not self.bg_pixmap.isNull():
            scaled = self.bg_pixmap.scaled(self.video_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_display.setPixmap(scaled)
        else:
            self.video_display.clear()   # nessuna scritta nella finestra quando non c'è un video

    def _scale_mouth(self):
        if hasattr(self, "slider_volume"):
            target = self._mouth_target()
            cap = getattr(self, "_mouth_row_cap", None)   # larghezza massima della riga bocca (corda dell'ellisse)
            if cap and hasattr(self, "mouth_row"):
                # solo se il pannello (già rialzato al massimo) non basta, la bocca si restringe, mai sotto la vecchia misura minima
                over = self._mouth_row_base() + target - cap
                if over > 0: target = max(96, target - over)
            self.slider_volume.set_target_width(target)
        self._align_mouth()

    def _align_mouth(self):
        if not (hasattr(self, "mouth_row") and hasattr(self, "slider_volume")): return
        mw = self.mouth_row.width()
        if mw <= 0: return
        mouth_center_x = self.mouth_row._c.x() + self.slider_volume.x() + self.slider_volume.width() / 2.0
        nose_center_x = mw / 2.0
        delta = nose_center_x - mouth_center_x
        desired = int(round(abs(delta)))
        max_margin = max(0, (mw - self.mouth_row._c.sizeHint().width()) // 2)
        desired = min(desired, max_margin)
        if abs(self.slider_volume._side_margin - desired) <= 1: return
        self.slider_volume.set_side_margin(desired)
        QTimer.singleShot(0, self._align_mouth)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale_mouth()
        self._update_shape()
        self._layout_grips()
        self._layout_overlay()
        if not self.isFullScreen():
            self._recompute_ears_geometry(float(self.width()), float(self.height()))
            self._last_size_key = None
        self.sync_ears_position()
        if not (self._drag_active or self._resize_active): QApplication.processEvents()
        if not self.worker.running: self.show_default_background()

    def moveEvent(self, event):
        super().moveEvent(event)
        self.sync_ears_position()
        if not (self._drag_active or self._resize_active or self._shaking): QApplication.processEvents()

    def changeEvent(self, event):
        if event.type() == QEvent.ActivationChange:
            self._ears_force_raise = True
            if self.isActiveWindow(): self._ensure_ears_behind()
        elif event.type() == QEvent.WindowStateChange:
            self._ears_key = None
            self._whiskers_key = None
            self._update_window_chrome()
            self._update_shape()
        super().changeEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        if delta > 0: self.slider_volume.setValue(min(100, self.slider_volume.value() + 5))
        elif delta < 0: self.slider_volume.setValue(max(0, self.slider_volume.value() - 5))
        event.accept()

    def on_volume_changed(self, val):
        self.current_volume = val
        self.lbl_volume.setText(f"{val}%")
        if not self.is_muted: self.audio_player.setVolume(val)
        self.btn_mute.setText("🔇" if val == 0 else "🔊")
        self.save_settings()

    def toggle_mute(self):
        self.is_muted = not self.is_muted
        if self.is_muted:
            self.audio_player.setMuted(True)
            self.btn_mute.setText("🔇")
        else:
            self.audio_player.setMuted(False)
            self.audio_player.setVolume(self.current_volume)
            self.btn_mute.setText("🔊" if self.current_volume > 0 else "🔇")
        self.save_settings()

    def save_settings(self): self._save_timer.start()

    def _normal_window_size(self):
        if self.isFullScreen():
            g = self._normal_geo if (BYPASS_WM and self._normal_geo is not None) else self.normalGeometry()
        else: g = self.geometry()
        return g.width(), g.height()

    def _write_settings(self):
        config_data = {
            "last_model": self.worker.upscaler_model_path if self.worker.upscaler_model_path else "",
            "upscale_enabled": self.chk_upscale.isChecked(),
            "last_denoise_model": self.worker.denoise_model_path if self.worker.denoise_model_path else "",
            "denoise_enabled": self.chk_denoise.isChecked(),
            "rife_enabled": self.chk_rife.isChecked(),
            "active_shaders": list(self.active_shaders),
            "shaders_force_when": bool(self.shader_force_when),
            "fullscreen": self.isFullScreen(),
            "volume": self.slider_volume.value(),
            "muted": self.is_muted,
            "show_osd": self.show_osd,
            "vsync_display": self.vsync_enabled,
            "always_on_top": self.always_on_top,
            "subtitles_enabled": bool(self.subs_enabled),
            "subtitles_zoom": bool(self.sub_zoom),
            "show_bg_texture": self.chk_texture.isChecked(),
            "video_fill_mode": getattr(self, "_video_fill_mode", "height"),
            "minimal_mode": bool(getattr(self, "_minimal_on", False)),
            "last_video_dir": self.last_video_dir,
            "last_model_dir": self.last_model_dir,
            # timestamp di ripresa per ogni video visto (mappa percorso -> millisecondi)
            "resume_positions": self.resume_positions,
            "theme_color": self.theme_color,
            "window_shape": "ellipse",
            "window_width": self._normal_window_size()[0],
            "window_height": self._normal_window_size()[1],
            # Posizione attuale della finestra (così, riaprendo con minimal attivo e poi
            # disattivandolo, si può tornare alla posizione dell'ultima sessione normale)
            "window_x": self.pos().x(),
            "window_y": self.pos().y()
        }
        # Dimensioni "minimal" salvate separatamente rispetto a window_width/height
        if getattr(self, "_minimal_on", False):
            # Con minimal attivo la finestra attuale è quella minimal
            mw, mh = self._normal_window_size()
        else:
            mw, mh = getattr(self, "_minimal_size", None) or (0, 0)
        if mw > 0 and mh > 0:
            config_data["window_minimal_w"] = mw
            config_data["window_minimal_h"] = mh
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=4)
        except Exception as e: print(f"[Config] Errore: {e}")

    def load_settings(self):
        if not os.path.isfile(CONFIG_FILE): return
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f: config_data = json.load(f)
            saved_vol = config_data.get("volume", 80)
            self.current_volume = saved_vol
            self.slider_volume.setValue(saved_vol)
            self.lbl_volume.setText(f"{saved_vol}%")
            self.is_muted = config_data.get("muted", False)
            if self.is_muted:
                self.audio_player.setMuted(True)
                self.btn_mute.setText("🔇")
            else:
                self.audio_player.setVolume(saved_vol)
                self.btn_mute.setText("🔊" if saved_vol > 0 else "🔇")
            saved_upscale = config_data.get("upscale_enabled", True)
            self.chk_upscale.setChecked(saved_upscale)
            self.worker.upscale_enabled = saved_upscale

            saved_denoise = config_data.get("denoise_enabled", False)
            self.chk_denoise.setChecked(saved_denoise)
            self.worker.denoise_enabled = saved_denoise

            if config_data.get("rife_enabled", False) and self.worker.load_rife():
                self.chk_rife.blockSignals(True)
                self.chk_rife.setChecked(True)
                self.chk_rife.blockSignals(False)
                self.worker.set_rife_enabled(True)

            saved_shaders = config_data.get("active_shaders", [])
            if isinstance(saved_shaders, list):
                self.active_shaders = list(dict.fromkeys(x for x in saved_shaders if isinstance(x, str)))
                self._push_shaders()
            self.shader_force_when = bool(config_data.get("shaders_force_when", False))
            self.worker.set_shader_force(self.shader_force_when)

            saved_texture = config_data.get("show_bg_texture", True)
            self.show_bg_texture = saved_texture
            self.chk_texture.setChecked(saved_texture)
            self.container.set_show_texture(saved_texture)
            saved_fill = config_data.get("video_fill_mode", "height")
            if saved_fill in ("width", "height", "contain"): self._video_fill_mode = saved_fill
            # Stato e dimensioni separate della modalità "minimal"
            # (le dimensioni minimal sostituiscono window_width/height: verranno salvate
            #  al prossimo write, così le due coppie restano sempre coerenti con lo stato)
            # Le dimensioni salvate dalle versioni con 5 finestre (rettangolo + 4 archi) si convertono una volta
            legacy_shape = config_data.get("window_shape") != "ellipse"
            min_w = config_data.get("window_minimal_w")
            min_h = config_data.get("window_minimal_h")
            if isinstance(min_w, int) and isinstance(min_h, int) and min_w > 0 and min_h > 0:
                self._minimal_size = legacy_to_ellipse_size(min_w, min_h) if legacy_shape else (min_w, min_h)
            self._minimal_on = bool(config_data.get("minimal_mode", False)) and self._minimal_size is not None
            # Se il salvataggio era in modalità minimal, abbassa subito il limite
            # minimo della finestra (serve anche per aprire dimensioni < 320x200)
            self._apply_minimal_size_limits(self._minimal_on)
            self.show_osd = config_data.get("show_osd", False)
            self.subs_enabled = bool(config_data.get("subtitles_enabled", True))
            self.sub_zoom = bool(config_data.get("subtitles_zoom", True))
            self.vsync_enabled = bool(config_data.get("vsync_display", True))
            self.last_video_dir = config_data.get("last_video_dir", "")
            self.last_model_dir = config_data.get("last_model_dir", "")
            # Timestamp di ripresa salvati per i video già visti
            saved_resume = config_data.get("resume_positions", {})
            if isinstance(saved_resume, dict):
                self.resume_positions = {k: int(v) for k, v in saved_resume.items()
                                         if isinstance(v, (int, float)) and int(v) > 0}
            saved_color = config_data.get("theme_color", FRAME_COLOR)
            self.apply_theme_color(saved_color)
            win_w = config_data.get("window_width")
            win_h = config_data.get("window_height")
            shift = QPoint(0, 0)   # la finestra ellittica è più grande del vecchio rettangolo: si tiene fermo il centro
            if legacy_shape and isinstance(win_w, int) and isinstance(win_h, int) and win_w > 0 and win_h > 0:
                nw, nh = legacy_to_ellipse_size(win_w, win_h)
                shift = QPoint((nw - win_w) // 2, (nh - win_h) // 2)
                win_w, win_h = nw, nh
            if self._minimal_on:
                win_w, win_h = self._minimal_size
            if isinstance(win_w, int) and isinstance(win_h, int) and win_w > 0 and win_h > 0:
                area = QApplication.primaryScreen().availableGeometry()
                min_size = self.minimumSizeHint().expandedTo(self.minimumSize())
                self.resize(
                    max(min_size.width(), min(win_w, area.width())),
                    max(min_size.height(), min(win_h, area.height())),
                )
            # Posizione della finestra (se salvata): usata anche come posizione "normale"
            # da ripristinare quando minimal viene disattivata
            win_x = config_data.get("window_x")
            win_y = config_data.get("window_y")
            if isinstance(win_x, int) and isinstance(win_y, int):
                win_x -= shift.x()
                win_y -= shift.y()
                vgeo = QApplication.primaryScreen().virtualGeometry()
                x = max(vgeo.left(), min(win_x, vgeo.right() - self.width() + 1))
                y = max(vgeo.top(), min(win_y, vgeo.bottom() - self.height() + 1))
                self.move(x, y)
                if not self._minimal_on:
                    self._normal_pos_before_minimal = QPoint(x, y)

            saved_model = config_data.get("last_model", "")
            if saved_model and os.path.isfile(saved_model):
                try:
                    self.worker.load_model(saved_model)
                    self.status.showMessage(f"Modello: {os.path.basename(saved_model)}")
                except Exception as e: print(f"[Config] Errore: {e}")

            saved_denoise_model = config_data.get("last_denoise_model", "")
            if saved_denoise_model and os.path.isfile(saved_denoise_model):
                try:
                    self.worker.load_denoise_model(saved_denoise_model)
                    self.status.showMessage(f"Denoise: {os.path.basename(saved_denoise_model)}")
                except Exception as e: print(f"[Config] Errore Denoise: {e}")

            if config_data.get("fullscreen", False): QTimer.singleShot(100, self.toggle_fullscreen)
        except Exception as e: print(f"[Config] Errore: {e}")

    def toggle_fullscreen(self):
        self._ears_key = None
        self._whiskers_key = None
        if self.isFullScreen():
            self._set_fullscreen(False)
            self.timeline_widget.show()
            self.controls_widget.show()
            c = QColor(self.theme_color)
            self.video_display.setStyleSheet(
                f"background-color: rgba({c.red()}, {c.green()}, {c.blue()}, 0.85); border: none; border-radius: 8px;"
            )
            self.btn_fullscreen.setText("⛶")
            self._update_window_chrome()
            self._update_shape()
            self._last_size_key = None
            self._ears_force_raise = True
            self._show_overlay()
            self.sync_ears_position()
            QTimer.singleShot(0, self._post_fullscreen_refresh)
            QTimer.singleShot(150, self._post_fullscreen_refresh)
        else:
            self._set_fullscreen(True)
            self._last_size_key = None
            self._show_overlay()
            QTimer.singleShot(0, self._layout_overlay)
            QTimer.singleShot(150, self._layout_overlay)
            self.video_display.setStyleSheet("background-color: #000000; border: none; border-radius: 0px;")
            self.btn_fullscreen.setText("🗗")
            self._update_window_chrome()
            self._update_shape()
            self.ear_left.hide()
            self.ear_right.hide()
            self.whiskers_left.hide()
            self.whiskers_right.hide()
        self.save_settings()

    def _post_fullscreen_refresh(self):
        if self.isFullScreen() or not self.isVisible(): return
        if BYPASS_WM and self._normal_geo is not None and self.geometry() != self._normal_geo:
            self.setGeometry(self._normal_geo)
        self._update_window_chrome()
        self._update_shape()
        self._scale_mouth()
        self._recompute_ears_geometry(float(self.width()), float(self.height()))
        self._last_size_key = None
        self._ears_key = None
        self._whiskers_key = None
        self._ears_force_raise = True
        self._layout_overlay()
        self._update_overlay_visibility()
        if not self._has_video() and not getattr(self, "_minimal_on", False) \
                and not self.controls_panel.isVisible(): self._show_overlay()
        self.sync_ears_position()

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key_Q:
            self.close()
            event.accept()
        elif key == Qt.Key_Space:
            self.ear_left.trigger_twitch()
            self.ear_right.trigger_twitch()
            self.trigger_whiskers_wiggle()
            if self.worker.running or self._audio_only: self.toggle_play_pause()
            event.accept()
        elif key == Qt.Key_W:
            self.trigger_whiskers_wiggle()
            event.accept()
        elif key == Qt.Key_C:
            self.choose_theme_color()
            event.accept()
        elif key == Qt.Key_I:
            self.show_osd = not self.show_osd
            if not self.show_osd: self.osd_label.hide()
            self.status.showMessage(f"HUD Info: {'ATTIVO' if self.show_osd else 'DISATTIVO'}", 2000)
            self.save_settings()
            event.accept()
        elif key in (Qt.Key_F, Qt.Key_F11):
            self.toggle_fullscreen()
            event.accept()
        elif key == Qt.Key_Escape:
            if self.isFullScreen(): self.toggle_fullscreen()
            event.accept()
        elif key == Qt.Key_M:
            self.toggle_mute()
            event.accept()
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            self.slider_volume.setValue(min(100, self.slider_volume.value() + 5))
            event.accept()
        elif key == Qt.Key_Minus:
            self.slider_volume.setValue(max(0, self.slider_volume.value() - 5))
            event.accept()
        elif key == Qt.Key_Left:
            self.seek_relative(-5.0)
            event.accept()
        elif key == Qt.Key_Right:
            self.seek_relative(5.0)
            event.accept()
        elif key == Qt.Key_Down:
            self.seek_relative(-15.0)
            event.accept()
        elif key == Qt.Key_Up:
            self.seek_relative(15.0)
            event.accept()
        else: super().keyPressEvent(event)

    def seek_relative(self, delta_sec):
        cur_pos_ms = self.audio_player.position()
        target_ms = max(0, cur_pos_ms + int(delta_sec * 1000))
        self.audio_player.setPosition(target_ms)
        if self._audio_only: return       # nessun decoder video da riposizionare
        self.worker.request_seek_seconds(delta_sec)

    def on_slider_pressed(self): self.is_slider_dragged = True
    def on_slider_released(self):
        self.is_slider_dragged = False
        target_frame = self.slider.value()
        if self._audio_only:              # nel modo audio lo slider e' in millisecondi
            self.audio_player.setPosition(int(target_frame))
            return
        if self.worker.video_path:
            fps = self.worker.video_fps or 25.0
            target_ms = int((target_frame / fps) * 1000.0)
            self.audio_player.setPosition(target_ms)
            self.worker.request_seek_frame(target_frame)

    def update_timeline(self, cur_frame, total_frames, cur_sec, total_sec):
        self.total_duration_sec = total_sec
        if not self.is_slider_dragged:
            for sl in (self.slider, self.slider_b):
                sl.blockSignals(True)
                sl.setMaximum(total_frames)
                sl.setValue(cur_frame)
                sl.blockSignals(False)
                sl.update()

    def _remember_current_position(self):
        """Salva il timestamp del video attualmente caricato (per riprenderlo alla riapertura)."""
        try:
            path = self._audio_path if self._audio_only else self.worker.video_path
            if not path: return
            # frame mostrati dal worker; eventuale fallback: posizione dell'audio player
            pos_ms = 0 if self._audio_only else int(self.worker._shown_pos * 1000.0 / max(1.0, self.worker.video_fps))
            if self.audio_player.isAvailable():
                ap = self.audio_player.position()
                if ap and ap > 0: pos_ms = ap
            total_ms = self.audio_player.duration() if self.audio_player.isAvailable() else 0
            if total_ms and pos_ms >= total_ms - 1500:
                # video arrivato alla fine: la ripresa non ha senso, azzera
                pos_ms = 0
            self.resume_positions[path] = int(pos_ms)
            # limita la dimensione della mappa in config
            if len(self.resume_positions) > 30:
                for k in list(self.resume_positions)[:-30]:
                    del self.resume_positions[k]
        except Exception: pass

    def play_video_file(self, file_path):
        if os.path.splitext(file_path)[1].lower().lstrip(".") in AUDIO_EXTS:
            return self.play_audio_file(file_path)
        # prima di cambiare video, ricorda dove eravamo arrivati nel precedente
        if self.worker.video_path or self._audio_only: self._remember_current_position()
        self._leave_audio_mode()
        self.worker.stop()
        self.audio_player.stop()
        self.worker.paused = False
        self.worker.audio_clock = None
        self._audio_last_pos = None
        has_subs = self._load_subtitles(file_path)
        url = QUrl.fromLocalFile(file_path)
        self.audio_player.setMedia(QMediaContent(url))
        self.worker.set_video(file_path)
        self.btn_play_pause.setEnabled(True)
        self.btn_play_pause.set_playing(True)
        self.status.showMessage(f"In riproduzione: {os.path.basename(file_path)}"
                                + ("  ·  sottotitoli karaoke caricati" if has_subs else ""))
        self._update_display_target()
        self.audio_player.play()
        self.worker.start()
        # ripristina il timestamp dell'ultima riproduzione, se presente e valido
        saved_ms = self.resume_positions.get(file_path, 0)
        if saved_ms > 2000:
            def _resume_at_saved():
                if self.worker.video_path != file_path: return
                dur = self.audio_player.duration()
                target = saved_ms
                if dur and target >= dur - 2000: target = 0   # era alla fine: riparti da capo
                if target <= 0: return
                self.audio_player.setPosition(target)
                self.worker.request_seek_time(target / 1000.0)
                self.status.showMessage(
                    f"Ripresa da {format_time(target / 1000.0)}: {os.path.basename(file_path)}")
            QTimer.singleShot(400, _resume_at_saved)

    # ------------------------------------------------------------ file solo audio
    def play_audio_file(self, file_path):
        """File solo audio: schermo nero, karaoke e sinusoide dei dB al posto del video."""
        if self.worker.video_path or self._audio_only: self._remember_current_position()
        self.worker.stop()
        self.worker.set_video(None)
        self._stop_envelope_threads()
        self.audio_player.stop()
        self.worker.paused = False
        self.worker.audio_clock = None
        self._audio_last_pos = None
        self._audio_only = True
        self._audio_path = file_path
        self._wave_anchor = None
        self._wave_last_t = 0.0
        has_subs = self._load_subtitles(file_path)
        self.audio_player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
        self.btn_play_pause.setEnabled(True)
        self.btn_play_pause.set_playing(True)
        self.osd_label.hide()
        self.wave_overlay.set_envelope(None)
        self.wave_overlay.show()
        self.wave_overlay.lower()
        th = AudioEnvelopeThread(file_path, self)
        th.ready.connect(self._on_envelope_ready)
        th.finished.connect(lambda t=th: self._env_threads.remove(t) if t in self._env_threads else None)
        self._env_threads.append(th)
        th.start()
        self._wave_timer.start()
        self.status.showMessage(f"In riproduzione (audio): {os.path.basename(file_path)}"
                                + ("  ·  sottotitoli karaoke caricati" if has_subs else ""))
        self.audio_player.play()
        saved_ms = self.resume_positions.get(file_path, 0)
        if saved_ms > 2000:
            def _resume_at_saved():
                if self._audio_path != file_path: return
                dur = self.audio_player.duration()
                target = saved_ms
                if dur and target >= dur - 2000: target = 0
                if target <= 0: return
                self.audio_player.setPosition(target)
                self.status.showMessage(
                    f"Ripresa da {format_time(target / 1000.0)}: {os.path.basename(file_path)}")
            QTimer.singleShot(400, _resume_at_saved)
        self._update_overlay_visibility()

    def _leave_audio_mode(self):
        if not self._audio_only: return
        self._audio_only = False
        self._audio_path = None
        self._wave_timer.stop()
        self._stop_envelope_threads()
        self.wave_overlay.set_envelope(None)
        self.wave_overlay.hide()

    def _stop_envelope_threads(self):
        for th in list(self._env_threads):
            th.stop()
            try: th.ready.disconnect(self._on_envelope_ready)
            except TypeError: pass

    def _on_envelope_ready(self, path, db):
        if self._audio_only and path == self._audio_path:
            self.wave_overlay.set_envelope(db)

    def _tick_audio_mode(self):
        """Ogni ~33 ms: tempo di riproduzione (interpolato) per l'onda e avanzamento della timeline."""
        if not self._audio_only: return
        ap = self.audio_player
        pos = ap.position() / 1000.0
        now = time.perf_counter()
        playing = ap.state() == QMediaPlayer.PlayingState
        a = self._wave_anchor
        if a is None or pos != a[0]:
            self._wave_anchor = a = (pos, now)
        t = pos + min(now - a[1], 1.0) if playing else pos
        if playing and 0.0 < self._wave_last_t - t < 0.3:
            t = self._wave_last_t          # niente piccoli passi indietro dovuti al campionamento
        self._wave_last_t = t
        self.wave_overlay.set_time(t)
        dur = ap.duration()
        if dur > 0:
            self.total_duration_sec = dur / 1000.0
            if not self.is_slider_dragged:
                for sl in (self.slider, self.slider_b):
                    sl.blockSignals(True)
                    sl.setMaximum(int(dur))
                    sl.setValue(int(pos * 1000.0))
                    sl.blockSignals(False)
                    sl.update()

    def _on_audio_status(self, status):
        if not self._audio_only or status != QMediaPlayer.EndOfMedia: return
        if LOOP_VIDEO:
            self._wave_anchor = None
            self._wave_last_t = 0.0
            self.audio_player.setPosition(0)
            self.audio_player.play()
        else:
            self._leave_audio_mode()
            self.on_video_finished()

    def _update_audio_clock(self):
        if self.audio_player.state() != QMediaPlayer.PlayingState:
            self.worker.audio_clock = None
            self._audio_last_pos = None
            return
        now = time.perf_counter()
        pos = self.audio_player.position()
        if self._audio_last_pos is None:
            self._audio_last_pos = pos
            self._audio_last_change = now
            return
        if pos != self._audio_last_pos:
            self._audio_last_change = now
            self.worker.audio_clock = (pos / 1000.0, now)
        elif now - self._audio_last_change > 1.5:
            self.worker.audio_clock = None

    def _update_display_target(self):
        s = self.video_display.size()
        self.worker.display_size = (s.width(), s.height())

    def _dialog_places(self, kind):
        loc = QStandardPaths.writableLocation
        paths = [MODELS_DIR] if kind == "model" else []
        paths += [
            self.last_video_dir if kind == "video" else self.last_model_dir,
            loc(QStandardPaths.HomeLocation),
            loc(QStandardPaths.DesktopLocation),
            loc(QStandardPaths.MoviesLocation),
            loc(QStandardPaths.DownloadLocation),
        ]
        for vol in QStorageInfo.mountedVolumes():
            root = vol.rootPath()
            if vol.isValid() and vol.isReady() and root.startswith(("/media/", "/run/media/", "/mnt/")):
                paths.append(root)
        return [p for p in dict.fromkeys(paths) if p and os.path.isdir(p)]

    def _pick_file(self, kind):
        is_video = (kind == "video")
        title = "Seleziona Video" if is_video else "Seleziona Modello AI"
        last_dir = self.last_video_dir if is_video else self.last_model_dir
        default_dir = QStandardPaths.writableLocation(QStandardPaths.MoviesLocation) if is_video else MODELS_DIR
        start = next((d for d in (last_dir, default_dir, BASE_DIR) if d and os.path.isdir(d)), BASE_DIR)
        filters = VIDEO_FILTERS if is_video else MODEL_FILTERS
        if USE_NATIVE_FILE_DIALOG:
            path, _ = QFileDialog.getOpenFileName(self, title, start, ";;".join(filters))
        else:
            if self._file_icons is None: self._file_icons = NekoIconProvider()
            path = open_file_dialog(
                self, title, start, filters, self._dialog_places(kind), self._file_icons,
                select=self.worker.video_path if is_video else (self.worker.upscaler_model_path or self.worker.denoise_model_path),
            )
        if BYPASS_WM: self._grab_focus()
        if path:
            if is_video: self.last_video_dir = os.path.dirname(path)
            else: self.last_model_dir = os.path.dirname(path)
            self.save_settings()
        return path

    def select_video(self):
        file_path = self._pick_file("video")
        if file_path: self.play_video_file(file_path)

    def select_model(self):
        model_path = self._pick_file("model")
        if model_path:
            try:
                self.worker.load_model(model_path)
                self.status.showMessage(f"Modello: {os.path.basename(model_path)}")
                self.save_settings()
            except Exception as e: self.status.showMessage(f"Errore: {e}")

    def select_denoise_model(self):
        model_path = self._pick_file("model")
        if model_path:
            try:
                self.worker.load_denoise_model(model_path)
                if not self.chk_denoise.isChecked(): self.chk_denoise.setChecked(True)
                self.status.showMessage(f"Denoise: {os.path.basename(model_path)}")
                self.save_settings()
            except Exception as e: self.status.showMessage(f"Errore Denoise: {e}")

    def toggle_upscale(self, state):
        self.worker.upscale_enabled = (state == Qt.Checked)
        self.save_settings()

    def toggle_denoise(self, state):
        self.worker.denoise_enabled = (state == Qt.Checked)
        self.save_settings()

    def toggle_rife(self, state):
        enabled = (state == Qt.Checked)
        if enabled and not self.worker.load_rife():
            self.chk_rife.blockSignals(True)
            self.chk_rife.setChecked(False)
            self.chk_rife.blockSignals(False)
            enabled = False
            QMessageBox.warning(self, "RIFE",
                f"Modello RIFE non trovato o non valido.\n\nScarica '{os.path.basename(RIFE_MODEL_PATH)}' e mettilo in:\n{MODELS_DIR}")
        self.worker.set_rife_enabled(enabled)
        self.save_settings()

    def on_loop_restarted(self):
        self.worker.audio_clock = None
        self._audio_last_pos = None
        self.audio_player.setPosition(0)
        if not self.worker.paused and self.audio_player.state() != QMediaPlayer.PlayingState:
            self.audio_player.play()

    def toggle_play_pause(self):
        self.worker.paused = not self.worker.paused
        if self.worker.paused:
            self.audio_player.pause()
            self.btn_play_pause.set_playing(False)
        else:
            self.audio_player.play()
            self.btn_play_pause.set_playing(True)

    def set_video_fill_mode(self, mode):
        """Cambia il riempimento della finestra per il video: "width", "height" o "contain"."""
        if mode not in ("width", "height", "contain"): return
        self._video_fill_mode = mode
        self.save_settings()
        # Ridisegna subito l'ultima cornice con il nuovo riempimento
        frame = self._last_display_frame
        if frame is not None:
            try:
                self.update_video_frame(frame, self.worker.real_fps, 0.0,
                                        frame.shape[1], frame.shape[0])
            except Exception: pass
        else:
            self.video_display.update()

    def update_video_frame(self, frame, fps, gpu_ms, out_w, out_h):
        h, w = frame.shape[:2]
        self._last_display_frame = np.ascontiguousarray(frame)
        q_img = QImage(frame.data, w, h, frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        target = self.video_display.size()
        tw, th = target.width(), target.height()
        fill_mode = getattr(self, "_video_fill_mode", "height")
        if fill_mode == "width":
            # Adatta alla larghezza dell'area centrale; se è più alto, taglia sopra/sotto (centrato)
            if tw > 0 and abs(pixmap.width() - tw) > 1:
                pixmap = pixmap.scaledToWidth(tw, Qt.FastTransformation)
            if th > 0 and pixmap.height() > th:
                pixmap = pixmap.copy(0, (pixmap.height() - th) // 2, pixmap.width(), th)
        elif fill_mode == "contain":
            # Adatta il video in modo che i 4 angoli siano contenuti nell'ellisse della finestra.
            # La finestra è un'ellisse x²/a² + y²/b² = 1 con semiassi a=tw/2, b=th/2; il display
            # è centrato nella finestra, quindi l'ellisse ha gli stessi assi del display.
            # Il rettangolo video centrato (±W/2, ±H/2), con aspect ratio r=w/h, è inscritto
            # quando W²/tw² + H²/th² = 1 con W = r·H, da cui:
            #   H_max = tw·th / sqrt((r·th)² + tw²),  W_max = r · H_max
            if tw > 0 and th > 0 and w > 0 and h > 0:
                if self.isFullScreen() or self.isMaximized():
                    pixmap = pixmap.scaled(tw, th, Qt.KeepAspectRatio, Qt.FastTransformation)
                else:
                    r = w / float(h)
                    denom = math.hypot(r * th, tw)
                    if denom > 0:
                        target_h = max(1, int(round((tw * th) / denom)))
                        target_w = max(1, int(round(r * target_h)))
                        pixmap = pixmap.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.FastTransformation)
        else:
            # Adatta all'altezza dell'area centrale; se è più largo, taglia i lati (centrato)
            if th > 0 and abs(pixmap.height() - th) > 1:
                pixmap = pixmap.scaledToHeight(th, Qt.FastTransformation)
            if tw > 0 and pixmap.width() > tw:
                pixmap = pixmap.copy((pixmap.width() - tw) // 2, 0, tw, pixmap.height())
        self.video_display.setPixmap(pixmap)
        self.worker.display_size = (target.width(), target.height())
        now = time.monotonic()
        if now - self._last_hud_update < 0.25: return
        self._last_hud_update = now
        if self.show_osd:
            models = []
            if self.worker.denoise_enabled and self.worker.denoise_model_path:
                models.append(f"🧹 {os.path.basename(self.worker.denoise_model_path)}")
            if self.worker.upscale_enabled and self.worker.upscaler_model_path:
                models.append(f"⚡ {os.path.basename(self.worker.upscaler_model_path)}")
            if self.worker.rife_enabled and self.worker.rife:
                models.append("🎞️ RIFE 2x")
            if not models:
                model_text = "Nativo"
            else:
                prec = "FP16" if (self.worker.upscaler and self.worker.upscaler['np_dtype'] == np.float16) else "FP32"
                model_text = " -> ".join(models) + f" [{prec}]"

            shader_line = ""
            if self.active_shaders:
                shader_line = "\n🎨 Shader: " + ", ".join(os.path.splitext(n)[0] for n in self.active_shaders)
            self.osd_label.setText(
                f"⚡ Framerate: {fps:.1f} FPS\n"
                f"⏱️ Latenza GPU: {gpu_ms:.1f} ms\n"
                f"🧠 Pipeline: {model_text}\n"
                f"🎬 Decodifica: {self.worker.decode_mode}\n"
                f"📺 Risoluzione Frame: {out_w}x{out_h}" + shader_line
            )
            self.osd_label.adjustSize()
            if self.isFullScreen() or self.isMaximized():
                self.osd_label.move(20, 20)
            else:
                # l'angolo (20, 20) cade fuori dall'ellisse: si parte dal profilo a un decimo dell'altezza
                lw, lh = self.video_display.width(), self.video_display.height()
                oy = int(lh * 0.10)
                self.osd_label.move(int(lw / 2.0 - ellipse_half_width(lw, lh, oy)) + 12, oy)
            self.osd_label.show()
        else: self.osd_label.hide()

    def on_video_finished(self):
        self.btn_play_pause.setEnabled(False)
        self.btn_play_pause.set_playing(False)
        self.audio_player.stop()
        self._sub_key = None
        self.sub_overlay.set_html("")
        self.show_default_background()
        self._update_overlay_visibility()

    def closeEvent(self, event):
        self._save_timer.stop()
        # prima di chiudere, ricorda il timestamp del video in riproduzione/pausa
        if self.worker.video_path or self._audio_only: self._remember_current_position()
        self._stop_envelope_threads()
        for th in list(self._env_threads): th.wait(1500)
        self._write_settings()
        if getattr(self, "_bubble", None) is not None:
            try: self._bubble.close()
            except RuntimeError: pass
        self.worker.stop()
        self.audio_player.stop()
        self.ear_left.close()
        self.ear_right.close()
        self.whiskers_left.close()
        self.whiskers_right.close()
        self._restore_stub.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app_font = QFont(resolve_ui_font_family(), 10)
    app_font.setBold(False)
    app.setFont(app_font)
    apply_system_icon_theme()
    qt_translator = QTranslator()
    if qt_translator.load(QLocale.system(), "qtbase", "_", QLibraryInfo.location(QLibraryInfo.TranslationsPath)):
        app.installTranslator(qt_translator)
    video_argument = None
    if len(sys.argv) > 1:
        candidate_path = sys.argv[1]
        if os.path.isfile(candidate_path): video_argument = os.path.abspath(candidate_path)
    player = NekoPlayer(initial_video=video_argument)
    player.show()
    if BYPASS_WM: QTimer.singleShot(150, player._grab_focus)
    sys.exit(app.exec_())
