import sys
import os
import glob
import json
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
from shaders import CAS_FRAGMENT_SHADER

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
MODELS_DIR = os.path.join(BASE_DIR, "animejanai")
BG_IMAGE_PATH = os.path.join(BASE_DIR, "sfondo.png")
RIFE_MODEL_PATH = os.path.join(MODELS_DIR, "rife49_ensemble_True_scale_1_sim.onnx")
RIFE_SCENE_CUT = 0.10   # differenza media oltre la quale non si interpola (cambio scena)
EARS_IMAGE_PATH = os.path.join(BASE_DIR, "ears.png")
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

FRAME_COLOR = "#0f0c12"
FRAME_BORDER_WIDTH = 10          # spessore del bordo nero della finestra ellittica
ELLIPSE_BAND = 14                # fascia a tinta unita (colore tema) tra bordo nero e video; 0 = nessuna
EARS_X_FRACTION = 0.27           # posizione orizzontale del centro delle orecchie (frazione della larghezza)
# Le dimensioni salvate dalle versioni a 5 finestre erano quelle del solo rettangolo centrale: gli archi
# lo ingrandivano di questo fattore (più il bordo) su ogni asse. Serve a convertirle alla prima apertura.
LEGACY_SIZE_GROWTH = 1.0 + 0.7 * (math.sqrt(2.0) - 1.0)
MOUTH_STEM_REACHES_NOSE = True   # la linea verticale della bocca sale fino al bordo del naso
WHISKERS_REF_WIDTH = 740   # larghezza finestra a cui i baffi hanno lunghezza 100%
WHISKERS_REF_HEIGHT = 520  # altezza finestra a cui distanza tra i baffi e spessore sono al 100%
# (i due riferimenti valgono per la vecchia finestra rettangolare; la finestra ellittica viene convertita)
WHISKERS_LENGTH_SCALE = 0.6  # i baffi ora partono dal bordo e non più dal bordo del video: accorciati di conseguenza

UI_FONT_FAMILY = "Nunito"
UI_FONT_FALLBACKS = ("Nunito", "Nunito Sans", "Quicksand", "Comfortaa", "sans-serif")

FREE_WINDOW_MOVE = True
BYPASS_WM = FREE_WINDOW_MOVE and sys.platform.startswith("linux")

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
    QPainterPathStroker, QIcon, QFont, QCursor, QFontMetrics, QLinearGradient, QPolygon
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QUrl, QPoint, QPointF, QRect, QRectF, QSize, QEvent,
    QFileInfo, QStandardPaths, QStorageInfo, QTranslator, QLocale, QLibraryInfo, QMimeDatabase
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

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
        "QSlider#videoSlider::groove:horizontal { height: 6px; background: rgba(49, 50, 68, 0.80); border-radius: 3px; } "
        "QSlider#videoSlider::sub-page:horizontal { background: #fab387; border-radius: 3px; } "
        "QSlider#videoSlider::handle:horizontal { background: #cdd6f4; border: 2px solid #fab387; width: 14px; margin-top: -4px; margin-bottom: -4px; border-radius: 7px; } "
        "QSlider#videoSlider::handle:horizontal:hover { background: #ffffff; } "
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
MODEL_EXTS = {"onnx", "pt", "pth"}
VIDEO_FILTERS = ["Video (" + "  ".join("*." + e for e in sorted(VIDEO_EXTS)) + ")", "Tutti i file (*)"]
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
        self._model = themed(["package-x-generic", "application-x-executable"], QStyle.SP_FileIcon)
        self._file = themed(["text-x-generic", "application-x-executable"], QStyle.SP_FileIcon)
    def icon(self, arg):
        if isinstance(arg, QFileInfo):
            if arg.isDir(): return self._dir
            ext = arg.suffix().lower()
            if ext in VIDEO_EXTS: return self._video
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
    BOX_SIZE = 420
    def __init__(self, is_left=True, parent=None):
        flags = Qt.FramelessWindowHint | Qt.WindowDoesNotAcceptFocus | Qt.Tool
        if sys.platform.startswith("linux"): flags |= Qt.X11BypassWindowManagerHint
        super().__init__(parent, flags)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.is_left = is_left
        self.setFixedSize(self.BOX_SIZE, self.BOX_SIZE)
        self.flare = 0.0
        self.squash = 0.0
        self.ear_w = float(self.BASE_WIDTH)
        self.ear_h = float(self.BASE_HEIGHT)
        self.angle = 0.0
        self.anchor_x = self.BOX_SIZE / 2.0
        self.anchor_y = self.BOX_SIZE - 40.0
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
    def top_reach(self, angle: float, ear_w: float, ear_h: float) -> float:
        # Distanza in px tra l'ancora e il punto più alto dell'orecchio ruotato (stessa trasformazione del paintEvent)
        th = math.radians(angle)
        s, c = math.sin(th), math.cos(th)
        pts = self._opaque_unit_points()
        if pts is None:
            return ear_h * c + (ear_w / 2.0) * abs(s)
        u, v = pts
        x = (u - 0.5) * ear_w
        y = (v - 1.0) * ear_h
        return float(-(x * s + y * c).min())
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
    def set_ear_config(self, angle: float, width: float, height: float):
        self.angle = angle
        self.ear_w = width
        self.ear_h = height
        self.update()
    def paintEvent(self, event):
        if self.ear_h <= 5: return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.translate(self.anchor_x, self.anchor_y)
        painter.rotate(self.angle)
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
    LINE_W = 7.0    # spessore linea nera (px a finestra di riferimento)
    GLOW_W = 10.0   # spessore alone bianco
    DOT_R = 5.0     # raggio puntino alla base
    def __init__(self, is_left=True, parent=None):
        flags = Qt.FramelessWindowHint | Qt.WindowDoesNotAcceptFocus | Qt.Tool | Qt.WindowTransparentForInput
        if sys.platform.startswith("linux"): flags |= Qt.X11BypassWindowManagerHint
        super().__init__(parent, flags)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.is_left = is_left
        self._sx = 1.0   # scala orizzontale (lunghezza): proporzionale alla larghezza della finestra
        self._sy = 1.0   # scala verticale (distanza tra i baffi, spessore): proporzionale all'altezza della finestra
        self.setFixedSize(self.WIDTH, self.HEIGHT)
        self.anim_timer = QTimer(self)
        self.anim_timer.setInterval(16)
        self.anim_timer.timeout.connect(self._on_anim_tick)
        self.anim_time = 0.0
        self.is_wiggling = False
    def set_scale(self, sx: float, sy: float):
        sx = max(0.1, float(sx))
        sy = max(0.1, float(sy))
        if abs(sx - self._sx) < 1e-4 and abs(sy - self._sy) < 1e-4: return
        self._sx, self._sy = sx, sy
        self.setFixedSize(max(1, int(round(self.WIDTH * sx))), max(1, int(round(self.HEIGHT * sy))))
        self.update()
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
        self.update()
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        sx, sy = self._sx, self._sy
        # Le forme sono definite in coordinate "logiche" 340x180; x scala con la larghezza, y con l'altezza.
        # Si converte ogni punto (senza scalare il painter) così lo spessore delle linee resta uniforme.
        def P(x, y): return QPointF(x * sx, y * sy)
        base_x = float(self.WIDTH - 15) if self.is_left else 15.0
        center_y = self.HEIGHT / 2.0
        whisker_configs = [
            {"base_y": center_y - 28, "angle": -12.0, "len": 280.0, "phase": 0.0},
            {"base_y": center_y + 2,  "angle": 2.0,   "len": 305.0, "phase": 0.40},
            {"base_y": center_y + 32, "angle": 17.0,  "len": 275.0, "phase": 0.80},
        ]
        direction = -1.0 if self.is_left else 1.0
        paths = []
        for cfg in whisker_configs:
            by = cfg["base_y"]
            base_angle = cfg["angle"]
            length = cfg["len"]
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
        self.set_target_width(96)

    def set_side_margin(self, m):
        m = max(0, int(m))
        if m != self._side_margin:
            self._side_margin = m
            self.set_target_width(self._mouth_w)

    def _apply_size(self):
        depth = self._depth_for_width(self._mouth_w)
        height = int(round(2 * depth + 10))
        total = self._mouth_w + 2 * self._side_margin
        if self.width() != total or self.height() != height:
            self.setFixedSize(total, height)
            self.updateGeometry()
            self._update_mask()

    @classmethod
    def _depth_for_width(cls, width):
        a = (width - 2 * cls.PAD) / 4.0
        return max(14.0, min(20.0, a * 0.4))

    def set_target_width(self, width):
        self._mouth_w = int(width)
        self._apply_size()

    def _geom(self):
        m = float(self._side_margin)
        x0, x1 = m + float(self.PAD), m + float(self.width() - self._side_margin * 2 - self.PAD)
        cx = (x0 + x1) / 2.0
        a = (x1 - x0) / 4.0
        depth = self._depth_for_width(self._mouth_w)
        y_top = 0.5
        y_j = y_top + depth
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
        pen = QPen(off, 3.2)
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
        p.drawEllipse(QPointF(hx, hy), 5.5, 5.5)
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


_Frame = collections.namedtuple("_Frame", "gen pos img is_seek")
_EOF = object()
_LOOP = object()

class VideoProcessorThread(QThread):
    frame_ready = pyqtSignal(object, float, float, int, int)
    position_changed = pyqtSignal(int, int, float, float)
    playback_finished = pyqtSignal()
    loop_restarted = pyqtSignal()
    status_message = pyqtSignal(str)
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
        self.audio_clock = None
        self._seek_gen = 0
        self._shown_pos = 0
        self.display_size = None

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

    def set_video(self, video_path): self.video_path = video_path
    def request_seek_frame(self, frame_no: int): self.seek_target_frame = frame_no
    def request_seek_seconds(self, delta_seconds: float): self.seek_delta_sec = delta_seconds

    def run(self):
        if not self.video_path: return
        cap, self.decode_mode = open_video_capture(self.video_path)
        if not cap.isOpened(): return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if not video_fps or video_fps <= 0 or video_fps > 120: video_fps = 25.0
        self.video_fps = video_fps
        frame_interval = 1.0 / video_fps
        duration_sec = total_frames / video_fps
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.running = True
        self._shown_pos = 0
        self._rife_prev = None
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
        if mid_frame is not None and interp:
            # frame intermedio subito, frame reale mezzo intervallo dopo
            self.frame_ready.emit(np.ascontiguousarray(mid_frame), fps_val * 2.0, gpu_time_ms, full_w, full_h)
            time.sleep(0.5 / max(self.video_fps, 1.0))
            fps_val *= 2.0
        self.frame_ready.emit(np.ascontiguousarray(display_frame), fps_val, gpu_time_ms, full_w, full_h)

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
    """Menu contestuale a forma di fumetto, con la codina che punta al cursore."""
    TAIL_H = 14
    TAIL_W = 20
    PAD = 8
    ROW_H = 30
    RADIUS = 16
    MARGIN = 3
    def __init__(self, items, parent=None):
        # items: lista di (testo, callback, abilitata) oppure (testo, callback, abilitata, checked) per i toggle
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setMouseTracking(True)
        self.items = [tuple(it[:3]) + tuple(it[3:]) if len(it) > 3 else tuple(it[:3]) for it in items]
        self._hover = -1
        self._tail_top = True
        self._tail_x = 24
        self._font = QFont(resolve_ui_font_family(), 10)
        self._font.setBold(True)
        fm = QFontMetrics(self._font)
        # le voci possono avere 3 elementi (testo, callback, abilitata) o 4 (toggle con checked)
        self._body_w = max(fm.horizontalAdvance(it[0]) for it in items) + 2 * self.PAD + 28
        self._body_h = len(items) * self.ROW_H + 2 * self.PAD
        self.resize(self._body_w + 2 * self.MARGIN, self._body_h + self.TAIL_H + 2 * self.MARGIN)
    def popup_at(self, gpos):
        screen = QApplication.screenAt(gpos) or QApplication.primaryScreen()
        sg = screen.geometry()
        W, H, M = self.width(), self.height(), self.MARGIN
        inset = M + self.RADIUS + 8
        x = gpos.x() - inset
        if x + W > sg.right() + 1: x = gpos.x() - (W - inset)
        x = max(sg.left(), min(x, sg.right() + 1 - W))
        self._tail_x = max(M + self.RADIUS, min(gpos.x() - x, W - M - self.RADIUS))
        if gpos.y() + H > sg.bottom() + 1:
            self._tail_top = False
            y = gpos.y() - H + M
        else:
            self._tail_top = True
            y = gpos.y() - M
        self.move(x, max(sg.top(), y))
        self.show()
    def _body_rect(self):
        top = self.MARGIN + (self.TAIL_H if self._tail_top else 0)
        return QRectF(self.MARGIN, top, self._body_w, self._body_h)
    def _row_rect(self, i):
        b = self._body_rect()
        return QRectF(b.x() + self.PAD, b.y() + self.PAD + i * self.ROW_H, b.width() - 2 * self.PAD, self.ROW_H)
    def _row_at(self, pos):
        for i in range(len(self.items)):
            if self._row_rect(i).contains(QPointF(pos)): return i
        return -1
    def _bubble_path(self):
        body = self._body_rect()
        path = QPainterPath()
        path.addRoundedRect(body, self.RADIUS, self.RADIUS)
        tx = float(self._tail_x)
        lean = 1.0 if tx < self.width() / 2.0 else -1.0
        base_y = body.top() if self._tail_top else body.bottom()
        tip_y = float(self.MARGIN) if self._tail_top else self.height() - float(self.MARGIN)
        bx1 = tx + lean * 3.0
        bx2 = tx + lean * (3.0 + self.TAIL_W)
        tail = QPainterPath()
        tail.moveTo(bx1, base_y)
        tail.lineTo(tx, tip_y)
        tail.lineTo(bx2, base_y)
        tail.closeSubpath()
        return path.united(tail)
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
        for i, item in enumerate(self.items):
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
        p.end()
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
            cb = self.items[i][1]
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
        self.show_bg_texture = True
        self._last_hud_update = 0.0
        self._audio_last_pos = None
        self._audio_last_change = 0.0
        self._ears_key = None
        self._ears_force_raise = False
        self._whiskers_key = None
        self.last_video_dir = ""
        self.last_model_dir = ""
        self._file_icons = None
        self.theme_color = FRAME_COLOR
        self.bg_pixmap = None
        if os.path.isfile(BG_IMAGE_PATH): self.bg_pixmap = QPixmap(BG_IMAGE_PATH)
        self.container = BackgroundContainer(self.bg_pixmap, self.theme_color)
        self.setCentralWidget(self.container)
        self._video_aspect = None
        self._video_fill_mode = "height"  # "height": adatta il video all'altezza; "width": alla larghezza
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
        # Trascina e rilascia: finestra principale + overlay (sono finestre separate e intercetterebbero il drop)
        self.setAcceptDrops(True)
        for w in (self.ear_left, self.ear_right):
            w.setAcceptDrops(True)
        self.ear_timer = QTimer(self)
        self.ear_timer.setInterval(8)
        self.ear_timer.timeout.connect(self.sync_ears_position)
        self.ear_timer.start()
        self.audio_player = QMediaPlayer(self)
        self.worker = VideoProcessorThread()
        self.worker.frame_ready.connect(self.update_video_frame)
        self.worker.position_changed.connect(self.update_timeline)
        self.worker.playback_finished.connect(self.on_video_finished)
        self.worker.loop_restarted.connect(self.on_loop_restarted)
        self.worker.status_message.connect(lambda m: self.status.showMessage(m))
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
        if BYPASS_WM:
            area = QApplication.primaryScreen().availableGeometry()
            self.move(area.center() - QPoint(self.width() // 2, self.height() // 2))
            self._restore_stub = RestoreStub()
            self._restore_stub.restore_requested.connect(self._restore_from_stub)
        if initial_video and os.path.isfile(initial_video):
            self.play_video_file(initial_video)

    def init_ui(self):
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
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
        self.timeline_widget = QWidget()
        timeline_layout = QHBoxLayout()
        timeline_layout.setContentsMargins(0, 0, 0, 0)
        self.slider = ClickableSlider(Qt.Horizontal)
        self.slider.setObjectName("videoSlider")
        self.slider.setRange(0, 1000)
        self.slider.setValue(0)
        self.slider.setFocusPolicy(Qt.NoFocus)
        self.slider.sliderPressed.connect(self.on_slider_pressed)
        self.slider.sliderReleased.connect(self.on_slider_released)
        self.slider.sliderMoved.connect(self.on_slider_moved)
        timeline_layout.addWidget(self.slider)
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setStyleSheet("font-family: monospace; font-weight: bold; padding-left: 8px;")
        timeline_layout.addWidget(self.lbl_time)
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
        self.chk_denoise = QCheckBox("Denoise")
        self.chk_denoise.setChecked(False)
        self.chk_denoise.setToolTip("Attiva/disattiva Denoise AI")
        self.chk_denoise.setFocusPolicy(Qt.NoFocus)
        self.chk_denoise.stateChanged.connect(self.toggle_denoise)

        self.chk_upscale = QCheckBox("AI 2x")
        self.chk_upscale.setChecked(True)
        self.chk_upscale.setToolTip("Attiva/disattiva Upscaling AI")
        self.chk_upscale.setFocusPolicy(Qt.NoFocus)
        self.chk_upscale.stateChanged.connect(self.toggle_upscale)

        self.chk_rife = QCheckBox("RIFE")
        self.chk_rife.setChecked(False)
        self.chk_rife.setToolTip("Attiva/disattiva interpolazione fotogrammi RIFE")
        self.chk_rife.setFocusPolicy(Qt.NoFocus)
        self.chk_rife.stateChanged.connect(self.toggle_rife)

        self.chk_texture = QCheckBox("🖼️ Sfondo")
        self.chk_texture.setChecked(self.show_bg_texture)
        self.chk_texture.setToolTip("Attiva o disattiva la texture sfondo.png")
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
        for b in (self.btn_fullscreen, self.btn_minimize, self.btn_close):
            right_lay.addWidget(b)

        nose_row = CenteredRow(left_box, self.btn_play_pause, right_box, spacing=10)

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
        ai_lay.addWidget(self.btn_color)
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
        panel_lay.setSpacing(5)
        controls_layout.setSpacing(4)
        controls_layout.addWidget(nose_row)
        controls_layout.addWidget(self.mouth_row)
        self.controls_widget.setLayout(controls_layout)
        panel_lay.addWidget(self.timeline_widget)
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
        ph = self.controls_panel.sizeHint().height()
        band = max(120, ph + 46)
        top = max(0, ch - band)
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

    def _update_overlay_visibility(self):
        if not hasattr(self, "controls_panel"): return
        # In modalità minimal: pannello compatto (solo play + volume), mostrato
        # all'hover come di consueto; senza video resta sempre visibile
        if getattr(self, "_minimal_on", False):
            self._set_minimal_compact(True)
            if not self._has_video():
                self._controls_hover = True
                if not self._controls_visible or not self.controls_panel.isVisible():
                    self._show_overlay()
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
        self.controls_panel.setGeometry(x, y, w, min(h, max(0, ch - 16)))
        self.controls_panel.raise_()


    def _hide_overlay(self):
        if not hasattr(self, "controls_panel"): return
        if not self._has_video():
            if not self._controls_visible: self._show_overlay()
            return
        if self._drag_active or self._resize_active:
            self._overlay_hide_timer.start()
            return
        if getattr(self, "is_slider_dragged", False):
            self._overlay_hide_timer.start()
            return
        if self.slider.isSliderDown() or getattr(self.slider_volume, "_dragging", False):
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
            if ext in VIDEO_EXTS or db.mimeTypeForFile(p).name().startswith("video/"):
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

    def _arc_drag_start(self, gpos):
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
            self._normal_pos_before_minimal = QPoint(self.pos())
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
        # Calcolo puro (nessun effetto collaterale): le orecchie poggiano sul profilo superiore dell'ellisse
        # (semiassi w/2 e h/2); y = 0 è il bordo superiore della finestra, cioè il vertice dell'ellisse
        a = max(1.0, w / 2.0)
        b = max(1.0, h / 2.0)
        def top_y(x):
            t = max(-1.0, min(1.0, (x - a) / a))
            return b - b * math.sqrt(max(0.0, 1.0 - t * t))
        def tilt(x):
            # inclinazione della normale all'ellisse rispetto alla verticale (negativa a sinistra)
            t = max(-1.0, min(1.0, (x - a) / a))
            return math.atan2(t / a, (b - top_y(x)) / (b * b))
        chord = w
        ear_w = min(float(CatEar.BASE_WIDTH), max(260.0, chord * 0.36), chord * 0.48)
        ear_h = ear_w * (float(CatEar.BASE_HEIGHT) / float(CatEar.BASE_WIDTH))
        x_min = chord * EARS_X_FRACTION
        x_center_l = x_min
        for _ in range(6):
            half_base = (ear_w / 2.0) * math.cos(tilt(x_center_l))
            x_center_l = max(x_min, half_base + 2.0)
        x_center_r = chord - x_center_l
        y_center_l = top_y(x_center_l)
        y_center_r = top_y(x_center_r)
        angle_l = math.degrees(tilt(x_center_l))
        angle_r = math.degrees(tilt(x_center_r))
        sink_depth = 16.0
        anchor_x = self.ear_left.anchor_x
        anchor_y = self.ear_left.anchor_y
        rel_x_l = int(round(x_center_l - anchor_x))
        rel_y_l = int(round(y_center_l + sink_depth - anchor_y))
        rel_x_r = int(round(x_center_r - anchor_x))
        rel_y_r = int(round(y_center_r + sink_depth - anchor_y))
        # Quota (rispetto al bordo superiore della finestra) della punta più alta delle orecchie
        tip_l = rel_y_l + anchor_y - self.ear_left.top_reach(angle_l, ear_w, ear_h)
        tip_r = rel_y_r + anchor_y - self.ear_right.top_reach(angle_r, ear_w, ear_h)
        return {
            "angle_l": angle_l, "angle_r": angle_r, "ear_w": ear_w, "ear_h": ear_h,
            "rel_l": QPoint(rel_x_l, rel_y_l), "rel_r": QPoint(rel_x_r, rel_y_r),
            "tip_top": min(tip_l, tip_r),
        }

    def _recompute_ears_geometry(self, w: float, h: float):
        lay = self._ears_layout(w, h)
        self.ear_left.set_ear_config(lay["angle_l"], lay["ear_w"], lay["ear_h"])
        self.ear_right.set_ear_config(lay["angle_r"], lay["ear_w"], lay["ear_h"])
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

    def show_context_bubble(self, gpos):
        if getattr(self, "_bubble", None) is not None:
            try: self._bubble.close()
            except RuntimeError: pass
        can_fit = not (self.isFullScreen() or self.isMaximized() or self.isMinimized())
        has_video = self._has_video()
        self._bubble = BubbleMenu([
            ("Adatta altezza", self.fit_height_to_screen, can_fit),
            ("Adatta larghezza", self.fit_width_to_screen, can_fit),
            ("Adatta tutto", self.fit_all_to_screen, can_fit),
            ("Adatta video L", lambda: self.set_video_fill_mode("width"), has_video),
            ("Adatta video H", lambda: self.set_video_fill_mode("height"), has_video),
            # Toggle: dimensioni separate "minimal" (salvate in config come window_minimal_*)
            ("Minimal", self.toggle_minimal_mode, can_fit, bool(self._minimal_on)),
        ])
        self._bubble.popup_at(gpos)

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
        sx = eq_w / float(WHISKERS_REF_WIDTH) * WHISKERS_LENGTH_SCALE   # lunghezza: segue la larghezza
        sy = eq_h / float(WHISKERS_REF_HEIGHT)                          # distanza tra i baffi e spessore: seguono l'altezza
        self.whiskers_left.set_scale(sx, sy)
        self.whiskers_right.set_scale(sx, sy)
        w_w = self.whiskers_left.width()
        w_h = self.whiskers_left.height()
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
        pos_l = QPoint(left_x, whisker_y)
        pos_r = QPoint(right_x, whisker_y)
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
            self.video_display.setText("🐾 Neko Player\nNessun video caricato.\n[Tasto Q = Chiudi | 🎨 = Colore]")

    def _scale_mouth(self):
        if hasattr(self, "slider_volume"):
            self.slider_volume.set_target_width(max(96, min(320, int(self.width() * 0.22))))
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
        if not (self._drag_active or self._resize_active): QApplication.processEvents()

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
            "fullscreen": self.isFullScreen(),
            "volume": self.slider_volume.value(),
            "muted": self.is_muted,
            "show_osd": self.show_osd,
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

            saved_texture = config_data.get("show_bg_texture", True)
            self.show_bg_texture = saved_texture
            self.chk_texture.setChecked(saved_texture)
            self.container.set_show_texture(saved_texture)
            saved_fill = config_data.get("video_fill_mode", "height")
            if saved_fill in ("width", "height"): self._video_fill_mode = saved_fill
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
            if self.worker.running: self.toggle_play_pause()
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
        self.worker.request_seek_seconds(delta_sec)

    def on_slider_pressed(self): self.is_slider_dragged = True
    def on_slider_released(self):
        self.is_slider_dragged = False
        target_frame = self.slider.value()
        if self.worker.video_path:
            fps = self.worker.video_fps or 25.0
            target_ms = int((target_frame / fps) * 1000.0)
            self.audio_player.setPosition(target_ms)
            self.worker.request_seek_frame(target_frame)

    def on_slider_moved(self, value):
        if self.slider.maximum() > 0:
            ratio = value / self.slider.maximum()
            cur_sec = ratio * self.total_duration_sec
            self.lbl_time.setText(f"{format_time(cur_sec)} / {format_time(self.total_duration_sec)}")

    def update_timeline(self, cur_frame, total_frames, cur_sec, total_sec):
        self.total_duration_sec = total_sec
        if not self.is_slider_dragged:
            self.slider.blockSignals(True)
            self.slider.setMaximum(total_frames)
            self.slider.setValue(cur_frame)
            self.slider.blockSignals(False)
            self.lbl_time.setText(f"{format_time(cur_sec)} / {format_time(total_sec)}")

    def _remember_current_position(self):
        """Salva il timestamp del video attualmente caricato (per riprenderlo alla riapertura)."""
        try:
            path = self.worker.video_path
            if not path: return
            # frame mostrati dal worker; eventuale fallback: posizione dell'audio player
            pos_ms = int(self.worker._shown_pos * 1000.0 / max(1.0, self.worker.video_fps))
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
        # prima di cambiare video, ricorda dove eravamo arrivati nel precedente
        if self.worker.video_path: self._remember_current_position()
        self.worker.stop()
        self.audio_player.stop()
        self.worker.paused = False
        self.worker.audio_clock = None
        self._audio_last_pos = None
        url = QUrl.fromLocalFile(file_path)
        self.audio_player.setMedia(QMediaContent(url))
        self.worker.set_video(file_path)
        self.btn_play_pause.setEnabled(True)
        self.btn_play_pause.set_playing(True)
        self.status.showMessage(f"In riproduzione: {os.path.basename(file_path)}")
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
                target_frame = int(target / 1000.0 * max(1, int(self.worker.video_fps)))
                self.worker.request_seek_frame(target_frame)
                self.status.showMessage(
                    f"Ripresa da {format_time(target / 1000.0)}: {os.path.basename(file_path)}")
            QTimer.singleShot(400, _resume_at_saved)

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
        """Cambia il riempimento della finestra per il video: "width" o "height"."""
        if mode not in ("width", "height"): return
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
        if getattr(self, "_video_fill_mode", "height") == "width":
            # Adatta alla larghezza dell'area centrale; se è più alto, taglia sopra/sotto (centrato)
            if tw > 0 and abs(pixmap.width() - tw) > 1:
                pixmap = pixmap.scaledToWidth(tw, Qt.FastTransformation)
            if th > 0 and pixmap.height() > th:
                pixmap = pixmap.copy(0, (pixmap.height() - th) // 2, pixmap.width(), th)
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

            self.osd_label.setText(
                f"⚡ Framerate: {fps:.1f} FPS\n"
                f"⏱️ Latenza GPU: {gpu_ms:.1f} ms\n"
                f"🧠 Pipeline: {model_text}\n"
                f"🎬 Decodifica: {self.worker.decode_mode}\n"
                f"📺 Risoluzione Frame: {out_w}x{out_h}"
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
        self.show_default_background()
        self._update_overlay_visibility()

    def closeEvent(self, event):
        self._save_timer.stop()
        # prima di chiudere, ricorda il timestamp del video in riproduzione/pausa
        if self.worker.video_path: self._remember_current_position()
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
        if BYPASS_WM: self._restore_stub.close()
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
