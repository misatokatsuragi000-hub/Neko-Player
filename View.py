import os
import sys
import re
import shlex
import shutil
import hashlib
import subprocess
from datetime import datetime
from collections import OrderedDict, Counter

from PySide6.QtCore import (
    Qt, QAbstractTableModel, QModelIndex, QSize, QThreadPool, 
    QRunnable, QObject, Signal, QSettings, QTimer, QUrl, QEvent, QRect, QPoint,
    QFile, QMimeData, QFileInfo, QMimeDatabase, QThread, QItemSelectionModel
)
from PySide6.QtGui import (
    QPixmap, QImage, QImageReader, QColor, QPainter, QKeyEvent, 
    QWheelEvent, QMouseEvent, QFont, QDesktopServices, QCursor, QIcon, QDrag
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTableView, QSpinBox, QLabel, 
    QHeaderView, QStyledItemDelegate, QComboBox, QStackedWidget,
    QGraphicsView, QGraphicsScene, QFrame, QSlider, QLineEdit, QMenu,
    QDialog, QListWidget, QListWidgetItem, QDialogButtonBox, QMessageBox,
    QInputDialog, QAbstractItemView, QStyle, QSizePolicy, QLayout, QSpacerItem,
    QSplitter, QPlainTextEdit
)

try:
    from PySide6.QtWidgets import QFileIconProvider
except ImportError:
    try:
        from PySide6.QtGui import QFileIconProvider
    except ImportError:
        QFileIconProvider = None

# Qt Multimedia per la riproduzione video con supporto a QVideoSink
try:
    from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput, QVideoSink, QVideoFrame
    MULTIMEDIA_AVAILABLE = True
except ImportError:
    MULTIMEDIA_AVAILABLE = False

KNOWN_IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif', '.gif',
    '.svg', '.ico', '.tga', '.dds', '.avif', '.heic', '.heif', '.jxl',
    '.cr2', '.cr3', '.nef', '.arw', '.dng', '.rw2', '.orf', '.pef',
    '.raf', '.psd', '.xcf', '.pbm', '.pgm', '.ppm', '.xbm', '.xpm'
}
KNOWN_VIDEO_EXTENSIONS = {
    '.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v',
    '.mpg', '.mpeg', '.3gp', '.ts', '.vob', '.ogv'
}
VIEWABLE_IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.svg', '.ico', 
    '.tiff', '.tif', '.pbm', '.pgm', '.ppm', '.xbm', '.xpm'
}

# Ruoli dati personalizzati (Qt.UserRole = 256)
ROLE_IS_DIR = 256 + 1
ROLE_NAME = 256 + 2

def format_size(bytes_size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def format_time(ms):
    seconds = max(0, ms // 1000)
    mins = seconds // 60
    secs = seconds % 60
    return f"{mins:02d}:{secs:02d}"

def clean_file_path(path_str):
    if not path_str:
        return ""
    if path_str.startswith("file://"):
        return QUrl(path_str).toLocalFile()
    return os.path.abspath(path_str)

INFO_TEXT_CHARS = 5000

def read_file_head_text(path, limit=INFO_TEXT_CHARS):
    """Primi 'limit' caratteri del file letti come testo (utile per i metadati incorporati,
    es. i prompt salvati nei PNG). I byte non stampabili compaiono come '·'."""
    try:
        with open(path, "rb") as f:
            raw = f.read(limit * 4)     # margine per i caratteri multibyte
    except OSError as e:
        return f"(impossibile leggere il file: {e})"
    text = raw.decode("utf-8", errors="replace")
    text = "".join(ch if (ch in "\n\t" or (ch.isprintable() and ch != "\ufffd")) else "·" for ch in text)
    return text[:limit]

def parse_input_target():
    """Scansiona tutti gli argomenti da riga di comando ignorando eventuali flag di sistema (-qwindowtitle, ecc.)."""
    for arg in sys.argv[1:]:
        if arg.startswith("-"):
            continue
        cleaned = clean_file_path(arg)
        if cleaned and os.path.exists(cleaned):
            return cleaned
    return None

def get_script_shebang(script_path):
    """Legge la riga shebang (#!) di uno script Python in modo robusto.

    Restituisce il path dell'interprete dichiarato nello script (es.
    '#!/usr/bin/env python3' -> '/usr/bin/python3'), oppure None se lo script
    non ha uno shebang eseguibile / valido.
    """
    try:
        with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline().strip()
    except Exception:
        return None
    if not first_line.startswith('#!'):
        return None
    interp = first_line[2:].strip()
    if not interp:
        return None
    parts = interp.split()
    if parts and os.path.basename(parts[0]) == 'env' and len(parts) > 1:
        parts = parts[1:]  # gestisce '#!/usr/bin/env python3'
    exe = parts[0]
    resolved = shutil.which(exe) if not os.path.isabs(exe) else exe
    if resolved and os.path.isfile(resolved) and os.access(resolved, os.X_OK):
        return resolved
    return None


def find_system_python():
    """Fallback: usa il Python DI SISTEMA (non venv).

    L'eseguibile di sistema è /usr/bin/python; se assente si ripiega su
    /usr/bin/python3 e infine sugli eseguibili "python"/"python3" trovati nel
    PATH, escludendo però i venv (sys.executable di View.py potrebbe puntare a
    un ambiente virtuale privo dei moduli richiesti, es. gi/PyGObject).
    """
    preferred = ['/usr/bin/python', '/usr/bin/python3']
    for py in preferred:
        if os.path.isfile(py) and os.access(py, os.X_OK):
            return py

    for name in ('python3', 'python'):
        p = shutil.which(name)
        if p:
            rp = os.path.realpath(p)
            if 'venv' not in rp.lower() and '.virtualenv' not in rp.lower():
                return p
    return sys.executable


def build_launch_command(app_cmd, file_path):
    abs_path = os.path.abspath(file_path)

    if '@@' in app_cmd:
        cmd_str = re.sub(r'@@[a-zA-Z]*\s*%(?:[uUfF])\s*@@', f'@@ {shlex.quote(abs_path)} @@', app_cmd)
        cmd_str = re.sub(r'@@.*?@@', f'@@ {shlex.quote(abs_path)} @@', cmd_str)
        cmd_str = re.sub(r'%[a-zA-Z]', '', cmd_str)
        parts = [p for p in shlex.split(cmd_str) if p.strip()]
    elif re.search(r'["\']?%[uUfF]["\']?', app_cmd):
        cmd_str = re.sub(r'["\']?%[uUfF]["\']?', shlex.quote(abs_path), app_cmd)
        cmd_str = re.sub(r'%[a-zA-Z]', '', cmd_str)
        parts = [p for p in shlex.split(cmd_str) if p.strip()]
    else:
        # Percorso puro (es. /path/wallpaper.py selezionato dal file dialog): non deve
        # essere spezzato da shlex, altrimenti i percorsi con spazi vengono troncati
        # ("sh: cannot execute .../wallpaper.py") e lo script non parte.
        if os.path.isfile(app_cmd):
            parts = [os.path.abspath(app_cmd), abs_path]
        else:
            cmd_str = re.sub(r'["\']?%[a-zA-Z]["\']?', '', app_cmd)
            try:
                parts = [p for p in shlex.split(cmd_str) if p.strip()]
            except ValueError:
                parts = cmd_str.split()

            if len(parts) >= 2 and parts[0].endswith('flatpak') and parts[1] == 'run':
                parts.extend(['@@', abs_path, '@@'])
            else:
                parts.append(abs_path)

    # MODIFICA: Se si lancia uno script Python (.py):
    # 1) rispetta lo shebang (#!/usr/bin/env python3 o #!/percorso/python) scritto
    #    in cima allo script, così ogni script può dichiarare il SUO interprete;
    # 2) se non c'è alcuno shebang, usa il python di sistema (non sys.executable,
    #    che potrebbe essere un venv senza PyGObject/gi).
    if parts and parts[0].endswith('.py'):
        interp = get_script_shebang(parts[0]) or find_system_python()
        parts.insert(0, interp)

    return parts

def get_installed_desktop_apps():
    search_dirs = [
        "/usr/share/applications",
        "/var/lib/flatpak/exports/share/applications",
        os.path.expanduser("~/.local/share/flatpak/exports/share/applications"),
        os.path.expanduser("~/.local/share/applications")
    ]
    apps = {}
    for s_dir in search_dirs:
        if not os.path.isdir(s_dir):
            continue
        for fname in os.listdir(s_dir):
            if fname.endswith(".desktop"):
                fpath = os.path.join(s_dir, fname)
                try:
                    name, exec_cmd, nodisplay = None, None, False
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        in_entry = False
                        for line in f:
                            line = line.strip()
                            if line == "[Desktop Entry]":
                                in_entry = True
                            elif line.startswith("[") and line != "[Desktop Entry]":
                                in_entry = False
                            if in_entry:
                                if line.startswith("Name=") and not name:
                                    name = line.split("=", 1)[1]
                                elif line.startswith("Exec=") and not exec_cmd:
                                    exec_cmd = line.split("=", 1)[1]
                                elif line.startswith("NoDisplay=true"):
                                    nodisplay = True

                    if name and exec_cmd and not nodisplay:
                        clean_cmd = re.sub(r'["\']?%[uUfF]["\']?', '', exec_cmd)
                        clean_cmd = re.sub(r'@@.*?@@', '', clean_cmd)
                        clean_cmd = re.sub(r'%[a-zA-Z]', '', clean_cmd).strip()
                        clean_cmd = ' '.join(clean_cmd.split())
                        if clean_cmd:
                            apps[name] = clean_cmd
                except Exception:
                    pass
    return apps

class ShrinkRowLayout(QLayout):
    """Riga orizzontale di widget con riduzione 'dal piu' largo'.

    - Spazio sufficiente: ogni elemento ha la sua larghezza ideale, allineati a sinistra;
      lo spazio extra va a stretch e widget espandibili (es. campo di ricerca).
    - Spazio insufficiente: si riducono per primi gli elementi piu' larghi, fino a
      raggiungere la larghezza degli altri; poi si riducono tutti insieme, fino al minimo.
    """

    def __init__(self, parent=None, spacing=6):
        super().__init__(parent)
        self._items = []
        self._gap = spacing

    # --- API simile a QBoxLayout
    def addItem(self, item):
        self._items.append(item)

    def addSpacing(self, size):
        self.addItem(QSpacerItem(size, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum))

    def addStretch(self):
        self.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation.Horizontal

    # --- calcolo
    def _active(self):
        return [it for it in self._items if it.spacerItem() is not None or not it.isEmpty()]

    @staticmethod
    def _expands(item):
        try:
            return bool(item.expandingDirections() & Qt.Orientation.Horizontal)
        except Exception:
            return False

    def _metrics(self, item):
        """(larghezza ideale, larghezza minima, espandibile)"""
        spacer = item.spacerItem()
        if spacer is not None:
            if spacer.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding:
                return 0, 0, True
            w = spacer.sizeHint().width()
            return w, w, False
        hint = item.sizeHint().width()
        return hint, min(item.minimumSize().width(), hint), self._expands(item)

    def _widths(self, avail):
        active = self._active()
        mets = [self._metrics(it) for it in active]
        hints = [m[0] for m in mets]
        mins = [m[1] for m in mets]
        n = len(active)
        gaps = self._gap * max(0, n - 1)
        widths = list(hints)

        if sum(hints) + gaps <= avail:
            grow = [i for i, m in enumerate(mets) if m[2]]
            if grow:
                share, rem = divmod(avail - gaps - sum(hints), len(grow))
                for k, i in enumerate(grow):
                    widths[i] += share + (rem if k == len(grow) - 1 else 0)
            return active, widths

        need = avail - gaps
        if need <= sum(mins):
            return active, list(mins)

        # "Livello dell'acqua": larghezza massima L tale che sum(clamp(L, min, hint)) == spazio
        lo, hi = 0.0, float(max(hints))
        for _ in range(40):
            mid = (lo + hi) / 2
            if sum(min(max(mid, mn), h) for mn, h in zip(mins, hints)) > need:
                hi = mid
            else:
                lo = mid
        widths = [int(min(max(lo, mn), h)) for mn, h in zip(mins, hints)]
        leftover = need - sum(widths)
        level = int(lo)
        for i in sorted(range(n), key=lambda k: -hints[k]):
            if leftover <= 0:
                break
            if widths[i] == level and widths[i] < hints[i]:
                widths[i] += 1
                leftover -= 1
        return active, widths

    def sizeHint(self):
        active = self._active()
        m = self.contentsMargins()
        w = sum(self._metrics(it)[0] for it in active) + self._gap * max(0, len(active) - 1)
        h = max([it.sizeHint().height() for it in active if it.spacerItem() is None] or [0])
        return QSize(w + m.left() + m.right(), h + m.top() + m.bottom())

    def minimumSize(self):
        active = self._active()
        m = self.contentsMargins()
        w = sum(self._metrics(it)[1] for it in active) + self._gap * max(0, len(active) - 1)
        h = max([it.minimumSize().height() for it in active if it.spacerItem() is None] or [0])
        return QSize(w + m.left() + m.right(), h + m.top() + m.bottom())

    def setGeometry(self, rect):
        super().setGeometry(rect)
        m = self.contentsMargins()
        eff = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        active, widths = self._widths(eff.width())
        x = eff.x()
        for item, w in zip(active, widths):
            if item.spacerItem() is not None:
                h = eff.height()
            else:
                h = min(item.sizeHint().height(), eff.height())
            y = eff.y() + (eff.height() - h) // 2
            item.setGeometry(QRect(x, y, w, h))
            x += w + self._gap

def make_toolbar_shrinkable(layout, min_w=20):
    """Permette ai widget di una toolbar di restringersi (testo tagliato) invece di
    imporre alla finestra una larghezza minima pari alla somma di tutti.
    I pulsanti allineano il testo a sinistra, cosi' si vede sempre l'inizio."""
    for i in range(layout.count()):
        w = layout.itemAt(i).widget()
        if w is not None:
            w.setMinimumWidth(min_w)
            if isinstance(w, QPushButton):
                w.setProperty("toolbarBtn", True)

class AppChooserDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Seleziona Applicazione")
        self.resize(520, 560)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; color: #fff; }
            QLabel { color: #ccc; font-size: 13px; }
            QLineEdit { 
                background-color: #2b2b2b; color: #fff; 
                border: 1px solid #444; padding: 6px; border-radius: 4px; 
            }
            QListWidget { 
                background-color: #242424; color: #fff; 
                border: 1px solid #3d3d3d; border-radius: 4px; font-size: 13px; 
            }
            QListWidget::item { padding: 8px; border-radius: 3px; }
            QListWidget::item:selected { background-color: #00adb5; color: #000; font-weight: bold; }
            QPushButton { 
                background-color: #333; color: #fff; 
                border: 1px solid #555; padding: 6px 14px; border-radius: 4px; 
            }
            QPushButton:hover { background-color: #444; }
        """)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Cerca tra le applicazioni installate nel sistema:"))
        self.txt_filter = QLineEdit()
        self.txt_filter.setPlaceholderText("🔍 Es. Krita, GIMP, VLC, Dolphin, Gwenview...")
        self.txt_filter.textChanged.connect(self.filter_apps)
        layout.addWidget(self.txt_filter)

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        self.apps_dict = get_installed_desktop_apps()
        self.populate_list(self.apps_dict)

        btn_layout = QHBoxLayout()
        self.btn_browse = QPushButton("📁 Sfoglia File / AppImage...")
        self.btn_browse.clicked.connect(self.browse_custom)
        btn_layout.addWidget(self.btn_browse)

        btn_layout.addStretch()

        self.btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.btn_box.accepted.connect(self.accept)
        self.btn_box.rejected.connect(self.reject)
        btn_layout.addWidget(self.btn_box)

        layout.addLayout(btn_layout)

        self.selected_app_name = None
        self.selected_app_cmd = None

    def populate_list(self, apps):
        self.list_widget.clear()
        for name in sorted(apps.keys(), key=lambda s: s.casefold()):
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, apps[name])
            self.list_widget.addItem(item)
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

    def filter_apps(self, text):
        query = text.strip().casefold()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setHidden(query not in item.text().casefold())

    def browse_custom(self):
        fpath, _ = QFileDialog.getOpenFileName(self, "Seleziona Eseguibile o AppImage", "/usr/bin", "Tutti i file (*)")
        if fpath:
            self.selected_app_name = os.path.basename(fpath).capitalize()
            self.selected_app_cmd = os.path.abspath(fpath)
            super().accept()

    def accept(self):
        curr = self.list_widget.currentItem()
        if curr:
            self.selected_app_name = curr.text()
            self.selected_app_cmd = curr.data(Qt.ItemDataRole.UserRole)
        super().accept()

class FileItem:
    __slots__ = ('path', 'name', 'mtime', 'size', 'ext', 'is_video', 'is_image', 'is_natively_viewable', 'is_dir')
    def __init__(self, path, name, mtime, size, ext, is_video, is_image, is_natively_viewable, is_dir=False):
        self.is_dir = is_dir
        self.path = path
        self.name = name
        self.mtime = mtime
        self.size = size
        self.ext = ext
        self.is_video = is_video
        self.is_image = is_image
        self.is_natively_viewable = is_natively_viewable

class DirectoryLoader(QThread):
    loaded = Signal(str, list)
    failed = Signal(str)

    def __init__(self, folder):
        super().__init__()
        self.folder = folder
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        items = []
        try:
            with os.scandir(self.folder) as it:
                for entry in it:
                    if self._is_cancelled:
                        return
                    is_dir = entry.is_dir()   # segue i link simbolici a cartelle
                    if is_dir or entry.is_file():
                        name = entry.name
                        _, ext = os.path.splitext(name)
                        ext_lower = "" if is_dir else (ext.lower() if ext else "(senza estensione)")
                        try:
                            st = entry.stat(follow_symlinks=False)
                            mtime = st.st_mtime
                            size = st.st_size
                        except OSError:
                            mtime = 0
                            size = 0

                        if is_dir:
                            size = 0
                        is_vid = ext_lower in KNOWN_VIDEO_EXTENSIONS and MULTIMEDIA_AVAILABLE
                        is_img = ext_lower in KNOWN_IMAGE_EXTENSIONS
                        is_natively_viewable = is_vid or (ext_lower in VIEWABLE_IMAGE_EXTENSIONS)
                        items.append(FileItem(
                            path=os.path.abspath(entry.path),
                            name=name,
                            mtime=mtime,
                            size=size,
                            ext=ext_lower,
                            is_video=is_vid,
                            is_image=is_img,
                            is_natively_viewable=is_natively_viewable,
                            is_dir=is_dir
                        ))
            if not self._is_cancelled:
                self.loaded.emit(self.folder, items)
        except Exception as e:
            if not self._is_cancelled:
                self.failed.emit(str(e))

class WorkerSignals(QObject):
    loaded = Signal(str, QImage)
    failed = Signal(str)

class ThumbnailWorker(QRunnable):
    def __init__(self, filepath, size, is_video, ext):
        super().__init__()
        self.filepath = filepath
        self.size = size
        self.is_video = is_video
        self.ext = ext
        self.signals = WorkerSignals()

    def run(self):
        img = None
        if self.is_video:
            img = self._extract_video_thumbnail()
        else:
            img = self._extract_image_thumbnail()

        if img and not img.isNull():
            self.signals.loaded.emit(self.filepath, img)
        else:
            self.signals.failed.emit(self.filepath)

    def _extract_image_thumbnail(self):
        reader = QImageReader(self.filepath)
        reader.setAutoTransform(True)
        orig_size = reader.size()
        if orig_size.isValid():
            scaled = orig_size.scaled(self.size, Qt.AspectRatioMode.KeepAspectRatio)
            reader.setScaledSize(scaled)
            img = reader.read()
            if not img.isNull():
                return img

        try:
            uri = QUrl.fromLocalFile(os.path.abspath(self.filepath)).toString()
            md5_hash = hashlib.md5(uri.encode('utf-8')).hexdigest()
            home = os.path.expanduser('~')
            for folder in ['large', 'normal', 'x-large']:
                cached_thumb = os.path.join(home, '.cache', 'thumbnails', folder, f"{md5_hash}.png")
                if os.path.isfile(cached_thumb):
                    img = QImage(cached_thumb)
                    if not img.isNull():
                        return img.scaled(self.size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        except Exception:
            pass

        return None

    def _extract_video_thumbnail(self):
        for ss_time in ['00:00:00.5', '00:00:00']:
            try:
                cmd = [
                    'ffmpeg', '-loglevel', 'error',
                    '-ss', ss_time,
                    '-i', self.filepath,
                    '-frames:v', '1',
                    '-f', 'image2pipe',
                    '-vcodec', 'png',
                    '-pix_fmt', 'rgba',
                    '-'
                ]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=2.0)
                if proc.returncode == 0 and len(proc.stdout) > 200:
                    img = QImage.fromData(proc.stdout)
                    if not img.isNull():
                        return img.scaled(self.size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
            except Exception:
                pass

        try:
            import cv2
            cap = cv2.VideoCapture(self.filepath)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                h, w, ch = frame.shape
                if ch == 4:
                    rgba = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
                    qimg = QImage(rgba.data, w, h, ch * w, QImage.Format.Format_RGBA8888)
                else:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                return qimg.scaled(self.size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        except Exception:
            pass

        try:
            s_dim = max(self.size.width(), self.size.height())
            cmd = ['ffmpegthumbnailer', '-i', self.filepath, '-o', '-', '-s', str(s_dim), '-c', 'png']
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=1.5)
            if proc.returncode == 0 and len(proc.stdout) > 200:
                img = QImage.fromData(proc.stdout)
                if not img.isNull():
                    return img.scaled(self.size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        except Exception:
            pass

        return None

class GalleryModel(QAbstractTableModel):
    def __init__(self, cols=5):
        super().__init__()
        self.files = []
        self.path_map = {}
        self.cols = cols
        
        self.cache = OrderedDict()
        self.ext_card_cache = {}
        self.MAX_CACHE_SIZE = 2500
        
        self.loading = set()
        self.pending = {}       # percorso -> worker in coda (annullabile finche' non parte)
        self._no_sys_thumb = set()   # file senza miniatura di sistema: evita md5 + stat a ogni disegno
        self.cache_px = {}      # percorso -> lato (px) a cui e' stata generata la miniatura
        self.loading_px = {}    # percorso -> lato richiesto al worker in corso
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(max(2, os.cpu_count() or 4))
        self.cell_size = QSize(180, 180)

    def set_files(self, file_items, criterion="name", reverse=False, clear_cache=False):
        self.beginResetModel()
        self.files = file_items
        self._apply_sort(criterion, reverse)
        if clear_cache:
            self.clear_caches()
        self.cancel_pending()
        self.loading.clear()
        self.endResetModel()

    def clear_caches(self):
        self.cache.clear()
        self.ext_card_cache.clear()
        self._no_sys_thumb.clear()

    def cancel_pending(self, keep=None):
        """Toglie dalla coda le miniature non ancora partite (tranne quelle in 'keep')."""
        for path, worker in list(self.pending.items()):
            if keep is not None and path in keep:
                continue
            if self.threadpool.tryTake(worker):
                self.pending.pop(path, None)
                self.loading.discard(path)
                self.loading_px.pop(path, None)

    def schedule_window(self, first_row, last_row):
        """Chiamato mentre si scorre: scarta dalla coda le miniature ormai fuori vista e
        precarica (a bassa priorita') quelle appena sotto / sopra la zona visibile."""
        n = len(self.files)
        if n == 0:
            return
        cols = self.cols
        vis_lo = first_row * cols
        vis_hi = min(n, (last_row + 1) * cols)
        margin = min(max(1, last_row - first_row + 1), 12)
        lo = max(0, (first_row - margin) * cols)
        hi = min(n, (last_row + 1 + 2 * margin) * cols)

        self.cancel_pending(keep={it.path for it in self.files[lo:hi]})

        order = list(range(vis_hi, hi)) + list(range(vis_lo - 1, lo - 1, -1))
        for idx in order:
            item = self.files[idx]
            if (item.is_natively_viewable and not item.is_dir
                    and item.path not in self.cache and item.path not in self.loading):
                self._start_thumb_worker(item, prefetch=True)

    def sort_data(self, criterion="name", reverse=False):
        if not self.files:
            return
        self.beginResetModel()
        self._apply_sort(criterion, reverse)
        self.endResetModel()

    def _apply_sort(self, criterion, reverse):
        if criterion == "name":
            self.files.sort(key=lambda x: x.name.casefold(), reverse=reverse)
        elif criterion == "mtime":
            self.files.sort(key=lambda x: x.mtime, reverse=reverse)
        elif criterion == "size":
            self.files.sort(key=lambda x: x.size, reverse=reverse)

        # Cartelle sempre in cima (l'ordinamento e' stabile: l'ordine interno resta quello scelto)
        self.files.sort(key=lambda x: not x.is_dir)

        self.path_map = {item.path: i for i, item in enumerate(self.files)}

    def set_columns(self, cols):
        self.beginResetModel()
        self.cols = max(1, cols)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        if not self.files:
            return 0
        return (len(self.files) + self.cols - 1) // self.cols

    def columnCount(self, parent=QModelIndex()):
        return self.cols

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        if index.row() * self.cols + index.column() >= len(self.files):
            return Qt.ItemFlag.NoItemFlags  # celle vuote in fondo alla griglia
        return (Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsDragEnabled)

    def get_system_card(self, ext):
        ext_clean = ext.lower().replace('.', '').strip()
        cache_key = f"{ext_clean}_{self.cell_size.width()}x{self.cell_size.height()}"
        if cache_key in self.ext_card_cache:
            return self.ext_card_cache[cache_key]

        dummy_name = f"file.{ext_clean}" if ext_clean else "file"
        icon = QIcon()

        try:
            db = QMimeDatabase()
            mime = db.mimeTypeForFile(dummy_name, QMimeDatabase.MatchMode.MatchExtension)
            if mime.isValid():
                icon = QIcon.fromTheme(mime.iconName())
                if icon.isNull():
                    icon = QIcon.fromTheme(mime.genericIconName())
        except Exception:
            pass

        if icon.isNull() and QFileIconProvider:
            try:
                provider = QFileIconProvider()
                icon = provider.icon(QFileInfo(dummy_name))
            except Exception:
                pass

        side = min(self.cell_size.width(), self.cell_size.height())
        target_dim = int(side * 0.70)

        req_dim = max(256, target_dim)
        icon_pix = icon.pixmap(QSize(req_dim, req_dim))

        if not icon_pix.isNull():
            icon_pix = icon_pix.scaled(
                QSize(target_dim, target_dim),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

        pix = QPixmap(self.cell_size)
        pix.fill(QColor("#181b22"))
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        p.setPen(QColor("#2d3340"))
        p.drawRoundedRect(pix.rect().adjusted(2, 2, -2, -2), 6, 6)

        if not icon_pix.isNull():
            ix = (self.cell_size.width() - icon_pix.width()) // 2
            iy = (self.cell_size.height() - icon_pix.height() - 10) // 2
            p.drawPixmap(ix, iy, icon_pix)

        badge_text = ext_clean.upper() if ext_clean else "FILE"
        f = p.font()
        f.setPointSize(8)
        f.setBold(True)
        p.setFont(f)
        fm = p.fontMetrics()
        text_w = fm.horizontalAdvance(badge_text) + 12
        badge_rect = QRect((self.cell_size.width() - text_w) // 2, self.cell_size.height() - 22, text_w, 16)
        
        p.fillRect(badge_rect, QColor(0, 0, 0, 210))
        p.setPen(QColor("#00adb5"))
        p.drawRoundedRect(badge_rect, 3, 3)
        p.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_text)
        
        p.end()
        self.ext_card_cache[cache_key] = pix
        return pix

    def get_folder_card(self):
        side = max(1, min(self.cell_size.width(), self.cell_size.height()))
        cache_key = f"__folder__{side}"
        if cache_key in self.ext_card_cache:
            return self.ext_card_cache[cache_key]

        icon = QIcon.fromTheme("folder")
        if icon.isNull() and QFileIconProvider:
            try:
                icon = QFileIconProvider().icon(QFileIconProvider.IconType.Folder)
            except Exception:
                pass
        if icon.isNull():
            icon = QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)

        req = max(256, side)
        pix = icon.pixmap(QSize(req, req))
        self.ext_card_cache[cache_key] = pix
        return pix

    def check_system_cached_thumb(self, filepath):
        try:
            uri = QUrl.fromLocalFile(os.path.abspath(filepath)).toString()
            md5_hash = hashlib.md5(uri.encode('utf-8')).hexdigest()
            home = os.path.expanduser('~')
            for folder in ['large', 'normal', 'x-large']:
                cached_thumb = os.path.join(home, '.cache', 'thumbnails', folder, f"{md5_hash}.png")
                if os.path.isfile(cached_thumb):
                    img = QImage(cached_thumb)
                    if not img.isNull():
                        scaled = img.scaled(self.cell_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
                        return QPixmap.fromImage(scaled)
        except Exception:
            pass
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        idx = index.row() * self.cols + index.column()
        if idx >= len(self.files):
            return None

        item = self.files[idx]

        if item.is_dir:
            if role == Qt.ItemDataRole.DecorationRole:
                return self.get_folder_card()
            elif role == Qt.ItemDataRole.UserRole:
                return False
            elif role == ROLE_IS_DIR:
                return True
            elif role == ROLE_NAME:
                return item.name
            elif role == Qt.ItemDataRole.ToolTipRole:
                dt_str = datetime.fromtimestamp(item.mtime).strftime("%d/%m/%Y %H:%M:%S")
                return f"[Cartella] {item.name}\nModificata: {dt_str}"
            return None

        if role == Qt.ItemDataRole.DecorationRole:
            if item.path in self.cache:
                self.cache.move_to_end(item.path)
                # Dopo uno zoom in avanti la miniatura in cache puo' essere troppo piccola:
                # la mostro comunque (scalata) e ne rigenero una nitida in background
                side = max(self.cell_size.width(), self.cell_size.height())
                if (self.cache_px.get(item.path, side) < side
                        and item.is_natively_viewable and item.path not in self.loading):
                    self._start_thumb_worker(item)
                return self.cache[item.path]

            if not item.is_natively_viewable:
                if item.path not in self._no_sys_thumb:
                    cached_pix = self.check_system_cached_thumb(item.path)
                    if cached_pix:
                        self.cache[item.path] = cached_pix
                        return cached_pix
                    self._no_sys_thumb.add(item.path)
                return self.get_system_card(item.ext)

            if item.path not in self.loading:
                self._start_thumb_worker(item)

            return None

        elif role == Qt.ItemDataRole.UserRole:
            return item.is_video

        elif role == Qt.ItemDataRole.ToolTipRole:
            dt_str = datetime.fromtimestamp(item.mtime).strftime("%d/%m/%Y %H:%M:%S")
            tipo = "Video" if item.is_video else ("Immagine" if item.is_image else "File generico")
            return f"[{tipo}] {item.name}\nDimensione: {format_size(item.size)}\nModificato: {dt_str}"

        return None

    def _start_thumb_worker(self, item, prefetch=False):
        # Lato arrotondato a multipli di 64: piccoli zoom non obbligano a rigenerare tutto
        side = max(self.cell_size.width(), self.cell_size.height())
        gen = ((side + 63) // 64) * 64
        self.loading.add(item.path)
        self.loading_px[item.path] = gen
        worker = ThumbnailWorker(item.path, QSize(gen, gen), item.is_video, item.ext)
        worker.signals.loaded.connect(self.on_thumb_loaded)
        worker.signals.failed.connect(self.on_thumb_failed)
        worker.setAutoDelete(False)     # il worker resta nostro finche' e' in coda (pending)
        self.pending[item.path] = worker
        # le miniature visibili hanno la precedenza su quelle precaricate
        self.threadpool.start(worker, 0 if prefetch else 1)

    def on_thumb_loaded(self, filepath, qimage):
        self.pending.pop(filepath, None)
        pixmap = QPixmap.fromImage(qimage)
        gen = self.loading_px.pop(filepath, None)
        if gen is not None:
            self.cache_px[filepath] = gen
        if len(self.cache) >= self.MAX_CACHE_SIZE:
            self.cache.popitem(last=False)
        self.cache[filepath] = pixmap
        
        if filepath in self.loading:
            self.loading.remove(filepath)

        idx = self.path_map.get(filepath)
        if idx is not None:
            r = idx // self.cols
            c = idx % self.cols
            m_idx = self.index(r, c)
            self.dataChanged.emit(m_idx, m_idx, [Qt.ItemDataRole.DecorationRole])

    def on_thumb_failed(self, filepath):
        self.pending.pop(filepath, None)
        self.loading_px.pop(filepath, None)
        if filepath in self.loading:
            self.loading.remove(filepath)

        ext = os.path.splitext(filepath)[1].lower()
        self.cache[filepath] = self.get_system_card(ext)

        idx = self.path_map.get(filepath)
        if idx is not None:
            r = idx // self.cols
            c = idx % self.cols
            m_idx = self.index(r, c)
            self.dataChanged.emit(m_idx, m_idx, [Qt.ItemDataRole.DecorationRole])

def build_drag_pixmap(thumb, count=1):
    """Anteprima mostrata sotto il cursore durante il trascinamento."""
    size = 96
    canvas = QPixmap(size + 16, size + 16)
    canvas.fill(Qt.GlobalColor.transparent)
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setOpacity(0.85)

    if isinstance(thumb, QImage):
        thumb = QPixmap.fromImage(thumb)
    if isinstance(thumb, QPixmap) and not thumb.isNull():
        scaled = thumb.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
        painter.drawPixmap((canvas.width() - scaled.width()) // 2,
                           (canvas.height() - scaled.height()) // 2, scaled)
    else:
        painter.fillRect(8, 8, size, size, QColor("#242830"))

    if count > 1:
        painter.setOpacity(1.0)
        painter.setBrush(QColor("#00adb5"))
        painter.setPen(Qt.PenStyle.NoPen)
        badge = QRect(canvas.width() - 30, 0, 30, 30)
        painter.drawEllipse(badge)
        painter.setPen(QColor("#ffffff"))
        f = painter.font()
        f.setBold(True)
        painter.setFont(f)
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, str(count))
    painter.end()
    return canvas


def start_file_drag(source_widget, paths, thumb=None):
    """Avvia un drag & drop di file verso altre app. Ritorna True se, a drag
    concluso, almeno uno dei file non esiste piu' (spostato altrove)."""
    paths = [os.path.abspath(p) for p in paths if p and os.path.exists(p)]
    if not paths:
        return False

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(p) for p in paths])

    drag = QDrag(source_widget)
    drag.setMimeData(mime)
    pix = build_drag_pixmap(thumb, len(paths))
    drag.setPixmap(pix)
    drag.setHotSpot(QPoint(pix.width() // 2, pix.height() // 2))
    drag.exec(Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
              Qt.DropAction.CopyAction)
    return any(not os.path.exists(p) for p in paths)


def url_drop_acceptable(event, own_widget):
    md = event.mimeData()
    return (md.hasUrls()
            and any(u.isLocalFile() for u in md.urls())
            and event.source() is not own_widget)   # niente drop su se stessi


def read_url_drop(event):
    """Legge un drop di file: ritorna (percorsi, sposta?). Copia di default,
    Shift = sposta. Imposta anche l'azione di drop sull'evento."""
    paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
    move = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier) \
        and bool(event.possibleActions() & Qt.DropAction.MoveAction)
    if move:
        event.setDropAction(Qt.DropAction.MoveAction)
    elif event.possibleActions() & Qt.DropAction.CopyAction:
        event.setDropAction(Qt.DropAction.CopyAction)
    event.accept()
    return paths, move


class GalleryTableView(QTableView):
    """QTableView con drag & drop in stile file manager.

    - Trascinamento IN USCITA: i file selezionati (anche piu' di uno) possono
      essere trascinati su altre applicazioni / file manager (URI list).
    - Trascinamento IN ENTRATA: i file rilasciati vengono copiati nella cartella
      corrente (Shift premuto = sposta).
    """
    filesDropped = Signal(list, bool)   # (percorsi, sposta?)
    dragFinished = Signal()
    resized = Signal()
    zoomRequested = Signal(int)   # passi di rotella (+ = ingrandisci, - = rimpicciolisci)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Accumula il delta: i touchpad inviano molti piccoli valori (120 = uno scatto)
            self._wheel_accum += event.angleDelta().y()
            steps = int(self._wheel_accum / 120)
            if steps:
                self._wheel_accum -= steps * 120
                self.zoomRequested.emit(steps)
            event.accept()
            return

        # Scorrimento fluido: la rotella muove di ~0.6 righe di miniature (il comportamento
        # predefinito di QTableView arriva a saltare quasi una pagina intera per scatto)
        pd = event.pixelDelta().y()
        if pd:
            dy = -float(pd)                                   # touchpad: pixel esatti
        else:
            dy = -event.angleDelta().y() / 120.0 * self._wheel_step()
        self._scroll_accum += dy
        px = int(self._scroll_accum)
        if px:
            self._scroll_accum -= px
            sb = self.verticalScrollBar()
            sb.setValue(sb.value() + px)
        event.accept()

    def _wheel_step(self):
        return max(60, int(self.verticalHeader().defaultSectionSize() * 0.6))

    def __init__(self, parent=None):
        super().__init__(parent)
        self._wheel_accum = 0
        self._scroll_accum = 0.0
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)
        self.setDropIndicatorShown(False)

    def selected_paths(self):
        model = self.model()
        paths, seen = [], set()
        for ix in sorted(self.selectedIndexes(), key=lambda i: (i.row(), i.column())):
            idx = ix.row() * model.cols + ix.column()
            if 0 <= idx < len(model.files):
                path = model.files[idx].path
                if path not in seen:
                    seen.add(path)
                    paths.append(path)
        return paths

    def startDrag(self, supportedActions):
        paths = self.selected_paths()
        if not paths:
            return
        cur, sel = self.currentIndex(), self.selectedIndexes()
        ref = cur if (cur.isValid() and cur in sel) else (sel[0] if sel else cur)
        thumb = ref.data(Qt.ItemDataRole.DecorationRole) if ref.isValid() else None
        if start_file_drag(self, paths, thumb):
            self.dragFinished.emit()

    def dragEnterEvent(self, event):
        if url_drop_acceptable(event, self):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if url_drop_acceptable(event, self):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not url_drop_acceptable(event, self):
            event.ignore()
            return
        paths, move = read_url_drop(event)
        self.filesDropped.emit(paths, move)


class CenteredImageDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scaled_cache = OrderedDict()

    def _scaled(self, pixmap, size):
        """Scalare con SmoothTransformation costa molto: il risultato si riusa a ogni ridisegno."""
        key = (pixmap.cacheKey(), size.width(), size.height())
        hit = self._scaled_cache.get(key)
        if hit is not None:
            self._scaled_cache.move_to_end(key)
            return hit
        scaled = pixmap.scaled(size, Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
        self._scaled_cache[key] = scaled
        if len(self._scaled_cache) > 600:
            self._scaled_cache.popitem(last=False)
        return scaled

    def paint(self, painter, option, index):
        pixmap = index.data(Qt.ItemDataRole.DecorationRole)
        is_video = index.data(Qt.ItemDataRole.UserRole)
        is_dir = index.data(ROLE_IS_DIR)
        painter.save()
        painter.fillRect(option.rect, QColor("#141414"))

        if is_dir:
            rect = option.rect
            label_h = 24
            icon_area = rect.adjusted(10, 8, -10, -(label_h + 6))
            if pixmap and not pixmap.isNull() and icon_area.width() > 4 and icon_area.height() > 4:
                scaled = self._scaled(pixmap, icon_area.size())
                painter.drawPixmap(icon_area.x() + (icon_area.width() - scaled.width()) // 2,
                                   icon_area.y() + (icon_area.height() - scaled.height()) // 2, scaled)
            f = painter.font()
            f.setPointSize(9)
            f.setBold(True)
            painter.setFont(f)
            label_rect = QRect(rect.x() + 6, rect.bottom() - label_h - 2, max(1, rect.width() - 12), label_h)
            name = index.data(ROLE_NAME) or ""
            elided = painter.fontMetrics().elidedText(name, Qt.TextElideMode.ElideMiddle, label_rect.width())
            painter.setPen(QColor("#e6e6e6"))
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, elided)
        elif pixmap and not pixmap.isNull():
            scaled = self._scaled(pixmap, option.rect.size() - QSize(8, 8))
            x = option.rect.x() + (option.rect.width() - scaled.width()) // 2
            y = option.rect.y() + (option.rect.height() - scaled.height()) // 2
            
            painter.drawPixmap(x, y, scaled)

            if is_video:
                badge_rect = option.rect.adjusted(8, option.rect.height() - 28, -option.rect.width() + 75, -8)
                painter.fillRect(badge_rect, QColor(0, 0, 0, 190))
                painter.setPen(QColor("#00adb5"))
                f = painter.font()
                f.setPointSize(9)
                f.setBold(True)
                painter.setFont(f)
                painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, "▶ VIDEO")
        else:
            painter.setPen(QColor("#242830"))
            painter.drawRect(option.rect.adjusted(4, 4, -4, -4))

        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, QColor(0, 173, 181, 60))
            painter.setPen(QColor("#00adb5"))
            painter.drawRect(option.rect.adjusted(1, 1, -2, -2))

        painter.restore()

class SingleImageViewerWidget(QGraphicsView):
    filesDropped = Signal(list, bool)
    dragFinished = Signal()
    fullscreenRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._file_drag_origin = None
        self.setAcceptDrops(True)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = None
        self.current_filepath = ""
        self.is_fit_mode = True

        # Render hints di alta qualità: bilineare/bicubico e antialiasing
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        # Disattiviamo completamente le barre per evitare interferenze e calcoli errati a tutto schermo
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QColor("#0d0d0d"))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def showEvent(self, event):
        super().showEvent(event)
        if self.is_fit_mode:
            QTimer.singleShot(40, lambda: self.fit_to_window(force=False))

    # --- Drag & drop dell'immagine corrente ---
    # Se l'immagine e' adattata alla finestra (niente da scorrere) basta trascinarla.
    # Se e' ingrandita, il trascinamento normale sposta l'immagine: usa Ctrl+trascina
    # per portarla fuori come file.
    def _can_pan(self):
        if self.is_fit_mode:
            return False
        return (self.horizontalScrollBar().maximum() > 0
                or self.verticalScrollBar().maximum() > 0)

    def mousePressEvent(self, event):
        self._file_drag_origin = None
        if (event.button() == Qt.MouseButton.LeftButton and self.pixmap_item
                and self.current_filepath and os.path.isfile(self.current_filepath)):
            pos = event.position().toPoint()
            over_image = self.pixmap_item.contains(self.mapToScene(pos))
            ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            if over_image and (ctrl or not self._can_pan()):
                self._file_drag_origin = pos
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._file_drag_origin is not None:
            if event.buttons() & Qt.MouseButton.LeftButton:
                dist = (event.position().toPoint() - self._file_drag_origin).manhattanLength()
                if dist >= QApplication.startDragDistance():
                    self._file_drag_origin = None
                    thumb = self.pixmap_item.pixmap() if self.pixmap_item else None
                    if start_file_drag(self, [self.current_filepath], thumb):
                        self.dragFinished.emit()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._file_drag_origin is not None:
            self._file_drag_origin = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._file_drag_origin = None
            self.fullscreenRequested.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def dragEnterEvent(self, event):
        if url_drop_acceptable(event, self):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if url_drop_acceptable(event, self):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not url_drop_acceptable(event, self):
            event.ignore()
            return
        paths, move = read_url_drop(event)
        self.filesDropped.emit(paths, move)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap_item and self.is_fit_mode:
            self.fit_to_window(force=True, target_size=event.size())

    def set_image(self, filepath):
        self.scene.clear()
        self.resetTransform()
        self.current_filepath = filepath
        self.is_fit_mode = True
        
        abs_p = os.path.abspath(filepath)
        reader = QImageReader(abs_p)
        reader.setAutoTransform(True)
        img = reader.read()
        
        pixmap = QPixmap.fromImage(img) if not img.isNull() else QPixmap(abs_p)

        if not pixmap.isNull():
            self.pixmap_item = self.scene.addPixmap(pixmap)
            # Forza l'interpolazione Smooth (Bilineare/Bicubica) sul pixmap
            self.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
            self.scene.setSceneRect(self.pixmap_item.boundingRect())
            self.fit_to_window(force=True)
            QTimer.singleShot(50, lambda: self.fit_to_window(force=False))
            QTimer.singleShot(150, lambda: self.fit_to_window(force=False))
        else:
            self.pixmap_item = None
            err_text = self.scene.addText(f"Impossibile caricare l'immagine:\n{os.path.basename(filepath)}")
            err_text.setDefaultTextColor(QColor("#ffffff"))
            f = err_text.font()
            f.setPointSize(14)
            f.setBold(True)
            err_text.setFont(f)
            self.scene.setSceneRect(err_text.boundingRect())

    def get_effective_viewport_size(self, target_size=None):
        """Ottiene la dimensione reale della vista, interrogando lo schermo in caso di fullscreen."""
        if target_size is not None and target_size.width() > 20 and target_size.height() > 20:
            vw = target_size.width()
            vh = target_size.height()
        else:
            vw = self.viewport().width()
            vh = self.viewport().height()

        win = self.window()
        if win and win.isFullScreen():
            scr = win.screen()
            if scr:
                s_geo = scr.geometry()
                if vw < s_geo.width() or vh < s_geo.height():
                    vw = s_geo.width()
                    vh = s_geo.height()

        return vw, vh

    def get_fit_scale(self, target_size=None):
        if not self.pixmap_item or self.pixmap_item.pixmap().isNull():
            return 1.0
        vw, vh = self.get_effective_viewport_size(target_size)
        rect = self.pixmap_item.boundingRect()
        if rect.width() <= 0 or rect.height() <= 0 or vw <= 0 or vh <= 0:
            return 1.0
        return min(vw / rect.width(), vh / rect.height())

    def fit_to_window(self, force=False, target_size=None):
        if not force and not self.is_fit_mode:
            return
        if not self.pixmap_item or self.pixmap_item.pixmap().isNull():
            return

        vw, vh = self.get_effective_viewport_size(target_size)
        if vw <= 20 or vh <= 20:
            QTimer.singleShot(50, lambda: self.fit_to_window(force=True))
            return

        rect = self.pixmap_item.boundingRect()
        pw = rect.width()
        ph = rect.height()
        if pw <= 0 or ph <= 0:
            return

        self.is_fit_mode = True
        self.resetTransform()
        
        # Calcolo matematico pulito
        scale_val = min(vw / pw, vh / ph)
        self.scale(scale_val, scale_val)
        self.setSceneRect(rect)
        self.centerOn(self.pixmap_item)

    def set_original_size(self):
        if self.pixmap_item:
            self.is_fit_mode = False
            self.resetTransform()
            self.centerOn(self.pixmap_item)

    def wheelEvent(self, event: QWheelEvent):
        if not self.pixmap_item or self.pixmap_item.pixmap().isNull():
            return

        delta = event.angleDelta().y()
        if delta == 0:
            return

        zoom_in_factor = 1.15
        zoom_out_factor = 1.0 / zoom_in_factor
        factor = zoom_in_factor if delta > 0 else zoom_out_factor

        fit_scale = self.get_fit_scale()
        current_scale = self.transform().m11()
        target_scale = current_scale * factor

        min_scale = max(0.01, fit_scale * 0.10)
        max_scale = 50.0

        if delta < 0 and current_scale <= min_scale:
            event.accept()
            return
        if delta > 0 and current_scale >= max_scale:
            event.accept()
            return

        self.is_fit_mode = False

        mouse_pos = event.position().toPoint()
        scene_pos = self.mapToScene(mouse_pos)
        is_over_item = self.pixmap_item.contains(scene_pos)

        if target_scale > fit_scale and is_over_item:
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        else:
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

        self.scale(factor, factor)
        event.accept()

class VideoCanvas(QWidget):
    dragFinished = Signal()
    fullscreenRequested = Signal()
    volumeWheel = Signal(int)   # scatti di rotella (+ = volume su, - = volume giu')

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_frame = None
        self.filepath = ""
        self._file_drag_origin = None
        self._wheel_accum = 0
        self._osd_text = ""
        self._osd_timer = QTimer(self)
        self._osd_timer.setSingleShot(True)
        self._osd_timer.setInterval(900)
        self._osd_timer.timeout.connect(self._hide_osd)

    def wheelEvent(self, event):
        # Accumula il delta: i touchpad inviano molti piccoli valori (120 = uno scatto)
        self._wheel_accum += event.angleDelta().y()
        steps = int(self._wheel_accum / 120)
        if steps:
            self._wheel_accum -= steps * 120
            self.volumeWheel.emit(steps)
        event.accept()

    def show_osd(self, text):
        self._osd_text = text
        self._osd_timer.start()
        self.update()

    def _hide_osd(self):
        self._osd_text = ""
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.filepath \
                and os.path.isfile(self.filepath):
            self._file_drag_origin = event.position().toPoint()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._file_drag_origin is not None:
            if event.buttons() & Qt.MouseButton.LeftButton:
                dist = (event.position().toPoint() - self._file_drag_origin).manhattanLength()
                if dist >= QApplication.startDragDistance():
                    self._file_drag_origin = None
                    thumb = self.current_frame
                    if start_file_drag(self, [self.filepath], thumb):
                        self.dragFinished.emit()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._file_drag_origin = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._file_drag_origin = None
            self.fullscreenRequested.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def set_frame(self, qimage):
        self.current_frame = qimage
        self.update()

    def clear(self):
        self.current_frame = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor("#0d0d0d"))

        if self.current_frame and not self.current_frame.isNull():
            scaled = self.current_frame.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawImage(x, y, scaled)

        if self._osd_text:
            f = painter.font()
            f.setPointSize(16)
            f.setBold(True)
            painter.setFont(f)
            fm = painter.fontMetrics()
            box = QRect(0, 0, fm.horizontalAdvance(self._osd_text) + 32, fm.height() + 16)
            box.moveTopRight(QPoint(self.width() - 20, 20))
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 170))
            painter.drawRoundedRect(box, 8, 8)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(box, Qt.AlignmentFlag.AlignCenter, self._osd_text)

class VideoPlayerWidget(QWidget):
    filesDropped = Signal(list, bool)
    dragFinished = Signal()
    fullscreenRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_filepath = ""
        self.setAcceptDrops(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if MULTIMEDIA_AVAILABLE:
            self.player = QMediaPlayer(self)
            self.audio_output = QAudioOutput(self)
            self.player.setAudioOutput(self.audio_output)
            
            try:
                self.player.setLoops(QMediaPlayer.Loops.Infinite)
            except AttributeError:
                self.player.setLoops(-1)

            self.player.mediaStatusChanged.connect(self.on_media_status_changed)

            self.video_sink = QVideoSink(self)
            self.player.setVideoSink(self.video_sink)
            self.video_sink.videoFrameChanged.connect(self.on_video_frame_changed)

            self.canvas = VideoCanvas(self)
            self.canvas.dragFinished.connect(self.dragFinished)
            self.canvas.fullscreenRequested.connect(self.fullscreenRequested)
            self.canvas.volumeWheel.connect(self.on_canvas_wheel)
            layout.addWidget(self.canvas, 1)

            self.ctrl_bar_widget = QWidget()
            ctrl_bar = QHBoxLayout(self.ctrl_bar_widget)
            ctrl_bar.setContentsMargins(15, 8, 15, 12)

            self.btn_play = QPushButton("⏸ Pausa")
            self.btn_play.clicked.connect(self.toggle_play)
            ctrl_bar.addWidget(self.btn_play)

            self.lbl_time = QLabel("00:00 / 00:00")
            ctrl_bar.addWidget(self.lbl_time)

            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setRange(0, 0)
            self.slider.sliderMoved.connect(self.set_position)
            ctrl_bar.addWidget(self.slider)

            # --- Audio: pulsante muto + barra del volume (valori salvati nelle impostazioni)
            self._settings = QSettings("FastViewer", "Settings")
            saved_volume = max(0, min(100, self._settings.value("volume", 80, type=int)))
            saved_muted = self._settings.value("muted", False, type=bool)

            self.btn_mute = QPushButton("🔊")
            self.btn_mute.setToolTip("Silenzia / riattiva audio (M)")
            self.btn_mute.setFixedWidth(46)
            self.btn_mute.clicked.connect(self.toggle_mute)
            ctrl_bar.addWidget(self.btn_mute)

            self.volume_slider = QSlider(Qt.Orientation.Horizontal)
            self.volume_slider.setRange(0, 100)
            self.volume_slider.setValue(saved_volume)
            self.volume_slider.setMaximumWidth(110)
            self.volume_slider.setMinimumWidth(40)
            self.volume_slider.setToolTip(f"Volume: {saved_volume}%  (↑ / ↓)")
            self.audio_output.setVolume(self._slider_to_gain(saved_volume))
            self.audio_output.setMuted(saved_muted)
            self.volume_slider.valueChanged.connect(self.on_volume_changed)
            ctrl_bar.addWidget(self.volume_slider)
            self.update_mute_icon()

            make_toolbar_shrinkable(ctrl_bar)
            self.btn_mute.setMinimumWidth(46)
            self.btn_mute.setProperty("toolbarBtn", False)

            layout.addWidget(self.ctrl_bar_widget)

            self.player.positionChanged.connect(self.on_position_changed)
            self.player.durationChanged.connect(self.on_duration_changed)
        else:
            lbl_err = QLabel("QtMultimedia non disponibile.")
            lbl_err.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(lbl_err)

    @staticmethod
    def _slider_to_gain(value):
        """Curva quadratica: il volume percepito scala meglio rispetto a un guadagno lineare."""
        return (max(0, min(100, value)) / 100.0) ** 2

    def update_mute_icon(self):
        if not MULTIMEDIA_AVAILABLE:
            return
        v = self.volume_slider.value()
        if self.audio_output.isMuted() or v == 0:
            self.btn_mute.setText("🔇")
        elif v < 50:
            self.btn_mute.setText("🔉")
        else:
            self.btn_mute.setText("🔊")

    def on_volume_changed(self, value):
        self.audio_output.setVolume(self._slider_to_gain(value))
        if value > 0 and self.audio_output.isMuted():
            self.audio_output.setMuted(False)   # muovere la barra riattiva l'audio
        self._settings.setValue("volume", value)
        self._settings.setValue("muted", self.audio_output.isMuted())
        self.volume_slider.setToolTip(f"Volume: {value}%  (↑ / ↓)")
        self.update_mute_icon()

    def toggle_mute(self):
        if not MULTIMEDIA_AVAILABLE:
            return
        muting = not self.audio_output.isMuted()
        if not muting and self.volume_slider.value() == 0:
            self.volume_slider.setValue(50)     # riattivare con volume a zero non avrebbe effetto
        self.audio_output.setMuted(muting)
        self._settings.setValue("muted", muting)
        self.update_mute_icon()

    def change_volume(self, delta):
        if not MULTIMEDIA_AVAILABLE:
            return
        if delta > 0 and self.audio_output.isMuted():
            self.audio_output.setMuted(False)
            self._settings.setValue("muted", False)
        self.volume_slider.setValue(max(0, min(100, self.volume_slider.value() + delta)))
        self.update_mute_icon()

    def on_canvas_wheel(self, steps):
        """Rotella sul video: regola il volume (5% a scatto, come le frecce su/giu')."""
        self.change_volume(5 * steps)
        if self.audio_output.isMuted() or self.volume_slider.value() == 0:
            self.canvas.show_osd("🔇 Muto")
        else:
            self.canvas.show_osd(f"🔊 Volume {self.volume_slider.value()}%")

    def on_video_frame_changed(self, frame: QVideoFrame):
        if frame.isValid():
            img = frame.toImage()
            self.canvas.set_frame(img)

    def on_media_status_changed(self, status):
        if MULTIMEDIA_AVAILABLE and status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.player.setPosition(0)
            self.player.play()

    def set_controls_visible(self, visible):
        if hasattr(self, 'ctrl_bar_widget'):
            self.ctrl_bar_widget.setVisible(visible)

    def load_video(self, filepath):
        self.current_filepath = os.path.abspath(filepath)
        if MULTIMEDIA_AVAILABLE:
            self.canvas.filepath = self.current_filepath
            self.player.setSource(QUrl.fromLocalFile(self.current_filepath))
            self.player.play()
            self.btn_play.setText("⏸ Pausa")

    def toggle_play(self):
        if not MULTIMEDIA_AVAILABLE:
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.btn_play.setText("▶ Riproduci")
        else:
            self.player.play()
            self.btn_play.setText("⏸ Pausa")

    def stop(self):
        if MULTIMEDIA_AVAILABLE:
            self.player.stop()
            self.canvas.clear()
            self.canvas.filepath = ""

    def dragEnterEvent(self, event):
        if url_drop_acceptable(event, self):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if url_drop_acceptable(event, self):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not url_drop_acceptable(event, self):
            event.ignore()
            return
        paths, move = read_url_drop(event)
        self.filesDropped.emit(paths, move)

    def set_position(self, pos):
        if MULTIMEDIA_AVAILABLE:
            self.player.setPosition(pos)

    def on_position_changed(self, pos):
        self.slider.setValue(pos)
        dur = self.player.duration()
        self.lbl_time.setText(f"{format_time(pos)} / {format_time(dur)}")

    def on_duration_changed(self, dur):
        self.slider.setRange(0, dur)

class FastGalleryViewer(QMainWindow):
    def __init__(self, target_path=None):
        super().__init__()
        self.setWindowTitle("Fast Media Gallery - Linux Mint Optimizer")
        self.resize(1320, 840)
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; color: #fff; }
            QLabel { color: #ccc; font-size: 13px; }
            QLineEdit, QSpinBox, QComboBox { 
                background-color: #242424; 
                color: #fff; 
                border: 1px solid #3d3d3d; 
                padding: 1px 6px; 
                border-radius: 4px;
            }
            QLineEdit:focus { border: 1px solid #00adb5; }
            QPushButton {
                background-color: #242424;
                border: 1px solid #3d3d3d;
                padding: 5px 12px;
                border-radius: 4px;
                color: #fff;
            }
            QPushButton:hover { background-color: #383838; }
            QPushButton[toolbarBtn="true"] { text-align: left; padding: 2px 8px; }
            QMenu {
                background-color: #242424;
                color: #fff;
                border: 1px solid #3d3d3d;
                padding: 5px;
            }
            QMenu::item {
                padding: 6px 22px;
                border-radius: 3px;
            }
            QMenu::item:selected {
                background-color: #00adb5;
                color: #000;
                font-weight: bold;
            }
            QMenu::separator {
                height: 1px;
                background: #3d3d3d;
                margin: 4px 6px;
            }
            QSlider::groove:horizontal { height: 6px; background: #333; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #00adb5; border-radius: 3px; }
            QSlider::handle:horizontal { background: #fff; width: 14px; margin-top: -4px; margin-bottom: -4px; border-radius: 7px; }
        """)

        self.settings = QSettings("FastViewer", "Settings")
        self.all_file_items = []
        self.current_single_idx = -1
        self.ui_hidden = False
        self._was_maximized_before_fs = True
        self.dir_loader_thread = None

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.init_gallery_ui()
        self.init_single_view_ui()

        self.stack.setCurrentIndex(0)
        QApplication.instance().installEventFilter(self)
        self.restore_window_state()

        clean_target = clean_file_path(target_path)
        initial_file_to_open = None

        if clean_target and os.path.exists(clean_target):
            if os.path.isfile(clean_target):
                folder = os.path.dirname(clean_target)
                initial_file_to_open = clean_target
            else:
                folder = clean_target
            self.settings.setValue("last_folder", folder)
        else:
            folder = self.settings.value("last_folder", "")

        # Se richiesto un file, mostralo istantaneamente prima di qualsiasi operazione lenta su disco
        if initial_file_to_open:
            self.open_single_file_direct(initial_file_to_open)

        # Scansione della cartella in thread parallelo (UI fluida a 60 FPS)
        if folder and os.path.isdir(folder):
            self.load_directory(folder)

    def save_window_state(self):
        """Salva dimensione/posizione 'normali' e se la finestra era massimizzata.
        In schermo intero si salva lo stato precedente al fullscreen."""
        geo = self.normalGeometry()
        if geo.width() > 0 and geo.height() > 0:
            self.settings.setValue("win_x", geo.x())
            self.settings.setValue("win_y", geo.y())
            self.settings.setValue("win_w", geo.width())
            self.settings.setValue("win_h", geo.height())
        maximized = self._was_maximized_before_fs if self.isFullScreen() else self.isMaximized()
        self.settings.setValue("win_maximized", bool(maximized))

    def restore_window_state(self):
        """Ripristina l'ultima dimensione/posizione; al primo avvio resta massimizzata."""
        w = self.settings.value("win_w", 0, type=int)
        h = self.settings.value("win_h", 0, type=int)
        maximized = self.settings.value("win_maximized", True, type=bool)

        if w >= 200 and h >= 150:
            self.resize(w, h)
            if self.settings.contains("win_x") and self.settings.contains("win_y"):
                x = self.settings.value("win_x", 0, type=int)
                y = self.settings.value("win_y", 0, type=int)
                rect = QRect(x, y, w, h)
                # Sposta solo se la finestra resta ben visibile su uno schermo attuale
                # (monitor scollegato / risoluzione cambiata)
                for screen in QApplication.screens():
                    inter = screen.availableGeometry().intersected(rect)
                    if inter.width() >= 120 and inter.height() >= 60:
                        self.move(x, y)
                        break

        self._was_maximized_before_fs = maximized
        if maximized:
            self.showMaximized()
        else:
            self.show()

    def closeEvent(self, event):
        self.save_window_state()
        if self.dir_loader_thread and self.dir_loader_thread.isRunning():
            self.dir_loader_thread.cancel()
            self.dir_loader_thread.wait(500)
        if self.cmp_loader_thread and self.cmp_loader_thread.isRunning():
            self.cmp_loader_thread.cancel()
            self.cmp_loader_thread.wait(500)
        super().closeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.stack.currentIndex() == 1 and hasattr(self, 'image_viewer'):
            if self.image_viewer.is_fit_mode:
                self.image_viewer.fit_to_window(force=True)

    def open_single_file_direct(self, file_path):
        abs_target = os.path.abspath(file_path)
        ext = os.path.splitext(abs_target)[1].lower()
        self.stack.setCurrentIndex(1)

        if ext in KNOWN_VIDEO_EXTENSIONS and MULTIMEDIA_AVAILABLE:
            self.btn_orig_size.hide()
            self.btn_fit.hide()
            self.single_media_stack.setCurrentIndex(1)
            self.video_player.load_video(abs_target)
            self.lbl_single_info.setText(f"[Video] {os.path.basename(abs_target)}")
        else:
            self.video_player.stop()
            self.btn_orig_size.show()
            self.btn_fit.show()
            self.single_media_stack.setCurrentIndex(0)
            self.image_viewer.set_image(abs_target)
            self.lbl_single_info.setText(f"[Immagine] {os.path.basename(abs_target)}")

    def scroll_gallery_rows(self, delta_rows):
        table = self.active_table()
        sb = table.verticalScrollBar()
        row_h = table.verticalHeader().defaultSectionSize()
        if row_h <= 0:
            row_h = self.spin_row_h.value()
        sb.setValue(sb.value() + (delta_rows * row_h))

    def toggle_ui_visibility(self):
        self.ui_hidden = not self.ui_hidden
        self.gallery_toolbar_widget.setVisible(not self.ui_hidden)
        self.single_toolbar_widget.setVisible(not self.ui_hidden)
        self.video_player.set_controls_visible(not self.ui_hidden)

        if self.stack.currentIndex() == 1 and 0 <= self.current_single_idx < len(self.single_model.files):
            if not self.single_model.files[self.current_single_idx].is_video:
                if self.image_viewer.is_fit_mode:
                    QTimer.singleShot(40, lambda: self.image_viewer.fit_to_window(force=True))

    def eventFilter(self, watched, event):
        if (event.type() == QEvent.Type.Resize and self.info_visible
                and watched is self.single_media_stack):
            self.position_info_overlay()

        if self.stack.currentIndex() == 1:
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.RightButton:
                if not isinstance(watched, QMenu):
                    if self.isFullScreen():
                        self.toggle_fullscreen()
                    else:
                        self.show_gallery()
                    return True

        if event.type() == QEvent.Type.MouseButtonPress:
            if watched != self.txt_search and self.txt_search.hasFocus():
                self.txt_search.clearFocus()

        if event.type() == QEvent.Type.KeyPress:
            is_typing = self.txt_search.hasFocus() or watched == self.txt_search
            if is_typing:
                if event.key() == Qt.Key.Key_Escape:
                    self.txt_search.clear()
                    self.table.setFocus()
                    return True
                return False

            if event.key() == Qt.Key.Key_Tab:
                self.toggle_ui_visibility()
                return True

            if event.key() == Qt.Key.Key_Q:
                self.close()
                return True

            if event.key() == Qt.Key.Key_F:
                self.toggle_fullscreen()
                return True

            if event.key() == Qt.Key.Key_Delete:
                if self.stack.currentIndex() == 1:
                    self.delete_current_single_file()
                    return True
                elif self.stack.currentIndex() == 0:
                    self.delete_selected_gallery_file()
                    return True

            if self.stack.currentIndex() == 0:
                if event.key() == Qt.Key.Key_F2:
                    self.rename_selected_gallery_file()
                    return True
                elif event.key() == Qt.Key.Key_Backspace and QApplication.activeModalWidget() is None:
                    self.go_to_parent_folder()
                    return True
                elif event.key() == Qt.Key.Key_W:
                    self.scroll_gallery_rows(-2)
                    return True
                elif event.key() == Qt.Key.Key_S:
                    self.scroll_gallery_rows(2)
                    return True

            if self.stack.currentIndex() == 1:
                if event.key() == Qt.Key.Key_Escape:
                    if self.isFullScreen():
                        self.toggle_fullscreen()
                    else:
                        self.show_gallery()
                    return True
                elif event.key() == Qt.Key.Key_I:
                    self.toggle_info_overlay()
                    return True
                elif event.key() in (Qt.Key.Key_Left, Qt.Key.Key_A):
                    prev_idx = self.get_adjacent_viewable_index(-1)
                    if prev_idx != -1:
                        self.current_single_idx = prev_idx
                        self.load_single_media(prev_idx)
                    return True
                elif event.key() in (Qt.Key.Key_Right, Qt.Key.Key_D):
                    next_idx = self.get_adjacent_viewable_index(1)
                    if next_idx != -1:
                        self.current_single_idx = next_idx
                        self.load_single_media(next_idx)
                    return True
                elif event.key() == Qt.Key.Key_Space:
                    item = self.single_model.files[self.current_single_idx]
                    if item.is_video:
                        self.video_player.toggle_play()
                        return True
                elif (event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_M)
                      and self.single_media_stack.currentIndex() == 1):
                    if event.key() == Qt.Key.Key_M:
                        self.video_player.toggle_mute()
                    else:
                        self.video_player.change_volume(5 if event.key() == Qt.Key.Key_Up else -5)
                    return True

        return super().eventFilter(watched, event)

    def get_adjacent_viewable_index(self, direction):
        idx = self.current_single_idx + direction
        while 0 <= idx < len(self.single_model.files):
            if self.single_model.files[idx].is_natively_viewable:
                return idx
            idx += direction
        return -1

    def init_gallery_ui(self):
        gallery_widget = QWidget()
        layout = QVBoxLayout(gallery_widget)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(3)

        self.gallery_toolbar_widget = QWidget()
        toolbar = ShrinkRowLayout(self.gallery_toolbar_widget)
        toolbar.setContentsMargins(0, 0, 0, 0)
        
        self.btn_open = QPushButton("📁 Apri Cartella")
        self.btn_open.clicked.connect(self.choose_folder)
        toolbar.addWidget(self.btn_open)

        self.btn_up = QPushButton("⬆ Su")
        self.btn_up.setToolTip("Cartella superiore (Backspace)")
        self.btn_up.clicked.connect(self.go_to_parent_folder)
        toolbar.addWidget(self.btn_up)

        toolbar.addSpacing(8)

        self.txt_search = QLineEdit()
        self.txt_search.setPlaceholderText("🔍 Cerca nome file...")
        self.txt_search.setClearButtonEnabled(True)
        self.txt_search.setMinimumWidth(40)
        self.txt_search.textChanged.connect(self.apply_filter_and_sort)
        self.txt_search.returnPressed.connect(lambda: self.table.setFocus())
        toolbar.addWidget(self.txt_search)

        toolbar.addSpacing(8)

        toolbar.addWidget(QLabel("Mostra:"))
        self.combo_filter = QComboBox()
        self.combo_filter.addItem("Tutti i file", "ALL")
        self.combo_filter.currentIndexChanged.connect(self.apply_filter_and_sort)
        toolbar.addWidget(self.combo_filter)

        toolbar.addSpacing(8)

        toolbar.addWidget(QLabel("Ordina:"))
        self.combo_sort = QComboBox()
        self.combo_sort.addItem("Nome", "name")
        self.combo_sort.addItem("Data Modifica", "mtime")
        self.combo_sort.addItem("Dimensione", "size")
        
        saved_sort = self.settings.value("sort_criterion", "name")
        idx_sort = self.combo_sort.findData(saved_sort)
        if idx_sort != -1:
            self.combo_sort.setCurrentIndex(idx_sort)
        self.combo_sort.currentIndexChanged.connect(self.apply_filter_and_sort)
        toolbar.addWidget(self.combo_sort)

        self.is_reverse = self.settings.value("sort_reverse", False, type=bool)
        self.btn_reverse = QPushButton()
        self.update_reverse_btn_label()
        self.btn_reverse.clicked.connect(self.toggle_sort_direction)
        toolbar.addWidget(self.btn_reverse)

        toolbar.addSpacing(12)

        toolbar.addWidget(QLabel("Colonne:"))
        self.spin_cols = QSpinBox()
        self.spin_cols.setRange(1, 20)
        self.spin_cols.setValue(self.settings.value("columns", 5, type=int))
        self.spin_cols.valueChanged.connect(self.update_grid_layout)
        toolbar.addWidget(self.spin_cols)

        toolbar.addWidget(QLabel("Altezza (px):"))
        self.spin_row_h = QSpinBox()
        self.spin_row_h.setRange(60, 800)
        self.spin_row_h.setValue(self.settings.value("row_height", 180, type=int))
        self.spin_row_h.valueChanged.connect(self.update_grid_layout)
        self.spin_row_h.setToolTip("Dimensione miniature (anche Ctrl + rotella del mouse)")
        toolbar.addWidget(self.spin_row_h)

        toolbar.addSpacing(8)

        self.btn_fs = QPushButton("⛶ Schermo Intero (F)")
        self.btn_fs.clicked.connect(self.toggle_fullscreen)
        toolbar.addWidget(self.btn_fs)

        self.btn_compare = QPushButton("⚖ Compara")
        self.btn_compare.setToolTip(
            "Confronta la cartella corrente con un'altra:\n"
            "a sinistra i file solo nella cartella originale, a destra quelli solo nella nuova")
        self.btn_compare.clicked.connect(self.toggle_compare)
        toolbar.addWidget(self.btn_compare)

        self.lbl_stats = QLabel("Nessuna cartella caricata")
        toolbar.addWidget(self.lbl_stats)
        toolbar.addStretch()
        make_toolbar_shrinkable(toolbar)

        self.gallery_toolbar_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.gallery_toolbar_widget)

        self.model = GalleryModel(cols=self.spin_cols.value())
        self.single_model = self.model    # modello da cui lo viewer singolo legge i file
        self.compare_folder = None        # seconda cartella (None = confronto spento)
        self.cmp_items = None
        self.cmp_loader_thread = None

        # Applica lo zoom dopo una breve pausa: la rotella genera molti eventi ravvicinati
        self._zoom_timer = QTimer(self)
        self._zoom_timer.setSingleShot(True)
        self._zoom_timer.setInterval(40)
        self._zoom_timer.timeout.connect(self.update_grid_layout)

        # Durante lo scorrimento: ogni 50 ms scarta le miniature fuori vista e precarica le prossime
        self._thumb_timer = QTimer(self)
        self._thumb_timer.setSingleShot(True)
        self._thumb_timer.setInterval(50)
        self._thumb_timer.timeout.connect(self.schedule_thumbnails)

        self.table = self._build_gallery_table(self.model)
        self.model_cmp = GalleryModel(cols=self.spin_cols.value())
        self.table_cmp = self._build_gallery_table(self.model_cmp, compare_side=True)

        header_style = ("padding: 2px 10px; background-color: #1b1f27; color: #e6e6e6;"
                        "font-weight: bold; border-bottom: 1px solid #00adb5;")

        left_panel = QWidget()
        left_lay = QVBoxLayout(left_panel)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(0)
        self.lbl_cmp_left = QLabel()
        self.lbl_cmp_left.setStyleSheet(header_style)
        self.lbl_cmp_left.hide()
        left_lay.addWidget(self.lbl_cmp_left)
        left_lay.addWidget(self.table)

        self.cmp_right_panel = QWidget()
        right_lay = QVBoxLayout(self.cmp_right_panel)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(0)
        self.lbl_cmp_right = QLabel()
        self.lbl_cmp_right.setStyleSheet(header_style)
        right_lay.addWidget(self.lbl_cmp_right)
        right_lay.addWidget(self.table_cmp)
        self.cmp_right_panel.hide()

        self.cmp_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.cmp_splitter.setChildrenCollapsible(False)
        self.cmp_splitter.setHandleWidth(3)
        self.cmp_splitter.setStyleSheet("QSplitter::handle { background-color: #00adb5; }")
        self.cmp_splitter.addWidget(left_panel)
        self.cmp_splitter.addWidget(self.cmp_right_panel)
        self.cmp_splitter.setStretchFactor(0, 1)
        self.cmp_splitter.setStretchFactor(1, 1)

        self.cmp_splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.cmp_splitter, 1)   # la galleria prende tutto lo spazio verticale in piu'
        self.stack.addWidget(gallery_widget)
        self.update_grid_layout()

    def _build_gallery_table(self, model, compare_side=False):
        table = GalleryTableView()
        table.setModel(model)
        table.setItemDelegate(CenteredImageDelegate(self))

        table.horizontalHeader().setVisible(False)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setMinimumSectionSize(1)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        table.resized.connect(self.update_effective_columns)
        table.zoomRequested.connect(self.on_gallery_zoom)
        table.setVerticalScrollMode(QTableView.ScrollMode.ScrollPerPixel)
        table.setHorizontalScrollMode(QTableView.ScrollMode.ScrollPerPixel)
        table.clicked.connect(lambda ix, t=table: self.on_cell_clicked(ix, t))
        if compare_side:
            # I file rilasciati sul lato destro finiscono nella seconda cartella
            table.filesDropped.connect(
                lambda paths, move: self.import_dropped_files(paths, move, self.compare_folder))
        else:
            table.filesDropped.connect(self.import_dropped_files)
        table.dragFinished.connect(self.reload_current_directory)
        table.verticalScrollBar().valueChanged.connect(self._kick_thumb_timer)
        model.modelReset.connect(self._kick_thumb_timer)

        table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        table.customContextMenuRequested.connect(
            lambda pos, t=table: self.show_gallery_context_menu(pos, t))
        return table

    def _kick_thumb_timer(self, *_):
        if not self._thumb_timer.isActive():
            self._thumb_timer.start()

    def schedule_thumbnails(self):
        pairs = [(self.table, self.model)]
        if self.compare_folder is not None:
            pairs.append((self.table_cmp, self.model_cmp))
        for table, model in pairs:
            rows = model.rowCount()
            if rows == 0 or not table.isVisible() or table.viewport().height() < 20:
                continue
            first = max(0, table.rowAt(0))
            last = table.rowAt(table.viewport().height() - 1)
            if last < 0:
                last = rows - 1
            model.schedule_window(first, max(first, last))

    def active_table(self):
        """Tabella su cui agiscono F2 / Canc / W / S (in confronto: quella con il focus)."""
        if self.compare_folder is not None and self.table_cmp.hasFocus():
            return self.table_cmp
        return self.table

    # ------------------------------------------------------------------ Compara
    def toggle_compare(self):
        if self.compare_folder is not None:
            self.exit_compare()
            return

        current = self.settings.value("last_folder", "")
        if not current or not os.path.isdir(current):
            QMessageBox.warning(self, "Compara", "Apri prima la cartella originale da confrontare.")
            return

        current = os.path.abspath(current)
        start_dir = os.path.dirname(current) or current
        folder = QFileDialog.getExistingDirectory(
            self, "Seleziona la seconda cartella da confrontare", start_dir)
        if not folder:
            return
        folder = os.path.abspath(folder)
        if folder == current:
            QMessageBox.information(self, "Compara", "Hai selezionato la stessa cartella: scegline un'altra.")
            return
        self.start_compare(folder)

    def start_compare(self, folder):
        self.compare_folder = folder
        self.cmp_items = None
        self.btn_compare.setText("✖ Esci da Compara")
        self.btn_up.setEnabled(False)
        self.lbl_cmp_left.show()
        self.cmp_right_panel.show()
        self.lbl_stats.setText("⏳ Confronto in corso...")
        self.load_compare_folder()
        QTimer.singleShot(0, self._balance_compare_splitter)

    def exit_compare(self):
        if self.compare_folder is None:
            return
        if self.cmp_loader_thread and self.cmp_loader_thread.isRunning():
            self.cmp_loader_thread.cancel()
            self.cmp_loader_thread.wait()
        self.compare_folder = None
        self.cmp_items = None
        self.single_model = self.model
        self.model_cmp.set_files([], clear_cache=True)
        self.lbl_cmp_left.hide()
        self.cmp_right_panel.hide()
        self.btn_compare.setText("⚖ Compara")

        current = os.path.abspath(self.settings.value("last_folder", "") or "")
        self.btn_up.setEnabled(bool(current) and os.path.dirname(current) != current)
        self.update_dynamic_filter_combo()
        self.apply_filter_and_sort()
        QTimer.singleShot(0, lambda: self.update_effective_columns(force=True))

    def _balance_compare_splitter(self):
        w = self.cmp_splitter.width()
        self.cmp_splitter.setSizes([w // 2, w - w // 2])
        self.update_effective_columns(force=True)

    def load_compare_folder(self):
        if self.compare_folder is None:
            return
        if self.cmp_loader_thread and self.cmp_loader_thread.isRunning():
            self.cmp_loader_thread.cancel()
            self.cmp_loader_thread.wait()
        self.cmp_items = None
        self.cmp_loader_thread = DirectoryLoader(self.compare_folder)
        self.cmp_loader_thread.loaded.connect(self.on_compare_loaded)
        self.cmp_loader_thread.failed.connect(self.on_compare_failed)
        self.cmp_loader_thread.start()

    def on_compare_loaded(self, folder, items):
        if folder != self.compare_folder:
            return
        self.cmp_items = items
        self.model_cmp.clear_caches()
        self.update_dynamic_filter_combo()
        self.apply_filter_and_sort()
        self.sync_single_view_index()

    def on_compare_failed(self, err):
        self.exit_compare()
        self.lbl_stats.setText(f"⛔ Errore lettura cartella di confronto: {err}")

    def compute_compare_sets(self):
        """(solo nell'originale, solo nella nuova): il confronto e' per nome, nel primo livello."""
        main_names = {x.name for x in self.all_file_items}
        cmp_names = {x.name for x in self.cmp_items}
        left = [x for x in self.all_file_items if x.name not in cmp_names]
        right = [x for x in self.cmp_items if x.name not in main_names]
        return left, right

    def _compare_label(self, folder, other):
        name = os.path.basename(folder.rstrip(os.sep)) or folder
        if name == (os.path.basename(other.rstrip(os.sep)) or other):
            parent = os.path.basename(os.path.dirname(folder.rstrip(os.sep)))
            name = f"{parent}/{name}" if parent else name
        return name

    def _filter_base_items(self):
        if self.compare_folder is not None and self.cmp_items is not None:
            left, right = self.compute_compare_sets()
            return left + right
        return self.all_file_items

    def show_gallery_context_menu(self, pos, table=None):
        if table is None:
            table = self.table
        model = table.model()
        # Sul lato destro del confronto "Incolla" va nella seconda cartella
        dest = self.compare_folder if table is not self.table else None

        viewport_pos = table.viewport().mapFromGlobal(QCursor.pos())
        index = table.indexAt(viewport_pos)
        if not index.isValid():
            index = table.indexAt(pos)

        target_item = None
        if index.isValid():
            idx = index.row() * model.cols + index.column()
            if idx < len(model.files):
                target_item = model.files[idx]
                if not table.selectionModel().isSelected(index):
                    table.setCurrentIndex(index)

        menu = QMenu(self)

        clipboard = QApplication.clipboard()
        mime = clipboard.mimeData()
        can_paste = bool(mime and (mime.hasUrls() or mime.hasImage()))

        if target_item:
            if target_item.is_dir:
                action_enter = menu.addAction("📂 Apri cartella")
                action_enter.triggered.connect(lambda: self.navigate_to(target_item.path))
                menu.addSeparator()

            menu_open_with = menu.addMenu("🚀 Apri con")
            self.populate_open_with_submenu(menu_open_with, target_item.path)

            # MODIFICA: "Imposta come Sfondo Animato" diretto nel menu contestuale principale
            if not target_item.is_dir:
                self.add_wallpaper_action(menu, target_item.path)

            menu.addSeparator()

            action_copy = menu.addAction("📋 Copia")
            action_copy.triggered.connect(lambda: self.copy_file_to_clipboard(target_item.path))

            action_paste = menu.addAction("📥 Incolla")
            action_paste.setEnabled(can_paste)
            action_paste.triggered.connect(lambda ch=False, d=dest: self.paste_from_clipboard(d))
            menu.addSeparator()

            action_rename = menu.addAction("✏️ Rinomina (F2)")
            action_rename.triggered.connect(lambda: self.rename_file(target_item.path))

            action_delete = menu.addAction("🗑 Elimina (Cestino)")
            action_delete.triggered.connect(lambda: self.delete_and_reload(target_item.path))
        else:
            action_paste = menu.addAction("📥 Incolla")
            action_paste.setEnabled(can_paste)
            action_paste.triggered.connect(lambda ch=False, d=dest: self.paste_from_clipboard(d))
            menu.addSeparator()

            action_reload = menu.addAction("🔄 Ricarica Cartella")
            action_reload.triggered.connect(self.reload_current_directory)

        menu.exec(QCursor.pos())

    def rename_file(self, file_path):
        if not file_path or not os.path.exists(file_path):
            return

        abs_path = os.path.abspath(file_path)
        old_name = os.path.basename(abs_path)
        folder = os.path.dirname(abs_path)

        new_name, ok = QInputDialog.getText(
            self,
            "Rinomina File",
            "Nuovo nome file (inclusa estensione):",
            QLineEdit.EchoMode.Normal,
            old_name
        )

        if not ok or not new_name.strip() or new_name.strip() == old_name:
            return

        new_name = new_name.strip()

        if '/' in new_name or '\\' in new_name or '\0' in new_name:
            QMessageBox.warning(self, "Nome non valido", "Il nome del file non può contenere caratteri come '/' o '\\'.")
            return

        new_path = os.path.join(folder, new_name)

        if os.path.exists(new_path):
            QMessageBox.warning(self, "File già esistente", f"Un file denominato '{new_name}' esiste già nella cartella.")
            return

        try:
            os.rename(abs_path, new_path)
            self.lbl_stats.setText(f"✏️ Rinominato in: {new_name}")
            self.reload_current_directory()
        except Exception as e:
            QMessageBox.critical(self, "Errore Rinomina", f"Impossibile rinominare il file:\n{e}")

    def rename_selected_gallery_file(self):
        table = self.active_table()
        model = table.model()
        curr_idx = table.currentIndex()
        if not curr_idx.isValid():
            return
        idx = curr_idx.row() * model.cols + curr_idx.column()
        if 0 <= idx < len(model.files):
            item = model.files[idx]
            self.rename_file(item.path)

    def copy_file_to_clipboard(self, file_path):
        if not file_path or not os.path.exists(file_path):
            return
        abs_path = os.path.abspath(file_path)
        clipboard = QApplication.clipboard()
        mime = QMimeData()
        url = QUrl.fromLocalFile(abs_path)
        mime.setUrls([url])
        mime.setText(abs_path)
        clipboard.setMimeData(mime)
        self.lbl_stats.setText(f"📋 Copiato negli appunti: {os.path.basename(abs_path)}")

    def paste_from_clipboard(self, dest_folder=None):
        current_folder = dest_folder or self.settings.value("last_folder", "")
        if not current_folder or not os.path.isdir(current_folder):
            QMessageBox.warning(self, "Attenzione", "Nessuna cartella di destinazione valida aperta.")
            return

        clipboard = QApplication.clipboard()
        mime = clipboard.mimeData()
        if not mime:
            return

        pasted_count = 0
        if mime.hasUrls():
            for url in mime.urls():
                src_path = url.toLocalFile()
                if src_path and os.path.exists(src_path):
                    filename = os.path.basename(src_path)
                    dest_path = os.path.join(current_folder, filename)

                    if os.path.exists(dest_path):
                        base, ext = os.path.splitext(filename)
                        counter = 1
                        while True:
                            suffix = " - Copia" if counter == 1 else f" - Copia ({counter})"
                            candidate = os.path.join(current_folder, f"{base}{suffix}{ext}")
                            if not os.path.exists(candidate):
                                dest_path = candidate
                                break
                            counter += 1

                    try:
                        if os.path.isdir(src_path):
                            shutil.copytree(src_path, dest_path)
                        else:
                            shutil.copy2(src_path, dest_path)
                        pasted_count += 1
                    except Exception as e:
                        QMessageBox.warning(self, "Errore Incolla", f"Impossibile incollare '{filename}':\n{e}")
        elif mime.hasImage():
            img = clipboard.image()
            if not img.isNull():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"Incolla_{timestamp}.png"
                dest_path = os.path.join(current_folder, filename)
                counter = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(current_folder, f"Incolla_{timestamp}_{counter}.png")
                    counter += 1
                if img.save(dest_path, "PNG"):
                    pasted_count += 1

        if pasted_count > 0:
            self.reload_current_directory()
            self.lbl_stats.setText(f"📥 Incollati {pasted_count} elemento/i con successo.")

    def import_dropped_files(self, paths, move=False, dest_folder=None):
        current_folder = dest_folder or self.settings.value("last_folder", "")
        if not current_folder or not os.path.isdir(current_folder):
            QMessageBox.warning(self, "Attenzione", "Nessuna cartella di destinazione valida aperta.")
            return

        dest_dir = os.path.abspath(current_folder)
        done, skipped = 0, 0
        for src in paths:
            src = os.path.abspath(src)
            if not os.path.exists(src):
                continue
            # stessa cartella o cartella trascinata dentro se stessa: ignora
            if os.path.dirname(src) == dest_dir or \
               (os.path.isdir(src) and (dest_dir + os.sep).startswith(src + os.sep)):
                skipped += 1
                continue

            filename = os.path.basename(src)
            dest_path = os.path.join(dest_dir, filename)
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(filename)
                counter = 1
                while True:
                    suffix = " - Copia" if counter == 1 else f" - Copia ({counter})"
                    candidate = os.path.join(dest_dir, f"{base}{suffix}{ext}")
                    if not os.path.exists(candidate):
                        dest_path = candidate
                        break
                    counter += 1
            try:
                if move:
                    shutil.move(src, dest_path)
                elif os.path.isdir(src):
                    shutil.copytree(src, dest_path)
                else:
                    shutil.copy2(src, dest_path)
                done += 1
            except Exception as e:
                QMessageBox.warning(self, "Errore Trascinamento", f"Impossibile importare '{filename}':\n{e}")

        if done:
            self.reload_current_directory()
            verbo = "Spostati" if move else "Copiati"
            self.lbl_stats.setText(f"📥 {verbo} {done} elemento/i nella cartella.")
        elif skipped:
            self.lbl_stats.setText("ℹ️ I file trascinati sono già in questa cartella.")

    def delete_file(self, file_path):
        if not file_path or not os.path.exists(file_path):
            return False

        abs_path = os.path.abspath(file_path)
        fname = os.path.basename(abs_path)
        kind = "la cartella" if os.path.isdir(abs_path) else "il file"

        msg_box = QMessageBox(
            QMessageBox.Icon.Question,
            "Conferma Eliminazione",
            f"Sei sicuro di voler spostare nel cestino {kind}:\n\n{fname}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            self
        )
        # Imposta 'Sì' con focus predefinito: premendo INVIO si conferma all'istante
        msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)
        reply = msg_box.exec()

        if reply != QMessageBox.StandardButton.Yes:
            return False

        success = QFile.moveToTrash(abs_path)
        if not success:
            try:
                os.remove(abs_path)
                success = True
            except Exception as e:
                QMessageBox.critical(self, "Errore Eliminazione", f"Impossibile eliminare il file:\n{e}")
                return False

        return True

    def delete_and_reload(self, file_path):
        if self.delete_file(file_path):
            self.reload_current_directory()

    def delete_selected_gallery_file(self):
        table = self.active_table()
        model = table.model()
        curr_idx = table.currentIndex()
        if not curr_idx.isValid():
            return
        idx = curr_idx.row() * model.cols + curr_idx.column()
        if 0 <= idx < len(model.files):
            item = model.files[idx]
            self.delete_and_reload(item.path)

    def delete_current_single_file(self):
        if not (0 <= self.current_single_idx < len(self.single_model.files)):
            return

        target_file = self.single_model.files[self.current_single_idx].path
        self.video_player.stop()

        if self.delete_file(target_file):
            target_idx = self.current_single_idx
            self.reload_current_directory()

            if not self.single_model.files:
                self.show_gallery()
            else:
                next_idx = self.get_adjacent_viewable_index(1)
                if next_idx == -1:
                    next_idx = self.get_adjacent_viewable_index(-1)
                
                if next_idx != -1:
                    self.current_single_idx = next_idx
                    self.load_single_media(next_idx)
                else:
                    self.show_gallery()
        else:
            if 0 <= self.current_single_idx < len(self.single_model.files):
                item = self.single_model.files[self.current_single_idx]
                if item.is_video:
                    self.video_player.load_video(item.path)

    def on_single_drag_finished(self):
        """Il file mostrato e' stato trascinato (e spostato) fuori dalla cartella."""
        if not (0 <= self.current_single_idx < len(self.single_model.files)):
            self.reload_current_directory()
            return
        moved_path = self.single_model.files[self.current_single_idx].path
        if os.path.exists(moved_path):
            return

        self.video_player.stop()
        next_idx = self.get_adjacent_viewable_index(1)
        if next_idx == -1:
            next_idx = self.get_adjacent_viewable_index(-1)
        self.reload_current_directory()
        if next_idx != -1:
            self.current_single_idx = next_idx
            self.load_single_media(next_idx)
        else:
            self.show_gallery()

    def reload_current_directory(self):
        current_folder = self.settings.value("last_folder", "")
        if current_folder and os.path.isdir(current_folder):
            self.load_directory(current_folder)
        if self.compare_folder is not None:
            self.load_compare_folder()

    def init_single_view_ui(self):
        single_widget = QWidget()
        layout = QVBoxLayout(single_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.single_toolbar_widget = QWidget()
        single_toolbar = ShrinkRowLayout(self.single_toolbar_widget)
        single_toolbar.setContentsMargins(10, 8, 10, 8)

        self.btn_back = QPushButton("⬅ Torna alla Galleria (Esc / Tasto Destro)")
        self.btn_back.setStyleSheet("font-weight: bold; padding: 6px 14px;")
        self.btn_back.clicked.connect(self.show_gallery)
        single_toolbar.addWidget(self.btn_back)

        self.btn_fit = QPushButton("📐 Adatta Schermo")
        self.btn_fit.clicked.connect(lambda: self.image_viewer.fit_to_window(force=True))
        single_toolbar.addWidget(self.btn_fit)

        self.btn_open_with = QPushButton("🚀 Apri con ▾")
        self.menu_open_with = QMenu(self)
        self.menu_open_with.aboutToShow.connect(self.populate_open_with_menu)
        self.btn_open_with.setMenu(self.menu_open_with)
        single_toolbar.addWidget(self.btn_open_with)

        # MODIFICA: pulsante dedicato "Imposta come Sfondo Animato" (visibile solo per video/GIF)
        self.btn_wallpaper = QPushButton("🎬 Sfondo Animato")
        self.btn_wallpaper.setStyleSheet("font-weight: bold; padding: 6px 12px;")
        self.btn_wallpaper.clicked.connect(
            lambda: self.launch_app(self.find_local_wallpaper_script(), self.get_active_single_path()))
        self.btn_wallpaper.setVisible(False)
        single_toolbar.addWidget(self.btn_wallpaper)

        self.lbl_single_info = QLabel("")
        self.lbl_single_info.setStyleSheet("color: #aaa; margin-left: 15px; font-weight: bold;")
        single_toolbar.addWidget(self.lbl_single_info)

        single_toolbar.addStretch()

        self.btn_fs_single = QPushButton("⛶ Fullscreen (F)")
        self.btn_fs_single.clicked.connect(self.toggle_fullscreen)
        single_toolbar.addWidget(self.btn_fs_single)

        self.btn_orig_size = QPushButton("🔍 Dimensioni Originali (100%)")
        self.btn_orig_size.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #1e7e34;
                padding: 6px 16px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #218838; }
        """)
        self.btn_orig_size.clicked.connect(lambda: self.image_viewer.set_original_size())
        single_toolbar.addWidget(self.btn_orig_size)
        make_toolbar_shrinkable(single_toolbar)

        layout.addWidget(self.single_toolbar_widget)

        self.single_media_stack = QStackedWidget()
        self.image_viewer = SingleImageViewerWidget(self)
        self.video_player = VideoPlayerWidget(self)

        self.single_media_stack.addWidget(self.image_viewer)
        self.single_media_stack.addWidget(self.video_player)

        # Riquadro info (tasto I) sovrapposto al contenuto
        self.info_visible = False
        self.info_overlay = QFrame(self.single_media_stack)
        self.info_overlay.setObjectName("InfoOverlay")
        self.info_overlay.setStyleSheet("""
            #InfoOverlay { background-color: rgba(10, 10, 10, 215); border: 1px solid #00adb5; border-radius: 6px; }
            #InfoOverlay QLabel { color: #00adb5; font-weight: bold; background: transparent; }
            #InfoOverlay QPlainTextEdit {
                background: transparent; color: #e6e6e6; border: none;
                font-family: monospace; font-size: 12px;
                selection-background-color: #00adb5; selection-color: #000;
            }
        """)
        info_lay = QVBoxLayout(self.info_overlay)
        info_lay.setContentsMargins(10, 8, 10, 8)
        info_lay.setSpacing(6)
        self.info_header = QLabel()
        self.info_header.setWordWrap(True)
        self.info_header.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info_lay.addWidget(self.info_header)
        self.info_text = QPlainTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.info_text.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        info_lay.addWidget(self.info_text, 1)
        self.info_overlay.hide()

        for w in (self.image_viewer, self.video_player):
            w.filesDropped.connect(self.import_dropped_files)
            w.dragFinished.connect(self.on_single_drag_finished)
            # Doppio click: entra/esci dal fullscreen (differito a fine evento mouse)
            w.fullscreenRequested.connect(lambda: QTimer.singleShot(0, self.toggle_fullscreen))

        layout.addWidget(self.single_media_stack)
        self.stack.addWidget(single_widget)

    def populate_open_with_menu(self):
        curr_path = None
        if 0 <= self.current_single_idx < len(self.single_model.files):
            curr_path = self.single_model.files[self.current_single_idx].path
        self.populate_open_with_submenu(self.menu_open_with, curr_path)

    def find_local_wallpaper_script(self):
        """Restituisce il percorso di wallpaper.py se presente accanto a View.py o in alcune cartelle note."""
        candidates = []
        try:
            candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wallpaper.py"))
        except Exception:
            pass
        # Cartelle tipiche di installazione (es. dischi secondari /mnt/...)
        candidates += [
            os.path.expanduser("~/Apps/wallpaper.py"),
            "/mnt/1CE16DBE48849731/Apps/wallpaper.py",
        ]
        # Cerca anche nelle directory presenti nel PATH (come farebbe una shell)
        for d in os.environ.get("PATH", "").split(os.pathsep):
            if d:
                candidates.append(os.path.join(d, "wallpaper.py"))
        for c in candidates:
            try:
                if c and os.path.isfile(c):
                    return c
            except OSError:
                continue
        return None

    def is_wallpaper_target(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        return ext in KNOWN_VIDEO_EXTENSIONS or ext == '.gif'

    def get_active_single_path(self):
        """Percorso del file attualmente visualizzato nella vista singola (o None)."""
        if 0 <= self.current_single_idx < len(self.single_model.files):
            return self.single_model.files[self.current_single_idx].path
        return None

    def add_wallpaper_action(self, menu, target_file_path):
        """Aggiunge la voce '🎬 Imposta come Sfondo Animato' al menu se applicabile.
        Ritorna True se l'azione è stata aggiunta."""
        local_wallpaper = self.find_local_wallpaper_script()
        if local_wallpaper and target_file_path and self.is_wallpaper_target(target_file_path):
            action_wp = menu.addAction("🎬 Imposta come Sfondo Animato")
            action_wp.triggered.connect(
                lambda ch=False, w=local_wallpaper, p=target_file_path: self.launch_app(w, p))
            return True
        return False

    def populate_open_with_submenu(self, menu, target_file_path):
        menu.clear()

        # MODIFICA: Integrazione diretta con wallpaper.py
        if self.add_wallpaper_action(menu, target_file_path):
            menu.addSeparator()

        saved_apps = self.settings.value("custom_apps_dict", {})
        if not isinstance(saved_apps, dict):
            old_apps = self.settings.value("custom_apps", [])
            saved_apps = {}
            if isinstance(old_apps, list):
                for path in old_apps:
                    name = os.path.basename(path).capitalize()
                    saved_apps[name] = path

        for app_name, app_cmd in saved_apps.items():
            action = menu.addAction(f"🎨 Apri con {app_name}")
            action.triggered.connect(lambda ch, cmd=app_cmd, p=target_file_path: self.launch_app(cmd, p))

        action_sys = menu.addAction("🌐 App predefinita di sistema (xdg-open)")
        action_sys.triggered.connect(lambda ch=False, p=target_file_path: self.launch_system_default(p))

        menu.addSeparator()

        # Selezione al volo di qualsiasi script Python (.py)
        action_py = menu.addAction("🐍 Esegui Script Python al volo...")
        action_py.triggered.connect(lambda ch=False, p=target_file_path: self.launch_python_script_direct(p))

        action_add = menu.addAction("➕ Configura / Aggiungi nuova app...")
        action_add.triggered.connect(lambda ch=False, p=target_file_path: self.configure_new_app(p))

        if saved_apps:
            action_clear = menu.addAction("🗑 Cancella lista app salvate")
            action_clear.triggered.connect(self.clear_configured_apps)

    def launch_python_script_direct(self, target_file_path=None):
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Seleziona Script Python",
            os.path.expanduser("~"), "Script Python (*.py);;Tutti i file (*)"
        )
        if fpath:
            self.launch_app(fpath, target_file_path)

    def open_in_file_manager(self, file_path, app_cmd=""):
        abs_path = os.path.abspath(file_path)
        cmd_lower = app_cmd.lower()

        if "flatpak" in cmd_lower and "dolphin" in cmd_lower:
            parts = [p for p in shlex.split(app_cmd) if p.strip() and not p.startswith('%')]
            cmd = parts + ['--select', '@@', abs_path, '@@']
        elif "flatpak" in cmd_lower:
            parts = [p for p in shlex.split(app_cmd) if p.strip() and not p.startswith('%')]
            cmd = parts + ['@@', abs_path, '@@']
        elif "dolphin" in cmd_lower:
            cmd = ['dolphin', '--select', abs_path]
        elif "nautilus" in cmd_lower:
            cmd = ['nautilus', '--select', abs_path]
        elif "nemo" in cmd_lower:
            cmd = ['nemo', '--no-desktop', abs_path]
        elif "thunar" in cmd_lower:
            cmd = ['thunar', os.path.dirname(abs_path)]
        else:
            cmd = build_launch_command(app_cmd, abs_path)

        try:
            subprocess.Popen(cmd)
            self.lbl_stats.setText(f"Aperto con: {os.path.basename(cmd[0])}")
        except Exception as e:
            QMessageBox.warning(self, "Errore Avvio", f"Errore nell'avvio del file manager:\n{e}")

    def launch_app(self, app_cmd, target_file_path=None):
        if target_file_path:
            file_path = target_file_path
        elif 0 <= self.current_single_idx < len(self.single_model.files):
            file_path = self.single_model.files[self.current_single_idx].path
        else:
            return

        abs_file_path = os.path.abspath(file_path)

        cmd_lower = app_cmd.lower()
        if any(fm in cmd_lower for fm in ['dolphin', 'nautilus', 'nemo', 'thunar', 'pcmanfm', 'caja']):
            self.open_in_file_manager(abs_file_path, app_cmd)
            return

        cmd = build_launch_command(app_cmd, abs_file_path)

        try:
            # start_new_session=True rende lo sfondo indipendente dal ciclo di vita di View.py
            work_dir = os.path.dirname(abs_file_path)
            proc = subprocess.Popen(cmd, cwd=work_dir, start_new_session=True,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

            # Se il processo muore entro ~1.5s (es. errore di sintassi nel parser di
            # wallpaper.py, modulo mancante, xprop/mpv non installati), mostra l'errore
            # invece di far credere che lo sfondo animato sia partito.
            def _check_early_exit(p=proc, c=cmd):
                try:
                    code = p.poll()
                    if code is not None and code != 0:
                        err = ""
                        try:
                            err = p.stderr.read().decode(errors="replace").strip()[-600:]
                        except Exception:
                            pass
                        QMessageBox.warning(
                            self, "Errore Avvio",
                            f"Il processo è terminato subito (codice {code}):\n"
                            f"{' '.join(c)}\n\n{err}"
                        )
                except Exception:
                    pass
            QTimer.singleShot(1500, _check_early_exit)

            # L'elemento [0] è un interprete Python se è python* oppure se è uno script .py
            # lanciato con shebang personalizzato (es. '#!/x/venv/bin/python3.11').
            is_interpreter = (os.path.basename(cmd[0]).startswith('python')
                              or cmd[0].endswith('.py'))
            prog = os.path.basename(cmd[1]) if len(cmd) > 1 and is_interpreter else os.path.basename(cmd[0])
            self.lbl_stats.setText(f"Avviato con: {prog}")
        except Exception as e:
            QMessageBox.warning(self, "Errore Avvio", f"Errore nell'avvio dell'applicazione:\n{e}")

    def launch_system_default(self, target_file_path=None):
        if target_file_path:
            file_path = target_file_path
        elif 0 <= self.current_single_idx < len(self.single_model.files):
            file_path = self.single_model.files[self.current_single_idx].path
        else:
            return

        abs_path = os.path.abspath(file_path)
        try:
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(abs_path)):
                subprocess.Popen(['xdg-open', abs_path])
            self.lbl_stats.setText(f"Aperto con app di sistema: {os.path.basename(abs_path)}")
        except Exception as e:
            QMessageBox.warning(self, "Errore Apertura", f"Impossibile aprire il file con l'app di sistema:\n{e}")

    def configure_new_app(self, target_file_path=None):
        dlg = AppChooserDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            app_name = dlg.selected_app_name
            app_cmd = dlg.selected_app_cmd

            if app_name and app_cmd:
                saved_apps = self.settings.value("custom_apps_dict", {})
                if not isinstance(saved_apps, dict):
                    saved_apps = {}
                saved_apps[app_name] = app_cmd
                self.settings.setValue("custom_apps_dict", saved_apps)
                self.launch_app(app_cmd, target_file_path)

    def clear_configured_apps(self):
        self.settings.setValue("custom_apps_dict", {})
        self.settings.setValue("custom_apps", [])

    def toggle_fullscreen(self):
        if self.isFullScreen():
            # Ritorno da Fullscreen a finestra
            if self._was_maximized_before_fs:
                self.showMaximized()
            else:
                self.showNormal()
            
            # Ripristina le barre
            if not self.ui_hidden:
                self.gallery_toolbar_widget.show()
                self.single_toolbar_widget.show()
                self.video_player.set_controls_visible(True)
        else:
            # Ingresso in Fullscreen
            self._was_maximized_before_fs = self.isMaximized()
            # Nasconde le toolbar per consentire al viewer di usare il 100% dell'altezza dello schermo
            self.gallery_toolbar_widget.hide()
            self.single_toolbar_widget.hide()
            self.video_player.set_controls_visible(False)
            self.showFullScreen()

        # Ricalcola immediatamente le dimensioni con forza di reset
        if self.stack.currentIndex() == 1:
            self.image_viewer.is_fit_mode = True
            self.image_viewer.fit_to_window(force=True)
            QTimer.singleShot(60, lambda: self.image_viewer.fit_to_window(force=True))
            QTimer.singleShot(200, lambda: self.image_viewer.fit_to_window(force=True))

    def on_cell_clicked(self, index: QModelIndex, table=None):
        if not index.isValid():
            return
        if QApplication.keyboardModifiers() & (Qt.KeyboardModifier.ControlModifier
                                               | Qt.KeyboardModifier.ShiftModifier):
            return  # selezione multipla: non aprire il file
        if table is None:
            table = self.table
        model = table.model()
        idx = index.row() * model.cols + index.column()
        if idx < len(model.files):
            self.txt_search.clearFocus()
            item = model.files[idx]

            if item.is_dir:
                self.navigate_to(item.path)
                return

            if not item.is_natively_viewable:
                self.launch_system_default(item.path)
                return

            self.single_model = model
            self.current_single_idx = idx
            self.load_single_media(idx)
            self.stack.setCurrentIndex(1)

    def load_single_media(self, idx):
        if not (0 <= idx < len(self.single_model.files)):
            return

        item = self.single_model.files[idx]
        tipo = "Video" if item.is_video else ("Immagine" if item.is_image else "File generico")
        self.lbl_single_info.setText(
            f"[{tipo}] {item.name}  —  [{idx + 1} / {len(self.single_model.files)}]  —  {format_size(item.size)}"
        )

        # Mostra il pulsante "Sfondo Animato" solo per video/GIF se wallpaper.py è disponibile
        if hasattr(self, 'btn_wallpaper'):
            self.btn_wallpaper.setVisible(bool(self.find_local_wallpaper_script())
                                          and self.is_wallpaper_target(item.path))

        if item.is_video:
            self.btn_orig_size.hide()
            self.btn_fit.hide()
            self.single_media_stack.setCurrentIndex(1)
            self.video_player.load_video(item.path)
        elif item.is_natively_viewable:
            self.video_player.stop()
            self.btn_orig_size.show()
            self.btn_fit.show()
            self.single_media_stack.setCurrentIndex(0)
            self.image_viewer.set_image(item.path)
        else:
            self.show_gallery()
            self.launch_system_default(item.path)

        if self.info_visible:
            self.refresh_info_overlay()
            if item.is_video:
                # le dimensioni del video si conoscono solo quando arriva il primo fotogramma
                QTimer.singleShot(400, self.refresh_info_overlay)

    def current_single_path(self):
        if self.single_media_stack.currentIndex() == 1:
            return self.video_player.current_filepath
        return self.image_viewer.current_filepath

    def toggle_info_overlay(self):
        self.info_visible = not self.info_visible
        if self.info_visible:
            self.refresh_info_overlay()
        else:
            self.info_overlay.hide()

    def position_info_overlay(self):
        st = self.single_media_stack
        m = 12
        w = min(max(260, min(760, int(st.width() * 0.5))), max(100, st.width() - 2 * m))
        h = min(max(160, int(st.height() * 0.6)), max(80, st.height() - 2 * m))
        self.info_overlay.setGeometry(m, m, w, h)

    def refresh_info_overlay(self):
        if not self.info_visible:
            return
        path = self.current_single_path()
        if not path or not os.path.isfile(path):
            self.info_overlay.hide()
            return

        dims = "n/d"
        if self.single_media_stack.currentIndex() == 1:
            frame = self.video_player.canvas.current_frame if MULTIMEDIA_AVAILABLE else None
            if frame is not None and not frame.isNull():
                dims = f"{frame.width()} × {frame.height()} px"
        elif self.image_viewer.pixmap_item is not None:
            pm = self.image_viewer.pixmap_item.pixmap()
            if not pm.isNull():
                dims = f"{pm.width()} × {pm.height()} px"

        try:
            size_txt = format_size(os.path.getsize(path))
        except OSError:
            size_txt = "?"
        text = read_file_head_text(path)
        self.info_header.setText(
            f"{os.path.basename(path)}\n{dims}  —  {size_txt}\n"
            f"Primi {len(text):,} caratteri del file (max {INFO_TEXT_CHARS:,}):")
        self.info_text.setPlainText(text)
        self.position_info_overlay()
        self.info_overlay.show()
        self.info_overlay.raise_()

    def show_gallery(self):
        self.info_visible = False
        self.info_overlay.hide()
        self.video_player.stop()
        self.stack.setCurrentIndex(0)

    def update_reverse_btn_label(self):
        self.btn_reverse.setText("⬇ Discendente" if self.is_reverse else "⬆ Ascendente")

    def toggle_sort_direction(self):
        self.is_reverse = not self.is_reverse
        self.update_reverse_btn_label()
        self.settings.setValue("sort_reverse", self.is_reverse)
        self.apply_filter_and_sort()

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleziona Cartella")
        if folder:
            self.exit_compare()
            abs_folder = os.path.abspath(folder)
            self.settings.setValue("last_folder", abs_folder)
            self.load_directory(abs_folder)

    def navigate_to(self, folder):
        abs_folder = os.path.abspath(folder)
        if self.compare_folder is not None:
            # Durante il confronto le due cartelle restano fisse: la sottocartella si apre nel file manager
            self.launch_system_default(abs_folder)
            return
        if not os.path.isdir(abs_folder) or not os.access(abs_folder, os.R_OK | os.X_OK):
            self.lbl_stats.setText(f"⛔ Impossibile aprire la cartella: {abs_folder}")
            return
        self.settings.setValue("last_folder", abs_folder)
        # La ricerca riguardava la cartella precedente: azzerala senza ricalcolare
        self.txt_search.blockSignals(True)
        self.txt_search.clear()
        self.txt_search.blockSignals(False)
        self.txt_search.clearFocus()
        self.load_directory(abs_folder)

    def go_to_parent_folder(self):
        if self.compare_folder is not None:
            return
        current = os.path.abspath(self.settings.value("last_folder", "") or "")
        if not current:
            return
        parent = os.path.dirname(current)
        if parent and parent != current:
            self.navigate_to(parent)

    def load_directory(self, folder):
        abs_folder = os.path.abspath(folder)
        self.lbl_stats.setText("⏳ Caricamento cartella in corso...")

        if self.dir_loader_thread and self.dir_loader_thread.isRunning():
            self.dir_loader_thread.cancel()
            self.dir_loader_thread.wait()

        self.dir_loader_thread = DirectoryLoader(abs_folder)
        self.dir_loader_thread.loaded.connect(self.on_directory_loaded)
        self.dir_loader_thread.failed.connect(lambda err: self.lbl_stats.setText(f"Errore lettura: {err}"))
        self.dir_loader_thread.start()

    def on_directory_loaded(self, folder, items):
        folder_changed = getattr(self, "_loaded_folder", None) != folder
        self._loaded_folder = folder

        self.all_file_items = items
        self.model.clear_caches()
        self.update_dynamic_filter_combo()
        self.apply_filter_and_sort()
        self.btn_up.setEnabled(os.path.dirname(folder) != folder and self.compare_folder is None)

        if folder_changed:
            self.table.clearSelection()
            self.table.scrollToTop()

        self.sync_single_view_index()

    def sync_single_view_index(self):
        showing_video = self.single_media_stack.currentIndex() == 1
        shown_path = self.video_player.current_filepath if showing_video else self.image_viewer.current_filepath
        if self.stack.currentIndex() == 1 and shown_path:
            curr_path = os.path.abspath(shown_path)
            for i, itm in enumerate(self.single_model.files):
                if os.path.abspath(itm.path) == curr_path:
                    self.current_single_idx = i
                    tipo = "Video" if itm.is_video else ("Immagine" if itm.is_image else "File generico")
                    self.lbl_single_info.setText(
                        f"[{tipo}] {itm.name}  —  [{i + 1} / {len(self.single_model.files)}]  —  {format_size(itm.size)}"
                    )
                    break

    def update_dynamic_filter_combo(self):
        self.combo_filter.blockSignals(True)
        self.combo_filter.clear()

        base_items = self._filter_base_items()
        total_files = len(base_items)
        self.combo_filter.addItem(f"Tutti gli elementi ({total_files:,})", "ALL")

        dir_count = sum(1 for x in base_items if x.is_dir)
        img_count = sum(1 for x in base_items if x.is_image)
        vid_count = sum(1 for x in base_items if x.is_video)
        other_count = sum(1 for x in base_items if not x.is_natively_viewable and not x.is_dir)

        if dir_count > 0:
            self.combo_filter.addItem(f"📁 Cartelle ({dir_count:,})", "MACRO_DIR")
        if img_count > 0:
            self.combo_filter.addItem(f"📷 Tutte le Immagini ({img_count:,})", "MACRO_IMG")
        if vid_count > 0:
            self.combo_filter.addItem(f"🎬 Tutti i Video ({vid_count:,})", "MACRO_VID")
        if other_count > 0:
            self.combo_filter.addItem(f"📄 Altri File di Sistema ({other_count:,})", "MACRO_OTHER")

        counts = Counter(x.ext for x in base_items if not x.is_dir)
        if counts:
            self.combo_filter.insertSeparator(self.combo_filter.count())
            for ext, count in sorted(counts.items(), key=lambda t: t[0]):
                icon = "🎬 " if ext in KNOWN_VIDEO_EXTENSIONS else ("📷 " if ext in KNOWN_IMAGE_EXTENSIONS else "📄 ")
                label = f"{icon}{ext.upper()} ({count:,})"
                self.combo_filter.addItem(label, ext)

        self.combo_filter.setCurrentIndex(0)
        self.combo_filter.blockSignals(False)

    @staticmethod
    def _filter_items(items, filter_mode, query):
        if filter_mode == "MACRO_DIR":
            filtered = [x for x in items if x.is_dir]
        elif filter_mode == "MACRO_IMG":
            filtered = [x for x in items if x.is_image]
        elif filter_mode == "MACRO_VID":
            filtered = [x for x in items if x.is_video]
        elif filter_mode == "MACRO_OTHER":
            filtered = [x for x in items if not x.is_natively_viewable and not x.is_dir]
        elif filter_mode == "ALL" or filter_mode is None:
            filtered = list(items)
        else:
            filtered = [x for x in items if x.ext == filter_mode]

        if query:
            filtered = [x for x in filtered if query in x.name.casefold()]
        return filtered

    def apply_filter_and_sort(self):
        if self.compare_folder is not None:
            self.apply_compare_filter_and_sort()
            return

        if not self.all_file_items:
            self.lbl_stats.setText("Nessun file trovato.")
            self.model.set_files([])
            return

        filter_mode = self.combo_filter.currentData()
        sort_criterion = self.combo_sort.currentData()
        query = self.txt_search.text().strip().casefold()

        self.settings.setValue("filter_type", filter_mode)
        self.settings.setValue("sort_criterion", sort_criterion)

        filtered = self._filter_items(self.all_file_items, filter_mode, query)

        self.model.set_files(filtered, criterion=sort_criterion, reverse=self.is_reverse, clear_cache=False)

        folder_name = os.path.basename(self.settings.value("last_folder", ""))
        self.lbl_stats.setText(
            f"Mostrati: {len(filtered):,} di {len(self.all_file_items):,} elementi in '{folder_name}'"
        )

    def apply_compare_filter_and_sort(self):
        if self.cmp_items is None:
            return   # seconda cartella ancora in lettura: si aggiorna appena e' pronta

        filter_mode = self.combo_filter.currentData()
        sort_criterion = self.combo_sort.currentData()
        query = self.txt_search.text().strip().casefold()
        self.settings.setValue("filter_type", filter_mode)
        self.settings.setValue("sort_criterion", sort_criterion)

        left_all, right_all = self.compute_compare_sets()
        left = self._filter_items(left_all, filter_mode, query)
        right = self._filter_items(right_all, filter_mode, query)
        self.model.set_files(left, criterion=sort_criterion, reverse=self.is_reverse, clear_cache=False)
        self.model_cmp.set_files(right, criterion=sort_criterion, reverse=self.is_reverse, clear_cache=False)

        main_folder = os.path.abspath(self.settings.value("last_folder", "") or "")
        left_name = self._compare_label(main_folder, self.compare_folder)
        right_name = self._compare_label(self.compare_folder, main_folder)

        def counter(shown, total):
            return f"{shown:,}" if shown == total else f"{shown:,} di {total:,}"

        self.lbl_cmp_left.setText(f"◀ Solo in «{left_name}» — {counter(len(left), len(left_all))}")
        self.lbl_cmp_left.setToolTip(main_folder)
        self.lbl_cmp_right.setText(f"Solo in «{right_name}» ▶ — {counter(len(right), len(right_all))}")
        self.lbl_cmp_right.setToolTip(self.compare_folder)

        if not left_all and not right_all:
            self.lbl_stats.setText("✅ Le due cartelle contengono gli stessi elementi (confronto per nome)")
        else:
            self.lbl_stats.setText(
                f"Solo in '{left_name}': {len(left_all):,} — Solo in '{right_name}': {len(right_all):,}")

    def update_grid_layout(self):
        cols = self.spin_cols.value()
        row_h = self.spin_row_h.value()

        self.settings.setValue("columns", cols)
        self.settings.setValue("row_height", row_h)

        self.model.cell_size = QSize(row_h, row_h)
        self.model_cmp.cell_size = QSize(row_h, row_h)
        self.table.verticalHeader().setDefaultSectionSize(row_h)
        self.table_cmp.verticalHeader().setDefaultSectionSize(row_h)
        self.update_effective_columns(force=True)

    def on_gallery_zoom(self, steps):
        """Ctrl + rotella: cambia l'altezza delle miniature (come la casella 'Altezza (px)')."""
        cur = self.spin_row_h.value()
        step_px = max(8, round(cur * 0.1))
        new = max(self.spin_row_h.minimum(), min(self.spin_row_h.maximum(), cur + steps * step_px))
        if new == cur:
            return
        self.spin_row_h.blockSignals(True)
        self.spin_row_h.setValue(new)
        self.spin_row_h.blockSignals(False)
        self._zoom_timer.start()

    def update_effective_columns(self, force=False):
        self._update_columns_for(self.table, self.model, force)
        if self.compare_folder is not None:
            self._update_columns_for(self.table_cmp, self.model_cmp, force)

    def _update_columns_for(self, table, model, force=False):
        """Le miniature mantengono la dimensione 'Altezza (px)': restringendo la
        finestra diminuisce il numero di colonne mostrate. 'Colonne' e' il massimo."""
        max_cols = max(1, self.spin_cols.value())
        row_h = max(1, self.spin_row_h.value())
        sb_w = table.verticalScrollBar().sizeHint().width()
        avail = table.width() - 2 * table.frameWidth() - sb_w
        fit = max_cols if avail <= 0 else max(1, avail // row_h)
        cols = max(1, min(max_cols, fit))

        if not force and cols == model.cols:
            return

        # Salva posizione di scorrimento e selezione, perche' il reset del modello le azzera
        old_cols = model.cols
        top_ix = table.indexAt(QPoint(0, 0))
        top_idx = top_ix.row() * old_cols + top_ix.column() if top_ix.isValid() else -1
        sel_paths = table.selected_paths()

        model.set_columns(cols)

        sel_model = table.selectionModel()
        for path in sel_paths:
            i = model.path_map.get(path)
            if i is not None:
                sel_model.select(model.index(i // cols, i % cols),
                                 QItemSelectionModel.SelectionFlag.Select)

        if top_idx >= 0:
            row = top_idx // cols
            QTimer.singleShot(0, lambda: table.scrollTo(
                model.index(row, 0), QAbstractItemView.ScrollHint.PositionAtTop))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    target_arg = parse_input_target()
    
    viewer = FastGalleryViewer(target_path=target_arg)
    sys.exit(app.exec())