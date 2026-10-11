#!/usr/bin/env python3
"""
Trascrivi Audio - GUI PyQt5 per estrarre il testo da un audio/video con faster-whisper,
convertirlo in Hiragana e riascoltarlo con evidenziazione "karaoke".

Versione basata su faster-whisper.

Requisiti:
- faster-whisper
- PyQt5 (con QtMultimedia)
- pykakasi
- numpy
- sentencepiece (traduzione locale NLLB; ctranslate2 e huggingface_hub arrivano con faster-whisper)
- Ollama (opzionale, per la traduzione con TranslateGemma): https://ollama.com
- ffmpeg nel PATH

Installazione:
python -m pip install faster-whisper PyQt5 pykakasi numpy sentencepiece
"""

import os
import re
import sys
import json
import collections
import csv
from html import escape
import wave
import subprocess
import tempfile
import atexit
import textwrap
import shlex
import socket
import time
import shutil
import threading
from bisect import bisect_left, bisect_right
from math import ceil


def _aggiungi_path_librerie_cuda():
    """Rende visibili a ctranslate2/faster-whisper le librerie CUDA installate via pip
    (nvidia-cublas-cu12, nvidia-cudnn-cu12, ...) dentro il venv corrente.

    Le librerie stanno in <site-packages>/nvidia/<nome>/lib (Linux) o .../bin (Windows).
    Windows: os.add_dll_directory. Linux: cambiare LD_LIBRARY_PATH a processo avviato NON
    basta (il loader la legge una volta sola all'avvio), quindi le librerie principali
    vengono pre-caricate con ctypes (RTLD_GLOBAL): quando ctranslate2 chiede
    'libcublas.so.12' il loader la trova gia' in memoria. LD_LIBRARY_PATH viene comunque
    aggiornata per eventuali processi figli."""
    try:
        from importlib.metadata import distributions
    except ImportError:
        return
    is_win = os.name == "nt"
    rx = re.compile(r"\.dll$" if is_win else r"\.so(\.\d+)*$")
    siti = {}
    for dist in distributions():
        nome = (dist.metadata.get("Name") or "").lower().replace("_", "-")
        if not nome.startswith(("nvidia-", "torch", "triton")):
            continue
        for f in (dist.files or []):
            if rx.search(f.name):
                try:
                    d = str(dist.locate_file(f.parent))
                except Exception:
                    continue
                siti.setdefault(d, []).append(str(dist.locate_file(f)))
    if not siti:
        return
    if is_win:
        for sito in siti:
            try:
                os.add_dll_directory(sito)
            except (AttributeError, OSError):
                os.environ["PATH"] = sito + os.pathsep + os.environ.get("PATH", "")
        return
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    nuove = [s for s in siti if s not in ld.split(os.pathsep)]
    if nuove:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(nuove) + (os.pathsep + ld if ld else "")
    try:
        import ctypes
    except ImportError:
        return
    # ordine: prima le dipendenze (runtime, cublasLt), poi cublas e cudnn
    ordine = ("libcudart", "libnvJitLink", "libcublasLt", "libcublas", "libcudnn.")
    tutte = [f for lst in siti.values() for f in lst]
    for pref in ordine:
        for f in sorted(tutte):
            n = os.path.basename(f)
            if n.startswith(pref) and os.path.exists(f):
                try:
                    ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


_aggiungi_path_librerie_cuda()

import numpy as np

from PyQt5.QtCore import (Qt, QThread, QTimer, QUrl, QPointF, QRectF, QSizeF, QProcess, QEventLoop,
                          pyqtSignal)
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QKeySequence, QLinearGradient, QPainter, QPainterPath, QPalette, QPen,
    QPolygonF, QTextCursor, QTextDocument, QTextOption, QImage, QPixmap,
)
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QLineEdit, QPushButton, QComboBox, QCheckBox, QDoubleSpinBox, QTextEdit,
    QFileDialog, QProgressBar, QMessageBox, QFrame, QSlider, QSplitter,
    QScrollArea, QShortcut, QStyle, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QListWidget, QListWidgetItem, QAbstractItemView,
)


MODELLI = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v2",
    "large-v3",
]

LINGUE = [
    ("Giapponese", "ja"),
    ("Italiano", "it"),
    ("Inglese", "en"),
    ("Cinese", "zh"),
    ("Coreano", "ko"),
    ("Francese", "fr"),
    ("Tedesco", "de"),
    ("Spagnolo", "es"),
    ("Rilevamento automatico", None),
]

EST_AUDIO = "*.wav *.mp3 *.m4a *.flac *.ogg *.opus *.aac *.wma *.aiff *.aif *.amr *.mka"
EST_VIDEO = ("*.mp4 *.mkv *.webm *.mov *.avi *.flv *.wmv *.m4v *.mpg *.mpeg "
              "*.ts *.m2ts *.3gp *.ogv *.mts")
FILTRO_FILE = (
    "Audio e video ({a} {v});;Audio ({a});;Video ({v});;Tutti i file (*)"
).format(a=EST_AUDIO, v=EST_VIDEO)

PREFISSO_COSYVOICE = "You are a helpful assistant.<|endofprompt|>"

# impostazioni salvate da una sessione all'altra (accanto allo script)
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hira_config.json")

# Separazione voce/musica: motori disponibili (etichetta, chiave). La chiave e' salvata nella config.
SEPARATORI = (
    ("Demucs (htdemucs)", "demucs"),
    ("BS-RoFormer (audio-separator)", "bsroformer"),
)
# Checkpoint BS-RoFormer (voce) scaricato da audio-separator alla prima esecuzione e tenuto in cache locale.
# Per provarne un altro basta cambiare il nome (python -m audio_separator.utils.cli --list_models).
BSROFORMER_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
SEPARATOR_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "separator_models")
# una riga per ogni traduzione AI: righe scartate e rifatte, per confrontare i modelli
SCARTI_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hira_scarti_traduzione.csv")

# Selezione dei modelli llama.cpp con il selettore di file del sistema (GTK su Linux Mint, tramite zenity).
# Lo script NON cerca i modelli da solo: usa il percorso scritto nel campo (file .gguf o cartella).
# Queste cartelle servono solo come punto di partenza del selettore, se il campo e' vuoto.
USA_DIALOG_SISTEMA = True
CARTELLE_GGUF = ("~/.lmstudio/models", "~/.cache/lm-studio/models", "~/.cache/llama.cpp",
                 "~/llama.cpp/models", "~/models", "~/Models", "~/.cache/huggingface/hub",
                 "~/Scaricati", "~/Downloads")
CARTELLE_LLAMA_EXE = ("~/llama.cpp/build/bin", "~/llama.cpp", "/usr/local/bin", "/opt/llama.cpp")


def cartella_iniziale(attuale, candidate):
    """Cartella da cui far partire il selettore: quella del percorso gia' nel campo, altrimenti
    la prima esistente tra le candidate, altrimenti la home."""
    pth = os.path.expanduser((attuale or "").strip())
    if pth:
        d = pth if os.path.isdir(pth) else os.path.dirname(pth)
        if d and os.path.isdir(d):
            return d
    for c in candidate:
        c = os.path.expanduser(c)
        if os.path.isdir(c):
            return c
    return os.path.expanduser("~")


def dialog_file_sistema(parent, titolo, cartella, filtri):
    """Selettore di file del sistema (zenity = GTK, o kdialog). 'filtri' = [(etichetta, "*.gguf *.GGUF")].
    Restituisce il percorso scelto, "" se l'utente annulla, None se non c'e' un selettore utilizzabile
    (in tal caso si ripiega sul dialogo di Qt). Non blocca il ciclo degli eventi di Qt."""
    if not USA_DIALOG_SISTEMA or not sys.platform.startswith("linux"):
        return None
    zen, kd = shutil.which("zenity"), shutil.which("kdialog")
    if zen:
        prog = zen
        args = ["--file-selection", "--title=" + titolo, "--filename=" + cartella.rstrip("/") + "/"]
        args += ["--file-filter={} | {}".format(et, pat) for et, pat in filtri]
    elif kd:
        prog = kd
        args = ["--getopenfilename", cartella,
                "\n".join("{}|{}".format(pat, et) for et, pat in filtri), "--title", titolo]
    else:
        return None

    proc = QProcess(parent)
    loop = QEventLoop()
    proc.finished.connect(lambda *_: loop.quit())
    proc.errorOccurred.connect(lambda *_: loop.quit())
    proc.start(prog, args)
    if not proc.waitForStarted(3000):
        return None
    if parent is not None:
        parent.setEnabled(False)          # come un dialogo modale: la finestra resta ferma finche' e' aperto
    try:
        loop.exec_()
    finally:
        if parent is not None:
            parent.setEnabled(True)
            parent.activateWindow()
    if proc.error() == QProcess.FailedToStart or proc.exitStatus() != QProcess.NormalExit:
        return None
    out = bytes(proc.readAllStandardOutput()).decode(errors="replace").strip()
    if proc.exitCode() == 0 and out:
        return out.splitlines()[0]
    return "" if proc.exitCode() == 1 else None


def nome_separatore(chiave):
    for etichetta, k in SEPARATORI:
        if k == chiave:
            return etichetta.split(" (")[0]
    return "Demucs"


def cerca_voce_pulita(path):
    """Restituisce il file di voce pulita (<nome>_voce.wav) gia' salvato accanto all'audio,
    se esiste; None altrimenti. E' la traccia creata da 'trascrivi' (o da 'Salva voce e
    musica'): riutilizzarla evita di separare di nuovo la voce da capo."""
    v = os.path.splitext(path)[0] + "_voce.wav"
    return v if os.path.isfile(v) else None


def cerca_musica_separata(path):
    """Restituisce il file di musica separata (<nome>_musica.wav) gia' salvato accanto
    all'audio, se esiste; None altrimenti."""
    m = os.path.splitext(path)[0] + "_musica.wav"
    return m if os.path.isfile(m) else None


def _nuovo_stem(dest, motore):
    """True se 'dest' (<nome>_voce.wav / <nome>_musica.wav) va rigenerato: non esiste,
    oppure l'e' stato creato da un separatore diverso da 'motore' usato ora (es. prima
    Demucs, ora BS-RoFormer). Senza meta-dati il file si considera comunque riusabile."""
    if not os.path.exists(dest):
        return True
    meta = {}
    try:
        with open(os.path.splitext(dest)[0] + ".json", encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return False
    salvato = (meta.get("separatore") or "").strip().lower()
    return bool(salvato) and salvato != motore


def _scrivi_stem_meta(stem_path, motore, device):
    """Affianca al wav una piccola copia .json ('<nome>_voce.json') con il separatore e il
    dispositivo usati per crearlo: serve a 'trascrivi' per capire se il file in cache e'
    ancora valido o va rifatto perche' ottenuto con un altro motore."""
    try:
        with open(os.path.splitext(stem_path)[0] + ".json", "w", encoding="utf-8") as fh:
            json.dump({"separatore": motore, "device": device}, fh, ensure_ascii=False)
    except OSError:
        pass


def _trova_stem_bsroformer(cartella, src, marca):
    """audio-separator nomina i file '<nome>_(Vocals)_<modello>.wav' e '<nome>_(Instrumental)_<modello>.wav'."""
    for f in os.listdir(cartella):
        if f.lower().endswith(".wav") and marca in f.lower() and os.path.join(cartella, f) != src:
            return os.path.join(cartella, f)
    return None


def separa_voce_musica(path, motore, device):
    """Separa voce e musica di 'path'. Restituisce (wav_voce, wav_musica, cartella_temporanea, errore).
    wav_voce/wav_musica sono None se fallisce; la cartella temporanea va cancellata dal chiamante."""
    d = tempfile.mkdtemp(prefix="hira_stem_")
    try:
        src = os.path.join(d, "audio.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", path, "-vn", "-ar", "44100", "-ac", "2", src],
            check=True, stdin=subprocess.DEVNULL, capture_output=True)

        ultimo = ""
        for dev in (["cpu"] if device == "cpu" else ["cuda", "cpu"]):
            if motore == "bsroformer":
                os.makedirs(SEPARATOR_MODELS_DIR, exist_ok=True)
                env = dict(os.environ)
                if dev == "cpu":
                    env["CUDA_VISIBLE_DEVICES"] = "-1"
                # niente --single_stem: servono entrambe le tracce (Vocals + Instrumental)
                r = subprocess.run(
                    [sys.executable, "-c",
                     "from audio_separator.utils.cli import main; main()",
                     src, "--model_filename", BSROFORMER_MODEL,
                     "--model_file_dir", SEPARATOR_MODELS_DIR,
                     "--output_dir", d, "--output_format", "WAV"],
                    stdin=subprocess.DEVNULL, capture_output=True, env=env)
                voce = _trova_stem_bsroformer(d, src, "(vocals)")
                musica = _trova_stem_bsroformer(d, src, "(instrumental)")
            else:
                r = subprocess.run(
                    [sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs",
                     "-d", dev, "-o", d, src],
                    stdin=subprocess.DEVNULL, capture_output=True)
                cart = os.path.join(d, "htdemucs", "audio")
                voce = os.path.join(cart, "vocals.wav")
                musica = os.path.join(cart, "no_vocals.wav")
            if (r.returncode == 0 and voce and musica
                    and os.path.isfile(voce) and os.path.isfile(musica)):
                return voce, musica, d, ""
            ultimo = (r.stderr or b"").decode(errors="replace").strip()[-300:]
            if "No module named" in ultimo:
                ultimo += ("  ->  python -m pip install " +
                           ('"audio-separator[gpu]"' if motore == "bsroformer" else "demucs"))
                break
        return None, None, d, ultimo or "errore sconosciuto"
    except FileNotFoundError:
        return None, None, d, "ffmpeg non trovato"
    except subprocess.CalledProcessError as e:
        return None, None, d, (e.stderr or b"").decode(errors="replace").strip()[-300:]


class SeparatoreStem(QThread):
    """Separa voce e musica del file originale e salva le due tracce nei percorsi indicati.

    Con riuso_voce=True 'path' e' gia' la traccia di voce pulita (<nome>_voce.wav) creata
    da 'trascrivi': non va salvata di nuovo (e mai sovrascritta, perche' sarebbe il file
    stesso usato come sorgente), basta separare la musica."""
    stato = pyqtSignal(str)
    fatto = pyqtSignal(str)          # messaggio di errore ("" se tutto ok)

    def __init__(self, path, motore, device, dest_voce, dest_musica, riuso_voce=False):
        super().__init__()
        self.path, self.motore, self.device = path, motore, device
        self.dest_voce, self.dest_musica = dest_voce, dest_musica
        self.riuso_voce = bool(riuso_voce)

    def run(self):
        d = None
        try:
            if self.riuso_voce:
                self.stato.emit("Voce gia' isolata: separo solo la musica con {} "
                                "(puo' richiedere alcuni minuti)...".format(nome_separatore(self.motore)))
            else:
                self.stato.emit("Separo voce e musica con {} (puo' richiedere alcuni minuti)...".format(
                    nome_separatore(self.motore)))
            voce, musica, d, err = separa_voce_musica(self.path, self.motore, self.device)
            if not voce or not musica:
                self.fatto.emit(err or "separazione non riuscita")
                return
            if not self.riuso_voce:
                shutil.copyfile(voce, self.dest_voce)
            shutil.copyfile(musica, self.dest_musica)
            self.fatto.emit("")
        except Exception as e:
            self.fatto.emit(str(e))
        finally:
            if d:
                shutil.rmtree(d, ignore_errors=True)


def esporta_mka(video, voce, musica, lossless=False):
    """Mette la traccia di voce e quella di musica in un unico file .mka (Matroska audio
    multitraccia) accanto al video/audio originale: <nome>.mka, con le tracce chiamate
    'Voce' e 'Musica'. Con lossless=True usa FLAC, altrimenti Opus a 192k.
    Restituisce il percorso del file creato; solleva CalledProcessError/FileNotFoundError
    se ffmpeg fallisce o non esiste."""
    base = os.path.splitext(video)[0]
    dest = base + ".mka"
    codec = ["-c:a", "flac"] if lossless else ["-c:a", "libopus", "-b:a", "192k"]
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-i", voce, "-i", musica,
        "-map", "0:a:0", "-map", "1:a:0",
        *codec,
        "-metadata:s:a:0", "title=Voce",
        "-metadata:s:a:1", "title=Musica",
        dest,
    ], check=True, stdin=subprocess.DEVNULL, capture_output=True)
    return dest


class EsportatoreMka(QThread):
    """Esegue esporta_mka() in un thread separato per non bloccare l'interfaccia."""
    stato = pyqtSignal(str)
    fatto = pyqtSignal(str)          # messaggio di errore ("" se tutto ok)

    def __init__(self, video, voce, musica, lossless=False):
        super().__init__()
        self.video, self.voce, self.musica = video, voce, musica
        self.lossless = bool(lossless)

    def run(self):
        try:
            self.stato.emit("Esporto le tracce in {}...".format(
                os.path.basename(os.path.splitext(self.video)[0] + ".mka")))
            esporta_mka(self.video, self.voce, self.musica, self.lossless)
            self.fatto.emit("")
        except FileNotFoundError:
            self.fatto.emit("ffmpeg non trovato: installalo e aggiungilo al PATH.")
        except subprocess.CalledProcessError as e:
            self.fatto.emit((e.stderr or b"").decode(errors="replace").strip()[-300:]
                            or "esportazione non riuscita")
        except Exception as e:
            self.fatto.emit(str(e))

# traduzione locale (NLLB-200 distilled, CTranslate2 int8; licenza CC-BY-NC 4.0: solo uso
# non commerciale). Il modello si scarica una volta sola, al primo uso.
MODELLI_TRAD = [
    ("NLLB 600M - veloce (~630 MB)", "nllb", "JustFrederik/nllb-200-distilled-600M-ct2-int8"),
    ("NLLB 1.3B - piu' accurato (~1,4 GB)", "nllb", "JustFrederik/nllb-200-distilled-1.3B-ct2-int8"),
    ("TranslateGemma 4B - via Ollama (leggero)", "ollama", "translategemma:4b"),
    ("TranslateGemma 12B - via Ollama (migliore, serve piu' VRAM)", "ollama", "translategemma:12b"),
    ("LM Studio - server locale (qualsiasi modello caricato)", "lmstudio", ""),
    ("llama.cpp - llama-server avviato dallo script (GGUF caricato solo per la traduzione)", "llamaserver", ""),
]
LM_URL_PREDEFINITO = "http://localhost:1235/v1"
# traduzione "a contesto completo": dimensione massima di un gruppo di righe (le finestre
# di contesto dei modelli locali sono spesso piccole, ad es. 4096 token)
LM_CTX_RIGHE = 30
LM_CTX_CAR = 1200
# modalita' 'un solo passaggio': gruppi piu' larghi (bastano ~8K di contesto)
LM_CTX_RIGHE_1 = 40
LM_CTX_CAR_1 = 1800
NLLB_FILE = ["model.bin", "config.json", "shared_vocabulary.*", "sentencepiece.bpe.model"]
# lingue per TranslateGemma: (nome, codice)
LINGUE_GEMMA = {
    "it": ("Italian", "it"), "en": ("English", "en"), "zh": ("Chinese", "zh-Hans"),
    "ko": ("Korean", "ko"), "fr": ("French", "fr"), "de": ("German", "de"), "es": ("Spanish", "es"),
}
NLLB_LINGUE = {
    "ja": "jpn_Jpan", "it": "ita_Latn", "en": "eng_Latn", "zh": "zho_Hans",
    "ko": "kor_Hang", "fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn",
}

SR = 16000  # frequenza di campionamento usata per decodifica/riproduzione

# colori
C_BG = "#14151c"
C_CARD = "#1d1f2b"
C_BORDER = "#2b2e3f"
C_TEXT = "#e6e8f2"
C_MUTED = "#8b90a8"
C_ACCENT = "#7c6cff"      # parte non ancora riprodotta
C_SPOKEN = "#ffb454"      # parte gia' pronunciata
C_ROSSO = "#ff2a2a"       # punto di riproduzione sulla sinusoide dei dB (schermo solo audio)
SUB_CURRENT = "#ff2a2a"       # picco di rosso del kanji/token pronunciato in questo istante
SUB_TRAD_CURRENT = "#00b0ff"  # azzurro intenso: parte della traduzione che corrisponde al token rosso
SUB_TRAD = "#bfe3ff"          # traduzione non ancora illuminata
SUB_PASSI = 16                # a quanti passi si divide la durata di un token (ridisegni del colore)
WAVE_SPAN = 8.0           # secondi visibili nella sinusoide
WAVE_CYCLES = 3           # oscillazioni visibili in tutta la larghezza (era 18, poi 6: ora lunghezza d'onda sestupla dell'originale)
WAVE_HEAD = 0.80          # posizione orizzontale della testina (0 = bordo sinistro, 1 = bordo destro)

STILE = """
#finestra, #interno { background: %(bg)s; }
QDialog, QFileDialog { background: %(bg)s; color: %(text)s; }
QScrollArea { background: transparent; border: none; }
QWidget { color: %(text)s; font-size: 13px;
          font-family: "Segoe UI", "Yu Gothic UI", "Meiryo", "Noto Sans CJK JP", "Hiragino Sans", sans-serif; }
QFrame#card { background: %(card)s; border: 1px solid %(border)s; border-radius: 12px; }
QLabel { background: transparent; }
QLabel#titolo { font-size: 22px; font-weight: 700; }
QLabel#sotto { color: %(muted)s; }
QLabel#sez { color: %(muted)s; font-weight: 700; font-size: 11px; letter-spacing: 1px; }
QLabel#tempo { font-family: "Consolas", "Menlo", monospace; font-size: 14px; }
QLineEdit, QComboBox, QDoubleSpinBox {
    background: #12131a; border: 1px solid %(border)s; border-radius: 8px; padding: 6px 8px;
    selection-background-color: %(accent)s; }
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus { border: 1px solid %(accent)s; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView { background: %(card)s; border: 1px solid %(border)s;
                              selection-background-color: %(accent)s; outline: none; }
QPushButton { background: #2a2d40; border: none; border-radius: 8px; padding: 8px 14px; color: %(text)s; }
QPushButton:hover { background: #34384f; }
QPushButton:pressed { background: #3d4260; }
QPushButton:disabled { color: #5b6078; background: #20222f; }
QPushButton#primario { background: %(accent)s; color: white; font-weight: 600; padding: 9px 22px; }
QPushButton#primario:hover { background: #917fff; }
QPushButton#primario:disabled { background: #3b3568; color: #8c87b5; }
QPushButton#play { background: %(accent)s; color: white; font-size: 16px; border-radius: 20px;
                   min-width: 40px; max-width: 40px; min-height: 40px; max-height: 40px; padding: 0; }
QPushButton#play:hover { background: #917fff; }
QPushButton#tondo { font-size: 14px; border-radius: 20px;
                    min-width: 40px; max-width: 40px; min-height: 40px; max-height: 40px; padding: 0; }
QPushButton#piccolo { padding: 4px 10px; font-size: 12px; }
QTextEdit { background: #12131a; border: 1px solid %(border)s; border-radius: 10px; padding: 8px;
            font-size: 18px; selection-background-color: %(accent)s; }
QProgressBar { background: #2a2d40; border: none; border-radius: 2px; max-height: 4px; }
QProgressBar::chunk { background: %(accent)s; border-radius: 2px; }
QCheckBox { background: transparent; spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 4px;
                       border: 1px solid #3a3e57; background: #12131a; }
QCheckBox::indicator:checked { background: %(accent)s; border-color: %(accent)s; }
QSlider::groove:horizontal { height: 4px; background: #2a2d40; border-radius: 2px; }
QSlider::sub-page:horizontal { background: %(accent)s; border-radius: 2px; }
QSlider::handle:horizontal { width: 12px; height: 12px; margin: -4px 0; border-radius: 6px; background: %(text)s; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #3a3e57; border-radius: 4px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle:horizontal { background: #3a3e57; border-radius: 4px; min-width: 30px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QSplitter::handle { background: transparent; }
QToolTip { background: %(card)s; color: %(text)s; border: 1px solid %(border)s; padding: 4px; }

/* Stile menu, viste ad albero e lista del selettore file */
QListView, QTreeView, QTableView {
    background: #12131a;
    alternate-background-color: %(card)s;
    border: 1px solid %(border)s;
    border-radius: 8px;
    color: %(text)s;
    selection-background-color: %(accent)s;
    selection-color: #ffffff;
    outline: none;
    padding: 4px;
}
QListView::item, QTreeView::item {
    padding: 4px 6px;
    border-radius: 4px;
}
QListView::item:hover, QTreeView::item:hover {
    background: #252838;
}
QListView::item:selected, QTreeView::item:selected {
    background: %(accent)s;
    color: #ffffff;
}
QHeaderView::section {
    background: %(card)s;
    color: %(muted)s;
    padding: 5px 8px;
    border: none;
    border-right: 1px solid %(border)s;
    border-bottom: 1px solid %(border)s;
    font-weight: 600;
}
QToolButton {
    background: #2a2d40;
    border: 1px solid %(border)s;
    border-radius: 6px;
    padding: 5px;
    color: %(text)s;
}
QToolButton:hover {
    background: #34384f;
}
QToolButton:pressed {
    background: #3d4260;
}
QMenu {
    background: %(card)s;
    border: 1px solid %(border)s;
    border-radius: 8px;
    padding: 4px;
}
QMenu::item {
    padding: 6px 20px 6px 12px;
    border-radius: 4px;
    color: %(text)s;
}
QMenu::item:selected {
    background: %(accent)s;
    color: #ffffff;
}
""" % dict(bg=C_BG, card=C_CARD, border=C_BORDER, text=C_TEXT, muted=C_MUTED, accent=C_ACCENT)


def imposta_palette_scura(app):
    """Imposta una palette scura di default a livello di applicazione
    per evitare contrasti errati o pannelli bianchi nei dialoghi nativi."""
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(C_BG))
    palette.setColor(QPalette.WindowText, QColor(C_TEXT))
    palette.setColor(QPalette.Base, QColor("#12131a"))
    palette.setColor(QPalette.AlternateBase, QColor(C_CARD))
    palette.setColor(QPalette.ToolTipBase, QColor(C_CARD))
    palette.setColor(QPalette.ToolTipText, QColor(C_TEXT))
    palette.setColor(QPalette.Text, QColor(C_TEXT))
    palette.setColor(QPalette.Button, QColor(C_CARD))
    palette.setColor(QPalette.ButtonText, QColor(C_TEXT))
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(C_ACCENT))
    palette.setColor(QPalette.Highlight, QColor(C_ACCENT))
    palette.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(palette)


# cache del modello faster-whisper caricato
_cache = {"key": None, "model": None}
_cache_nllb = {"tr": None, "sp": None, "dev": None, "repo": None}
_kakasi_instance = None


def get_kakasi():
    """Inizializza pykakasi in modo lazy per non rallentare l'avvio della GUI."""
    global _kakasi_instance
    if _kakasi_instance is None:
        import pykakasi
        _kakasi_instance = pykakasi.kakasi()
    return _kakasi_instance


def a_hiragana(testo: str) -> str:
    """Converte kanji e katakana in hiragana, lasciando intatti simboli/timestamp."""
    if not testo:
        return ""
    try:
        kks = get_kakasi()
        risultati = kks.convert(testo)
        return "".join(item["hira"] for item in risultati)
    except ImportError:
        return "[Libreria 'pykakasi' non trovata. Installala con: pip install pykakasi]"
    except Exception as e:
        return f"[Errore conversione hiragana: {e}]"


def _vram_usata_mb():
    """VRAM usata (MB) sulla prima GPU NVIDIA, oppure None se nvidia-smi non c'e'."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            return int(float(r.stdout.strip().splitlines()[0]))
    except Exception:
        pass
    return None


def _svuota_cache_gpu():
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    # Se PyTorch e' comunque installato, libera anche la sua cache CUDA.
    # faster-whisper usa CTranslate2, ma questo non guasta.
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def libera_whisper():
    """Scarica solo Whisper dalla memoria (RAM e VRAM)."""
    _cache["key"] = None
    _cache["model"] = None
    _svuota_cache_gpu()


def libera_modello():
    """Scarica dalla memoria Whisper e il modello di traduzione NLLB."""
    libera_whisper()
    _cache_nllb.update(tr=None, sp=None, dev=None, repo=None)
    _svuota_cache_gpu()


def log_hira(msg):
    """Messaggi diagnostici in console (stderr), con l'ora: servono a capire dove si ferma il caricamento di un file."""
    print("[hira %s] %s" % (time.strftime("%H:%M:%S"), msg), file=sys.stderr, flush=True)


def ha_flusso_video(path):
    """True se il file contiene una vera traccia video (la copertina
    incorporata in un mp3/m4a non conta). Usa ffprobe; se manca -> False."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v",
             "-show_entries", "stream=index:stream_disposition=attached_pic",
             "-of", "json", path],
            capture_output=True, stdin=subprocess.DEVNULL, timeout=20)
        dati = json.loads(r.stdout.decode(errors="replace") or "{}")
        for st in dati.get("streams", []):
            if not st.get("disposition", {}).get("attached_pic", 0):
                return True
    except subprocess.TimeoutExpired:
        log_hira("ffprobe: nessuna risposta in 20 s, tratto il file come solo audio")
    except Exception as e:
        log_hira("ffprobe: %s (tratto il file come solo audio)" % e)
    return False


def info_video(path):
    """Dimensioni (ridotte, 8 bit) e frequenza dei fotogrammi con cui mostrare il video, o None se non si puo'.
    Tiene conto del rapporto d'aspetto dei pixel (sorgenti anamorfiche)."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,sample_aspect_ratio,avg_frame_rate,r_frame_rate",
             "-of", "json", path],
            capture_output=True, stdin=subprocess.DEVNULL, timeout=20)
        st = json.loads(r.stdout.decode(errors="replace") or "{}")["streams"][0]
        w, h = int(st["width"]), int(st["height"])
        sar = 1.0
        try:
            a, b = str(st.get("sample_aspect_ratio", "1:1")).split(":")
            if float(a) > 0 and float(b) > 0:
                sar = float(a) / float(b)
        except Exception:
            pass
        fps = 24.0
        for k in ("avg_frame_rate", "r_frame_rate"):
            try:
                n, d = str(st.get(k, "0/1")).split("/")
                v = float(n) / float(d)
                if 1.0 < v < 250.0:
                    fps = v
                    break
            except Exception:
                pass
        fps = min(fps, 30.0)
        ew = w * sar
        W = min(960, int(round(ew)))
        W -= W % 2
        H = int(round(W * h / ew))
        H -= H % 2
        return dict(w=max(2, W), h=max(2, H), fps=fps)
    except Exception as e:
        log_hira("info video: %r" % (e,))
        return None


class SorgenteFrame(QThread):
    """Decodifica i fotogrammi del video con ffmpeg (RGB 8 bit, ridotti) e li accoda con il loro istante. Il video non
    passa piu' dal backend multimediale di Qt (GStreamer), che con H.264 a 10 bit (High 10) si blocca o resta nero:
    l'audio suona sempre dal wav gia' estratto e le immagini seguono la sua posizione."""
    MAX = 30                      # fotogrammi tenuti in coda: ffmpeg aspetta se la coda e' piena

    def __init__(self, path, inizio, w, h, fps):
        super().__init__()
        self.path, self.inizio, self.w, self.h, self.fps = path, float(inizio), int(w), int(h), float(fps)
        self.coda = collections.deque()      # (istante in secondi, QImage)
        self.ultimo = self.inizio            # istante dell'ultimo fotogramma accodato
        self.fermo = False
        self.finito = False
        self.proc = None

    def ferma(self):
        self.fermo = True
        p = self.proc
        if p is not None:
            try:
                p.kill()
            except Exception:
                pass

    def run(self):
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-ss", "%.3f" % self.inizio, "-i", self.path,
               "-an", "-sn", "-dn",
               "-vf", "fps=%.5f,scale=%d:%d:flags=fast_bilinear,format=rgb24" % (self.fps, self.w, self.h),
               "-f", "rawvideo", "-"]
        size = self.w * self.h * 3
        t0 = time.perf_counter()
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         stdin=subprocess.DEVNULL, bufsize=size * 4)
            n = 0
            while not self.fermo:
                while len(self.coda) >= self.MAX and not self.fermo:
                    self.msleep(10)
                if self.fermo:
                    break
                buf = self.proc.stdout.read(size)
                if not buf or len(buf) < size:
                    break
                img = QImage(buf, self.w, self.h, self.w * 3, QImage.Format_RGB888).copy()
                t = self.inizio + n / self.fps
                n += 1
                self.coda.append((t, img))
                self.ultimo = t
                if n == 1:
                    log_hira("video: primo fotogramma dopo %.2f s (da %.1f s, %dx%d a %.3f fps)"
                             % (time.perf_counter() - t0, self.inizio, self.w, self.h, self.fps))
        except Exception as e:
            log_hira("video: errore di decodifica: %r" % (e,))
        finally:
            p = self.proc
            if p is not None:
                try:
                    p.kill()
                    p.wait(2)
                except Exception:
                    pass
            self.finito = True


def fmt_tempo(s):
    m, sec = divmod(float(s), 60)
    return "{:02d}:{:04.1f}".format(int(m), sec)


def l16(s):
    """Lunghezza in unita' UTF-16 (le posizioni di QTextCursor usano queste)."""
    return len(s.encode("utf-16-le")) // 2


def distribuisci(testo, t0, t1):
    """Se mancano i tempi per parola: divide il segmento in token (kakasi)
    e assegna a ciascuno un tempo proporzionale alla lunghezza della lettura."""
    items = get_kakasi().convert(testo)
    pesi = []
    for it in items:
        h = it["hira"]
        pesi.append(float(len(h)) if any(c.isalnum() for c in h) else 0.3)

    tot = sum(pesi) or 1.0
    out, t = [], t0

    for it, w in zip(items, pesi):
        dt = (t1 - t0) * w / tot
        out.append((it["orig"], it["hira"], t, t + dt))
        t += dt

    return out


# ---------------------------------------------------------------------------
# Formato karaoke compatto (versione 2)
#
# Il formato v1 salvava ogni informazione piu' volte (testo completo, testo
# hiragana, intervalli di ogni token, token, indice del segmento, tempi con 15+
# decimali). Nella v2 si salvano solo i token con i loro tempi: testi e intervalli
# si ricostruiscono al caricamento.
#   "s": [ [prefisso_segmento, [token, ...]], ... ]
#   token = [orig, t0, t1]            se hiragana == orig
#         = [orig, hira, t0, t1]      altrimenti
# ---------------------------------------------------------------------------
def testo_righe(righe):
    """Unisce le righe con a-capo e restituisce (testo, intervalli UTF-16 di ogni riga)."""
    spans, pos = [], 0
    for r in righe:
        n = l16(r)
        spans.append((pos, pos + n))
        pos += n + 1
    return "\n".join(righe), spans


def _u16_slice(s, a, b):
    return s.encode("utf-16-le")[2 * a:2 * b].decode("utf-16-le")


def allinea_traduzione(t0, t1, trad):
    """Riallinea i tempi delle righe tradotte a quelli dei token giapponesi.

    Le righe nascono da un secondo passaggio di whisper con tempi propri (di solito
    in anticipo rispetto alle parole giapponesi). Ogni token giapponese viene
    assegnato alla riga con cui si sovrappone di piu' (o alla piu' vicina) e la riga
    appare quando parte il suo primo token e scompare quando finisce l'ultimo.
    Righe senza token vicini: tempi originali. Ripetibile senza effetti collaterali."""
    if not trad or not trad.get("righe") or not t0:
        return trad
    if trad.get("fissa"):      # tempi presi dai prefissi [N|mm:ss.s]: non toccarli
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


def segmenti_giapponesi(seg):
    """Da tok['seg'] (indice di segmento per ogni token) a una lista di intervalli [a, b) di token."""
    out = []
    for i, sg in enumerate(seg):
        if i == 0 or sg != seg[i - 1]:
            out.append([i, i + 1])
        else:
            out[-1][1] = i + 1
    return out


_RE_PREF_NUM = re.compile(
    r"^\s*[\[\(\uff08\u3010\uff3b]\s*(\d+)\s*(?:[|:/\uff5c\uff1a]\s*[\d:.,]+\s*)?[\]\)\uff09\u3011\uff3d]\s*(.*)$")


def testo_numerato(tok, t0):
    """Segmenti giapponesi come righe '[N|mm:ss.s] testo', da dare a un traduttore esterno:
    il prefisso non si traduce e permette di riagganciare ogni riga al suo segmento."""
    segs = segmenti_giapponesi(tok["seg"])
    righe = []
    for n, (a, b) in enumerate(segs, 1):
        testo = "".join(tok["o"][a:b]).replace("\n", " ").strip()
        righe.append("[{}|{}] {}".format(n, fmt_tempo(t0[a]), testo))
    return "\n".join(righe), len(segs)


def parse_tempo(s):
    """'13:23.4' / '1:02:03.5' / '23.4' -> secondi (float), None se non valido."""
    try:
        parti = [float(x) for x in s.strip().replace(",", ".").split(":")]
    except ValueError:
        return None
    if not parti or len(parti) > 3:
        return None
    v = 0.0
    for x in parti:
        v = v * 60.0 + x
    return v


def leggi_numerate_tempi(testo, n_seg):
    """Legge righe '[N|tempo] testo' (anche solo '[N] testo'; le righe senza prefisso si
    accodano alla precedente). Restituisce (righe, trovate, scartate) con una riga per
    segmento (vuota se manca), oppure None se il testo non e' in questo formato."""
    grezze = (testo or "").replace("\r", "").split("\n")
    piene = [r for r in grezze if r.strip()]
    if not piene or sum(1 for r in piene if _RE_PREF_NUM.match(r)) < 0.6 * len(piene):
        return None
    trovate, ultimo, scartate, tempi = {}, None, 0, {}
    for r in grezze:
        m = _RE_PREF_NUM.match(r)
        if m:
            n = int(m.group(1))
            if 1 <= n <= n_seg and n not in trovate:
                trovate[n] = m.group(2).strip()
                ultimo = n
                mt = re.match(r"^\s*[\[\(\uff08\u3010\uff3b]\s*\d+\s*[|:/\uff5c\uff1a]\s*([\d:.,]+)", r)
                if mt:
                    v = parse_tempo(mt.group(1))
                    if v is not None:
                        tempi[n] = v
            else:
                scartate += 1
                ultimo = None
        elif ultimo is not None and r.strip():
            trovate[ultimo] += " " + r.strip()
    return ([trovate.get(i, "") for i in range(1, n_seg + 1)], len(trovate), scartate,
            [tempi.get(i) for i in range(1, n_seg + 1)])


def spans_riquadro_numerato(testo, n_seg):
    """Intervalli UTF-16 del testo (dopo il prefisso [N|tempo]) di ogni segmento, calcolati sul
    riquadro cosi' com'e': serve a evidenziare senza riscrivere il riquadro. Le righe senza
    prefisso accodano alla precedente (come in leggi_numerate_tempi); i segmenti senza riga
    hanno un intervallo vuoto."""
    spans, visti, ultimo, pos = [None] * n_seg, set(), None, 0
    for r in (testo or "").replace("\r", "").split("\n"):
        m = _RE_PREF_NUM.match(r)
        if m:
            n = int(m.group(1))
            if 1 <= n <= n_seg and n not in visti:
                visti.add(n)
                ultimo = n
                a = pos + l16(r[:m.start(2)])
                spans[n - 1] = [a, a + l16(m.group(2).rstrip())]
            else:
                ultimo = None
        elif ultimo is not None and r.strip():
            spans[ultimo - 1][1] = pos + l16(r.rstrip())
        pos += l16(r) + 1
    prev, out = 0, []
    for sp in spans:
        sp = tuple(sp) if sp is not None else (prev, prev)
        out.append(sp)
        prev = sp[1]
    return out


def spans_riquadro_righe(testo, n):
    """Intervalli UTF-16 delle prime n righe del riquadro, cosi' come sono scritte."""
    spans, pos = [], 0
    for r in (testo or "").replace("\r", "").split("\n")[:n]:
        spans.append((pos, pos + l16(r)))
        pos += l16(r) + 1
    return spans


_RE_INTERVALLO = re.compile(
    r"\[\s*([\d:.,]+)\s*(?:->|-->|\u2192)\s*\[?\s*([\d:.,]+)\s*\]")   # [mm:ss.s -> mm:ss.s]


def leggi_intervalli_tempo(testo):
    """Legge un testo fatto di frasi precedute da '[inizio -> fine]' (formato della trascrizione con
    timestamp; tollerato anche '[inizio -> [fine]' e piu' frasi sulla stessa riga). Restituisce
    (righe, t0, t1, spans) con gli intervalli UTF-16 del solo testo di ogni frase nel riquadro, oppure
    None se il testo non e' in questo formato."""
    testo = (testo or "").replace("\r", "")
    piene = [r for r in testo.split("\n") if r.strip()]
    ms = list(_RE_INTERVALLO.finditer(testo))
    if not ms or len(ms) < 0.6 * len(piene) or testo[:ms[0].start()].strip():
        return None
    righe, t0, t1, spans, prev = [], [], [], [], 0.0
    for k, m in enumerate(ms):
        a, b = m.end(), (ms[k + 1].start() if k + 1 < len(ms) else len(testo))
        corpo = testo[a:b]
        ini = a + (len(corpo) - len(corpo.lstrip()))
        fin = a + len(corpo.rstrip())
        riga = " ".join(testo[ini:fin].split())
        va, vb = parse_tempo(m.group(1)), parse_tempo(m.group(2))
        if not riga or va is None or vb is None:
            continue
        va = max(va, prev)
        prev = va
        righe.append(riga)
        t0.append(va)
        t1.append(max(vb, va))
        spans.append((l16(testo[:ini]), l16(testo[:fin])))
    if not righe:
        return None
    return righe, t0, t1, spans


def dividi_frasi(testo, max_car=80):
    """Spezza un testo in frasi (fine frase . ! ? ... e simili); le frasi piu' lunghe di
    max_car vengono divise alle virgole/punti e virgola e, se serve, a capo parola."""
    testo = " ".join((testo or "").split())
    if not testo:
        return []
    frasi = [f.strip() for f in re.split(r"(?<=[.!?\u2026\u3002\uff01\uff1f])\s+", testo) if f.strip()]
    out = []
    for f in frasi:
        if len(f) <= max_car:
            out.append(f)
            continue
        cur = ""
        for pezzo in re.split(r"(?<=[,;:\u3001])\s+", f):
            if cur and len(cur) + 1 + len(pezzo) > max_car:
                out.append(cur)
                cur = pezzo
            else:
                cur = (cur + " " + pezzo).strip()
        if cur:
            out.append(cur)
    finale = []
    for r in out:
        finale.extend(textwrap.wrap(r, width=max_car + 20, break_long_words=False) or [r])
    return finale


def righe_su_tempi(righe, t0, t1, pesi):
    """Distribuisce M righe (traduzione incollata a mano) sul tempo dei token giapponesi,
    in proporzione al 'peso' cumulato dei token (lunghezza della lettura). E' una stima:
    serve quando le righe incollate non coincidono con i segmenti. Restituisce dict(righe, t0, t1)."""
    n, m = len(t0), len(righe)
    cw = [0.0]
    for w in pesi:
        cw.append(cw[-1] + w)
    tot = cw[-1] or 1.0
    bordi = [0]
    for k in range(1, m):
        j = bisect_left(cw, tot * k / m)
        bordi.append(min(max(j, bordi[-1]), n))
    bordi.append(n)
    a0, a1, prev = [], [], 0.0
    for k in range(m):
        a = min(bordi[k], n - 1)
        b = max(bordi[k + 1], a + 1)
        ini = max(t0[a], prev)
        fin = max(t1[min(b, n) - 1], ini)
        a0.append(ini)
        a1.append(fin)
        prev = ini
    return dict(righe=list(righe), t0=a0, t1=a1)


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


def karaoke_compatto(r):
    """Dal risultato della trascrizione al dizionario v2 (senza meta-dati del file).
    Restituisce None se i dati non sono ricostruibili esattamente: in quel caso
    il chiamante ricade sul formato v1."""
    tok = r["tok"]
    n = len(r["t0"])
    segs = []
    fine = 0
    for i in range(n):
        a, b = r["span_o"][i]
        if i == 0 or tok["seg"][i] != tok["seg"][i - 1]:
            segs.append([_u16_slice(r["orig"], fine, a), []])
        elif a != fine:
            return None
        o, h = tok["o"][i], tok["h"][i]
        a0, a1 = round(float(r["t0"][i]), 2), round(float(r["t1"][i]), 2)
        segs[-1][1].append([o, a0, a1] if h == o else [o, h, a0, a1])
        fine = b

    d = dict(versione=2, s=segs)
    tr = r.get("trad")
    if tr and tr.get("righe"):
        d["tr"] = [[t, round(float(a), 2), round(float(b), 2)]
                   for t, a, b in zip(tr["righe"], tr["t0"], tr["t1"])]
        if tr.get("fissa"):
            d["trf"] = 1
        if tr.get("num"):
            d["trn"] = 1

    # verifica: la ricostruzione deve riprodurre esattamente testi e intervalli
    try:
        x = karaoke_espandi(d)
    except Exception:
        return None
    if (x["orig"] != r["orig"] or x["hira"] != r["hira"]
            or [tuple(v) for v in r["span_o"]] != x["span_o"]
            or [tuple(v) for v in r["span_h"]] != x["span_h"]
            or tok["o"] != x["tok"]["o"] or tok["h"] != x["tok"]["h"]
            or list(tok["seg"]) != x["tok"]["seg"]):
        return None
    return d


# ---------------------------------------------------------------------------
# Batch: ricerca file e salvataggio dei risultati senza passare dalla GUI
# ---------------------------------------------------------------------------
EXT_MEDIA = tuple(e[1:].lower() for e in (EST_AUDIO + " " + EST_VIDEO).split())


def _chiave_naturale(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def trova_media(percorso):
    """Elenco dei file audio/video: il file stesso, oppure tutti quelli contenuti
    in una cartella (sottocartelle comprese), in ordine naturale (2 prima di 10)."""
    if os.path.isfile(percorso):
        return [percorso] if percorso.lower().endswith(EXT_MEDIA) else []
    trovati = []
    if os.path.isdir(percorso):
        for rad, sotto, files in os.walk(percorso):
            sotto.sort(key=_chiave_naturale)
            for f in sorted(files, key=_chiave_naturale):
                if f.lower().endswith(EXT_MEDIA):
                    trovati.append(os.path.join(rad, f))
    return trovati


def scrivi_karaoke_json(r, media, percorso):
    """Scrive il file _karaoke.json del risultato r (formato compatto v2, se possibile)."""
    dati = dict(
        file=os.path.basename(media),
        percorso_file=os.path.abspath(media),
    )
    comp = karaoke_compatto(r)
    if comp is not None:
        dati.update(comp)
    else:  # fallback: formato completo v1
        dati.update(
            versione=1,
            orig=r["orig"],
            hira=r["hira"],
            span_o=[list(x) for x in r["span_o"]],
            span_h=[list(x) for x in r["span_h"]],
            t0=[round(float(t), 2) for t in r["t0"]],
            t1=[round(float(t), 2) for t in r["t1"]],
            tok=r["tok"],
        )
    with open(percorso, "w", encoding="utf-8") as fh:
        json.dump(dati, fh, ensure_ascii=False, separators=(",", ":"))


def traduzione_numerata(trad):
    """Traduzione come righe '[N|mm:ss.s] testo' (stesso formato di 'Copia con numeri'):
    e' il formato dei file _traduzione.txt, con il tempo di inizio di ogni frase."""
    return "\n".join(
        "[{}|{}] {}".format(k, fmt_tempo(a), " ".join(str(rg).split()))
        for k, (rg, a) in enumerate(zip(trad["righe"], trad["t0"]), 1))


def salva_risultato(r, media):
    """Salva accanto a 'media' gli stessi file di "Salva accanto all'audio":
    _trascrizione.txt, _hiragana.txt, _traduzione.txt (se c'e') e _karaoke.json.
    Se la trascrizione ha isolato la voce, conserva anche quella (<nome>_voce.wav),
    cosi' da non doverla separare di nuovo in seguito."""
    base = os.path.splitext(media)[0]
    testi = [("_trascrizione.txt", r["orig"]), ("_hiragana.txt", r["hira"])]
    trad = r.get("trad")
    if trad and trad.get("righe"):
        testi.append(("_traduzione.txt", traduzione_numerata(trad)))
    scritti = []
    for suff, t in testi:
        if t.strip():
            with open(base + suff, "w", encoding="utf-8") as fh:
                fh.write(t)
            scritti.append(suff)
    scrivi_karaoke_json(r, media, base + "_karaoke.json")
    scritti.append("_karaoke.json")
    if r.get("voce_file") and not r.get("voce_temporanea") \
            and os.path.isfile(r["voce_file"]) and os.path.abspath(r["voce_file"]) != base + "_voce.wav":
        try:
            shutil.copyfile(r["voce_file"], base + "_voce.wav")
            scritti.append("_voce.wav")
        except OSError:
            pass
    return scritti


# ---------------------------------------------------------------------------
# Traduzione locale (NLLB-200 via CTranslate2)
# ---------------------------------------------------------------------------
def _traduci_nllb(testi, lingua, device="auto", progresso=None, stato=None, batch=16, repo=None):
    """Traduce dal giapponese con NLLB-200 (CTranslate2), tutto in locale.
    Ogni riga e' una frase indipendente e i risultati tornano nello stesso ordine:
    l'allineamento con i tempi e' garantito. Restituisce una lista lunga come 'testi'
    ("" per le righe vuote, None dove il modello non ha dato testo).
    Solleva ImportError se manca sentencepiece/ctranslate2/huggingface_hub."""
    if lingua not in NLLB_LINGUE:
        raise ValueError("lingua '{}' non supportata dalla traduzione locale".format(lingua))
    import ctranslate2
    import sentencepiece as spm
    from huggingface_hub import snapshot_download

    repo = repo or MODELLI_TRAD[0][2]
    src, tgt = NLLB_LINGUE["ja"], NLLB_LINGUE[lingua]
    risultati = ["" if not t else None for t in testi]
    da_fare = [i for i, t in enumerate(testi) if t]
    if not da_fare:
        return risultati

    if _cache_nllb["tr"] is None or _cache_nllb["repo"] != repo:
        _cache_nllb.update(tr=None, sp=None, dev=None, repo=None)
        if stato:
            stato("Cerco il modello di traduzione {} (la prima volta lo scarica)...".format(repo.split("/")[-1]))
        cartella = snapshot_download(repo, allow_patterns=NLLB_FILE)
        sp = spm.SentencePieceProcessor()
        sp.load(os.path.join(cartella, "sentencepiece.bpe.model"))

        tr, ultimo, usato = None, None, None
        for dev in (["cuda", "cpu"] if device in ("auto", "cuda") else ["cpu"]):
            try:
                if dev == "cuda" and ctranslate2.get_cuda_device_count() == 0:
                    continue
                if stato:
                    stato("Carico il modello di traduzione su {}...".format(dev))
                tr = ctranslate2.Translator(cartella, device=dev, compute_type="auto")
                usato = dev
                break
            except Exception as e:
                ultimo = e
        if tr is None:
            raise RuntimeError("impossibile caricare il modello NLLB: {}".format(ultimo))
        _cache_nllb.update(tr=tr, sp=sp, dev=usato, repo=repo)

    tr, sp = _cache_nllb["tr"], _cache_nllb["sp"]
    fatte = 0
    for k in range(0, len(da_fare), batch):
        idx = da_fare[k:k + batch]
        sorgenti = [[src] + sp.encode(testi[i].replace("\n", " "), out_type=str) + ["</s>"] for i in idx]
        lim = min(256, 3 * max(len(x) for x in sorgenti) + 10)
        res = tr.translate_batch(
            sorgenti, target_prefix=[[tgt]] * len(idx), beam_size=5, max_decoding_length=lim)
        for i, r in zip(idx, res):
            toks = list(r.hypotheses[0])
            if toks and toks[0] == tgt:
                toks = toks[1:]
            risultati[i] = sp.decode(toks).strip() or None
        fatte += len(idx)
        if progresso:
            progresso(fatte, len(da_fare))
    return risultati


def _prompt_gemma(lingua, testo):
    """Prompt nel formato di TranslateGemma (giapponese -> lingua scelta)."""
    t, c = LINGUE_GEMMA[lingua]
    return (
        "You are a professional Japanese (ja) to {t} ({c}) translator. Your goal is to accurately "
        "convey the meaning and nuances of the original Japanese text while adhering to {t} grammar, "
        "vocabulary, and cultural sensitivities.\n"
        "Produce only the {t} translation, without any additional explanations or commentary. "
        "Please translate the following Japanese text into {t}:\n\n\n"
    ).format(t=t, c=c) + testo.replace("\n", " ")


def _prompt_plamo(lingua, testo):
    """Prompt nel formato di PLaMo-2-translate (giapponese -> lingua scelta).
    Il modello va interrogato con /v1/completions (non /chat/completions): il formato
    e' quello 'dataset translation' con i token speciali <|plamo:op|>."""
    t = LINGUE_GEMMA[lingua][0]
    return (
        "<|plamo:op|>dataset\ntranslation\n"
        "<|plamo:op|>input lang=Japanese\n{}\n"
        "<|plamo:op|>output lang={}\n"
    ).format(testo.replace("\n", " "), t)


# ---------------------------------------------------------------- llama-server (llama.cpp)
_LLAMA = {"proc": None, "chiave": None, "url": None, "logf": None, "logpath": None}


def ferma_llamaserver():
    """Termina il llama-server avviato dallo script (libera subito la VRAM)."""
    pr = _LLAMA.get("proc")
    if pr is not None and pr.poll() is None:
        try:
            pr.terminate()
            pr.wait(timeout=15)
        except Exception:
            try:
                pr.kill()
                pr.wait(timeout=5)
            except Exception:
                pass
    lf = _LLAMA.get("logf")
    if lf is not None:
        try:
            lf.close()
        except Exception:
            pass
    lp = _LLAMA.get("logpath")
    if lp:
        try:
            os.remove(lp)
        except OSError:
            pass
    _LLAMA.update(proc=None, chiave=None, url=None, logf=None, logpath=None)


atexit.register(ferma_llamaserver)


def _coda_log(path, n=12):
    try:
        with open(path, "rb") as fh:
            righe = fh.read().decode("utf-8", "replace").strip().splitlines()
    except OSError:
        return ""
    return "\n".join(righe[-n:])


def _risolvi_exe(percorso):
    """Accetta il file llama-server oppure una cartella che lo contiene (anche in
    sottocartelle: es. ~/.lmstudio/extensions/backends)."""
    percorso = (percorso or "").strip()
    if percorso and os.path.isdir(percorso):
        for root, dirs, files in os.walk(percorso):
            if root[len(percorso):].count(os.sep) > 5:
                dirs[:] = []
                continue
            for nome in ("llama-server", "llama-server.exe"):
                if nome in files:
                    return os.path.join(root, nome)
        raise RuntimeError("nessun llama-server trovato dentro {}".format(percorso))
    return percorso or shutil.which("llama-server") or ""


def _risolvi_gguf(percorso):
    """Accetta il file .gguf oppure la cartella del modello (sceglie il .gguf, escludendo
    mmproj; per i modelli spezzati in parti usa la prima)."""
    percorso = (percorso or "").strip()
    if not percorso or not os.path.isdir(percorso):
        return percorso
    trovati = []
    for root, dirs, files in os.walk(percorso):
        for f in sorted(files):
            if f.lower().endswith(".gguf") and "mmproj" not in f.lower():
                trovati.append(os.path.join(root, f))
    if not trovati:
        raise RuntimeError("nessun file .gguf dentro {}".format(percorso))
    if len(trovati) == 1:
        return trovati[0]
    prime = [f for f in trovati if "-00001-of-" in f]
    if len(prime) == 1:
        return prime[0]
    raise RuntimeError("piu' file .gguf in {}: scegli quello giusto con 'Sfoglia...':\n{}".format(
        percorso, "\n".join(os.path.basename(f) for f in trovati[:10])))


def _llama_avvia(cfg, stato=None):
    """Avvia (o riusa) llama-server con il GGUF scelto; restituisce l'URL base /v1."""
    import urllib.request
    import urllib.error
    exe = _risolvi_exe(cfg.get("exe"))
    if not exe or not os.path.isfile(exe):
        raise RuntimeError("llama-server non trovato: indica il percorso dell'eseguibile "
                           "(da https://github.com/ggml-org/llama.cpp/releases) oppure mettilo nel PATH.")
    gguf = _risolvi_gguf(cfg.get("gguf"))
    if not gguf or not os.path.isfile(gguf):
        raise RuntimeError("File .gguf non trovato: {}".format(gguf or "(vuoto)"))
    try:
        ctx = int(str(cfg.get("ctx") or "").strip() or 16384)
    except ValueError:
        ctx = 16384
    ngl = str(cfg.get("ngl") or "").strip()
    try:
        extra = shlex.split(cfg.get("extra") or "", posix=(os.name != "nt"))
    except ValueError as e:
        raise RuntimeError("argomenti extra non validi: {}".format(e))
    chiave = (exe, gguf, ctx, ngl, tuple(extra))

    pr = _LLAMA["proc"]
    if pr is not None and pr.poll() is None and _LLAMA["chiave"] == chiave:
        return _LLAMA["url"]
    ferma_llamaserver()

    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        porta = sk.getsockname()[1]
    cmd = [exe, "-m", gguf, "-c", str(ctx)]
    if ngl and ngl.lower() != "auto":
        cmd += ["-ngl", ngl]
    cmd += ["--host", "127.0.0.1", "--port", str(porta)] + extra
    logf = tempfile.NamedTemporaryFile(prefix="llama_server_", suffix=".log", delete=False)
    kw = {}
    if os.name != "nt":
        # le build incluse in LM Studio hanno le librerie (libllama, libggml-*) accanto all'eseguibile
        env = dict(os.environ)
        cartella = os.path.dirname(os.path.abspath(exe))
        dirs = [cartella]
        # le librerie CUDA (libcudart, libcublas) di LM Studio stanno in backends/vendor/<...>
        vendor = os.path.join(os.path.dirname(cartella), "vendor")
        if os.path.isdir(vendor):
            for v in sorted(os.listdir(vendor)):
                if os.path.isdir(os.path.join(vendor, v)) and (
                        "cuda12" not in cartella or "cuda12" in v):
                    dirs.append(os.path.join(vendor, v))
        if env.get("LD_LIBRARY_PATH"):
            dirs.append(env["LD_LIBRARY_PATH"])
        env["LD_LIBRARY_PATH"] = os.pathsep.join(dirs)
        kw["env"] = env
        kw["cwd"] = cartella
    if os.name == "nt":
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        pr = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL, **kw)
    except OSError as e:
        logf.close()
        os.remove(logf.name)
        raise RuntimeError("impossibile avviare llama-server: {}".format(e))
    _LLAMA.update(proc=pr, chiave=chiave, logf=logf, logpath=logf.name)

    nome = os.path.basename(gguf)
    t0 = time.time()
    ultimo = 0
    while True:
        if pr.poll() is not None:
            coda = _coda_log(logf.name)
            ferma_llamaserver()
            raise RuntimeError("llama-server si e' chiuso (codice {}) durante il caricamento di {}. "
                               "Ultime righe del log:\n{}".format(pr.returncode, nome, coda))
        try:
            with urllib.request.urlopen("http://127.0.0.1:{}/health".format(porta), timeout=2) as r:
                if r.status == 200:
                    break
        except (urllib.error.URLError, OSError):
            pass
        sec = int(time.time() - t0)
        if sec > 900:
            ferma_llamaserver()
            raise RuntimeError("llama-server non e' diventato pronto entro 15 minuti.")
        if stato and sec - ultimo >= 3:
            ultimo = sec
            stato("Carico {} in llama.cpp ({} s)...".format(nome, sec))
        time.sleep(0.5)
    _LLAMA["url"] = "http://127.0.0.1:{}/v1".format(porta)
    return _LLAMA["url"]


def _traduci_llamaserver(testi, lingua, cfg, progresso=None, stato=None):
    if lingua not in LINGUE_GEMMA:
        raise ValueError("lingua '{}' non supportata".format(lingua))
    if not any(testi):
        return ["" for _ in testi]
    cfg = dict(cfg, gguf=_risolvi_gguf(cfg.get("gguf")))
    url = _llama_avvia(cfg, stato)
    # il nome del file serve anche a riconoscere translategemma / plamo
    nome = os.path.splitext(os.path.basename(cfg["gguf"]))[0]
    return _traduci_lmstudio(testi, lingua, url, nome, progresso, stato,
                             contesto_completo=bool(cfg.get("contesto")), tutto=bool(cfg.get("tutto")),
                             veloce=bool(cfg.get("veloce")))


def _ollama_url():
    h = os.environ.get("OLLAMA_HOST", "").strip() or "127.0.0.1:11434"
    if not h.startswith("http"):
        h = "http://" + h
    return h.replace("0.0.0.0", "127.0.0.1").rstrip("/")


def _ollama_json(percorso, dati=None, timeout=300):
    """GET (dati=None) o POST JSON verso Ollama. Errori in messaggi comprensibili."""
    import urllib.request
    import urllib.error
    url = _ollama_url() + percorso
    req = urllib.request.Request(
        url, data=None if dati is None else json.dumps(dati).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            det = json.loads(e.read().decode("utf-8")).get("error", "")
        except Exception:
            det = ""
        raise RuntimeError("Ollama ha risposto con errore {}: {}".format(e.code, det or e.reason))
    except urllib.error.URLError as e:
        raise RuntimeError(
            "Ollama non raggiungibile su {} ({}). Installalo da https://ollama.com e assicurati "
            "che sia in esecuzione (comando:  ollama serve).".format(_ollama_url(), e.reason))


def _ollama_assicura_modello(nome, stato=None):
    """Se il modello non e' ancora presente lo scarica con Ollama (solo la prima volta)."""
    import urllib.request
    elenco = _ollama_json("/api/tags", timeout=20).get("models", [])
    presenti = {m.get("name", "") for m in elenco} | {m.get("model", "") for m in elenco}
    if nome in presenti or (":" not in nome and nome + ":latest" in presenti):
        return
    if stato:
        stato("Scarico il modello {} con Ollama (solo la prima volta, puo' richiedere molti minuti)...".format(nome))
    req = urllib.request.Request(
        _ollama_url() + "/api/pull",
        data=json.dumps({"model": nome, "name": nome, "stream": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for riga in r:
                if not riga.strip():
                    continue
                d = json.loads(riga.decode("utf-8"))
                if d.get("error"):
                    raise RuntimeError("download del modello {} non riuscito: {}".format(nome, d["error"]))
                if stato and d.get("total") and d.get("completed") is not None:
                    stato("Scarico {} con Ollama: {:.0f}%".format(nome, 100.0 * d["completed"] / d["total"]))
    except OSError as e:
        raise RuntimeError("download del modello {} non riuscito: {}".format(nome, e))


def _traduci_ollama(testi, lingua, nome_modello, progresso=None, stato=None):
    """Traduce dal giapponese con TranslateGemma tramite Ollama, una riga per richiesta
    (quindi l'allineamento con i tempi e' garantito). Nessun limite di richieste."""
    if lingua not in LINGUE_GEMMA:
        raise ValueError("lingua '{}' non supportata".format(lingua))
    tgt, tgt_cod = LINGUE_GEMMA[lingua]
    risultati = ["" if not t else None for t in testi]
    da_fare = [i for i, t in enumerate(testi) if t]
    if not da_fare:
        return risultati

    _ollama_assicura_modello(nome_modello, stato)


    for n, i in enumerate(da_fare, 1):
        if stato and n == 1:
            stato("Carico {} in Ollama (la prima richiesta puo' richiedere un po')...".format(nome_modello))
        r = _ollama_json("/api/generate", dict(
            model=nome_modello,
            prompt=_prompt_gemma(lingua, testi[i]),
            stream=False,
            options=dict(temperature=0.0, num_predict=400),
        ), timeout=600)
        out = " ".join((r.get("response") or "").split())
        risultati[i] = out or None
        if progresso:
            progresso(n, len(da_fare))
    return risultati


def _lm_base(url):
    u = (url or "").strip() or LM_URL_PREDEFINITO
    if not u.startswith("http"):
        u = "http://" + u
    u = u.rstrip("/")
    return u if u.endswith("/v1") else u + "/v1"


def _lm_json(base, percorso, dati=None, timeout=300):
    """GET (dati=None) o POST JSON verso un server OpenAI-compatibile (LM Studio)."""
    import urllib.request
    import urllib.error
    req = urllib.request.Request(
        base + percorso, data=None if dati is None else json.dumps(dati).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer lm-studio"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        det = ""
        try:
            d = json.loads(e.read().decode("utf-8")).get("error", "")
            det = d.get("message", "") if isinstance(d, dict) else str(d)
        except Exception:
            pass
        raise RuntimeError("LM Studio ha risposto con errore {}: {}".format(e.code, det or e.reason))
    except urllib.error.URLError as e:
        raise RuntimeError(
            "LM Studio non raggiungibile su {} ({}). Apri LM Studio, avvia il server locale "
            "(scheda Developer) e controlla l'indirizzo.".format(base, e.reason))
    except OSError as e:
        raise RuntimeError("LM Studio non ha risposto: {}".format(e))


_RE_GIAPP = re.compile("[\u3040-\u30ff\u4e00-\u9fff]")
_RE_NUM = re.compile(r"^\s*\[(\d+)\]\s*(.*)$")


def _pulisci_lm(out):
    out = re.sub(r"<think>.*?</think>", "", out or "", flags=re.S)
    out = " ".join(out.split())
    if len(out) > 1 and out[0] in '"\u201c\u300c\u00ab' and out[-1] in '"\u201d\u300d\u00bb':
        out = out[1:-1].strip()
    return out


_LM_ULTIMO = {"info": ""}

# Statistiche di scarto dell'ultima traduzione (azzerate da traduzione_segmenti). Nei metodi a gruppi
# ('un solo passaggio' e 'contesto completo') una riga generata dal modello e' SCARTATA se manca nel
# formato [N], e' vuota o e' rimasta in giapponese: viene allora rifatta una per una.
_STAT_TRAD = {}


def _stat_add(chiave, n=1):
    _STAT_TRAD[chiave] = _STAT_TRAD.get(chiave, 0) + n


def _stat_gruppo(n_righe, risposta_ok):
    _stat_add("gruppi")
    _stat_add("tot", n_righe)
    if not risposta_ok:
        _stat_add("gruppi_vuoti")


def _stat_scarto(pezzo):
    _stat_add("scartate")
    _stat_add("mancanti" if not pezzo else "giapponese")


def _stat_rifatta(out, lingua):
    ok = bool(out) and (lingua == "zh" or not _RE_GIAPP.search(out))
    _stat_add("rifatte_ok" if ok else "rifatte_ko")


def _report_scarti(p, testi, righe_tmp):
    """Testo per la barra di stato e riga nel CSV con le righe scartate dal modello."""
    st = _STAT_TRAD
    richieste = [i for i, t in enumerate(testi) if t]
    if not richieste:
        return ""
    zh = p.get("lingua") == "zh"
    senza = sum(1 for i in richieste if righe_tmp[i] is None)
    giapp = 0 if zh else sum(1 for i in richieste if righe_tmp[i] and _RE_GIAPP.search(righe_tmp[i]))
    if st.get("gruppi"):
        tot = max(1, st.get("tot", len(richieste)))
        scart = st.get("scartate", 0)
        msg = (" (righe AI scartate: {} su {} = {:.1f}% [{} mancanti o vuote, {} rimaste in giapponese]; "
               "rifatte bene {}, ancora non valide {})").format(
            scart, tot, 100.0 * scart / tot, st.get("mancanti", 0), st.get("giapponese", 0),
            st.get("rifatte_ok", 0), st.get("rifatte_ko", 0))
    else:
        msg = (" (questo metodo non scarta righe: {} senza risultato, {} ancora con caratteri giapponesi)"
               ).format(senza, giapp)
    try:
        nuovo = not os.path.isfile(SCARTI_LOG)
        with open(SCARTI_LOG, "a", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            if nuovo:
                w.writerow(["data", "file", "modello", "metodo", "lingua", "righe", "scartate", "mancanti_o_vuote",
                            "rimaste_in_giapponese", "rifatte_ok", "rifatte_ko", "gruppi", "gruppi_senza_risposta",
                            "finali_senza_risultato", "finali_con_giapponese"])
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), os.path.basename(str(p.get("file", ""))),
                        st.get("modello", ""), st.get("metodo", ""), p.get("lingua", ""), len(richieste),
                        st.get("scartate", 0) if st.get("gruppi") else "", st.get("mancanti", 0),
                        st.get("giapponese", 0), st.get("rifatte_ok", 0), st.get("rifatte_ko", 0),
                        st.get("gruppi", 0), st.get("gruppi_vuoti", 0), senza, giapp])
    except OSError:
        pass
    return msg
# i modelli Qwen3 & co. "pensano" prima di rispondere e il ragionamento consuma i token: lo si
# disattiva nel template di chat (llama-server lo supporta; chi non lo conosce lo ignora)
_SENZA_THINKING = dict(enable_thinking=False)


def _info_risposta(r):
    try:
        ch = r["choices"][0]
        rag = bool((ch.get("message") or {}).get("reasoning_content"))
        return "finish_reason={}{}".format(ch.get("finish_reason"), ", il modello ha solo 'pensato'" if rag else "")
    except Exception:
        return ""


def _lm_chat(base, modello, messaggi, max_tokens):
    r = _lm_json(base, "/chat/completions", dict(
        model=modello, messages=messaggi, temperature=0.0, max_tokens=max_tokens, stream=False,
        chat_template_kwargs=_SENZA_THINKING), timeout=900)
    _LM_ULTIMO["info"] = _info_risposta(r)
    try:
        out = r["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("risposta inattesa da LM Studio: {}".format(str(r)[:150]))
    return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()


def _lm_completion(base, modello, prompt, max_tokens):
    """Richiesta /v1/completions (prompt grezzo) usata da PLaMo-2-translate."""
    r = _lm_json(base, "/completions", dict(
        model=modello, prompt=prompt, temperature=0.1, max_tokens=max_tokens,
        stop=["<|plamo:op|>"], stream=False), timeout=600)
    try:
        out = r["choices"][0]["text"] or ""
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("risposta inattesa da LM Studio: {}".format(str(r)[:150]))
    return out.split("<|plamo:op|>")[0].strip()


def _leggi_numerate(testo, attesi):
    """Legge righe nel formato '[N] testo' (le righe di continuazione si accodano)."""
    trovate, ultimo = {}, None
    for riga in testo.splitlines():
        m = _RE_NUM.match(riga)
        if m:
            n = int(m.group(1))
            if n in attesi and n not in trovate:
                trovate[n] = m.group(2)
                ultimo = n
            else:
                ultimo = None
        elif ultimo is not None and riga.strip():
            trovate[ultimo] += " " + riga.strip()
    return trovate


def _traduci_lm_numerato(testi, lingua, base, modello, progresso=None, stato=None, tutto=False):
    """Un solo passaggio: ogni gruppo di righe numerate viene tradotto direttamente in righe
    numerate (meta' dei token rispetto al metodo a due passaggi). Il modello vede tutto il
    gruppo e le ultime righe gia' tradotte, quindi capisce soggetti e pronomi omessi.
    Le righe mancanti o rimaste in giapponese vengono rifatte una per una."""
    tgt = LINGUE_GEMMA[lingua][0]
    risultati = ["" if not t else None for t in testi]
    da_fare = [i for i, t in enumerate(testi) if t]

    gruppi, cur, car = [], [], 0
    for i in da_fare:
        lung = len(testi[i])
        if cur and not tutto and (len(cur) >= LM_CTX_RIGHE_1 or car + lung > LM_CTX_CAR_1):
            gruppi.append(cur)
            cur, car = [], 0
        cur.append(i)
        car += lung
    if cur:
        gruppi.append(cur)

    sistema = (
        "You are a professional Japanese to {t} translator of song lyrics, subtitles and dialogue. "
        "You receive numbered Japanese lines that belong to ONE continuous text; a sentence may be "
        "split across lines. Japanese omits subjects and pronouns, so use the whole text (and the "
        "previous part, if given) to work out who does what, and keep a consistent tone. Translate "
        "EACH line into {t}. Output exactly one line per input line in the format [N] translation, "
        "keep the same numbers and order, never merge or split lines, translate a fragment as a "
        "fragment, and output nothing else: no notes, no romanization."
    ).format(t=tgt)
    sistema_riga = (
        "You are a professional Japanese to {t} translator of song lyrics, subtitles and dialogue. "
        "You receive a Japanese text for context and ONE line from it. Reply with ONLY the {t} "
        "translation of that line: no quotes, no notes, no explanations."
    ).format(t=tgt)

    fatte, coda = 0, []
    for gn, g in enumerate(gruppi, 1):
        righe = [testi[i].replace("\n", " ") for i in g]
        car = sum(len(x) for x in righe)
        if stato:
            stato("LM Studio: traduco il gruppo {}/{} ({} righe)...".format(gn, len(gruppi), len(g)))
        utente = ""
        if coda:
            utente += ("Previous part of the same text, already translated (context only, do not "
                       "repeat it):\n" + "\n".join("{} => {}".format(a, b) for a, b in coda) + "\n\n")
        utente += "Translate these lines:\n" + "\n".join("[{}] {}".format(k, r) for k, r in enumerate(righe, 1))
        try:
            ris = _lm_chat(base, modello,
                           [dict(role="system", content=sistema), dict(role="user", content=utente)],
                           min(8000, 600 + 3 * car + 10 * len(g)))
        except RuntimeError:
            ris = ""
        trovate = _leggi_numerate(ris, set(range(1, len(g) + 1)))
        _stat_gruppo(len(g), bool(ris))

        da_rifare = []
        for k, i in enumerate(g, 1):
            pezzo = _pulisci_lm(trovate.get(k, ""))
            if pezzo and (lingua == "zh" or not _RE_GIAPP.search(pezzo)):
                risultati[i] = pezzo
            else:
                da_rifare.append(i)
                _stat_scarto(pezzo)
        if da_rifare and not trovate and not ris:
            raise RuntimeError("il modello non ha restituito la traduzione ({}). Se ha 'pensato' troppo, "
                               "aggiungi --reasoning-budget 0 agli argomenti extra o usa un modello "
                               "non-thinking".format(_LM_ULTIMO["info"]))
        for n, i in enumerate(da_rifare, 1):
            if stato:
                stato("LM Studio: rifaccio le righe non valide ({}/{}, gruppo {}/{})...".format(
                    n, len(da_rifare), gn, len(gruppi)))
            u = "Japanese text for context:\n{}\n\nTranslate only this line into {}:\n{}".format(
                "\n".join(righe), tgt, testi[i].replace("\n", " "))
            out = _pulisci_lm(_lm_chat(base, modello,
                                       [dict(role="system", content=sistema_riga), dict(role="user", content=u)], 400))
            risultati[i] = out or None
            _stat_rifatta(out, lingua)

        coda = [(righe[k], risultati[g[k]]) for k in range(max(0, len(g) - 3), len(g)) if risultati[g[k]]]
        fatte += len(g)
        if progresso:
            progresso(fatte, len(da_fare))
    return risultati


def _traduci_lm_contesto(testi, lingua, base, modello, progresso=None, stato=None, tutto=False):
    """Traduzione a contesto completo, in due passaggi per ogni gruppo di righe:
    1) il modello traduce TUTTO il testo come un discorso unico (il giapponese omette soggetti e
       pronomi: servono i versi vicini per capirli);
    2) gli si chiede di assegnare a ogni riga giapponese numerata la parte corrispondente di
       quella traduzione. Le righe mancanti, in giapponese o sospette vengono rifatte una per
       una usando la traduzione completa come riferimento: l'allineamento resta sempre per riga."""
    tgt = LINGUE_GEMMA[lingua][0]
    risultati = ["" if not t else None for t in testi]
    da_fare = [i for i, t in enumerate(testi) if t]

    gruppi, cur, car = [], [], 0
    for i in da_fare:
        lung = len(testi[i])
        if not tutto and cur and (len(cur) >= LM_CTX_RIGHE or car + lung > LM_CTX_CAR):
            gruppi.append(cur)
            cur, car = [], 0
        cur.append(i)
        car += lung
    if cur:
        gruppi.append(cur)

    sistema1 = (
        "You are a professional Japanese to {t} translator of song lyrics, subtitles and dialogue. "
        "Japanese is highly context dependent: it often omits subjects and pronouns, so use the whole "
        "text to work out who does what and to whom, and keep a consistent tone. Translate the whole "
        "text naturally and faithfully into {t}. Output ONLY the {t} translation, with no notes, "
        "no romanization and no explanations."
    ).format(t=tgt)
    sistema2 = (
        "You align a translation to its source lines. You receive a Japanese text split into numbered "
        "lines and its complete {t} translation. REPLACE each Japanese line with the part of the "
        "{t} translation that corresponds to it. Rules: reuse the wording of the given translation (only "
        "cut it into pieces; adjust punctuation or capitalization if needed); add nothing; keep the "
        "order; if the translation does not render some line, translate that line briefly in the same "
        "style. Output exactly one line per Japanese line in the format [N] text, and nothing else."
    ).format(t=tgt)
    sistema_riga = (
        "You are a professional Japanese to {t} translator of song lyrics, subtitles and dialogue. "
        "You receive a reference translation of the whole text and ONE Japanese line from it. Reply with "
        "ONLY the {t} translation of that line, consistent with the reference translation: no quotes, "
        "no notes, no explanations."
    ).format(t=tgt)

    fatte, coda = 0, []
    for gn, g in enumerate(gruppi, 1):
        righe = [testi[i].replace("\n", " ") for i in g]
        car = sum(len(x) for x in righe)

        # --- passo 1: traduzione libera di tutto il testo
        if stato:
            stato("LM Studio: traduco tutto il testo (gruppo {}/{})...".format(gn, len(gruppi)))
        utente = ""
        if coda:
            utente += ("Previous part of the same text, already translated (for continuity only, "
                       "do not repeat it):\n" + "\n".join("{} => {}".format(a, b) for a, b in coda) + "\n\n")
        utente += "Text to translate:\n" + "\n".join(righe)
        completa = _lm_chat(base, modello,
                            [dict(role="system", content=sistema1), dict(role="user", content=utente)],
                            min(16000, 600 + 3 * car))
        if not completa:
            raise RuntimeError("il modello non ha restituito la traduzione del testo ({}). Se ha 'pensato' troppo, "
                               "aggiungi --reasoning-budget 0 agli argomenti extra o usa un modello non-thinking".format(_LM_ULTIMO["info"]))

        # --- passo 2: assegna a ogni riga la sua parte
        if stato:
            stato("LM Studio: riallineo la traduzione alle righe (gruppo {}/{})...".format(gn, len(gruppi)))
        numerate = "\n".join("[{}] {}".format(k, r) for k, r in enumerate(righe, 1))
        utente2 = "Japanese lines:\n{}\n\n{} translation:\n{}\n\nOutput the aligned lines now.".format(
            numerate, tgt, completa)
        try:
            ris2 = _lm_chat(base, modello,
                            [dict(role="system", content=sistema2), dict(role="user", content=utente2)],
                            min(16000, 800 + 4 * car))
        except RuntimeError:
            ris2 = ""
        trovate = _leggi_numerate(ris2, set(range(1, len(g) + 1)))
        _stat_gruppo(len(g), bool(ris2))

        da_rifare = []
        for k, i in enumerate(g, 1):
            pezzo = _pulisci_lm(trovate.get(k, ""))
            if pezzo and (lingua == "zh" or not _RE_GIAPP.search(pezzo)):
                risultati[i] = pezzo
            else:
                da_rifare.append(i)
                _stat_scarto(pezzo)

        # --- righe non valide: una per una, con la traduzione completa come riferimento
        for i in da_rifare:
            if stato:
                stato("LM Studio: rifaccio {} righe non allineate (gruppo {}/{})...".format(
                    len(da_rifare), gn, len(gruppi)))
            u = "Reference translation of the whole text:\n{}\n\nTranslate only this Japanese line into {}:\n{}".format(
                completa, tgt, testi[i].replace("\n", " "))
            out = _pulisci_lm(_lm_chat(base, modello,
                                       [dict(role="system", content=sistema_riga), dict(role="user", content=u)], 400))
            risultati[i] = out or None
            _stat_rifatta(out, lingua)

        coda = [(righe[k], risultati[g[k]]) for k in range(max(0, len(g) - 3), len(g)) if risultati[g[k]]]
        fatte += len(g)
        if progresso:
            progresso(fatte, len(da_fare))
    return risultati


def _traduci_lmstudio(testi, lingua, url, modello, progresso=None, stato=None, contesto=2, contesto_completo=False, tutto=False, veloce=False):
    """Traduce dal giapponese con un modello servito da LM Studio (API OpenAI), una riga per
    richiesta: l'allineamento con i tempi e' garantito. Con i modelli generici si passano
    anche le righe vicine come contesto (da non tradurre), utile per i versi spezzati;
    con TranslateGemma si usa il suo prompt specifico, senza altro."""
    if lingua not in LINGUE_GEMMA:
        raise ValueError("lingua '{}' non supportata".format(lingua))
    tgt = LINGUE_GEMMA[lingua][0]
    risultati = ["" if not t else None for t in testi]
    da_fare = [i for i, t in enumerate(testi) if t]
    if not da_fare:
        return risultati

    base = _lm_base(url)
    modello = (modello or "").strip()
    if not modello:
        elenco = _lm_json(base, "/models", timeout=20).get("data", [])
        ids = [m["id"] for m in elenco if m.get("id") and "embed" not in m["id"].lower()]
        if not ids:
            raise RuntimeError("LM Studio non ha nessun modello disponibile: caricane uno "
                               "oppure scrivi il suo nome nel campo del modello.")
        modello = ids[0]
    _STAT_TRAD["modello"] = modello
    if stato:
        stato("Traduco con LM Studio ({})...".format(modello))
    gemma = "translategemma" in modello.lower()
    plamo = "plamo" in modello.lower()
    _STAT_TRAD["metodo"] = "riga per riga"
    if (contesto_completo or veloce) and not gemma and not plamo:
        if veloce:
            _STAT_TRAD["metodo"] = "un solo passaggio (numerato)"
            return _traduci_lm_numerato(testi, lingua, base, modello, progresso, stato, tutto=tutto)
        _STAT_TRAD["metodo"] = "contesto completo (2 passaggi)"
        return _traduci_lm_contesto(testi, lingua, base, modello, progresso, stato, tutto=tutto)

    sistema = (
        "You are a professional Japanese to {t} translator of song lyrics, subtitles and dialogue. "
        "You translate ONE line at a time, faithfully and naturally; the line may be a sentence "
        "fragment, so keep its meaning and tone. Reply with ONLY the {t} translation of the requested "
        "line: no quotes, no notes, no romanization, no explanations, and never translate the context lines."
    ).format(t=tgt)

    for n, i in enumerate(da_fare, 1):
        riga = testi[i].replace("\n", " ")
        if plamo:
            out = _pulisci_lm(_lm_completion(base, modello, _prompt_plamo(lingua, riga), 400))
            risultati[i] = out or None
            if progresso:
                progresso(n, len(da_fare))
            continue
        if gemma:
            messaggi = [dict(role="user", content=_prompt_gemma(lingua, riga))]
        else:
            prec = [testi[j].replace("\n", " ") for j in range(max(0, i - contesto), i) if testi[j]]
            succ = [testi[j].replace("\n", " ") for j in range(i + 1, min(len(testi), i + 1 + contesto)) if testi[j]]
            utente = ""
            if prec:
                utente += "Previous lines (context only, do not translate):\n" + "\n".join(prec) + "\n\n"
            if succ:
                utente += "Next lines (context only, do not translate):\n" + "\n".join(succ) + "\n\n"
            utente += "Translate this line into {}:\n{}".format(tgt, riga)
            messaggi = [dict(role="system", content=sistema), dict(role="user", content=utente)]
        r = _lm_json(base, "/chat/completions", dict(
            model=modello, messages=messaggi, temperature=0.0, max_tokens=400, stream=False,
            chat_template_kwargs=_SENZA_THINKING), timeout=600)
        try:
            out = r["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise RuntimeError("risposta inattesa da LM Studio: {}".format(str(r)[:150]))
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.S)
        out = " ".join(out.split())
        if len(out) > 1 and out[0] in '"\u201c\u300c\u00ab' and out[-1] in '"\u201d\u300d\u00bb':
            out = out[1:-1].strip()
        risultati[i] = out or None
        if progresso:
            progresso(n, len(da_fare))
    return risultati


def traduci_locale(testi, lingua, device="auto", progresso=None, stato=None, modello=None, lm=None):
    """Traduzione locale dal giapponese. 'modello' e' una voce di MODELLI_TRAD
    (nome, tipo, id): NLLB (CTranslate2) oppure TranslateGemma (Ollama).
    Restituisce una lista lunga come 'testi' ("" per le righe vuote, None se senza risultato)."""
    nome, tipo, ident = modello or MODELLI_TRAD[0]
    _STAT_TRAD["modello"] = ident or nome
    _STAT_TRAD["metodo"] = {"nllb": "NLLB", "ollama": "riga per riga (Ollama)"}.get(tipo, tipo)
    if tipo == "llamaserver":
        return _traduci_llamaserver(testi, lingua, lm or {}, progresso, stato)
    if tipo == "lmstudio":
        lm = lm or {}
        return _traduci_lmstudio(testi, lingua, lm.get("url", ""), lm.get("modello", ""), progresso, stato,
                                 contesto_completo=bool(lm.get("contesto")), tutto=bool(lm.get("tutto")),
                                 veloce=bool(lm.get("veloce")))
    if tipo == "ollama":
        return _traduci_ollama(testi, lingua, ident, progresso, stato)
    return _traduci_nllb(testi, lingua, device, progresso, stato, repo=ident)


_STOP = threading.Event()        # impostato dal pulsante "Interrompi": i cicli lunghi lo controllano


class Interrotto(Exception):
    """Operazione interrotta dall'utente."""


def controlla_stop():
    if _STOP.is_set():
        raise Interrotto()


_RE_TS_AI = re.compile(
    r"\[\s*\d+\s*[|\uff5c:/]\s*[\d:.,]+\s*\]"                      # [N|mm:ss.s]
    r"|\[\s*[\d:.,]+\s*(?:->|-->|\u2192)\s*[\d:.,]+\s*\]")       # [mm:ss.s -> mm:ss.s]


def pulisci_per_ai(t):
    """Toglie timestamp e numeri di segmento dal testo prima di darlo al modello di traduzione."""
    return _RE_TS_AI.sub("", t or "").strip()


def _progresso_trad(p, stato):
    def f(i, n):
        controlla_stop()
        if stato:
            stato("Passaggio 2/2: traduco le liriche in '{}' ({}/{})...".format(p["lingua"], i, n))
    return f


def traduzione_segmenti(p, testi, tt0, tt1, stato=None):
    """Traduce i segmenti giapponesi con il modello scelto in 'p' (impostazioni del Worker).
    Restituisce (trad, avviso): 'trad' e' un dict(righe, t0, t1) con una riga per segmento,
    oppure None se la traduzione non e' riuscita; 'avviso' spiega eventuali problemi."""
    avviso, righe_tmp = "", None
    _STAT_TRAD.clear()
    testi = [pulisci_per_ai(t) for t in testi]
    try:
        righe_tmp = traduci_locale(
            testi, p["lingua"], p["device"],
            modello=MODELLI_TRAD[min(max(int(p.get("trad_modello", 0)), 0), len(MODELLI_TRAD) - 1)],
            lm=dict(url=p.get("lm_url", ""), modello=p.get("lm_modello", ""), contesto=p.get("lm_contesto", True),
                    tutto=p.get("lm_tutto", False), veloce=p.get("lm_veloce", True),
                    exe=p.get("ls_exe", ""), gguf=p.get("ls_gguf", ""), ctx=p.get("ls_ctx", ""),
                    ngl=p.get("ls_ngl", ""), extra=p.get("ls_extra", "")),
            progresso=_progresso_trad(p, stato),
            stato=stato)
    except Interrotto:
        raise
    except ImportError as e:
        avviso += (" (traduzione locale non disponibile: manca {}. "
                   "Installa:  python -m pip install sentencepiece)").format(e.name or "un modulo")
    except Exception as e:
        avviso += " (traduzione locale non riuscita: {})".format(str(e).strip()[:600])
    finally:
        if not p.get("batch"):
            ferma_llamaserver()

    trad = None
    if righe_tmp is not None:
        richieste = [i for i, t in enumerate(testi) if t]
        ko = [i for i in richieste if righe_tmp[i] is None]
        if richieste and len(ko) < len(richieste):
            trad = dict(righe=[r if r is not None else testi[i] for i, r in enumerate(righe_tmp)], t0=tt0, t1=tt1)
            if ko:
                avviso += " (traduzione: {} righe su {} senza risultato, lasciate in giapponese)".format(
                    len(ko), len(richieste))
        else:
            avviso += " (traduzione NON riuscita: il modello non ha restituito testo)"
        avviso += _report_scarti(p, testi, righe_tmp)
    return trad, avviso


# ---------------------------------------------------------------------------
# Thread: decodifica audio + calcolo livelli in dB
# ---------------------------------------------------------------------------
class Decoder(QThread):
    pronto = pyqtSignal(str, str, object, float, float, bool)  # file, wav, db, hop, durata, ha_video
    errore = pyqtSignal(str, str)

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.info = None          # dimensioni/fps con cui mostrare il video (se c'e')

    def run(self):
        wav = None
        t0 = time.perf_counter()
        log_hira("decodifica: avvio ffmpeg su %s" % self.path)
        try:
            wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            subprocess.run(
                ["ffmpeg", "-y", "-nostdin", "-v", "error", "-i", self.path, "-vn",
                 "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", wav],
                check=True, stdin=subprocess.DEVNULL, capture_output=True)
            log_hira("decodifica: ffmpeg finito in %.1f s" % (time.perf_counter() - t0))

            with wave.open(wav, "rb") as w:
                raw = w.readframes(w.getnframes())

            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            win = SR // 100                      # finestre da 10 ms
            m = len(x) // win

            if m == 0:
                raise ValueError("Audio vuoto o non leggibile (il file ha una traccia audio?).")

            rms = np.sqrt(np.mean(x[:m * win].reshape(m, win) ** 2, axis=1))
            db = np.clip(20.0 * np.log10(rms + 1e-9), -60.0, 0.0)

            ha_video = ha_flusso_video(self.path)
            self.info = info_video(self.path) if ha_video else None
            log_hira("decodifica: finita in %.1f s (durata %.1f s, traccia video: %s)"
                     % (time.perf_counter() - t0, len(x) / float(SR), ha_video))
            self.pronto.emit(self.path, wav, db, 0.01, len(x) / float(SR), ha_video)

        except FileNotFoundError:
            log_hira("decodifica: ffmpeg non trovato")
            self.errore.emit(self.path, "ffmpeg non trovato: installalo e aggiungilo al PATH.")
        except subprocess.CalledProcessError as e:
            msg = (e.stderr or b"").decode(errors="replace") or str(e)
            log_hira("decodifica: ffmpeg ha fallito: " + msg[-500:])
            self.errore.emit(self.path, msg)
        except BaseException as e:
            import traceback
            log_hira("decodifica: errore imprevisto: %r\n%s" % (e, traceback.format_exc()))
            self.errore.emit(self.path, str(e) or repr(e))


# ---------------------------------------------------------------------------
# Thread: trascrizione faster-whisper
# ---------------------------------------------------------------------------
class Worker(QThread):
    stato = pyqtSignal(str)
    finito = pyqtSignal(object)
    errore = pyqtSignal(str)

    def __init__(self, p):
        super().__init__()
        self.p = p

    def _device_attempts(self, requested):
        """Elenco di (device, compute_type) da provare."""
        attempts = []

        if requested in ("auto", "cuda"):
            attempts.extend([
                ("cuda", "int8_float16"),
                ("cuda", "float16"),
                ("cuda", "int8"),
            ])

        if requested in ("auto", "cpu"):
            attempts.append(("cpu", "int8"))

        if requested == "cuda":
            attempts.append(("cpu", "int8"))

        return attempts

    def _load_model(self):
        p = self.p

        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "faster-whisper non installato. Installalo con:\n"
                "python -m pip install faster-whisper"
            ) from e

        last_error = None

        for device, compute_type in self._device_attempts(p["device"]):
            key = (p["modello"], device, compute_type)

            if _cache["key"] == key and _cache["model"] is not None:
                self.stato.emit(
                    f"Uso modello già caricato: {p['modello']} su {device} ({compute_type})."
                )
                return _cache["model"], device, compute_type

            try:
                self.stato.emit(
                    f"Carico faster-whisper {p['modello']} su {device} ({compute_type})..."
                )

                if _cache["model"] is not None:
                    libera_whisper()

                model = WhisperModel(
                    p["modello"],
                    device=device,
                    compute_type=compute_type,
                )

                _cache["key"] = key
                _cache["model"] = model

                return model, device, compute_type

            except Exception as e:
                msg = str(e)
                if "libcublas" in msg or "cannot be loaded" in msg:
                    _aggiungi_path_librerie_cuda()
                last_error = e
                self.stato.emit(f"Tentativo {device}/{compute_type} fallito: {e}")
                continue

        msg_err = str(last_error)
        guida = ""
        if "libcublas" in msg_err or "cudnn" in msg_err.lower() or "cannot be loaded" in msg_err:
            if os.name == "nt":
                guida = (
                    "\n\nLibrerie CUDA mancanti o non trovate. Soluzioni:\n"
                    "1) Installa le librerie nel venv con pip:\n"
                    "   python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12\n"
                    "2) Oppure installa il CUDA Toolkit da developer.nvidia.com e "
                    "assicurati che la cartella ...\\CUDA\\v12.x\\bin sia nel PATH.\n"
                    "3) Riavvia il programma dopo aver modificato il PATH.\n"
                    "4) In alternativa scegli 'cpu' come dispositivo."
                )
            else:
                guida = (
                    "\n\nLibrerie CUDA 12 mancanti o non trovate (Linux). Soluzioni:\n"
                    "1) Installa le librerie nel venv con pip:\n"
                    "   python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12\n"
                    "   (il programma aggiunge automaticamente i loro percorsi a\n"
                    "   LD_LIBRARY_PATH all'avvio)\n"
                    "2) Oppure installa il CUDA Toolkit 12 di sistema, ad es. su\n"
                    "   Linux Mint/Ubuntu:  apt install libcublas-12-6 libcudnn9-cuda-12\n"
                    "   poi verifica con:  ldconfig -p | grep cublas\n"
                    "3) Controlla che la GPU sia visibile:  nvidia-smi\n"
                    "4) In alternativa scegli 'cpu' come dispositivo."
                )
        raise RuntimeError(
            f"Impossibile caricare faster-whisper {p['modello']}: {msg_err}{guida}"
        )

    def _isola_voce(self, path, audio_orig=None):
        """Separa voce e musica con il motore scelto (Demucs htdemucs oppure BS-RoFormer
        tramite audio-separator), in un'unica passata: il modello produce entrambe le
        tracce, quindi non ha senso lanciarlo due volte. Restituisce
        (wav_voce, wav_musica, cartella_temporanea, messaggio_errore); i due wav sono
        None se fallisce.

        Se 'audio_orig' (il file originale scelto dall'utente) viene passato insieme alle
        tracce ottenute dall'intero brano, le due tracce vengono salvate direttamente
        accanto all'originale come <nome>_voce.wav e <nome>_musica.wav: cosi' 'trascrivi'
        popola subito la cache degli stem, senza dover poi rilanciare la separazione per
        'Salva voce e musica' o per l'esportazione .mka.
        Richiede:  python -m pip install demucs
              o:   python -m pip install "audio-separator[gpu]"   (oppure [cpu])"""
        motore = self.p.get("separatore") or "demucs"
        d = tempfile.mkdtemp(prefix="hira_sep_")
        try:
            src = os.path.join(d, "audio.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", path, "-vn", "-ar", "44100", "-ac", "2", src],
                check=True, stdin=subprocess.DEVNULL, capture_output=True)

            ultimo = ""
            for dev in (["cpu"] if self.p["device"] == "cpu" else ["cuda", "cpu"]):
                voce = musica = None
                if motore == "bsroformer":
                    os.makedirs(SEPARATOR_MODELS_DIR, exist_ok=True)
                    env = dict(os.environ)
                    if dev == "cpu":
                        env["CUDA_VISIBLE_DEVICES"] = "-1"     # audio-separator sceglie da solo la GPU
                    # niente --single_stem: il modello genera Vocals + Instrumental in un colpo
                    # solo, salvare anche la musica adesso evita di separare di nuovo dopo
                    r = subprocess.run(
                        [sys.executable, "-c",
                         "from audio_separator.utils.cli import main; main()",
                         src, "--model_filename", BSROFORMER_MODEL,
                         "--model_file_dir", SEPARATOR_MODELS_DIR,
                         "--output_dir", d, "--output_format", "WAV"],
                        stdin=subprocess.DEVNULL, capture_output=True, env=env)
                    voce = _trova_stem_bsroformer(d, src, "(vocals)")
                    musica = _trova_stem_bsroformer(d, src, "(instrumental)")
                else:
                    r = subprocess.run(
                        [sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs",
                         "-d", dev, "-o", d, src],
                        stdin=subprocess.DEVNULL, capture_output=True)
                    cart = os.path.join(d, "htdemucs", "audio")
                    voce = os.path.join(cart, "vocals.wav")
                    musica = os.path.join(cart, "no_vocals.wav")
                if r.returncode == 0 and voce and os.path.isfile(voce):
                    if musica and not os.path.isfile(musica):
                        musica = None
                    if audio_orig:
                        # salvataggio diretto dei 2 stem accanto all'originale (una volta sola)
                        base = os.path.splitext(audio_orig)[0]
                        try:
                            shutil.copyfile(voce, base + "_voce.wav")
                            _scrivi_stem_meta(base + "_voce.wav", motore, dev)
                            if musica:
                                shutil.copyfile(musica, base + "_musica.wav")
                                _scrivi_stem_meta(base + "_musica.wav", motore, dev)
                        except OSError:
                            pass
                    return voce, musica, d, ""
                ultimo = (r.stderr or b"").decode(errors="replace").strip()[-300:]
                if "No module named" in ultimo:
                    ultimo += ("  ->  python -m pip install " +
                               ('"audio-separator[gpu]"' if motore == "bsroformer" else "demucs"))
                    break      # inutile riprovare su CPU: manca proprio il pacchetto
            return None, None, d, ultimo or "errore sconosciuto"
        except FileNotFoundError:
            return None, None, d, "ffmpeg non trovato"
        except subprocess.CalledProcessError as e:
            return None, None, d, (e.stderr or b"").decode(errors="replace").strip()[-300:]

    def run(self):
        tmp = None
        dir_demucs = None
        try:
            p = self.p
            path = p["file"]

            if p["inizio"] > 0 or p["fine"] > 0:
                self.stato.emit("Estraggo il segmento con ffmpeg...")
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                cmd = ["ffmpeg", "-y", "-v", "error", "-ss", str(p["inizio"]), "-i", path]

                if p["fine"] > p["inizio"]:
                    cmd += ["-t", str(p["fine"] - p["inizio"])]

                cmd += ["-ar", "16000", "-ac", "1", tmp]
                subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL, capture_output=True)
                path = tmp

            # Musica di sottofondo: Demucs / BS-RoFormer isola la voce, poi si trascrive solo quella.
            # La separazione si fa UNA VOLTA SOLA e produce entrambi gli stem: voce e musica
            # vengono salvati subito accanto all'audio (<nome>_voce.wav / <nome>_musica.wav),
            # cosi' 'Salva voce e musica' e l'export .mka li trovano gia' pronti senza rilanciare
            # il modello. Se le tracce esistono gia' (e sono state create con lo stesso
            # separatore) si riutilizzano, senza rifare la separazione da capo.
            voce_isolata = False
            voce_persistente = None
            musica_persistente = None
            avviso_pre = ""
            if p.get("demucs"):
                motore = p.get("separatore") or "demucs"
                nome_sep = nome_separatore(motore)
                riuso = p.get("riusa_voce", True)
                dest_voce = os.path.splitext(p["file"])[0] + "_voce.wav"
                voce_cache = cerca_voce_pulita(p["file"]) if riuso else None
                se_esiste = bool(voce_cache) and not _nuovo_stem(dest_voce, motore)
                musica_cache = cerca_musica_separata(p["file"]) if riuso else None
                brano_intero = p["inizio"] <= 0 and p["fine"] <= 0

                def _sep(sorg):
                    """Separa voce+musica in un colpo solo; salva i 2 stem accanto all'originale."""
                    nonlocal dir_demucs
                    self.stato.emit("Isolo voce e musica con {} (una volta sola, puo' richiedere "
                                    "alcuni minuti)...".format(nome_sep))
                    v, m, d, err = self._isola_voce(
                        sorg, audio_orig=p["file"] if brano_intero else None)
                    if v:
                        dir_demucs = d
                    return v, m, err

                if se_esiste and brano_intero:
                    self.stato.emit("Uso la voce gia' isolata: {}".format(os.path.basename(voce_cache)))
                    path = voce_cache
                    voce_isolata = True
                    voce_persistente = voce_cache       # niente copia: il file e' gia' quello giusto
                    if not musica_cache:
                        # manca solo la musica: la si ottiene dalla stessa separazione completa
                        # (che riscrive anche la voce, identica) senza rifarla due volte
                        _, m, _ = _sep(p["file"])
                        musica_persistente = m
                elif se_esiste:
                    # voce pulita dell'intero brano ma la trascrizione e' di un segmento: si ritaglia
                    # la voce salvata sul segmento richiesto, senza rifare la separazione da capo
                    self.stato.emit("Ritaglio la voce gia' isolata sul segmento scelto...")
                    tmp2 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                    try:
                        subprocess.run(
                            ["ffmpeg", "-y", "-v", "error", "-ss", str(p["inizio"]), "-i", voce_cache]
                            + (["-t", str(p["fine"] - p["inizio"])] if p["fine"] > p["inizio"] else [])
                            + ["-ar", "44100", "-ac", "2", tmp2],
                            check=True, stdin=subprocess.DEVNULL, capture_output=True)
                        path = tmp2
                        if tmp and os.path.exists(tmp):
                            try:
                                os.remove(tmp)
                            except OSError:
                                pass
                        tmp = tmp2          # il ritaglio va ripulito alla fine
                        voce_isolata = True
                        if not musica_cache:
                            # gli stem si salvano una volta sola: ricavo la musica dall'originale
                            # (la voce e' gia' a destino: qui '_isola_voce' non la sovrascrive)
                            _, musica_persistente, _ = self._isola_voce(p["file"], audio_orig=p["file"])
                    except Exception:
                        # ritaglio fallito: si torna indietro e si separa l'originale come una volta
                        if os.path.exists(tmp2):
                            try:
                                os.remove(tmp2)
                            except OSError:
                                pass
                        voce, m, err = _sep(path)
                        if voce:
                            path = voce
                            voce_isolata = True
                            musica_persistente = m
                        else:
                            avviso_pre = " ({} non utilizzabile, uso l'audio originale: {})".format(nome_sep, err)
                else:
                    voce, m, err = _sep(path)
                    if voce:
                        path = voce
                        voce_isolata = True
                        musica_persistente = m
                    else:
                        avviso_pre = " ({} non utilizzabile, uso l'audio originale: {})".format(nome_sep, err)

            model, device, compute_type = self._load_model()

            self.stato.emit("Trascrivo con faster-whisper{}...".format(
                " (passaggio 1/2: giapponese)" if p.get("doppio") and p["lingua"] not in (None, "ja") else ""))

            use_words = bool(p["parole"])

            # Lingua diversa dal giapponese + "doppio passaggio": il testo giapponese
            # (kanji + hiragana, con tempi per parola) si ricava con un passaggio in
            # giapponese; il testo nella lingua scelta (es. italiano) con un secondo.
            doppio = bool(p.get("doppio")) and p["lingua"] not in (None, "ja")
            lingua_jp = "ja" if doppio else p["lingua"]

            # Musica (anche con voce isolata): il filtro VAD e le soglie "no speech" / compressione
            # scartano spesso i testi cantati, quindi si rilassano; con la voce isolata il VAD resta
            # attivo ma piu' sensibile, con la musica ancora presente viene disattivato.
            musica = bool(p.get("musica")) or bool(p.get("demucs"))

            def _opzioni(lang, con_prompt, musica_forzata=None, task="transcribe"):
                musica_eff = musica if musica_forzata is None else bool(musica_forzata)
                o = dict(
                    language=lang,
                    task=task,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                    condition_on_previous_text=False,
                    initial_prompt=(p["prompt_iniziale"] or None) if con_prompt else None,
                    temperature=0.0,
                    compression_ratio_threshold=2.0,
                )
                if musica_eff:
                    o.update(
                        no_speech_threshold=None,
                        compression_ratio_threshold=3.0,
                        temperature=[0.0, 0.2, 0.4],
                    )
                    if voce_isolata:
                        o["vad_parameters"] = dict(threshold=0.3, min_silence_duration_ms=700, speech_pad_ms=400)
                    else:
                        o["vad_filter"] = False
                        o.pop("vad_parameters")
                return o

            def _consume(lang, word_timestamps, con_prompt=True, musica_forzata=None, task="transcribe"):
                segments_iter, info = model.transcribe(
                    path,
                    word_timestamps=word_timestamps,
                    **_opzioni(lang, con_prompt, musica_forzata, task)
                )
                out = []
                for sg in segments_iter:
                    controlla_stop()
                    out.append(sg)
                return out, info

            raw_segments = None
            info = None

            try:
                raw_segments, info = _consume(lingua_jp, use_words)
            except Interrotto:
                raise
            except Exception as e:
                if use_words:
                    self.stato.emit(
                        "Word timestamps non disponibili: uso sincronizzazione per segmento."
                    )
                    use_words = False
                    raw_segments, info = _consume(lingua_jp, False)
                else:
                    raise

            lingua = lingua_jp or getattr(info, "language", "") or ""

            self.stato.emit("Converto in hiragana e calcolo la sincronizzazione...")

            avviso = avviso_pre
            try:
                get_kakasi()
                kk_ok = True
            except ImportError:
                kk_ok = False
                avviso = " (pykakasi mancante: hiragana non convertito)"

            off = p["inizio"]

            def hira_di(t):
                if not kk_ok:
                    return t
                try:
                    return "".join(i["hira"] for i in get_kakasi().convert(t))
                except Exception:
                    return t

            segmenti = []
            precedente = None

            for seg in raw_segments:
                testo = (getattr(seg, "text", "") or "").strip()

                if not testo or testo == precedente:
                    continue

                precedente = testo

                s0 = float(getattr(seg, "start", 0.0)) + off
                s1 = float(getattr(seg, "end", s0)) + off

                toks = []
                words = getattr(seg, "words", None)

                if use_words and words:
                    for i, w in enumerate(words):
                        t = (getattr(w, "word", "") or "")

                        if i == 0:
                            t = t.lstrip()
                        if i == len(words) - 1:
                            t = t.rstrip()

                        if not t:
                            continue

                        ws = float(getattr(w, "start", s0)) + off
                        we = float(getattr(w, "end", s1)) + off

                        toks.append((t, hira_di(t), ws, we))

                elif kk_ok:
                    toks = distribuisci(testo, s0, s1)
                else:
                    toks = [(testo, testo, s0, s1)]

                segmenti.append((s0, s1, toks))

            prev = 0.0
            for n, (s0, s1, toks) in enumerate(segmenti):
                nuovi = []
                for o, h, a, b in toks:
                    a = max(a, prev)
                    b = max(b, a)
                    prev = a
                    nuovi.append((o, h, a, b))
                segmenti[n] = (s0, s1, nuovi)

            trad = None

            # Fuori dal batch Whisper non serve piu' dopo la trascrizione: lo scarico subito dalla
            # memoria (VRAM compresa), cosi' resta spazio per il modello di traduzione (LM Studio,
            # Ollama...) e per il resto del sistema. Nel batch resta caricato per non ricaricarlo a ogni
            # file: lo scarica _batch_fine alla fine.
            if not p.get("batch") and p.get("scarica_whisper", True):
                prima = _vram_usata_mb() if device == "cuda" else None
                model = None  # anche il riferimento locale usato da _consume
                libera_whisper()
                if prima is not None:
                    import time as _t
                    _t.sleep(0.5)
                    dopo = _vram_usata_mb()
                    if dopo is not None:
                        if prima - dopo >= 300:
                            avviso += " (VRAM: {:.1f} -> {:.1f} GB dopo aver scaricato Whisper)".format(
                                prima / 1024.0, dopo / 1024.0)
                        else:
                            avviso += (" (VRAM non rilasciata da Whisper: {:.1f} -> {:.1f} GB; CTranslate2 puo' "
                                       "tenerla finche' il programma e' aperto: se il server locale non riesce a "
                                       "caricare il modello, riavvia hira.py)").format(prima / 1024.0, dopo / 1024.0)

            da_tradurre = None
            if doppio and segmenti:
                testi_da_tradurre = ["".join(tok[0] for tok in toks).strip() for _, _, toks in segmenti]
                tt0 = [s[0] for s in segmenti]
                tt1 = [s[1] for s in segmenti]

                if p.get("solo_trascrizione"):
                    # batch a due fasi: le traduzioni si fanno dopo, per tutti i file insieme
                    da_tradurre = dict(testi=testi_da_tradurre, t0=tt0, t1=tt1)
                else:
                    self.stato.emit("Passaggio 2/2: traduco le liriche in '{}'...".format(p["lingua"]))
                    # una riga per segmento: i risultati tornano nello stesso ordine, quindi
                    # restano allineati ai tempi
                    trad, av = traduzione_segmenti(p, testi_da_tradurre, tt0, tt1, self.stato.emit)
                    avviso += av

                    # verso l'inglese Whisper sa tradurre da solo: ripiego se il resto non va
                    if trad is None and p["lingua"] == "en":
                        self.stato.emit("Traduzione locale non disponibile: uso Whisper translate verso l'inglese...")
                        try:
                            if model is None:
                                model, device, compute_type = self._load_model()
                            raw_tr, _ = _consume("en", False, False, musica_forzata=musica, task="translate")
                            righe, tt0_w, tt1_w = [], [], []
                            prec_t, prev_t = None, 0.0
                            for seg in raw_tr:
                                t = (getattr(seg, "text", "") or "").strip()
                                if not t or t == prec_t:
                                    continue
                                prec_t = t
                                a_ = max(float(getattr(seg, "start", 0.0)) + off, prev_t)
                                b_ = max(float(getattr(seg, "end", a_)) + off, a_)
                                prev_t = a_
                                righe.append(t.replace("\n", " "))
                                tt0_w.append(a_)
                                tt1_w.append(b_)
                            if righe:
                                trad = dict(righe=righe, t0=tt0_w, t1=tt1_w)
                                avviso += " (traduzione fatta da Whisper)"
                        except Exception as e:
                            avviso += " (anche Whisper translate e' fallito: {})".format(e)

            sep = "\n" if (p["timestamp"] or p["a_capo"]) else ("" if lingua in ("ja", "zh") else " ")

            po, ph = [], []
            pos_o = pos_h = 0
            span_o, span_h, tempi = [], [], []
            tok_o, tok_h, tok_seg = [], [], []

            if p["prefisso"] and not p["timestamp"]:
                po.append(PREFISSO_COSYVOICE)
                ph.append(PREFISSO_COSYVOICE)
                pos_o = pos_h = l16(PREFISSO_COSYVOICE)

            for n, (s0, s1, toks) in enumerate(segmenti):
                if n > 0 and sep:
                    po.append(sep)
                    ph.append(sep)
                    pos_o += l16(sep)
                    pos_h += l16(sep)

                if p["timestamp"]:
                    et = "[{} -> {}] ".format(fmt_tempo(s0), fmt_tempo(s1))
                    po.append(et)
                    ph.append(et)
                    pos_o += l16(et)
                    pos_h += l16(et)

                for o, h, a, b in toks:
                    po.append(o)
                    ph.append(h)

                    span_o.append((pos_o, pos_o + l16(o)))
                    span_h.append((pos_h, pos_h + l16(h)))

                    pos_o += l16(o)
                    pos_h += l16(h)

                    tempi.append((a, b))
                    tok_o.append(o)
                    tok_h.append(h)
                    tok_seg.append(n)

            self.finito.emit(dict(
                orig="".join(po),
                hira="".join(ph),
                span_o=span_o,
                span_h=span_h,
                t0=[t[0] for t in tempi],
                t1=[t[1] for t in tempi],
                avviso=avviso,
                da_tradurre=da_tradurre,
                tok=dict(o=tok_o, h=tok_h, seg=tok_seg),
                trad=allinea_traduzione([t[0] for t in tempi], [t[1] for t in tempi], trad),
                # tracce pulite prodotte/salvate da questa trascrizione: la voce (e la musica)
                # sono gia' accanto all'audio come <nome>_voce.wav / <nome>_musica.wav, percio'
                # 'Salva voce e musica' e l'export .mka non devono rifare la separazione
                voce_file=(voce_persistente if voce_persistente else
                           (path if (voce_isolata and dir_demucs) else None)),
                voce_temporanea=bool(voce_isolata and dir_demucs),
                musica_file=musica_persistente,
            ))

        except FileNotFoundError:
            self.errore.emit("ffmpeg non trovato: installalo e aggiungilo al PATH.")

        except subprocess.CalledProcessError as e:
            self.errore.emit((e.stderr or b"").decode(errors="replace") or str(e))

        except Interrotto:
            self.errore.emit("__interrotto__")

        except Exception as e:
            msg = str(e)

            if "out of memory" in msg.lower():
                msg += (
                    "\nVRAM insufficiente: chiudi altri programmi che usano la GPU, "
                    "scegli un modello più piccolo oppure usa CPU."
                )

            if "cuda" in msg.lower() or "cudnn" in msg.lower():
                msg += (
                    "\nErrore CUDA/cuDNN con faster-whisper/CTranslate2. "
                    "Prova device=cpu, oppure installa le librerie CUDA/cuDNN compatibili."
                )

            self.errore.emit(msg)

        finally:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            if dir_demucs:
                shutil.rmtree(dir_demucs, ignore_errors=True)


# ---------------------------------------------------------------------------
# Widget: forma d'onda (dB su Y, tempo su X)
# ---------------------------------------------------------------------------
def etichetta_tempo(t, step):
    if step < 1:
        tot, d = divmod(int(round(max(t, 0) * 10)), 10)
        s_txt = ":{:02d}.{}".format(tot % 60, d)
    else:
        tot = int(round(max(t, 0)))
        s_txt = ":{:02d}".format(tot % 60)

    m_tot = tot // 60
    if m_tot >= 60:
        return "{}:{:02d}{}".format(m_tot // 60, m_tot % 60, s_txt)

    return "{}{}".format(m_tot, s_txt)


class Forma(QWidget):
    seek = pyqtSignal(float)

    DB_MIN = -60.0
    DB_TICKS = (0, -12, -24, -36, -48, -60)
    STEPS = (0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(130)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(
            "Clic/trascina: posizione  |  Rotella: zoom  |  "
            "Maiusc+rotella: scorri  |  Doppio clic: reset zoom"
        )

        self.db = None
        self.n = 0
        self.hop = 0.01
        self.durata = 0.0
        self.pos = 0.0
        self.v0, self.v1 = 0.0, 1.0
        self.sel = (0.0, 0.0)
        self.msg = "Carica un file per vedere la forma d'onda"
        self._drag = False

    def pulisci(self, msg):
        self.db = None
        self.n = 0
        self.durata = 0.0
        self.pos = 0.0
        self.msg = msg
        self.update()

    def imposta(self, db, hop, durata):
        self.n = len(db)
        self.db = np.append(db.astype(np.float32), np.float32(self.DB_MIN))
        self.hop = hop
        self.durata = durata
        self.pos = 0.0
        self.v0, self.v1 = 0.0, max(durata, 0.1)
        self.update()

    def imposta_selezione(self, a, b):
        self.sel = (a, b)
        self.update()

    def imposta_pos(self, t):
        self.pos = t
        span = self.v1 - self.v0

        if not self._drag and self.durata and span < self.durata - 1e-6 and (t > self.v1 or t < self.v0):
            self.v0 = t - 0.1 * span
            self.v1 = self.v0 + span
            self._limita_vista()

        self.update()

    def _plot(self):
        return QRectF(58, 12, max(10, self.width() - 58 - 14), max(10, self.height() - 12 - 28))

    def _x(self, t):
        r = self._plot()
        return r.left() + (t - self.v0) / max(self.v1 - self.v0, 1e-9) * r.width()

    def _t(self, x):
        r = self._plot()
        return self.v0 + (x - r.left()) / r.width() * (self.v1 - self.v0)

    def _limita_vista(self):
        span = self.v1 - self.v0
        self.v0 = max(0.0, min(self.v0, self.durata - span))
        self.v1 = self.v0 + span

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        p.setPen(QPen(QColor(C_BORDER)))
        p.setBrush(QColor("#12131a"))
        p.drawRoundedRect(QRectF(0.5, 0.5, self.width() - 1, self.height() - 1), 10, 10)

        r = self._plot()

        f = QFont(self.font())
        f.setPointSizeF(8.5)
        p.setFont(f)

        for d in self.DB_TICKS:
            y = r.top() + (-d) / 60.0 * r.height()
            p.setPen(QPen(QColor("#262a3b"), 1))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setPen(QColor(C_MUTED))
            p.drawText(QRectF(0, y - 8, r.left() - 8, 16), Qt.AlignRight | Qt.AlignVCenter, "{} dB".format(d))

        if self.db is None:
            p.setPen(QColor(C_MUTED))
            p.drawText(r, Qt.AlignCenter, self.msg)
            return

        span = self.v1 - self.v0
        step = self.STEPS[-1]

        for s in self.STEPS:
            if s / span * r.width() >= 90:
                step = s
                break

        t = ceil(self.v0 / step) * step

        for _i in range(300):
            if t > self.v1 + 1e-9:
                break

            x = self._x(t)
            p.setPen(QPen(QColor("#262a3b"), 1))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.setPen(QColor(C_MUTED))
            p.drawText(QRectF(x - 32, r.bottom() + 6, 64, 16), Qt.AlignCenter, etichetta_tempo(t, step))
            t += step

        a, b = self.sel
        if b > a:
            xa, xb = self._x(a), self._x(b)
            p.fillRect(
                QRectF(
                    max(xa, r.left()),
                    r.top(),
                    max(0, min(xb, r.right()) - max(xa, r.left())),
                    r.height()
                ),
                QColor(255, 255, 255, 20)
            )

        w = int(r.width())
        e = np.linspace(self.v0 / self.hop, self.v1 / self.hop, w + 1).astype(np.int64)
        e = np.clip(e, 0, self.n)

        idx = np.empty(2 * w, dtype=np.int64)
        idx[0::2] = e[:-1]
        idx[1::2] = e[1:]

        val = np.maximum.reduceat(self.db, idx)[0::2]
        ys = r.top() + (-np.clip(val, self.DB_MIN, 0.0)) / 60.0 * r.height()

        poly = QPolygonF()
        poly.append(QPointF(r.left(), r.bottom()))

        for i, y in enumerate(ys):
            poly.append(QPointF(r.left() + i + 0.5, float(y)))

        poly.append(QPointF(r.left() + w, r.bottom()))

        xp = self._x(self.pos)

        p.save()
        p.setPen(Qt.NoPen)
        p.setClipRect(r)

        c = QColor(C_ACCENT)
        c.setAlpha(210)
        p.setBrush(c)
        p.drawPolygon(poly)

        p.setClipRect(QRectF(r.left(), r.top(), min(r.width(), max(0.0, xp - r.left())), r.height()))

        c = QColor(C_SPOKEN)
        c.setAlpha(235)
        p.setBrush(c)
        p.drawPolygon(poly)

        p.restore()

        if self.v0 <= self.pos <= self.v1:
            p.setPen(QPen(QColor("#ffffff"), 1.6))
            p.drawLine(QPointF(xp, r.top()), QPointF(xp, r.bottom()))

            tri = QPolygonF([
                QPointF(xp - 5, r.top() - 1),
                QPointF(xp + 5, r.top() - 1),
                QPointF(xp, r.top() + 6)
            ])

            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#ffffff"))
            p.drawPolygon(tri)

    def _seek_x(self, x):
        t = min(max(self._t(x), 0.0), self.durata)
        self.seek.emit(t)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.db is not None:
            self._drag = True
            self._seek_x(e.x())

    def mouseMoveEvent(self, e):
        if self._drag:
            self._seek_x(e.x())

    def mouseReleaseEvent(self, e):
        self._drag = False

    def mouseDoubleClickEvent(self, e):
        if self.db is not None:
            self.v0, self.v1 = 0.0, max(self.durata, 0.1)
            self.update()

    def wheelEvent(self, e):
        if self.db is None:
            return

        dy = e.angleDelta().y() or e.angleDelta().x()
        span = self.v1 - self.v0

        if e.modifiers() & Qt.ShiftModifier:
            self.v0 += span * 0.15 * (-1 if dy > 0 else 1)
            self.v1 = self.v0 + span
        else:
            nuovo = min(max(span * (0.8 if dy > 0 else 1.25), min(1.0, self.durata)), max(self.durata, 0.1))
            tm = self._t(e.x())
            ratio = (tm - self.v0) / span
            self.v0 = tm - ratio * nuovo
            self.v1 = self.v0 + nuovo

        self._limita_vista()
        self.update()


# ---------------------------------------------------------------------------
# Casella di testo con doppio clic + evidenziatore "karaoke"
# ---------------------------------------------------------------------------
class CasellaTesto(QTextEdit):
    doppio_clic = pyqtSignal(int)

    def mouseDoubleClickEvent(self, e):
        super().mouseDoubleClickEvent(e)
        self.doppio_clic.emit(self.cursorForPosition(e.pos()).position())


class Evidenziatore:
    def __init__(self, editor):
        self.ed = editor
        self.spans = []
        self.ultimo = None

    @property
    def attivo(self):
        return bool(self.spans)

    def imposta(self, spans):
        self.spans = list(spans)
        self.ultimo = None
        self.ed.setExtraSelections([])

    def invalida(self):
        self.spans = []
        self.ultimo = None
        self.ed.setExtraSelections([])

    def aggiorna(self, idx, in_corso):
        if not self.spans:
            return

        stato = (idx, in_corso)
        if stato == self.ultimo:
            return

        self.ultimo = stato
        sels = []

        if idx >= 0:
            idx = min(idx, len(self.spans) - 1)

            fine_detto = self.spans[idx][1]

            if fine_detto > 0:
                s = QTextEdit.ExtraSelection()
                c = QTextCursor(self.ed.document())
                c.setPosition(0)
                c.setPosition(fine_detto, QTextCursor.KeepAnchor)
                s.cursor = c
                s.format.setForeground(QBrush(QColor(C_SPOKEN)))
                sels.append(s)

            if in_corso:
                self._scorri(self.spans[idx][0])

        self.ed.setExtraSelections(sels)

    def aggiorna_frazione(self, idx, f):
        """Come aggiorna(), ma la riga idx e' illuminata solo per la frazione f (0..1)."""
        if not self.spans:
            return

        fine = 0
        a = 0
        if idx >= 0:
            idx = min(idx, len(self.spans) - 1)
            a, b = self.spans[idx]
            fine = b if f >= 1.0 else a + int(round((b - a) * max(0.0, f)))

        stato = (idx, fine)
        if stato == self.ultimo:
            return

        self.ultimo = stato
        sels = []

        if fine > 0:
            s = QTextEdit.ExtraSelection()
            c = QTextCursor(self.ed.document())
            c.setPosition(0)
            c.setPosition(fine, QTextCursor.KeepAnchor)
            s.cursor = c
            s.format.setForeground(QBrush(QColor(C_SPOKEN)))
            sels.append(s)

        if idx >= 0 and f < 1.0:
            self._scorri(a)

        self.ed.setExtraSelections(sels)

    def _scorri(self, pos):
        c = QTextCursor(self.ed.document())
        c.setPosition(pos)
        r = self.ed.cursorRect(c)
        vh = self.ed.viewport().height()

        if r.top() < 20 or r.bottom() > vh - 20:
            sb = self.ed.verticalScrollBar()
            sb.setValue(sb.value() + r.top() - vh // 3)


# ---------------------------------------------------------------------------
# Finestra principale
# ---------------------------------------------------------------------------
class SliderPos(QSlider):
    """Barra di posizione: clic e trascinamento saltano subito al punto."""
    seek = pyqtSignal(float)

    def __init__(self):
        super().__init__(Qt.Horizontal)
        self.trascina = False

    def _vai(self, x):
        v = QStyle.sliderValueFromPosition(self.minimum(), self.maximum(),
                                            int(x), max(1, self.width()))
        self.setValue(v)
        self.seek.emit(v / 1000.0)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.trascina = True
            self._vai(e.x())

    def mouseMoveEvent(self, e):
        if self.trascina:
            self._vai(e.x())

    def mouseReleaseEvent(self, e):
        self.trascina = False


class OndaDB:
    """Livelli in dB condivisi tra lo schermo principale e quello a schermo intero: servono a disegnare,
    nei file solo audio, la sinusoide la cui ampiezza segue il volume nel tempo."""

    def __init__(self):
        self.attiva = False      # True: file senza video, schermo nero con sinusoide
        self.db = None
        self.amp = None
        self.times = None
        self.t = 0.0

    def imposta(self, db, hop):
        db = np.asarray(db, dtype=np.float32)
        if len(db) == 0:
            return self.pulisci()
        lo = Forma.DB_MIN
        hi = max(float(np.percentile(db, 99.5)), lo + 12.0)
        amp = np.clip((db - lo) / (hi - lo), 0.0, 1.0)
        amp = np.convolve(amp, np.array([1, 2, 3, 2, 1], dtype=np.float32) / 9.0, mode="same")
        self.db = db
        self.amp = amp.astype(np.float32)
        self.times = np.arange(len(db), dtype=np.float32) * float(hop)
        self.t = 0.0

    def pulisci(self):
        self.db = self.amp = self.times = None
        self.t = 0.0


class VideoSchermo(QGraphicsView):
    """Visualizzatore video: mostra i fotogrammi decodificati da ffmpeg (SorgenteFrame) in un QGraphicsPixmapItem."""
    doppio_clic = pyqtSignal()

    def __init__(self, sorgente=None):
        super().__init__()
        self.setFocusPolicy(Qt.NoFocus)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor("#000000")))
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setRenderHint(QPainter.SmoothPixmapTransform)

        if sorgente is not None:
            self._scena = sorgente._scena
            self.item = sorgente.item
        else:
            self._scena = QGraphicsScene(self)
            self.item = QGraphicsPixmapItem()
            self.item.setTransformationMode(Qt.SmoothTransformation)
            self._scena.addItem(self.item)
        self.setScene(self._scena)
        self.onda = sorgente.onda if sorgente is not None else OndaDB()
        self._sub_html = ""
        self._sub_cache = None

    def _adatta(self, *_):
        br = self.item.boundingRect()
        if br.isValid() and not br.isEmpty():
            self._scena.setSceneRect(br)
            self.fitInView(self.item, Qt.KeepAspectRatio)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._adatta()

    def mouseDoubleClickEvent(self, e):
        self.doppio_clic.emit()

    def imposta_sub(self, html):
        self._sub_html = html
        self._sub_cache = None
        self.viewport().update()

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

    def _disegna_onda(self):
        """Schermo nero dei file solo audio: sinusoide che scorre, ampiezza = livello in dB.
        Si vede solo la parte prima della testina (gia' riprodotta), arancione-rossa; il futuro e' nascosto."""
        o = self.onda
        w, h = self.viewport().width(), self.viewport().height()
        if w < 60 or h < 60:
            return
        p = QPainter(self.viewport())
        p.setRenderHint(QPainter.Antialiasing)
        cx, cy = w * WAVE_HEAD, h * 0.30
        amp_px = h * 0.18
        lw = max(1.5, h * 0.006)
        p.setPen(QPen(QColor(255, 255, 255, 28), 1))
        p.drawLine(QPointF(w * 0.04, cy), QPointF(w * 0.96, cy))
        if o.amp is None:
            p.end()
            return

        t = o.t
        period = WAVE_SPAN / WAVE_CYCLES
        step = max(2, int(w / 360))
        xs = np.arange(0, w + 1, step, dtype=np.float64)
        u = np.where(xs < cx, (xs - cx) / (cx * 0.96), (xs - cx) / max(1.0, (w - cx) * 0.96))
        taper = np.clip(1.0 - u ** 4, 0.0, 1.0)          # i bordi sfumano
        tx = t + (xs - cx) / w * WAVE_SPAN
        a = np.interp(tx, o.times, o.amp, left=0.0, right=0.0)
        ys = cy - a * amp_px * taper * np.sin(2.0 * np.pi * tx / period)

        path = QPainterPath()
        path.moveTo(float(xs[0]), float(ys[0]))
        for x, y in zip(xs[1:], ys[1:]):
            path.lineTo(float(x), float(y))

        past = QLinearGradient(cx - cx * 0.80, 0, cx, 0)
        past.setColorAt(0.0, QColor(255, 180, 84, 40))
        past.setColorAt(0.6, QColor(C_SPOKEN))
        past.setColorAt(1.0, QColor(C_ROSSO))
        # la parte non ancora riprodotta (a destra della testina) non viene disegnata
        for clip, grad in ((QRectF(0, 0, cx, h), past),):
            p.save()
            p.setClipRect(clip)
            p.setOpacity(0.18)
            p.setPen(QPen(QBrush(grad), lw * 4.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawPath(path)
            p.setOpacity(1.0)
            p.setPen(QPen(QBrush(grad), lw, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawPath(path)
            p.restore()

        # testina: linea verticale, punto sulla sinusoide e livello in dB
        a0 = float(np.interp(t, o.times, o.amp, left=0.0, right=0.0))
        y0 = cy - a0 * amp_px * float(np.sin(2.0 * np.pi * t / period))
        p.setPen(QPen(QColor(255, 255, 255, 70), 1))
        p.drawLine(QPointF(cx, cy - amp_px * 1.1), QPointF(cx, cy + amp_px * 1.1))
        r = max(3.0, h * 0.014)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 42, 42, 90))
        p.drawEllipse(QPointF(cx, y0), r * 1.9, r * 1.9)
        p.setBrush(QColor(C_ROSSO))
        p.drawEllipse(QPointF(cx, y0), r, r)

        i = int(min(len(o.db) - 1, max(0, round(t / max(float(o.times[1]) if len(o.times) > 1 else 0.01, 1e-6)))))
        val = float(o.db[i])
        txt = "< -60 dB" if val <= Forma.DB_MIN + 0.5 else "%d dB" % round(val)
        f = QFont(self.font())
        f.setPixelSize(max(11, int(h * 0.034)))
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(255, 255, 255, 150))
        p.drawText(QRectF(cx - 70, cy - amp_px * 1.1 - h * 0.06, 140, h * 0.05), Qt.AlignCenter, txt)
        p.end()

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.onda.attiva:
            self._disegna_onda()
        if not self._sub_html:
            return

        w, h = self.viewport().width(), self.viewport().height()
        px = max(16, min(int(h * 0.055), 64))
        larg = int(w * 0.92)
        chiave = (self._sub_html, larg, px)

        if self._sub_cache is None or self._sub_cache[0] != chiave:
            nero = re.sub(r"color:\s*#[0-9a-fA-F]{6}", "color:#000000", self._sub_html)
            self._sub_cache = (chiave, self._doc(self._sub_html, larg, px),
                               self._doc(nero, larg, px))

        _, doc, doc_nero = self._sub_cache
        x = (w - larg) / 2.0
        y = h - doc.size().height() - h * 0.05
        r = max(2, px // 12)

        p = QPainter(self.viewport())
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
        p.end()


class ContenitoreVideo(QWidget):
    """Video con i sottotitoli karaoke sovrapposti (contorno nero)."""
    doppio_clic = pyqtSignal()
    esci = pyqtSignal()
    play_pausa = pyqtSignal()
    salta = pyqtSignal(float)

    def __init__(self, sorgente=None):
        super().__init__()
        self._html = ""
        self.setFocusPolicy(Qt.StrongFocus)
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor("#000000"))
        self.setPalette(pal)
        self.setAutoFillBackground(True)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.video = VideoSchermo(sorgente)
        self.video.setMinimumHeight(300)
        self.video.doppio_clic.connect(self.doppio_clic)
        lay.addWidget(self.video, 1)

        self._righe = 4

    def imposta_righe(self, n):
        self._righe = n

    def imposta_sub(self, html):
        self._html = html
        self.video.imposta_sub(html)

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key_Escape:
            self.esci.emit()
        elif k == Qt.Key_F:
            self.doppio_clic.emit()
        elif k == Qt.Key_Space:
            self.play_pausa.emit()
        elif k == Qt.Key_Left:
            self.salta.emit(-5.0)
        elif k == Qt.Key_Right:
            self.salta.emit(5.0)
        else:
            super().keyPressEvent(e)

    def closeEvent(self, e):
        e.ignore()
        self.esci.emit()


def card(titolo=None):
    f = QFrame()
    f.setObjectName("card")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(16, 14, 16, 14)
    lay.setSpacing(10)

    if titolo:
        lb = QLabel(titolo.upper())
        lb.setObjectName("sez")
        lay.addWidget(lb)

    return f, lay


class TraduttoreBatch(QThread):
    """Seconda fase del batch: traduce, uno dopo l'altro, i segmenti di tutti i file gia'
    trascritti. Whisper e' gia' stato scaricato dalla memoria; il modello di traduzione
    (LM Studio, Ollama, NLLB) resta caricato per tutta la fase."""
    stato = pyqtSignal(str)
    inizio = pyqtSignal(int)
    fatto = pyqtSignal(int, object, str)   # indice, trad (o None), avviso
    finito = pyqtSignal()

    def __init__(self, p, lavori):
        super().__init__()
        self.p = p
        self.lavori = lavori               # lista di (testi, t0, t1)
        self.fermo = False

    def run(self):
        try:
            for k, (testi, t0, t1) in enumerate(self.lavori):
                if self.fermo:
                    break
                self.inizio.emit(k)
                try:
                    trad, av = traduzione_segmenti(self.p, testi, t0, t1, self.stato.emit)
                except Interrotto:
                    trad, av = None, " (interrotta)"
                    self.fermo = True
                except Exception as e:
                    trad, av = None, " (traduzione non riuscita: {})".format(str(e).strip()[:200])
                self.fatto.emit(k, trad, av)
        finally:
            ferma_llamaserver()
            self.finito.emit()


class Finestra(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("finestra")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setWindowTitle("Trascrivi Audio/Video (faster-whisper)")
        scr = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1180, scr.width() - 40), min(920, scr.height() - 90))
        self.setAcceptDrops(True)

        self.worker = None
        self.decoder = None
        self._batch_attivo = False
        self._batch_stop = False
        self._batch_i = -1
        self._batch_corrente = ""
        self._batch_due_fasi = False
        self._batch_fase = 1
        self._batch_lavori = []
        self._batch_trad = None
        self._trad_solo = None
        self._stem = None
        self._mka = None
        self._batch_params = None
        self._batch_interrotto1 = False
        self._batch_esiti = dict(ok=0, salt=0, vuoti=[], err=[], avvisi=[], note=[])
        self._vecchi = []
        self.wav_tmp = None
        self.file_corrente = ""
        self.modo_video = False
        self.onda = None
        self._onda_ancora = None
        self._onda_ultimo = 0.0
        self._fs = False
        self.cont_fs = None
        self.ultimo_ris = None
        self.tok_sub = None
        self.seg_range = {}
        self._sub_chiave = None
        self.t0 = []
        self.t1 = []
        self.t0_t = []
        self.t1_t = []
        self.trad_righe = []
        self._cw = [0.0]
        self._prog = False

        self.player = QMediaPlayer(None, QMediaPlayer.LowLatency)
        self.player.setVolume(80)

        self._tm_stop = QTimer(self)
        self._tm_stop.setInterval(300)
        self._tm_stop.timeout.connect(self._aggiorna_stop)
        self._tm_stop.start()

        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._tick)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        inner.setObjectName("interno")
        inner.setAttribute(Qt.WA_StyledBackground, True)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        root = QVBoxLayout(inner)
        self.root = root
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        # --- intestazione ---
        h = QVBoxLayout()
        h.setSpacing(0)

        t = QLabel("Trascrivi Audio / Video")
        t.setObjectName("titolo")

        s = QLabel("faster-whisper · Hiragana · riproduzione con evidenziazione sincronizzata")
        s.setObjectName("sotto")

        h.addWidget(t)
        h.addWidget(s)
        root.addLayout(h)

        # --- sorgente + impostazioni ---
        c1, l1 = card("Sorgente e impostazioni")

        riga = QHBoxLayout()
        self.ed_file = QLineEdit()
        self.ed_file.setPlaceholderText("Scegli o trascina qui un file audio o video...")
        self.ed_file.editingFinished.connect(self._file_modificato)

        b_sfoglia = QPushButton("Sfoglia...")
        b_sfoglia.clicked.connect(self.scegli_file)

        riga.addWidget(self.ed_file, 1)
        riga.addWidget(b_sfoglia)
        l1.addLayout(riga)

        g = QGridLayout()
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(8)

        self.cb_modello = QComboBox()
        self.cb_modello.addItems(MODELLI)
        self.cb_modello.setCurrentText("large-v2")

        self.cb_lingua = QComboBox()
        for nome, _ in LINGUE:
            self.cb_lingua.addItem(nome)

        self.cb_device = QComboBox()
        self.cb_device.addItems(["auto", "cuda", "cpu"])

        self.sp_inizio = QDoubleSpinBox()
        self.sp_inizio.setRange(0, 100000)
        self.sp_inizio.setSuffix(" s")
        self.sp_inizio.setDecimals(1)

        self.sp_fine = QDoubleSpinBox()
        self.sp_fine.setRange(0, 100000)
        self.sp_fine.setSuffix(" s")
        self.sp_fine.setDecimals(1)
        self.sp_fine.setSpecialValueText("fino alla fine")

        self.sp_inizio.valueChanged.connect(self._aggiorna_selezione)
        self.sp_fine.valueChanged.connect(self._aggiorna_selezione)

        g.addWidget(QLabel("Modello"), 0, 0)
        g.addWidget(self.cb_modello, 0, 1)
        g.addWidget(QLabel("Lingua"), 0, 2)
        g.addWidget(self.cb_lingua, 0, 3)
        g.addWidget(QLabel("Dispositivo"), 0, 4)
        g.addWidget(self.cb_device, 0, 5)

        g.addWidget(QLabel("Da"), 1, 0)
        g.addWidget(self.sp_inizio, 1, 1)
        g.addWidget(QLabel("A"), 1, 2)
        g.addWidget(self.sp_fine, 1, 3)

        g.setColumnStretch(1, 1)
        g.setColumnStretch(3, 1)
        g.setColumnStretch(5, 1)

        l1.addLayout(g)

        self.ck_timestamp = QCheckBox("Mostra i timestamp")
        self.ck_acapo = QCheckBox("Una frase per riga")
        self.ck_parole = QCheckBox("Sincronizzazione per parola (piu' precisa, un po' piu' lenta)")
        self.ck_parole.setChecked(True)
        self.ck_prefisso = QCheckBox("Aggiungi il prefisso CosyVoice3 (You are a helpful assistant.<|endofprompt|>)")

        riga2 = QHBoxLayout()
        riga2.setSpacing(20)
        riga2.addWidget(self.ck_timestamp)
        riga2.addWidget(self.ck_acapo)
        riga2.addWidget(self.ck_parole)
        riga2.addStretch(1)
        l1.addLayout(riga2)

        l1.addWidget(self.ck_prefisso)

        self.ck_doppio = QCheckBox(
            "Se la lingua non e' il giapponese: ricava anche kanji + hiragana (2 passaggi, piu' lento)")
        self.ck_doppio.setChecked(True)
        self.ck_doppio.setToolTip(
            "1° passaggio: whisper in giapponese (kanji, hiragana e tempi per parola)\n"
            "2° passaggio: traduzione diretta dei segmenti giapponesi nella lingua scelta\n"
            "(traduzione locale con NLLB-200 / CTranslate2: il modello locale si scarica la prima volta;\n"
            "se non va e la lingua e' l'inglese, whisper task='translate')")
        l1.addWidget(self.ck_doppio)

        rt = QHBoxLayout()
        self.lb_trad = QLabel("Modello di traduzione")
        self.cb_trad = QComboBox()
        for nome, _, _ in MODELLI_TRAD:
            self.cb_trad.addItem(nome)
        self.cb_trad.setCurrentIndex(4)
        self.cb_trad.setToolTip(
            "Traduzione locale (nessun limite di richieste). NLLB: modelli generici, si scaricano da soli.\n"
            "TranslateGemma: modello specializzato nella traduzione, richiede Ollama installato e in\n"
            "esecuzione (https://ollama.com); se manca, il modello viene scaricato da Ollama al primo uso.\n"
            "LM Studio: usa il server locale di LM Studio con il modello che hai caricato (indirizzo sotto).")
        rt.addWidget(self.lb_trad)
        rt.addWidget(self.cb_trad, 1)
        rt.addStretch(2)
        l1.addLayout(rt)

        rl = QHBoxLayout()
        self.lb_lm = QLabel("LM Studio")
        self.ed_lm_url = QLineEdit(LM_URL_PREDEFINITO)
        self.ed_lm_url.setToolTip("Indirizzo del server locale di LM Studio (API compatibile con OpenAI)")
        self.ed_lm_modello = QLineEdit()
        self.ed_lm_modello.setPlaceholderText("nome del modello (vuoto = il primo disponibile)  [solo LM Studio]")
        self.ed_lm_modello.setToolTip(
            "Nome del modello come appare in LM Studio. Lasciando vuoto si usa il primo modello\n"
            "disponibile sul server. Se contiene 'translategemma' viene usato il prompt di TranslateGemma,\n"
            "se contiene 'plamo' il formato di PLaMo-2-translate (/v1/completions),\n"
            "altrimenti un prompt generico con le righe vicine come contesto.")
        rl.addWidget(self.lb_lm)
        rl.addWidget(self.ed_lm_url, 2)
        rl.addWidget(self.ed_lm_modello, 2)
        l1.addLayout(rl)
        self.ck_lm_ctx = QCheckBox("LM Studio: traduci prima tutto il testo, poi riallinea riga per riga (piu' contesto)")
        self.ck_lm_ctx.setChecked(True)
        self.ck_lm_ctx.setToolTip(
            "Il giapponese omette soggetti e pronomi: il modello traduce prima tutto il testo come un\n"
            "discorso unico e poi assegna a ogni riga la sua parte. Le righe che non tornano vengono\n"
            "rifatte una per una. Non si applica ai modelli TranslateGemma e PLaMo (prompt specifico).")
        l1.addWidget(self.ck_lm_ctx)
        self.ck_lm_tutto = QCheckBox("LM Studio: manda tutto il file in un solo blocco (serve un contesto ampio, es. Bonsai)")
        self.ck_lm_tutto.setChecked(False)
        self.ck_lm_tutto.setToolTip(
            "Invece di gruppi di ~30 righe, il modello riceve l'intero testo: traduce tutto d'un fiato e\n"
            "poi sostituisce ogni riga giapponese con la corrispondente frase tradotta. Richiede che nel\n"
            "modello caricato in LM Studio il 'Context Length' sia abbastanza grande per tutto il testo.")
        l1.addWidget(self.ck_lm_tutto)
        self.ck_lm_veloce = QCheckBox("LM Studio / llama.cpp: un solo passaggio (piu' veloce: traduce direttamente le righe numerate)")
        self.ck_lm_veloce.setChecked(True)
        self.ck_lm_veloce.setToolTip(
            "Salta il passaggio 'traduci tutto, poi riallinea': il modello traduce subito gruppi di righe\n"
            "numerate vedendo anche le ultime righe gia' tradotte. Dimezza i tempi e bastano ~8K di\n"
            "contesto. Se e' spuntato 'tutto il file in un solo blocco' il gruppo e' l'intero testo.")
        l1.addWidget(self.ck_lm_veloce)
        self.ck_scarica_whisper = QCheckBox("Scarica Whisper dalla VRAM prima della traduzione")
        self.ck_scarica_whisper.setChecked(True)
        self.ck_scarica_whisper.setToolTip(
            "Libera la VRAM occupata da Whisper dopo la trascrizione (utile se il modello di traduzione\n"
            "ne ha bisogno). Togli la spunta se hai VRAM in abbondanza: Whisper resta caricato e la\n"
            "trascrizione successiva parte subito. Il pulsante 'Libera modello' lo scarica a mano.")
        l1.addWidget(self.ck_scarica_whisper)

        # --- llama.cpp: llama-server avviato dallo script solo per la traduzione
        ra = QHBoxLayout()
        self.lb_ls = QLabel("llama.cpp")
        self.ed_ls_exe = QLineEdit()
        self.ed_ls_exe.setPlaceholderText("llama-server (vuoto = cerca nel PATH)")
        self.b_ls_exe = QPushButton("Sfoglia...")
        self.b_ls_exe.clicked.connect(self._sfoglia_ls_exe)
        ra.addWidget(self.lb_ls)
        ra.addWidget(self.ed_ls_exe, 1)
        ra.addWidget(self.b_ls_exe)
        l1.addLayout(ra)
        rb = QHBoxLayout()
        self.ed_ls_gguf = QLineEdit()
        self.ed_ls_gguf.setPlaceholderText("modello .gguf (es. Bonsai o TranslateGemma)")
        self.ed_ls_gguf.setToolTip(
            "Percorso del file .gguf (o della cartella che lo contiene). Lo script non cerca i modelli da solo:\n"
            "usa questo percorso. 'Sfoglia...' apre il selettore di file del sistema.")
        self.b_ls_gguf = QPushButton("Sfoglia...")
        self.b_ls_gguf.clicked.connect(self._sfoglia_ls_gguf)
        rb.addWidget(self.ed_ls_gguf, 1)
        rb.addWidget(self.b_ls_gguf)
        l1.addLayout(rb)
        rc = QHBoxLayout()
        self.ed_ls_ctx = QLineEdit("16384")
        self.ed_ls_ctx.setToolTip("Context Length (-c). Piu' e' grande, piu' VRAM usa la KV cache.")
        self.ed_ls_ngl = QLineEdit("")
        self.ed_ls_ngl.setPlaceholderText("auto")
        self.ed_ls_ngl.setToolTip(
            "Layer sulla GPU (-ngl). Vuoto = llama.cpp decide da solo in base alla VRAM libera\n"
            "(consigliato). Un numero forza quel valore: 99 = tutti, ma se la VRAM non basta\n"
            "llama-server si chiude con 'failed to fit params', quindi abbassa (es. 20, 10).")
        self.ed_ls_extra = QLineEdit()
        self.ed_ls_extra.setPlaceholderText("argomenti extra, es. -ctk q8_0 -ctv q8_0 -fa on")
        rc.addWidget(QLabel("Contesto"))
        rc.addWidget(self.ed_ls_ctx, 1)
        rc.addWidget(QLabel("Layer GPU"))
        rc.addWidget(self.ed_ls_ngl, 1)
        rc.addWidget(self.ed_ls_extra, 3)
        l1.addLayout(rc)
        self.cb_trad.currentIndexChanged.connect(self._trad_ui)
        self._trad_ui()

        self.ck_musica = QCheckBox(
            "Musica di sottofondo: non scartare i testi cantati (niente filtro VAD, soglie rilassate)")
        self.ck_musica.setToolTip(
            "Whisper tende a ignorare i testi quando c'e' musica: il filtro VAD li scambia per\n"
            "non-parlato e le soglie 'no speech' / compressione scartano i segmenti.\n"
            "Questa opzione le disattiva o le rilassa (puo' aggiungere qualche frase in piu' nei silenzi).")
        l1.addWidget(self.ck_musica)

        self.ck_demucs = QCheckBox(
            "Isola prima la voce dalla musica (molto piu' efficace con la musica, ma lento)")
        self.ck_demucs.setToolTip(
            "Separa la voce dalla musica prima della trascrizione.\n"
            "Demucs:      python -m pip install demucs\n"
            "BS-RoFormer: python -m pip install \"audio-separator[gpu]\"  (o [cpu])\n"
            "Se il motore non e' disponibile si usa l'audio originale. Implica le impostazioni 'musica'.")
        l1.addWidget(self.ck_demucs)

        rs = QHBoxLayout()
        self.lb_sep = QLabel("Motore di separazione")
        self.cb_sep = QComboBox()
        for etichetta, chiave in SEPARATORI:
            self.cb_sep.addItem(etichetta, chiave)
        self.cb_sep.setToolTip(
            "Demucs: modello htdemucs, veloce e robusto.\n"
            "BS-RoFormer: spesso piu' pulito sulla voce cantata, ma piu' pesante; il checkpoint\n"
            "({}) viene scaricato alla prima esecuzione in:\n{}".format(BSROFORMER_MODEL, SEPARATOR_MODELS_DIR))
        rs.addWidget(self.lb_sep)
        rs.addWidget(self.cb_sep, 1)
        l1.addLayout(rs)

        def _sep_ui(att):
            self.lb_sep.setEnabled(att)
            self.cb_sep.setEnabled(att)
        self.ck_demucs.toggled.connect(_sep_ui)
        _sep_ui(self.ck_demucs.isChecked())
        self.cb_lingua.currentIndexChanged.connect(self._lingua_cambiata)
        self._lingua_cambiata()

        self.ed_prompt = QLineEdit()
        self.ed_prompt.setPlaceholderText("Suggerimento iniziale opzionale (nomi propri, termini difficili...)")
        l1.addWidget(self.ed_prompt)

        az = QHBoxLayout()

        self.b_avvia = QPushButton("▶ Trascrivi")
        self.b_avvia.setObjectName("primario")
        self.b_avvia.clicked.connect(self.avvia)

        self.b_libera = QPushButton("Libera memoria")
        self.b_libera.clicked.connect(self.libera)

        self.b_salva_accanto = QPushButton("💾 Salva accanto all'audio")
        self.b_salva_accanto.setToolTip(
            "Salva trascrizione e hiragana (.txt) e i dati del karaoke (_karaoke.json) "
            "nella stessa cartella del file originale"
        )
        self.b_salva_accanto.clicked.connect(self.salva_accanto)

        self.b_salva_stem = QPushButton("🎙 Salva voce e musica")
        self.b_salva_stem.setToolTip(
            "Separa la voce dalla musica del file scelto (con il motore selezionato sotto) e salva\n"
            "accanto all'originale due file WAV: <nome>_voce.wav e <nome>_musica.wav.\n"
            "Usa sempre il file intero (Da/A ignorati) e non richiede di aver trascritto prima."
        )
        self.b_salva_stem.clicked.connect(self.salva_voce_musica)

        self.b_esporta_mka = QPushButton("🎛 Esporta tracce .mka")
        self.b_esporta_mka.setToolTip(
            "Unisce le due tracce gia' separate (<nome>_voce.wav e <nome>_musica.wav, create da\n"
            "'trascrivi' con separazione o da 'Salva voce e musica') in un unico file Matroska\n"
            "multitraccia <nome>.mka accanto all'originale: traccia 1 = Voce, traccia 2 = Musica.\n"
            "Con il tasto destro del mouse puoi scegliere la codifica lossless (FLAC) invece di\n"
            "Opus 192k. Non fa alcuna separazione: se gli stem non ci sono, avvisa."
        )
        self.b_esporta_mka.clicked.connect(self.esporta_tracce_mka)

        self.b_ferma = QPushButton("■ Interrompi")
        self.b_ferma.setEnabled(False)
        self.b_ferma.setToolTip(
            "Interrompe il processo in corso (trascrizione, traduzione o batch). Si ferma al\n"
            "prossimo segmento/riga: una richiesta gia' partita verso il modello deve finire.")
        self.b_ferma.clicked.connect(self.interrompi)

        az.addWidget(self.b_avvia)
        az.addWidget(self.b_ferma)
        self.b_carica_k = QPushButton("📂 Carica karaoke...")
        self.b_carica_k.setToolTip(
            "Carica un file _karaoke.json salvato in precedenza: riapre il video/audio "
            "e ripristina testi e sincronizzazione senza trascrivere di nuovo"
        )
        self.b_carica_k.clicked.connect(self.scegli_karaoke)

        az.addWidget(self.b_salva_accanto)
        az.addWidget(self.b_salva_stem)
        az.addWidget(self.b_esporta_mka)
        az.addWidget(self.b_carica_k)
        az.addStretch(1)
        az.addWidget(self.b_libera)
        l1.addLayout(az)

        self.barra = QProgressBar()
        self.barra.setRange(0, 1)
        self.barra.setTextVisible(False)
        l1.addWidget(self.barra)

        self.lb_stato = QLabel("Pronto.")
        self.lb_stato.setObjectName("sotto")
        l1.addWidget(self.lb_stato)

        self.c1 = c1
        root.addWidget(c1)

        # --- batch ---
        cbt, lbt = card("Batch (piu' file in coda)")

        nota = QLabel(
            "Ogni file viene trascritto per intero con le impostazioni qui sopra (Da/A ignorati) "
            "e salvato accanto all'originale: _trascrizione.txt, _hiragana.txt, "
            "_traduzione.txt e _karaoke.json. Con la traduzione attiva il batch lavora in due fasi: "
            "prima tutte le trascrizioni (Whisper caricato una volta sola), poi lo scarica dalla memoria "
            "e traduce tutti i file. Puoi anche trascinare qui piu' file o una cartella."
        )
        nota.setObjectName("sotto")
        nota.setWordWrap(True)
        lbt.addWidget(nota)

        self.lst_batch = QListWidget()
        self.lst_batch.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.lst_batch.setFixedHeight(140)
        lbt.addWidget(self.lst_batch)

        rb = QHBoxLayout()
        self.b_b_file = QPushButton("Aggiungi file...")
        self.b_b_file.clicked.connect(self.scegli_batch_file)
        self.b_b_cart = QPushButton("Aggiungi cartella...")
        self.b_b_cart.setToolTip("Aggiunge tutti i file audio/video della cartella (sottocartelle comprese)")
        self.b_b_cart.clicked.connect(self.scegli_batch_cartella)
        self.b_b_rimuovi = QPushButton("Rimuovi selezionati")
        self.b_b_rimuovi.setToolTip("Rimuove i file selezionati dalla coda (tasto Canc)")
        self.b_b_rimuovi.clicked.connect(self.rimuovi_batch)
        self.b_b_svuota = QPushButton("Svuota")
        self.b_b_svuota.clicked.connect(self.svuota_batch)
        rb.addWidget(self.b_b_file)
        rb.addWidget(self.b_b_cart)
        rb.addWidget(self.b_b_rimuovi)
        rb.addWidget(self.b_b_svuota)
        rb.addStretch(1)
        self.lb_batch = QLabel("")
        self.lb_batch.setObjectName("sotto")
        rb.addWidget(self.lb_batch)
        lbt.addLayout(rb)

        sc_canc = QShortcut(QKeySequence("Delete"), self.lst_batch)
        sc_canc.setContext(Qt.WidgetShortcut)
        sc_canc.activated.connect(self.rimuovi_batch)

        self.ck_salta = QCheckBox("Salta i file che hanno gia' un _karaoke.json (utile per riprendere un batch interrotto)")
        self.ck_salta.setChecked(True)
        lbt.addWidget(self.ck_salta)

        rb2 = QHBoxLayout()
        self.b_batch = QPushButton("▶▶ Trascrivi batch")
        self.b_batch.setObjectName("primario")
        self.b_batch.clicked.connect(self.avvia_batch)
        self.b_batch_stop = QPushButton("■ Interrompi dopo il file corrente")
        self.b_batch_stop.setEnabled(False)
        self.b_batch_stop.clicked.connect(self.ferma_batch)
        rb2.addWidget(self.b_batch)
        rb2.addWidget(self.b_batch_stop)
        rb2.addStretch(1)
        lbt.addLayout(rb2)

        self.cbt = cbt
        root.addWidget(cbt)
        self._batch_conteggio()

        # --- player ---
        c2, l2 = card("Player")

        self.l2, self.c2 = l2, c2
        self.cont_video = ContenitoreVideo()
        self.video = self.cont_video.video
        self.onda = self.video.onda
        self._collega_cont(self.cont_video)
        self.cont_video.hide()
        l2.addWidget(self.cont_video, 1)

        self.forma = Forma()
        self.sl_pos = SliderPos()
        self.sl_pos.setRange(0, 0)
        self.sl_pos.seek.connect(self.vai_a)
        l2.addWidget(self.sl_pos)

        ctrl = QHBoxLayout()
        ctrl.setSpacing(10)

        self.b_play = QPushButton("▶")
        self.b_play.setObjectName("play")
        self.b_play.clicked.connect(self.play_pausa)

        self.b_stop = QPushButton("■")
        self.b_stop.setObjectName("tondo")
        self.b_stop.clicked.connect(self.ferma)

        self.lb_tempo = QLabel("00:00.0 / 00:00.0")
        self.lb_tempo.setObjectName("tempo")

        self.cb_vel = QComboBox()
        for v in ("0.5x", "0.75x", "1x", "1.25x", "1.5x", "2x"):
            self.cb_vel.addItem(v)
        self.cb_vel.setCurrentText("1x")
        self.cb_vel.currentTextChanged.connect(
            lambda tx: self.player.setPlaybackRate(float(tx.rstrip("x")))
        )

        self.sl_vol = QSlider(Qt.Horizontal)
        self.sl_vol.setRange(0, 100)
        self.sl_vol.setValue(80)
        self.sl_vol.setFixedWidth(110)
        self.sl_vol.valueChanged.connect(self.player.setVolume)

        ctrl.addWidget(self.b_play)
        ctrl.addWidget(self.b_stop)
        ctrl.addWidget(self.lb_tempo)
        ctrl.addStretch(1)

        self.lb_sub = QLabel("Sul video")
        self.cb_sub = QComboBox()
        self.cb_sub.addItems(["Originale + Hiragana", "Solo originale", "Solo hiragana", "Nessun testo",
                              "Originale + Hiragana + Traduzione", "Solo traduzione"])
        self.cb_sub.currentIndexChanged.connect(self._cambia_sub)

        self.b_fs = QPushButton("⛶ Schermo intero")
        self.b_fs.setObjectName("piccolo")
        self.b_fs.setToolTip(
            "Tasto F o doppio clic sul video: schermo intero  |  In schermo intero: Esc esce, "
            "Spazio play/pausa, frecce ±5 s"
        )
        self.b_fs.clicked.connect(self.schermo_intero)

        ctrl.addWidget(self.lb_sub)
        ctrl.addWidget(self.cb_sub)
        ctrl.addWidget(self.b_fs)
        ctrl.addSpacing(10)
        ctrl.addWidget(QLabel("Velocita'"))
        ctrl.addWidget(self.cb_vel)
        ctrl.addSpacing(10)
        ctrl.addWidget(QLabel("🔊"))
        ctrl.addWidget(self.sl_vol)

        l2.addLayout(ctrl)
        root.addWidget(c2)

        self.player.stateChanged.connect(self._stato_player)
        self.player.positionChanged.connect(self.aggiorna_pos)
        self.player.error[QMediaPlayer.Error].connect(self._errore_player)
        self.player.mediaStatusChanged.connect(self._stato_media)
        # guardia del video: il backend multimediale (GStreamer) puo' bloccarsi senza emettere alcun errore
        self._t_decod = 0.0
        self._tm_decod = QTimer(self)
        self._tm_decod.setInterval(1000)
        self._tm_decod.timeout.connect(self._decodifica_attiva)
        # video: fotogrammi decodificati da ffmpeg e mostrati in sincronia con la posizione dell'audio
        self._frame_src = None
        self._frame_vecchi = []
        self._info_video = None
        self._frame_t_mostrato = None
        self._frame_ultimo_avvio = 0.0
        self._tm_frame = QTimer(self)
        self._tm_frame.setInterval(25)
        self._tm_frame.timeout.connect(self._aggiorna_frame)

        # --- testi affiancati ---
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)
        split.setMinimumHeight(200)

        c3, l3 = card()
        h3 = QHBoxLayout()

        lb3 = QLabel("TRASCRIZIONE (ORIGINALE)")
        lb3.setObjectName("sez")

        self.b_copia = QPushButton("Copia")
        self.b_copia.setObjectName("piccolo")
        self.b_copia.clicked.connect(self.copia)

        self.b_salva = QPushButton("Salva .txt")
        self.b_salva.setObjectName("piccolo")
        self.b_salva.clicked.connect(self.salva)

        self.b_copia_num = QPushButton("Copia con numeri")
        self.b_copia_num.setObjectName("piccolo")
        self.b_copia_num.setToolTip(
            "Copia i segmenti come righe '[N|mm:ss.s] testo', da far tradurre a un programma esterno.\n"
            "Il traduttore non tocca i numeri: incollando il risultato nel riquadro TRADUZIONE e premendo\n"
            "'Applica' ogni riga torna al suo segmento, con i tempi esatti.")
        self.b_copia_num.clicked.connect(self.copia_numerata)
        h3.addWidget(lb3)
        h3.addStretch(1)
        h3.addWidget(self.b_copia_num)
        h3.addWidget(self.b_copia)
        h3.addWidget(self.b_salva)
        l3.addLayout(h3)

        self.out = CasellaTesto()
        self.out.setPlaceholderText("Il testo trascritto comparira' qui e puoi correggerlo a mano.")
        self.out.doppio_clic.connect(lambda pos: self.salta_a_testo(self.ev_o, pos))
        self.out.textChanged.connect(self._testo_o_modificato)
        l3.addWidget(self.out, 1)

        split.addWidget(c3)

        c4, l4 = card()
        h4 = QHBoxLayout()

        lb4 = QLabel("HIRAGANA")
        lb4.setObjectName("sez")

        self.b_aggiorna_hira = QPushButton("Rigenera")
        self.b_aggiorna_hira.setObjectName("piccolo")
        self.b_aggiorna_hira.setToolTip("Riconverte il testo della trascrizione in hiragana")
        self.b_aggiorna_hira.clicked.connect(self.aggiorna_hiragana)

        self.b_copia_hira = QPushButton("Copia")
        self.b_copia_hira.setObjectName("piccolo")
        self.b_copia_hira.clicked.connect(self.copia_hiragana)

        h4.addWidget(lb4)
        h4.addStretch(1)
        h4.addWidget(self.b_aggiorna_hira)
        h4.addWidget(self.b_copia_hira)
        l4.addLayout(h4)

        self.out_hira = CasellaTesto()
        self.out_hira.setPlaceholderText("Il testo in hiragana comparira' qui (puoi modificarlo a mano).")
        self.out_hira.doppio_clic.connect(lambda pos: self.salta_a_testo(self.ev_h, pos))
        self.out_hira.textChanged.connect(self._testo_h_modificato)
        l4.addWidget(self.out_hira, 1)

        split.addWidget(c4)

        self.c5, l5 = card()
        h5 = QHBoxLayout()
        lb5 = QLabel("TRADUZIONE")
        lb5.setObjectName("sez")
        self.b_copia_trad = QPushButton("Copia")
        self.b_copia_trad.setObjectName("piccolo")
        self.b_copia_trad.clicked.connect(self.copia_traduzione)
        self.b_applica_trad = QPushButton("Applica")
        self.b_applica_trad.setObjectName("piccolo")
        self.b_applica_trad.setToolTip(
            "Usa il testo di questo riquadro (anche incollato a mano) come traduzione ufficiale:\n"
            "viene agganciata ai tempi del giapponese e salvata nel karaoke (.json) e nei sottotitoli.")
        self.b_applica_trad.clicked.connect(self.applica_traduzione)
        self.b_traduci = QPushButton("Traduci")
        self.b_traduci.setObjectName("piccolo")
        self.b_traduci.setToolTip(
            "Traduce di nuovo solo il testo (senza rifare la trascrizione), con il modello e la lingua\n"
            "scelti sopra, e mette il risultato in questo riquadro agganciato ai tempi del giapponese.")
        self.b_traduci.clicked.connect(self.traduci_solo)
        h5.addWidget(lb5)
        h5.addStretch(1)
        h5.addWidget(self.b_traduci)
        h5.addWidget(self.b_applica_trad)
        h5.addWidget(self.b_copia_trad)
        l5.addLayout(h5)

        self.out_trad = CasellaTesto()
        self.out_trad.setPlaceholderText("Il testo nella lingua scelta comparira' qui (una riga per frase).")
        self.out_trad.doppio_clic.connect(lambda pos: self.salta_a_testo(self.ev_t, pos, self.t0_t))
        self.out_trad.textChanged.connect(self._testo_t_modificato)
        l5.addWidget(self.out_trad, 1)

        split.addWidget(self.c5)
        self.c5.setVisible(False)
        root.addWidget(split, 1)

        self.ev_o = Evidenziatore(self.out)
        self.ev_h = Evidenziatore(self.out_hira)
        self.ev_t = Evidenziatore(self.out_trad)

        self._sp_o_orig, self._sp_h_orig, self._hira_orig = [], [], ""

        self._mostra_video(False)
        self._cambia_sub()

        self.sc_f = QShortcut(QKeySequence("F"), self)
        self.sc_f.setContext(Qt.WindowShortcut)
        self.sc_f.activated.connect(self._tasto_f)

        self.ultima_cartella = ""
        self._carica_impostazioni()
        # dopo il caricamento della configurazione i campi llama-server possono essere
        # stati popolati ma restare grigi: riappliciamo lo stato di abilitazione
        # (con la lingua salvata "giapponese" cb_trad risulta disabilitato e _trad_ui
        # spegneva i campi anche se il backend scelto era "llama.cpp")
        self._lingua_cambiata()

    # ------------------------------------------------------------ impostazioni
    def _impostazioni(self):
        return dict(
            modello=self.cb_modello.currentText(),
            lingua=LINGUE[self.cb_lingua.currentIndex()][1] or "auto",
            device=self.cb_device.currentText(),
            timestamp=self.ck_timestamp.isChecked(),
            a_capo=self.ck_acapo.isChecked(),
            parole=self.ck_parole.isChecked(),
            prefisso=self.ck_prefisso.isChecked(),
            doppio=self.ck_doppio.isChecked(),
            musica=self.ck_musica.isChecked(),
            demucs=self.ck_demucs.isChecked(),
            separatore=self.cb_sep.currentData(),
            batch_salta=self.ck_salta.isChecked(),
            trad_modello=self.cb_trad.currentIndex(),
            lm_url=self.ed_lm_url.text(),
            lm_modello=self.ed_lm_modello.text(),
            lm_contesto=self.ck_lm_ctx.isChecked(),
            lm_tutto=self.ck_lm_tutto.isChecked(),
            lm_veloce=self.ck_lm_veloce.isChecked(),
            scarica_whisper=self.ck_scarica_whisper.isChecked(),
            ls_exe=self.ed_ls_exe.text().strip(),
            ls_gguf=self.ed_ls_gguf.text().strip(),
            ls_ctx=self.ed_ls_ctx.text().strip(),
            ls_ngl=self.ed_ls_ngl.text().strip(),
            ls_extra=self.ed_ls_extra.text().strip(),
            prompt_iniziale=self.ed_prompt.text(),
            sottotitoli=self.cb_sub.currentIndex(),
            velocita=self.cb_vel.currentText(),
            volume=self.sl_vol.value(),
            cartella=self.ultima_cartella,
            finestra=[self.width(), self.height()],
        )

    def _salva_impostazioni(self):
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as fh:
                json.dump(self._impostazioni(), fh, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def _carica_impostazioni(self):
        try:
            with open(CONFIG_FILE, encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            return
        if not isinstance(d, dict):
            return

        def combo(cb, valore):
            i = cb.findText(valore) if isinstance(valore, str) else -1
            if i >= 0:
                cb.setCurrentIndex(i)

        def spunta(ck, chiave):
            if isinstance(d.get(chiave), bool):
                ck.setChecked(d[chiave])

        combo(self.cb_modello, d.get("modello"))
        combo(self.cb_device, d.get("device"))
        isep = self.cb_sep.findData(d.get("separatore"))
        if isep >= 0:
            self.cb_sep.setCurrentIndex(isep)
        # i campi llama-server vanno popolati PRIMA di _trad_ui: se il modello scelto
        # e' "llama.cpp", i widget restano disabilitati finche' non sono tutti compilati
        tm = d.get("trad_modello")
        if isinstance(tm, int) and 0 <= tm < self.cb_trad.count():
            self.cb_trad.setCurrentIndex(tm)
        if isinstance(d.get("lm_url"), str) and d["lm_url"].strip():
            self.ed_lm_url.setText(d["lm_url"])
        if isinstance(d.get("lm_modello"), str):
            self.ed_lm_modello.setText(d["lm_modello"])
        for ed, k in ((self.ed_ls_exe, "ls_exe"), (self.ed_ls_gguf, "ls_gguf"),
                       (self.ed_ls_ctx, "ls_ctx"), (self.ed_ls_ngl, "ls_ngl"),
                       (self.ed_ls_extra, "ls_extra")):
            if isinstance(d.get(k), str):
                ed.setText(d[k])
        cod = d.get("lingua")
        for i, (_, c) in enumerate(LINGUE):
            if (c or "auto") == cod:
                self.cb_lingua.setCurrentIndex(i)
                break
        for ck, k in ((self.ck_timestamp, "timestamp"), (self.ck_acapo, "a_capo"),
                      (self.ck_parole, "parole"), (self.ck_prefisso, "prefisso"),
                      (self.ck_doppio, "doppio"), (self.ck_musica, "musica"),
                      (self.ck_demucs, "demucs"), (self.ck_salta, "batch_salta"),
                      (self.ck_lm_ctx, "lm_contesto"),
                      (self.ck_lm_tutto, "lm_tutto"),
                      (self.ck_lm_veloce, "lm_veloce"),
                      (self.ck_scarica_whisper, "scarica_whisper")):
            spunta(ck, k)
        if isinstance(d.get("prompt_iniziale"), str):
            self.ed_prompt.setText(d["prompt_iniziale"])
        sub = d.get("sottotitoli")
        if isinstance(sub, int) and 0 <= sub < self.cb_sub.count():
            self.cb_sub.setCurrentIndex(sub)
        combo(self.cb_vel, d.get("velocita"))
        vol = d.get("volume")
        if isinstance(vol, int) and 0 <= vol <= 100:
            self.sl_vol.setValue(vol)
        if isinstance(d.get("cartella"), str) and os.path.isdir(d["cartella"]):
            self.ultima_cartella = d["cartella"]
        fin = d.get("finestra")
        if (isinstance(fin, list) and len(fin) == 2 and all(isinstance(v, int) for v in fin)
                and fin[0] >= 400 and fin[1] >= 300):
            scr = QApplication.primaryScreen().availableGeometry()
            self.resize(min(fin[0], scr.width()), min(fin[1], scr.height()))

    # ------------------------------------------------------------------ file
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        files = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if not files:
            return
        if len(files) == 1 and os.path.isfile(files[0]):
            f = files[0]
            if f.lower().endswith(".json"):
                self.carica_karaoke_json(f)
            else:
                self.imposta_file(f)
        else:
            # piu' file o una cartella: finiscono nella coda del batch
            self._aggiungi_batch(files)

    def scegli_file(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Scegli un file", self.ultima_cartella,
            FILTRO_FILE,
            options=QFileDialog.DontUseNativeDialog
        )
        if f:
            self.imposta_file(f)

    def imposta_file(self, path):
        if os.path.isdir(os.path.dirname(path)):
            self.ultima_cartella = os.path.dirname(path)
        self.ed_file.setText(path)
        self._file_modificato()

    def _file_modificato(self):
        path = self.ed_file.text().strip()
        if not path:
            return
        if not os.path.isfile(path):
            self.lb_stato.setText("File non trovato: " + path)
            log_hira("file non trovato: " + path)
            return
        # stesso file di prima: si ricarica solo se la volta scorsa non e' arrivato a fine decodifica (prima non succedeva nulla)
        in_corso = self.decoder is not None and self.decoder.isRunning() and self.decoder.path == path
        if path != self.file_corrente or (self.wav_tmp is None and not in_corso):
            self.carica_audio(path)
            self._cerca_karaoke(path)

    def _rimuovi_wav(self):
        if self.wav_tmp and os.path.exists(self.wav_tmp):
            try:
                os.remove(self.wav_tmp)
            except OSError:
                pass
        self.wav_tmp = None

    def carica_audio(self, path):
        self.file_corrente = path
        self._tm_frame.stop()
        self._ferma_frame()
        self._info_video = None
        self.video.item.setPixmap(QPixmap())
        self.player.stop()
        self.player.setMedia(QMediaContent())
        self._rimuovi_wav()
        self.onda.pulisci()
        self._mostra_video(False, schermo=False)

        self.forma.pulisci("Decodifico l'audio...")
        self.sl_pos.setRange(0, 0)
        self.lb_stato.setText("Decodifico il file...")
        self.lb_tempo.setText("00:00.0 / 00:00.0")
        log_hira("carico %s" % path)
        self._t_decod = time.perf_counter()
        self._tm_decod.start()

        if self.decoder is not None:
            self._vecchi.append(self.decoder)
            self._vecchi = [t for t in self._vecchi if t.isRunning()]

        self.decoder = Decoder(path)
        self.decoder.pronto.connect(self._audio_pronto)
        self.decoder.errore.connect(self._audio_errore)
        self.decoder.start()

    def _decodifica_attiva(self):
        """Mostra il tempo trascorso: se il contatore avanza, ffmpeg sta ancora lavorando (file lungo), non e' bloccato."""
        self.lb_stato.setText("Decodifico il file... %d s" % int(time.perf_counter() - self._t_decod))

    def _audio_pronto(self, path, wav, db, hop, durata, ha_video):
        if path == self.file_corrente:
            self._tm_decod.stop()
            self.lb_stato.setText("Pronto: %s (%s)" % (os.path.basename(path), "video" if ha_video else "solo audio"))
        if path != self.file_corrente:
            try:
                os.remove(wav)
            except OSError:
                pass
            return

        self.wav_tmp = wav
        self.durata = durata
        self.forma.imposta(db, hop, durata)
        self.onda.imposta(db, hop)

        emittente = self.sender()
        info = getattr(emittente, "info", None) if emittente is not None else None
        self.modo_video = bool(ha_video) and info is not None
        self._mostra_video(self.modo_video)
        # l'audio suona sempre dal wav estratto (affidabile con qualunque codec); il video lo disegna ffmpeg
        self._log_player("apro l'audio (%s), video=%s" % (wav, self.modo_video))
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(wav)))
        self.player.setPlaybackRate(float(self.cb_vel.currentText().rstrip("x")))
        if self.modo_video:
            self._info_video = info
            self._frame_t_mostrato = None
            self._avvia_frame(0.0)
            self._tm_frame.start()
        else:
            self._tm_frame.stop()

        self.sl_pos.setRange(0, int(durata * 1000))
        self._aggiorna_selezione()
        self.aggiorna_pos(0)

    def _audio_errore(self, path, msg):
        if path == self.file_corrente:
            self._tm_decod.stop()
            self.forma.pulisci("Impossibile decodificare l'audio")
            self.lb_stato.setText("Errore di decodifica audio.")
            QMessageBox.warning(self, "Audio", msg)

    def _mostra_video(self, on, schermo=True):
        """on: il file ha una vera traccia video. Con schermo=True lo schermo si vede comunque: nei
        file solo audio e' nero, con i sottotitoli karaoke e la sinusoide dei dB."""
        self.modo_video = bool(on)
        if not schermo:
            self._fullscreen(False)
        self.cont_video.setVisible(bool(schermo))
        self.root.setStretchFactor(self.c1, 0)
        self.root.setStretchFactor(self.c2, 4 if schermo else 0)
        self.lb_sub.setVisible(bool(schermo))
        self.cb_sub.setVisible(bool(schermo))
        self.b_fs.setVisible(bool(schermo))
        self.onda.attiva = bool(schermo) and not on
        self._onda_ancora = None
        self._sub_chiave = None
        self._aggiorna_schermi()

    def _aggiorna_schermi(self):
        self.video.viewport().update()
        if self.cont_fs is not None and self.cont_fs.isVisible():
            self.cont_fs.video.viewport().update()

    def _tempo_onda(self, ms):
        """Tempo di riproduzione interpolato tra un aggiornamento e l'altro del player (movimento fluido)."""
        t = ms / 1000.0
        now = time.perf_counter()
        a = self._onda_ancora
        if a is None or t != a[0]:
            self._onda_ancora = a = (t, now)
        if self.player.state() == QMediaPlayer.PlayingState:
            rate = self.player.playbackRate() or 1.0
            te = t + min((now - a[1]) * rate, 1.0)
            if 0.0 < self._onda_ultimo - te < 0.3:
                te = self._onda_ultimo      # niente piccoli passi indietro
        else:
            te = t
        self._onda_ultimo = te
        return te

    def _tasto_f(self):
        w = QApplication.focusWidget()
        if isinstance(w, (QLineEdit, QTextEdit, QDoubleSpinBox)):
            return
        self.schermo_intero()

    def schermo_intero(self):
        if self.wav_tmp is not None:
            self._fullscreen(not self._fs)

    def _collega_cont(self, c):
        c.doppio_clic.connect(self.schermo_intero)
        c.esci.connect(lambda: self._fullscreen(False))
        c.play_pausa.connect(self.play_pausa)
        c.salta.connect(self._salta_rel)

    def _sub_set(self, html):
        self.cont_video.imposta_sub(html)
        if self.cont_fs is not None:
            self.cont_fs.imposta_sub(html)

    def _righe_set(self, n):
        self.cont_video.imposta_righe(n)
        if self.cont_fs is not None:
            self.cont_fs.imposta_righe(n)

    def _fullscreen(self, on):
        if on == self._fs:
            return
        self._fs = on

        if on:
            if self.cont_fs is None:
                self.cont_fs = ContenitoreVideo(self.video)
                self.cont_fs.setWindowTitle("Video")
                self._collega_cont(self.cont_fs)
            self.cont_fs.imposta_righe(self.cont_video._righe)
            self.cont_fs.imposta_sub(self.cont_video._html)
            self.cont_fs.showFullScreen()
            self.cont_fs.setFocus()
            self.cont_fs.activateWindow()
        elif self.cont_fs is not None:
            self.cont_fs.hide()
            self.activateWindow()

        self._sub_chiave = None
        self.aggiorna_pos(self.player.position())

    def _salta_rel(self, delta):
        t = self.player.position() / 1000.0 + delta
        self.vai_a(max(0.0, min(getattr(self, "durata", 0.0), t)))

    def _cambia_sub(self, *_):
        m = self.cb_sub.currentIndex()
        self._righe_set({0: 4, 1: 2, 2: 2, 4: 6, 5: 2}.get(m, 0))
        self._sub_chiave = None
        self.aggiorna_pos(self.player.position())

    def _prog_jp(self, t):
        """Quantita' (pesata) di giapponese gia' pronunciata al tempo t."""
        i = bisect_right(self.t0, t) - 1
        if i < 0:
            return 0.0
        d = self.t1[i] - self.t0[i]
        if t >= self.t1[i] or d <= 0:
            return self._cw[i + 1]
        return self._cw[i] + (self._cw[i + 1] - self._cw[i]) * (t - self.t0[i]) / d

    def _frazione_trad(self, i, t):
        """Frazione (0..1) della riga di traduzione i da illuminare al tempo t:
        e' la stessa proporzione di giapponese gia' illuminato nell'intervallo
        della riga. Se non c'e' giapponese in quell'intervallo, va a tempo."""
        a, b = self.t0_t[i], self.t1_t[i]
        if t <= a:
            return 0.0
        if t >= b or b <= a:
            return 1.0
        ca, cb = self._prog_jp(a), self._prog_jp(b)
        if self.t0 and cb - ca > 1e-9:
            f = (self._prog_jp(t) - ca) / (cb - ca)
        else:
            f = (t - a) / (b - a)
        return min(1.0, max(0.0, f))

    @staticmethod
    def _mix(c1, c2, t):
        """Interpola linearmente due colori '#rrggbb' (t=0 -> c1, t=1 -> c2)."""
        t = min(1.0, max(0.0, t))
        a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
        b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
        return "#%02x%02x%02x" % tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))

    @staticmethod
    def _liscia(x):
        x = min(1.0, max(0.0, x))
        return x * x * (3.0 - 2.0 * x)

    def _env(self, passo):
        """Intensita' 0..1 al passo 'passo': sale in fretta, tiene, poi scende."""
        u = (passo + 0.5) / SUB_PASSI
        return self._liscia(u / 0.25) if u < 0.5 else self._liscia((1.0 - u) / 0.25)

    def _colore_corrente(self, passo):
        """Token in riproduzione: dal bianco sale al rosso, poi si raffredda verso l'arancione."""
        u = (passo + 0.5) / SUB_PASSI
        base = "#ffffff" if u < 0.5 else C_SPOKEN
        return self._mix(base, SUB_CURRENT, self._env(passo))

    def _colore_traduzione(self, passo):
        """Azzurro intenso del tratto di traduzione in corso: stessa curva del rosso dei kanji."""
        u = (passo + 0.5) / SUB_PASSI
        base = SUB_TRAD if u < 0.5 else C_SPOKEN
        return self._mix(base, SUB_TRAD_CURRENT, self._env(passo))

    def _intervallo_trad(self, i, idx):
        """Intervallo [c0, c1) di caratteri della riga tradotta i che corrisponde al token
        giapponese idx (stessa proporzione dell'illuminazione); esteso a parole intere se ci sono spazi."""
        testo = self.trad_righe[i]
        n = len(testo)
        a, b = self.t0_t[i], self.t1_t[i]
        ca, cb = self._prog_jp(a), self._prog_jp(b)
        if cb - ca > 1e-9:
            f0 = (self._cw[idx] - ca) / (cb - ca)
            f1 = (self._cw[idx + 1] - ca) / (cb - ca)
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
                c1 -= 1
        return c0, c1

    def _html_trad(self, testo, cut, c1=None, passo=None):
        if c1 is not None and passo is not None:
            parti = ((testo[:cut], C_SPOKEN),
                     (testo[cut:c1], self._colore_traduzione(passo)),
                     (testo[c1:], SUB_TRAD))
        else:
            parti = ((testo[:cut], C_SPOKEN), (testo[cut:], SUB_TRAD))
        out = ""
        for frag, col in parti:
            if frag:
                out += '<span style="color:%s;">%s</span>' % (col, escape(frag))
        return '<div style="white-space: pre-wrap;">%s</div>' % out

    def _html_riga(self, tok, a, b, idx, in_corso, corrente=None):
        out = []
        for i in range(a, b):
            tx = escape(tok[i])
            if i == idx and corrente:
                col = corrente                # token in riproduzione: sfumatura verso il rosso
            elif i <= idx:
                col = C_SPOKEN
            else:
                col = "#ffffff"
            out.append('<span style="color:%s;">%s</span>' % (col, tx))
        return '<div style="white-space: pre-wrap;">%s</div>' % "".join(out)

    def _aggiorna_sub(self, t):
        if not self.tok_sub or not self.t0:
            return

        m = self.cb_sub.currentIndex()
        mostra_jp = m in (0, 1, 2, 4)
        mostra_tr = m in (4, 5) and bool(self.t0_t)

        idx, in_corso = -1, False
        chiave_jp, chiave_tr, k_c = None, None, None

        if mostra_jp:
            idx = bisect_right(self.t0, t) - 1
            if idx >= 0:
                in_corso = t <= self.t1[idx]
                if in_corso or t - self.t1[idx] <= 1.5:
                    chiave_jp = (idx, in_corso)
                if in_corso and self.tok_sub["o"][idx].strip():
                    d = self.t1[idx] - self.t0[idx]
                    u = (t - self.t0[idx]) / d if d > 0 else 0.5
                    k_c = (idx, min(SUB_PASSI - 1, max(0, int(u * SUB_PASSI))))

        if mostra_tr:
            it = bisect_right(self.t0_t, t) - 1
            if it >= 0 and t <= self.t1_t[it] + 1.5:
                n_car = len(self.trad_righe[it])
                c0 = c1 = 0
                if k_c is not None:
                    c0, c1 = self._intervallo_trad(it, k_c[0])
                if c1 > c0:
                    chiave_tr = (it, c0, c1)      # tratto azzurro = dove i kanji sono rossi
                else:
                    chiave_tr = (it, int(round(n_car * self._frazione_trad(it, t))), None)

        chiave = None
        if chiave_jp is not None or chiave_tr is not None:
            chiave = (chiave_jp, chiave_tr, k_c, m)

        if chiave == self._sub_chiave:
            return
        self._sub_chiave = chiave

        if chiave is None:
            self._sub_set("")
            return

        righe = []
        corrente = self._colore_corrente(k_c[1]) if k_c else None
        if chiave_jp is not None:
            a, b = self.seg_range[self.tok_sub["seg"][idx]]
            if m in (0, 1, 4):
                righe.append(self._html_riga(self.tok_sub["o"], a, b, idx, in_corso, corrente))
            if m in (0, 2, 4):
                righe.append(self._html_riga(self.tok_sub["h"], a, b, idx, in_corso, corrente))
        if chiave_tr is not None:
            if chiave_tr[2] is not None and k_c:
                righe.append(self._html_trad(self.trad_righe[chiave_tr[0]], chiave_tr[1],
                                             chiave_tr[2], k_c[1]))
            else:
                righe.append(self._html_trad(self.trad_righe[chiave_tr[0]], chiave_tr[1]))
        self._sub_set("".join(righe))

    @staticmethod
    def _log_player(msg):
        log_hira(msg)

    def _errore_player(self, codice):
        self._log_player("errore del player (%s): %s" % (codice, self.player.errorString()))
        self.lb_stato.setText("Errore del player: " + self.player.errorString())

    def _stato_media(self, st):
        nomi = {QMediaPlayer.UnknownMediaStatus: "sconosciuto", QMediaPlayer.NoMedia: "nessun media",
                QMediaPlayer.LoadingMedia: "caricamento", QMediaPlayer.LoadedMedia: "caricato",
                QMediaPlayer.StalledMedia: "bloccato (dati non arrivano)", QMediaPlayer.BufferingMedia: "buffering",
                QMediaPlayer.BufferedMedia: "bufferizzato", QMediaPlayer.EndOfMedia: "fine",
                QMediaPlayer.InvalidMedia: "NON VALIDO per il backend"}
        self._log_player("stato media: " + nomi.get(st, str(st)))

    # ---------------------------------------------------------------- video (fotogrammi da ffmpeg)
    def _ferma_frame(self):
        src = self._frame_src
        if src is not None:
            src.ferma()
            self._frame_vecchi.append(src)
            self._frame_src = None
        self._frame_vecchi = [t for t in self._frame_vecchi if t.isRunning()]

    def _avvia_frame(self, t):
        self._ferma_frame()
        info = self._info_video
        if not info or not self.file_corrente:
            return
        self._frame_ultimo_avvio = time.perf_counter()
        self._frame_src = SorgenteFrame(self.file_corrente, max(0.0, t), info["w"], info["h"], info["fps"])
        self._frame_src.start()

    def _mostra_frame(self, img):
        pm = QPixmap.fromImage(img)
        cambiata = pm.size() != self.video.item.pixmap().size()
        self.video.item.setPixmap(pm)
        if cambiata:
            self.video._adatta()
            if self.cont_fs is not None:
                self.cont_fs.video._adatta()

    def _aggiorna_frame(self):
        """Ogni 25 ms: mostra il fotogramma che corrisponde alla posizione dell'audio; dopo un salto (o se la
        decodifica e' rimasta troppo indietro) riparte da li'."""
        src = self._frame_src
        if not self.modo_video or src is None:
            return
        t = self._tempo_onda(self.player.position())
        coda = src.coda
        dovuto = None
        while coda and coda[0][0] <= t + 0.005:
            dovuto = coda.popleft()
        if dovuto is not None:
            self._frame_t_mostrato = dovuto[0]
            self._mostra_frame(dovuto[1])
        if time.perf_counter() - self._frame_ultimo_avvio < 0.6:
            return
        mostrato = self._frame_t_mostrato
        indietro = mostrato is not None and t < mostrato - 0.3          # salto all'indietro
        prima = t < src.inizio - 0.05
        in_ritardo = (not coda) and t > src.ultimo + 1.0 and (not src.finito or t < getattr(self, "durata", 0.0) - 1.0)
        if indietro or prima or in_ritardo:
            self._avvia_frame(t)

    def _aggiorna_selezione(self):
        a = self.sp_inizio.value()
        b = self.sp_fine.value()

        if a > 0 or b > 0:
            self.forma.imposta_selezione(a, b if b > a else getattr(self, "durata", 0.0))
        else:
            self.forma.imposta_selezione(0, 0)

    # ---------------------------------------------------------------- player
    def play_pausa(self):
        if self.wav_tmp is None:
            self.lb_stato.setText("Nessun audio caricato.")
            return

        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def ferma(self):
        self.player.stop()
        self.aggiorna_pos(0)

    def vai_a(self, secondi):
        if self.wav_tmp is None:
            return

        self.player.setPosition(int(secondi * 1000))
        self.aggiorna_pos(int(secondi * 1000))

    def _stato_player(self, stato):
        if stato == QMediaPlayer.PlayingState:
            self.b_play.setText("❚❚")
            self.timer.start()
        else:
            self.b_play.setText("▶")
            self.timer.stop()
            self.aggiorna_pos(self.player.position())

    def _tick(self):
        self.aggiorna_pos(self.player.position())

    def aggiorna_pos(self, ms):
        t = ms / 1000.0
        self.forma.imposta_pos(t)
        if not self.sl_pos.trascina:
            self.sl_pos.setValue(int(ms))
        self.lb_tempo.setText("{} / {}".format(fmt_tempo(t), fmt_tempo(getattr(self, "durata", 0.0))))

        o = getattr(self, "onda", None)
        if o is not None and o.attiva:
            o.t = self._tempo_onda(ms)
            self._aggiorna_schermi()

        if self.t0:
            idx = bisect_right(self.t0, t) - 1
            in_corso = idx >= 0 and t <= self.t1[idx]
            self.ev_o.aggiorna(idx, in_corso)
            self.ev_h.aggiorna(idx, in_corso)

        if self.t0_t:
            it = bisect_right(self.t0_t, t) - 1
            self.ev_t.aggiorna_frazione(it, self._frazione_trad(it, t) if it >= 0 else 0.0)

        self._aggiorna_sub(t)

    def salta_a_testo(self, ev, pos, tempi=None):
        if not ev.attivo or self.wav_tmp is None:
            return

        tempi = self.t0 if tempi is None else tempi
        starts = [s[0] for s in ev.spans]
        i = min(max(0, bisect_right(starts, pos) - 1), len(tempi) - 1)

        self.vai_a(tempi[i])

        if self.player.state() != QMediaPlayer.PlayingState:
            self.player.play()

    # ------------------------------------------------------------------ batch
    def _parametri(self, path, intero=False):
        """Parametri del Worker dalle impostazioni correnti. Con intero=True
        (batch) l'intervallo Da/A viene ignorato: si trascrive tutto il file."""
        return dict(
            file=path,
            modello=self.cb_modello.currentText(),
            lingua=LINGUE[self.cb_lingua.currentIndex()][1],
            device=self.cb_device.currentText(),
            inizio=0.0 if intero else self.sp_inizio.value(),
            fine=0.0 if intero else self.sp_fine.value(),
            batch=bool(intero),
            timestamp=self.ck_timestamp.isChecked(),
            a_capo=self.ck_acapo.isChecked(),
            parole=self.ck_parole.isChecked(),
            prefisso=self.ck_prefisso.isChecked(),
            doppio=self.ck_doppio.isEnabled() and self.ck_doppio.isChecked(),
            trad_modello=self.cb_trad.currentIndex(),
            lm_url=self.ed_lm_url.text().strip(),
            lm_modello=self.ed_lm_modello.text().strip(),
            lm_contesto=self.ck_lm_ctx.isChecked(),
            lm_tutto=self.ck_lm_tutto.isChecked(),
            lm_veloce=self.ck_lm_veloce.isChecked(),
            scarica_whisper=self.ck_scarica_whisper.isChecked(),
            ls_exe=self.ed_ls_exe.text().strip(),
            ls_gguf=self.ed_ls_gguf.text().strip(),
            ls_ctx=self.ed_ls_ctx.text().strip(),
            ls_ngl=self.ed_ls_ngl.text().strip(),
            ls_extra=self.ed_ls_extra.text().strip(),
            musica=self.ck_musica.isChecked(),
            demucs=self.ck_demucs.isChecked(),
            separatore=self.cb_sep.currentData(),
            prompt_iniziale=self.ed_prompt.text().strip(),
        )

    def _batch_etichetta(self, item, icona):
        nome = os.path.basename(item.data(Qt.UserRole))
        item.setText("{}  {}".format(icona, nome) if icona else nome)

    def _batch_conteggio(self):
        n = self.lst_batch.count()
        self.lb_batch.setText("{} file in coda".format(n) if n else "Coda vuota")
        libero = not self._batch_attivo
        self.b_batch.setEnabled(libero and n > 0)
        self.b_b_rimuovi.setEnabled(libero and n > 0)
        self.b_b_svuota.setEnabled(libero and n > 0)

    def _batch_ui(self):
        """Abilita/disabilita i controlli in base al batch in corso."""
        libero = not self._batch_attivo
        self.b_b_file.setEnabled(libero)
        self.b_b_cart.setEnabled(libero)
        self.ck_salta.setEnabled(libero)
        self.b_batch_stop.setEnabled(not libero)
        self.b_avvia.setEnabled(libero)
        self.b_libera.setEnabled(libero)
        if libero:
            self.b_batch_stop.setText("■ Interrompi dopo il file corrente")
        self._batch_conteggio()

    def _aggiungi_batch(self, percorsi):
        if self._batch_attivo:
            return
        presenti = {self.lst_batch.item(i).data(Qt.UserRole) for i in range(self.lst_batch.count())}
        nuovi = 0
        for p in percorsi:
            for f in trova_media(p):
                f = os.path.abspath(f)
                if f in presenti:
                    continue
                presenti.add(f)
                it = QListWidgetItem()
                it.setData(Qt.UserRole, f)
                it.setToolTip(f)
                self._batch_etichetta(it, "")
                self.lst_batch.addItem(it)
                nuovi += 1
        self._batch_conteggio()
        if nuovi:
            self.lb_stato.setText("Aggiunti {} file alla coda del batch.".format(nuovi))
        else:
            self.lb_stato.setText("Nessun nuovo file audio/video da aggiungere.")

    def scegli_batch_file(self):
        fs, _ = QFileDialog.getOpenFileNames(
            self, "Aggiungi file al batch", self.ultima_cartella,
            FILTRO_FILE,
            options=QFileDialog.DontUseNativeDialog
        )
        if fs:
            if os.path.isdir(os.path.dirname(fs[0])):
                self.ultima_cartella = os.path.dirname(fs[0])
            self._aggiungi_batch(fs)

    def scegli_batch_cartella(self):
        d = QFileDialog.getExistingDirectory(
            self, "Aggiungi cartella al batch", self.ultima_cartella,
            options=QFileDialog.DontUseNativeDialog | QFileDialog.ShowDirsOnly
        )
        if d:
            self.ultima_cartella = d
            self._aggiungi_batch([d])

    def rimuovi_batch(self):
        if self._batch_attivo:
            return
        for it in self.lst_batch.selectedItems():
            self.lst_batch.takeItem(self.lst_batch.row(it))
        self._batch_conteggio()

    def svuota_batch(self):
        if self._batch_attivo:
            return
        self.lst_batch.clear()
        self._batch_conteggio()

    def avvia_batch(self):
        if self._batch_attivo:
            return
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Trascrizione in corso", "Aspetta la fine della trascrizione corrente.")
            return
        n = self.lst_batch.count()
        if n == 0:
            QMessageBox.warning(self, "Coda vuota", "Aggiungi prima almeno un file alla coda del batch.")
            return

        self._salva_impostazioni()
        _STOP.clear()
        for i in range(n):
            self._batch_etichetta(self.lst_batch.item(i), "")

        # con la traduzione attiva il batch si fa in due fasi: prima TUTTE le trascrizioni (Whisper
        # caricato una volta sola), poi Whisper viene scaricato e si fanno TUTTE le traduzioni
        pb = self._parametri("", intero=True)
        self._batch_params = pb
        self._batch_due_fasi = bool(pb["doppio"]) and pb["lingua"] not in (None, "ja")
        self._batch_fase = 1
        self._batch_lavori = []
        self._batch_interrotto1 = False

        self._batch_i = -1
        self._batch_stop = False
        self._batch_esiti = dict(ok=0, salt=0, vuoti=[], err=[], avvisi=[], note=[])
        self._batch_attivo = True
        self._batch_ui()
        self.barra.setRange(0, n)
        self.barra.setValue(0)
        self._batch_avanti()

    def ferma_batch(self):
        self._batch_stop = True
        if self._batch_fase == 2 and self._batch_trad is not None:
            self._batch_trad.fermo = True
        self.b_batch_stop.setEnabled(False)
        self.b_batch_stop.setText("Mi fermo dopo il file corrente...")

    def _batch_avanti(self):
        """Passa al file successivo (saltando quelli da non rifare) o chiude il batch."""
        if not self._batch_attivo:
            return
        n = self.lst_batch.count()
        while True:
            self._batch_i += 1
            if self._batch_stop or self._batch_i >= n:
                self._batch_fine_fase1()
                return
            it = self.lst_batch.item(self._batch_i)
            path = it.data(Qt.UserRole)
            if not os.path.isfile(path):
                self._batch_etichetta(it, "✗")
                self._batch_esiti["err"].append((path, "file non trovato"))
            elif self.ck_salta.isChecked() and self._batch_gia_fatto(path):
                self._batch_etichetta(it, "↷")
                self._batch_esiti["salt"] += 1
            else:
                break
            self.barra.setValue(self._batch_i + 1)

        self._batch_corrente = path
        self._batch_etichetta(it, "⏳")
        self.lst_batch.scrollToItem(it)

        prefisso = "Batch {}/{} · {}: ".format(self._batch_i + 1, n, os.path.basename(path))
        self.lb_stato.setText(prefisso + "avvio...")

        # il vecchio worker puo' non aver ancora finito di uscire: lo tengo vivo
        if self.worker is not None:
            self._vecchi.append(self.worker)
            self._vecchi = [t for t in self._vecchi if t.isRunning()]

        par = self._parametri(path, intero=True)
        par["solo_trascrizione"] = self._batch_due_fasi
        w = Worker(par)
        w.stato.connect(lambda m, pre=prefisso: self.lb_stato.setText(pre + m))
        w.finito.connect(self._batch_ok)
        w.errore.connect(self._batch_errore)
        self.worker = w
        w.start()

    def _batch_ok(self, r):
        if not self._batch_attivo:
            return
        it = self.lst_batch.item(self._batch_i)
        path = self._batch_corrente
        if not r["t0"]:
            # niente testo: non scrivo il json, cosi' "salta i file gia' fatti" non lo salta alla prossima
            self._batch_etichetta(it, "⚠")
            self._batch_esiti["vuoti"].append(path)
        elif self._batch_due_fasi and r.get("da_tradurre"):
            # fase 1: salvo subito la trascrizione (senza traduzione) e metto il file in coda per la fase 2
            try:
                salva_risultato(r, path)
                self._batch_etichetta(it, "✎")
                self._batch_lavori.append(dict(i=self._batch_i, path=path, r=r, fatto=False))
            except OSError as e:
                self._batch_etichetta(it, "✗")
                self._batch_esiti["err"].append((path, "salvataggio non riuscito: {}".format(e)))
        else:
            try:
                salva_risultato(r, path)
                if "traduzione" in r.get("avviso", ""):
                    self._batch_etichetta(it, "⚠")
                    self._batch_esiti["avvisi"].append((path, r["avviso"].strip()))
                else:
                    self._batch_etichetta(it, "✓")
                self._batch_esiti["ok"] += 1
            except OSError as e:
                self._batch_etichetta(it, "✗")
                self._batch_esiti["err"].append((path, "salvataggio non riuscito: {}".format(e)))
        self.barra.setValue(self._batch_i + 1)
        self._batch_avanti()

    def _batch_gia_fatto(self, path):
        """Un file e' da saltare se ha il _karaoke.json e, quando si traduce, anche la _traduzione.txt."""
        base = os.path.splitext(path)[0]
        if not os.path.isfile(base + "_karaoke.json"):
            return False
        return not self._batch_due_fasi or os.path.isfile(base + "_traduzione.txt")

    def _batch_fine_fase1(self):
        """Finite (o interrotte) le trascrizioni: scarico Whisper e passo alle traduzioni."""
        if not (self._batch_due_fasi and self._batch_lavori):
            self._batch_fine()
            return
        self._batch_interrotto1 = self._batch_stop
        self._batch_fase = 2
        self._batch_stop = False

        # lascio uscire l'ultimo Worker, poi scarico Whisper dalla memoria (VRAM compresa)
        if self.worker is not None:
            self.worker.wait(5000)
        import time as _t
        prima = _vram_usata_mb()
        libera_whisper()
        if prima is not None:
            _t.sleep(0.5)
            dopo = _vram_usata_mb()
            if dopo is not None:
                if prima - dopo >= 300:
                    self._batch_esiti["note"].append(
                        "VRAM: {:.1f} -> {:.1f} GB dopo aver scaricato Whisper".format(prima / 1024.0, dopo / 1024.0))
                else:
                    self._batch_esiti["note"].append(
                        "VRAM non rilasciata da Whisper ({:.1f} -> {:.1f} GB): CTranslate2 puo' tenerla finche' il "
                        "programma e' aperto; se il server locale non riesce a caricare il modello, riavvia hira.py"
                        .format(prima / 1024.0, dopo / 1024.0))

        n = len(self._batch_lavori)
        self.barra.setRange(0, n)
        self.barra.setValue(0)
        self.b_batch_stop.setEnabled(True)
        self.b_batch_stop.setText("■ Interrompi le traduzioni dopo il file corrente")
        self.lb_stato.setText("Fase 2/2: traduco {} file...".format(n))

        t = TraduttoreBatch(
            self._batch_params,
            [(l["r"]["da_tradurre"]["testi"], l["r"]["da_tradurre"]["t0"], l["r"]["da_tradurre"]["t1"])
             for l in self._batch_lavori])
        t.inizio.connect(self._batch_trad_inizio)
        t.stato.connect(self._batch_trad_stato)
        t.fatto.connect(self._batch_trad_fatto)
        t.finito.connect(self._batch_fine)
        self._batch_trad = t
        t.start()

    def _batch_trad_stato(self, m):
        if self._batch_attivo:
            k = self.barra.value() + 1
            n = len(self._batch_lavori)
            nome = os.path.basename(self._batch_lavori[min(k, n) - 1]["path"])
            self.lb_stato.setText("Traduzione {}/{} · {}: {}".format(min(k, n), n, nome, m))

    def _batch_trad_inizio(self, k):
        if not self._batch_attivo:
            return
        it = self.lst_batch.item(self._batch_lavori[k]["i"])
        self._batch_etichetta(it, "⏳")
        self.lst_batch.scrollToItem(it)

    def _batch_trad_fatto(self, k, trad, avviso):
        if not self._batch_attivo:
            return
        lav = self._batch_lavori[k]
        lav["fatto"] = True
        it = self.lst_batch.item(lav["i"])
        r = lav["r"]
        r["trad"] = allinea_traduzione(r["t0"], r["t1"], trad)
        r["avviso"] = (r.get("avviso", "") + avviso)
        try:
            salva_risultato(r, lav["path"])
            if trad is None or "traduzione" in avviso:
                self._batch_etichetta(it, "⚠")
                self._batch_esiti["avvisi"].append((lav["path"], (avviso.strip() or "traduzione non riuscita")))
            else:
                self._batch_etichetta(it, "✓")
            self._batch_esiti["ok"] += 1
        except OSError as e:
            self._batch_etichetta(it, "✗")
            self._batch_esiti["err"].append((lav["path"], "salvataggio non riuscito: {}".format(e)))
        lav["r"] = None  # libero la memoria del risultato
        self.barra.setValue(k + 1)

    def _batch_errore(self, msg):
        if not self._batch_attivo:
            return
        it = self.lst_batch.item(self._batch_i)
        if msg == "__interrotto__":
            msg = "interrotto dall'utente"
            self._batch_stop = True
        riga = (msg.strip().splitlines() or ["errore sconosciuto"])[0][:200]
        self._batch_etichetta(it, "✗")
        self._batch_esiti["err"].append((self._batch_corrente, riga))

        # errori che si ripeterebbero identici su ogni file: inutile proseguire
        if ("Impossibile caricare faster-whisper" in msg or "faster-whisper non installato" in msg
                or "ffmpeg non trovato" in msg):
            self._batch_stop = True

        self.barra.setValue(self._batch_i + 1)
        self._batch_avanti()

    def _batch_fine(self):
        self._batch_attivo = False
        libera_whisper()  # nel batch Whisper e' rimasto in memoria tra un file e l'altro
        if self._batch_trad is not None:  # il thread puo' non essere ancora uscito: lo tengo vivo
            self._vecchi.append(self._batch_trad)
            self._vecchi = [t for t in self._vecchi if t.isRunning()]
            self._batch_trad = None
        self._batch_ui()
        self.barra.setRange(0, 1)
        self.barra.setValue(0)

        e = self._batch_esiti
        for lav in self._batch_lavori:
            if not lav["fatto"]:
                e["avvisi"].append((lav["path"], "trascritto ma non tradotto (batch interrotto)"))
        msg = "Batch {}: {} completati, {} saltati, {} senza testo, {} con errori.".format(
            "interrotto" if (self._batch_stop or self._batch_interrotto1) else "terminato",
            e["ok"], e["salt"], len(e["vuoti"]), len(e["err"]))
        self.lb_stato.setText(msg)

        dett = ["✗ {}: {}".format(os.path.basename(p), m) for p, m in e["err"][:15]]
        dett += ["⚠ {}: nessun testo riconosciuto".format(os.path.basename(p)) for p in e["vuoti"][:15]]
        dett += ["⚠ {}: {}".format(os.path.basename(p), m) for p, m in e["avvisi"][:15]]
        dett += list(e["note"])
        if dett:
            QMessageBox.warning(self, "Batch", msg + "\n\n" + "\n".join(dett))
        else:
            QMessageBox.information(self, "Batch", msg)

    # ------------------------------------------------------- trascrizione
    def _in_corso(self):
        return (self._batch_attivo
                or (self.worker is not None and self.worker.isRunning())
                or (self._trad_solo is not None and self._trad_solo.isRunning()))

    def _aggiorna_stop(self):
        if not self._in_corso():
            _STOP.clear()          # niente da interrompere: riparte pulito
        self.b_ferma.setEnabled(self._in_corso() and not _STOP.is_set())

    def interrompi(self):
        """Ferma il processo in corso: trascrizione, traduzione o batch."""
        if not self._in_corso():
            return
        _STOP.set()
        if self._trad_solo is not None and self._trad_solo.isRunning():
            self._trad_solo.fermo = True
        if self._batch_attivo:
            self.ferma_batch()
        self.b_ferma.setEnabled(False)
        self.lb_stato.setText("Interrompo: attendo la fine del passo corrente...")

    def avvia(self):
        path = self.ed_file.text().strip()

        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "File mancante", "Scegli un file audio/video valido.")
            return

        if self._batch_attivo:
            QMessageBox.information(self, "Batch in corso", "Aspetta la fine del batch o interrompilo.")
            return

        params = self._parametri(path)

        self._salva_impostazioni()

        _STOP.clear()
        self.b_avvia.setEnabled(False)
        self.barra.setRange(0, 0)
        self.ultimo_ris = None
        self._imposta_testi("", "", [], [], [], [])

        self.worker = Worker(params)
        self.worker.stato.connect(self.lb_stato.setText)
        self.worker.finito.connect(self.fine_ok)
        self.worker.errore.connect(self.fine_errore)
        self.worker.start()

    def _ripristina(self):
        self.barra.setRange(0, 1)
        self.b_avvia.setEnabled(True)

    def _lingua_cambiata(self, *_):
        cod = LINGUE[self.cb_lingua.currentIndex()][1]
        self.ck_doppio.setEnabled(cod not in (None, "ja"))
        self.cb_trad.setEnabled(cod not in (None, "ja"))
        self.lb_trad.setEnabled(cod not in (None, "ja"))
        self._trad_ui()

    def _trad_ui(self, *_):
        """I campi del backend di traduzione sono attivi solo se e' scelto quel backend.
        Nota: con la lingua 'giapponese' il menu cb_trad e' disabilitato (la trascrizione
        giapponese non passa dalla traduzione), ma i campi vanno lasciati modificabili:
        servono comunque per preparare/conservare la configurazione."""
        tipo = MODELLI_TRAD[self.cb_trad.currentIndex()][1]
        lm, ls = tipo == "lmstudio", tipo == "llamaserver"
        for w in (self.lb_lm, self.ed_lm_url, self.ed_lm_modello):
            w.setEnabled(lm)
        for w in (self.ck_lm_ctx, self.ck_lm_tutto, self.ck_lm_veloce):
            w.setEnabled(lm or ls)
        for w in (self.lb_ls, self.ed_ls_exe, self.b_ls_exe, self.ed_ls_gguf, self.b_ls_gguf,
                  self.ed_ls_ctx, self.ed_ls_ngl, self.ed_ls_extra):
            w.setEnabled(ls)

    def _sfoglia_ls_exe(self):
        ini = cartella_iniziale(self.ed_ls_exe.text(), CARTELLE_LLAMA_EXE)
        f = dialog_file_sistema(self, "Scegli llama-server", ini, [("Tutti i file", "*")])
        if f is None:     # nessun selettore di sistema: dialogo di Qt
            f, _ = QFileDialog.getOpenFileName(
                self, "Scegli llama-server", ini, "Tutti i file (*)",
                options=QFileDialog.DontUseNativeDialog)
        if f:
            self.ed_ls_exe.setText(f)

    def _sfoglia_ls_gguf(self):
        ini = cartella_iniziale(self.ed_ls_gguf.text(), CARTELLE_GGUF)
        f = dialog_file_sistema(self, "Scegli il modello GGUF", ini,
                                [("Modelli GGUF", "*.gguf *.GGUF"), ("Tutti i file", "*")])
        if f is None:
            f, _ = QFileDialog.getOpenFileName(
                self, "Scegli il modello GGUF", ini, "Modelli GGUF (*.gguf);;Tutti i file (*)",
                options=QFileDialog.DontUseNativeDialog)
        if f:
            self.ed_ls_gguf.setText(f)

    def _imposta_testi(self, orig, hira, span_o, span_h, t0, t1, tok=None, trad=None):
        if trad and trad.get("righe"):
            if trad.get("num"):
                # traduzione numerata: il riquadro conserva i prefissi [N|mm:ss.s]; le
                # evidenziazioni coprono solo il testo, dopo il prefisso
                parti, span_t, pos = [], [], 0
                for k, (rg, ta) in enumerate(zip(trad["righe"], trad["t0"]), 1):
                    pref = "[{}|{}] ".format(k, fmt_tempo(ta))
                    parti.append(pref + rg)
                    span_t.append((pos + l16(pref), pos + l16(pref) + l16(rg)))
                    pos += l16(pref) + l16(rg) + 1
                testo_t = "\n".join(parti)
            else:
                testo_t, span_t = testo_righe(trad["righe"])
            self.trad_righe = list(trad["righe"])
            self.t0_t, self.t1_t = list(trad["t0"]), list(trad["t1"])
        else:
            testo_t, span_t = "", []
            self.trad_righe, self.t0_t, self.t1_t = [], [], []

        self._prog = True
        self.out.setPlainText(orig)
        self.out_hira.setPlainText(hira)
        self.out_trad.setPlainText(testo_t)
        self._prog = False

        self.ev_t.imposta(span_t)
        self.c5.setVisible(bool(span_t))

        self.t0, self.t1 = t0, t1

        # peso cumulato del giapponese (stessa stima di distribuisci()): serve ad
        # allineare l'illuminazione della traduzione a quella del giapponese
        if tok:
            pesi = [float(len(h)) if any(c.isalnum() for c in h) else 0.3 for h in tok["h"]]
        else:
            pesi = [1.0] * len(t0)
        self._cw = [0.0]
        for w in pesi:
            self._cw.append(self._cw[-1] + w)

        self.ev_o.imposta(span_o)
        self.ev_h.imposta(span_h)

        self._sp_o_orig, self._sp_h_orig, self._hira_orig = span_o, span_h, hira

        self.tok_sub = tok
        self.seg_range = {}
        if tok:
            for i, sg in enumerate(tok["seg"]):
                r = self.seg_range.setdefault(sg, [i, i + 1])
                r[1] = i + 1
        self._sub_chiave = None
        self._sub_set("")

    def fine_ok(self, r):
        self._ripristina()
        self.ultimo_ris = r
        self._imposta_testi(r["orig"], r["hira"], r["span_o"], r["span_h"], r["t0"], r["t1"], r.get("tok"), r.get("trad"))

        # se la trascrizione ha usato una voce isolata (temporanea, che sta per essere cancellata),
        # la teniamo in cache accanto all'audio: "Salva voce e musica" e le trascrizioni successive
        # la riutilizzano senza separare di nuovo. La musica (<nome>_musica.wav) l'ha gia' salvata
        # il worker stesso nella stessa separazione, quindi qui non serve toccarla.
        if r.get("voce_file") and r.get("voce_temporanea") and os.path.isfile(r["voce_file"]):
            audio = self.ed_file.text().strip()
            motore = self.cb_sep.currentData() or "demucs"
            dest = (os.path.splitext(audio)[0] + "_voce.wav") if audio else None
            if dest and not os.path.exists(dest):
                try:
                    shutil.copyfile(r["voce_file"], dest)
                    _scrivi_stem_meta(dest, motore, self.cb_device.currentText())
                except OSError:
                    pass

        self.lb_stato.setText(
            "Fatto{}. Premi ▶ per vedere le parole colorarsi mentre vengono pronunciate, anche sul video "
            "(doppio clic su una parola per partire da li').".format(r["avviso"])
        )

        self.aggiorna_pos(self.player.position())

        if "traduzione" in r["avviso"]:
            QMessageBox.warning(self, "Traduzione", r["avviso"].strip())

    def fine_errore(self, msg):
        self._ripristina()
        if msg == "__interrotto__":
            self.lb_stato.setText("Trascrizione interrotta.")
            return
        self.lb_stato.setText("Errore.")
        QMessageBox.critical(self, "Errore", msg)

    def _testo_o_modificato(self):
        if not self._prog and self.ev_o.attivo:
            self.ev_o.invalida()
            self.lb_stato.setText("Testo modificato: evidenziazione disattivata per questo riquadro.")

    def _testo_h_modificato(self):
        if not self._prog and self.ev_h.attivo:
            self.ev_h.invalida()
            self.lb_stato.setText("Hiragana modificato: evidenziazione disattivata per questo riquadro.")

    def _testo_t_modificato(self):
        if not self._prog and self.ev_t.attivo:
            self.ev_t.invalida()
            self.lb_stato.setText("Traduzione modificata: evidenziazione disattivata per questo riquadro.")

    def aggiorna_hiragana(self):
        if self.ev_o.attivo and self._hira_orig:
            self._prog = True
            self.out_hira.setPlainText(self._hira_orig)
            self._prog = False

            self.ev_h.imposta(self._sp_h_orig)
            self.aggiorna_pos(self.player.position())
        else:
            self._prog = True
            self.out_hira.setPlainText(a_hiragana(self.out.toPlainText()))
            self._prog = False

            self.ev_h.invalida()
            self.lb_stato.setText("Hiragana aggiornato dal testo superiore.")

    # --------------------------------------------------------------- azioni
    # ------------------------------------------------------- karaoke (.json)
    def _scrivi_json(self, percorso):
        scrivi_karaoke_json(self.ultimo_ris, self.ed_file.text().strip(), percorso)

    @staticmethod
    def _valida_karaoke(d):
        for k in ("orig", "hira", "span_o", "span_h", "t0", "t1", "tok"):
            if k not in d:
                raise ValueError("File karaoke non valido: manca '{}'.".format(k))
        n = len(d["t0"])
        tok = d["tok"]
        if not (len(d["t1"]) == len(d["span_o"]) == len(d["span_h"]) == n
                and all(len(tok.get(k, [])) == n for k in ("o", "h", "seg"))):
            raise ValueError("File karaoke non valido: dati incoerenti.")

    def _applica_karaoke(self, d):
        # Espandi solo i dati nel formato compatto v2 ("versione" >= 2 con "s");
        # i dati già espansi (v1 o output di karaoke_espandi, che rimuove "s")
        # passano senza doppia espansione.
        if isinstance(d, dict) and d.get("versione", 1) >= 2 and "s" in d:
            d = karaoke_espandi(d)
        self._valida_karaoke(d)
        r = dict(
            orig=d["orig"], hira=d["hira"],
            span_o=[tuple(x) for x in d["span_o"]],
            span_h=[tuple(x) for x in d["span_h"]],
            t0=list(d["t0"]), t1=list(d["t1"]),
            tok=d["tok"], avviso="", trad=d.get("trad"),
        )
        self.ultimo_ris = r
        self._imposta_testi(r["orig"], r["hira"], r["span_o"], r["span_h"],
                            r["t0"], r["t1"], r["tok"], r["trad"])
        self.aggiorna_pos(self.player.position())

    def _cerca_karaoke(self, media):
        pj = os.path.splitext(media)[0] + "_karaoke.json"
        if not os.path.isfile(pj):
            return
        try:
            with open(pj, encoding="utf-8") as fh:
                self._applica_karaoke(json.load(fh))
            self.lb_stato.setText(
                "Karaoke caricato da {}: premi ▶ (non serve trascrivere di nuovo).".format(
                    os.path.basename(pj)))
        except Exception as e:
            self.lb_stato.setText("Karaoke salvato non utilizzabile: {}".format(e))

    def scegli_karaoke(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Carica karaoke", os.path.dirname(self.ed_file.text().strip()),
            "Karaoke (*.json);;Tutti i file (*)",
            options=QFileDialog.DontUseNativeDialog
        )
        if f:
            self.carica_karaoke_json(f)

    def carica_karaoke_json(self, pj):
        try:
            with open(pj, encoding="utf-8") as fh:
                raw = json.load(fh)
            # Espandi solo se necessario (formato compatto v2 con "versione" >= 2);
            # i dati già espansi o in formato v1 passano senza doppia espansione.
            if isinstance(raw, dict) and raw.get("versione", 1) >= 2 and "s" in raw:
                dati = karaoke_espandi(raw)
            else:
                dati = raw
            self._valida_karaoke(dati)
        except Exception as e:
            QMessageBox.warning(self, "Karaoke", "Impossibile leggere il file:\n{}".format(e))
            return

        media = dati.get("percorso_file") or ""
        if not os.path.isfile(media):
            media = os.path.join(os.path.dirname(pj), dati.get("file", ""))

        if os.path.isfile(media):
            if media != self.file_corrente:
                self.imposta_file(media)
        else:
            QMessageBox.information(
                self, "File non trovato",
                "Il file audio/video originale non e' stato trovato.\n"
                "Scegline uno con 'Sfoglia...': testi e sincronizzazione sono comunque caricati.")

        self._applica_karaoke(dati)
        self.lb_stato.setText("Karaoke caricato da {}.".format(os.path.basename(pj)))

    def copia_numerata(self):
        r = self.ultimo_ris
        if not r or not r.get("tok") or not r.get("t0"):
            QMessageBox.information(self, "Copia con numeri",
                                    "Prima trascrivi un file o carica un karaoke.")
            return
        testo, n = testo_numerato(r["tok"], r["t0"])
        QApplication.clipboard().setText(testo)
        self.lb_stato.setText(
            "Copiati {} segmenti numerati [N|tempo]. Chiedi al traduttore di tradurre solo il testo, "
            "una riga per riga, tenendo i prefissi; poi incolla in TRADUZIONE e premi Applica.".format(n))

    def copia(self):
        QApplication.clipboard().setText(self.out.toPlainText())
        self.lb_stato.setText("Trascrizione originale copiata negli appunti.")

    def applica_traduzione(self):
        """Rende 'ufficiale' il testo del riquadro Traduzione (incollato o modificato a mano)."""
        r = self.ultimo_ris
        if not r or not r.get("tok") or not r.get("t0"):
            QMessageBox.information(
                self, "Traduzione",
                "Prima trascrivi un file o carica un karaoke: la traduzione va agganciata ai tempi del giapponese.")
            return
        righe = self.out_trad.toPlainText().replace("\r", "").split("\n")
        while righe and not righe[-1].strip():
            righe.pop()
        if not righe:
            QMessageBox.information(self, "Traduzione", "Il riquadro della traduzione e' vuoto.")
            return
        tok, t0, t1 = r["tok"], r["t0"], r["t1"]
        diviso = False
        testo_riquadro = self.out_trad.toPlainText()
        iv = leggi_intervalli_tempo(testo_riquadro)
        if iv is not None:
            # frasi con '[inizio -> fine]': il testo mostrato e' senza timestamp, i tempi decidono
            # quando appare la frase; il riquadro non viene modificato
            trad = dict(righe=iv[0], t0=iv[1], t1=iv[2], fissa=True)
            r["trad"] = trad
            self.trad_righe = list(trad["righe"])
            self.t0_t, self.t1_t = list(trad["t0"]), list(trad["t1"])
            self.ev_t.imposta(iv[3])
            self.c5.setVisible(True)
            self._sub_chiave = None
            self._sub_set("")
            self.aggiorna_pos(self.player.position())
            self.lb_stato.setText("Traduzione applicata: {} frasi con i tempi del testo. Premi 'Salva "
                                  "accanto' per scriverla nei file.".format(len(trad["righe"])))
            return
        segs = segmenti_giapponesi(tok["seg"])
        num = leggi_numerate_tempi(testo_riquadro, len(segs))
        if num is not None:
            righe = num[0]
        if num is None and len(segs) >= 3 and len(righe) < max(2, len(segs) // 3):
            ris = QMessageBox.question(
                self, "Traduzione senza a capo",
                "La traduzione ha solo {} righe, mentre il giapponese ha {} segmenti: sembra un unico "
                "paragrafo, che resterebbe sullo schermo per tutto il video.\n\n"
                "Vuoi che la divida in frasi e la distribuisca sul tempo del giapponese?".format(
                    len(righe), len(segs)))
            if ris != QMessageBox.Yes:
                return
            righe = dividi_frasi("\n".join(righe))
            diviso = True
        if len(righe) == len(segs):
            trad = dict(righe=righe, t0=[t0[a] for a, b in segs], t1=[t1[b - 1] for a, b in segs])
            esito = "una riga per segmento, sincronizzata con il giapponese"
            if num is not None and any(v is not None for v in num[3]):
                # i tempi scritti nei prefissi [N|mm:ss.s] decidono quando mostrare la frase
                a0 = [v if v is not None else t0[a] for v, (a, b) in zip(num[3], segs)]
                prev = 0.0
                for i in range(len(a0)):
                    a0[i] = max(a0[i], prev)
                    prev = a0[i]
                a1 = []
                for i, (a, b) in enumerate(segs):
                    fine = t1[b - 1]
                    prossimo = a0[i + 1] if i + 1 < len(a0) else None
                    if fine <= a0[i]:
                        fine = prossimo if prossimo is not None else a0[i] + 3.0
                    if prossimo is not None:
                        fine = min(fine, prossimo)
                    a1.append(max(fine, a0[i]))
                trad = dict(righe=righe, t0=a0, t1=a1, fissa=True)
            if num is not None:
                trad["num"] = True
                esito = "{} segmenti su {} agganciati per numero ({})".format(
                    num[1], len(segs),
                    "tempi presi dai prefissi" if trad.get("fissa") else "tempi esatti")
                if num[1] < len(segs):
                    esito += ", {} senza traduzione (restano vuoti)".format(len(segs) - num[1])
                if num[2]:
                    esito += ", {} righe con numero non valido ignorate".format(num[2])
        else:
            ris = QMessageBox.question(
                self, "Righe diverse dai segmenti",
                "La traduzione ha {} righe, il giapponese {} segmenti.\n\n"
                "Posso distribuire le righe sul tempo del giapponese in modo proporzionale "
                "(sincronizzazione approssimativa). Procedo?".format(len(righe), len(segs)))
            if ris != QMessageBox.Yes:
                return
            pesi = [float(len(h)) if any(c.isalnum() for c in h) else 0.3 for h in tok["h"]]
            trad = righe_su_tempi(righe, t0, t1, pesi)
            esito = "{} righe distribuite sul tempo (sincronizzazione approssimativa)".format(len(righe))
        r["trad"] = trad
        if diviso:
            # l'unico caso in cui il testo cambia: l'utente ha chiesto di dividerlo in frasi
            self._imposta_testi(r["orig"], r["hira"], r["span_o"], r["span_h"], r["t0"], r["t1"], r["tok"], trad)
        else:
            # Applica non tocca i campi: aggancia solo le frasi agli intervalli di tempo
            if num is not None:
                spans = spans_riquadro_numerato(testo_riquadro, len(segs))
            else:
                spans = spans_riquadro_righe(testo_riquadro, len(trad["righe"]))
            self.trad_righe = list(trad["righe"])
            self.t0_t, self.t1_t = list(trad["t0"]), list(trad["t1"])
            self.ev_t.imposta(spans)
            self.c5.setVisible(bool(spans))
            self._sub_chiave = None
            self._sub_set("")
        self.aggiorna_pos(self.player.position())
        self.lb_stato.setText("Traduzione applicata: {}. Premi 'Salva accanto' per scriverla nel .txt e nel .json.".format(esito))

    def traduci_solo(self):
        """Rifa' solo la traduzione dei segmenti gia' trascritti (nessun Whisper)."""
        r = self.ultimo_ris
        if not r or not r.get("tok") or not r.get("t0"):
            QMessageBox.information(self, "Traduzione",
                                    "Prima trascrivi un file o carica un karaoke: serve il giapponese da tradurre.")
            return
        if self._batch_attivo or (self.worker is not None and self.worker.isRunning()) \
                or (self._trad_solo is not None and self._trad_solo.isRunning()):
            QMessageBox.information(self, "Traduzione", "C'e' gia' un'operazione in corso: attendi che finisca.")
            return
        lingua = LINGUE[self.cb_lingua.currentIndex()][1]
        if lingua in (None, "ja"):
            QMessageBox.information(self, "Traduzione",
                                    "Scegli come lingua una diversa dal giapponese: e' la lingua della traduzione.")
            return
        if self.out_trad.toPlainText().strip():
            ris = QMessageBox.question(self, "Traduci",
                                       "Il riquadro Traduzione contiene gia' del testo: verra' sostituito. Procedo?")
            if ris != QMessageBox.Yes:
                return
        tok, t0, t1 = r["tok"], r["t0"], r["t1"]
        segs = segmenti_giapponesi(tok["seg"])
        testi = ["".join(tok["o"][a:b]).replace("\n", " ").strip() for a, b in segs]
        tt0 = [t0[a] for a, b in segs]
        tt1 = [t1[b - 1] for a, b in segs]
        p = self._parametri(self.ed_file.text().strip())
        p["batch"] = False
        self.b_traduci.setEnabled(False)
        self.lb_stato.setText("Traduco {} segmenti in '{}'...".format(len(testi), lingua))
        _STOP.clear()
        t = TraduttoreBatch(p, [(testi, tt0, tt1)])
        t.stato.connect(self.lb_stato.setText)
        t.fatto.connect(self._traduci_solo_fatto)
        t.finito.connect(lambda: self.b_traduci.setEnabled(True))
        self._trad_solo = t
        t.start()

    def _traduci_solo_fatto(self, k, trad, avviso):
        r = self.ultimo_ris
        if not trad or not trad.get("righe") or r is None:
            self.lb_stato.setText("Traduzione interrotta." if _STOP.is_set()
                                  else "Traduzione non riuscita.{}".format(avviso or ""))
            return
        trad = allinea_traduzione(r["t0"], r["t1"], trad)
        r["trad"] = trad
        # cambia solo il riquadro Traduzione: trascrizione e hiragana non si toccano
        testo_t, span_t = testo_righe(trad["righe"])
        self._prog = True
        self.out_trad.setPlainText(testo_t)
        self._prog = False
        self.trad_righe = list(trad["righe"])
        self.t0_t, self.t1_t = list(trad["t0"]), list(trad["t1"])
        self.ev_t.imposta(span_t)
        self.c5.setVisible(bool(span_t))
        self._sub_chiave = None
        self._sub_set("")
        self.aggiorna_pos(self.player.position())
        self.lb_stato.setText("Traduzione fatta ({} righe).{} Premi 'Salva accanto' per scriverla nei file.".format(
            len(trad["righe"]), avviso or ""))

    def copia_traduzione(self):
        QApplication.clipboard().setText(self.out_trad.toPlainText())
        self.lb_stato.setText("Traduzione copiata negli appunti.")

    def copia_hiragana(self):
        QApplication.clipboard().setText(self.out_hira.toPlainText())
        self.lb_stato.setText("Hiragana copiato negli appunti.")

    def salva(self):
        base = os.path.splitext(self.ed_file.text().strip())[0] or "trascrizione"
        f, _ = QFileDialog.getSaveFileName(
            self, "Salva testo", base + ".txt", "Testo (*.txt)",
            options=QFileDialog.DontUseNativeDialog
        )

        if f:
            with open(f, "w", encoding="utf-8") as fh:
                fh.write(self.out.toPlainText())
            msg = "Salvato: " + f

            if self.ultimo_ris:
                pj = os.path.splitext(f)[0] + "_karaoke.json"
                try:
                    self._scrivi_json(pj)
                    msg += " + " + os.path.basename(pj)
                except OSError as e:
                    QMessageBox.critical(self, "Errore di salvataggio", str(e))
            self.lb_stato.setText(msg)

    def salva_accanto(self):
        audio = self.ed_file.text().strip()

        if not audio or not os.path.isfile(audio):
            QMessageBox.warning(self, "File mancante", "Scegli prima un file audio/video valido.")
            return

        txt_trad = self.out_trad.toPlainText()
        tr_ris = (self.ultimo_ris or {}).get("trad")
        if (tr_ris and tr_ris.get("righe") and self.ev_t.attivo
                and not tr_ris.get("num") and not tr_ris.get("fissa")):
            # traduzione generata dal programma (riquadro senza prefissi): il file li ha comunque
            txt_trad = traduzione_numerata(tr_ris)
        testi = [
            ("_trascrizione.txt", self.out.toPlainText()),
            ("_hiragana.txt", self.out_hira.toPlainText()),
            ("_traduzione.txt", txt_trad),
        ]

        testi = [(suff, t) for suff, t in testi if t.strip()]

        con_json = self.ultimo_ris is not None

        if not testi and not con_json:
            QMessageBox.information(self, "Niente da salvare", "Non ci sono ancora testi da salvare.")
            return

        base = os.path.splitext(audio)[0]
        nomi = [s for s, _ in testi] + (["_karaoke.json"] if con_json else [])
        esistenti = [os.path.basename(base + s) for s in nomi if os.path.exists(base + s)]

        if esistenti:
            r = QMessageBox.question(
                self, "Sovrascrivere?",
                "Esistono gia' questi file:\n" + "\n".join(esistenti) + "\nVuoi sovrascriverli?"
            )
            if r != QMessageBox.Yes:
                return

        try:
            for suff, t in testi:
                with open(base + suff, "w", encoding="utf-8") as fh:
                    fh.write(t)
            if con_json:
                self._scrivi_json(base + "_karaoke.json")
        except OSError as e:
            QMessageBox.critical(self, "Errore di salvataggio", str(e))
            return

        self.lb_stato.setText("Salvato in {}: {}".format(
            os.path.dirname(audio),
            ", ".join(os.path.basename(base + s) for s in nomi)
        ))

    def salva_voce_musica(self):
        audio = self.ed_file.text().strip()

        if not audio or not os.path.isfile(audio):
            QMessageBox.warning(self, "File mancante", "Scegli prima un file audio/video valido.")
            return

        base = os.path.splitext(audio)[0]
        dest_voce, dest_musica = base + "_voce.wav", base + "_musica.wav"

        # Voce pulita e musica gia' create da 'trascrivi' (Demucs/BS-RoFormer salvano ora
        # entrambi gli stem accanto all'originale, in un'unica passata): non ha senso
        # rifare la separazione da capo. Se esiste anche la musica (<nome>_musica.wav)
        # non c'e' proprio nulla da fare; se manca, si separa solo la musica partendo
        # dall'audio pulito che abbiamo gia'.
        voce_esistente = cerca_voce_pulita(audio)
        if voce_esistente and os.path.abspath(voce_esistente) == os.path.abspath(dest_voce):
            if os.path.exists(dest_musica):
                print("Voce e musica gia' salvate da 'trascrivi', nessuna separazione da rifare:\n  {}\n  {}".format(
                    dest_voce, dest_musica))
                self.lb_stato.setText("Nessun audio da salvare: voce e musica esistono gia' ({}, {}).".format(
                    os.path.basename(dest_voce), os.path.basename(dest_musica)))
                return
            sorgente, riuso_voce = dest_voce, True
        else:
            # nessun audio pulito creato con 'trascrivi': non c'e' niente da salvare e
            # non va rifatta la separazione di tutto il brano da capo
            print("Nessun audio pulito da salvare: trascrivi prima il file isolando la voce "
                  "(Demucs/BS-RoFormer), oppure usa 'Salva accanto all'audio'.")
            self.lb_stato.setText("Nessun audio da salvare: manca la voce pulita creata da 'trascrivi'.")
            return

        if self._batch_attivo or (self.worker is not None and self.worker.isRunning()) \
                or (self._stem is not None and self._stem.isRunning()):
            QMessageBox.information(self, "Operazione in corso", "Attendi la fine dell'operazione in corso.")
            return

        esistenti = [os.path.basename(dest_musica)] if os.path.exists(dest_musica) else []

        if esistenti:
            r = QMessageBox.question(
                self, "Sovrascrivere?",
                "Esistono gia' questi file:\n" + "\n".join(esistenti) + "\nVuoi sovrascriverli?"
            )
            if r != QMessageBox.Yes:
                return

        self.b_salva_stem.setEnabled(False)
        self.b_avvia.setEnabled(False)
        self.barra.setRange(0, 0)

        t = SeparatoreStem(sorgente, self.cb_sep.currentData() or "demucs",
                           self.cb_device.currentText(), dest_voce, dest_musica,
                           riuso_voce=riuso_voce)
        t.stato.connect(self.lb_stato.setText)
        t.fatto.connect(lambda err, d=os.path.dirname(audio), v=dest_voce, m=dest_musica:
                        self._salva_voce_musica_fatto(err, d, v, m))
        self._stem = t
        t.start()

    def _salva_voce_musica_fatto(self, err, cartella, dest_voce, dest_musica):
        self.b_salva_stem.setEnabled(True)
        self.b_avvia.setEnabled(True)
        self.barra.setRange(0, 1)

        if err:
            self.lb_stato.setText("Separazione non riuscita.")
            QMessageBox.critical(self, "Separazione voce/musica", err)
            return

        # con la voce gia' isolata da 'trascrivi' qui si e' salvata solo la musica
        if os.path.exists(dest_voce) and os.path.exists(dest_musica):
            self.lb_stato.setText("Salvato in {}: {}, {}".format(
                cartella, os.path.basename(dest_voce), os.path.basename(dest_musica)))
        else:
            self.lb_stato.setText("Salvato in {}: {}".format(
                cartella, os.path.basename(dest_musica if os.path.exists(dest_musica) else dest_voce)))

    def esporta_tracce_mka(self):
        """Unisce gli stem gia' presenti in cache (<nome>_voce.wav + <nome>_musica.wav) in un
        unico .mka multitraccia. Non separa nulla: se mancano, dice che non c'e' audio da salvare."""
        audio = self.ed_file.text().strip()
        if not audio or not os.path.isfile(audio):
            QMessageBox.warning(self, "File mancante", "Scegli prima un file audio/video valido.")
            return

        base = os.path.splitext(audio)[0]
        voce = cerca_voce_pulita(audio) or (base + "_voce.wav")
        musica = base + "_musica.wav"
        mancanti = [os.path.basename(x) for x in (voce, musica) if not os.path.isfile(x)]
        if mancanti:
            print("Nessun audio da salvare in .mka: mancano {}\n"
                  "(separa prima le tracce: trascrivi con separazione attiva, oppure usa "
                  "'Salva voce e musica').".format(", ".join(mancanti)))
            self.lb_stato.setText("Nessun audio da salvare in .mka: {} non trovati."
                                  .format(", ".join(mancanti)))
            QMessageBox.information(
                self, "Nessuno stem da esportare",
                "Non ci sono tracce in cache da esportare:\n{}\n\n"
                "Trascrivi il file con la separazione attiva, oppure premi "
                "'Salva voce e musica'.".format("\n".join(mancanti)))
            return

        lossless = False
        r = QMessageBox.question(
            self, "Esporta tracce .mka",
            "Tracce trovate:\n  {}\n  {}\n\nCodifica del file <nome>.mka:\n"
            "  Yes    -> lossless (FLAC, file piu' grande)\n"
            "  No     -> Opus 192k (compatto)\n"
            "  Cancel -> annulla".format(os.path.basename(voce), os.path.basename(musica)))
        if r == QMessageBox.Yes:
            lossless = True
        elif r != QMessageBox.No:
            return

        if self._batch_attivo or (self.worker is not None and self.worker.isRunning()) \
                or (self._stem is not None and self._stem.isRunning()) \
                or (self._mka is not None and self._mka.isRunning()):
            QMessageBox.information(self, "Operazione in corso", "Attendi la fine dell'operazione in corso.")
            return

        dest = base + ".mka"
        if os.path.exists(dest):
            r = QMessageBox.question(self, "Sovrascrivere?",
                                     "Esiste gia':\n{}\nVuoi sovrascriverlo?".format(
                                         os.path.basename(dest)))
            if r != QMessageBox.Yes:
                return

        self.b_esporta_mka.setEnabled(False)
        t = EsportatoreMka(audio, voce, musica, lossless)
        t.stato.connect(self.lb_stato.setText)
        t.fatto.connect(lambda err, d=dest, c=os.path.dirname(audio):
                        self._esporta_mka_fatto(err, d, c))
        self._mka = t
        t.start()

    def _esporta_mka_fatto(self, err, dest, cartella):
        self.b_esporta_mka.setEnabled(True)
        if err:
            self.lb_stato.setText("Esportazione .mka non riuscita.")
            QMessageBox.critical(self, "Esporta tracce .mka", err)
            return
        self.lb_stato.setText("Tracce unite in {}: {}".format(cartella, os.path.basename(dest)))

    def libera(self):
        libera_modello()
        self.lb_stato.setText("Modello scaricato dalla memoria.")

    def closeEvent(self, e):
        self._tm_frame.stop()
        self._ferma_frame()
        self._batch_stop = True
        self._batch_attivo = False
        if self._batch_trad is not None:
            self._batch_trad.fermo = True
            self._batch_trad.wait(1500)
        if self._trad_solo is not None and self._trad_solo.isRunning():
            self._trad_solo.fermo = True
            self._trad_solo.wait(1500)
        if self._stem is not None and self._stem.isRunning():
            self._stem.wait(1500)
        if self._mka is not None and self._mka.isRunning():
            self._mka.wait(1500)
        self._salva_impostazioni()
        ferma_llamaserver()
        self._fullscreen(False)
        if self.cont_fs is not None:
            self.cont_fs.hide()
        self.player.stop()
        self.player.setMedia(QMediaContent())
        self._rimuovi_wav()

        try:
            if self.worker is not None and self.worker.isRunning():
                self.worker.wait(500)
            if self.decoder is not None and self.decoder.isRunning():
                self.decoder.wait(500)
        except Exception:
            pass

        super().closeEvent(e)


if __name__ == "__main__":
    log_hira("avvio hiraFF (diagnostica del caricamento file attiva)")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    imposta_palette_scura(app)
    app.setStyleSheet(STILE)

    w = Finestra()
    w.show()

    sys.exit(app.exec_())
