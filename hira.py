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
- ffmpeg nel PATH

Installazione:
python -m pip install faster-whisper PyQt5 pykakasi numpy
"""

import os
import re
import sys
import json
from html import escape
import wave
import subprocess
import tempfile
from bisect import bisect_right
from math import ceil


def _aggiungi_path_librerie_cuda():
    """Aggiunge ai percorsi di ricerca le librerie CUDA installate via pip
    (cublas, cudnn, ...) dentro il venv corrente.

    Su Windows servono i DLL (cartelle *\\bin), su Linux/Mint servono i .so
    (cartelle */lib): qui li aggiungo a LD_LIBRARY_PATH prima che
    ctranslate2/faster-whisper provino a caricarli."""
    try:
        from importlib.metadata import distributions
    except ImportError:
        return
    is_win = os.name == "nt"
    est = (".dll",) if is_win else (".so", ".so.12")
    siti = set()
    for dist in distributions():
        nome = (dist.metadata.get("Name") or "").lower()
        if nome.startswith(("nvidia-", "torch", "triton")):
            try:
                loc = str(dist.locate_file(""))
            except Exception:
                continue
            candidati = []
            for subdir in ("bin", "lib"):
                d = os.path.join(loc, subdir)
                if os.path.isdir(d):
                    candidati.append(d)
            if not candidati and os.path.isdir(loc):
                candidati.append(loc)
            for d in candidati:
                try:
                    if any(f.endswith(est) for f in os.listdir(d)):
                        siti.add(d)
                except OSError:
                    pass
    if not siti:
        return
    if is_win:
        for sito in siti:
            try:
                os.add_dll_directory(sito)
            except (AttributeError, OSError):
                # Python < 3.8 oppure cartella inesistente: usa il PATH
                os.environ["PATH"] = sito + os.pathsep + os.environ.get("PATH", "")
    else:
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        nuove = [s for s in siti if s not in ld.split(os.pathsep)]
        if nuove:
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(nuove) + (
                os.pathsep + ld if ld else ""
            )


_aggiungi_path_librerie_cuda()

import numpy as np

from PyQt5.QtCore import Qt, QThread, QTimer, QUrl, QPointF, QRectF, QSizeF, pyqtSignal
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QKeySequence, QPainter, QPalette, QPen, QPolygonF, QTextCursor,
    QTextDocument, QTextOption,
)
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QLineEdit, QPushButton, QComboBox, QCheckBox, QDoubleSpinBox, QTextEdit,
    QFileDialog, QProgressBar, QMessageBox, QFrame, QSlider, QSplitter,
    QScrollArea, QShortcut, QStyle, QGraphicsView, QGraphicsScene,
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

SR = 16000  # frequenza di campionamento usata per decodifica/riproduzione

# colori
C_BG = "#14151c"
C_CARD = "#1d1f2b"
C_BORDER = "#2b2e3f"
C_TEXT = "#e6e8f2"
C_MUTED = "#8b90a8"
C_ACCENT = "#7c6cff"      # parte non ancora riprodotta
C_SPOKEN = "#ffb454"      # parte gia' pronunciata

STILE = """
#finestra, #interno { background: %(bg)s; }
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
QPushButton { background: #2a2d40; border: none; border-radius: 8px; padding: 8px 14px; }
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
QSplitter::handle { background: transparent; }
QToolTip { background: %(card)s; color: %(text)s; border: 1px solid %(border)s; padding: 4px; }
""" % dict(bg=C_BG, card=C_CARD, border=C_BORDER, text=C_TEXT, muted=C_MUTED, accent=C_ACCENT)


# cache del modello faster-whisper caricato
_cache = {"key": None, "model": None}
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


def libera_modello():
    """Scarica il modello dalla memoria."""
    _cache["key"] = None
    _cache["model"] = None

    try:
        import gc
        gc.collect()
    except Exception:
        pass

    # Se PyTorch è comunque installato, libera anche la sua cache CUDA.
    # faster-whisper usa CTranslate2, ma questo non guasta.
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


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
    except Exception:
        pass
    return False


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
# Thread: decodifica audio + calcolo livelli in dB
# ---------------------------------------------------------------------------
class Decoder(QThread):
    pronto = pyqtSignal(str, str, object, float, float, bool)  # file, wav, db, hop, durata, ha_video
    errore = pyqtSignal(str, str)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        wav = None
        try:
            wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", self.path, "-vn",
                 "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", wav],
                check=True, stdin=subprocess.DEVNULL, capture_output=True)

            with wave.open(wav, "rb") as w:
                raw = w.readframes(w.getnframes())

            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            win = SR // 100                      # finestre da 10 ms
            m = len(x) // win

            if m == 0:
                raise ValueError("Audio vuoto o non leggibile (il file ha una traccia audio?).")

            rms = np.sqrt(np.mean(x[:m * win].reshape(m, win) ** 2, axis=1))
            db = np.clip(20.0 * np.log10(rms + 1e-9), -60.0, 0.0)

            self.pronto.emit(self.path, wav, db, 0.01, len(x) / float(SR),
                             ha_flusso_video(self.path))

        except FileNotFoundError:
            self.errore.emit(self.path, "ffmpeg non trovato: installalo e aggiungilo al PATH.")
        except subprocess.CalledProcessError as e:
            self.errore.emit(self.path, (e.stderr or b"").decode(errors="replace") or str(e))
        except Exception as e:
            self.errore.emit(self.path, str(e))


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
            # int8_float16: buon compromesso VRAM/qualità
            # float16: classico
            # int8: fallback
            attempts.extend([
                ("cuda", "int8_float16"),
                ("cuda", "float16"),
                ("cuda", "int8"),
            ])

        if requested in ("auto", "cpu"):
            attempts.append(("cpu", "int8"))

        # Se l'utente ha chiesto esplicitamente CUDA ma non funziona,
        # permette comunque un fallback CPU.
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
                    libera_modello()

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
                    # Mancano le librerie CUDA: potrebbe essere un problema di
                    # percorso di ricerca (DLL su Windows, .so via
                    # LD_LIBRARY_PATH su Linux) dei pacchetti pip nel venv.
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

    def run(self):
        tmp = None
        try:
            p = self.p
            path = p["file"]

            # Estrazione eventuale del segmento Da/A
            if p["inizio"] > 0 or p["fine"] > 0:
                self.stato.emit("Estraggo il segmento con ffmpeg...")
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                cmd = ["ffmpeg", "-y", "-v", "error", "-ss", str(p["inizio"]), "-i", path]

                if p["fine"] > p["inizio"]:
                    cmd += ["-t", str(p["fine"] - p["inizio"])]

                cmd += ["-ar", "16000", "-ac", "1", tmp]
                subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL, capture_output=True)
                path = tmp

            model, device, compute_type = self._load_model()

            self.stato.emit("Trascrivo con faster-whisper...")

            use_words = bool(p["parole"])

            common = dict(
                language=p["lingua"],
                task="transcribe",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                condition_on_previous_text=False,
                initial_prompt=(p["prompt_iniziale"] or None),
                temperature=0.0,
                compression_ratio_threshold=2.0,
            )

            def _consume(word_timestamps):
                segments_iter, info = model.transcribe(
                    path,
                    word_timestamps=word_timestamps,
                    **common
                )
                return list(segments_iter), info

            raw_segments = None
            info = None

            # Primo tentativo con word_timestamps.
            # Se fallisce, riprova senza word_timestamps.
            try:
                raw_segments, info = _consume(use_words)
            except Exception as e:
                if use_words:
                    self.stato.emit(
                        "Word timestamps non disponibili: uso sincronizzazione per segmento."
                    )
                    use_words = False
                    raw_segments, info = _consume(False)
                else:
                    raise

            lingua = p["lingua"] or getattr(info, "language", "") or ""

            self.stato.emit("Converto in hiragana e calcolo la sincronizzazione...")

            avviso = ""
            try:
                get_kakasi()
                kk_ok = True
            except ImportError:
                kk_ok = False
                avviso = " (pykakasi mancante: hiragana non convertito)"

            def hira_di(t):
                if not kk_ok:
                    return t
                try:
                    return "".join(i["hira"] for i in get_kakasi().convert(t))
                except Exception:
                    return t

            off = p["inizio"]
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

            # tempi monotoni (necessari per la ricerca binaria durante la riproduzione)
            prev = 0.0
            for n, (s0, s1, toks) in enumerate(segmenti):
                nuovi = []
                for o, h, a, b in toks:
                    a = max(a, prev)
                    b = max(b, a)
                    prev = a
                    nuovi.append((o, h, a, b))
                segmenti[n] = (s0, s1, nuovi)

            # costruzione dei due testi + posizioni (UTF-16) di ogni token
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
                tok=dict(o=tok_o, h=tok_h, seg=tok_seg),
            ))

        except FileNotFoundError:
            self.errore.emit("ffmpeg non trovato: installalo e aggiungilo al PATH.")

        except subprocess.CalledProcessError as e:
            self.errore.emit((e.stderr or b"").decode(errors="replace") or str(e))

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

    # --- dati ---
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

    # --- geometria ---
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

    # --- disegno ---
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

        # griglia dB (asse Y)
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

        # griglia tempo (asse X)
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

        # intervallo "Da/A"
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

        # forma d'onda: massimo per ogni colonna di pixel
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

        # cursore di riproduzione
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

    # --- mouse ---
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


class VideoSchermo(QGraphicsView):
    """Visualizzatore video basato su QGraphicsVideoItem: viene disegnato da Qt
    come un normale widget (niente finestra nativa sovrapposta), quindi non si
    blocca nel passaggio a schermo intero."""
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
            # seconda vista sulla STESSA scena (usata per lo schermo intero):
            # il video del player non viene mai spostato ne' riagganciato
            self._scena = sorgente._scena
            self.item = sorgente.item
        else:
            self._scena = QGraphicsScene(self)
            self.item = QGraphicsVideoItem()
            self._scena.addItem(self.item)
        self.setScene(self._scena)
        self._sub_html = ""
        self._sub_cache = None
        self.item.nativeSizeChanged.connect(self._adatta)

    def _adatta(self, *_):
        sz = self.item.nativeSize()
        if sz.isValid() and not sz.isEmpty():
            self.item.setSize(QSizeF(sz))
            self._scena.setSceneRect(self.item.boundingRect())
            self.fitInView(self.item, Qt.KeepAspectRatio)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._adatta()

    def mouseDoubleClickEvent(self, e):
        self.doppio_clic.emit()

    # --- sottotitoli sovrapposti al video (testo colorato con contorno nero)
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

    def paintEvent(self, e):
        super().paintEvent(e)
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
        # chiusura (Alt+F4) da schermo intero: esce solo dal fullscreen
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
        self._vecchi = []              # thread in chiusura (evita distruzione prematura)
        self.wav_tmp = None
        self.file_corrente = ""
        self.modo_video = False        # True se il player riproduce il video originale
        self._fs = False               # True se il video e' a schermo intero
        self.cont_fs = None            # finestra a schermo intero (creata alla prima richiesta)
        self.ultimo_ris = None         # ultimo risultato di trascrizione (dati del karaoke)
        self.tok_sub = None            # token per i sottotitoli karaoke sul video
        self.seg_range = {}
        self._sub_chiave = None
        self.t0 = []
        self.t1 = []
        self._prog = False             # True mentre il testo viene impostato da codice

        self.player = QMediaPlayer(None, QMediaPlayer.LowLatency)
        self.player.setVolume(80)

        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._tick)

        # tutto il contenuto sta dentro una scroll area: se lo schermo e'
        # piccolo si scorre invece di uscire dai bordi
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

        az.addWidget(self.b_avvia)
        self.b_carica_k = QPushButton("📂 Carica karaoke...")
        self.b_carica_k.setToolTip(
            "Carica un file _karaoke.json salvato in precedenza: riapre il video/audio "
            "e ripristina testi e sincronizzazione senza trascrivere di nuovo"
        )
        self.b_carica_k.clicked.connect(self.scegli_karaoke)

        az.addWidget(self.b_salva_accanto)
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

        # --- player ---
        c2, l2 = card("Player")

        self.l2, self.c2 = l2, c2
        self.cont_video = ContenitoreVideo()
        self.video = self.cont_video.video
        self._collega_cont(self.cont_video)
        self.cont_video.hide()
        self.player.setVideoOutput(self.video.item)
        l2.addWidget(self.cont_video, 1)

        # il grafico dell'audio non e' piu' mostrato (resta solo come contenitore
        # dei dati): al suo posto c'e' una barra di posizione
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
        self.cb_sub.addItems(["Originale + Hiragana", "Solo originale", "Solo hiragana", "Nessun testo"])
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

        h3.addWidget(lb3)
        h3.addStretch(1)
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
        root.addWidget(split, 1)

        self.ev_o = Evidenziatore(self.out)
        self.ev_h = Evidenziatore(self.out_hira)

        self._sp_o_orig, self._sp_h_orig, self._hira_orig = [], [], ""

        self._mostra_video(False)
        self._cambia_sub()

        self.sc_f = QShortcut(QKeySequence("F"), self)
        self.sc_f.setContext(Qt.WindowShortcut)
        self.sc_f.activated.connect(self._tasto_f)

    # ------------------------------------------------------------------ file
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            f = urls[0].toLocalFile()
            if f.lower().endswith(".json"):
                self.carica_karaoke_json(f)
            else:
                self.imposta_file(f)

    def scegli_file(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Scegli un file", "",
            FILTRO_FILE
        )
        if f:
            self.imposta_file(f)

    def imposta_file(self, path):
        self.ed_file.setText(path)
        self._file_modificato()

    def _file_modificato(self):
        path = self.ed_file.text().strip()
        if path and path != self.file_corrente and os.path.isfile(path):
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
        self.player.stop()
        self.player.setMedia(QMediaContent())
        self._rimuovi_wav()
        self._mostra_video(False)

        self.forma.pulisci("Decodifico l'audio...")
        self.sl_pos.setRange(0, 0)
        self.lb_stato.setText("Decodifico il file...")
        self.lb_tempo.setText("00:00.0 / 00:00.0")

        if self.decoder is not None:
            self._vecchi.append(self.decoder)
            self._vecchi = [t for t in self._vecchi if t.isRunning()]

        self.decoder = Decoder(path)
        self.decoder.pronto.connect(self._audio_pronto)
        self.decoder.errore.connect(self._audio_errore)
        self.decoder.start()

    def _audio_pronto(self, path, wav, db, hop, durata, ha_video):
        if path != self.file_corrente:       # nel frattempo e' stato scelto un altro file
            try:
                os.remove(wav)
            except OSError:
                pass
            return

        self.wav_tmp = wav
        self.durata = durata
        self.forma.imposta(db, hop, durata)

        # Video: il player riproduce il file originale (immagine + audio
        # sincronizzati). Audio: riproduce il wav decodificato.
        self.modo_video = bool(ha_video)
        self._mostra_video(self.modo_video)
        sorgente = path if self.modo_video else wav
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(sorgente)))
        self.player.setPlaybackRate(float(self.cb_vel.currentText().rstrip("x")))

        self.sl_pos.setRange(0, int(durata * 1000))
        self._aggiorna_selezione()
        self.aggiorna_pos(0)

    def _audio_errore(self, path, msg):
        if path == self.file_corrente:
            self.forma.pulisci("Impossibile decodificare l'audio")
            self.lb_stato.setText("Errore di decodifica audio.")
            QMessageBox.warning(self, "Audio", msg)

    def _mostra_video(self, on):
        """Mostra/nasconde il riquadro video e i suoi controlli."""
        self.modo_video = bool(on)
        if not on:
            self._fullscreen(False)
        self.cont_video.setVisible(bool(on))
        self.root.setStretchFactor(self.c1, 0)
        self.root.setStretchFactor(self.c2, 4 if on else 0)
        self.lb_sub.setVisible(bool(on))
        self.cb_sub.setVisible(bool(on))
        self.b_fs.setVisible(bool(on))
        self._sub_chiave = None

    def _tasto_f(self):
        # non rubare la lettera "f" mentre si scrive in una casella
        w = QApplication.focusWidget()
        if isinstance(w, (QLineEdit, QTextEdit, QDoubleSpinBox)):
            return
        self.schermo_intero()

    def schermo_intero(self):
        if self.modo_video:
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
        """Schermo intero: una seconda finestra mostra la STESSA scena del
        video. Il video del player non viene toccato (spostarlo/riagganciarlo
        mentre e' in riproduzione causava 'Internal data stream error')."""
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
        self._righe_set({0: 4, 1: 2, 2: 2}.get(m, 0))
        self._sub_chiave = None
        self.aggiorna_pos(self.player.position())

    def _html_riga(self, tok, a, b, idx, in_corso):
        out = []
        for i in range(a, b):
            tx = escape(tok[i])
            if i <= idx:
                out.append('<span style="color:%s;">%s</span>' % (C_SPOKEN, tx))
            else:
                out.append('<span style="color:#ffffff;">%s</span>' % tx)
        return '<div style="white-space: pre-wrap;">%s</div>' % "".join(out)

    def _aggiorna_sub(self, t):
        """Sottotitoli karaoke sul video: la frase corrente, con le parole
        gia' pronunciate colorate e quella in corso evidenziata."""
        if not self.modo_video or not self.tok_sub or not self.t0:
            return

        m = self.cb_sub.currentIndex()
        idx = bisect_right(self.t0, t) - 1
        chiave = None

        if m != 3 and idx >= 0:
            in_corso = t <= self.t1[idx]
            if in_corso or t - self.t1[idx] <= 1.5:
                chiave = (idx, in_corso, m)

        if chiave == self._sub_chiave:
            return
        self._sub_chiave = chiave

        if chiave is None:
            self._sub_set("")
            return

        a, b = self.seg_range[self.tok_sub["seg"][idx]]
        righe = []
        if m in (0, 1):
            righe.append(self._html_riga(self.tok_sub["o"], a, b, idx, in_corso))
        if m in (0, 2):
            righe.append(self._html_riga(self.tok_sub["h"], a, b, idx, in_corso))
        self._sub_set("".join(righe))

    def _errore_player(self, _codice):
        """Il backend multimediale non riesce a riprodurre il video (codec
        mancanti): ripiega sul solo audio gia' decodificato."""
        if self.modo_video and self.wav_tmp and os.path.exists(self.wav_tmp):
            self._mostra_video(False)
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.wav_tmp)))
            self.player.setPlaybackRate(float(self.cb_vel.currentText().rstrip("x")))
            self.lb_stato.setText(
                "Video non riproducibile dal backend multimediale: uso solo l'audio."
            )
        elif not self.modo_video:
            self.lb_stato.setText("Errore del player: " + self.player.errorString())

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

        if self.t0:
            idx = bisect_right(self.t0, t) - 1
            in_corso = idx >= 0 and t <= self.t1[idx]
            self.ev_o.aggiorna(idx, in_corso)
            self.ev_h.aggiorna(idx, in_corso)

        self._aggiorna_sub(t)

    def salta_a_testo(self, ev, pos):
        """Doppio clic su una parola: riproduce da li'."""
        if not ev.attivo or self.wav_tmp is None:
            return

        starts = [s[0] for s in ev.spans]
        i = max(0, bisect_right(starts, pos) - 1)

        self.vai_a(self.t0[i])

        if self.player.state() != QMediaPlayer.PlayingState:
            self.player.play()

    # ------------------------------------------------------- trascrizione
    def avvia(self):
        path = self.ed_file.text().strip()

        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "File mancante", "Scegli un file audio/video valido.")
            return

        params = dict(
            file=path,
            modello=self.cb_modello.currentText(),
            lingua=LINGUE[self.cb_lingua.currentIndex()][1],
            device=self.cb_device.currentText(),
            inizio=self.sp_inizio.value(),
            fine=self.sp_fine.value(),
            timestamp=self.ck_timestamp.isChecked(),
            a_capo=self.ck_acapo.isChecked(),
            parole=self.ck_parole.isChecked(),
            prefisso=self.ck_prefisso.isChecked(),
            prompt_iniziale=self.ed_prompt.text().strip(),
        )

        self.b_avvia.setEnabled(False)
        self.barra.setRange(0, 0)   # barra indeterminata
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

    def _imposta_testi(self, orig, hira, span_o, span_h, t0, t1, tok=None):
        self._prog = True
        self.out.setPlainText(orig)
        self.out_hira.setPlainText(hira)
        self._prog = False

        self.t0, self.t1 = t0, t1

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
        self._imposta_testi(r["orig"], r["hira"], r["span_o"], r["span_h"], r["t0"], r["t1"], r.get("tok"))

        self.lb_stato.setText(
            "Fatto{}. Premi ▶ per vedere le parole colorarsi mentre vengono pronunciate, anche sul video "
            "(doppio clic su una parola per partire da li').".format(r["avviso"])
        )

        self.aggiorna_pos(self.player.position())

    def fine_errore(self, msg):
        self._ripristina()
        self.lb_stato.setText("Errore.")
        QMessageBox.critical(self, "Errore", msg)

    # Se l'utente modifica a mano un riquadro, le posizioni salvate non valgono piu':
    # l'evidenziazione di quel riquadro viene disattivata.
    def _testo_o_modificato(self):
        if not self._prog and self.ev_o.attivo:
            self.ev_o.invalida()
            self.lb_stato.setText("Testo modificato: evidenziazione disattivata per questo riquadro.")

    def _testo_h_modificato(self):
        if not self._prog and self.ev_h.attivo:
            self.ev_h.invalida()
            self.lb_stato.setText("Hiragana modificato: evidenziazione disattivata per questo riquadro.")

    def aggiorna_hiragana(self):
        """Riconverte in hiragana il testo del riquadro superiore."""
        if self.ev_o.attivo and self._hira_orig:
            # testo originale intatto: ripristina l'hiragana sincronizzato
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
        """Salva i dati per rivedere il karaoke senza trascrivere di nuovo."""
        r = self.ultimo_ris
        media = self.ed_file.text().strip()
        dati = dict(
            versione=1,
            file=os.path.basename(media),
            percorso_file=os.path.abspath(media),
            orig=r["orig"],
            hira=r["hira"],
            span_o=[list(x) for x in r["span_o"]],
            span_h=[list(x) for x in r["span_h"]],
            t0=r["t0"],
            t1=r["t1"],
            tok=r["tok"],
        )
        with open(percorso, "w", encoding="utf-8") as fh:
            json.dump(dati, fh, ensure_ascii=False)

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
        self._valida_karaoke(d)
        r = dict(
            orig=d["orig"], hira=d["hira"],
            span_o=[tuple(x) for x in d["span_o"]],
            span_h=[tuple(x) for x in d["span_h"]],
            t0=list(d["t0"]), t1=list(d["t1"]),
            tok=d["tok"], avviso="",
        )
        self.ultimo_ris = r
        self._imposta_testi(r["orig"], r["hira"], r["span_o"], r["span_h"],
                            r["t0"], r["t1"], r["tok"])
        self.aggiorna_pos(self.player.position())

    def _cerca_karaoke(self, media):
        """Se accanto al file c'e' un _karaoke.json lo carica al posto della trascrizione."""
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
            "Karaoke (*.json);;Tutti i file (*)")
        if f:
            self.carica_karaoke_json(f)

    def carica_karaoke_json(self, pj):
        try:
            with open(pj, encoding="utf-8") as fh:
                dati = json.load(fh)
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

    def copia(self):
        QApplication.clipboard().setText(self.out.toPlainText())
        self.lb_stato.setText("Trascrizione originale copiata negli appunti.")

    def copia_hiragana(self):
        QApplication.clipboard().setText(self.out_hira.toPlainText())
        self.lb_stato.setText("Hiragana copiato negli appunti.")

    def salva(self):
        base = os.path.splitext(self.ed_file.text().strip())[0] or "trascrizione"
        f, _ = QFileDialog.getSaveFileName(self, "Salva testo", base + ".txt", "Testo (*.txt)")

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
        """Salva trascrizione e hiragana nella stessa cartella dell'audio originale."""
        audio = self.ed_file.text().strip()

        if not audio or not os.path.isfile(audio):
            QMessageBox.warning(self, "File mancante", "Scegli prima un file audio/video valido.")
            return

        testi = [
            ("_trascrizione.txt", self.out.toPlainText()),
            ("_hiragana.txt", self.out_hira.toPlainText()),
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

    def libera(self):
        libera_modello()
        self.lb_stato.setText("Modello scaricato dalla memoria.")

    def closeEvent(self, e):
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
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STILE)

    w = Finestra()
    w.show()

    sys.exit(app.exec_())
