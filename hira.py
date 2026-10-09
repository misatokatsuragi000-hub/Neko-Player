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
import sys
import wave
import subprocess
import tempfile
from bisect import bisect_right
from math import ceil

import numpy as np

from PyQt5.QtCore import Qt, QThread, QTimer, QUrl, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QPainter, QPen, QPolygonF, QTextCursor,
)
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QLineEdit, QPushButton, QComboBox, QCheckBox, QDoubleSpinBox, QTextEdit,
    QFileDialog, QProgressBar, QMessageBox, QFrame, QSlider, QSplitter,
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
#finestra { background: %(bg)s; }
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
    pronto = pyqtSignal(str, str, object, float, float)  # file, wav, db, hop, durata
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
                raise ValueError("Audio vuoto o non leggibile.")

            rms = np.sqrt(np.mean(x[:m * win].reshape(m, win) ** 2, axis=1))
            db = np.clip(20.0 * np.log10(rms + 1e-9), -60.0, 0.0)

            self.pronto.emit(self.path, wav, db, 0.01, len(x) / float(SR))

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
                last_error = e
                self.stato.emit(f"Tentativo {device}/{compute_type} fallito: {e}")
                continue

        raise RuntimeError(
            f"Impossibile caricare faster-whisper {p['modello']}: {last_error}"
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
            sep = "
            if (p["timestamp"] or p["a_capo"]) else ("" if lingua in ("ja", "zh") else " ")

            po, ph = [], []
            pos_o = pos_h = 0
            span_o, span_h, tempi = [], [], []

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

            self.finito.emit(dict(
                orig="".join(po),
                hira="".join(ph),
                span_o=span_o,
                span_h=span_h,
                t0=[t[0] for t in tempi],
                t1=[t[1] for t in tempi],
                avviso=avviso,
            ))

        except FileNotFoundError:
            self.errore.emit("ffmpeg non trovato: installalo e aggiungilo al PATH.")

        except subprocess.CalledProcessError as e:
            self.errore.emit((e.stderr or b"").decode(errors="replace") or str(e))

        except Exception as e:
            msg = str(e)

            if "out of memory" in msg.lower():
                msg += (
                    "
VRAM insufficiente: chiudi altri programmi che usano la GPU, "
                    "scegli un modello più piccolo oppure usa CPU."
                )

            if "cuda" in msg.lower() or "cudnn" in msg.lower():
                msg += (
                    "
Errore CUDA/cuDNN con faster-whisper/CTranslate2. "
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
        self.setMinimumHeight(200)
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

            fine_detto = self.spans[idx][0] if in_corso else self.spans[idx][1]

            if fine_detto > 0:
                s = QTextEdit.ExtraSelection()
                c = QTextCursor(self.ed.document())
                c.setPosition(0)
                c.setPosition(fine_detto, QTextCursor.KeepAnchor)
                s.cursor = c
                s.format.setForeground(QBrush(QColor(C_SPOKEN)))
                sels.append(s)

            if in_corso:
                a, b = self.spans[idx]
                s = QTextEdit.ExtraSelection()
                c = QTextCursor(self.ed.document())
                c.setPosition(a)
                c.setPosition(b, QTextCursor.KeepAnchor)
                s.cursor = c
                s.format.setBackground(QBrush(QColor(C_SPOKEN)))
                s.format.setForeground(QBrush(QColor("#14151c")))
                sels.append(s)
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
        self.setWindowTitle("Trascrivi Audio (faster-whisper)")
        self.resize(1180, 920)
        self.setAcceptDrops(True)

        self.worker = None
        self.decoder = None
        self._vecchi = []              # thread in chiusura (evita distruzione prematura)
        self.wav_tmp = None
        self.file_corrente = ""
        self.t0 = []
        self.t1 = []
        self._prog = False             # True mentre il testo viene impostato da codice

        self.player = QMediaPlayer(None, QMediaPlayer.LowLatency)
        self.player.setVolume(80)

        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._tick)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        # --- intestazione ---
        h = QVBoxLayout()
        h.setSpacing(0)

        t = QLabel("Trascrivi Audio")
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
        self.ed_file.setPlaceholderText("Scegli o trascina qui un file audio/video...")
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
            "Salva trascrizione e hiragana come file .txt nella stessa cartella dell'audio originale"
        )
        self.b_salva_accanto.clicked.connect(self.salva_accanto)

        az.addWidget(self.b_avvia)
        az.addWidget(self.b_salva_accanto)
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

        root.addWidget(c1)

        # --- player ---
        c2, l2 = card("Player")

        self.forma = Forma()
        self.forma.seek.connect(self.vai_a)
        l2.addWidget(self.forma)

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
        ctrl.addWidget(QLabel("Velocita'"))
        ctrl.addWidget(self.cb_vel)
        ctrl.addSpacing(10)
        ctrl.addWidget(QLabel("🔊"))
        ctrl.addWidget(self.sl_vol)

        l2.addLayout(ctrl)
        root.addWidget(c2)

        self.player.stateChanged.connect(self._stato_player)
        self.player.positionChanged.connect(self.aggiorna_pos)

        # --- testi affiancati ---
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)

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

    # ------------------------------------------------------------------ file
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            self.imposta_file(urls[0].toLocalFile())

    def scegli_file(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Scegli un file", "",
            "Audio/Video (*.wav *.mp3 *.m4a *.flac *.ogg *.opus *.aac *.mp4 *.mkv *.webm *.mov);;Tutti i file (*)"
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

        self.forma.pulisci("Decodifico l'audio...")
        self.lb_tempo.setText("00:00.0 / 00:00.0")

        if self.decoder is not None:
            self._vecchi.append(self.decoder)
            self._vecchi = [t for t in self._vecchi if t.isRunning()]

        self.decoder = Decoder(path)
        self.decoder.pronto.connect(self._audio_pronto)
        self.decoder.errore.connect(self._audio_errore)
        self.decoder.start()

    def _audio_pronto(self, path, wav, db, hop, durata):
        if path != self.file_corrente:       # nel frattempo e' stato scelto un altro file
            try:
                os.remove(wav)
            except OSError:
                pass
            return

        self.wav_tmp = wav
        self.durata = durata
        self.forma.imposta(db, hop, durata)

        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(wav)))
        self.player.setPlaybackRate(float(self.cb_vel.currentText().rstrip("x")))

        self._aggiorna_selezione()
        self.aggiorna_pos(0)

    def _audio_errore(self, path, msg):
        if path == self.file_corrente:
            self.forma.pulisci("Impossibile decodificare l'audio")
            self.lb_stato.setText("Errore di decodifica audio.")
            QMessageBox.warning(self, "Audio", msg)

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
        self.lb_tempo.setText("{} / {}".format(fmt_tempo(t), fmt_tempo(getattr(self, "durata", 0.0))))

        if self.t0:
            idx = bisect_right(self.t0, t) - 1
            in_corso = idx >= 0 and t <= self.t1[idx]
            self.ev_o.aggiorna(idx, in_corso)
            self.ev_h.aggiorna(idx, in_corso)

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
        self._imposta_testi("", "", [], [], [], [])

        self.worker = Worker(params)
        self.worker.stato.connect(self.lb_stato.setText)
        self.worker.finito.connect(self.fine_ok)
        self.worker.errore.connect(self.fine_errore)
        self.worker.start()

    def _ripristina(self):
        self.barra.setRange(0, 1)
        self.b_avvia.setEnabled(True)

    def _imposta_testi(self, orig, hira, span_o, span_h, t0, t1):
        self._prog = True
        self.out.setPlainText(orig)
        self.out_hira.setPlainText(hira)
        self._prog = False

        self.t0, self.t1 = t0, t1

        self.ev_o.imposta(span_o)
        self.ev_h.imposta(span_h)

        self._sp_o_orig, self._sp_h_orig, self._hira_orig = span_o, span_h, hira

    def fine_ok(self, r):
        self._ripristina()
        self._imposta_testi(r["orig"], r["hira"], r["span_o"], r["span_h"], r["t0"], r["t1"])

        self.lb_stato.setText(
            "Fatto{}. Premi ▶ per vedere le parole colorarsi mentre vengono pronunciate "
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
            self.lb_stato.setText("Salvato: " + f)

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

        if not testi:
            QMessageBox.information(self, "Niente da salvare", "Non ci sono ancora testi da salvare.")
            return

        base = os.path.splitext(audio)[0]
        esistenti = [os.path.basename(base + s) for s, _ in testi if os.path.exists(base + s)]

        if esistenti:
            r = QMessageBox.question(
                self, "Sovrascrivere?",
                "Esistono gia' questi file:
" + "
".join(esistenti) + "
Vuoi sovrascriverli?"
            )
            if r != QMessageBox.Yes:
                return

        try:
            for suff, t in testi:
                with open(base + suff, "w", encoding="utf-8") as fh:
                    fh.write(t)
        except OSError as e:
            QMessageBox.critical(self, "Errore di salvataggio", str(e))
            return

        self.lb_stato.setText("Salvato in {}: {}".format(
            os.path.dirname(audio),
            ", ".join(os.path.basename(base + s) for s, _ in testi)
        ))

    def libera(self):
        libera_modello()
        self.lb_stato.setText("Modello scaricato dalla memoria.")

    def closeEvent(self, e):
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
