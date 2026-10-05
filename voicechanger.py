#!/usr/bin/env python3
"""
Spostamento selettivo delle frequenze (pitch per banda) con curva libera.

Dipendenze (Linux Mint / Ubuntu):
  sudo apt install python3-tk python3-numpy python3-matplotlib python3-soundfile libportaudio2
  pip install sounddevice      (meglio in un venv con --system-site-packages)

Uso: python3 equalizzatore.py
  - Click sinistro sulla linea arancione  -> aggiunge un'ancora (trascinala su/giu')
  - Click sinistro su un'ancora           -> la sposti in verticale
  - Click destro su un'ancora             -> la elimina
  - Altezza ancora = spostamento in semitoni delle frequenze in quel punto
  - Rotella: zoom sulle frequenze; Maiusc+rotella: zoom sullo spostamento (semitoni)
  - Trascina sul vuoto: sposta la vista
  - Pulsanti Vista/Mostra: spettrogramma (attivo di default) o spettro medio
  - Barra in basso: click/trascina per scegliere il punto di riproduzione
"""
import json
import os
import shutil
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import soundfile as sf
import sounddevice as sd
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.patheffects as pe
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

FMIN, FMAX = 20.0, 20000.0   # range udibile
SMAX = 12.0                  # spostamento massimo in semitoni (+/-)
PICK_PX = 12                 # tolleranza click in pixel


# ---------------------------------------------------------------- config e dialoghi file
CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".config", "equalizzatore", "config.json")
AUDIO_EXTS = ["wav", "flac", "ogg", "mp3", "aiff", "aif"]


def load_config():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(cfg):
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


def ask_file(save, title, initdir, initname=""):
    """Dialogo nativo GTK (zenity) se disponibile, altrimenti quello standard di Tk.
    Restituisce il percorso scelto oppure '' se annullato."""
    if not os.path.isdir(initdir):
        initdir = os.path.expanduser("~")
    if shutil.which("zenity"):
        start = os.path.join(initdir, initname if save else "")
        cmd = ["zenity", "--file-selection", "--title=" + title, "--filename=" + start]
        if save:
            cmd += ["--save", "--confirm-overwrite", "--file-filter=WAV | *.wav *.WAV"]
        else:
            pats = " ".join(f"*.{e} *.{e.upper()}" for e in AUDIO_EXTS)
            cmd += ["--file-filter=File audio | " + pats, "--file-filter=Tutti i file | *"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            return r.stdout.strip() if r.returncode == 0 else ""
        except Exception:
            pass
    if save:
        return filedialog.asksaveasfilename(title=title, initialdir=initdir, initialfile=initname,
                                            defaultextension=".wav", filetypes=[("WAV", "*.wav")])
    return filedialog.askopenfilename(
        title=title, initialdir=initdir,
        filetypes=[("Audio", " ".join("*." + e for e in AUDIO_EXTS)), ("Tutti", "*.*")])


# ---------------------------------------------------------------- DSP
def spectral_envelope(mag, N, sr, iters=4):
    """Inviluppo spettrale (formanti) per ogni riga di `mag` (T, K) col metodo del
    "true envelope": lisciatura cepstrale iterata che segue i picchi dei dati.
    Il lifter taglia sotto il periodo del pitch piu' acuto (~400 Hz)."""
    q = max(8, int(sr / 400.0))
    logm = np.log(mag + 1e-10)
    env = logm
    for _ in range(iters):
        c = np.fft.irfft(env, n=N, axis=1)
        c[:, q + 1:N - q] = 0.0
        c[:, q] *= 0.5
        c[:, N - q] *= 0.5
        sm = np.fft.rfft(c, axis=1).real
        env = np.maximum(logm, sm)
    return np.exp(sm)


def pitch_warp(x, sr, shift_fn, progress=None, N=2048, osamp=8, formant=None):
    """Phase vocoder con identity phase locking e rotazione di fase continua per bin.
    A 0 st ricostruisce l'originale (errore ~1e-16). Le regioni di picco che si
    sovrappongono sul bin di arrivo vengono sommate (np.add.at).

    formant=None  -> nessuna preservazione (i formanti seguono il pitch).
    formant=f     -> preserva i formanti: lo spettro viene "appiattito" dividendolo per
                     l'inviluppo, spostato, e poi rimoltiplicato per l'inviluppo originale
                     traslato di f semitoni (f=0: formanti invariati)."""
    L = len(x)
    hop = N // osamp
    K = N // 2 + 1
    k = np.arange(K)
    TWO_PI = 2 * np.pi
    win = 0.5 - 0.5 * np.cos(TWO_PI * np.arange(N) / N)
    expct = TWO_PI * hop / N
    xp = np.concatenate([np.zeros(N), x, np.zeros(N)])
    frames = sliding_window_view(xp, N)[::hop]
    T = len(frames)
    out = np.zeros((T - 1) * hop + N)
    freqs_hz = k * (sr / N)
    st = shift_fn(freqs_hz)
    ratio = 2.0 ** (st / 12.0)
    delta_phi = k * (ratio - 1.0) * expct
    phi_rot = np.zeros(K)
    if formant is not None:                       # indici per traslare l'inviluppo
        pos = np.clip(k / 2.0 ** (formant / 12.0), 0, K - 1)
        j0 = np.minimum(pos.astype(int), K - 2)
        wj = pos - j0
    CH = 128
    for t0 in range(0, T, CH):
        t1 = min(T, t0 + CH)
        spec = np.fft.rfft(frames[t0:t1] * win, axis=1)
        mag = np.abs(spec)
        mp = np.pad(mag, ((0, 0), (2, 2)))
        pkm = ((mag > 0) & (mag > mp[:, 1:-3]) & (mag > mp[:, :-4])
               & (mag >= mp[:, 3:-1]) & (mag >= mp[:, 4:]))
        if formant is not None:
            env = spectral_envelope(mag, N, sr)
            env_t = (1 - wj) * env[:, j0] + wj * env[:, j0 + 1]
            spec = spec / env
        Z = np.zeros((t1 - t0, K), dtype=complex)
        for i in range(t1 - t0):
            phi_rot = np.mod(phi_rot + delta_phi, TWO_PI)
            pk = np.flatnonzero(pkm[i])
            if pk.size == 0:
                Z[i] = spec[i]
                continue
            tp = np.rint(pk * ratio[pk]).astype(int)
            off = tp - pk
            own = np.searchsorted((pk[:-1] + pk[1:]) // 2, k)
            dst = k + off[own]
            ok = (dst >= 0) & (dst < K)
            src, dst_ok, o = k[ok], dst[ok], own[ok]
            rot = np.exp(1j * phi_rot[pk[o]])
            np.add.at(Z[i], dst_ok, spec[i, src] * rot)
        if formant is not None:
            Z *= env_t
        y = np.fft.irfft(Z, n=N, axis=1) * win
        for i in range(t1 - t0):
            s0 = (t0 + i) * hop
            out[s0:s0 + N] += y[i]
        if progress:
            progress(t1 / T)
    return out[N:N + L] / (osamp * 3.0 / 8.0)


def pitch_warp_v1(x, sr, shift_fn, progress=None, N=2048, osamp=8):
    """Phase vocoder con rapporto di pitch dipendente dalla frequenza e
    identity phase locking (Laroche & Dolson, 1999).

    shift_fn(f_hz) -> semitoni di spostamento per ogni frequenza.

    Per ogni frame si individuano i picchi spettrali; ogni picco ha una regione
    d'influenza (fino al punto medio col picco vicino). Tutta la regione viene
    traslata dello stesso numero intero di bin del suo picco, e le fasi dei bin
    attorno al picco restano agganciate a quella del picco (coerenza verticale).
    Solo il picco accumula fase nel tempo. Se piu' bin cadono sullo stesso bin
    di arrivo vince quello con ampiezza maggiore (nessuna somma di ampiezze)."""
    L = len(x)
    hop = N // osamp
    K = N // 2 + 1
    k = np.arange(K)
    TWO_PI = 2 * np.pi

    win = 0.5 - 0.5 * np.cos(TWO_PI * np.arange(N) / N)
    expct = TWO_PI * hop / N
    xp = np.concatenate([np.zeros(N), x, np.zeros(N)])
    frames = sliding_window_view(xp, N)[::hop]
    T = len(frames)
    out = np.zeros((T - 1) * hop + N)

    last_ph = np.zeros(K)       # fase di analisi del frame precedente
    psi = np.zeros(K)           # fase di sintesi per ogni bin di arrivo (frame precedente)
    CH = 128
    for t0 in range(0, T, CH):
        t1 = min(T, t0 + CH)
        spec = np.fft.rfft(frames[t0:t1] * win, axis=1)
        mag, ph = np.abs(spec), np.angle(spec)
        prev = np.vstack([last_ph[None, :], ph[:-1]])
        last_ph = ph[-1]
        d = ph - prev - k * expct
        d -= TWO_PI * np.round(d / TWO_PI)
        fb = k + d * osamp / TWO_PI                       # frequenza vera in bin
        ratio = 2.0 ** (shift_fn(np.maximum(fb, 0.0) * sr / N) / 12.0)
        newf = fb * ratio                                 # frequenza spostata in bin

        # massimi locali (+-2 bin); ">" a sinistra e ">=" a destra evita doppioni sui plateau
        mp = np.pad(mag, ((0, 0), (2, 2)))
        pkm = ((mag > 0) & (mag > mp[:, 1:-3]) & (mag > mp[:, :-4])
               & (mag >= mp[:, 3:-1]) & (mag >= mp[:, 4:]))

        Z = np.zeros((t1 - t0, K), dtype=complex)
        for i in range(t1 - t0):
            pk = np.flatnonzero(pkm[i])
            if pk.size == 0:
                continue
            tp = np.rint(newf[i, pk]).astype(int)         # bin di arrivo dei picchi
            off = tp - pk
            own = np.searchsorted((pk[:-1] + pk[1:]) // 2, k)   # picco proprietario di ogni bin
            # fase di sintesi dei picchi: dal frame precedente + avanzamento alla nuova frequenza
            th = np.mod(psi[np.clip(tp, 0, K - 1)] + newf[i, pk] * expct, TWO_PI)
            dst = k + off[own]
            ok = (dst >= 0) & (dst < K)
            src, dst, o = k[ok], dst[ok], own[ok]
            phs = th[o] + ph[i, src] - ph[i, pk[o]]       # identity phase locking
            order = np.argsort(mag[i, src], kind="stable")  # l'ultima scrittura (max) vince
            src, dst, phs = src[order], dst[order], phs[order]
            Z[i, dst] = mag[i, src] * np.exp(1j * phs)
            psi[dst] = np.mod(phs, TWO_PI)

        y = np.fft.irfft(Z, n=N, axis=1) * win
        for i in range(t1 - t0):
            s = (t0 + i) * hop
            out[s:s + N] += y[i]
        if progress:
            progress(t1 / T)
    return out[N:N + L] / (osamp * 3.0 / 8.0)


# ---------------------------------------------------------------- player
class Player:
    def __init__(self):
        self.buf = None
        self.pos = 0
        self.playing = False
        self.stream = None

    def open(self, sr, channels):
        self.close()
        self.stream = sd.OutputStream(samplerate=sr, channels=channels,
                                      dtype="float32", callback=self._cb)
        self.stream.start()

    def close(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.playing = False

    def _cb(self, outdata, frames, t, status):
        buf = self.buf
        if not self.playing or buf is None:
            outdata.fill(0)
            return
        p = self.pos
        chunk = buf[p:p + frames]
        n = len(chunk)
        outdata[:n] = chunk
        if n < frames:
            outdata[n:] = 0
            self.playing = False
            self.pos = len(buf)
        else:
            self.pos = p + n


# ---------------------------------------------------------------- app
BG, PANEL, PANEL2 = "#14161b", "#1d2027", "#2a2e38"
FG, MUTED, GRID = "#dfe3ea", "#8b93a1", "#2c313c"
ACCENT, BLUE, GREEN = "#ff9f43", "#4aa3ff", "#3ddc97"
YLIM = SMAX * 1.1            # semitoni visibili di default
SEEK_PAD = 8                # margine laterale barra di riproduzione
SEEK_STEPS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600]
LIVE_N = 8192               # finestra (campioni) per lo spettro in tempo reale


def hz_label(f):
    return f"{f / 1000:.4g}k" if f >= 1000 else f"{f:.4g}"


class App:
    def __init__(self, root):
        self.root = root
        root.title("Spostamento selettivo delle frequenze")
        root.geometry("1320x780")
        root.minsize(1000, 580)
        root.configure(bg=BG)

        self.data = None
        self.sr = 44100
        self.anchors = []         # [freq_hz, semitoni]
        self.drag = None
        self.pan = None
        self.player = Player()
        self.cache = None         # (chiave, array float32)
        self.worker = None
        self.result = None
        self.progress = 0.0
        self.duration = 1.0
        self.mode = "orig"        # cosa si sta ascoltando: "orig" | "mod"
        self.want_play = False
        self.extra = []           # callback da eseguire a render finito (es. salvataggio)
        self.auto_job = None
        self.anchor_texts = []
        self.orig_line = self.res_line = self.fill = None
        self.spectro_on = True      # True: spettro nel punto di riproduzione; False: spettro medio
        self.view_t0, self.view_t1 = 0.0, 1.0   # tratto di file visibile nella barra
        self.seek_drag = None
        self.seek_state = None
        self.prev_t = 0.0
        self.mono = None
        self.mono_mod = None        # (chiave, audio elaborato in mono)
        self.legend = None
        self.last_head = -1.0
        self.cfg = load_config()
        self.path = ""
        self.keep_formants = tk.BooleanVar(value=True)
        self.formant_st = tk.StringVar(value="0")

        self.setup_style()
        self.build_ui()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.tick()

    # ------------------------------------------------------------ interfaccia
    def setup_style(self):
        st = ttk.Style()
        st.theme_use("clam")
        base = ("DejaVu Sans", 10)
        st.configure(".", background=BG, foreground=FG, font=base, borderwidth=0)
        st.configure("TFrame", background=BG)
        st.configure("Panel.TFrame", background=PANEL)
        st.configure("TLabel", background=PANEL, foreground=FG)
        st.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=("DejaVu Sans", 9))
        st.configure("Time.TLabel", background=PANEL, foreground=FG, font=("DejaVu Sans Mono", 10))
        st.configure("TSeparator", background=GRID)
        for name, bg, fg, act in [("TButton", PANEL2, FG, "#383e4b"),
                                  ("Blue.TButton", BLUE, "#0b1420", "#7dbcff"),
                                  ("Accent.TButton", ACCENT, "#1a1205", "#ffb866"),
                                  ("Green.TButton", GREEN, "#06170f", "#6ee8b3")]:
            st.configure(name, background=bg, foreground=fg, padding=(11, 7),
                         focusthickness=0, focuscolor=bg, relief="flat",
                         font=("DejaVu Sans", 10, "bold" if name != "TButton" else "normal"))
            st.map(name, background=[("active", act), ("pressed", act)])
        st.configure("Small.TButton", padding=(9, 5))
        st.configure("TCheckbutton", background=PANEL, foreground=FG)
        st.map("TCheckbutton", background=[("active", PANEL)])
        st.configure("TSpinbox", fieldbackground=PANEL2, foreground=FG, arrowcolor=FG)
        st.configure("Horizontal.TScale", background=ACCENT, troughcolor=PANEL2,
                     bordercolor=PANEL, lightcolor=ACCENT, darkcolor=ACCENT)

    def build_ui(self):
        def btn(parent, text, cmd, style="TButton", side=tk.LEFT):
            def run(cmd=cmd):
                cmd()
                self.root.focus_set()      # la barra spaziatrice resta per play/pausa
            b = ttk.Button(parent, text=text, command=run, style=style, takefocus=False)
            b.pack(side=side, padx=3)
            return b

        def sep(parent, side=tk.LEFT):
            ttk.Separator(parent, orient="vertical").pack(side=side, fill=tk.Y, padx=10, pady=2)

        # barra superiore
        tb = ttk.Frame(self.root, style="Panel.TFrame", padding=(10, 8))
        tb.pack(side=tk.TOP, fill=tk.X)
        btn(tb, "Apri file…", self.open_file)
        sep(tb)
        self.btn_orig = btn(tb, "▶  Originale", lambda: self.play("orig"), "Blue.TButton")
        self.btn_mod = btn(tb, "▶  Modificato", lambda: self.play("mod"), "Accent.TButton")
        self.btn_pause = btn(tb, "❚❚  Pausa", self.toggle_pause)
        sep(tb)
        self.btn_view = btn(tb, "Vista: Spettrogramma", self.toggle_view)
        sep(tb)
        ttk.Checkbutton(tb, text="Preserva formanti", variable=self.keep_formants,
                        takefocus=False, command=self.formant_changed).pack(side=tk.LEFT, padx=3)
        ttk.Label(tb, text="Formanti (st)", style="Muted.TLabel").pack(side=tk.LEFT, padx=(8, 4))
        sp = ttk.Spinbox(tb, from_=-12, to=12, increment=0.5, width=5, takefocus=False,
                         textvariable=self.formant_st, command=self.formant_changed)
        sp.pack(side=tk.LEFT)
        sp.bind("<Return>", lambda e: (self.formant_changed(), self.root.focus_set()))
        sp.bind("<FocusOut>", lambda e: self.formant_changed())
        btn(tb, "Salva WAV…", self.save_wav, "Green.TButton", side=tk.RIGHT)
        sep(tb, tk.RIGHT)
        btn(tb, "Azzera curva", self.reset_curve, side=tk.RIGHT)
        sep(tb, tk.RIGHT)
        btn(tb, "Reset", self.reset_zoom, "Small.TButton", side=tk.RIGHT)
        btn(tb, "＋", lambda: self.zoom_btn(0.6), "Small.TButton", side=tk.RIGHT)
        btn(tb, "－", lambda: self.zoom_btn(1 / 0.6), "Small.TButton", side=tk.RIGHT)
        ttk.Label(tb, text="Zoom", style="Muted.TLabel").pack(side=tk.RIGHT, padx=(0, 4))

        # barra di stato (in fondo) e barra di riproduzione
        sb = ttk.Frame(self.root, style="Panel.TFrame", padding=(12, 5))
        sb.pack(side=tk.BOTTOM, fill=tk.X)
        self.status = tk.StringVar(value="Apri un file audio per iniziare.")
        ttk.Label(sb, textvariable=self.status).pack(side=tk.LEFT)
        ttk.Label(sb, style="Muted.TLabel",
                  text="Rotella: zoom · Maiusc+rotella: zoom semitoni · Trascina sul vuoto: sposta "
                       "· Spazio: play/pausa · ← →: ±0,5 s · Rotella sulla barra: zoom tempo · Click destro su ancora: elimina").pack(side=tk.RIGHT)

        pb = ttk.Frame(self.root, style="Panel.TFrame", padding=(12, 8))
        pb.pack(side=tk.BOTTOM, fill=tk.X, pady=(1, 0))
        self.mode_lbl = tk.Label(pb, text="● ORIGINALE", bg=PANEL, fg=BLUE,
                                 font=("DejaVu Sans", 9, "bold"), width=13, anchor="w")
        self.mode_lbl.pack(side=tk.LEFT)
        self.seekc = tk.Canvas(pb, height=50, bg=PANEL, highlightthickness=0, takefocus=0)
        self.seekc.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        c = self.seekc
        c.bind("<Button-1>", self.on_seek_press)
        c.bind("<B1-Motion>", self.on_seek_motion)
        c.bind("<ButtonRelease-1>", lambda e: setattr(self, "seek_drag", None))
        c.bind("<Double-Button-1>", self.on_seek_double)
        c.bind("<Button-2>", self.on_seek_pan_start)
        c.bind("<B2-Motion>", self.on_seek_pan)
        c.bind("<Button-4>", lambda e: self.on_seek_wheel(e, 1))
        c.bind("<Button-5>", lambda e: self.on_seek_wheel(e, -1))
        c.bind("<MouseWheel>", lambda e: self.on_seek_wheel(e, 1 if e.delta > 0 else -1))
        c.bind("<Configure>", lambda e: setattr(self, "seek_state", None))
        self.time_lbl = ttk.Label(pb, text="0:00 / 0:00", style="Time.TLabel", width=12)
        self.time_lbl.pack(side=tk.LEFT)

        # grafico
        self.fig = Figure(figsize=(10, 5), facecolor=BG, tight_layout=True)
        self.ax = self.fig.add_subplot(111, facecolor="#0d0f13")
        self.ax.set_xscale("log")
        self.ax.set_xlim(FMIN, FMAX)
        self.ax.set_xlabel("Frequenza (Hz)", color=MUTED)
        self.ax.set_ylabel("Livello spettro (dB)", color=MUTED)
        self.ax.tick_params(colors=MUTED)
        self.ax.grid(True, color=GRID, alpha=0.8, lw=0.6)
        for sp in self.ax.spines.values():
            sp.set_color(GRID)

        self.ax2 = self.ax.twinx()
        self.ax2.set_ylim(-YLIM, YLIM)
        self.ax2.set_ylabel("Spostamento frequenza (semitoni)", color=ACCENT)
        self.ax2.tick_params(colors=ACCENT)
        for sp in self.ax2.spines.values():
            sp.set_color(GRID)
        self.ax2.axhline(0, color=MUTED, lw=0.7, ls="--", alpha=0.7)
        self.fgrid = np.geomspace(FMIN, FMAX, 1500)
        self.curve_line, = self.ax2.plot(self.fgrid, np.zeros_like(self.fgrid),
                                         color=ACCENT, lw=2.2)
        self.anchor_pts, = self.ax2.plot([], [], "o", color=BG, mec=ACCENT, mew=2.2, ms=10)

        self.curve_line.set_path_effects([pe.withStroke(linewidth=4.5, foreground="#000000", alpha=0.45)])
        self.ax.set_title("", color=MUTED, fontsize=10)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        w = self.canvas.get_tk_widget()
        w.configure(bg=BG, highlightthickness=0)
        w.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        w.bind("<Button-4>", lambda e: self.on_wheel(e, 1))
        w.bind("<Button-5>", lambda e: self.on_wheel(e, -1))
        w.bind("<MouseWheel>", lambda e: self.on_wheel(e, 1 if e.delta > 0 else -1))
        self.root.bind("<space>", lambda e: (self.toggle_pause(), "break")[1])
        self.root.bind("<Left>", lambda e: self.seek_by(-0.5))
        self.root.bind("<Right>", lambda e: self.seek_by(0.5))
        self.update_ticks()

    def set_mode(self, mode):
        self.mode = mode
        if mode == "orig":
            self.mode_lbl.config(text="● ORIGINALE", fg=BLUE)
        else:
            self.mode_lbl.config(text="● MODIFICATO", fg=ACCENT)

    # ------------------------------------------------------------ curva
    def nodes(self):
        pts = [[FMIN, 0.0]] + sorted(self.anchors, key=lambda a: a[0]) + [[FMAX, 0.0]]
        return np.log10([p[0] for p in pts]), np.array([p[1] for p in pts])

    def shift_st(self, f):
        xs, ys = self.nodes()
        return np.interp(np.log10(np.maximum(f, 1e-3)), xs, ys)

    def formant_value(self):
        """Spostamento dei formanti in semitoni, oppure None se la preservazione e' spenta."""
        if not self.keep_formants.get():
            return None
        try:
            return float(np.clip(float(self.formant_st.get().replace(",", ".")), -12, 12))
        except ValueError:
            return 0.0

    def formant_changed(self):
        k = self.key()
        if getattr(self, "_last_fkey", None) != k[-1:]:
            self._last_fkey = k[-1:]
            self.curve_changed()

    def key(self):
        pts = tuple((round(a[0], 3), round(a[1], 3)) for a in sorted(self.anchors))
        fv = self.formant_value()
        return pts + (("formanti", None if fv is None else round(fv, 2)),)

    def curve_changed(self):
        """Dopo una modifica: se si sta ascoltando il modificato, aggiorna l'audio da solo."""
        if self.mode == "mod" and self.data is not None:
            if self.auto_job:
                self.root.after_cancel(self.auto_job)
            self.auto_job = self.root.after(350, self.request_render)

    # ------------------------------------------------------------ file
    def open_file(self):
        path = ask_file(False, "Apri file audio",
                        self.cfg.get("open_dir", os.path.expanduser("~")))
        if not path:
            return
        try:
            data, sr = sf.read(path, dtype="float64", always_2d=True)
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile aprire il file:\n{e}")
            return
        self.data, self.sr = data, sr
        self.path = path
        self.cfg["open_dir"] = os.path.dirname(path)
        save_config(self.cfg)
        self.anchors, self.cache, self.mono_mod = [], None, None
        self.duration = len(data) / sr
        self.view_t0, self.view_t1 = 0.0, self.duration
        self.prev_t, self.seek_state = 0.0, None
        try:
            self.player.open(sr, data.shape[1])
        except Exception as e:
            messagebox.showerror("Errore audio", str(e))
            return
        self.orig32 = data.astype(np.float32)
        self.player.buf, self.player.pos = self.orig32, 0
        self.set_mode("orig")
        self.compute_spectrum()
        self.setup_live()
        self.draw_spectrum()
        self.apply_view()
        self.reset_zoom()
        self.status.set(f"{os.path.basename(path)} — {sr} Hz, {data.shape[1]} canali, "
                        f"{self.duration:.1f} s")

    def compute_spectrum(self):
        mono = self.data.mean(axis=1)
        n = 32768                          # alta risoluzione per lo zoom
        if len(mono) < n:
            mono = np.pad(mono, (0, n - len(mono)))
        count = int(min(200, max(1, (len(mono) - n) // (n // 2) + 1)))
        starts = np.linspace(0, len(mono) - n, count).astype(int)
        win = np.hanning(n)
        acc = np.zeros(n // 2 + 1)
        for s in starts:
            acc += np.abs(np.fft.rfft(mono[s:s + n] * win)) ** 2
        mag = np.sqrt(acc / len(starts)) / (win.sum() / 2)
        self.f_full = np.fft.rfftfreq(n, 1 / self.sr)
        self.db_full = 20 * np.log10(mag + 1e-10)
        keep = (self.f_full >= FMIN) & (self.f_full <= min(FMAX, self.sr / 2))
        self.freqs, self.spec_db = self.f_full[keep], self.db_full[keep]
        self.res_vals = self.spec_db.copy()

    def setup_live(self):
        """Prepara l'analisi in tempo reale (spettro nel punto di riproduzione)."""
        n = LIVE_N
        self.mono = self.data.mean(axis=1)
        self.f_live = np.fft.rfftfreq(n, 1 / self.sr)
        self.live_keep = (self.f_live >= FMIN) & (self.f_live <= min(FMAX, self.sr / 2))
        self.live_x = self.f_live[self.live_keep]
        self.live_win = np.hanning(n)
        m = self.mono if len(self.mono) >= n else np.pad(self.mono, (0, n - len(self.mono)))
        cnt = int(min(150, max(1, (len(m) - n) // n + 1)))
        starts = np.linspace(0, len(m) - n, cnt).astype(int)
        mag = np.abs(np.fft.rfft(sliding_window_view(m, n)[starts] * self.live_win, axis=1))
        mag /= self.live_win.sum() / 2
        self.live_top = float(np.max(20 * np.log10(mag.max(axis=1) + 1e-10))) + 6
        self.last_head = -1.0

    def frame_db(self, mono, pos):
        """Spettro in dB della finestra centrata sul campione pos."""
        n = LIVE_N
        a = int(pos) - n // 2
        seg = np.zeros(n)
        lo, hi = max(a, 0), min(a + n, len(mono))
        if hi > lo:
            seg[lo - a:hi - a] = mono[lo:hi]
        mag = np.abs(np.fft.rfft(seg * self.live_win)) / (self.live_win.sum() / 2)
        db = 20 * np.log10(mag + 1e-10)
        return np.convolve(db, np.ones(3) / 3, mode="same")

    def curve_data(self):
        """Restituisce (x, originale, modificato, descrizione) per la vista attiva."""
        live = self.spectro_on
        if live:
            pos = self.player.pos
            fF, dbF = self.f_live, self.frame_db(self.mono, pos)
            x, orig = self.live_x, dbF[self.live_keep]
        else:
            fF, dbF = self.f_full, self.db_full
            x, orig = self.freqs, self.spec_db
        if not self.anchors:
            return x, orig, orig, "nessuna modifica"
        if live and self.mono_mod and self.mono_mod[0] == self.key():
            res = self.frame_db(self.mono_mod[1], pos)[self.live_keep]
            return x, orig, res, "audio elaborato"
        g = fF * 2.0 ** (self.shift_st(fF) / 12.0)
        o = np.argsort(g)
        res = np.interp(x, g[o], dbF[o], left=-120, right=-120)
        return x, orig, res, "anteprima"

    def set_curves(self):
        x, orig, res, src = self.curve_data()
        self.res_vals = res
        self.orig_line.set_data(x, orig)
        self.res_line.set_data(x, res)
        if self.fill is not None:
            self.fill.remove()
        self.fill = self.ax.fill_between(x, orig, -300, color=BLUE, alpha=0.22, lw=0)
        if self.spectro_on:
            t = self.player.pos / self.sr
            self.ax.set_ylim(self.live_top - 100, self.live_top)
            self.ax.set_title(f"Spettro a {t:.1f} s  ·  modificato: {src}", color=MUTED, fontsize=10)
        else:
            self.fit_db()
            self.ax.set_title(f"Spettro medio dell'intero file  ·  modificato: {src}",
                              color=MUTED, fontsize=10)

    def apply_view(self):
        self.btn_view.config(text="Vista: Spettrogramma" if self.spectro_on else "Vista: Spettro medio")
        self.last_head = -1.0
        self.refresh()

    def toggle_view(self):
        self.spectro_on = not self.spectro_on
        self.apply_view()

    def draw_spectrum(self):
        for a in (self.orig_line, self.res_line, self.fill):
            if a is not None:
                a.remove()
        self.fill = self.ax.fill_between(self.freqs, self.spec_db, -300,
                                         color=BLUE, alpha=0.22, lw=0)
        self.orig_line, = self.ax.plot(self.freqs, self.spec_db, color=BLUE, lw=0.9,
                                       label="Originale")
        self.res_line, = self.ax.plot(self.freqs, self.spec_db, color=GREEN, lw=1.1,
                                      label="Modificato")
        leg = self.ax.legend(loc="upper left", facecolor=PANEL, edgecolor=GRID)
        for t in leg.get_texts():
            t.set_color(FG)
        self.legend = leg

    # ------------------------------------------------------------ disegno
    def update_ticks(self):
        lo, hi = self.ax.get_xlim()
        if hi / lo > 4:
            ticks = [m * 10 ** e for e in range(1, 5) for m in (1, 2, 5) if lo <= m * 10 ** e <= hi]
        else:
            raw = (hi - lo) / 8
            mag = 10 ** np.floor(np.log10(raw))
            step = min(c * mag for c in (1, 2, 5, 10) if c * mag >= raw)
            ticks = list(np.arange(np.ceil(lo / step) * step, hi + 1e-9, step))
        self.ax.set_xticks(ticks)
        self.ax.set_xticklabels([hz_label(t) for t in ticks])
        self.ax.minorticks_off()

    def fit_db(self):
        if self.res_line is None:
            return
        lo, hi = self.ax.get_xlim()
        m = (self.freqs >= lo) & (self.freqs <= hi)
        if m.any():
            top = max(self.spec_db[m].max(), self.res_vals[m].max()) + 10
            self.ax.set_ylim(top - 90, top)

    def refresh(self):
        self.curve_line.set_ydata(self.shift_st(self.fgrid))
        a = sorted(self.anchors, key=lambda a: a[0])
        self.anchor_pts.set_data([p[0] for p in a], [p[1] for p in a])
        for t in self.anchor_texts:
            t.remove()
        self.anchor_texts = [
            self.ax2.annotate(f"{hz_label(p[0])} Hz   {p[1]:+.1f} st", (p[0], p[1]),
                              xytext=(0, 14), textcoords="offset points", ha="center",
                              fontsize=8, color=FG,
                              bbox=dict(boxstyle="round,pad=0.3", fc=PANEL, ec=ACCENT, lw=0.8))
            for p in a]
        if self.res_line is not None and self.mono is not None:
            self.set_curves()
        self.update_ticks()
        self.canvas.draw_idle()

    def reset_curve(self):
        self.anchors = []
        self.refresh()
        self.curve_changed()

    # ------------------------------------------------------------ zoom
    def zoom_x(self, fc, z):
        lo, hi = self.ax.get_xlim()
        fc = float(np.clip(fc, lo, hi))
        a = fc * (lo / fc) ** z
        b = fc * (hi / fc) ** z
        if b / a >= FMAX / FMIN:
            a, b = FMIN, FMAX
        if b / a < 1.02:
            return
        if a < FMIN:
            a, b = FMIN, b * FMIN / a
        if b > FMAX:
            a, b = a * FMAX / b, FMAX
        self.ax.set_xlim(max(a, FMIN), min(b, FMAX))

    def zoom_y(self, v, z):
        lo, hi = self.ax2.get_ylim()
        v = float(np.clip(v, lo, hi))
        lo, hi = v + (lo - v) * z, v + (hi - v) * z
        if hi - lo >= 2 * YLIM:
            lo, hi = -YLIM, YLIM
        if hi - lo < 0.5:
            return
        if lo < -YLIM:
            lo, hi = -YLIM, hi - lo - YLIM
        if hi > YLIM:
            lo, hi = lo - (hi - YLIM), YLIM
        self.ax2.set_ylim(lo, hi)

    def zoom_btn(self, z):
        if self.data is None:
            return
        lo, hi = self.ax.get_xlim()
        self.zoom_x(np.sqrt(lo * hi), z)
        self.refresh()

    def reset_zoom(self):
        self.ax.set_xlim(FMIN, FMAX)
        self.ax2.set_ylim(-YLIM, YLIM)
        self.refresh()

    def on_wheel(self, e, d):
        if self.data is None:
            return
        f, v = self.to_data_px(e.x, self.fig.bbox.height - e.y)
        z = 0.8 if d > 0 else 1.25
        if e.state & 0x1:          # Maiusc
            self.zoom_y(v, z)
        else:
            self.zoom_x(f, z)
        self.refresh()

    # ------------------------------------------------------------ elaborazione
    def render(self, xs, ys, fv):
        if not np.any(ys) and not fv:
            return self.orig32
        fn = lambda f: np.interp(np.log10(np.maximum(f, 1e-3)), xs, ys)
        ch = self.data.shape[1]
        out = np.empty_like(self.data)
        for c in range(ch):
            def prog(p, c=c):
                self.progress = (c + p) / ch
            out[:, c] = pitch_warp(self.data[:, c], self.sr, fn, prog, formant=fv)
        peak = np.abs(out).max()
        if peak > 0.999:
            out *= 0.999 / peak
        return out.astype(np.float32)

    def request_render(self, play=False, then=None):
        """Assicura che il file modificato per la curva attuale sia pronto (thread in background)."""
        if self.data is None:
            return
        self.auto_job = None
        if play:
            self.want_play = True
        if then:
            self.extra.append(then)
        key = self.key()
        if self.cache and self.cache[0] == key:
            self.finish_render(self.cache[1])
            return
        if self.worker and self.worker.is_alive():
            return                      # il tick rilancerà se la curva è cambiata
        self.progress, self.result = 0.0, None
        xs, ys = self.nodes()
        fv = self.formant_value()      # letto nel thread principale (Tk)

        def run():
            try:
                self.result = (key, self.render(xs, ys, fv), None)
            except Exception as e:
                self.result = (key, None, e)
        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()

    def finish_render(self, arr):
        p = self.player
        if self.mode == "mod":
            p.buf = arr                 # cambio al volo, mantenendo la posizione
            if self.want_play:
                if p.pos >= len(arr) - 1:
                    p.pos = 0
                p.playing = True
            self.status.set("Audio modificato aggiornato.")
        self.want_play = False
        cbs, self.extra = self.extra, []
        for cb in cbs:
            cb(arr)

    # ------------------------------------------------------------ riproduzione
    def play(self, which):
        if self.data is None:
            return
        self.set_mode(which)
        if which == "orig":
            p = self.player
            p.buf = self.orig32
            if p.pos >= len(self.orig32) - 1:
                p.pos = 0
            p.playing = True
            self.want_play = False
        else:
            self.request_render(play=True)

    def pause(self):
        self.player.playing = False
        self.want_play = False

    def toggle_pause(self):
        """Pausa se sta suonando, altrimenti riprende dal punto attuale."""
        if self.data is None:
            return
        if self.player.playing or self.want_play:
            self.pause()
        else:
            self.play(self.mode)

    def seek_by(self, dt):
        """Sposta la posizione di riproduzione di dt secondi (frecce sinistra/destra)."""
        if self.data is None:
            return "break"
        pos = self.player.pos + int(dt * self.sr)
        self.player.pos = int(min(max(pos, 0), len(self.data) - 1))
        t = self.player.pos / self.sr
        if not self.view_t0 <= t <= self.view_t1:
            span = self.view_t1 - self.view_t0
            self.set_view(t - 0.1 * span, span)
        return "break"

    # ---- barra di riproduzione con zoom sul tempo
    def set_view(self, t0, span):
        d = self.duration
        span = min(max(span, min(d, 1.0)), d)
        t0 = min(max(t0, 0.0), d - span)
        self.view_t0, self.view_t1 = t0, t0 + span

    def seek_time(self, x):
        w = max(1, self.seekc.winfo_width() - 2 * SEEK_PAD)
        return self.view_t0 + (x - SEEK_PAD) / w * (self.view_t1 - self.view_t0)

    def seek_apply(self, e):
        w = max(1, self.seekc.winfo_width() - 2 * SEEK_PAD)
        if self.seek_drag == "strip":      # striscia superiore: sposta la finestra visibile
            t = (e.x - SEEK_PAD) / w * self.duration
            span = self.view_t1 - self.view_t0
            self.set_view(t - span / 2, span)
        else:                              # barra: sposta la riproduzione
            t = min(max(self.seek_time(e.x), 0.0), self.duration)
            self.player.pos = int(min(t * self.sr, len(self.data) - 1))

    def on_seek_press(self, e):
        if self.data is None:
            return
        self.seek_drag = "strip" if e.y < 13 else "bar"
        self.seek_apply(e)

    def on_seek_motion(self, e):
        if self.data is not None and self.seek_drag:
            self.seek_apply(e)

    def on_seek_double(self, e):
        if self.data is not None and e.y < 13:      # doppio click sulla striscia: reset zoom
            self.set_view(0.0, self.duration)

    def on_seek_pan_start(self, e):
        self.pan_seek = (e.x, self.view_t0)

    def on_seek_pan(self, e):
        if self.data is None or not hasattr(self, "pan_seek"):
            return
        x0, t0 = self.pan_seek
        span = self.view_t1 - self.view_t0
        w = max(1, self.seekc.winfo_width() - 2 * SEEK_PAD)
        self.set_view(t0 - (e.x - x0) / w * span, span)

    def on_seek_wheel(self, e, d):
        if self.data is None:
            return
        span = self.view_t1 - self.view_t0
        if e.state & 0x1:                                   # Maiusc: scorri
            self.set_view(self.view_t0 - d * 0.2 * span, span)
            return
        tc = min(max(self.seek_time(e.x), self.view_t0), self.view_t1)
        frac = (tc - self.view_t0) / span
        nspan = span * (0.75 if d > 0 else 1 / 0.75)
        nspan = min(max(nspan, min(self.duration, 1.0)), self.duration)
        self.set_view(tc - frac * nspan, nspan)

    def draw_seek(self):
        c = self.seekc
        c.delete("all")
        W = c.winfo_width()
        if W < 40:
            return
        d = self.duration if self.data is not None else 1.0
        t0, t1 = self.view_t0, self.view_t1
        x0, x1 = SEEK_PAD, W - SEEK_PAD
        X = lambda t: x0 + (t - t0) / (t1 - t0) * (x1 - x0)
        pt = self.player.pos / self.sr if self.data is not None else 0.0
        # striscia panoramica (tutto il file, con la finestra visibile)
        a, b = x0 + t0 / d * (x1 - x0), x0 + t1 / d * (x1 - x0)
        c.create_rectangle(x0, 3, x1, 9, fill=PANEL2, outline="")
        c.create_rectangle(a, 3, max(b, a + 3), 9, fill="#5b6478", outline="")
        xs = x0 + pt / d * (x1 - x0)
        c.create_line(xs, 2, xs, 10, fill=ACCENT, width=2)
        # barra principale
        c.create_rectangle(x0, 16, x1, 30, fill=PANEL2, outline="")
        xp = X(pt)
        if xp > x0:
            c.create_rectangle(x0, 16, min(xp, x1), 30, fill="#7a4f1c", outline="")
        span = t1 - t0
        step = next((st for st in SEEK_STEPS if st >= span / 8), SEEK_STEPS[-1])
        k = int(np.ceil(t0 / step - 1e-9))
        while k * step <= t1 + 1e-9:
            t = k * step
            x = X(t)
            c.create_line(x, 30, x, 35, fill=MUTED)
            m, sec = int(t // 60), t - 60 * int(t // 60)
            lab = f"{m}:{sec:04.1f}" if step < 1 else f"{m}:{int(round(sec)):02d}"
            c.create_text(x, 43, text=lab, fill=MUTED, font=("DejaVu Sans", 8))
            k += 1
        if t0 <= pt <= t1:
            c.create_line(xp, 14, xp, 32, fill=ACCENT, width=2)
            c.create_oval(xp - 5, 18, xp + 5, 28, fill=ACCENT, outline=BG)

    def tick(self):
        if self.worker is not None:
            if self.worker.is_alive():
                self.status.set(f"Elaborazione… {self.progress * 100:.0f}%")
            elif self.result is not None:
                key, arr, err = self.result
                self.result = None
                if err:
                    self.extra, self.want_play = [], False
                    messagebox.showerror("Errore", str(err))
                else:
                    self.cache = (key, arr)
                    self.mono_mod = (key, arr.mean(axis=1))
                    self.refresh()
                    if key == self.key():
                        self.finish_render(arr)
                    else:
                        self.request_render()   # la curva è cambiata nel frattempo
        txt = "❚❚  Pausa" if (self.player.playing or self.want_play) else "▶  Play"
        if self.btn_pause.cget("text") != txt:
            self.btn_pause.config(text=txt)
        if self.data is not None:
            t = self.player.pos / self.sr
            span = self.view_t1 - self.view_t0
            if self.player.playing and self.prev_t <= self.view_t1 < t:
                self.set_view(t - 0.1 * span, span)      # la vista segue la riproduzione
            self.prev_t = t
            state = (self.player.pos, self.view_t0, self.view_t1, self.seekc.winfo_width())
            if state != self.seek_state:
                self.seek_state = state
                self.draw_seek()
            if self.spectro_on and self.mono is not None and t != self.last_head:
                self.last_head = t
                self.set_curves()
                self.canvas.draw_idle()
            fmt = lambda s: f"{int(s) // 60}:{int(s) % 60:02d}"
            self.time_lbl.config(text=f"{fmt(t)} / {fmt(self.duration)}")
        self.root.after(60, self.tick)

    def save_wav(self):
        if self.data is None:
            return
        base = os.path.splitext(os.path.basename(self.path))[0] or "audio"
        path = ask_file(True, "Salva WAV",
                        self.cfg.get("save_dir", self.cfg.get("open_dir", os.path.expanduser("~"))),
                        base + "_modificato.wav")
        if not path:
            return
        if not path.lower().endswith(".wav"):
            path += ".wav"
        self.cfg["save_dir"] = os.path.dirname(path)
        save_config(self.cfg)

        def write(arr):
            sf.write(path, arr, self.sr, subtype="PCM_24")
            self.status.set(f"Salvato: {path}")
        self.request_render(then=write)

    # ------------------------------------------------------------ mouse
    def to_data_px(self, x, y):
        return self.ax2.transData.inverted().transform((x, y))

    def on_press(self, e):
        if self.data is None or e.inaxes is None:
            return
        tr = self.ax2.transData
        click = np.array([e.x, e.y])
        best, bd = None, PICK_PX
        for a in self.anchors:
            d = np.hypot(*(tr.transform((a[0], a[1])) - click))
            if d < bd:
                best, bd = a, d
        if e.button == 3:
            if best is not None:
                self.anchors.remove(best)
                self.refresh()
                self.curve_changed()
            return
        if e.button != 1:
            return
        if best is None:
            f, _ = self.to_data_px(e.x, e.y)
            f = float(np.clip(f, FMIN * 1.05, FMAX / 1.05))
            cy = float(self.shift_st(np.array([f]))[0])
            if abs(tr.transform((f, cy))[1] - e.y) > PICK_PX:
                # click sul vuoto: sposta la vista
                self.pan = dict(x=e.x, y=e.y, xl=self.ax.get_xlim(), yl=self.ax2.get_ylim())
                return
            best = [f, cy]
            self.anchors.append(best)
        self.drag = best
        self.refresh()

    def on_motion(self, e):
        if e.x is None:
            return
        if self.pan:
            p = self.pan
            lo, hi = p["xl"]
            la, lb = np.log10(lo), np.log10(hi)
            sh = (e.x - p["x"]) / self.ax.bbox.width * (lb - la)
            na, nb = la - sh, lb - sh
            if na < np.log10(FMIN):
                na, nb = np.log10(FMIN), nb + np.log10(FMIN) - na
            if nb > np.log10(FMAX):
                na, nb = na - (nb - np.log10(FMAX)), np.log10(FMAX)
            self.ax.set_xlim(10 ** na, 10 ** nb)
            ylo, yhi = p["yl"]
            if yhi - ylo < 2 * YLIM - 0.01:
                dy = (e.y - p["y"]) / self.ax.bbox.height * (yhi - ylo)
                dy = max(min(dy, ylo + YLIM), yhi - YLIM)
                self.ax2.set_ylim(ylo - dy, yhi - dy)
            self.refresh()
            return
        if self.drag is None:
            return
        _, s = self.to_data_px(e.x, e.y)
        self.drag[1] = float(np.clip(s, -SMAX, SMAX))
        f = self.drag[0]
        self.status.set(f"Ancora a {f:.0f} Hz: {self.drag[1]:+.2f} semitoni "
                        f"(→ {f * 2 ** (self.drag[1] / 12):.0f} Hz)")
        self.refresh()

    def on_release(self, e):
        changed = self.drag is not None
        self.drag = None
        self.pan = None
        if changed:
            self.curve_changed()

    def on_close(self):
        self.player.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
