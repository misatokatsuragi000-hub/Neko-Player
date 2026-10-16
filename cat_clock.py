#!/usr/bin/env python3
"""
Orologio gatto da desktop
-------------------------
- Mostra solo l'immagine (sfondo trasparente) con le lancette che girano sul quadrante
- Sempre in primo piano, senza bordi e senza icona nella taskbar
- Parte minimizzato nella tray (click sull'icona = mostra/nascondi)
- Si sposta trascinando, si ridimensiona con la rotella del mouse
  oppure con Ctrl + trascinamento; ricorda posizione, dimensione e opzioni
- Un clic sulla testa fa camminare il gatto per qualche secondo in una direzione
  casuale, dondolando buffamente (un altro clic lo ferma); se sbatte contro il
  bordo dello schermo barcolla, cade a terra, si scuote e dopo un po' si rialza
- Si puo' afferrare la coda col mouse e tirarla: si raddrizza e poi si allunga fino al 150%;
  la coda resta tirata verso il mouse finche' si tiene premuto; al 150% il gatto parte a
  camminare in quella direzione (seguendo il mouse finche' la coda e' tesa); al rilascio la coda
  torna indietro con un rimbalzo elastico
- Quando l'orologio e' nascosto (clic sull'icona della tray o Esc) compare un gatto in miniatura
  (MINI_WIDTH px, senza quadrante, lancette e numeri) in piedi sulla barra delle applicazioni,
  all'altezza dell'icona nella tray; cammina avanti e indietro lungo la barra, ondeggiando, e
  quando arriva ai bordi dello schermo barcolla e cade; quando si rialza torna indietro.
  Con la matrioska attiva, accanto al gatto in miniatura cammina anche un'altra miniatura, piu'
  piccola, che lo segue restando indietro e ogni tanto scatta per raggiungerlo (e si scansiona
  quando lui cade).
  Un clic su di lui riapre l'orologio
- Opzione "Matrioska" (menu): compare un secondo gatto, piu' piccolo, che funziona come il primo
  (orologio, coda, tremolio, camminata, cadute, coda tirata, sveglia) e lo segue sempre a breve
  distanza (se lo trascini col mouse resta dove lo lasci e quella diventa la sua posizione standard
  rispetto al grande); usa il tipo di orologio opposto (se il grande e' a lancette il piccolo e' digitale e viceversa): quando il grande si sposta, il piccolo gli cammina dietro
- Dal menu "Orologio a numeri": al posto delle lancette compare l'ora in cifre
  (riempimento col colore delle lancette, bordo col colore dei numeri del quadrante)
- Sveglia (menu "Sveglia attiva" / "Imposta ora sveglia..."): all'ora impostata, ogni giorno, il gatto
  oscilla sul posto come quando cammina ma con un angolo maggiore e miagola (miao.mp3 in
  loop). Un clic sul quadrante permette di scrivere l'ora della sveglia direttamente sul quadrante
  (cifre con lo stesso font, dimensione e colori dell'orologio a numeri; Invio conferma, Esc annulla). Un clic sul gatto (o sul menu) la ferma; si ferma da sola dopo ALARM_DURATION secondi

Requisiti:  pip install PyQt5
File necessari nella stessa cartella:  cat_clock_body.png, cat_clock_dial.png, cat_clock_numbers.png, cat_clock_tail.png, cat_clock_details.png, miao.mp3
"""
import sys
import math
import random
import time
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer, QPointF, QRectF, QSettings, QPoint, QRect, QSize, QTime, QUrl
from PyQt5.QtGui import (QPixmap, QPainter, QColor, QPen, QIcon, QBrush, QLinearGradient,
                         QFont, QFontMetricsF, QPainterPath, QTransform)
from PyQt5.QtWidgets import (QApplication, QWidget, QSystemTrayIcon, QMenu,
                             QAction, QMessageBox, QInputDialog, QColorDialog)
try:                                   # audio della sveglia (se manca, si usa il beep di sistema)
    from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
    HAVE_MULTIMEDIA = True
except ImportError:
    HAVE_MULTIMEDIA = False

# ----------------------------------------------------------------- CONFIG --
BODY_FILE = Path(__file__).with_name("cat_clock_body.png")        # sagoma del corpo (si colora)
DIAL_FILE = Path(__file__).with_name("cat_clock_dial.png")        # disco del quadrante (si colora)
TAIL_FILE = Path(__file__).with_name("cat_clock_tail.png")        # coda (oscilla come un pendolo)
NUMBERS_FILE = Path(__file__).with_name("cat_clock_numbers.png")  # numeri delle ore (si colorano)
DETAILS_FILE = Path(__file__).with_name("cat_clock_details.png")  # contorno, occhi, strisce, coda
DEFAULT_BODY_COLOR = "#DECDBE"  # colore originale del legno, ora a tinta unita
DEFAULT_COLOR2 = "#E58FB5"      # secondo colore della sfumatura (default)
DEFAULT_DIAL_COLOR = "#17171B"  # nero del quadrante originale
DEFAULT_NUMBERS_COLOR = "#CDBAAB"  # colore originale dei numeri
DEFAULT_TAIL_COLOR = "#141212"     # nero della coda
START_HIDDEN = False           # False = l'orologio appare subito all'avvio; True = parte nascosto nella tray
DEFAULT_WIDTH = 300            # larghezza iniziale (px)
MIN_WIDTH, MAX_WIDTH = 80, 2000
WHEEL_STEP = 0.06              # 6% per scatto di rotella

# Geometria del quadrante nell'immagine ritagliata (461 x 720 px)
CENTER = (228, 374)            # centro del quadrante
HOUR_LEN, MIN_LEN, SEC_LEN = 82, 114, 122
HAND_W = 9                     # spessore lancette ore/minuti
HUB_R = 10.5

# Orologio a numeri (opzione del menu)
DIGITAL_FONTS = ["Yu Gothic UI Semibold" , "DejaVu Sans" ,  "Sans Serif"]   # il primo disponibile
DIGITAL_24H = True             # False = formato 12 ore
DIGITAL_W = 200                # larghezza di "HH:MM" sul quadrante (px dell'immagine)
DIGITAL_SEC_SCALE = 0.5        # grandezza dei secondi rispetto a ore:minuti
DIGITAL_DY = 0                 # sposta il blocco verso l'alto (-) o il basso (+), px dell'immagine
DIGITAL_OUTLINE = 5.0          # spessore del bordo (in unita' di un carattere alto 100 px)
DIGITAL_HIDE_RING = True       # nasconde la corona dei numeri delle ore disegnata nell'immagine

# Coda a pendolo
TAIL_PIVOT = (226, 500)        # punto di sospensione (nascosto dietro il corpo)
TAIL_ANGLE = 15.0              # ampiezza massima in gradi (da -15 a +15)
TAIL_PERIOD = 2.0              # secondi per un'oscillazione completa (andata e ritorno)
TAIL_FPS = 60
# Coda e movimento del gatto (barcollamento, caduta): la coda e' un pendolo agganciato al corpo con
# una molla smorzata; subisce l'inerzia del corpo che si inclina. Quando il gatto cade la coda
# si alza (e freme) in segno di panico; quando tenta di rialzarsi scatta verso il basso e si
# incurva sul pavimento per fare da puntello
TAIL_FALL_WN = 14.0            # rad/s: rigidita' del giunto coda-corpo
TAIL_FALL_ZETA = 0.55          # smorzamento del giunto
TAIL_REL_MAX = 100.0           # massimo angolo della coda rispetto al corpo (gradi)
TAIL_FLOOR_FROM = 35.0         # da questa inclinazione del gatto la coda comincia a cercare il pavimento
TAIL_FLOOR_TO = 75.0           # e da qui lo cerca del tutto
TAIL_UP_ANGLE = 120.0          # coda alzata: angolo nel mondo dalla verticale in giu', dal lato opposto alla caduta
TAIL_FLUTTER = 60.0             # fremito della coda alzata (gradi) mentre il gatto e' a terra
TAIL_FLUTTER_HZ = 2.5
TAIL_PUSH_CURL = 1.35          # curvatura della coda quando fa da puntello (1 = come la PNG)
TAIL_STRIP = 8.0               # lunghezza delle strisce in cui viene tagliata la coda (px dell'immagine)
TAIL_HALF_W = 40.0             # mezza larghezza di ogni striscia (deve superare lo spessore della coda)
TAIL_PAD = 2.5                 # leggera sovrapposizione tra strisce, per non lasciare righe
TAIL_TIP = 45.0                # lunghezza extra dell'ultima striscia, per includere la punta arrotondata

# Coda tirata (si afferra la coda col mouse e la si tira)
TAIL_PULL_MAX = 2.5            # allungamento massimo (150%): raggiunto quello, il gatto parte in quella direzione
TAIL_GRAB_SLACK = 16.0          # tolleranza attorno alla coda per afferrarla (px dell'immagine)
TAIL_GRAB_MIN_PX = 8.0         # ...ma almeno questi px sullo schermo, cosi' si afferra anche col gatto piccolo
TAIL_PULL_DECAY = 8.0          # ritorno elastico al rilascio: smorzamento (1/s)
TAIL_PULL_OMEGA = 14.0         # ...e pulsazione (rad/s): la coda rimbalza un po' prima di fermarsi
TAIL_PULL_REL_TIME = 0.7       # durata del ritorno (secondi)

# Gatto in miniatura sulla barra delle applicazioni (quando l'orologio e' nascosto)
MINI_WIDTH = 30                # larghezza del gatto in miniatura (px)
TASKBAR_HEIGHT = 35            # altezza della barra delle applicazioni (px): il gatto sta in piedi sul suo bordo superiore
MINI_FALLBACK_RIGHT = 120      # se il sistema non da' la posizione dell'icona: distanza dal bordo destro dello schermo (px)
MINI_FOLLOW_GAIN = 6.0         # miniatura piccola (matrioska): piu' e' indietro, piu' corre (velocita' extra per px di distanza, 1/s)
MINI_LAG_DIST = (120.0, 240.0)   # miniatura piccola: a che distanza (px) resta indietro prima di scattare
MINI_LAG_TIME = (5.0, 12.0)     # ...e per quanto tempo (s) resta indietro
MINI_DRIFT = 0.75              # dopo lo scatto la piccola rallenta e si stacca piano: va a questa frazione della velocita' della principale
MINI_SPRINT_SPEED = 2       # velocita' dello scatto (multiplo di MINI_WALK_SPEED)
MINI_SPRINT_STRIDE = 0.55      # durata del passo durante lo scatto (frazione di quello normale: passi piu' rapidi)
MINI_REFRESH_MS = 1500         # ogni quanto la miniatura si riposiziona e torna in primo piano
MINI_WALK_SPEED = 45.0         # velocita' media della camminata lungo la barra (px al secondo); 0 = resta fermo sull'icona
# (dondolio, saltelli, barcollamento e caduta usano gli stessi parametri del gatto grande: WOBBLE_*, WALK_HOP, STAGGER_*, FALL_*, LIE_TIME, WIGGLE_*, UP_TIME)

# Matrioska: secondo gatto, piu' piccolo, che funziona come il primo e lo segue
MATRIOSKA_SCALE = 0.6          # larghezza del gatto piccolo rispetto a quello grande
MATRIOSKA_MIN_WIDTH = 30       # larghezza minima del gatto piccolo (px)
MATRIOSKA_GAP = 0.06           # distanza tra i due gatti (frazione della larghezza del grande)
MATRIOSKA_START = 0.45         # comincia a camminare dietro al grande quando e' piu' lontano di cosi' (larghezze del piccolo)
MATRIOSKA_STOP = 0.10          # ...e si ferma quando e' piu' vicino di cosi'
MATRIOSKA_CALM = 0.5           # secondi di calma (vicino al grande) prima di fermarsi del tutto
MATRIOSKA_GAIN = 1.5           # piu' e' indietro, piu' corre: velocita' extra per px di distanza (1/s)
MATRIOSKA_MAX_SPEED = 3.0      # velocita' massima nell'inseguimento (larghezze del piccolo al secondo)

# Tremolio "sveglia" a ogni cambio di minuto
SHAKE_DURATION = 1.6           # secondi
SHAKE_FREQ = 22.0              # vibrazioni al secondo
SHAKE_AMP = 0.022              # ampiezza = 2.2% della larghezza (minimo 3 px)

# Camminata buffa (clic sulla testa)
HEAD_Y_MAX = 230               # il "clic sulla testa" vale sopra questa riga dell'immagine (px, 461x720)
CLICK_TOL = 5                  # px di tolleranza: oltre, e' un trascinamento e non un clic
WALK_DURATION = (3.0, 6.0)     # durata casuale della camminata (secondi, min e max)
WALK_SPEED = 0.30              # velocita' media in larghezze del gatto al secondo
WALK_PAD = 0.15                # margine temporaneo attorno al gatto, serve all'inclinazione
WOBBLE_ANGLE = 8.0             # inclinazione massima a destra/sinistra (gradi)
WOBBLE_PERIOD = 0.85           # secondi per un'oscillazione completa (un passo per lato)
WALK_RAMP = 0.4                # secondi di avvio e di arresto morbidi
WALK_HOP = 34.0                # altezza del saltello a ogni passo (px dell'immagine, 461x720); 0 = niente saltello
FOOT_L = (113, 535)            # punto d'appoggio del piede sinistro (nell'immagine)
FOOT_R = (340, 540)            # punto d'appoggio del piede destro
FALL_TIME = 0.55               # secondi per cadere a terra
FALL_REBOUND = 7.0             # piccolo rimbalzo all'impatto (gradi)
BOUNCE_TIME = 0.35
LIE_TIME = (1.5, 3.0)          # quanto resta a terra (secondi, min e max)
UP_TIME = 1.6                  # secondi per rialzarsi (con un primo tentativo fallito)
STAGGER_TIME = 1.1             # secondi di barcollamento dopo l'urto, prima di cadere
STAGGER_SWINGS = 2.75          # quante oscillazioni fa mentre barcolla
STAGGER_AMP = 24.0             # inclinazione massima mentre barcolla (gradi)
WIGGLE_TIME = 1.0              # secondi in cui si scuote da terra prima di rialzarsi
WIGGLE_FREQ = 7.0              # scosse al secondo
WIGGLE_ROCK = 7.0              # quanto dondola mentre si scuote (gradi)
WIGGLE_JITTER = 0.012          # tremolio della finestra (frazione della larghezza, minimo 2 px)

# Sveglia
ALARM_FILE = Path(__file__).with_name("miao.mp3")   # suono riprodotto in loop mentre suona
DEFAULT_ALARM_TIME = "07:00"   # ora di default (HH:MM)
ALARM_DURATION = 60.0          # secondi di suoneria, poi si ferma da sola (un clic la ferma prima)
ALARM_WOBBLE_ANGLE = 18.0      # inclinazione massima a destra/sinistra durante la sveglia (gradi; camminata: WOBBLE_ANGLE)
ALARM_WOBBLE_PERIOD = WOBBLE_PERIOD   # secondi per un'oscillazione completa
ALARM_MEOW_GAP = 0.5           # pausa tra un miagolio e il successivo (secondi)
ALARM_VOLUME = 100             # 0-100
EDIT_BLINK_HZ = 2.0            # lampeggi al secondo del testo mentre si imposta l'ora della sveglia

HAND_COLOR = QColor(239, 199, 167)   # pesca, come le lancette originali
HUB_RING = QColor(217, 164, 72)
SEC_COLOR = QColor(255, 150, 110)
# ---------------------------------------------------------------------------


class CatClock(QWidget):
    def __init__(self, settings: QSettings, leader=None):
        super().__init__()
        self.settings = settings
        self.leader = leader                      # None = gatto principale; altrimenti e' il piccolo (matrioska) che lo segue
        self.child = None                         # il gatto piccolo che segue questo (se la matrioska e' attiva)
        self._min_w = MATRIOSKA_MIN_WIDTH if leader is not None else MIN_WIDTH
        self._follow = False                      # la camminata in corso e' un inseguimento del gatto grande
        self._follow_calm = 0.0
        self._sync_key = None
        self.window_menu = None
        # il gatto piccolo riusa le immagini e le tabelle gia' calcolate dal grande
        self.body = leader.body if leader is not None else QPixmap(str(BODY_FILE))
        self.dial = leader.dial if leader is not None else QPixmap(str(DIAL_FILE))
        self.tail = leader.tail if leader is not None else QPixmap(str(TAIL_FILE))
        self.numbers = leader.numbers if leader is not None else QPixmap(str(NUMBERS_FILE))
        self.details = leader.details if leader is not None else QPixmap(str(DETAILS_FILE))
        for f, pm in ((BODY_FILE, self.body), (DIAL_FILE, self.dial),
                      (NUMBERS_FILE, self.numbers), (TAIL_FILE, self.tail),
                      (DETAILS_FILE, self.details)):
            if pm.isNull():
                raise FileNotFoundError(f"Immagine non trovata: {f}")
        self.img_w, self.img_h = self.body.width(), self.body.height()
        self._body_img = leader._body_img if leader is not None else self.body.toImage()   # per capire se il clic cade sul gatto
        self._dial_img = leader._dial_img if leader is not None else self.dial.toImage()   # ...o sul quadrante (clic = imposta la sveglia)
        self._dial_click = None
        self._alarm_edit = None                   # durante l'impostazione: le 4 cifre di HH:MM
        self._edit_pos = 0                        # cifra in modifica (0-3)
        self._edit_t0 = 0.0                       # inizio del ciclo di lampeggio (riparte a ogni tasto)
        self.alarm_changed_cb = None              # impostato da build_tray (aggiorna il menu)
        self.tray = None                          # impostato da build_tray: serve a trovare l'icona nella tray
        self.mini = None                          # gatto in miniatura (creato a fine __init__)
        self.mini_on = settings.value("mini_on", True, type=bool)
        self.matrioska_on = settings.value("matrioska_on", False, type=bool)
        # posizione standard del piccolo rispetto al grande, in larghezze del grande (None = automatica)
        self.matrioska_off = None
        if settings.contains("matrioska_dx") and settings.contains("matrioska_dy"):
            self.matrioska_off = (settings.value("matrioska_dx", 0.0, type=float),
                                  settings.value("matrioska_dy", 0.0, type=float))
        self._moved = False                       # il gatto e' stato trascinato col mouse (per il piccolo: nuova posizione standard)
        self._lift_tab = leader._lift_tab if leader is not None else self._build_lift_table()
        self._tail_geo = leader._tail_geo if leader is not None else self._build_tail_geometry()
        if self._tail_geo:                    # distanza perno-punta: serve a farla appoggiare a terra
            nx_, ny_ = self._tail_geo[0][-1]
            self._tail_reach = math.hypot(nx_ - TAIL_PIVOT[0], ny_ - TAIL_PIVOT[1])
        else:
            self._tail_reach = 215.0
        self._tail_t = time.monotonic()
        self._tail_init = False
        self._tail_omega = self._tail_omega_v = 0.0   # angolo della coda nel mondo (gradi, gradi/s)
        self._tail_w_prev = self._tail_wv = 0.0
        self._tail_push = 0.0
        self._tail_pose = None        # (angolo, curvatura) della coda all'ultimo disegno: serve a capire se la si afferra
        self._tp_drag = False         # coda afferrata e tirata col mouse
        self._tp_j = 0                # nodo della coda afferrato
        self._tp_off = (0.0, 0.0)     # scarto clic-nodo (px dell'immagine), cosi' all'inizio non c'e' scatto
        self._tp_sign = 1.0           # verso della curvatura della coda al momento della presa
        self._tp_pose = (0.0, 0.0, 1.0)   # posa tirata: (angolo, curvatura, allungamento)
        self._tp_rel = (0.0, 0.0, 1.0)    # posa al momento del rilascio
        self._tp_gpos = QPoint()      # ultima posizione globale del mouse mentre si tira
        self._tp_g = 1.0              # quanto e' tesa la coda (0-1): il gatto cammina solo se e' tesa
        self._tp_rel_t0 = None        # inizio del ritorno elastico (None = nessun ritorno in corso)
        self.body_color = QColor(settings.value("body_color", DEFAULT_BODY_COLOR, type=str))
        if not self.body_color.isValid():
            self.body_color = QColor(DEFAULT_BODY_COLOR)
        self.body_color2 = QColor(settings.value("body_color2", DEFAULT_COLOR2, type=str))
        if not self.body_color2.isValid():
            self.body_color2 = QColor(DEFAULT_COLOR2)
        self.gradient_on = settings.value("gradient_on", False, type=bool)
        self.dial_color = QColor(settings.value("dial_color", DEFAULT_DIAL_COLOR, type=str))
        if not self.dial_color.isValid():
            self.dial_color = QColor(DEFAULT_DIAL_COLOR)
        self.numbers_color = QColor(settings.value("numbers_color", DEFAULT_NUMBERS_COLOR, type=str))
        if not self.numbers_color.isValid():
            self.numbers_color = QColor(DEFAULT_NUMBERS_COLOR)
        self.tail_on = settings.value("tail_on", True, type=bool)
        self.shake_on = settings.value("shake_on", True, type=bool)
        self.alarm_on = settings.value("alarm_on", False, type=bool)
        self.alarm_time = QTime.fromString(settings.value("alarm_time", DEFAULT_ALARM_TIME, type=str), "HH:mm")
        if not self.alarm_time.isValid():
            self.alarm_time = QTime.fromString(DEFAULT_ALARM_TIME, "HH:mm")
        self._shake_start = None
        self._shake_base = None
        self._walk_start = None       # camminata: None = ferma
        self._walk_last = 0.0
        self._walk_dur = 0.0
        self._walk_dir = (0.0, 0.0)
        self._walk_pos = QPointF()    # angolo alto-sinistra del gatto sullo schermo
        self._wobble = 0.0            # inclinazione attuale (gradi, + = verso destra)
        self._hop = 0.0               # saltello verticale attuale (px dell'immagine, + = in alto)
        self._pad = (0, 0, 0, 0)      # margini extra (sx, alto, dx, basso) durante la camminata
        self._walk_phase = "walk"      # walk -> fall -> lie -> up
        self._phase_t0 = 0.0
        self._fall_sign = 1           # +1 cade verso destra, -1 verso sinistra
        self._fall_a0 = 0.0
        self._lie_dur = 0.0
        self._last_speed = 0.0
        self._stag_speed = 0.0
        self._cat_w = self._cat_h = 0   # dimensione 'logica' del gatto: mai ricavata dalla finestra
        self._jit = 0                   # tremolio orizzontale (solo disegno) mentre si scuote
        self._off = (0, 0)              # dove sta il gatto dentro la finestra (px)
        self._rest_geo = None           # geometria finale da ribadire a fine camminata
        self._head_click = None
        self._alarm_active = False    # la sveglia sta suonando
        self._alarm_fired = None      # (anno, mese, giorno, ora, minuto) dell'ultima suoneria
        self._player = None           # QMediaPlayer, creato alla prima suoneria
        self._player_failed = False
        self._meow_token = 0          # invalida i miagolii programmati di una suoneria gia' finita
        n = datetime.now()
        self._last_min = (n.hour, n.minute)
        self.tail_color = QColor(settings.value("tail_color", DEFAULT_TAIL_COLOR, type=str))
        if not self.tail_color.isValid():
            self.tail_color = QColor(DEFAULT_TAIL_COLOR)
        self._tail_cache = None
        self.hand_color = QColor(settings.value("hand_color", HAND_COLOR.name(), type=str))
        if not self.hand_color.isValid():
            self.hand_color = QColor(HAND_COLOR)
        self.ratio = self.img_h / self.img_w

        self.show_seconds = settings.value("show_seconds", True, type=bool)
        self.digital_on = settings.value("digital_on", False, type=bool)
        self._dig = None              # metriche del font dell'orologio a numeri
        self.locked = settings.value("locked", False, type=bool)
        if leader is not None:                    # il piccolo non si blocca e non ha una sveglia sua
            self.locked = False
            self.alarm_on = False

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.StrongFocus)       # serve alla tastiera per scrivere l'ora della sveglia
        self.setWindowTitle("Cat Clock")

        if leader is not None:
            self.set_width(round(leader._cat_w * MATRIOSKA_SCALE))
        else:
            w = settings.value("width", DEFAULT_WIDTH, type=int)
            self.set_width(w)
            self._restore_position()

        self._cache = None            # (size, dpr, pixmap scalato)
        self._drag_off = None
        self._resize_start = None

        # tick ogni secondo (sempre allineato al secondo di sistema)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._apply_timer()
        # ogni tanto riporta l'orologio davanti agli altri
        self.raise_timer = QTimer(self)
        self.raise_timer.timeout.connect(self._keep_on_top)
        self.raise_timer.start(3000)
        # salvataggio ritardato
        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.timeout.connect(self.save_settings)
        # controllo della sveglia, una volta al secondo (attivo anche se l'orologio e' nascosto)
        self.alarm_timer = QTimer(self)
        self.alarm_timer.timeout.connect(self._alarm_check)
        self.alarm_timer.start(1000)
        self.mini = MiniCat(self) if leader is None else None

    def set_mini(self, on):
        self.mini_on = on
        if self.mini is not None:
            if on and not self.isVisible():
                self.mini.show_mini()
            elif not on:
                self.mini.hide_mini()
        self.save_settings()

    def _apply_timer(self):
        if self._walk_start is not None or self._pull_busy():
            self.timer.start(16)
            return
        # con la coda che oscilla servono ~30 fps, altrimenti basta 4 aggiornamenti al secondo
        self.timer.start(int(1000 / TAIL_FPS) if self.tail_on else 250)

    def set_tail(self, on):
        self.tail_on = on
        self._apply_timer()
        self.save_settings()
        self.update()

    def _tick(self):
        if self.leader is not None:
            self.sync_from_leader()
        n = datetime.now()
        key = (n.hour, n.minute)
        if key != self._last_min:
            self._last_min = key
            if (self.shake_on and self.isVisible() and self._shake_start is None
                    and self._walk_start is None and not self._pull_busy()):
                self._start_shake()
        if self._shake_start is not None:
            self._shake_step()
        if self._walk_start is not None:
            self._walk_step()
        self._pull_step()
        if self.leader is not None:
            self._follow_check()
        self.update()

    def _start_shake(self):
        self._rest_geo = None
        self._shake_start = time.monotonic()
        self._shake_base = self.pos()
        self.timer.start(16)               # ~60 fps durante il tremolio

    def _shake_step(self):
        t = time.monotonic() - self._shake_start
        if t >= SHAKE_DURATION:
            self._end_shake()
            return
        # a piena intensita' per i primi 3/4 del tempo, poi si smorza
        fade = SHAKE_DURATION * 0.25
        env = 1.0 if t < SHAKE_DURATION - fade else (SHAKE_DURATION - t) / fade
        amp = max(3.0, self._cat_w * SHAKE_AMP)
        dx = amp * env * math.sin(2 * math.pi * SHAKE_FREQ * t)
        dy = amp * 0.4 * env * math.sin(2 * math.pi * SHAKE_FREQ * 1.7 * t)
        self.move(self._shake_base + QPoint(round(dx), round(dy)))

    def _end_shake(self):
        if self._shake_start is None:
            return
        self.move(self._shake_base)        # torna esattamente dov'era
        self._shake_start = None
        self._shake_base = None
        self._apply_timer()

    def set_shake(self, on):
        self.shake_on = on
        if not on:
            self._end_shake()
        self.save_settings()

    # ----------------------------------------------------------- camminata --
    def _cat_size(self):
        """Dimensione del gatto, senza i margini temporanei della camminata."""
        return QSize(self._cat_w, self._cat_h)

    def _cat_pos(self):
        """Angolo alto-sinistra del gatto sullo schermo (senza tremolio ne' margini temporanei)."""
        if self._shake_start is not None:
            return QPoint(self._shake_base)
        if self._walk_start is not None or self._pull_busy():
            return QPoint(round(self._walk_pos.x()), round(self._walk_pos.y()))
        return self.pos()

    def _apply_pad(self, pad):
        """Imposta i margini extra attorno al gatto e riposiziona la finestra."""
        self._pad = pad
        self._place()

    def _on_head(self, pos):
        k = self._cat_w / self.img_w
        ix, iy = int(pos.x() / k), int(pos.y() / k)
        if not (0 <= ix < self.img_w and 0 <= iy < self.img_h):
            return False
        return iy < HEAD_Y_MAX and self._body_img.pixelColor(ix, iy).alpha() > 40

    def _on_dial(self, pos):
        k = self._cat_w / self.img_w
        ix, iy = int(pos.x() / k), int(pos.y() / k)
        if not (0 <= ix < self._dial_img.width() and 0 <= iy < self._dial_img.height()):
            return False
        return self._dial_img.pixelColor(ix, iy).alpha() > 40

    def _build_lift_table(self):
        """Di quanto va alzato il gatto mentre si sdraia, perche' il fianco non affondi
        sotto la linea dei piedi (tabella ogni 3 gradi, per i due piedi-perno)."""
        img, pts = self._body_img, []
        rows = list(range(0, self.img_h, 6)) + list(range(480, 545, 2))
        for y in rows:
            if not 0 <= y < self.img_h:
                continue
            xl = next((x for x in range(self.img_w) if img.pixelColor(x, y).alpha() > 40), None)
            if xl is None:
                continue
            xr = next(x for x in range(self.img_w - 1, -1, -1) if img.pixelColor(x, y).alpha() > 40)
            pts += [(xl, y), (xr, y)]
        tabs = {}
        for sign, (px, py) in ((1, FOOT_R), (-1, FOOT_L)):
            vals = []
            for a in range(0, 91, 3):
                th = math.radians(a * sign)
                sn, cs = math.sin(th), math.cos(th)
                vals.append(max((x - px) * sn + (y - py) * cs for x, y in pts))
            tabs[sign] = [max(0.0, v - vals[0]) for v in vals]
        return tabs

    def _lift(self, angle):
        tab = self._lift_tab[1 if angle >= 0 else -1]
        f = min(abs(angle), 90.0) / 3.0
        i = min(int(f), len(tab) - 2)
        return tab[i] + (tab[i + 1] - tab[i]) * (f - i)

    def _place(self):
        """Posiziona la finestra (gatto + margini) e, se la geometria vera e' diversa (ritardi o
        'rimbalzi' del window manager), la ribadisce ad ogni fotogramma.
        La finestra resta sempre interamente dentro lo schermo (i window manager la rimetterebbero
        dentro da soli, e a fine animazione il gatto salterebbe): se i margini sporgerebbero oltre
        il bordo, la finestra viene spostata verso l'interno e il gatto si disegna al suo interno
        nel punto giusto (self._off). Cio' che sporge dallo schermo comunque non si vedrebbe."""
        l, t, r, b = self._pad
        cx, cy = round(self._walk_pos.x()), round(self._walk_pos.y())
        ww, wh = self._cat_w + l + r, self._cat_h + t + b
        wx, wy = cx - l, cy - t
        if (l, t, r, b) != (0, 0, 0, 0):
            c = QPoint(cx + self._cat_w // 2, cy + self._cat_h // 2)
            geo = (QApplication.screenAt(c) or QApplication.primaryScreen()).availableGeometry()
            wx = max(geo.left(), min(wx, geo.right() + 1 - ww))
            wy = max(geo.top(), min(wy, geo.bottom() + 1 - wh))
        self._off = (cx - wx, cy - wy)
        rect = QRect(wx, wy, ww, wh)
        if self.geometry() != rect:
            self.setGeometry(rect)

    def _settle(self):
        """A fine camminata ribadisce la geometria finale, se nel frattempo nessuno l'ha cambiata."""
        g = self._rest_geo
        if g is None or self._walk_start is not None or self._drag_off is not None:
            return
        if self.isVisible() and self.geometry() != g:
            self.setGeometry(g)

    def _start_walk(self, direction=None, from_pull=False, follow=False):
        """Fa camminare il gatto: direzione casuale, oppure quella data (vettore unitario).
        from_pull: parte dalla coda tirata, la finestra e' gia' allargata per la coda allungata."""
        if self.locked or self._walk_start is not None or not self.isVisible():
            return
        self._end_shake()
        self._follow = follow
        self._follow_calm = 0.0
        cat = self._cat_size()
        if from_pull:
            pos = QPoint(round(self._walk_pos.x()), round(self._walk_pos.y()))
            keep = self._pad
        else:
            pos, keep = self.pos(), (0, 0, 0, 0)
        self._rest_geo = None
        if direction is None:
            ang = random.uniform(0.0, 2.0 * math.pi)
            direction = (math.cos(ang), math.sin(ang))
        self._walk_dir = direction
        self._walk_dur = random.uniform(*WALK_DURATION)
        self._walk_pos = QPointF(pos)
        # la finestra si allarga un po' attorno al gatto, cosi' l'inclinazione non viene tagliata
        m = max(8, round(cat.width() * WALK_PAD))
        self._apply_pad(tuple(max(m, q) for q in keep))
        self._wobble = 0.0
        self._hop = 0.0
        self._walk_phase = "walk"
        self._walk_start = self._walk_last = time.monotonic()
        self.timer.start(16)

    def _walk_step(self):
        now = time.monotonic()
        if self._walk_phase != "walk":
            self._hop *= 0.75            # se l'urto lo coglie in aria, atterra in pochi fotogrammi
            if self._hop < 0.05:
                self._hop = 0.0
            self._fallen_step(now)
            return
        t = now - self._walk_start
        dt = min(now - self._walk_last, 0.1)
        self._walk_last = now
        if self._tp_drag:                    # coda tirata: la camminata dura finche' si tiene premuto
            self._walk_dur = t + WALK_RAMP + 1.0
        fol = self._follow_state(t, dt) if (self._follow and not self._alarm_active) else None
        if t >= self._walk_dur:
            self._end_walk()
            return
        # avvio e arresto morbidi
        env = min(1.0, t / WALK_RAMP, (self._walk_dur - t) / WALK_RAMP)
        env = env * env * (3 - 2 * env)
        if self._tp_drag and not self._alarm_active:
            # segue il mouse finche' la coda e' tesa: piu' e' tesa, piu' in fretta; a coda
            # solo dritta (nessun allungamento) si ferma, restando attaccato al mouse
            tension = max(0.0, min(1.0, (self._tp_pose[2] - 1.0) / (TAIL_PULL_MAX - 1.0)))
            self._tp_g += (tension - self._tp_g) * min(1.0, dt * 8.0)
            env *= self._tp_g
            kk = self._cat_w / self.img_w
            vx = self._tp_gpos.x() - (self._walk_pos.x() + TAIL_PIVOT[0] * kk)
            vy = self._tp_gpos.y() - (self._walk_pos.y() + TAIL_PIVOT[1] * kk)
            dd = math.hypot(vx, vy)
            if dd > 1.0:
                self._walk_dir = (vx / dd, vy / dd)
        if fol is not None:                  # matrioska: cammina verso il gatto grande, piano se e' vicino
            env *= fol[2]
            self._walk_dir = fol[1]
        period = ALARM_WOBBLE_PERIOD if self._alarm_active else WOBBLE_PERIOD
        angle = ALARM_WOBBLE_ANGLE if self._alarm_active else WOBBLE_ANGLE
        phase = 2 * math.pi * t / period
        self._wobble = angle * env * math.sin(phase)
        # saltello: parte quando il gatto, alla massima inclinazione su un piede (phase = pi/2 + n*pi),
        # comincia a ruotare verso l'altro; e' al massimo quando passa dalla verticale (wobble = 0)
        # e torna a zero quando la rotazione si conclude sul nuovo piede d'appoggio. Nella fase
        # iniziale (phase < pi/2, entrambi i piedi a terra) non c'e' saltello.
        if phase < math.pi / 2:
            self._hop = 0.0
        else:
            self._hop = WALK_HOP * env * abs(math.cos(phase))
        if self._alarm_active:               # sveglia: oscilla sul posto, senza avanzare
            self._last_speed = 0.0
            self._place()
            return
        # avanza un po' di piu' quando il peso passa da un piede all'altro
        cat = self._cat_size()
        base = WALK_SPEED * cat.width()
        if fol is not None:                  # piu' e' indietro, piu' corre
            base = min(base + MATRIOSKA_GAIN * fol[0], MATRIOSKA_MAX_SPEED * cat.width())
        speed = base * env * (0.55 + 0.45 * abs(math.cos(phase)))
        self._last_speed = speed
        dx, dy = self._walk_dir
        adv = speed * dt
        if fol is not None:
            adv = min(adv, fol[0])           # non supera il punto da raggiungere
        x = self._walk_pos.x() + dx * adv
        y = self._walk_pos.y() + dy * adv
        # bordi dello schermo
        c = QPoint(int(x + cat.width() / 2), int(y + cat.height() / 2))
        scr = QApplication.screenAt(c) or QApplication.primaryScreen()
        geo = scr.availableGeometry()
        left, top = geo.left(), geo.top()
        right, bottom = geo.right() + 1 - cat.width(), geo.bottom() + 1 - cat.height()
        hit = False
        if x < left:
            x, dx, hit = left, abs(dx), True
        elif x > right:
            x, dx, hit = right, -abs(dx), True
        if y < top:
            y, dy, hit = top, abs(dy), True
        elif y > bottom:
            y, dy, hit = bottom, -abs(dy), True
        self._walk_dir = (dx, dy)
        self._walk_pos = QPointF(x, y)
        # sbatte contro il bordo: barcolla e cade (se e' appena partito rimbalza e basta)
        if hit and t > WALK_RAMP and not self._follow:
            self._start_stagger(1 if c.x() < geo.center().x() else -1)
        else:
            self._place()

    def _overflow(self, sign, maxdeg):
        """Quanto il gatto (tutta la tela dell'immagine) sporge dal suo rettangolo mentre si
        inclina fino a maxdeg gradi attorno al piede del lato 'sign'; in unita' dell'immagine."""
        px, py = FOOT_R if sign > 0 else FOOT_L
        W, H = self.img_w, self.img_h
        out = [0.0, 0.0, 0.0, 0.0]                      # sinistra, alto, destra, basso
        for a in sorted(set(range(0, int(maxdeg), 3)) | {int(maxdeg)}):
            th = math.radians(a * sign)
            sn, cs = math.sin(th), math.cos(th)
            lift = self._lift(a * sign)
            for x, y in ((0, 0), (W, 0), (0, H), (W, H)):
                dx, dy = x - px, y - py
                X = px + dx * cs - dy * sn
                Y = py + dx * sn + dy * cs - lift
                out = [max(out[0], -X), max(out[1], -Y), max(out[2], X - W), max(out[3], Y - H)]
        return out

    def _grow_pad(self, specs):
        """Allarga la finestra quanto serve per le inclinazioni (segno, gradi_massimi) date."""
        cat = self._cat_size()
        k = cat.width() / self.img_w
        pad = list(self._pad)
        for sign, maxdeg in specs:
            ov = self._overflow(sign, maxdeg)
            pad = [max(p_, math.ceil(o * k) + 8) for p_, o in zip(pad, ov)]
        self._apply_pad(tuple(pad))

    def _start_stagger(self, sign):
        """Urto contro il bordo: prima barcolla, poi cade verso la parte di schermo con piu' spazio."""
        self._pull_release()                        # se la coda era tirata, ora la molla
        self._walk_phase = "stagger"
        self._phase_t0 = self._walk_last = time.monotonic()
        self._fall_sign = sign
        self._fall_a0 = self._wobble
        self._stag_speed = self._last_speed
        self._lie_dur = random.uniform(*LIE_TIME)
        # da barcollante e da sdraiato il gatto sporge molto: la finestra si allarga in anticipo
        self._grow_pad(((1, STAGGER_AMP + 3), (-1, STAGGER_AMP + 3), (sign, 90)))

    def _recoil(self, now, f):
        """Rinculo: dopo l'urto scivola un po' indietro, sempre meno."""
        dt = min(now - self._walk_last, 0.1)
        self._walk_last = now
        cat = self._cat_size()
        dx, dy = self._walk_dir
        x = self._walk_pos.x() + dx * self._stag_speed * f * dt
        y = self._walk_pos.y() + dy * self._stag_speed * f * dt
        c = QPoint(int(x + cat.width() / 2), int(y + cat.height() / 2))
        geo = (QApplication.screenAt(c) or QApplication.primaryScreen()).availableGeometry()
        x = min(max(x, geo.left()), geo.right() + 1 - cat.width())
        y = min(max(y, geo.top()), geo.bottom() + 1 - cat.height())
        self._walk_pos = QPointF(x, y)

    def _fallen_step(self, now):
        el = now - self._phase_t0
        self._jit = 0
        sg = self._fall_sign
        target = 90.0 * sg
        ph = self._walk_phase
        if ph == "stagger":                         # barcolla, con l'ultima oscillazione verso il lato della caduta
            u = min(1.0, el / STAGGER_TIME)
            s0 = min(1.0, u * 5.0)
            s0 = s0 * s0 * (3 - 2 * s0)
            amp = STAGGER_AMP * (0.5 + 0.5 * u)
            osc = -sg * amp * math.sin(2 * math.pi * STAGGER_SWINGS * u)
            self._wobble = self._fall_a0 * (1 - s0) + osc
            self._recoil(now, (1 - u) ** 2)
            if u >= 1.0:
                self._walk_phase, self._phase_t0 = "fall", now
                self._fall_a0 = self._wobble
        elif ph == "fall":
            if el < FALL_TIME:                      # cade accelerando
                u = el / FALL_TIME
                self._wobble = self._fall_a0 + (target - self._fall_a0) * u * u
            elif el < FALL_TIME + BOUNCE_TIME:      # rimbalzino all'impatto
                b = (el - FALL_TIME) / BOUNCE_TIME
                self._wobble = target - sg * FALL_REBOUND * math.sin(math.pi * b) * (1 - b)
            else:
                self._wobble = target
                self._walk_phase, self._phase_t0 = "lie", now
        elif ph == "lie":
            self._wobble = target
            if el >= self._lie_dur:
                self._walk_phase, self._phase_t0 = "wiggle", now
        elif ph == "wiggle":                        # si scuote un poco prima di rialzarsi
            u = el / WIGGLE_TIME
            if u >= 1.0:
                self._wobble = target
                self._walk_phase, self._phase_t0 = "up", now
            else:
                env = math.sin(math.pi * u)
                rock = WIGGLE_ROCK * (0.5 + 0.5 * math.sin(2 * math.pi * WIGGLE_FREQ * el)) * env
                self._wobble = target - sg * rock
                self._jit = round(max(2.0, self._cat_w * WIGGLE_JITTER) * env
                                  * math.sin(2 * math.pi * WIGGLE_FREQ * 3.1 * el))
        else:                                       # "up": un tentativo fallito, poi si rialza
            u = el / UP_TIME
            if u >= 1.0:
                self._end_walk()
                return
            if u < 0.45:
                mag = 90.0 - 28.0 * math.sin(math.pi * u / 0.45)
            else:
                v = (u - 0.45) / 0.55
                mag = 90.0 * (1 - v * v * (3 - 2 * v))
            self._wobble = sg * mag
        if self._walk_start is not None:
            self._place()

    def _end_walk(self):
        self._pull_cancel()                 # ...e annulla anche la coda tirata
        self._alarm_stop()                  # qualsiasi fine della camminata ferma anche la sveglia
        if self._walk_start is None:
            return
        cat = self._cat_size()
        pos = QPoint(round(self._walk_pos.x()), round(self._walk_pos.y()))
        self._walk_start = None
        self._follow = False
        self._walk_phase = "walk"
        self._wobble = 0.0
        self._hop = 0.0
        self._jit = 0
        self._apply_pad((0, 0, 0, 0))
        self._rest_geo = QRect(pos, cat)
        QTimer.singleShot(80, self._settle)
        QTimer.singleShot(250, self._settle)
        self._apply_timer()
        self._schedule_save()
        self.update()

    # ------------------------------------------------------------ coda tirata --
    def _pull_busy(self):
        """True mentre la coda e' tirata o sta tornando a posto (la finestra e' allargata)."""
        return self._tp_drag or self._tp_rel_t0 is not None

    def _tail_local(self, m, s=1.0):
        """Nodi della coda (px dell'immagine, relativi al perno, coda non ruotata) con
        curvatura m e allungamento s: stessa costruzione di _draw_tail."""
        nodes, ds, sig = self._tail_geo[0], self._tail_geo[1], self._tail_geo[2]
        x, y = m * (nodes[0][0] - TAIL_PIVOT[0]), nodes[0][1] - TAIL_PIVOT[1]
        pts = []
        for i in range(len(nodes)):
            pts.append((x, y))
            if i < len(ds):
                a = math.radians(m * sig[i])
                x -= ds[i] * s * math.sin(a)
                y += ds[i] * s * math.cos(a)
        return pts

    def _tail_points(self, base, m, s=1.0):
        """Nodi della coda in coordinate-immagine, per la posa (base, m, s)."""
        c, sn = math.cos(math.radians(base)), math.sin(math.radians(base))
        return [(TAIL_PIVOT[0] + x * c - y * sn, TAIL_PIVOT[1] + x * sn + y * c)
                for x, y in self._tail_local(m, s)]

    def _tail_hit(self, pos):
        """Se pos (px del widget) cade sulla parte visibile della coda restituisce
        (nodo afferrato, scarto clic-nodo, verso della curvatura), altrimenti None."""
        if self._tail_geo is None or self._tail_pose is None:
            return None
        k = self._cat_w / self.img_w
        cx, cy = (pos.x() - self._off[0]) / k, (pos.y() - self._off[1]) / k
        ix, iy = int(cx), int(cy)
        if (cx >= 0 and cy >= 0 and ix < self.img_w and iy < self.img_h
                and self._body_img.pixelColor(ix, iy).alpha() > 40):
            return None                                   # li' c'e' il corpo: la coda e' dietro
        psi, m = self._tail_pose
        hw = self._tail_geo[6]
        best = None
        for j, (x, y) in enumerate(self._tail_points(psi, m)):
            d = math.hypot(cx - x, cy - y)
            if d <= max(hw[j] + TAIL_GRAB_SLACK, TAIL_GRAB_MIN_PX / k) and (best is None or d < best[0]):
                best = (d, j, cx - x, cy - y)
        if best is None:
            return None
        return max(best[1], 1), (best[2], best[3]), (1.0 if m >= 0 else -1.0)

    def _pull_pad(self):
        """Margini che bastano a contenere la coda allungata al massimo, in qualunque direzione."""
        k = self._cat_w / self.img_w
        reach = (TAIL_PULL_MAX * (sum(self._tail_geo[1]) + TAIL_TIP) + 10.0) * k
        px, py = TAIL_PIVOT[0] * k, TAIL_PIVOT[1] * k
        W, H = self._cat_w, self._cat_h
        return tuple(max(0, math.ceil(v)) for v in (reach - px, reach - py, reach - (W - px), reach - (H - py)))

    def _pull_begin(self, gpos, grab):
        self._tp_j, self._tp_off, self._tp_sign = grab
        self._tp_gpos = QPoint(gpos)
        self._tp_g = 1.0
        self._tp_drag = True
        self._tp_rel_t0 = None
        self._walk_pos = QPointF(self.pos())          # posizione del gatto: resta ferma durante la presa
        self._apply_pad(self._pull_pad())
        self._tp_pose = (self._tail_pose[0], self._tail_pose[1], 1.0)
        self.timer.start(16)
        self._pull_move(gpos)

    def _pull_solve(self, vx, vy):
        """Posa della coda (angolo, curvatura, allungamento) per cui il nodo afferrato arriva
        nel punto (vx, vy) relativo al perno. Prima la coda si raddrizza (curvatura -> 0), poi,
        a coda dritta, si allunga. Ultimo valore: True se ha raggiunto l'allungamento massimo."""
        j, sign = self._tp_j, self._tp_sign
        d = math.hypot(vx, vy)
        straight = max(1.0, math.hypot(*self._tail_local(0.0)[j]))
        if d >= straight:
            m, s = 0.0, min(TAIL_PULL_MAX, d / straight)
        else:
            if math.hypot(*self._tail_local(1.0)[j]) >= d:
                mm = 1.0                                  # piu' di cosi' non si accorcia
            else:
                lo, hi = 0.0, 1.0                         # distanza(lo) >= d > distanza(hi)
                for _ in range(16):
                    mid = (lo + hi) / 2
                    if math.hypot(*self._tail_local(mid)[j]) >= d:
                        lo = mid
                    else:
                        hi = mid
                mm = (lo + hi) / 2
            m, s = sign * mm, 1.0
        cx, cy = self._tail_local(m, s)[j]
        base = math.degrees(math.atan2(vy, vx) - math.atan2(cy, cx))
        base = (base + 180.0) % 360.0 - 180.0
        return base, m, s, d >= TAIL_PULL_MAX * straight * 0.999

    def _pull_target(self, ix, iy):
        """Posa della coda per un mouse nel punto (ix, iy) (px dell'immagine, gatto non inclinato)."""
        vx = ix - self._tp_off[0] - TAIL_PIVOT[0]
        vy = iy - self._tp_off[1] - TAIL_PIVOT[1]
        if math.hypot(vx, vy) < 1e-3:
            return None
        base, m, s, full = self._pull_solve(vx, vy)
        self._tp_pose = (base, m, s)
        return vx, vy, full

    def _pull_move(self, gpos):
        self._tp_gpos = QPoint(gpos)
        if self._walk_start is not None:
            return          # il gatto cammina: la posa si ricalcola a ogni fotogramma (vedi paintEvent)
        k = self._cat_w / self.img_w
        mv = QPointF(gpos) - self._walk_pos               # mouse rispetto all'angolo del gatto (px)
        r = self._pull_target(mv.x() / k, mv.y() / k)
        if r is not None and r[2]:                        # coda al 150%: il gatto parte in quella direzione
            vx, vy, _ = r                                 # (la coda resta tirata finche' si tiene premuto)
            d = math.hypot(vx, vy)
            self._tp_g = 1.0
            self._start_walk((vx / d, vy / d), from_pull=True)

    def _pull_release(self):
        """La coda viene lasciata: torna a posto con un rimbalzo elastico."""
        if not self._tp_drag:
            return
        self._tp_drag = False
        self._tp_rel = self._tp_pose
        self._tp_rel_t0 = time.monotonic()
        if (self._walk_start is not None and self._walk_phase == "walk"
                and not self._alarm_active):              # il gatto si ferma piano
            self._walk_dur = min(self._walk_dur, self._tp_rel_t0 - self._walk_start + WALK_RAMP)

    def _pull_pose(self):
        """(peso, angolo, curvatura, allungamento) della posa tirata, oppure None.
        Il peso vale 1 mentre si tira e poi oscilla smorzato verso 0 (rimbalzo)."""
        if self._tp_drag:
            return (1.0,) + tuple(self._tp_pose)
        if self._tp_rel_t0 is not None:
            t = time.monotonic() - self._tp_rel_t0
            if t < TAIL_PULL_REL_TIME:
                w = math.exp(-TAIL_PULL_DECAY * t) * math.cos(TAIL_PULL_OMEGA * t)
                return (w,) + tuple(self._tp_rel)
        return None

    def _pull_step(self):
        if self._tp_rel_t0 is not None and time.monotonic() - self._tp_rel_t0 >= TAIL_PULL_REL_TIME:
            self._pull_finish()

    def _pull_finish(self):
        self._tp_rel_t0 = None
        if self._walk_start is None:                      # se sta camminando, la finestra la gestisce la camminata
            self._pull_restore()

    def _pull_cancel(self):
        if not self._pull_busy():
            return
        self._tp_drag = False
        self._tp_rel_t0 = None
        if self._walk_start is None:
            self._pull_restore()

    def _pull_restore(self):
        """Toglie i margini della coda tirata e rimette la finestra attorno al solo gatto."""
        cat = self._cat_size()
        pos = QPoint(round(self._walk_pos.x()), round(self._walk_pos.y()))
        self._apply_pad((0, 0, 0, 0))
        self._rest_geo = QRect(pos, cat)
        QTimer.singleShot(80, self._settle)
        QTimer.singleShot(250, self._settle)
        self._apply_timer()
        self._schedule_save()
        self.update()

    # --------------------------------------------------------------- matrioska --
    def set_matrioska(self, on, save=True):
        """Crea / elimina il gatto piccolo che segue questo."""
        self.matrioska_on = on
        if on and self.child is None:
            c = CatClock(self.settings, leader=self)
            c.window_menu = self.window_menu
            self.child = c
            if self.isVisible():
                c.show_near()
        elif not on and self.child is not None:
            c, self.child = self.child, None
            c.hide()
            c.deleteLater()
        if self.mini is not None:
            self.mini.update_child()          # se l'orologio e' nascosto, anche la miniatura ha (o perde) la sua piccola
        if save:
            self.save_settings()

    def sync_from_leader(self, force=False):
        """Il piccolo copia aspetto e opzioni del grande (colori, numeri, coda, tremolio) e ne segue la dimensione; l'orologio e' di tipo opposto (a lancette <-> digitale)."""
        L = self.leader
        key = (L.body_color.rgb(), L.body_color2.rgb(), L.gradient_on, L.dial_color.rgb(),
               L.numbers_color.rgb(), L.tail_color.rgb(), L.hand_color.rgb(), L.digital_on,
               L.show_seconds, L.tail_on, L.shake_on, L._cat_w)
        if key == self._sync_key and not force:
            return
        self._sync_key = key
        self.body_color, self.body_color2 = QColor(L.body_color), QColor(L.body_color2)
        self.gradient_on = L.gradient_on
        self.dial_color, self.numbers_color = QColor(L.dial_color), QColor(L.numbers_color)
        self.tail_color, self.hand_color = QColor(L.tail_color), QColor(L.hand_color)
        self.digital_on, self.show_seconds = not L.digital_on, L.show_seconds   # tipo di orologio opposto al grande
        self.tail_on, self.shake_on = L.tail_on, L.shake_on
        self._cache = None
        self._tail_cache = None
        w = max(MATRIOSKA_MIN_WIDTH, round(L._cat_w * MATRIOSKA_SCALE))
        if w != self._cat_w:
            self.set_width(w, keep_center=True)
        self._apply_timer()
        self.update()

    def _follow_target(self):
        """Dove deve stare il piccolo (angolo alto-sinistra): accanto al grande, dal lato con piu'
        spazio sullo schermo, con i piedi alla stessa altezza."""
        L = self.leader
        lp = L._cat_pos()
        c = lp + QPoint(L._cat_w // 2, L._cat_h // 2)
        geo = (QApplication.screenAt(c) or QApplication.primaryScreen()).availableGeometry()
        if L.matrioska_off is not None:      # posizione scelta dall'utente trascinando il piccolo
            x = lp.x() + L.matrioska_off[0] * L._cat_w
            y = lp.y() + L.matrioska_off[1] * L._cat_w
        else:
            gap = round(L._cat_w * MATRIOSKA_GAP)
            x = lp.x() - gap - self._cat_w if c.x() > geo.center().x() else lp.x() + L._cat_w + gap
            foot = (FOOT_L[1] + FOOT_R[1]) / 2.0
            y = lp.y() + foot * L._cat_w / self.img_w - foot * self._cat_w / self.img_w
        x = max(geo.left(), min(x, geo.right() + 1 - self._cat_w))
        y = max(geo.top(), min(y, geo.bottom() + 1 - self._cat_h))
        return QPointF(x, y)

    def set_matrioska_offset(self, child):
        """Il piccolo e' stato trascinato: dove l'utente lo lascia diventa la sua posizione standard
        rispetto al grande (misurata in larghezze del grande, cosi' vale anche se cambia la dimensione)."""
        lp, cp = self._cat_pos(), child.pos()
        self.matrioska_off = ((cp.x() - lp.x()) / self._cat_w, (cp.y() - lp.y()) / self._cat_w)
        self.save_settings()

    def show_near(self):
        """Il piccolo compare subito accanto al grande."""
        if self.leader is None:
            return
        self.sync_from_leader(force=True)
        t = self._follow_target()
        self.move(round(t.x()), round(t.y()))
        self.show()
        self.raise_()

    def _follow_check(self):
        """Fermo e troppo lontano dal grande: si mette a camminargli dietro."""
        if (self._walk_start is not None or self._pull_busy() or self._shake_start is not None
                or self._drag_off is not None or self._resize_start is not None
                or self._alarm_edit is not None or self._alarm_active or not self.isVisible()):
            return
        t, cur = self._follow_target(), self.pos()
        dx, dy = t.x() - cur.x(), t.y() - cur.y()
        d = math.hypot(dx, dy)
        if d > MATRIOSKA_START * self._cat_w:
            self._start_walk((dx / d, dy / d), follow=True)

    def _follow_state(self, t, dt):
        """Durante l'inseguimento: (distanza dal punto da raggiungere, direzione, intensita' 0-1).
        L'intensita' sale con la distanza (a pochi px il gatto sta fermo); l'inseguimento finisce
        dopo MATRIOSKA_CALM secondi passati vicini, ma riparte da solo se il grande si muove."""
        tgt = self._follow_target()
        vx, vy = tgt.x() - self._walk_pos.x(), tgt.y() - self._walk_pos.y()
        d = math.hypot(vx, vy)
        w = self._cat_w
        stop, start = MATRIOSKA_STOP * w, MATRIOSKA_START * w
        fi = max(0.0, min(1.0, (d - stop) / max(1.0, start - stop)))
        self._follow_calm = self._follow_calm + dt if d < stop else 0.0
        if self._follow_calm > MATRIOSKA_CALM:
            self._walk_dur = min(self._walk_dur, t + WALK_RAMP)      # si ferma piano
        else:
            self._walk_dur = t + WALK_RAMP + 1.0
        direction = (vx / d, vy / d) if d > 1e-3 else self._walk_dir
        return d, direction, fi

    # ------------------------------------------------------------- sveglia --
    def set_alarm_on(self, on):
        self.alarm_on = on
        if on:
            self._alarm_fired = None
        else:
            self._alarm_stop_all()
        self.save_settings()

    def set_alarm_time(self, qtime):
        self.alarm_time = QTime(qtime.hour(), qtime.minute())
        self._alarm_fired = None
        self.save_settings()

    def start_alarm_edit(self):
        """Scrive l'ora della sveglia direttamente sul quadrante, dove si legge l'ora."""
        if self.leader is not None:          # la sveglia e' una sola: quella del gatto grande
            self.leader.start_alarm_edit()
            return
        if self._alarm_edit is not None:
            return
        if not self.isVisible():
            self.show()
        self._end_walk()
        self._end_shake()
        t = self.alarm_time
        self._alarm_edit = [t.hour() // 10, t.hour() % 10, t.minute() // 10, t.minute() % 10]
        self._edit_pos = 0
        self._edit_t0 = time.monotonic()
        self.activateWindow()
        self.setFocus()
        self.update()

    def _alarm_edit_commit(self):
        d = self._alarm_edit
        if d is None:
            return
        self._alarm_edit = None
        self.set_alarm_time(QTime(d[0] * 10 + d[1], d[2] * 10 + d[3]))
        self.set_alarm_on(True)                 # impostare l'ora attiva la sveglia
        if self.alarm_changed_cb:
            self.alarm_changed_cb()
        self.update()

    def _alarm_edit_cancel(self):
        self._alarm_edit = None
        self.update()

    def _alarm_edit_adjust(self, delta):
        d = self._alarm_edit
        if self._edit_pos < 2:
            v = (d[0] * 10 + d[1] + delta) % 24
            d[0], d[1] = v // 10, v % 10
        else:
            v = (d[2] * 10 + d[3] + delta) % 60
            d[2], d[3] = v // 10, v % 10
        self.update()

    def _alarm_edit_type(self, n):
        d, i = self._alarm_edit, self._edit_pos
        if i == 0 and n > 2:                    # "7" -> 07:
            d[0], d[1], i = 0, n, 2
        elif i == 1 and d[0] == 2 and n > 3:    # 24-29 non esistono
            return
        elif i == 2 and n > 5:                  # minuti 60-99 non esistono
            return
        else:
            d[i] = n
            i += 1
        if d[0] == 2 and d[1] > 3:
            d[1] = 3
        self._edit_pos = min(i, 3)
        self.update()

    def _alarm_edit_key(self, e):
        self._edit_t0 = time.monotonic()        # il testo resta visibile mentre si scrive
        key = e.key()
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self._alarm_edit_commit()
        elif key == Qt.Key_Escape:
            self._alarm_edit_cancel()
        elif Qt.Key_0 <= key <= Qt.Key_9:
            self._alarm_edit_type(key - Qt.Key_0)
        elif key == Qt.Key_Backspace:
            self._edit_pos = max(0, self._edit_pos - 1)
            self._alarm_edit[self._edit_pos] = 0
            self.update()
        elif key == Qt.Key_Left:
            self._edit_pos = max(0, self._edit_pos - 1)
            self.update()
        elif key == Qt.Key_Right:
            self._edit_pos = min(3, self._edit_pos + 1)
            self.update()
        elif key == Qt.Key_Up:
            self._alarm_edit_adjust(1)
        elif key == Qt.Key_Down:
            self._alarm_edit_adjust(-1)

    def focusOutEvent(self, e):
        self._alarm_edit_commit()               # cliccando altrove l'ora scritta viene confermata
        super().focusOutEvent(e)

    def _alarm_check(self):
        """Una volta al secondo: se e' l'ora della sveglia (e non ha gia' suonato in questo minuto)."""
        if not self.alarm_on or self._alarm_active:
            return
        n = datetime.now()
        if (n.hour, n.minute) != (self.alarm_time.hour(), self.alarm_time.minute()):
            return
        key = (n.year, n.month, n.day, n.hour, n.minute)
        if key == self._alarm_fired:
            return
        if self._walk_start is not None and self._walk_phase != "walk":
            return                           # sta cadendo/e' a terra: riprova al prossimo secondo
        self._alarm_fired = key
        self._start_alarm()

    def _start_alarm(self):
        """Il gatto oscilla sul posto come quando cammina, ma con un angolo maggiore, e miagola."""
        if self._alarm_active:
            return
        self._alarm_edit = None
        self._end_walk()
        self._end_shake()
        if not self.isVisible():
            self.show()
        self._rest_geo = None
        self._alarm_active = True
        self._walk_dir = (0.0, 0.0)
        self._walk_dur = ALARM_DURATION
        self._walk_pos = QPointF(self.pos())
        self._wobble = 0.0
        self._hop = 0.0
        self._walk_phase = "walk"
        # margini abbastanza larghi per l'inclinazione maggiore e per il saltello
        self._pad = (0, 0, 0, 0)
        self._grow_pad(((1, ALARM_WOBBLE_ANGLE), (-1, ALARM_WOBBLE_ANGLE)))
        l, t, r, b = self._pad
        k = self._cat_w / self.img_w
        self._apply_pad((l, t + math.ceil(WALK_HOP * k) + 4, r, b))
        self._walk_start = self._walk_last = time.monotonic()
        self.timer.start(16)
        self._meow_token += 1
        self._meow(self._meow_token)
        if self.child is not None:           # il piccolo oscilla insieme (senza audio)
            self.child._start_alarm()

    def _alarm_stop(self):
        if not self._alarm_active:
            return
        self._alarm_active = False
        self._meow_token += 1
        if self._player is not None:
            self._player.stop()
        if self.child is not None:
            self.child._end_walk()

    def _alarm_stop_all(self):
        """Disattivazione dal menu: ferma anche una suoneria in corso."""
        if self._alarm_active:
            self._end_walk()

    def _meow(self, token):
        if not self._alarm_active or token != self._meow_token:
            return
        if self.leader is not None:          # il piccolo non miagola: lo fa gia' il grande
            return
        if (self._player is None and not self._player_failed
                and HAVE_MULTIMEDIA and ALARM_FILE.exists()):
            self._player = QMediaPlayer(self)
            self._player.setMedia(QMediaContent(QUrl.fromLocalFile(str(ALARM_FILE))))
            self._player.setVolume(ALARM_VOLUME)
            self._player.mediaStatusChanged.connect(self._on_media_status)
            self._player.error.connect(self._on_media_error)
        if self._player is not None and not self._player_failed:
            self._player.setPosition(0)
            self._player.play()
        else:                                # niente audio disponibile: beep di sistema
            QApplication.beep()
            QTimer.singleShot(int((ALARM_MEOW_GAP + 0.6) * 1000), lambda: self._meow(token))

    def _on_media_status(self, status):
        if status == QMediaPlayer.EndOfMedia and self._alarm_active:
            token = self._meow_token
            QTimer.singleShot(int(ALARM_MEOW_GAP * 1000), lambda: self._meow(token))

    def _on_media_error(self, *_):
        self._player_failed = True           # codec mancante o file illeggibile: passa al beep
        if self._alarm_active:
            self._meow(self._meow_token)

    def showEvent(self, e):
        n = datetime.now()
        self._last_min = (n.hour, n.minute)    # niente tremolio "arretrato" quando riappare
        self._apply_timer()
        if self.mini is not None:
            self.mini.hide_mini()              # l'orologio e' tornato: la miniatura sparisce
        if self.child is not None:
            self.child.show_near()             # ...e il gatto piccolo gli si rimette accanto
        super().showEvent(e)

    def hideEvent(self, e):
        self._alarm_edit = None
        self._end_shake()
        self._end_walk()
        self.timer.stop()          # nascosto = nessun consumo di CPU
        if self.child is not None:
            self.child.hide()
        if self.mini is not None and self.mini_on:
            self.mini.show_mini()              # ...e al suo posto compare il gatto in miniatura
        super().hideEvent(e)

    # ---------------------------------------------------------- geometria --
    def set_width(self, w, keep_center=False):
        self._end_walk()
        self._rest_geo = None
        w = max(self._min_w, min(MAX_WIDTH, int(w)))
        h = round(w * self.ratio)
        self._cat_w, self._cat_h = w, h
        if keep_center and self.isVisible():
            c = self.geometry().center()
            self.setGeometry(c.x() - w // 2, c.y() - h // 2, w, h)
        else:
            self.resize(w, h)
        self._cache = None
        self.update()

    def _restore_position(self):
        x = self.settings.value("x", None)
        y = self.settings.value("y", None)
        if x is None or y is None:
            geo = QApplication.primaryScreen().availableGeometry()
            self.move(geo.right() - self.width() - 40, geo.top() + 40)
            return
        pos = QPoint(int(x), int(y))
        rect = QRect(pos, self.size())
        # se lo schermo salvato non c'è più, riporta l'orologio in vista
        if not any(s.availableGeometry().intersects(rect) for s in QApplication.screens()):
            geo = QApplication.primaryScreen().availableGeometry()
            pos = QPoint(geo.right() - self.width() - 40, geo.top() + 40)
        self.move(pos)

    def reset_geometry(self):
        self.settings.remove("x")
        self.settings.remove("y")
        self.set_width(DEFAULT_WIDTH)
        self._restore_position()
        self.save_settings()

    # ----------------------------------------------------------- impostaz. --
    def save_settings(self):
        if self.leader is not None:           # il gatto piccolo non salva nulla: dipende dal grande
            return
        s = self.settings
        pos = self._cat_pos()
        s.setValue("x", pos.x())
        s.setValue("y", pos.y())
        s.setValue("width", self._cat_size().width())
        s.setValue("show_seconds", self.show_seconds)
        s.setValue("digital_on", self.digital_on)
        s.setValue("locked", self.locked)
        s.setValue("mini_on", self.mini_on)
        s.setValue("matrioska_on", self.matrioska_on)
        if self.matrioska_off is not None:
            s.setValue("matrioska_dx", self.matrioska_off[0])
            s.setValue("matrioska_dy", self.matrioska_off[1])
        s.setValue("body_color", self.body_color.name())
        s.setValue("body_color2", self.body_color2.name())
        s.setValue("gradient_on", self.gradient_on)
        s.setValue("hand_color", self.hand_color.name())
        s.setValue("dial_color", self.dial_color.name())
        s.setValue("tail_on", self.tail_on)
        s.setValue("shake_on", self.shake_on)
        s.setValue("alarm_on", self.alarm_on)
        s.setValue("alarm_time", self.alarm_time.toString("HH:mm"))
        s.setValue("tail_color", self.tail_color.name())
        s.setValue("numbers_color", self.numbers_color.name())
        s.sync()

    def _schedule_save(self):
        self.save_timer.start(400)

    def _keep_on_top(self):
        if self.isVisible():
            self.raise_()

    # ------------------------------------------------------------ disegno --
    def _fill_brush(self, rect, c1, c2, gradient):
        if not gradient:
            return QBrush(c1)
        # sfumatura a 45 gradi veri (alto-sinistra -> basso-destra), indipendente dalle proporzioni
        t = (rect.width() + rect.height()) / 2.0
        g = QLinearGradient(QPointF(0, 0), QPointF(t, t))
        g.setColorAt(0.0, c1)
        g.setColorAt(1.0, c2)
        return QBrush(g)

    @staticmethod
    def _tinted(layer, brush, size):
        """Copia del livello (solo la forma) riempita con il pennello dato."""
        pm = QPixmap(size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        p.drawPixmap(pm.rect(), layer)
        p.setCompositionMode(QPainter.CompositionMode_SourceIn)
        p.fillRect(pm.rect(), brush)
        p.end()
        return pm

    def _compose(self, size, dpr, c1, c2, gradient, ring=True, with_dial=True, tail=False):
        """Corpo (tinta unita o sfumatura) + disco quadrante + dettagli, alla dimensione richiesta.
        with_dial=False: senza quadrante e numeri; tail=True: con la coda dietro al corpo."""
        target = size * dpr
        pm = QPixmap(target)
        pm.fill(Qt.transparent)
        rect = pm.rect()
        body = self._tinted(self.body, self._fill_brush(rect, c1, c2, gradient), target)
        dial = self._tinted(self.dial, QBrush(self.dial_color), target)
        nums = self._tinted(self.numbers, QBrush(self.numbers_color), target)
        p = QPainter(pm)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        if tail:
            p.drawPixmap(0, 0, self._tinted(self.tail, QBrush(self.tail_color), target))
        p.drawPixmap(0, 0, body)
        if with_dial:
            p.drawPixmap(0, 0, dial)
            if ring:
                p.drawPixmap(0, 0, nums)
        p.drawPixmap(rect, self.details)
        p.end()
        pm.setDevicePixelRatio(dpr)
        return pm

    @property
    def face(self):
        pm = self._compose(self.body.size(), 1.0, self.body_color,
                           self.body_color2, self.gradient_on)
        p = QPainter(pm)
        p.setCompositionMode(QPainter.CompositionMode_DestinationOver)
        p.drawPixmap(0, 0, self._tinted(self.tail, QBrush(self.tail_color), self.tail.size()))  # coda dietro al corpo
        p.end()
        return pm

    def mini_face(self, w, dpr):
        """Gatto per la miniatura: corpo e dettagli, senza coda (animata a parte), quadrante, numeri e lancette."""
        key = (w, dpr, self.body_color.rgb(), self.body_color2.rgb(), self.gradient_on)
        cache = self.__dict__.setdefault("_mini_cache", {})
        pm = cache.get(key)
        if pm is None:
            if len(cache) > 6:
                cache.clear()
            pm = self._compose(QSize(w, round(w * self.ratio)), dpr, self.body_color,
                               self.body_color2, self.gradient_on, ring=False, with_dial=False)
            cache[key] = pm
        return pm

    def set_colors(self, c1=None, c2=None, gradient=None, save=True):
        if c1 is not None:
            self.body_color = QColor(c1)
        if c2 is not None:
            self.body_color2 = QColor(c2)
        if gradient is not None:
            self.gradient_on = gradient
        self._cache = None
        self.update()
        if save:
            self.save_settings()

    def set_hand_color(self, color, save=True):
        self.hand_color = QColor(color)
        self.update()
        if save:
            self.save_settings()

    def set_digital(self, on):
        self.digital_on = on
        self._cache = None
        self.update()
        self.save_settings()

    def set_numbers_color(self, color, save=True):
        self.numbers_color = QColor(color)
        self._cache = None
        self.update()
        if save:
            self.save_settings()

    def set_dial_color(self, color, save=True):
        self.dial_color = QColor(color)
        self._cache = None
        self.update()
        if save:
            self.save_settings()

    def set_body_color(self, color, save=True):      # compatibilita'
        self.set_colors(c1=color, save=save)

    def _scaled_face(self):
        dpr = self.devicePixelRatioF()
        ring = not (self.digital_on and DIGITAL_HIDE_RING)
        key = (self._cat_size(), dpr, self.body_color.rgb(), self.body_color2.rgb(), self.gradient_on,
               self.dial_color.rgb(), self.numbers_color.rgb(), ring)
        if self._cache is None or self._cache[0] != key:
            self._cache = (key, self._compose(self._cat_size(), dpr, self.body_color,
                                              self.body_color2, self.gradient_on, ring))
        return self._cache[1]

    def _build_tail_geometry(self):
        """Ricava dalla PNG la linea d'asse della coda (dal perno alla punta), seguendola a passi
        di TAIL_STRIP px. Serve a tagliare la coda in strisce perpendicolari all'asse e a
        ricomporla con la curvatura moltiplicata per un fattore m: m = 1 e' l'immagine
        originale, m = -1 l'immagine riflessa sull'asse verticale, m = 0 la coda dritta."""
        img = self.tail.toImage()
        W, H = img.width(), img.height()

        def solid(x, y):
            xi, yi = int(round(x)), int(round(y))
            return 0 <= xi < W and 0 <= yi < H and (img.pixel(xi, yi) >> 24) > 128

        y0 = int(TAIL_PIVOT[1])
        xs = [x for x in range(W) if solid(x, y0)]
        if not xs:
            return None
        p, d = (sum(xs) / len(xs), float(y0)), (0.0, 1.0)
        nodes = [p]
        hws = [0.0]                                               # semispessore della coda in ogni nodo
        for _ in range(300):
            qx, qy = p[0] + TAIL_STRIP * d[0], p[1] + TAIL_STRIP * d[1]
            nx, ny = d[1], -d[0]                                  # normale all'asse
            ts = [i * 0.5 for i in range(-int(TAIL_HALF_W * 2), int(TAIL_HALF_W * 2) + 1)
                  if solid(qx + nx * i * 0.5, qy + ny * i * 0.5)]
            if len(ts) < 10:                                      # la coda e' finita
                break
            mid = (ts[0] + ts[-1]) / 2                            # ricentra sulla sezione
            q = (qx + nx * mid, qy + ny * mid)
            vx, vy = q[0] - p[0], q[1] - p[1]
            L = math.hypot(vx, vy)
            d, p = (vx / L, vy / L), q
            nodes.append(p)
            hws.append((ts[-1] - ts[0]) / 2)
        if len(nodes) < 4:
            return None
        hws[0] = hws[1]
        nodes = [(x + 0.5, y + 0.5) for x, y in nodes]            # coordinate continue dei pixel
        for _ in range(2):                                        # leggera lisciatura 1-2-1
            nodes = ([nodes[0]]
                     + [(0.25 * a[0] + 0.5 * b[0] + 0.25 * c[0], 0.25 * a[1] + 0.5 * b[1] + 0.25 * c[1])
                        for a, b, c in zip(nodes, nodes[1:], nodes[2:])]
                     + [nodes[-1]])
        n = len(nodes)
        ds, sig = [], []
        for a, b in zip(nodes, nodes[1:]):
            vx, vy = b[0] - a[0], b[1] - a[1]
            ds.append(math.hypot(vx, vy))
            sig.append(math.degrees(math.atan2(-vx, vy)))         # angolo di rotate() che porta (0,1) sul segmento
        tau = []
        for i in range(n):
            a, b = nodes[max(i - 1, 0)], nodes[min(i + 1, n - 1)]
            tau.append(math.degrees(math.atan2(-(b[0] - a[0]), b[1] - a[1])))
        back = [0.5 * ds[max(i - 1, 0)] + TAIL_PAD for i in range(n)]
        fwd = [0.5 * ds[i] + TAIL_PAD for i in range(n - 1)] + [TAIL_TIP + TAIL_PAD]
        return nodes, ds, sig, tau, back, fwd, hws

    @staticmethod
    def _ss(x):
        x = min(1.0, max(0.0, x))
        return x * x * (3 - 2 * x)

    def _tail_phase(self, now):
        """Pesi del comportamento della coda nella fase corrente della caduta:
        a = oscilla come un pendolo, r = alzata in segno di panico, p = fa da puntello sul pavimento,
        stiff = moltiplicatore della rigidita' del giunto (scatti), fl = intensita' del fremito.
        Quello che resta (1 - a) e' la coda che penzola per gravita' (vedi _tail_rest)."""
        a, r, p, stiff, fl = 1.0, 0.0, 0.0, 1.0, 0.0
        if self._walk_start is None or self._walk_phase == "walk":
            return a, r, p, stiff, fl
        ph, el = self._walk_phase, now - self._phase_t0
        ss = self._ss
        if ph == "stagger":                       # barcolla: la coda si irrigidisce e comincia ad alzarsi
            u = min(1.0, el / STAGGER_TIME)
            a, r = 1.0 - 0.6 * u, 0.4 * ss(u)
        elif ph == "fall":                        # cade: la coda scatta verso l'alto
            a, r, stiff = 0.4 * max(0.0, 1.0 - el / FALL_TIME), 0.4 + 0.6 * ss(el / FALL_TIME), 1.4
        elif ph in ("lie", "wiggle"):             # a terra: resta alzata e freme
            a, r, stiff = 0.0, 1.0, 1.4
            fl = ss(el / 0.4) if ph == "lie" else 1.0
        elif ph == "up":                          # prova a rialzarsi: scatta in basso, fa da puntello
            u = el / UP_TIME
            alive = ss((u - 0.7) / 0.3)
            a, r = alive, 1.0 - ss(u / 0.12)
            p = ss(u / 0.12) * (1.0 - alive)
            stiff = 1.0 + 0.9 * (1.0 - ss(u / 0.3))
        return a, r, p, stiff, fl

    def _tail_rest(self, w):
        """Angolo della coda rispetto al corpo (gradi) quando penzola per gravita' col gatto
        inclinato di w gradi: in verticale nel mondo; da sdraiato non puo' scendere sotto il
        pavimento, quindi si appoggia lasciando la punta a terra."""
        aw = abs(w)
        if aw < TAIL_FLOOR_FROM:
            return -w
        fx, fy = FOOT_R if w > 0 else FOOT_L
        th = math.radians(w)
        # altezza del perno della coda sul pavimento (px dell'immagine)
        h = self._lift(w) - ((TAIL_PIVOT[0] - fx) * math.sin(th) + (TAIL_PIVOT[1] - fy) * math.cos(th))
        floor = math.degrees(math.acos(max(-1.0, min(1.0, h / self._tail_reach))))
        v = min(1.0, (aw - TAIL_FLOOR_FROM) / (TAIL_FLOOR_TO - TAIL_FLOOR_FROM))
        s = v * v * (3 - 2 * v)
        return math.copysign(floor * s, w) - w

    def _tail_update(self):
        """Fa avanzare la coda e ne restituisce l'angolo rispetto al corpo (gradi). Il giunto e'
        una molla smorzata che insegue un angolo-obiettivo: l'oscillazione a pendolo quando il gatto
        e' vivo, la posizione di riposo per gravita' quando e' a terra. Il moto e' integrato
        nel sistema 'mondo', quindi l'inclinazione brusca del corpo la fa ritardare e frustare."""
        now = time.monotonic()
        dt = now - self._tail_t
        w = self._wobble
        stale = dt > 0.3
        if dt < 1e-4 and self._tail_init:
            return self._tail_omega - w
        dt = min(max(dt, 1e-3), 0.05)
        self._tail_t = now
        raw = max(-3000.0, min(3000.0, (w - self._tail_w_prev) / dt)) if not stale else 0.0
        self._tail_wv = 0.5 * self._tail_wv + 0.5 * raw
        self._tail_w_prev = w
        wv = self._tail_wv
        a, r_up, push, stiff, fl = self._tail_phase(now)
        self._tail_push = push
        wt = 2 * math.pi / TAIL_PERIOD
        rr = wt / TAIL_FALL_WN                      # il giunto smorza/ritarda un po': si compensa
        gain = 1.0 / math.sqrt((1 - rr * rr) ** 2 + (2 * TAIL_FALL_ZETA * rr) ** 2)
        osc = osc_v = 0.0
        if self.tail_on:
            osc = TAIL_ANGLE / gain * math.cos(wt * time.time())
            osc_v = -TAIL_ANGLE / gain * wt * math.sin(wt * time.time())
        # coda alzata: dal lato opposto alla caduta (contrappeso), nel mondo a TAIL_UP_ANGLE
        flutter = fl * TAIL_FLUTTER * math.sin(2 * math.pi * TAIL_FLUTTER_HZ * now)
        up = self._fall_sign * TAIL_UP_ANGLE - w + flutter
        up = max(-TAIL_REL_MAX, min(TAIL_REL_MAX, up))
        target = a * osc + (1 - a) * ((1 - r_up) * self._tail_rest(w) + r_up * up)
        if stale or not self._tail_init:
            self._tail_omega = w + target
            self._tail_omega_v = wv + a * osc_v
            self._tail_init = True
        sub = max(1, int(dt / 0.004))
        h = dt / sub
        wn = TAIL_FALL_WN * stiff
        wn2, zw = wn ** 2, 2 * TAIL_FALL_ZETA * wn
        om, ov = self._tail_omega, self._tail_omega_v
        for _ in range(sub):
            ov += (-wn2 * (om - w - target) - zw * (ov - wv)) * h
            om += ov * h
        psi = om - w
        if abs(psi) > TAIL_REL_MAX:                 # fine corsa del giunto
            psi = math.copysign(TAIL_REL_MAX, psi)
            om = w + psi
            if (ov - wv) * psi > 0:
                ov = wv
        self._tail_omega, self._tail_omega_v = om, ov
        return psi

    def _draw_tail(self, p, k, base, m, s=1.0):
        """Coda: base ruotata attorno al perno di 'base' gradi e curvatura moltiplicata per m
        (1 = come la PNG, -1 = PNG riflessa sull'asse verticale, 0 = dritta).
        s = allungamento lungo l'asse (1 = lunghezza originale, 1.5 = +50%)."""
        tail = self._scaled_tail()
        dpr = tail.devicePixelRatio()
        pvx, pvy = TAIL_PIVOT
        px, py = pvx * k, pvy * k
        W = self._cat_w
        p.save()
        p.translate(px, py)
        p.rotate(base)
        # parte sopra il perno (nascosta dal corpo): rigida
        p.drawPixmap(QRectF(-px, -py, W, py), tail, QRectF(0, 0, W * dpr, py * dpr))
        geo = self._tail_geo
        if geo is None:                                           # fallback: coda rigida
            p.drawPixmap(QRectF(-px, 0, W, self._cat_h - py), tail,
                         QRectF(0, py * dpr, W * dpr, (self._cat_h - py) * dpr))
            p.restore()
            return
        nodes, ds, sig, tau, back, fwd = geo[:6]
        mirror = m < 0
        x, y = m * (nodes[0][0] - pvx), nodes[0][1] - pvy         # posizione (rispetto al perno) del 1o nodo
        hw = TAIL_HALF_W * k
        for i, (cx, cy) in enumerate(nodes):
            p.save()
            p.translate(x * k, y * k)
            p.rotate(m * tau[i])
            if mirror:
                p.scale(-1.0, 1.0)                                # specchia la sezione attorno all'asse
            if s != 1.0:
                p.scale(1.0, s)                                   # allunga la striscia lungo l'asse
            p.setClipRect(QRectF(-hw, -back[i] * k, 2 * hw, (back[i] + fwd[i]) * k), Qt.IntersectClip)
            p.rotate(-tau[i])                                     # porta la striscia originale in verticale
            p.translate(-(cx - pvx) * k, -(cy - pvy) * k)
            p.drawPixmap(QPointF(-px, -py), tail)
            p.restore()
            if i < len(ds):                                       # nodo successivo: segmento con angolo m*sig
                a = math.radians(m * sig[i])
                x -= ds[i] * s * math.sin(a)
                y += ds[i] * s * math.cos(a)
        p.restore()

    def _scaled_tail(self):
        dpr = self.devicePixelRatioF()
        key = (self._cat_size(), dpr, self.tail_color.rgb())
        if self._tail_cache is None or self._tail_cache[0] != key:
            pm = self._tinted(self.tail, QBrush(self.tail_color), self._cat_size() * dpr)
            pm.setDevicePixelRatio(dpr)
            self._tail_cache = (key, pm)
        return self._tail_cache[1]

    def set_tail_color(self, color, save=True):
        self.tail_color = QColor(color)
        self._tail_cache = None
        self.update()
        if save:
            self.save_settings()

    def _digital_metrics(self):
        if self._dig is None:
            font = QFont()
            font.setFamilies(DIGITAL_FONTS)
            font.setBold(True)
            font.setPixelSize(100)
            fm = QFontMetricsF(font)
            cell = max(fm.horizontalAdvance(str(d)) for d in range(10))   # cifre a larghezza fissa
            self._dig = (font, fm, cell, fm.horizontalAdvance(":"), fm.tightBoundingRect("0").height())
        return self._dig

    def _draw_digital(self, p, k, now, edit=False):
        """Ora in cifre al centro del quadrante: riempimento = colore lancette,
        bordo = colore dei numeri del quadrante. (p e' gia' centrato sul quadrante.)"""
        font, fm, cell, colon_w, h0 = self._digital_metrics()
        hh = now.hour if DIGITAL_24H else (now.hour % 12 or 12)
        s1 = DIGITAL_W / (cell * 4 + colon_w)             # scala di ore:minuti
        s2 = s1 * DIGITAL_SEC_SCALE                       # scala dei secondi
        h1, h2 = h0 * s1, h0 * s2
        gap = 0.28 * h1
        block = h1 + (gap + h2 if self.show_seconds else 0.0)
        top = -block / 2 + DIGITAL_DY
        lines = [(f"{hh:02d}:{now.minute:02d}", s1, top + h1)]
        if edit:                                          # ora della sveglia al posto dell'ora (stessa posizione)
            if int((time.monotonic() - self._edit_t0) * EDIT_BLINK_HZ * 2) % 2 == 1:
                return                                    # fase "spento" del lampeggio: si attende un input
            d = self._alarm_edit
            lines = [(f"{d[0]}{d[1]}:{d[2]}{d[3]}", s1, top + h1)]
        elif self.show_seconds:
            lines.append((f"{now.second:02d}", s2, top + h1 + gap + h2))
        path = QPainterPath()
        for text, sc, base in lines:
            x = -sum(colon_w if ch == ":" else cell for ch in text) * sc / 2
            for ch in text:
                adv = colon_w if ch == ":" else cell
                g = QPainterPath()
                g.addText(0, 0, font, ch)
                off = (adv - fm.horizontalAdvance(ch)) / 2         # cifra centrata nella sua casella
                path.addPath(QTransform().translate(x + off * sc, base).scale(sc, sc).map(g))
                x += adv * sc
        if edit:                                          # sottolineatura sotto la cifra in modifica
            text, sc, base = lines[0]
            x0 = -sum(colon_w if ch == ":" else cell for ch in text) * sc / 2
            ci = (0, 1, 3, 4)[self._edit_pos]
            xs = x0 + sum(colon_w if ch == ":" else cell for ch in text[:ci]) * sc
            ul = QPainterPath()
            ul.addRect(xs + cell * sc * 0.08, base + 0.12 * h1, cell * sc * 0.84, 0.07 * h1)
            path.addPath(ul)
        p.save()
        p.scale(k, k)
        pen = QPen(self.numbers_color, 2 * DIGITAL_OUTLINE * s1, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.strokePath(path, pen)                            # bordo (tutto il testo)...
        p.fillPath(path, QBrush(self.hand_color))          # ...poi il riempimento sopra
        p.restore()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        k = self._cat_w / self.img_w
        p.translate(self._off[0] + self._jit, self._off[1])

        # camminata: tutto il gatto (coda e lancette comprese) si inclina attorno al piede
        # del lato verso cui pende; a inclinazione zero non cambia nulla, quindi il cambio
        # di piede e' continuo
        if self._hop:
            p.translate(0, -self._hop * k)                  # saltello verticale del passo
        if self._wobble:
            fx, fy = FOOT_R if self._wobble > 0 else FOOT_L
            p.translate(0, -self._lift(self._wobble) * k)   # da sdraiato non affonda nel pavimento
            p.translate(fx * k, fy * k)
            p.rotate(self._wobble)
            p.translate(-fx * k, -fy * k)

        # coda: disegnata per prima (dietro al corpo). Il suo angolo rispetto al corpo (psi) viene
        # dalla dinamica (vedi _tail_update); la curvatura segue psi: a un estremo coincide con la
        # PNG, all'estremo opposto con la PNG riflessa (m = -psi/ampiezza, da -1 a 1)
        if self._tp_drag and self._walk_start is not None:
            # il gatto cammina e si inclina: il mouse va riportato nel riferimento del gatto
            inv, ok = p.worldTransform().inverted()
            if ok:
                pt = inv.map(QPointF(self.mapFromGlobal(self._tp_gpos)))
                self._pull_target(pt.x() / k, pt.y() / k)
        psi = self._tail_update()
        m0 = 0.0 if self.tail_on else 1.0
        m = max(-1.0, min(1.0, m0 - psi / TAIL_ANGLE))
        if self._tail_push > 0:                   # fa da puntello: si incurva verso il pavimento
            m = m * (1 - self._tail_push) + self._fall_sign * TAIL_PUSH_CURL * self._tail_push
        s = 1.0
        pull = self._pull_pose()                  # coda tirata col mouse (o che rimbalza dopo il rilascio)
        if pull is None:
            self._tail_pose = (psi, m)            # serve a capire se un clic afferra la coda
        else:
            w, pp, pm, ps = pull
            psi += w * ((pp - psi + 180.0) % 360.0 - 180.0)
            m += w * (pm - m)
            s = max(0.9, 1.0 + w * (ps - 1.0))
        self._draw_tail(p, k, psi, m, s)

        p.drawPixmap(0, 0, self._scaled_face())
        now = datetime.now()
        sec = now.second + now.microsecond / 1e6
        minute = now.minute + now.second / 60.0
        hour = (now.hour % 12) + minute / 60.0

        p.translate(CENTER[0] * k, CENTER[1] * k)

        if self._alarm_edit is not None:          # impostazione della sveglia: cifre al posto delle lancette
            self._draw_digital(p, k, now, edit=True)
            p.end()
            return

        if self.digital_on:                       # orologio a numeri: niente lancette
            self._draw_digital(p, k, now)
            p.end()
            return

        def hand(angle, length, width, color, tail=0.0):
            p.save()
            p.rotate(angle)
            pen = QPen(color, width * k, Qt.SolidLine, Qt.RoundCap)
            p.setPen(pen)
            half = width / 2
            p.drawLine(QPointF(0, tail * k), QPointF(0, -(length - half) * k))
            p.restore()

        hand(hour * 30, HOUR_LEN, HAND_W, self.hand_color)
        hand(minute * 6, MIN_LEN, HAND_W, self.hand_color)
        if self.show_seconds:
            # lancetta dei secondi: movimento a scatti come un orologio al quarzo
            hand(int(sec) * 6, SEC_LEN, 2.4, SEC_COLOR, tail=22)

        # perno centrale
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(HUB_RING))
        p.drawEllipse(QPointF(0, 0), HUB_R * k, HUB_R * k)
        p.setBrush(QBrush(self.hand_color))
        p.drawEllipse(QPointF(0, 0), (HUB_R - 2.4) * k, (HUB_R - 2.4) * k)
        p.end()

    # -------------------------------------------------------------- mouse --
    def mousePressEvent(self, e):
        if self._alarm_edit is not None:         # un clic conferma l'ora scritta
            self._alarm_edit_commit()
            if e.button() == Qt.LeftButton:
                return
        if (self._walk_start is not None and self._walk_phase != "walk"
                and e.button() == Qt.LeftButton):
            return                       # e' a terra: non si lascia interrompere
        self._rest_geo = None
        self._end_shake()
        was_walking = self._walk_start is not None or self._pull_busy()
        self._end_walk()                 # qualsiasi clic ferma la camminata
        self._head_click = None
        self._dial_click = None
        if (e.button() == Qt.LeftButton and not was_walking
                and not (e.modifiers() & Qt.ControlModifier)
                and not self._on_head(e.pos()) and self._on_dial(e.pos())):
            self._dial_click = e.globalPos()      # clic sul quadrante: imposta la sveglia
        if e.button() == Qt.LeftButton and not self.locked:
            if (e.modifiers() & Qt.ControlModifier) and self.leader is None:
                self._resize_start = (e.globalPos(), self._cat_w)
            else:
                grab = None if was_walking else self._tail_hit(e.pos())
                if grab is not None:
                    self._pull_begin(e.globalPos(), grab)     # afferra la coda: la finestra non si trascina
                else:
                    self._drag_off = e.globalPos() - self.frameGeometry().topLeft()
                    if not was_walking and self._on_head(e.pos()):
                        self._head_click = e.globalPos()
        elif e.button() == Qt.RightButton:
            self.window_menu.popup(e.globalPos())

    def mouseMoveEvent(self, e):
        if self.locked:
            return
        if self._resize_start and e.buttons() & Qt.LeftButton:
            start, w0 = self._resize_start
            d = e.globalPos() - start
            self.set_width(w0 + (d.x() + d.y() * 0.5) )
        elif self._tp_drag and e.buttons() & Qt.LeftButton:
            self._pull_move(e.globalPos())
        elif self._drag_off is not None and e.buttons() & Qt.LeftButton:
            target = e.globalPos() - self._drag_off
            if target != self.pos():
                self._moved = True
            self.move(target)

    def mouseReleaseEvent(self, e):
        self._pull_release()
        if self._drag_off is not None or self._resize_start is not None:
            self._schedule_save()
        if (self._head_click is not None and e.button() == Qt.LeftButton
                and (e.globalPos() - self._head_click).manhattanLength() <= CLICK_TOL):
            self._start_walk()
        if (self._dial_click is not None and e.button() == Qt.LeftButton
                and (e.globalPos() - self._dial_click).manhattanLength() <= CLICK_TOL
                and self._alarm_edit is None):
            self.start_alarm_edit()
        if self.leader is not None and self._moved and self._drag_off is not None:
            self.leader.set_matrioska_offset(self)    # lasciato li': e' la nuova posizione standard
        self._moved = False
        self._dial_click = None
        self._head_click = None
        self._drag_off = None
        self._resize_start = None

    def wheelEvent(self, e):
        if self._alarm_edit is not None:         # rotella: cambia ore o minuti (secondo la cifra selezionata)
            self._edit_t0 = time.monotonic()
            steps = e.angleDelta().y()
            if steps:
                self._alarm_edit_adjust(1 if steps > 0 else -1)
            return
        if self.leader is not None:              # la dimensione del piccolo dipende da quella del grande
            return
        self._end_shake()
        self._end_walk()
        self._rest_geo = None
        if self.locked:
            return
        steps = e.angleDelta().y() / 120.0
        self.set_width(self._cat_w * (1 + WHEEL_STEP * steps), keep_center=True)
        self._schedule_save()

    def keyPressEvent(self, e):
        if self._alarm_edit is not None:
            self._alarm_edit_key(e)
            return
        if e.key() == Qt.Key_Escape:
            (self.leader if self.leader is not None else self).hide()


class MiniCat(QWidget):
    """Gatto in miniatura che sta in piedi sulla barra delle applicazioni mentre l'orologio e'
    nascosto. Parte dall'icona nella tray e cammina lungo la barra, ondeggiando; ai bordi dello
    schermo barcolla, cade, si scuote, si rialza e riparte in senso opposto.
    La coda e' animata come quella del gatto grande: pendolo mentre cammina, si alza e freme in
    segno di panico quando cade, fa da puntello quando tenta di rialzarsi. La dinamica e' quella
    di CatClock (stesse funzioni, prese in prestito), con lo stato proprio della miniatura.
    Clic sinistro = riapre l'orologio, destro = menu."""

    # funzioni della coda prese da CatClock (lavorano solo su angoli e tempi, non sulle dimensioni)
    _ss = staticmethod(CatClock._ss)
    _tail_phase = CatClock._tail_phase
    _tail_rest = CatClock._tail_rest
    _tail_update = CatClock._tail_update
    _draw_tail = CatClock._draw_tail

    # nomi che quelle funzioni si aspettano, collegati allo stato della miniatura
    _walk_phase = property(lambda s: s._phase)
    _phase_t0 = property(lambda s: s._t0)
    _walk_start = property(lambda s: 0.0)              # per la miniatura la "camminata" e' sempre attiva
    tail_on = property(lambda s: s.clock.tail_on)
    _tail_geo = property(lambda s: s.clock._tail_geo)
    _tail_reach = property(lambda s: s.clock._tail_reach)

    def _lift(self, angle):
        return self.clock._lift(angle)

    def _scaled_tail(self):
        dpr = self.devicePixelRatioF()
        key = (dpr, self.clock.tail_color.rgb())
        if self._tail_cache is None or self._tail_cache[0] != key:
            pm = CatClock._tinted(self.clock.tail, QBrush(self.clock.tail_color), QSize(self._cat_w, self._cat_h) * dpr)
            pm.setDevicePixelRatio(dpr)
            self._tail_cache = (key, pm)
        return self._tail_cache[1]

    def __init__(self, clock, leader=None, width=MINI_WIDTH):
        super().__init__()
        self.clock = clock
        self._leader = leader            # None = miniatura principale; altrimenti e' la piu' piccola che la segue
        self._child = None
        self._w = width
        self._env = 0.0                  # intensita' della camminata (solo per la miniatura che segue)
        self._wph = 0.0                  # fase del dondolio (accumulata: cambia ritmo senza salti)
        self._sp_mode = "lag"            # "lag" = resta indietro, "sprint" = scatta per raggiungere la principale
        self._sp_t0 = 0.0
        self._lag_dist = MINI_LAG_DIST[0]
        self._lag_time = MINI_LAG_TIME[0]
        self._cat_w = width
        self._cat_h = round(width * clock.ratio)
        self._tail_t = time.monotonic()
        self._tail_init = False
        self._tail_omega = self._tail_omega_v = 0.0
        self._tail_w_prev = self._tail_wv = 0.0
        self._tail_push = 0.0
        self._tail_cache = None
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setWindowTitle("Cat Clock mini")
        self.setCursor(Qt.PointingHandCursor)
        self._k = width / clock.img_w
        k = self._k
        # la finestra e' una "scena" piu' larga del gatto (serve per l'inclinazione e la caduta) e
        # arriva solo fino alla linea dei piedi: il resto (la coda che pende) resta "dietro" la barra
        ovs = [clock._overflow(s, 90.0) for s in (1, -1)]       # sinistra, alto, destra, basso
        extra = math.ceil(40 * k)        # la coda alzata sporge un po' dall'immagine
        self._pad_x = math.ceil(max(max(o[0], o[2]) for o in ovs) * k) + 2 + extra
        self._pad_top = math.ceil(max(o[1] for o in ovs) * k) + math.ceil(WALK_HOP * k) + 2 + extra
        feet = math.ceil(max(FOOT_L[1], FOOT_R[1]) * k) + 1
        self.setFixedSize(width + 2 * self._pad_x, self._pad_top + feet)
        self._geo = QRect()
        self._xmin, self._xmax, self._win_y = 0, 0, 0
        self._x = 0.0                    # angolo sinistro del gatto sullo schermo (px, con decimali)
        self._dir = -1                   # verso della camminata: -1 sinistra, +1 destra
        self._phase = "walk"             # walk -> stagger -> fall -> lie -> wiggle -> up -> walk
        self._t0 = self._last = 0.0
        self._wobble = self._hop = self._jit = 0.0
        self._fall_sign = 1
        self._fall_a0 = 0.0
        self._lie_dur = 0.0
        self._speed = 0.0
        self._off_x = float(self._pad_x)
        self._timer = QTimer(self)       # fotogrammi dell'animazione
        self._timer.timeout.connect(self._frame)
        self._slow = QTimer(self)        # riposizionamento e primo piano
        self._slow.timeout.connect(self._refresh)

    # -------------------------------------------------------------- posizione --
    def _screen_and_icon_x(self):
        """Schermo con la barra e ascissa dell'icona nella tray (o un'ascissa di ripiego)."""
        g = self.clock.tray.geometry() if self.clock.tray is not None else QRect()
        if g.isValid() and not g.isEmpty():
            c = g.center()
            scr = QApplication.screenAt(c) or QApplication.primaryScreen()
            return scr.geometry(), c.x()
        geo = QApplication.primaryScreen().geometry()
        return geo, geo.right() - MINI_FALLBACK_RIGHT

    def _set_geo(self, geo):
        self._geo = geo
        self._xmin = geo.left()
        self._xmax = geo.right() + 1 - self._w
        self._win_y = geo.bottom() + 1 - TASKBAR_HEIGHT - self.height()   # piedi sul bordo superiore della barra

    def _apply(self):
        """Mette la finestra attorno al gatto (sempre dentro lo schermo) e lo ridisegna."""
        wx = math.floor(self._x) - self._pad_x
        wx = max(self._geo.left(), min(wx, self._geo.right() + 1 - self.width()))
        self._off_x = self._x - wx       # dove sta il gatto dentro la finestra (con decimali: moto fluido)
        if self.pos() != QPoint(wx, self._win_y):
            self.move(wx, self._win_y)
        self.update()

    # ------------------------------------------------------------ vita/morte --
    def show_mini(self):
        geo, cx = self._screen_and_icon_x()
        self._set_geo(geo)
        self._x = float(max(self._xmin, min(cx - self._w // 2, self._xmax)))
        self._dir = -1 if self._x > geo.center().x() else 1       # verso la parte di barra piu' lunga
        self._tail_init = False                                   # la coda riparte da ferma
        self._begin_walk(time.monotonic())
        self._apply()
        self.show()
        self.raise_()
        self._slow.start(MINI_REFRESH_MS)
        self._timer.start(16)                                     # anche da fermo: serve alla coda
        self.update_child()

    def hide_mini(self):
        self._kill_child()
        self._timer.stop()
        self._slow.stop()
        self.hide()

    # ---------------------------------------------------- miniatura matrioska --
    def update_child(self):
        """Con la matrioska attiva accanto alla miniatura ce n'e' un'altra, piu' piccola, che la segue."""
        if self._leader is not None:
            return
        if self.clock.matrioska_on and self.isVisible():
            if self._child is None:
                w = max(8, round(self._w * MATRIOSKA_SCALE))
                self._child = MiniCat(self.clock, leader=self, width=w)
                self._child.show_near()
        else:
            self._kill_child()

    def _kill_child(self):
        c, self._child = self._child, None
        if c is not None:
            c.hide_mini()
            c.deleteLater()

    def show_near(self):
        """(Solo la piccola) compare subito accanto alla miniatura principale."""
        self._set_geo(self._leader._geo)
        self._x = self._behind_x()
        self._env = 0.0
        self._sp_mode, self._sp_t0 = "lag", time.monotonic()
        self._lag_dist = random.uniform(*MINI_LAG_DIST)
        self._lag_time = random.uniform(*MINI_LAG_TIME)
        self._tail_init = False
        self._begin_walk(time.monotonic())
        self._apply()
        self.show()
        self.raise_()
        self._slow.start(MINI_REFRESH_MS)
        self._timer.start(16)

    def _behind_x(self):
        """Ascissa dietro la principale: dal lato opposto a quello in cui sta camminando."""
        L = self._leader
        gap = round(L._w * MATRIOSKA_GAP) + 2
        x = L._x - gap - self._w if L._dir > 0 else L._x + L._w + gap
        return max(float(self._xmin), min(x, float(self._xmax)))

    def _follow_x(self):
        """(ascissa, modo) per la piccola. Sta sempre DIETRO la principale (mai davanti):
        - "aside": la principale e' caduta, la piccola resta dal suo lato e si scansiona oltre
          il suo ingombro da sdraiata;
        - "hold": la principale ha appena cambiato verso (dopo una caduta) e la piccola e' davanti
          a lei: resta ferma e la lascia passare;
        - "behind": segue dietro (ascissa = subito dietro la principale)."""
        L = self._leader
        gap = round(L._w * MATRIOSKA_GAP) + 2
        if L._phase != "walk":
            right = self._x + self._w / 2 > L._x + L._w / 2
            if L._fall_sign == (1 if right else -1):
                gap += round(L._w * 1.3)
            x = L._x + L._w + gap if right else L._x - gap - self._w
            return max(float(self._xmin), min(x, float(self._xmax))), "aside"
        tol = 3.0                                              # tolleranza: durante l'inseguimento puo' sfiorarla
        if L._dir > 0:
            passed = self._x + self._w <= L._x + tol          # la piccola e' tutta alle spalle della principale
        else:
            passed = self._x >= L._x + L._w - tol
        if not passed:
            return max(float(self._xmin), min(self._x, float(self._xmax))), "hold"
        return self._behind_x(), "behind"

    def _follow_frame(self, now, dt):
        """Camminata della piccola, con lo stesso dondolio e gli stessi saltelli della principale.
        Resta indietro: dopo lo scatto continua a camminare ma piu' piano della principale (MINI_DRIFT),
        cosi' si stacca gradualmente fino a _lag_dist px, poi la segue a quella distanza; dopo
        _lag_time secondi SCATTA (passi rapidi, velocita' multipla) fino a raggiungerla, e ricomincia
        da capo con una nuova distanza."""
        L = self._leader
        w = self._w
        x, mode = self._follow_x()
        stop, start = MATRIOSKA_STOP * w, MATRIOSKA_START * w
        sprint = drift = False
        tx = x
        dirs = 1 if L._dir > 0 else -1
        if mode == "behind":
            dist = (L._x - (self._x + w)) if dirs > 0 else (self._x - (L._x + L._w))   # px dietro la principale
            if self._sp_mode == "sprint":
                sprint = True
                if abs(x - self._x) <= 2 * stop:                  # raggiunta
                    self._sp_mode, self._sp_t0, sprint = "lag", now, False
                    self._lag_dist = random.uniform(*MINI_LAG_DIST)
                    self._lag_time = random.uniform(*MINI_LAG_TIME)
            elif now - self._sp_t0 > self._lag_time:
                self._sp_mode, sprint = "sprint", True
            if not sprint:
                lag_x = L._x - self._lag_dist - w if dirs > 0 else L._x + L._w + self._lag_dist
                lag_x = max(float(self._xmin), min(lag_x, float(self._xmax)))
                if dist >= self._lag_dist:
                    tx = lag_x
                else:
                    drift = True                                    # non e' ancora abbastanza indietro: si stacca piano
        else:
            self._sp_mode, self._sp_t0 = "lag", now
        d = float(dirs) if drift else tx - self._x
        ad = abs(d)
        if drift:
            fi = 1.0                                                # continua a camminare, senza fermarsi
        elif sprint:
            fi = min(1.0, ad / max(1.0, start))
        else:
            fi = max(0.0, min(1.0, (ad - stop) / max(1.0, start - stop)))
        self._env += (fi - self._env) * min(1.0, dt * (10.0 if sprint else 6.0))
        self._wph += dt * 2 * math.pi / (WOBBLE_PERIOD * (MINI_SPRINT_STRIDE if sprint else 1.0))
        self._wobble = WOBBLE_ANGLE * (1.2 if sprint else 1.0) * self._env * math.sin(self._wph)
        self._hop = 0.0 if self._wph < math.pi / 2 else WALK_HOP * self._env * abs(math.cos(self._wph))
        if sprint:
            base = MINI_SPRINT_SPEED * MINI_WALK_SPEED          # scatto a velocita' fissa e ben visibile
        elif drift:
            base = MINI_DRIFT * MINI_WALK_SPEED
        else:
            base = min(MINI_WALK_SPEED + MINI_FOLLOW_GAIN * ad, 3.0 * MINI_WALK_SPEED)
        self._speed = base * self._env * (0.55 + 0.45 * abs(math.cos(self._wph)))
        adv = self._speed * dt if drift else min(self._speed * dt, ad)
        self._x = max(float(self._xmin), min(self._x + math.copysign(adv, d), float(self._xmax)))

    def _refresh(self):
        geo = self._leader._geo if self._leader is not None else self._screen_and_icon_x()[0]
        self._set_geo(geo)
        self._x = max(float(self._xmin), min(self._x, float(self._xmax)))
        self._apply()
        self.raise_()

    # -------------------------------------------------------------- animazione --
    def _begin_walk(self, now):
        self._phase = "walk"
        self._t0 = self._last = now
        self._wobble = self._hop = self._jit = 0.0

    def _frame(self):
        now = time.monotonic()
        dt = min(now - self._last, 0.1)
        self._last = now
        self._jit = 0.0
        if self._leader is not None:                              # la piccola insegue la principale
            self._follow_frame(now, dt)
            self._apply()
            return
        if MINI_WALK_SPEED <= 0:                                  # fermo sull'icona: si anima solo la coda
            self.update()
            return
        if self._phase == "walk":
            self._walk_frame(now, dt)
        else:
            self._hop *= 0.75
            if self._hop < 0.05:
                self._hop = 0.0
            self._fallen_frame(now, dt)
        self._apply()

    def _walk_frame(self, now, dt):
        t = now - self._t0
        env = min(1.0, t / WALK_RAMP)
        env = env * env * (3 - 2 * env)                         # avvio morbido
        phase = 2 * math.pi * t / WOBBLE_PERIOD
        self._wobble = WOBBLE_ANGLE * env * math.sin(phase)
        self._hop = 0.0 if phase < math.pi / 2 else WALK_HOP * env * abs(math.cos(phase))
        # avanza un po' di piu' quando il peso passa da un piede all'altro
        self._speed = MINI_WALK_SPEED * env * (0.55 + 0.45 * abs(math.cos(phase)))
        x = self._x + self._dir * self._speed * dt
        hit = False
        if x < self._xmin:
            x, self._dir, hit = float(self._xmin), 1, True
        elif x > self._xmax:
            x, self._dir, hit = float(self._xmax), -1, True
        self._x = x
        if hit and t > WALK_RAMP:        # ora self._dir punta lontano dal bordo: cade da quella parte
            self._fall_sign = self._dir
            self._fall_a0 = self._wobble
            self._lie_dur = random.uniform(*LIE_TIME)
            self._phase, self._t0 = "stagger", now

    def _recoil(self, dt, f):
        """Dopo l'urto scivola un po' indietro, sempre meno."""
        x = self._x + self._dir * self._speed * f * dt
        self._x = max(float(self._xmin), min(x, float(self._xmax)))

    def _fallen_frame(self, now, dt):
        el = now - self._t0
        sg = self._fall_sign
        target = 90.0 * sg
        ph = self._phase
        if ph == "stagger":                           # barcolla, con l'ultima oscillazione verso il lato della caduta
            u = min(1.0, el / STAGGER_TIME)
            s0 = min(1.0, u * 5.0)
            s0 = s0 * s0 * (3 - 2 * s0)
            amp = STAGGER_AMP * (0.5 + 0.5 * u)
            self._wobble = self._fall_a0 * (1 - s0) - sg * amp * math.sin(2 * math.pi * STAGGER_SWINGS * u)
            self._recoil(dt, (1 - u) ** 2)
            if u >= 1.0:
                self._phase, self._t0 = "fall", now
                self._fall_a0 = self._wobble
        elif ph == "fall":
            if el < FALL_TIME:                        # cade accelerando
                u = el / FALL_TIME
                self._wobble = self._fall_a0 + (target - self._fall_a0) * u * u
            elif el < FALL_TIME + BOUNCE_TIME:        # rimbalzino all'impatto
                b = (el - FALL_TIME) / BOUNCE_TIME
                self._wobble = target - sg * FALL_REBOUND * math.sin(math.pi * b) * (1 - b)
            else:
                self._wobble = target
                self._phase, self._t0 = "lie", now
        elif ph == "lie":
            self._wobble = target
            if el >= self._lie_dur:
                self._phase, self._t0 = "wiggle", now
        elif ph == "wiggle":                          # si scuote un po' prima di rialzarsi
            u = el / WIGGLE_TIME
            if u >= 1.0:
                self._wobble = target
                self._phase, self._t0 = "up", now
            else:
                env = math.sin(math.pi * u)
                rock = WIGGLE_ROCK * (0.5 + 0.5 * math.sin(2 * math.pi * WIGGLE_FREQ * el)) * env
                self._wobble = target - sg * rock
                self._jit = max(0.5, self._w * WIGGLE_JITTER) * env * math.sin(2 * math.pi * WIGGLE_FREQ * 3.1 * el)
        else:                                         # "up": un tentativo fallito, poi si rialza
            u = el / UP_TIME
            if u >= 1.0:
                self._begin_walk(now)                 # riparte: self._dir e' gia' il verso opposto al bordo
                return
            if u < 0.45:
                mag = 90.0 - 28.0 * math.sin(math.pi * u / 0.45)
            else:
                v = (u - 0.45) / 0.55
                mag = 90.0 * (1 - v * v * (3 - 2 * v))
            self._wobble = sg * mag

    # ------------------------------------------------------------------ disegno --
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        k = self._k
        p.translate(self._off_x + self._jit, self._pad_top)
        if self._hop:
            p.translate(0, -self._hop * k)                       # saltello del passo
        if self._wobble:
            fx, fy = FOOT_R if self._wobble > 0 else FOOT_L
            p.translate(0, -self.clock._lift(self._wobble) * k)  # da sdraiato non affonda nella barra
            p.translate(fx * k, fy * k)
            p.rotate(self._wobble)
            p.translate(-fx * k, -fy * k)
        # coda: disegnata per prima (dietro al corpo), come nel gatto grande
        psi = self._tail_update()
        m = max(-1.0, min(1.0, (0.0 if self.tail_on else 1.0) - psi / TAIL_ANGLE))
        if self._tail_push > 0:                       # fa da puntello: si incurva verso il pavimento
            m = m * (1 - self._tail_push) + self._fall_sign * TAIL_PUSH_CURL * self._tail_push
        self._draw_tail(p, k, psi, m)
        p.drawPixmap(0, 0, self.clock.mini_face(self._w, self.devicePixelRatioF()))
        p.end()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clock.show()
            self.clock.raise_()
        elif e.button() == Qt.RightButton:
            self.clock.window_menu.popup(e.globalPos())


# ------------------------------------------------------------------- tray --
def build_tray(app, clock: CatClock):
    icon_pm = clock.face.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    icon = QIcon(icon_pm)
    app.setWindowIcon(icon)

    tray = QSystemTrayIcon(icon, app)
    tray.setToolTip("Cat Clock")
    clock.tray = tray                   # la miniatura si posiziona sull'icona

    menu = QMenu()
    act_toggle = QAction("Mostra / Nascondi orologio", menu)
    act_sec = QAction("Lancetta dei secondi", menu, checkable=True)
    act_digital = QAction("Orologio a numeri", menu, checkable=True)
    act_digital.setChecked(clock.digital_on)
    act_tail = QAction("Coda oscillante", menu, checkable=True)
    act_tail.setChecked(clock.tail_on)
    act_shake = QAction("Trema ogni minuto", menu, checkable=True)
    act_shake.setChecked(clock.shake_on)
    act_lock = QAction("Blocca posizione e dimensione", menu, checkable=True)
    act_mini = QAction("Gatto in miniatura sulla barra", menu, checkable=True)
    act_mini.setChecked(clock.mini_on)
    act_matr = QAction("Matrioska (gatto piccolo che ti segue)", menu, checkable=True)
    act_matr.setChecked(clock.matrioska_on)
    act_alarm = QAction("Sveglia attiva", menu, checkable=True)
    act_alarm.setChecked(clock.alarm_on)
    act_alarm_set = QAction("Imposta ora sveglia", menu)
    act_alarm_test = QAction("Prova la sveglia", menu)
    act_sec.setChecked(clock.show_seconds)
    act_lock.setChecked(clock.locked)

    size_menu = menu.addMenu("Dimensione")
    for label, w in (("Piccolo (150)", 150), ("Medio (300)", 300),
                     ("Grande (450)", 450), ("Molto grande (700)", 700)):
        a = size_menu.addAction(label)
        a.triggered.connect(lambda _=False, w=w: (clock.set_width(w, keep_center=True),
                                                  clock.save_settings()))
    act_size = QAction("Imposta dimensione...", menu)
    act_color = QAction("Colore del gatto...", menu)
    act_hands = QAction("Colore delle lancette...", menu)
    act_dial = QAction("Colore sfondo quadrante...", menu)
    act_numbers = QAction("Colore numeri delle ore...", menu)
    act_tailcol = QAction("Colore della coda...", menu)
    act_color_reset = QAction("Ripristina colori originali", menu)
    grad_menu = QMenu("Sfumatura a 45\u00b0", menu)
    act_grad = QAction("Usa sfumatura", grad_menu, checkable=True)
    act_grad.setChecked(clock.gradient_on)
    act_c1 = QAction("Colore iniziale (alto-sinistra)...", grad_menu)
    act_c2 = QAction("Colore finale (basso-destra)...", grad_menu)
    act_swap = QAction("Inverti i due colori", grad_menu)
    for a_ in (act_grad, act_c1, act_c2, act_swap):
        grad_menu.addAction(a_)
    act_reset = QAction("Ripristina posizione e dimensione", menu)
    act_quit = QAction("Esci", menu)

    def ask_size():
        dlg = QInputDialog()
        dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowStaysOnTopHint)
        dlg.setWindowTitle("Cat Clock")
        dlg.setInputMode(QInputDialog.IntInput)
        dlg.setLabelText(f"Larghezza in pixel ({MIN_WIDTH}-{MAX_WIDTH}).\n"
                         "L'altezza si adatta da sola:")
        dlg.setIntRange(MIN_WIDTH, MAX_WIDTH)
        dlg.setIntValue(clock._cat_w)
        dlg.setIntStep(10)
        dlg.setOkButtonText("Applica")
        dlg.setCancelButtonText("Annulla")
        if dlg.exec_() == QInputDialog.Accepted:
            clock.set_width(dlg.intValue(), keep_center=True)
            clock.save_settings()

    def refresh_alarm_label():
        act_alarm.setText(f"Sveglia attiva ({clock.alarm_time.toString('HH:mm')})")

    def pick(title, current, apply):
        original = QColor(current)
        dlg = QColorDialog(original)
        dlg.setWindowTitle(title)
        dlg.setOption(QColorDialog.DontUseNativeDialog, True)   # include il contagocce da schermo
        dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowStaysOnTopHint)
        dlg.currentColorChanged.connect(lambda c: apply(c, False))   # anteprima dal vivo
        if dlg.exec_() == QColorDialog.Accepted:
            apply(dlg.selectedColor(), True)
        else:
            apply(original, False)

    def pick_color():
        pick("Colore del gatto" if not clock.gradient_on else "Sfumatura: colore iniziale",
             clock.body_color, lambda c, sv: clock.set_colors(c1=c, save=sv))

    def pick_color2():
        clock.set_colors(gradient=True, save=False)
        act_grad.setChecked(True)
        pick("Sfumatura: colore finale", clock.body_color2,
             lambda c, sv: clock.set_colors(c2=c, save=sv))

    def pick_hands():
        pick("Colore delle lancette", clock.hand_color,
             lambda c, sv: clock.set_hand_color(c, save=sv))

    def pick_dial():
        pick("Colore dello sfondo del quadrante", clock.dial_color,
             lambda c, sv: clock.set_dial_color(c, save=sv))

    def pick_numbers():
        pick("Colore dei numeri delle ore", clock.numbers_color,
             lambda c, sv: clock.set_numbers_color(c, save=sv))

    def pick_tail():
        pick("Colore della coda", clock.tail_color,
             lambda c, sv: clock.set_tail_color(c, save=sv))

    def set_gradient(v):
        clock.set_colors(gradient=v)

    def swap_colors():
        clock.set_colors(c1=clock.body_color2, c2=clock.body_color)

    def reset_colors():
        act_grad.setChecked(False)
        clock.set_colors(c1=QColor(DEFAULT_BODY_COLOR), c2=QColor(DEFAULT_COLOR2), gradient=False)
        clock.set_hand_color(HAND_COLOR)
        clock.set_dial_color(QColor(DEFAULT_DIAL_COLOR))
        clock.set_numbers_color(QColor(DEFAULT_NUMBERS_COLOR))
        clock.set_tail_color(QColor(DEFAULT_TAIL_COLOR))

    def toggle():
        if clock.isVisible():
            clock.hide()
        else:
            clock.show()
            clock.raise_()

    def set_seconds(v):
        clock.show_seconds = v
        clock.save_settings()
        clock.update()

    def set_lock(v):
        clock.locked = v
        clock.save_settings()

    def quit_app():
        clock.save_settings()
        tray.hide()
        app.quit()

    act_toggle.triggered.connect(toggle)
    act_sec.toggled.connect(set_seconds)
    act_digital.toggled.connect(clock.set_digital)
    act_tail.toggled.connect(clock.set_tail)
    act_shake.toggled.connect(clock.set_shake)
    act_lock.toggled.connect(set_lock)
    act_mini.toggled.connect(clock.set_mini)
    act_matr.toggled.connect(clock.set_matrioska)
    act_alarm.toggled.connect(clock.set_alarm_on)
    act_alarm_set.triggered.connect(clock.start_alarm_edit)
    clock.alarm_changed_cb = lambda: (act_alarm.setChecked(True), refresh_alarm_label())
    act_alarm_test.triggered.connect(clock._start_alarm)
    act_size.triggered.connect(ask_size)
    act_color.triggered.connect(pick_color)
    act_hands.triggered.connect(pick_hands)
    act_dial.triggered.connect(pick_dial)
    act_numbers.triggered.connect(pick_numbers)
    act_tailcol.triggered.connect(pick_tail)
    act_color_reset.triggered.connect(reset_colors)
    act_grad.toggled.connect(set_gradient)
    act_c1.triggered.connect(pick_color)
    act_c2.triggered.connect(pick_color2)
    act_swap.triggered.connect(swap_colors)
    act_reset.triggered.connect(clock.reset_geometry)
    act_quit.triggered.connect(quit_app)

    for a in (act_toggle, act_sec, act_digital, act_tail, act_shake, act_lock, act_mini, act_matr,
              act_alarm, act_alarm_set, act_alarm_test):
        menu.insertAction(size_menu.menuAction(), a)
    refresh_alarm_label()
    menu.insertAction(size_menu.menuAction(), act_size)
    menu.insertAction(size_menu.menuAction(), act_color)
    menu.insertMenu(size_menu.menuAction(), grad_menu)
    menu.insertAction(size_menu.menuAction(), act_hands)
    menu.insertAction(size_menu.menuAction(), act_dial)
    menu.insertAction(size_menu.menuAction(), act_numbers)
    menu.insertAction(size_menu.menuAction(), act_tailcol)
    menu.insertAction(size_menu.menuAction(), act_color_reset)
    menu.addAction(act_reset)
    menu.addSeparator()
    menu.addAction(act_quit)

    # lo stesso menu serve anche per il tasto destro sull'orologio
    clock.window_menu = menu
    menu.aboutToShow.connect(clock._end_walk)   # dimensione/posizione tornano normali
    menu.aboutToShow.connect(refresh_alarm_label)
    tray.setContextMenu(menu)
    tray.activated.connect(lambda reason: toggle()
                           if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick) else None)
    tray.show()
    return tray


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    if not QSystemTrayIcon.isSystemTrayAvailable():
        QMessageBox.critical(None, "Cat Clock", "System tray non disponibile.")
        return 1

    settings = QSettings("CatClock", "CatClock")
    clock = CatClock(settings)
    tray = build_tray(app, clock)       # noqa: F841  (deve restare in vita)

    if not START_HIDDEN:
        clock.show()
    elif clock.mini_on:
        clock.mini.show_mini()
    if clock.matrioska_on:
        clock.set_matrioska(True, save=False)
    app.aboutToQuit.connect(clock.save_settings)
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
