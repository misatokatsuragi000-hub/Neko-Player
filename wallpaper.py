import json
import os
import select
import socket
import subprocess
import sys
import time
from urllib.parse import unquote, urlparse
import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gdk, GLib, Gtk, Pango

SOCKET_PATH = "/tmp/mpv_wallpaper.sock"
CONFIG_DIR = os.path.expanduser("~/.config/wallpaper_animato")
CONFIG_FILE = os.path.join(CONFIG_DIR, "settings.json")
PLAYLIST_FILE = "/tmp/wallpaper_playlist.m3u"
ICON_PATH = "/home/actarus/.icons/Windows-7/apps/fox.png"
# Canale con cui un secondo lancio (python3 wallpaper.py /percorso/video.mp4) passa il file all'istanza
# gia' in esecuzione. Socket "astratto" di Linux: sparisce da solo quando il programma termina.
CTL_ADDR = "\0wallpaper_animato_ctl"


def normalizza_percorso(arg):
  """Percorso assoluto da un argomento di lancio (accetta anche indirizzi file:///...)."""
  if arg.startswith("file://"):
    arg = unquote(urlparse(arg).path)
  return os.path.abspath(os.path.expanduser(arg))


def file_da_argv(argv):
  """Primo argomento che non e' un'opzione (--show, ...): il file video da usare."""
  for a in argv:
    if a and not a.startswith("--"):
      return normalizza_percorso(a)      # assoluto subito: l'istanza che lo riceve ha un'altra cartella corrente
  return ""


def invia_a_istanza_attiva(percorso):
  """True se un'altra istanza e' in esecuzione e ha ricevuto il file."""
  s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
  s.settimeout(1.0)
  try:
    s.connect(CTL_ADDR)
    s.sendall((percorso + "\n").encode("utf-8"))
    return True
  except OSError:
    return False
  finally:
    s.close()


class WallpaperApp(Gtk.Window):

  def __init__(self):
    super().__init__(title="Sfondo Animato")
    self.set_border_width(16)
    self.set_default_size(470, 780)
    self.set_position(Gtk.WindowPosition.CENTER)

    if os.path.exists(ICON_PATH):
      self.set_icon_from_file(ICON_PATH)

    self.is_initialized = False
    self.ipc_sock = None
    self.ctl_sock = None
    self.ipc_buffer = ""
    self.request_counter = 0
    self.is_running = False
    self.current_playing_file = ""
    self.current_playing_path = ""
    self.last_adapted_video = None

    self.config = self.load_config()
    self.selected_file = self.config.get("file", "")
    self.selected_folder = self.config.get("folder", "")

    vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
    self.add(vbox)

    # --- 1. Scelta Sorgente ---
    mode_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    mode_label = Gtk.Label(
        label="<b>Cosa vuoi usare come sfondo?</b>", xalign=0
    )
    mode_label.set_use_markup(True)
    mode_box.pack_start(mode_label, False, False, 0)

    radio_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=20)
    self.radio_single = Gtk.RadioButton.new_with_label_from_widget(
        None, "Un singolo file video"
    )
    self.radio_folder = Gtk.RadioButton.new_with_label_from_widget(
        self.radio_single, "Una cartella intera di video"
    )
    self.radio_single.connect("toggled", self.on_mode_changed)

    radio_box.pack_start(self.radio_single, False, False, 0)
    radio_box.pack_start(self.radio_folder, False, False, 0)
    mode_box.pack_start(radio_box, False, False, 0)

    # File Singolo
    self.file_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
    btn_file = Gtk.Button(label="📁 Scegli un file video dal PC...")
    btn_file.connect("clicked", self.choose_file_dialog)

    file_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
    self.lbl_file = Gtk.Label(
        label=self.selected_file
        if self.selected_file
        else "Nessun file selezionato",
        xalign=0,
    )
    self.lbl_file.set_ellipsize(Pango.EllipsizeMode.MIDDLE)
    self.lbl_file.get_style_context().add_class("dim-label")

    self.btn_open_file_loc = Gtk.Button(label="📂")
    self.btn_open_file_loc.set_tooltip_text(
        "Apri la posizione di questo file nell'esplora risorse"
    )
    self.btn_open_file_loc.set_sensitive(
        bool(self.selected_file and os.path.exists(self.selected_file))
    )
    self.btn_open_file_loc.connect(
        "clicked", lambda w: self.open_path_in_file_manager(self.selected_file)
    )

    file_row.pack_start(self.lbl_file, True, True, 0)
    file_row.pack_start(self.btn_open_file_loc, False, False, 0)

    self.file_box.pack_start(btn_file, False, False, 0)
    self.file_box.pack_start(file_row, False, False, 0)
    mode_box.pack_start(self.file_box, False, False, 0)

    # Cartella
    self.folder_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
    btn_folder = Gtk.Button(label="📂 Scegli la cartella dal PC...")
    btn_folder.connect("clicked", self.choose_folder_dialog)

    folder_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
    self.lbl_folder = Gtk.Label(
        label=self.selected_folder
        if self.selected_folder
        else "Nessuna cartella selezionata",
        xalign=0,
    )
    self.lbl_folder.set_ellipsize(Pango.EllipsizeMode.MIDDLE)
    self.lbl_folder.get_style_context().add_class("dim-label")

    self.btn_open_folder_loc = Gtk.Button(label="📂")
    self.btn_open_folder_loc.set_tooltip_text(
        "Apri la cartella nell'esplora risorse"
    )
    self.btn_open_folder_loc.set_sensitive(
        bool(self.selected_folder and os.path.exists(self.selected_folder))
    )
    self.btn_open_folder_loc.connect(
        "clicked",
        lambda w: self.open_path_in_file_manager(self.selected_folder),
    )

    folder_row.pack_start(self.lbl_folder, True, True, 0)
    folder_row.pack_start(self.btn_open_folder_loc, False, False, 0)

    self.check_recursive = Gtk.CheckButton(
        label="Cerca file anche nelle sottocartelle"
    )
    self.check_recursive.set_active(self.config.get("recursive", True))
    self.check_recursive.connect("toggled", lambda w: self.save_config())

    self.folder_box.pack_start(btn_folder, False, False, 0)
    self.folder_box.pack_start(folder_row, False, False, 0)
    self.folder_box.pack_start(self.check_recursive, False, False, 4)
    mode_box.pack_start(self.folder_box, False, False, 0)

    if self.config.get("mode") == "folder":
      self.radio_folder.set_active(True)
      self.file_box.hide()
      self.folder_box.show()
    else:
      self.radio_single.set_active(True)
      self.folder_box.hide()
      self.file_box.show()

    vbox.pack_start(mode_box, False, False, 0)

    vbox.pack_start(
        Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL), False, False, 2
    )

    # --- 2. In Riproduzione ---
    now_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
    now_title = Gtk.Label(label="<b>In riproduzione:</b>", xalign=0)
    now_title.set_use_markup(True)

    now_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
    self.lbl_now_playing = Gtk.Label(
        label="Nessun video in riproduzione", xalign=0
    )
    self.lbl_now_playing.set_ellipsize(Pango.EllipsizeMode.END)
    self.lbl_now_playing.get_style_context().add_class("dim-label")

    self.btn_open_now_playing = Gtk.Button(label="📂")
    self.btn_open_now_playing.set_tooltip_text(
        "Apri la posizione di questo file nell'esplora risorse"
    )
    self.btn_open_now_playing.set_sensitive(False)
    self.btn_open_now_playing.connect(
        "clicked",
        lambda w: self.open_path_in_file_manager(self.current_playing_path),
    )

    now_row.pack_start(self.lbl_now_playing, True, True, 0)
    now_row.pack_start(self.btn_open_now_playing, False, False, 0)

    now_box.pack_start(now_title, False, False, 0)
    now_box.pack_start(now_row, False, False, 0)
    vbox.pack_start(now_box, False, False, 0)

    # --- 3. Barra Pulsanti ---
    btn_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
    btn_box.set_homogeneous(True)

    self.btn_start = Gtk.Button(label="▶ Avvia")
    self.btn_start.get_style_context().add_class("suggested-action")
    self.btn_start.connect("clicked", lambda w: self.start_wallpaper())
    btn_box.pack_start(self.btn_start, True, True, 0)

    self.btn_pause = Gtk.Button(label="⏸ Pausa")
    self.btn_pause.set_sensitive(False)
    self.btn_pause.connect("clicked", self.toggle_pause)
    btn_box.pack_start(self.btn_pause, True, True, 0)

    self.btn_next = Gtk.Button(label="⏭ Prossimo")
    self.btn_next.set_sensitive(False)
    self.btn_next.connect("clicked", self.next_track)
    btn_box.pack_start(self.btn_next, True, True, 0)

    self.btn_stop = Gtk.Button(label="⏹ Ferma")
    self.btn_stop.get_style_context().add_class("destructive-action")
    self.btn_stop.set_sensitive(False)
    self.btn_stop.connect("clicked", self.stop_wallpaper)
    btn_box.pack_start(self.btn_stop, True, True, 0)

    vbox.pack_start(btn_box, False, False, 0)

    vbox.pack_start(
        Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL), False, False, 4
    )

    # --- 4. Barra del Volume ---
    vol_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
    vol_label = Gtk.Label(label="<b>Volume Audio:</b>", xalign=0)
    vol_label.set_use_markup(True)
    vol_box.pack_start(vol_label, False, False, 0)

    self.scale_volume = Gtk.Scale.new_with_range(
        Gtk.Orientation.HORIZONTAL, 0, 100, 1
    )
    initial_vol = self.config.get("volume", 70)
    if initial_vol == 0:
      initial_vol = 70
    self.scale_volume.set_value(initial_vol)
    self.scale_volume.add_mark(0, Gtk.PositionType.BOTTOM, "Muto")
    self.scale_volume.add_mark(50, Gtk.PositionType.BOTTOM, "50%")
    self.scale_volume.add_mark(100, Gtk.PositionType.BOTTOM, "100%")
    self.scale_volume.connect("value-changed", self.on_volume_changed)
    vol_box.pack_start(self.scale_volume, False, False, 0)

    vbox.pack_start(vol_box, False, False, 0)

    vbox.pack_start(
        Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL), False, False, 4
    )

    # --- 5. Inquadratura Video ---
    adj_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    adj_label = Gtk.Label(label="<b>Inquadratura Video:</b>", xalign=0)
    adj_label.set_use_markup(True)
    adj_box.pack_start(adj_label, False, False, 0)

    fit_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    lbl_fit = Gtk.Label(label="Adattamento Base:", xalign=0)
    self.combo_fit = Gtk.ComboBoxText()
    self.combo_fit.append("height", "📐 Adatta all'Altezza (100% Schermo)")
    self.combo_fit.append("width", "📐 Adatta alla Larghezza (100% Schermo)")
    self.combo_fit.append("fill", "🖼 Riempi Schermo (Crop)")
    self.combo_fit.append("fit", "🔲 Mostra Intero (Originale / Fit)")
    self.combo_fit.append("stretch", "↔ Stira a Tutto Schermo (Stretch)")

    self.combo_fit.set_active_id(self.config.get("fit_mode", "height"))
    self.combo_fit.connect("changed", self.on_fit_changed)
    fit_row.pack_start(lbl_fit, False, False, 0)
    fit_row.pack_start(self.combo_fit, True, True, 0)
    adj_box.pack_start(fit_row, False, False, 0)

    adj_box.pack_start(
        Gtk.Label(label="Scala / Zoom Aggiuntivo:", xalign=0), False, False, 0
    )
    self.scale_zoom = Gtk.Scale.new_with_range(
        Gtk.Orientation.HORIZONTAL, -1.5, 2.0, 0.05
    )
    self.scale_zoom.set_value(self.config.get("zoom", 0.0))
    self.scale_zoom.add_mark(0.0, Gtk.PositionType.BOTTOM, "Normale")
    self.scale_zoom.connect("value-changed", self.on_slider_changed)
    adj_box.pack_start(self.scale_zoom, False, False, 0)

    adj_box.pack_start(
        Gtk.Label(
            label="Offset Orizzontale (Sinistra / Destra):", xalign=0
        ),
        False,
        False,
        0,
    )
    self.scale_pan_x = Gtk.Scale.new_with_range(
        Gtk.Orientation.HORIZONTAL, -1.0, 1.0, 0.02
    )
    self.scale_pan_x.set_value(self.config.get("pan_x", 0.0))
    self.scale_pan_x.add_mark(0.0, Gtk.PositionType.BOTTOM, "Centro")
    self.scale_pan_x.connect("value-changed", self.on_slider_changed)
    adj_box.pack_start(self.scale_pan_x, False, False, 0)

    adj_box.pack_start(
        Gtk.Label(label="Offset Verticale (Alto / Basso):", xalign=0),
        False,
        False,
        0,
    )
    self.scale_pan_y = Gtk.Scale.new_with_range(
        Gtk.Orientation.HORIZONTAL, -1.0, 1.0, 0.02
    )
    self.scale_pan_y.set_value(self.config.get("pan_y", 0.0))
    self.scale_pan_y.add_mark(0.0, Gtk.PositionType.BOTTOM, "Centro")
    self.scale_pan_y.connect("value-changed", self.on_slider_changed)
    adj_box.pack_start(self.scale_pan_y, False, False, 0)

    btn_reset = Gtk.Button(label="↺ Ripristina Centro e Scala")
    btn_reset.connect("clicked", self.reset_all)
    adj_box.pack_start(btn_reset, False, False, 6)

    vbox.pack_start(adj_box, True, True, 0)

    # --- 6. Tray Icon ---
    if os.path.exists(ICON_PATH):
      self.tray_icon = Gtk.StatusIcon.new_from_file(ICON_PATH)
    else:
      self.tray_icon = Gtk.StatusIcon.new_from_icon_name(
          "preferences-desktop-wallpaper"
      )

    self.tray_icon.set_tooltip_text("Sfondo Animato")
    self.tray_icon.connect("activate", self.toggle_window_visibility)
    self.tray_icon.connect("popup-menu", self.on_tray_popup_menu)

    self.connect("delete-event", self.on_delete_event)

    GLib.timeout_add(1000, self.update_now_playing)

    self.is_initialized = True

    # Avvio automatico del playback (silenzioso se manca file/cartella)
    GLib.timeout_add(300, self.autostart_playback)

  def usa_file_esterno(self, percorso):
    """Imposta il file nel campo 'singolo file' e lo mette subito in play."""
    percorso = normalizza_percorso(percorso)
    if not os.path.isfile(percorso):
      print("Wallpaper: file non trovato: " + percorso, file=sys.stderr)
      return False
    self.selected_file = percorso
    self.lbl_file.set_text(percorso)
    self.btn_open_file_loc.set_sensitive(True)
    self.radio_single.set_active(True)       # modalita' "singolo file" (salva anche la configurazione)
    self.save_config()
    self.start_wallpaper(silent=False)
    return True

  def avvia_server_comandi(self):
    """Riceve da altre istanze (lanciate con un file come parametro) il file da riprodurre."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
      s.bind(CTL_ADDR)
      s.listen(4)
    except OSError:
      s.close()              # il canale e' gia' di un'altra istanza: questa non lo gestisce
      return
    s.setblocking(False)
    self.ctl_sock = s
    GLib.io_add_watch(
        s.fileno(), GLib.PRIORITY_DEFAULT, GLib.IO_IN, self.on_comando_remoto
    )

  def on_comando_remoto(self, fd, condition):
    try:
      conn, _ = self.ctl_sock.accept()
    except OSError:
      return True
    try:
      conn.settimeout(0.5)
      dati = conn.recv(65536).decode("utf-8", "replace")
    except OSError:
      dati = ""
    finally:
      conn.close()
    for riga in dati.split("\n"):
      riga = riga.rstrip("\r")
      if riga:
        self.usa_file_esterno(riga)
    return True

  def autostart_playback(self):
    if not self.is_running:
      self.start_wallpaper(silent=True)
    return False

  def apply_mode_visibility(self):
    """Mostra solo il box (file o cartella) coerente con la modalita' scelta."""
    if self.radio_folder.get_active():
      self.file_box.hide()
      self.folder_box.show()
    else:
      self.folder_box.hide()
      self.file_box.show()

  # --- GESTIONE APERTURA CARTELLA FILE MANAGER ---
  def open_path_in_file_manager(self, target_path):
    """Apre la cartella di sistema evidenziando il file, o apre la cartella indicata."""
    if not target_path:
      return

    target_path = os.path.abspath(os.path.expanduser(target_path))
    if not os.path.exists(target_path):
      return

    folder = (
        os.path.dirname(target_path)
        if os.path.isfile(target_path)
        else target_path
    )

    # Tenta di aprire evidenziando il file tramite protocollo standard D-Bus FileManager1
    if os.path.isfile(target_path):
      uri = f"file://{target_path}"
      try:
        res = subprocess.run(
            [
                "dbus-send",
                "--session",
                "--dest=org.freedesktop.FileManager1",
                "--type=method_call",
                "/org/freedesktop/FileManager1",
                "org.freedesktop.FileManager1.ShowItems",
                f"array:string:{uri}",
                'string:""',
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1,
        )
        if res.returncode == 0:
          return
      except Exception:
        pass

    # Fallback su xdg-open aprendo la cartella
    try:
      subprocess.Popen(
          ["xdg-open", folder],
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
      )
    except Exception:
      pass

  # --- ADATTAMENTO E OFFSET ---
  def get_screen_aspect_ratio(self):
    """Rileva le dimensioni reali dello schermo o della finestra del render."""
    win_w = self.query_ipc(["get_property", "osd-width"])
    win_h = self.query_ipc(["get_property", "osd-height"])

    if win_w and win_h and win_w > 100 and win_h > 100:
      return float(win_w) / float(win_h)

    display = Gdk.Display.get_default()
    if display:
      monitor = display.get_primary_monitor() or display.get_monitor(0)
      if monitor:
        geom = monitor.get_geometry()
        if geom.width > 0 and geom.height > 0:
          return float(geom.width) / float(geom.height)

    screen = Gdk.Screen.get_default()
    if screen and screen.get_height() > 0:
      return float(screen.get_width()) / float(screen.get_height())

    return 16.0 / 9.0

  def apply_adaptation(self):
    """Calcola e applica l'inquadratura al video corrente."""
    if not self.is_running:
      return False

    screen_aspect = self.get_screen_aspect_ratio()

    # 1. Recupera l'aspect ratio nativo e reale del video
    video_aspect = None
    vp = self.query_ipc(["get_property", "video-params"])
    if isinstance(vp, dict) and vp.get("w") and vp.get("h") and vp["h"] > 0:
      if vp.get("aspect") and float(vp["aspect"]) > 0:
        video_aspect = float(vp["aspect"])
      else:
        video_aspect = float(vp["w"]) / float(vp["h"])
    else:
      dwidth = self.query_ipc(["get_property", "dwidth"])
      dheight = self.query_ipc(["get_property", "dheight"])
      if dwidth and dheight and dheight > 0:
        video_aspect = float(dwidth) / float(dheight)

    if not video_aspect or video_aspect <= 0:
      return False

    mode = self.combo_fit.get_active_id() or "height"
    keepaspect = True
    panscan = 0.0

    if mode == "stretch":
      keepaspect = False
      panscan = 0.0
    elif mode == "fill":
      keepaspect = True
      panscan = 1.0
    elif mode == "fit":
      keepaspect = True
      panscan = 0.0
    else:
      keepaspect = True
      if abs(video_aspect - screen_aspect) < 0.03:
        panscan = 0.0
      elif mode == "height":
        panscan = 1.0 if video_aspect > screen_aspect else 0.0
      elif mode == "width":
        panscan = 0.0 if video_aspect > screen_aspect else 1.0

    self.send_ipc(["set_property", "keepaspect", keepaspect])
    self.send_ipc(["set_property", "panscan", panscan])
    self.send_ipc(
        ["set_property", "video-zoom", float(self.scale_zoom.get_value())]
    )
    self.send_ipc(
        ["set_property", "video-pan-x", float(self.scale_pan_x.get_value())]
    )
    self.send_ipc(
        ["set_property", "video-pan-y", float(self.scale_pan_y.get_value())]
    )

    return True

  def check_and_adapt_video(self, retries_left=25):
    """Verifica e adatta il video corrente non appena mpv espone le dimensioni del flusso."""
    if not self.is_running:
      return False

    current = self.query_ipc(["get_property", "path"]) or self.query_ipc(
        ["get_property", "filename"]
    )
    if not current:
      if retries_left > 0:
        GLib.timeout_add(
            150, lambda: self.check_and_adapt_video(retries_left - 1)
        )
      return False

    if self.apply_adaptation():
      self.last_adapted_video = current
      return False
    else:
      if retries_left > 0:
        GLib.timeout_add(
            150, lambda: self.check_and_adapt_video(retries_left - 1)
        )
      return False

  def on_slider_changed(self, widget):
    if not self.is_running:
      self.save_config()
      return
    self.send_ipc(
        ["set_property", "video-zoom", float(self.scale_zoom.get_value())]
    )
    self.send_ipc(
        ["set_property", "video-pan-x", float(self.scale_pan_x.get_value())]
    )
    self.send_ipc(
        ["set_property", "video-pan-y", float(self.scale_pan_y.get_value())]
    )
    self.save_config()

  def on_fit_changed(self, widget):
    self.save_config()
    self.apply_adaptation()

  def on_delete_event(self, widget, event):
    self.save_config()
    self.hide()
    return True

  def toggle_window_visibility(self, icon):
    if self.get_visible():
      self.hide()
    else:
      self.show_all()
      self.apply_mode_visibility()
      self.present()

  def on_tray_popup_menu(self, icon, button, activate_time):
    menu = Gtk.Menu()

    label_toggle = (
        "Nascondi Finestra" if self.get_visible() else "Mostra Finestra"
    )
    item_toggle = Gtk.MenuItem(label=label_toggle)
    item_toggle.connect(
        "activate", lambda w: self.toggle_window_visibility(icon)
    )
    menu.append(item_toggle)

    menu.append(Gtk.SeparatorMenuItem())

    item_pause = Gtk.MenuItem(label="⏸ Pausa / Riprendi")
    item_pause.connect("activate", self.toggle_pause)
    menu.append(item_pause)

    item_next = Gtk.MenuItem(label="⏭ Prossimo Video")
    item_next.connect("activate", self.next_track)
    menu.append(item_next)

    item_stop = Gtk.MenuItem(label="⏹ Ferma Sfondo")
    item_stop.connect("activate", self.stop_wallpaper)
    menu.append(item_stop)

    menu.append(Gtk.SeparatorMenuItem())

    item_quit = Gtk.MenuItem(label="❌ Esci Definitivamente")
    item_quit.connect("activate", self.quit_completely)
    menu.append(item_quit)

    menu.show_all()
    menu.popup(None, None, None, None, button, activate_time)

  def quit_completely(self, widget=None):
    self.save_config()
    self.close_socket()
    Gtk.main_quit()

  def load_config(self):
    defaults = {
        "mode": "single",
        "file": "",
        "folder": "",
        "recursive": True,
        "volume": 70,
        "fit_mode": "height",
        "zoom": 0.0,
        "pan_x": 0.0,
        "pan_y": 0.0,
    }
    if os.path.exists(CONFIG_FILE):
      try:
        with open(CONFIG_FILE, "r") as f:
          defaults.update(json.load(f))
      except Exception:
        pass
    return defaults

  def save_config(self):
    if not self.is_initialized:
      return
    os.makedirs(CONFIG_DIR, exist_ok=True)
    cfg = {
        "mode": "folder" if self.radio_folder.get_active() else "single",
        "file": self.selected_file,
        "folder": self.selected_folder,
        "recursive": self.check_recursive.get_active(),
        "volume": int(self.scale_volume.get_value()),
        "fit_mode": self.combo_fit.get_active_id() or "height",
        "zoom": round(float(self.scale_zoom.get_value()), 2),
        "pan_x": round(float(self.scale_pan_x.get_value()), 2),
        "pan_y": round(float(self.scale_pan_y.get_value()), 2),
    }
    try:
      with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=4)
    except Exception:
      pass

  def on_mode_changed(self, widget):
    if not self.is_initialized:
      return
    if self.radio_single.get_active():
      self.folder_box.hide()
      self.file_box.show()
    else:
      self.file_box.hide()
      self.folder_box.show()
    self.save_config()

  def on_volume_changed(self, widget):
    vol = int(widget.get_value())
    self.send_ipc(["set_property", "volume", vol])
    self.save_config()

  def choose_file_dialog(self, widget):
    dialog = Gtk.FileChooserDialog(
        title="Seleziona un video dal computer",
        parent=self,
        action=Gtk.FileChooserAction.OPEN,
    )
    dialog.add_buttons(
        "_Annulla", Gtk.ResponseType.CANCEL, "_Apri", Gtk.ResponseType.OK
    )

    filt = Gtk.FileFilter()
    filt.set_name("Video e GIF animati")
    filt.add_pattern("*.mp4")
    filt.add_pattern("*.webm")
    filt.add_pattern("*.mkv")
    filt.add_pattern("*.gif")
    filt.add_pattern("*.avi")
    filt.add_pattern("*.mov")
    dialog.add_filter(filt)

    if dialog.run() == Gtk.ResponseType.OK:
      self.selected_file = dialog.get_filename()
      self.lbl_file.set_text(self.selected_file)
      self.btn_open_file_loc.set_sensitive(True)
      self.save_config()
    dialog.destroy()

  def choose_folder_dialog(self, widget):
    dialog = Gtk.FileChooserDialog(
        title="Seleziona la cartella contenente i tuoi video",
        parent=self,
        action=Gtk.FileChooserAction.SELECT_FOLDER,
    )
    dialog.add_buttons(
        "_Annulla", Gtk.ResponseType.CANCEL, "_Seleziona", Gtk.ResponseType.OK
    )

    if dialog.run() == Gtk.ResponseType.OK:
      self.selected_folder = dialog.get_filename()
      self.lbl_folder.set_text(self.selected_folder)
      self.btn_open_folder_loc.set_sensitive(True)
      self.save_config()
    dialog.destroy()

  def get_socket(self):
    if self.ipc_sock is not None:
      return self.ipc_sock
    if not os.path.exists(SOCKET_PATH):
      return None
    try:
      s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
      s.connect(SOCKET_PATH)
      s.setblocking(False)
      self.ipc_sock = s
      self.ipc_buffer = ""
      return s
    except Exception:
      return None

  def close_socket(self):
    if self.ipc_sock is not None:
      try:
        self.ipc_sock.close()
      except Exception:
        pass
      self.ipc_sock = None
      self.ipc_buffer = ""

  def send_ipc(self, command):
    s = self.get_socket()
    if s is None:
      return
    try:
      payload = json.dumps({"command": command}) + "\n"
      s.sendall(payload.encode("utf-8"))
    except (BrokenPipeError, ConnectionResetError, OSError):
      self.close_socket()

  def query_ipc(self, command, timeout=0.3):
    s = self.get_socket()
    if s is None:
      return None

    self.request_counter = getattr(self, "request_counter", 0) + 1
    req_id = self.request_counter

    payload = json.dumps({"command": command, "request_id": req_id}) + "\n"
    try:
      s.sendall(payload.encode("utf-8"))
    except (BrokenPipeError, ConnectionResetError, OSError):
      self.close_socket()
      return None

    end_time = time.time() + timeout
    while time.time() < end_time:
      while "\n" in self.ipc_buffer:
        line, self.ipc_buffer = self.ipc_buffer.split("\n", 1)
        line = line.strip()
        if not line:
          continue
        try:
          resp = json.loads(line)
          if resp.get("request_id") == req_id:
            if resp.get("error") == "success":
              return resp.get("data")
            return None
        except json.JSONDecodeError:
          continue

      rem = max(0.01, end_time - time.time())
      ready = select.select([s], [], [], rem)
      if not ready[0]:
        break
      try:
        chunk = s.recv(4096)
        if not chunk:
          self.close_socket()
          return None
        self.ipc_buffer += chunk.decode("utf-8", errors="ignore")
      except (BlockingIOError, InterruptedError):
        continue
      except (BrokenPipeError, ConnectionResetError, OSError):
        self.close_socket()
        return None

    while "\n" in self.ipc_buffer:
      line, self.ipc_buffer = self.ipc_buffer.split("\n", 1)
      line = line.strip()
      if not line:
        continue
      try:
        resp = json.loads(line)
        if resp.get("request_id") == req_id:
          if resp.get("error") == "success":
            return resp.get("data")
          return None
      except json.JSONDecodeError:
        continue

    return None

  def update_now_playing(self):
    if not self.is_running:
      self.lbl_now_playing.set_text("Nessun video in riproduzione")
      self.current_playing_file = ""
      self.current_playing_path = ""
      self.last_adapted_video = None
      self.btn_open_now_playing.set_sensitive(False)
      return True

    filename = self.query_ipc(["get_property", "filename"])
    path = self.query_ipc(["get_property", "path"]) or filename
    if filename:
      self.lbl_now_playing.set_text(f"🎬 {filename}")
      self.current_playing_path = path or ""
      self.btn_open_now_playing.set_sensitive(
          bool(path and os.path.exists(path))
      )
      if path and path != self.last_adapted_video:
        self.check_and_adapt_video()
    return True

  def toggle_pause(self, widget=None):
    self.send_ipc(["cycle", "pause"])

  def next_track(self, widget=None):
    self.send_ipc(["playlist-next"])
    self.last_adapted_video = None
    GLib.timeout_add(150, self.check_and_adapt_video)
    GLib.timeout_add(300, self.update_now_playing)

  def reset_all(self, widget):
    self.scale_zoom.set_value(0.0)
    self.scale_pan_x.set_value(0.0)
    self.scale_pan_y.set_value(0.0)
    self.save_config()
    self.apply_adaptation()

  def start_wallpaper(self, silent=False):
    self.save_config()

    if self.radio_single.get_active():
      target = self.selected_file
      if not target or not os.path.isfile(target):
        if not silent:
          self.show_error(
              "Clicca su 'Scegli un file video' per selezionare un file."
          )
        return
      play_flags = ["--loop-file=inf"]
      media_input = [target]
      self.btn_next.set_sensitive(False)
    else:
      target_dir = self.selected_folder
      if not target_dir or not os.path.isdir(target_dir):
        if not silent:
          self.show_error(
              "Clicca su 'Scegli la cartella' per selezionare la cartella con i"
              " video."
          )
        return

      valid_exts = (
          ".mp4",
          ".webm",
          ".mkv",
          ".gif",
          ".avi",
          ".mov",
          ".m4v",
          ".wmv",
      )
      video_files = []

      if self.check_recursive.get_active():
        for root, dirs, files in os.walk(target_dir):
          for f in files:
            if f.lower().endswith(valid_exts):
              video_files.append(os.path.join(root, f))
      else:
        for f in os.listdir(target_dir):
          full_p = os.path.join(target_dir, f)
          if os.path.isfile(full_p) and f.lower().endswith(valid_exts):
            video_files.append(full_p)

      if not video_files:
        if not silent:
          self.show_error(
              "Nessun file video trovato con i criteri di ricerca attuali!"
          )
        return

      with open(PLAYLIST_FILE, "w") as pl:
        for vf in video_files:
          pl.write(vf + "\n")

      play_flags = ["--loop-file=no", "--loop-playlist=inf", "--shuffle"]
      media_input = [f"--playlist={PLAYLIST_FILE}"]
      self.btn_next.set_sensitive(True)

    self.stop_wallpaper(None)
    time.sleep(0.1)

    vol = int(self.scale_volume.get_value())

    cmd = [
        "xwinwrap",
        "-fs",
        "-fdt",
        "-ni",
        "-b",
        "-nf",
        "-un",
        "--",
        "mpv",
        "-wid",
        "WID",
        "--hwdec=auto",
        "--ao=pulse,pipewire,alsa",
        "--aid=auto",
        "--mute=no",
        f"--volume={vol}",
        f"--input-ipc-server={SOCKET_PATH}",
        *play_flags,
        *media_input,
    ]

    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    self.is_running = True
    self.last_adapted_video = None

    self.btn_start.set_sensitive(False)
    self.btn_pause.set_sensitive(True)
    self.btn_stop.set_sensitive(True)

    GLib.timeout_add(250, self.check_and_adapt_video)
    GLib.timeout_add(500, self.update_now_playing)

  def stop_wallpaper(self, widget=None):
    self.close_socket()
    subprocess.run(["killall", "-q", "xwinwrap", "mpv"])
    if os.path.exists(SOCKET_PATH):
      try:
        os.remove(SOCKET_PATH)
      except OSError:
        pass
    self.is_running = False
    self.last_adapted_video = None
    self.current_playing_path = ""
    self.lbl_now_playing.set_text("Nessun video in riproduzione")
    self.btn_open_now_playing.set_sensitive(False)
    self.btn_start.set_sensitive(True)
    self.btn_pause.set_sensitive(False)
    self.btn_next.set_sensitive(False)
    self.btn_stop.set_sensitive(False)

  def show_error(self, message):
    dialog = Gtk.MessageDialog(
        transient_for=self,
        flags=0,
        message_type=Gtk.MessageType.WARNING,
        buttons=Gtk.ButtonsType.OK,
        text=message,
    )
    dialog.run()
    dialog.destroy()


if __name__ == "__main__":
  # Uso: python3 wallpaper.py [--show] [file_video]
  # Con un file: viene messo nel campo "singolo file" e riprodotto subito. Se il programma e' gia'
  # aperto, il file viene passato all'istanza attiva (nessuna seconda copia).
  file_arg = file_da_argv(sys.argv[1:])
  if file_arg and invia_a_istanza_attiva(file_arg):
    sys.exit(0)
  app = WallpaperApp()
  app.avvia_server_comandi()
  if file_arg:
    # prima del ciclo principale: l'avvio automatico dell'ultimo file (300 ms) trova gia' il playback attivo
    app.usa_file_esterno(file_arg)
  # Parte con il playback attivo e la finestra nascosta nel tray
  # (usa --show per aprire anche la finestra all'avvio)
  if "--show" in sys.argv:
    app.show_all()
    app.apply_mode_visibility()
  Gtk.main()
