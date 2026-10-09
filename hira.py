
    # --------------------------------------------------------------- azioni
    # ------------------------------------------------------- karaoke (.json)
    def _scrivi_json(self, percorso):
        r = self.ultimo_ris
        media = self.ed_file.text().strip()

        tok = r.get("tok") or {}
        tokens = []
        if tok and "o" in tok and "h" in tok:
            for i, (o, h, s, e) in enumerate(
                zip(
                    tok.get("o", []),
                    tok.get("h", []),
                    r.get("t0", []),
                    r.get("t1", []),
                )
            ):
                tokens.append({
                    "o": o,
                    "h": h,
                    "t": [float(s), float(e)],
                    "seg": tok.get("seg", [0])[i] if len(tok.get("seg", [])) > i else 0,
                })

        dati = dict(
            versione=2,
            version=2,
            file=os.path.basename(media),
            percorso_file=os.path.abspath(media),
            orig=r["orig"],
            hira=r["hira"],
            tokens=tokens,
        )
        with open(percorso, "w", encoding="utf-8") as fh:
            json.dump(dati, fh, ensure_ascii=False)

    @staticmethod
    def _valida_karaoke(d):
        if "tokens" in d:
            if not isinstance(d["tokens"], list):
                raise ValueError("File karaoke non valido: 'tokens' deve essere una lista.")
            for item in d["tokens"]:
                if not isinstance(item, dict):
                    raise ValueError("File karaoke non valido: ogni token deve essere un oggetto JSON.")
                if set(("o", "h", "t")) - set(item):
                    raise ValueError("File karaoke non valido: un token manca di 'o', 'h' o 't'.")
                if not isinstance(item["t"], (list, tuple)) or len(item["t"]) != 2:
                    raise ValueError("File karaoke non valido: il campo 't' deve essere [inizio, fine].")
            return

        for k in ("orig", "hira", "span_o", "span_h", "t0", "t1", "tok"):
            if k not in d:
                raise ValueError("File karaoke non valido: manca '{}'.".format(k))
        n = len(d["t0"])
        tok = d["tok"]
        if not (len(d["t1"]) == len(d["span_o"]) == len(d["span_h"]) == n
                and all(len(tok.get(k, [])) == n for k in ("o", "h", "seg"))):
            raise ValueError("File karaoke non valido: dati incoerenti.")

    def _normalizza_karaoke(self, d):
        if "tokens" in d:
            tokens = d["tokens"]
            t0, t1, tok_o, tok_h, tok_seg = [], [], [], [], []
            span_o, span_h = [], []
            pos_o = pos_h = 0

            for item in tokens:
                o = item.get("o", "")
                h = item.get("h", "")
                s, e = item["t"]
                t0.append(float(s))
                t1.append(float(e))
                tok_o.append(o)
                tok_h.append(h)
                tok_seg.append(item.get("seg", 0))

                span_o.append((pos_o, pos_o + l16(o)))
                span_h.append((pos_h, pos_h + l16(h)))

                pos_o += l16(o)
                pos_h += l16(h)

            return dict(
                orig=d["orig"],
                hira=d["hira"],
                span_o=span_o,
                span_h=span_h,
                t0=t0,
                t1=t1,
                tok={"o": tok_o, "h": tok_h, "seg": tok_seg},
                avviso="",
            )

        return dict(
            orig=d["orig"],
            hira=d["hira"],
            span_o=[tuple(x) for x in d["span_o"]],
            span_h=[tuple(x) for x in d["span_h"]],
            t0=list(d["t0"]),
            t1=list(d["t1"]),
            tok=d["tok"],
            avviso="",
        )

    def _applica_karaoke(self, d):
        self._valida_karaoke(d)
        r = self._normalizza_karaoke(d)
        self.ultimo_ris = r
        self._imposta_testi(r["orig"], r["hira"], r["span_o"], r["span_h"],
                            r["t0"], r["t1"], r["tok"])
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
    imposta_palette_scura(app)
    app.setStyleSheet(STILE)

    w = Finestra()
    w.show()

    sys.exit(app.exec_())
