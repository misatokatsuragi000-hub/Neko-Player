@@
 def karaoke_espandi(d):
-    """Da formato v2 a v1 (quello usato internamente). I dati v1 passano invariati."""
-    if d.get("versione", 1) < 2:
-        return d
-    if "s" not in d:
-        raise ValueError("File karaoke non valido: manca 's'.")
+    """Da formato v2 a v1 (quello usato internamente).
+
+    Il supporto al vecchio formato v1 e' stato rimosso: i file karaoke devono
+    contenere il campo 's' del nuovo formato compresso (versione >= 2).
+    """
+    if "s" not in d:
+        raise ValueError(
+            "Formato karaoke non supportato: usa solo il nuovo formato v2 (campo 's'). "
+            "Ricarica o rigenera il file .json dal programma."
+        )
@@
     def _scrivi_json(self, percorso):
         r = self.ultimo_ris
         media = self.ed_file.text().strip()
         dati = dict(
             file=os.path.basename(media),
             percorso_file=os.path.abspath(media),
         )
         comp = karaoke_compatto(r)
-        if comp is not None:
-            dati.update(comp)
-        else:  # fallback: formato completo v1
-            dati.update(
-                versione=1,
-                orig=r["orig"],
-                hira=r["hira"],
-                span_o=[list(x) for x in r["span_o"]],
-                span_h=[list(x) for x in r["span_h"]],
-                t0=[round(float(t), 2) for t in r["t0"]],
-                t1=[round(float(t), 2) for t in r["t1"]],
-                tok=r["tok"],
-            )
+        if comp is None:
+            raise ValueError(
+                "Impossibile serializzare il karaoke nel formato v2. "
+                "I dati attuali non sono compatibili con la compattazione."
+            )
+        dati.update(comp)
+        dati["versione"] = 2
         with open(percorso, "w", encoding="utf-8") as fh:
             json.dump(dati, fh, ensure_ascii=False, separators=(",", ":"))
@@
     def _applica_karaoke(self, d):
-        d = karaoke_espandi(d)
+        try:
+            d = karaoke_espandi(d)
+        except ValueError as e:
+            raise ValueError(
+                "Formato karaoke non supportato: il file e' vecchio o incompatibile. "
+                "Genera un nuovo file con il programma (formato v2 only)."
+            ) from e
         self._valida_karaoke(d)
         r = dict(
             orig=d["orig"], hira=d["hira"],
             span_o=[tuple(x) for x in d["span_o"]],
             span_h=[tuple(x) for x in d["span_h"]],
@@
     def carica_karaoke_json(self, pj):
         try:
             with open(pj, encoding="utf-8") as fh:
                 dati = karaoke_espandi(json.load(fh))
             self._valida_karaoke(dati)
         except Exception as e:
-            QMessageBox.warning(self, "Karaoke", "Impossibile leggere il file:\n{}".format(e))
+            QMessageBox.warning(
+                self,
+                "Karaoke",
+                "Formato non supportato. Usa solo file .json generati dal programma in formato v2.\n\n"
+                "Dettaglio: {}".format(e),
+            )
             return
