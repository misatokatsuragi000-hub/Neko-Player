
*** Begin Patch
*** Update File: hira.py
@@
     def _scrivi_json(self, percorso):
         r = self.ultimo_ris
+        if r is None:
+            raise ValueError("Nessun karaoke da salvare.")
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
+                "Impossibile salvare il karaoke nel formato v2: i dati non sono compatibili "
+                "con la serializzazione nuova."
+            )
+        dati.update(comp)
         with open(percorso, "w", encoding="utf-8") as fh:
             json.dump(dati, fh, ensure_ascii=False, separators=(",", ":"))
 
     @staticmethod
     def _valida_karaoke(d):
+        if not isinstance(d, dict):
+            raise ValueError("File karaoke non valido: il contenuto non e' un oggetto JSON.")
+        if "s" not in d:
+            raise ValueError(
+                "File karaoke non supportato: usa solo il formato JSON nuovo (versione v2)."
+            )
+        if d.get("versione", 2) != 2:
+            raise ValueError(
+                "File karaoke non supportato: il formato legacy non e' piu' accettato."
+            )
         for k in ("orig", "hira", "span_o", "span_h", "t0", "t1", "tok"):
             if k not in d:
                 raise ValueError("File karaoke non valido: manca '{}'.".format(k))
         n = len(d["t0"])
         tok = d["tok"]
@@
     def _applica_karaoke(self, d):
+        if not isinstance(d, dict):
+            raise ValueError("File karaoke non valido: il contenuto non e' un oggetto JSON.")
+        if "s" not in d:
+            raise ValueError(
+                "File karaoke non supportato: usa solo il formato JSON nuovo (versione v2)."
+            )
+        if d.get("versione", 2) != 2:
+            raise ValueError(
+                "File karaoke non supportato: il formato legacy non e' piu' accettato."
+            )
         d = karaoke_espandi(d)
         self._valida_karaoke(d)
         r = dict(
             orig=d["orig"], hira=d["hira"],
             span_o=[tuple(x) for x in d["span_o"]],
@@
     def carica_karaoke_json(self, pj):
         try:
             with open(pj, encoding="utf-8") as fh:
-                dati = karaoke_espandi(json.load(fh))
+                dati = json.load(fh)
+            if not isinstance(dati, dict):
+                raise ValueError("File karaoke non valido: il contenuto non e' un oggetto JSON.")
+            if "s" not in dati:
+                raise ValueError(
+                    "File karaoke non supportato: usa solo il formato JSON nuovo (versione v2)."
+                )
+            if dati.get("versione", 2) != 2:
+                raise ValueError(
+                    "File karaoke non supportato: il formato legacy non e' piu' accettato."
+                )
+            dati = karaoke_espandi(dati)
             self._valida_karaoke(dati)
         except Exception as e:
             QMessageBox.warning(self, "Karaoke", "Impossibile leggere il file:\n{}".format(e))
             return
*** End Patch