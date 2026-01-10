# Model Testing - Quick Reference

**Datum:** 2026-01-10

## ⚡ TL;DR - Auf dem Jetson

### Status prüfen:
```bash
cd ~/dev/HandTrackingV3
./scripts/check_models.sh
```

### Full Models aktivieren:
```bash
# 1. ONNX konvertieren (falls noch nicht geschehen)
python3 scripts/convert_to_onnx.py

# 2. Full Models aktivieren
./scripts/switch_models.sh full

# 3. Neu bauen und starten
cd cmake-build-debug-remote-host
ninja
sudo systemctl restart hand-tracking

# 4. Log beobachten
journalctl -u hand-tracking -f
```

### Zurück zu Lite:
```bash
./scripts/switch_models.sh lite
ninja -C cmake-build-debug-remote-host
sudo systemctl restart hand-tracking
```

---

## 📊 Was zu beobachten

### Im Service Log:
```
═══════════════════════════════════════════════
MODEL CONFIGURATION
  Mode: FULL (High Accuracy)          ← Aktiver Modus
  Palm Model: models/palm_detection_full.onnx
  Landmark Model: models/hand_landmark_full.onnx
═══════════════════════════════════════════════

[... TensorRT Building (2-3 min beim ersten Mal) ...]

✅ TensorRT inference initialized successfully
FPS: 24.8                              ← FPS (sollte >20 sein)
Hands Detected: 2
```

### Performance Vergleich:
```
Lite:  FPS: 28-30  |  Inference: ~15ms
Full:  FPS: 22-26  |  Inference: ~30ms
```

---

## 🎯 Entscheidung

### Bleibe bei Lite wenn:
- ✅ Gesten funktionieren gut genug
- ✅ FPS bleibt bei 28-30

### Wechsel zu Full wenn:
- ✅ Gesten signifikant besser (>10% improvement)
- ✅ FPS bleibt >20
- ✅ Keine neuen Probleme

**Siehe `MODEL_TESTING.md` für detaillierte Anleitung**

