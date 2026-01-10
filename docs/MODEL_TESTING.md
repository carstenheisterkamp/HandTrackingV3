# Model Testing Guide: Lite vs Full

**Datum:** 2026-01-10  
**Ziel:** Vergleiche `lite` vs `full` Modelle für bessere Gestenerkennung

---

## 🎯 Schnellstart

### Schritt 0: Status prüfen (auf Jetson)

```bash
cd ~/dev/HandTrackingV3
./scripts/check_models.sh
```

**Zeigt:**
- Welche TFLite Models vorhanden sind
- Welche ONNX Models konvertiert sind
- Welche TensorRT Engines gecached sind
- Was noch zu tun ist

### Schritt 1: Prüfe vorhandene Models (auf Jetson)

```bash
cd ~/dev/HandTrackingV3
ls -lh models/*.tflite
```

**Erwartete Modelle (sollten bereits vorhanden sein):**
- `palm_detection_lite.tflite` ✅
- `hand_landmark_lite.tflite` ✅
- `palm_detection_full.tflite` ✅ (oder via sh4/sh6 blob)
- `hand_landmark_full.tflite` ✅ (oder via sh4/sh6 blob)

**Falls Full Models fehlen:**
```bash
# Option 1: Von MediaPipe herunterladen
wget -P models/ https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

# Option 2: Download-Script ausführen
python3 scripts/download_tflite_models.py
```


### Schritt 2: Zu ONNX konvertieren (auf Jetson)

```bash
cd ~/dev/HandTrackingV3
python3 scripts/convert_to_onnx.py
```

**Dies konvertiert (wenn noch nicht vorhanden):**
- `palm_detection_lite.tflite` → `palm_detection.onnx` ✅
- `hand_landmark_lite.tflite` → `hand_landmark.onnx` ✅
- `palm_detection_full.tflite` → `palm_detection_full.onnx` ← NEU
- `hand_landmark_full.tflite` → `hand_landmark_full.onnx` ← NEU

**Hinweis:** Script überspringt bereits vorhandene ONNX-Dateien automatisch.

**Wichtig:** TensorRT baut beim ersten Start `.engine` Dateien (dauert ~2-3 Minuten).

### Schritt 3: Full Models aktivieren

**In `src/main.cpp` Zeile 38:**
```cpp
const bool USE_FULL_MODELS = true;  // ← Auf true setzen
```

### Schritt 4: Neu kompilieren und testen

```bash
# Auf Jetson
cd ~/dev/HandTrackingV3/cmake-build-debug-remote-host
ninja
sudo systemctl restart hand-tracking
```

### Schritt 5: Zurück zu Lite Models

**In `src/main.cpp` Zeile 38:**
```cpp
const bool USE_FULL_MODELS = false;  // ← Auf false setzen
```

Neu kompilieren: `ninja`

---

## 📊 Was zu testen

### Performance Metrics

**Überwache im Log:**
```
FPS: XX.X                    ← Sollte bei 25-30 bleiben
TensorRT: Ready              ← Initalisierung erfolgreich
Hands Detected: X            ← Erkennungsrate
```

**Bei viel langsamerer FPS (<20):** Zurück zu Lite Models

### Gestenerkennung

Teste die problematischen Gesten:

| Geste | Problem (Lite) | Erwartung (Full) |
|-------|----------------|------------------|
| FIVE | Wird als FOUR erkannt | Besser? |
| FIST (2 Hände) | Inkonsistent | Stabiler? |
| THUMBS_UP | Verwechslung mit FIST | Zuverlässiger? |
| POINTING | Verwechslung mit TWO | Präziser? |

**Test bei:**
- 40cm Abstand
- 80cm Abstand
- 120cm Abstand
- Verschiedene Winkel

### False Positives

- Gesicht noch erkannt? (sollte 0 sein mit Haar Cascade)
- Andere Objekte als Hand erkannt?

---

## 🔍 Model Unterschiede

### Lite Models (Default)

**Palm Detection Lite:**
- Input: 192×192
- Params: ~100K
- Inference: ~5-8ms auf Jetson

**Hand Landmark Lite:**
- Input: 224×224
- Params: ~200K
- Keypoints: 21
- Inference: ~5-7ms auf Jetson

**Total:** ~12-15ms → 60-80 FPS möglich

### Full Models

**Palm Detection Full:**
- Input: 256×256 (größer)
- Params: ~500K (5x mehr)
- Inference: ~10-15ms (langsamer)

**Hand Landmark Full:**
- Input: 256×256 (größer)
- Params: ~1M (5x mehr)
- Keypoints: 21 (gleich)
- Inference: ~15-20ms (langsamer)

**Total:** ~25-35ms → 28-40 FPS möglich

**Vorteil Full:**
- Bessere Erkennung bei weiter Entfernung
- Robuster bei schwierigen Winkeln
- Präzisere Landmark-Positionen

**Nachteil Full:**
- 2-3x langsamer
- Größere TensorRT Engines (mehr VRAM)
- Längere Init-Zeit

---

## 📈 Benchmark Template

### Test 1: FPS Impact

| Modell | FPS Avg | FPS Min | FPS Max | Inference Time |
|--------|---------|---------|---------|----------------|
| Lite   | 28.5    | 25.1    | 30.2    | ~15ms          |
| Full   | ?       | ?       | ?       | ?              |

### Test 2: Gestenerkennung Accuracy

| Geste | Lite Accuracy | Full Accuracy | Improvement |
|-------|---------------|---------------|-------------|
| FIVE  | 70%           | ?             | ?           |
| FIST  | 80%           | ?             | ?           |
| THUMBS_UP | 75%       | ?             | ?           |
| POINTING | 85%        | ?             | ?           |

*(Accuracy = richtig erkannt / 20 Versuche)*

### Test 3: False Positives

| Szenario | Lite | Full |
|----------|------|------|
| Gesicht im Bild | 0 | ? |
| Kein Hand im Bild | 0 | ? |
| Objekt (Tasse) | 0 | ? |

---

## 🎯 Entscheidungskriterien

### Bleibe bei Lite wenn:
- ✅ FPS bleibt bei 25-30
- ✅ Gesten-Accuracy >85%
- ✅ Keine häufigen False Positives

### Wechsel zu Full wenn:
- ✅ FPS bleibt >20 FPS
- ✅ Gesten-Accuracy signifikant besser (>10% Verbesserung)
- ✅ Keine neuen False Positives

### Verwerfe Full wenn:
- ❌ FPS fällt unter 20
- ❌ Keine signifikante Verbesserung (<5%)
- ❌ Mehr False Positives

---

## 🔧 Troubleshooting

### Engine Build dauert ewig (>5 Minuten)
- Normal beim ersten Start mit Full Models
- TensorRT optimiert für Jetson Hardware
- Nur einmal nötig, danach cached

### Out of Memory Error
- Full Models benötigen mehr VRAM
- Jetson Orin Nano hat 8GB shared RAM/VRAM
- Lösung: Zurück zu Lite oder MAXN Mode aktivieren

### Keine FPS Verbesserung erkennbar
- Log zeigt "TensorRT: Building..." → warte bis fertig
- Check `tegrastats` für GPU Utilization
- Stelle sicher, dass MAXN Mode aktiv ist

### Service startet nicht mehr
- Check Log: `journalctl -u hand-tracking -n 50`
- Prüfe ob ONNX Dateien existieren: `ls -lh models/*.onnx`
- Fallback: Setze `USE_FULL_MODELS = false` und rebuild

---

## 📝 Notizen

**Wichtig:**
- Engine Dateien (`*.engine`) werden automatisch erstellt
- Sind Hardware-spezifisch (Jetson Orin Nano)
- Müssen neu gebaut werden bei Model-Wechsel
- Liegen in `models/` neben `.onnx` Dateien

**Tipp:**
- Teste Full Models nur bei konkreten Erkennungsproblemen
- Lite Models sind für die meisten Anwendungen ausreichend
- Full Models könnten bei schlechten Lichtverhältnissen helfen


