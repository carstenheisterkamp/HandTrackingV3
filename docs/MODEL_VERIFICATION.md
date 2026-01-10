# Model Verification Guide

**Datum:** 2026-01-10  
**Problem:** Wie prüfe ich, ob FULL oder LITE Models wirklich aktiv sind?

---

## ✅ 3 Wege zur Verifikation

### 1️⃣ Im Service Log (beim Start)

```bash
journalctl -u hand-tracking -n 100 | grep -A 10 "MODEL CONFIGURATION"
```

**Erwartete Ausgabe:**

**FULL Models aktiv:**
```
═══════════════════════════════════════════════
MODEL CONFIGURATION
  Mode: FULL (High Accuracy)
  Palm Model: models/palm_detection_full.onnx
  Landmark Model: models/hand_landmark_full.onnx
═══════════════════════════════════════════════

🔍 Checking for ONNX models...
   Palm model path: models/palm_detection_full.onnx
   Landmark model path: models/hand_landmark_full.onnx
   Palm exists: YES
   Landmark exists: YES
   Palm model size: 3500 KB          ← FULL: ~3.5 MB
   Landmark model size: 6800 KB      ← FULL: ~6.8 MB
```

**LITE Models aktiv:**
```
═══════════════════════════════════════════════
MODEL CONFIGURATION
  Mode: LITE (Fast)
  Palm Model: models/palm_detection.onnx
  Landmark Model: models/hand_landmark.onnx
═══════════════════════════════════════════════

🔍 Checking for ONNX models...
   Palm model path: models/palm_detection.onnx
   Landmark model path: models/hand_landmark.onnx
   Palm exists: YES
   Landmark exists: YES
   Palm model size: 1200 KB          ← LITE: ~1.2 MB
   Landmark model size: 2300 KB      ← LITE: ~2.3 MB
```

**Key Indicator:** Die Dateigrößen!
- FULL: Palm ~3.5MB, Landmark ~6.8MB
- LITE: Palm ~1.2MB, Landmark ~2.3MB

---

### 2️⃣ Im MJPEG Preview (http://100.101.16.21:8080)

**Overlay oben links im Bild (gespiegelt wie ein Spiegel):**

```
┌──────────────────────────┐
│ 2026-01-10 15:30:45     │
│ FPS: 28.5               │
│ TensorRT: Ready         │
│ Models: LITE            │ ← Zeigt aktive Models
│ Stereo: Active          │
│ Hands Detected: 1 / 2   │
└──────────────────────────┘
```

**Farbe:**
- `Models: LITE` → Grün
- `Models: FULL` → Orange

---

### 3️⃣ Im 2-Sekunden Stats Log

```bash
journalctl -u hand-tracking -f | grep "Models:"
```

**Ausgabe alle 2 Sekunden:**

```
Models: FULL (full models)
Models: FULL (full models)
Models: FULL (full models)
```

oder

```
Models: LITE (lite models)
Models: LITE (lite models)
Models: LITE (lite models)
```

---

## 🔍 Schnell-Check

**Einzeiler um aktive Models zu sehen:**
```bash
journalctl -u hand-tracking --since "1 minute ago" | grep -E "(MODEL CONFIGURATION|Models:|model size)" | head -10
```

**Erwartete Ausgabe:**
```
MODEL CONFIGURATION
  Mode: FULL (High Accuracy)
   Palm model size: 3500 KB
   Landmark model size: 6800 KB
Models: FULL (full models)
Models: FULL (full models)
```

---

## ❓ Troubleshooting

### "Models: LITE" obwohl USE_FULL_MODELS = true

**Ursache:** Code wurde nicht neu kompiliert

**Lösung:**
```bash
cd ~/dev/HandTrackingV3/cmake-build-debug-remote-host
ninja
sudo systemctl restart hand-tracking
```

### "Model size" wird nicht geloggt

**Ursache:** Alte Binary ohne neuen Code

**Lösung:** Neu kompilieren und deployen

### FULL Models zeigen gleiche FPS wie LITE

**Mögliche Ursachen:**
1. **LITE Models laufen tatsächlich** (check Log)
2. **GPU ist underutilized** (check mit `tegrastats`)
3. **Bottleneck ist woanders** (Mono Stereo, MJPEG Encoding)
4. **FULL Models sind gut optimiert** von TensorRT

**Verifikation:**
```bash
# Während Service läuft
tegrastats

# Achte auf:
# GR3D_FREQ: GPU Auslastung (sollte >80% sein)
# RAM: Speichernutzung (FULL braucht mehr)
```

### Kein Unterschied in Genauigkeit

**Mögliche Ursachen:**
1. **LITE Models sind schon gut genug** für deine Szene
2. **Testbedingungen optimal** (gute Beleuchtung, frontale Ansicht)
3. **Thresholds verdecken Verbesserung** (Gestenerkennung nutzt Thresholds)

**Empfehlung:** Teste in schwierigeren Szenarien:
- Schlechte Beleuchtung
- Extreme Winkel (Seitenansicht)
- Weit entfernt (>1.5m)
- Schnelle Bewegungen

---

## 📊 Erwartete Werte

### Model File Sizes (ONNX):

| Model | LITE | FULL | Faktor |
|-------|------|------|--------|
| palm_detection | 1.2 MB | 3.5 MB | 3x |
| hand_landmark | 2.3 MB | 6.8 MB | 3x |

### TensorRT Engine Sizes (nach Build):

| Model | LITE | FULL | Faktor |
|-------|------|------|--------|
| palm_detection.onnx.engine | ~15 MB | ~45 MB | 3x |
| hand_landmark.onnx.engine | ~30 MB | ~90 MB | 3x |

### Performance:

| Metric | LITE | FULL | Delta |
|--------|------|------|-------|
| FPS | 28-30 | 22-26 | -4 bis -6 |
| Inference Time | ~15ms | ~25-30ms | +10-15ms |
| Init Time | 30-60s | 2-3 min | +1.5-2 min |
| VRAM Usage | ~200 MB | ~350 MB | +150 MB |

---

## ✅ Checkliste: FULL Models sind aktiv

- [ ] Log zeigt "Mode: FULL (High Accuracy)"
- [ ] Model Paths enthalten "_full.onnx"
- [ ] File Sizes: Palm ~3.5MB, Landmark ~6.8MB
- [ ] Preview zeigt "Models: FULL" (Orange)
- [ ] Stats Log zeigt "Models: FULL (full models)"
- [ ] FPS ist niedriger als mit LITE (~22-26 statt 28-30)

**Wenn alle Punkte ✅ → FULL Models sind definitiv aktiv!**

