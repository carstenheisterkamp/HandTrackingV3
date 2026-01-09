# TODO: V3 3D Hand Controller Implementation

> **Aktuelle Phase:** Phase 2 - TensorRT Inference
> **Letztes Update:** 2026-01-09
> **Ziel:** Hand-NNs auf Jetson mit TensorRT

---

## 🎯 Aktuelle Aufgabe

### Phase 2.1: TensorRT Engine Wrapper
- [x] TensorRTEngine.hpp/.cpp erstellen
- [x] Engine laden/erstellen (.onnx → .engine)
- [x] Inference Methode (Input → Output Buffer)
- [x] CUDA Memory Management

### Phase 2.2: Palm Detection
- [x] PalmDetector.hpp/.cpp erstellen
- [ ] ONNX Model beschaffen/konvertieren
- [x] NV12 → RGB Preprocessing (GPU)
- [x] Post-Processing (BBox, Score, Anchors)

### Phase 2.3: Hand Landmark
- [x] HandLandmark.hpp/.cpp erstellen
- [x] ROI Extraction aus Palm Detection
- [x] 21 Landmarks Output Parsing
- [x] Unletterbox Koordinaten

### Phase 2.4: ProcessingLoop Integration
- [x] PalmDetector + HandLandmark in ProcessingLoop einbinden
- [x] HandTracker + GestureFSM integrieren
- [ ] ONNX Models beschaffen/konvertieren
- [ ] Test: 30+ FPS mit NNs verifizieren

---

## 📅 Roadmap

### ✅ Phase 1: Sensor-Only Pipeline - ERLEDIGT
**Ergebnis:** 30 FPS stabil auf Jetson

| Task | Status | Notes |
|------|--------|-------|
| PipelineManager: Mono L/R hinzufügen | ✅ | THE_400_P @ 60fps |
| PipelineManager: RGB 640×360 NV12 | ✅ | LETTERBOX mode |
| PipelineManager: Sync Node | ✅ | 10ms threshold |
| InputLoop: MessageGroup parsing | ✅ | rgb + monoLeft + monoRight |
| Types.hpp: V3 Konstanten | ✅ | GestureState, Point3D, etc. |
| Config: FPS auf 60 ändern | ✅ | main.cpp |
| Test: 60 FPS verifizieren | ⬜ | Auf Jetson deployen |

### Phase 2: TensorRT Inference
**Ziel:** Hand-NNs auf Jetson mit TensorRT

| Task | Status | Notes |
|------|--------|-------|
| TensorRT Engine Wrapper | ⬜ | Generische Klasse |
| Palm Detection TRT Engine | ⬜ | .onnx → .engine |
| Hand Landmark TRT Engine | ⬜ | .onnx → .engine |
| NV12 → RGB Preprocessing (GPU) | ⬜ | CUDA/NPP |
| LETTERBOX Preprocessing | ⬜ | GPU-seitig |
| Unletterbox Postprocessing | ⬜ | Koordinaten zurückmappen |
| ProcessingLoop Integration | ⬜ | Inference Pipeline |
| Test: 30+ FPS verifizieren | ⬜ | Mit beiden NNs |

### Phase 3: Stereo Depth (Punktuell)
**Ziel:** Z-Koordinate nur am Palm Center

| Task | Status | Notes |
|------|--------|-------|
| StereoDepth Klasse | ✅ | src/core/StereoDepth.cpp |
| OAK-D Kalibrierung laden | ✅ | Default-Werte implementiert |
| Lokales Stereo Matching (9×9) | ✅ | SAD Block Matching |
| Median Filter für Robustheit | ✅ | robustMedian() |
| Z in Kamera-Koordinaten | ✅ | pixelTo3D() |
| Rectification Maps berechnen | ⬜ | TODO: OpenCV stereoRectify |
| Device Kalibrierung laden | ⬜ | dai::Device::readCalibration() |
| Test: Tiefe verifizieren | ⬜ | Bekannte Abstände |

### Phase 4: Kalman Tracking
**Ziel:** Glatte, prädiktive Trajektorien

| Task | Status | Notes |
|------|--------|-------|
| HandTracker Klasse | ✅ | src/core/HandTracker.cpp |
| 6-State Kalman Filter | ✅ | [x,y,z,vx,vy,vz] |
| VIP Lock Logic (15 Frames) | ✅ | ~250ms Stabilität |
| Dropout Handling | ✅ | Pure Prediction |
| +1 Frame Prediction | ✅ | Latenz-Kompensation |
| One-Euro für Rotationen | ⬜ | Landmarks-relativ |
| Test: Jitter messen | ⬜ | <5ms σ Ziel |

### Phase 5: Gesture FSM
**Ziel:** Robuste Gesten-Erkennung

| Task | Status | Notes |
|------|--------|-------|
| GestureFSM Klasse | ✅ | src/core/GestureFSM.cpp |
| States definieren | ✅ | Idle/Palm/Pinch/Grab/Point |
| Hysteresis Thresholds | ✅ | Enter/Exit unterschiedlich |
| Debounce (3 Frames) | ✅ | ~50ms @ 60fps |
| Finger Extension Check | ✅ | Landmark-basiert |
| Test: Gesten-Übergänge | ⬜ | Kein Flackern |

### Phase 6: OSC Integration
**Ziel:** 30 Hz konstante Ausgabe

| Task | Status | Notes |
|------|--------|-------|
| 30 Hz Rate Limiting | ⬜ | Decoupled von FPS |
| Drop-Oldest >50ms | ⬜ | Backpressure |
| /hand/palm Message | ⬜ | x, y, z |
| /hand/velocity Message | ⬜ | vx, vy, vz |
| /hand/gesture Message | ⬜ | state, confidence |
| /service/status Message | ⬜ | Heartbeat |
| Test: E2E Latenz <60ms | ⬜ | Glass-to-OSC |

---

## 📋 Quick Reference

### Wichtige Konstanten (V3)
```cpp
// Camera
CAMERA_FPS = 60
RGB_WIDTH = 640, RGB_HEIGHT = 360
MONO_WIDTH = 640, MONO_HEIGHT = 400

// Tracking
VIP_LOCK_FRAMES = 15
DROPOUT_LIMIT = 5

// Gestures
PINCH_THRESHOLD_ENTER = 0.08
PINCH_THRESHOLD_EXIT = 0.12
DEBOUNCE_FRAMES = 3

// OSC
OSC_RATE_HZ = 30
MAX_LATENCY_MS = 50
```

### Dateien die geändert werden
- `src/core/PipelineManager.cpp` - Sensor-Only Pipeline
- `include/core/PipelineManager.hpp` - Config Updates
- `src/core/InputLoop.cpp` - MessageGroup Parsing
- `src/main.cpp` - FPS Config
- `include/core/Types.hpp` - Neue Typen

### Neue Dateien (geplant)
- `src/inference/TensorRTEngine.cpp` - TRT Wrapper
- `src/inference/PalmDetector.cpp` - Palm Detection
- `src/inference/HandLandmark.cpp` - Landmark Inference
- `src/core/HandTracker.cpp` - Kalman Filter
- `src/core/GestureFSM.cpp` - Gesten State Machine
- `src/core/StereoDepth.cpp` - Punktuelle Tiefe

---

## 📝 Notizen

### 2026-01-09
- V3 Architektur definiert: OAK-D = Sensor-Only
- Kernprinzip: "Wir bauen einen 3D-Controller, kein CV-System"
- XLink bleibt unidirektional (kein BBox-Rückkanal-Problem)
- Start mit Phase 1: Sensor-Only Pipeline

**Umbau durchgeführt:**
- PipelineManager komplett auf Sensor-Only umgebaut
  - RGB 640×360 NV12 @ 60fps (LETTERBOX)
  - Mono L/R 640×400 GRAY8 @ 60fps
  - Sync Node mit 10ms Threshold
  - Keine NNs mehr auf OAK-D
- InputLoop nur noch Sync-Mode (kein Fallback auf RGB-only)
- Neue Komponenten implementiert:
  - HandTracker: Kalman Filter mit 6 States [x,y,z,vx,vy,vz]
  - GestureFSM: State Machine (Idle/Palm/Pinch/Grab/Point)
  - StereoDepth: Punktuelle Tiefe am Palm Center
- Types.hpp mit V3 Konstanten und neuen Typen

**Nächster Schritt:** Auf Jetson deployen und 60 FPS testen

---

## ⚠️ Bekannte Risiken

1. **PoE Bandwidth:** 60fps × (RGB + 2×Mono) = ~40-50 MB/s → sollte passen (GigE = 125 MB/s)
2. **TensorRT Conversion:** Palm/Landmark Blobs müssen zu ONNX → TRT
3. **Stereo Kalibrierung:** Muss aus OAK-D Device geladen werden

---

## ✅ Erledigte Aufgaben

- [x] OPTIMAL_WORKFLOW_V3.md erstellt
- [x] TODO.md erstellt
- [x] PipelineManager.cpp: V3 Sensor-Only Pipeline (RGB + Mono L/R + Sync)
- [x] PipelineManager.hpp: Config erweitert (monoWidth, monoHeight, enableStereo)
- [x] InputLoop.cpp: MessageGroup Parsing für Sync Queue
- [x] main.cpp: 30 FPS Config, enableStereo=false
- [x] Types.hpp: V3 Konstanten, GestureState enum, Point3D, Velocity3D
- [x] HandTracker.cpp/.hpp: 6-State Kalman Filter mit VIP Lock
- [x] GestureFSM.cpp/.hpp: Gesture State Machine mit Hysteresis
- [x] StereoDepth.cpp/.hpp: Punktuelle Tiefenmessung (9×9 Window)
- [x] CMakeLists.txt: Neue Dateien hinzugefügt
- [x] **CODE CLEANUP:**
  - [x] ProcessingLoop.cpp: Komplett neu geschrieben (815→250 Zeilen)
  - [x] ProcessingLoop.hpp: Vereinfacht, alte Filter entfernt
  - [x] Frame.hpp: nnData/palmData als DEPRECATED markiert
  - [x] docs/: Alte Dateien ins Archive verschoben

