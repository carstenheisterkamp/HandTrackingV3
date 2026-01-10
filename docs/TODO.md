# TODO: V3 3D Hand Controller Implementation

> **Aktuelle Phase:** Phase 3 - Stereo Depth Testing 🧪
> **Letztes Update:** 2026-01-10
> **Status:** 2D Tracking ✅ | 3D Code ✅ | Testing ⬜

---

## 🎯 Aktuelle Aufgabe: Phase 3 Testing

**Implementiert (2026-01-10):**
- ✅ Stereo Depth am Palm Center
- ✅ Z-Koordinate in OSC Output
- ✅ Debug Overlay mit Volume (16:9), Delta, Z-Werten
- ✅ Gesten-Thresholds optimiert (FIVE/FIST verbessert)
- ✅ FULL Models aktiviert (bessere Accuracy @ 30 FPS)
- ✅ Preview gespiegelt (Mirror-View, Text lesbar)
- ✅ OSC Dokumentation konsolidiert (nur Unreal Engine C++)

**TODO:**
- ⬜ **TEST auf Jetson:** Tiefenwerte bei 50cm, 100cm, 150cm verifizieren
- ⬜ **TEST:** Gesten-Erkennung (FIVE vs FOUR, FIST bei 2 Händen)
- ⬜ Bei Bedarf: Device-Kalibrierung laden (statt Default)
- ⬜ Bei Bedarf: Rectification Maps für bessere Stereo-Accuracy

**Nächster Schritt:** Testen wenn Kamera verfügbar 🎥

---

## 📅 Development Roadmap

### ✅ Phase 1: Sensor-Only Pipeline (Abgeschlossen)
- RGB 640×360 NV12 @ 30 FPS
- Mono L/R 640×400 GRAY8 @ 30 FPS
- Sync Node für synchronisierte Streams
- **Ergebnis:** Stabile 30 FPS auf Jetson

### ✅ Phase 2: TensorRT Inference (Abgeschlossen)
- Palm Detection TensorRT Engine (.onnx → .engine)
- Hand Landmark TensorRT Engine
- 2-Hand Tracking mit Kalman Filter
- MCP+Angle Gestenerkennung (13 Gesten)
- Haar Cascade Face Filter (0 False Positives)
- **Ergebnis:** 25-30 FPS mit beiden Händen

### 🧪 Phase 3: Stereo Depth (Testing)
**Implementiert (2026-01-10):**

| Komponente | Status | Details |
|------------|--------|---------|
| Pipeline: Mono L/R | ✅ | enableStereo=true aktiviert |
| StereoDepth Class | ✅ | Punktuelle Messung am Palm |
| Z-Koordinate Output | ✅ | In OSC /hand/{id}/palm [x,y,z] |
| Debug Overlay | ✅ | Volume (16:9) + Delta + Z-Werte |
| Gesten-Optimierung | ✅ | FIVE/FIST Thresholds verbessert (5% statt 10%) |
| Model Testing | ✅ | FULL Models @ 30 FPS (besser als LITE) |
| Preview Mirror | ✅ | Kamera gespiegelt, Text lesbar |
| Bounding Box Text | ✅ | Text horizontal gespiegelt für Lesbarkeit |

**Ausstehend:**
- ⬜ Testen bei 50cm, 100cm, 150cm (remote dev blockiert)
- ⬜ Optional: Device-Kalibrierung laden
- ⬜ Optional: Rectification Maps

**Ergebnis:** Code komplett, wartet auf Testing vor Ort

### 📋 Phase 4: Player Lock System (In Progress 🚧)
**Ziel:** Stabiles Single-User Gaming

**Design:** `PLAYER_LOCK_DESIGN.md` ✅

| Komponente | Status | Details |
|------------|--------|---------|
| 3D Play Volume (16:9) | ✅ Implementiert | Preview + Filtering aktiv |
| Volume Filtering Logic | ✅ Implementiert | 2D Filter vor Landmark Inference |
| Face-Anchored Tracking | ⬜ | Haar Cascade Hand-zu-Gesicht |
| Session FSM (IDLE/ACTIVE/LOST) | ⬜ | State Machine für Player Session |
| OSC Events (/player/*) | ⬜ | enter/active/lost/exit Events |
| Multi-Person Ignoring | ⬜ | Ignoriere Personen außerhalb Volume |

**Implementiert (2026-01-10):**
- ✅ PlayVolume Klasse mit 16:9 Aspect Ratio
- ✅ 2D Volume-Filtering vor Landmark Inference (Performance-Optimierung)
- ✅ Debug-Visualisierung: Rejected palms (rote Kreise + "OUT" Label)
- ✅ Volume Status im Preview: "PLAY VOLUME (16:9) - ACTIVE"
- ✅ Filtering-Stats im Log

**Priorität:** Aktiv in Entwicklung

### 📋 Phase 5: Dynamische Gesten
**Ziel:** Velocity-basierte Gesten

| Geste | Trigger | Status |
|-------|---------|--------|
| SWIPE_LEFT/RIGHT | \|vx\| > 0.4 | ⬜ |
| SWIPE_UP/DOWN | \|vy\| > 0.4 | ⬜ |
| PUSH | vz > 0.3 | ⬜ |
| PUNCH | FIST + vz > 0.4 | ⬜ |

**Voraussetzung:** Phase 3 (Velocity.vz verfügbar)

---

## 📋 Backlog (Optional Features)

### 🎛️ One-Euro Filter
**Wann:** Falls Kalman Filter bei schnellen Richtungswechseln laggt
- Adaptive Cutoff-Frequenz basierend auf Velocity
- Bessere Reaktion für schnelle Gaming-Bewegungen
- **Referenz:** http://cristal.univ-lille.fr/~casiez/1euro/

### 🔌 Service Resilience
**Wann:** Für Production-Umgebungen
- Automatische OAK-D Reconnect bei Disconnect
- Watchdog für Device-Health
- Graceful Degradation bei Netzwerkproblemen

### 🎨 Advanced Debug Features
- Z-Depth Heatmap im Preview
- Landmark IDs als Nummern anzeigen
- Performance-Graphen (FPS über Zeit)

---

## 📝 Quick Reference

### Implementierte Features (Stand 2026-01-10)
- ✅ 2-Hand Tracking (max. 2 Hände gleichzeitig)
- ✅ 13 Statische Gesten (FIST, FIVE, PEACE, METAL, etc.)
- ✅ 3D Position mit Stereo Depth (x, y, z)
- ✅ Kalman Filter (Position + Velocity + Delta)
- ✅ Haar Cascade Face Filter (0 False Positives)
- ✅ OSC Output @ 30 Hz non-blocking
- ✅ MJPEG Debug Preview mit Play Volume
- ✅ 25-30 FPS stabil auf Jetson Orin Nano

### Konstanten
```cpp
CAMERA_FPS = 30
RGB_PREVIEW = 640×360 NV12
MONO_STEREO = 640×400 GRAY8
OSC_RATE = 30 Hz
DEBOUNCE = 3 frames (~100ms)
```

---

## ⚠️ Bekannte Issues

1. **Gesten-Erkennung:**
   - FIVE wird manchmal als FOUR erkannt → Thresholds optimiert (2026-01-10)
   - FIST bei 2 Händen inkonsistent → Curl-Check hinzugefügt (2026-01-10)
   - **Status:** Verbesserungen implementiert, Testing ausstehend

2. **Stereo Depth:**
   - Nutzt Default-Kalibrierung (75mm Baseline)
   - Keine Rectification Maps (kann Accuracy reduzieren)
   - **Status:** Funktioniert, aber ungetestet bei bekannten Abständen

3. **OAK-D PoE Reconnect:**
   - Service verbindet sich manchmal nicht nach Neustart
   - **Workaround:** Jetson neu starten oder `scripts/fix_oak_reconnect.sh`

---

## ✅ Erledigte Aufgaben (Archiv)

### 🎉 Meilenstein: 2026-01-10 - Phase 3 Code Complete
**Stereo Depth + Overlay Improvements**
- [x] enableStereo=true aktiviert
- [x] Z-Koordinate in OSC Output
- [x] Debug Overlay: Play Volume Box
- [x] Debug Overlay: Delta/Acceleration Display
- [x] Debug Overlay: Persistente Hand-Slots (kein Flickering)
- [x] Gesten-Thresholds optimiert (5% statt 10%)
- [x] FIST Curl-Check hinzugefügt

### 🎉 Meilenstein: 2026-01-09 - Phase 2 Complete
**2D Hand Tracking Fully Functional**
- [x] 2-Hand Detection mit NMS
- [x] TensorRT Palm + Landmark
- [x] Kalman Filter [x,y,z,vx,vy,vz]
- [x] 13 Gesten (Y-basiert + MCP-Angle Fallback)
- [x] Haar Cascade Face Filter
- [x] OSC Non-Blocking Output
- [x] 25-30 FPS stabil

### Phase 2 Sub-Tasks (2026-01-06 bis 2026-01-09)
- [x] TensorRT Engine Wrapper
- [x] Palm Detection TensorRT
- [x] Hand Landmark TensorRT
- [x] TFLite → ONNX Conversion
- [x] NV12 → RGB Preprocessing (CUDA/NPP)
- [x] Multi-Hand Support (detectAll + nmsMulti)
- [x] HandTracker + GestureFSM Integration
- [x] Gesture Recognition (MCP+Angle)
- [x] False Positive Filter (Haar Cascade)
- [x] OSC Integration (/hand/{id}/*)
- [x] MJPEG Debug Preview

### Phase 1: Sensor-Only Pipeline (2026-01-05)
- [x] PipelineManager: RGB + Mono L/R
- [x] Sync Node für synchronized streams
- [x] InputLoop: MessageGroup parsing
- [x] Types.hpp: V3 Konstanten
- [x] 30 FPS auf Jetson verifiziert

### Initial Setup
- [x] OPTIMAL_WORKFLOW_V3.md erstellt
- [x] TODO.md erstellt
- [x] CMakeLists.txt angepasst
- [x] Code Cleanup (ProcessingLoop 815→250 Zeilen)

