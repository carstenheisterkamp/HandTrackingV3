# TODO: V3 3D Hand Controller Implementation

> **Aktuelle Phase:** Phase 4 - Lite vs. Full Modelle Testing 🧪
> **Letztes Update:** 2026-01-12
> **Status:** 2D Tracking ✅ | 3D Code ✅ | Session FSM ✅ | Lite ONNX ✅

---

## 📋 TFLite → ONNX Konvertierung (Reference)

**Funktionierende Kombination (Jetson aarch64, Python 3.10) - 2026-01-12:**

```bash
# Isolierte temporäre venv in /tmp (keine System-Änderungen)
cd /tmp
python3 -m venv convert_env
source /tmp/convert_env/bin/activate
pip install --upgrade pip

# Bewährte, kompatible Versionen (WICHTIG: Diese Kombination ist getestet!)
pip install numpy==1.23.5
pip install protobuf==3.20.3
pip install tensorflow==2.12.0
pip install tf2onnx==1.13.0
pip install onnx==1.14.0

# Konvertierung (Lite TFLite → ONNX)
python3 -m tf2onnx.convert \
  --tflite /home/nvidia/dev/HandTrackingV3/models/palm_detection_lite.tflite \
  --output /home/nvidia/dev/HandTrackingV3/models/palm_detection.onnx \
  --opset 13

python3 -m tf2onnx.convert \
  --tflite /home/nvidia/dev/HandTrackingV3/models/hand_landmark_lite.tflite \
  --output /home/nvidia/dev/HandTrackingV3/models/hand_landmark.onnx \
  --opset 13

# Aufräumen
deactivate
rm -rf /tmp/convert_env

# Verifizieren
ls -lh /home/nvidia/dev/HandTrackingV3/models/*.onnx
```

**Wichtige Hinweise:**
- TensorFlow 2.12.0 ist kompatibel mit protobuf 3.20.3
- tf2onnx 1.13.0 + onnx 1.14.0 arbeiten mit protobuf 3.20.3
- Diese Versionen sind aufeinander abgestimmt (keine Dependency-Konflikte)
- Beim nächsten Service-Start baut TensorRT automatisch `.engine` aus den ONNX-Files
- Konvertierung in isolierter /tmp-venv verhindert System-Konflikte

**Verfügbare Modelle:**
- `palm_detection.onnx` (Lite) + `hand_landmark.onnx` (Lite)
- `palm_detection_full.onnx` (Full) + `hand_landmark_full.onnx` (Full)

---

## 🎯 Aktuelle Aufgabe: 1536×864 Balanced Quality Test

**Test-Ergebnisse 1920×1080 (2026-01-11):**
- ❌ FPS: 12.5 (zu niedrig, nicht nutzbar)
- ✅ Erkennung: Besser bei Faust & starken Winkeln
- 📊 Resultat: Hand-Größe gut, aber TensorRT Bottleneck

**Aktuell (1536×864):**
- ✅ Resolution: 1536×864 (Balanced: +140% vs. 640×360)
- ✅ FPS Ziel: 25-30 FPS
- ✅ Hand-Größe @ 2m: ~240px (deutlich besser als 640×360)
- ✅ Face Margin: 5% (optimiert)
- ✅ Y-Achse: Raw camera coords (nicht invertiert)
- ✅ OSC Ziel: 169.254.1.100:9000

**TODO:**
- ⬜ **TEST auf Jetson:** FPS bei 1536×864 messen
- ⬜ Falls FPS <25: Fallback auf 1280×720
- ⬜ Hand-Tracking-Stabilität bei 2m Distance testen
- ⬜ Face-Anchored Tracking validieren

**Nächster Schritt:** Build & Test 1536×864 🎥

---

## 📅 Development Roadmap

### ✅ Phase 1: Sensor-Only Pipeline (Abgeschlossen)
- RGB NV12 @ 30 FPS (adaptive resolution)
- Mono L/R 640×400 GRAY8 @ 30 FPS
- Sync Node für synchronisierte Streams
- **Ergebnis:** Stabile 30 FPS auf Jetson

### ✅ Phase 2: TensorRT Inference (Abgeschlossen)
- Palm Detection TensorRT Engine (.onnx → .engine)
- Hand Landmark TensorRT Engine
- 2-Hand Tracking mit Kalman Filter
- MCP+Angle Gestenerkennung (13 Gesten)
- Haar Cascade Face Filter (optimiert: 5% margin)
- **Ergebnis:** 25-30 FPS mit beiden Händen @ 1280×720

### 🧪 Phase 3: Stereo Depth (Code Ready, Testing Blocked)
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
| Session FSM (IDLE/ACTIVE/LOST) | ✅ Implementiert | Per-Hand State Machine + Transitions |
| OSC Events (/player/*) | ✅ Implementiert | enter/active/lost/exit Events |
| Multi-Person Ignoring | ⬜ | Ignoriere Personen außerhalb Volume |

**Implementiert (2026-01-11):**
- ✅ SessionFSM Klasse mit 3 States (IDLE/ACTIVE/LOST)
- ✅ Stabile Frames: 15 für IDLE→ACTIVE, 3 für LOST→IDLE
- ✅ OSC Events: /player/session/{enter,active,lost,exit}
- ✅ Per-Hand Tracking (Hand 0 + Hand 1)
- ✅ State Transition Logging

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

# TODO (Stand: 2026-01-10)

## Status & Phasen
- [x] Phase 2: 2D Hand-Tracking (Palm + Landmarks, 2 Hände, Preview)
- [ ] Phase 3: Stereo Depth (Palm-Z, Volume-3D Filter) – Tests offen
- [x] Phase 4: Player Lock System – Session FSM ✅ | Face-Anchoring ⬜ | Multi-Person ⬜

## Aktuell in Arbeit
- [x] OSC: Alle Koordinaten normalisiert (X/Y/Z → 0-1)
- [x] OSC: Y-Achse invertiert (0=oben, 1=unten) für Unreal Engine
- [x] Preview: Gespiegelt (nur Kamerabild/Boxes/Skelette), Overlay lesbar
- [x] Face-Filter: Haar Cascade aktiv – False Positives im Gesicht unterdrückt
- [x] Gesten: MCP-basierte Erkennung mit Angle-Fallback
- [x] Zwei Hände stabil erkannt und getrackt
- [ ] Stereo Z-Validierung: Testen bei bekannten Distanzen (0.5m, 1m, 1.5m, 2m, 2.5m)
- [ ] Volume-Filtering (3D): Outside-Volume verwerfen, 2 Hände priorisieren (First-Come-First-Serve)
- [ ] OSC: Deltas werden gesendet (Verifizierung im Client-Log)
- [ ] Models: FULL aktiv? Overlay zeigt korrekten Typ (prüfen)

## Nächste Schritte
1. StereoDepth Tests (Palm-Z): Messreihe mit Markierungen am Boden
2. Play Volume: 16:9 Volumen finalisieren und im Overlay anzeigen (bereits aktiv)
3. Player Lock: Implementieren gemäß `PLAYER_LOCK_DESIGN.md`
4. Gesten-Tuning:
   - FIVE vs FOUR (Daumen-Schwellen)
   - FIST vs THUMB_UP (Winkel-Fallback)
   - MIDDLE_FINGER - Robustheit erhöhen
5. Optional: One-Euro Filter (Client-seitig empfohlen; Server-seitig als Option aufnehmen)

## Qualität & Regeln
- Immer erst TESTEN, dann commit/push
- Keine Architekturänderungen ohne explizite Freigabe
- Niemals ungefragt revertieren – erst Ursache finden und fixen
- Clang-Tidy aktiv (performance/readability), `-Wall -Wextra -Werror` auf Jetson

## Backlog
- Service-Resilienz: Gegen LAN/Kamera-Verlust robust machen
- OSC: Session-Events (Spawn/Despawn) bei Betreten/Verlassen des Volumens
- MJPEG: Farbartefakte beobachten (aktuell unkritisch)
- Gesten: Zweihändige dynamische Gesten (geplant)
