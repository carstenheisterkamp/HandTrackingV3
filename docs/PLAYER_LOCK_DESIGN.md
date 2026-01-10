# Player Lock System Design (Phase 4)

> **Ziel:** Stabiles Single-User-Tracking für Gaming trotz mehrerer Personen im Bild
> 
> **Status:** Design-Phase (nach Phase 3 - Stereo Depth)

---

## 🎯 Problem

**Aktuell (Phase 3):**
- System trackt Top-2 Hände nach Detection-Score
- Score wechselt bei Bewegung/Okklusion
- **Resultat:** Hand-IDs können zwischen Personen springen

**Gewünscht für Gaming:**
- **Stabile Session:** 1 Spieler = 2 Hände, unabhängig von anderen Personen
- **First-Come-First-Serve:** Erste Person im "Play Volume" wird Owner
- **Kein Flickering:** Hand-IDs bleiben stabil bis Session-Ende

---

## 🏗 Architektur: 3-Layer System

```
┌─────────────────────────────────────────────────────────────────┐
│                    Layer 1: Volume Filter                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  3D Play Volume (konfigurierbar, 16:9 format)            │  │
│  │    X: [0.1, 0.9] (80% horizontal, symmetrisch)           │  │
│  │    Y: [0.1, 0.9] (80% vertikal, symmetrisch)             │  │
│  │    Z: [0.5m, 2.5m] (50cm-2.5m from camera)               │  │
│  │                                                            │  │
│  │  → Filtert alle Detections außerhalb Play Volume          │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Layer 2: Player Detection                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Face Detector (Haar Cascade)                             │  │
│  │    → Detektiert Gesichter im Volume                       │  │
│  │    → Jedes Gesicht = potenzieller Player                  │  │
│  │                                                            │  │
│  │  Hand-to-Face Association                                 │  │
│  │    → Ordne Hände dem nächsten Gesicht zu                  │  │
│  │    → Max Distance: 0.4 normalized units horizontal        │  │
│  │    → Preference: Hände links/rechts vom Gesicht           │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Layer 3: Session Manager                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Player Session FSM                                        │  │
│  │    States: IDLE → CALIBRATING → ACTIVE → LOST             │  │
│  │                                                            │  │
│  │  IDLE:                                                     │  │
│  │    • Warte auf Player im Volume mit 2 Händen + Gesicht    │  │
│  │                                                            │  │
│  │  CALIBRATING (1-2 Sekunden):                              │  │
│  │    • Player erkannt, warte auf stabile Tracking           │  │
│  │    • Sende: /player/calibrating [progress: 0.0-1.0]       │  │
│  │                                                            │  │
│  │  ACTIVE:                                                   │  │
│  │    • Player ist "Owner", Hand-IDs sind locked             │  │
│  │    • Ignoriere andere Hände/Gesichter                     │  │
│  │    • Sende: /player/active [player_id: 0]                 │  │
│  │    • Normale Hand-Tracking OSC Messages                   │  │
│  │                                                            │  │
│  │  LOST (Grace Period 3 Sekunden):                          │  │
│  │    • Player temporär nicht erkannt                        │  │
│  │    • Warte auf Rückkehr ins Volume                        │  │
│  │    • Sende: /player/lost [time_remaining: 3.0-0.0]        │  │
│  │    • Nach Timeout → IDLE                                  │  │
│  │    • Sende: /player/exit                                  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📐 Play Volume Definition

### Konfigurierbare Parameter:

```cpp
struct PlayVolume {
    // Normalized coordinates (0-1)
    // 16:9 aspect ratio to match camera (640x360) and game format
    float minX = 0.1f;   // 10% margin left
    float maxX = 0.9f;   // 10% margin right
    
    float minY = 0.1f;   // 10% margin top
    float maxY = 0.9f;   // 10% margin bottom
    
    // Absolute depth (mm)
    float minZ = 500.0f;   // 50cm minimum
    float maxZ = 2500.0f;  // 2.5m maximum
    
    // Face requirement
    bool requireFace = true;
    
    // Calibration time
    float calibrationTime = 2.0f;  // seconds
    
    // Grace period after lost
    float lostTimeout = 3.0f;  // seconds
};
```

### Warum dieses Volume?

- **X: [0.1, 0.9]** - 80% horizontal, symmetrisch (16:9 Format)
- **Y: [0.1, 0.9]** - 80% vertikal, symmetrisch (16:9 Format)
- **Aspect Ratio:** Matches camera (640×360 = 16:9) and typical game viewport
- **Z: [0.5m, 2.5m]** - Optimal für Stereo-Accuracy + Spielbereich
- **Face Required:** Verhindert falsche Hand-Detections ohne Person

---

## 🔄 State Machine: Player Session

```
     ┌─────────────────────────────────────────┐
     │             IDLE                        │
     │  • No player locked                     │
     │  • Scan for candidates                  │
     └─────────────────────────────────────────┘
                    │
                    │ Player enters volume
                    │ (Face + 2 Hands detected)
                    ▼
     ┌─────────────────────────────────────────┐
     │         CALIBRATING                     │
     │  • Player found, wait for stability     │
     │  • Require N consecutive frames         │
     │  • OSC: /player/calibrating [progress]  │
     └─────────────────────────────────────────┘
                    │
                    │ N frames stable
                    ▼
     ┌─────────────────────────────────────────┐
     │            ACTIVE                       │
     │  • Player is "Owner"                    │
     │  • Hand IDs locked to this player       │
     │  • Ignore other hands/faces             │
     │  • OSC: /player/active                  │
     │  • OSC: /hand/{0,1}/...                 │
     └─────────────────────────────────────────┘
                    │
                    │ Player lost (out of volume)
                    ▼
     ┌─────────────────────────────────────────┐
     │             LOST                        │
     │  • Grace period (3s)                    │
     │  • Wait for player return               │
     │  • OSC: /player/lost [time_remaining]   │
     └─────────────────────────────────────────┘
                    │
       ┌────────────┴────────────┐
       │                         │
       │ Returns                 │ Timeout
       ▼                         ▼
    ACTIVE                     IDLE
                              (OSC: /player/exit)
```

---

## 🖐 Hand-to-Player Association

### Algorithmus:

```cpp
struct Player {
    cv::Rect faceRect;        // From Haar Cascade
    Point3D faceCenter3D;     // With depth
    
    int leftHandId = -1;      // Track which detection is left hand
    int rightHandId = -1;     // Track which detection is right hand
    
    Point3D leftHandPos;
    Point3D rightHandPos;
    
    SessionState state;
    float stateTimer;
};

// Für jedes Frame:
void associateHandsToPlayer(Player& player, 
                            const std::vector<Detection>& hands,
                            const std::vector<cv::Rect>& faces) {
    
    // 1. Find face in volume
    for (const auto& face : faces) {
        if (isInVolume(face)) {
            player.faceRect = face;
            break;
        }
    }
    
    // 2. Find hands near this face
    std::vector<Detection> candidateHands;
    for (const auto& hand : hands) {
        if (isInVolume(hand)) {
            float distToFace = distance2D(hand.center, player.faceRect.center);
            if (distToFace < MAX_HAND_FACE_DISTANCE) {
                candidateHands.push_back(hand);
            }
        }
    }
    
    // 3. Assign left/right based on X position relative to face
    if (candidateHands.size() >= 2) {
        std::sort(candidateHands.begin(), candidateHands.end(),
            [](const auto& a, const auto& b) { return a.x < b.x; });
        
        player.leftHandId = candidateHands[0].trackId;   // Leftmost
        player.rightHandId = candidateHands[1].trackId;  // Rightmost
    }
}
```

---

## 📡 OSC Protocol Extensions

### Neue Player-Events:

| OSC Address | Type | Description | When |
|-------------|------|-------------|------|
| `/player/enter` | int | player_id (always 0) | IDLE → CALIBRATING |
| `/player/calibrating` | float | progress (0.0-1.0) | During CALIBRATING |
| `/player/active` | int | player_id | CALIBRATING → ACTIVE |
| `/player/lost` | float | time_remaining (3.0-0.0) | ACTIVE → LOST |
| `/player/exit` | int | player_id | LOST → IDLE |
| `/player/volume` | [6 floats] | [minX, maxX, minY, maxY, minZ, maxZ] | On config change |

### Modified Hand Messages:

**Während ACTIVE State:**
```
/hand/0/palm [x, y, z]           # Immer left hand des locked players
/hand/1/palm [x, y, z]           # Immer right hand des locked players
/hand/0/velocity [vx, vy, vz]
/hand/1/velocity [vx, vy, vz]
/hand/0/gesture [state, conf, name]
/hand/1/gesture [state, conf, name]
```

**Während IDLE/CALIBRATING/LOST:**
- Keine `/hand/...` Messages (oder optional mit `available=0` Flag)

---

## 🎮 Unreal Engine Integration

### Blueprint Beispiel:

```cpp
// Event: /player/enter
void OnPlayerEnter(int playerId) {
    // Spawn Player Avatar/UI
    ShowCalibrationOverlay();
}

// Event: /player/calibrating
void OnPlayerCalibrating(float progress) {
    UpdateCalibrationProgress(progress);
    if (progress >= 1.0f) {
        HideCalibrationOverlay();
    }
}

// Event: /player/active
void OnPlayerActive(int playerId) {
    EnableGameplay();
    SpawnPlayerCursor();
}

// Event: /player/lost
void OnPlayerLost(float timeRemaining) {
    ShowWarning("Zurück ins Spielfeld! " + timeRemaining + "s");
}

// Event: /player/exit
void OnPlayerExit(int playerId) {
    DisableGameplay();
    DespawnPlayerCursor();
    ShowIdleScreen();
}
```

---

## ⚡ Performance Impact Analysis

### Overhead pro Frame:

| Komponente | CPU Zeit | GPU Zeit | Cache | Gesamt |
|------------|----------|----------|-------|--------|
| **Face Detection (Haar Cascade)** | 2-3ms | - | ✅ Alle 5 Frames | ~0.5ms avg |
| **Volume Check (Hände)** | <0.1ms | - | - | <0.1ms |
| **Volume Check (Gesicht)** | <0.05ms | - | - | <0.05ms |
| **Hand-to-Face Distance** | <0.1ms | - | - | <0.1ms |
| **Session State Update** | <0.05ms | - | - | <0.05ms |
| **GESAMT** | - | - | - | **~0.8ms** |

### FPS Impact:

**Aktuell (Phase 3):**
- Palm Detection: ~8ms
- Hand Landmark: ~7ms
- Stereo Depth: <1ms
- Kalman + Gesture: <0.5ms
- **Total: ~16.5ms → 60 FPS möglich**

**Mit Player Lock (Phase 4):**
- Palm Detection: ~8ms
- Hand Landmark: ~7ms
- Stereo Depth: <1ms
- **Player Lock: ~0.8ms** ← NEU
- Kalman + Gesture: <0.5ms
- **Total: ~17.3ms → noch immer 57+ FPS** ✅

### Optimierungen:

1. **Face Detection Caching:**
   - Nur alle 5 Frames (bei 30 FPS = alle 166ms)
   - Face bewegt sich langsamer als Hände
   - Spart 80% der Face-Detection Zeit

2. **Early Exit bei ACTIVE State:**
   - Wenn Player locked: ignoriere andere Detections sofort
   - Keine NMS für ignorierte Hände nötig
   - Spart ~0.2ms

3. **SIMD für Distance Checks:**
   - Hand-to-Face Distance kann mit SIMD optimiert werden
   - Arm NEON auf Jetson

**Fazit: <1ms Overhead, vernachlässigbar für 30 FPS Target** ✅

---

## 🎨 Debug Visualization (MJPEG Preview)

### Overlay-Elemente:

```cpp
void ProcessingLoop::drawDebugOverlay(cv::Mat& debugFrame, Frame* frame) {
    if (!_playerSession) return;
    
    // 1. Draw Play Volume (3D Box projected to 2D)
    drawPlayVolume(debugFrame);
    
    // 2. Draw Face Detection
    if (_playerSession->hasFace()) {
        auto faceRect = _playerSession->getFaceRect();
        cv::Scalar faceColor = _playerSession->isActive() 
            ? cv::Scalar(0, 255, 0)   // Green = Active Player
            : cv::Scalar(255, 255, 0); // Yellow = Calibrating
        
        cv::rectangle(debugFrame, faceRect, faceColor, 2);
        cv::putText(debugFrame, "FACE", 
            cv::Point(faceRect.x, faceRect.y - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, faceColor, 2);
    }
    
    // 3. Draw Hand-to-Face Connections
    if (_playerSession->isActive()) {
        auto faceCenter = _playerSession->getFaceCenter();
        
        for (int h = 0; h < 2; ++h) {
            if (_handStates[h].palmX > 0) {
                cv::Point handPos(
                    _handStates[h].palmX * debugFrame.cols,
                    _handStates[h].palmY * debugFrame.rows
                );
                
                // Line from face to hand
                cv::line(debugFrame, faceCenter, handPos, 
                    cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                
                // Hand label
                std::string label = h == 0 ? "L" : "R";
                cv::putText(debugFrame, label, handPos + cv::Point(10, -10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            }
        }
    }
    
    // 4. Draw Session State
    drawSessionState(debugFrame);
    
    // 5. Draw Volume Violations (if any)
    drawVolumeViolations(debugFrame);
}

void ProcessingLoop::drawPlayVolume(cv::Mat& frame) {
    const auto& vol = _playerSession->getVolume();
    
    // 2D Projection (X, Y)
    int x1 = vol.minX * frame.cols;
    int x2 = vol.maxX * frame.cols;
    int y1 = vol.minY * frame.rows;
    int y2 = vol.maxY * frame.rows;
    
    cv::Scalar color = _playerSession->isActive()
        ? cv::Scalar(0, 255, 0)      // Green = Active
        : cv::Scalar(100, 100, 100); // Gray = Idle
    
    // Draw rectangle
    cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), 
        color, 2, cv::LINE_AA);
    
    // Draw corner markers (3D effect)
    int markerSize = 20;
    // Top-left
    cv::line(frame, cv::Point(x1, y1), cv::Point(x1 + markerSize, y1), color, 3);
    cv::line(frame, cv::Point(x1, y1), cv::Point(x1, y1 + markerSize), color, 3);
    // Top-right
    cv::line(frame, cv::Point(x2, y1), cv::Point(x2 - markerSize, y1), color, 3);
    cv::line(frame, cv::Point(x2, y1), cv::Point(x2, y1 + markerSize), color, 3);
    // Bottom-left
    cv::line(frame, cv::Point(x1, y2), cv::Point(x1 + markerSize, y2), color, 3);
    cv::line(frame, cv::Point(x1, y2), cv::Point(x1, y2 - markerSize), color, 3);
    // Bottom-right
    cv::line(frame, cv::Point(x2, y2), cv::Point(x2 - markerSize, y2), color, 3);
    cv::line(frame, cv::Point(x2, y2), cv::Point(x2, y2 - markerSize), color, 3);
    
    // Z-Depth indication (text)
    char depthText[64];
    snprintf(depthText, sizeof(depthText), 
        "Z: %.1fm - %.1fm", vol.minZ / 1000.0f, vol.maxZ / 1000.0f);
    cv::putText(frame, depthText, 
        cv::Point(x1 + 10, y1 + 25),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
}

void ProcessingLoop::drawSessionState(cv::Mat& frame) {
    if (!_playerSession) return;
    
    auto state = _playerSession->getState();
    std::string stateText;
    cv::Scalar stateColor;
    
    switch (state) {
        case SessionState::IDLE:
            stateText = "IDLE - Waiting for player...";
            stateColor = cv::Scalar(128, 128, 128);
            break;
        case SessionState::CALIBRATING:
            {
                float progress = _playerSession->getCalibrationProgress();
                char buf[64];
                snprintf(buf, sizeof(buf), "CALIBRATING... %.0f%%", progress * 100);
                stateText = buf;
                stateColor = cv::Scalar(0, 255, 255); // Yellow
            }
            break;
        case SessionState::ACTIVE:
            stateText = "ACTIVE - Player locked";
            stateColor = cv::Scalar(0, 255, 0); // Green
            break;
        case SessionState::LOST:
            {
                float remaining = _playerSession->getLostTimeRemaining();
                char buf[64];
                snprintf(buf, sizeof(buf), "LOST - Return in %.1fs", remaining);
                stateText = buf;
                stateColor = cv::Scalar(0, 0, 255); // Red
            }
            break;
    }
    
    // Draw banner at top
    cv::rectangle(frame, cv::Point(0, 0), cv::Point(frame.cols, 40),
        cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(frame, stateText, cv::Point(10, 25),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, stateColor, 2, cv::LINE_AA);
}

void ProcessingLoop::drawVolumeViolations(cv::Mat& frame) {
    // Draw hands that are OUTSIDE volume in red
    for (const auto& hand : _rejectedHands) {
        cv::Scalar color(0, 0, 255); // Red
        cv::rectangle(frame, hand.bbox, color, 2);
        cv::putText(frame, "OUT OF VOLUME", 
            cv::Point(hand.bbox.x, hand.bbox.y - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
    }
    
    // Draw faces that are OUTSIDE volume
    for (const auto& face : _rejectedFaces) {
        cv::Scalar color(255, 0, 255); // Magenta
        cv::rectangle(frame, face, color, 1, cv::LINE_AA);
        cv::putText(frame, "IGNORED", 
            cv::Point(face.x, face.y - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
    }
}
```

### Visualisierungs-Modi (Config):

```cpp
struct DebugVisualization {
    bool showPlayVolume = true;        // 3D Box overlay
    bool showFaceDetection = true;     // Face rectangles
    bool showHandToFaceLinks = true;   // Lines connecting hands to face
    bool showSessionState = true;      // Banner with state
    bool showVolumeViolations = true;  // Rejected detections
    bool showDepthHeatmap = false;     // Z-Depth color coding
    bool showLandmarkIDs = false;      // Number labels on keypoints
};
```

### Screenshot-Beispiele (Mockup):

**IDLE State:**
```
┌─────────────────────────────────────────┐
│ IDLE - Waiting for player...           │ ← Gray banner
├─────────────────────────────────────────┤
│                                         │
│     ┌───────────────────┐               │
│     │                   │ ← Gray volume │
│     │   PLAY VOLUME     │    box        │
│     │   Z: 0.5m - 2.5m  │               │
│     │                   │               │
│     └───────────────────┘               │
│                                         │
│  [Person außerhalb]                     │
│  └─ Magenta Box: "IGNORED"              │
│                                         │
└─────────────────────────────────────────┘
```

**CALIBRATING State:**
```
┌─────────────────────────────────────────┐
│ CALIBRATING... 67%                      │ ← Yellow banner
├─────────────────────────────────────────┤
│                                         │
│     ┌───────────────────┐               │
│     │   ┌─────┐         │ ← Green vol  │
│     │   │FACE │         │    Green face│
│     │   └─────┘         │               │
│     │     ╱   ╲         │               │
│     │   🖐L   R🖐       │ ← Hands      │
│     │                   │    with lines│
│     └───────────────────┘               │
│                                         │
└─────────────────────────────────────────┘
```

**ACTIVE State:**
```
┌─────────────────────────────────────────┐
│ ACTIVE - Player locked ✓                │ ← Green banner
├─────────────────────────────────────────┤
│                                         │
│     ┌───────────────────┐               │
│     │   ┌─────┐         │ ← Green vol  │
│     │   │FACE │         │    Everything│
│     │   └─────┘         │    green     │
│     │     ╱   ╲         │               │
│     │   🖐L   R🖐       │ ← Stable IDs │
│     │                   │               │
│     └───────────────────┘               │
│                                         │
│  [Person außerhalb - ignoriert]         │
│  └─ Kein Overlay (komplett ignoriert)   │
│                                         │
└─────────────────────────────────────────┘
```

**LOST State:**
```
┌─────────────────────────────────────────┐
│ LOST - Return in 2.3s                   │ ← Red banner
├─────────────────────────────────────────┤
│                                         │
│     ┌───────────────────┐               │
│     │                   │ ← Green vol  │
│     │   [Player left]   │    but empty │
│     │                   │               │
│     │                   │               │
│     └───────────────────┘               │
│                                         │
└─────────────────────────────────────────┘
```

---

## 🔧 Implementation Roadmap (Updated mit Visualization)

### Phase 4A: Volume Filter & Session Manager

**Neue Dateien:**
- `include/core/PlayVolume.hpp` - Volume definition & checking
- `include/core/PlayerSession.hpp` - Session FSM
- `src/core/PlayerSession.cpp` - State machine logic

**Integration:**
- ProcessingLoop: Volume-Filter vor Hand-Processing
- ProcessingLoop: Session state check vor OSC send
- **ProcessingLoop: Debug Visualization** ← NEU

**Geschätzter Aufwand:** 2-3 Tage

### Phase 4B: Hand-to-Face Association

**Ergänzungen:**
- PalmDetector: Track detection IDs frame-to-frame
- ProcessingLoop: Hand-Face distance calculation
- PlayerSession: Left/Right hand assignment

**Geschätzter Aufwand:** 1 Tag

### Phase 4C: OSC Session Events + Visualization Polish

**Ergänzungen:**
- OscSender: Neue `/player/*` Messages
- ProcessingLoop: Session state change events
- **Debug Overlay: Alle Visualisierungs-Modi** ← NEU
- **Config: Visualization flags** ← NEU

**Geschätzter Aufwand:** 1 Tag

**GESAMT: 4-5 Tage** (inkl. Visualization)

---

## 📊 Performance Budget (Final)

| Komponente | Zeit | % von 33ms @ 30 FPS |
|------------|------|---------------------|
| Palm Detection | 8ms | 24% |
| Hand Landmark | 7ms | 21% |
| Stereo Depth | 1ms | 3% |
| **Player Lock** | **0.8ms** | **2.4%** |
| Kalman + Gesture | 0.5ms | 1.5% |
| **Debug Overlay** | **0.5ms** | **1.5%** |
| OSC Send | 0.2ms | 0.6% |
| **GESAMT** | **18ms** | **54%** |

**Margin: 15ms (45%) für System-Overhead und Jitter** ✅

**Worst-Case (mit allen Overlays):**
- 18ms + 15ms Margin = 33ms → 30 FPS garantiert ✅
- Bei 60 FPS Target (16.6ms): 18ms → **55 FPS minimal** ✅

---

### Phase 4A: Volume Filter & Session Manager

**Neue Dateien:**
- `include/core/PlayVolume.hpp` - Volume definition & checking
- `include/core/PlayerSession.hpp` - Session FSM
- `src/core/PlayerSession.cpp` - State machine logic

**Integration:**
- ProcessingLoop: Volume-Filter vor Hand-Processing
- ProcessingLoop: Session state check vor OSC send

**Geschätzter Aufwand:** 1-2 Tage

### Phase 4B: Hand-to-Face Association

**Ergänzungen:**
- PalmDetector: Track detection IDs frame-to-frame
- ProcessingLoop: Hand-Face distance calculation
- PlayerSession: Left/Right hand assignment

**Geschätzter Aufwand:** 1 Tag

### Phase 4C: OSC Session Events

**Ergänzungen:**
- OscSender: Neue `/player/*` Messages
- ProcessingLoop: Session state change events

**Geschätzter Aufwand:** 0.5 Tage

---

## 🧪 Testing Plan

### Test 1: Single Player Stability
- **Setup:** 1 Person im Volume, 1 Person außerhalb
- **Expected:** Nur Person im Volume wird getrackt

### Test 2: First-Come-First-Serve
- **Setup:** Person A betritt Volume → ACTIVE, Person B betritt Volume
- **Expected:** Person A bleibt locked, Person B ignoriert

### Test 3: Grace Period
- **Setup:** Player verlässt Volume für 2s, kehrt zurück
- **Expected:** Session bleibt ACTIVE (kein exit)

### Test 4: Timeout
- **Setup:** Player verlässt Volume für >3s
- **Expected:** Session → IDLE, /player/exit gesendet

---

## 🎛 Config Example (JSON)

```json
{
  "player_lock": {
    "enabled": true,
    "play_volume": {
      "min_x": 0.2,
      "max_x": 0.8,
      "min_y": 0.1,
      "max_y": 0.9,
      "min_z_mm": 500,
      "max_z_mm": 2500
    },
    "face_required": true,
    "calibration_frames": 60,
    "lost_timeout_seconds": 3.0
  }
}
```

---

## ❓ Alternativen & Trade-offs

### Alternative 1: Nur Depth-basiert (ohne Face)
**Pro:** Einfacher, weniger Dependencies
**Contra:** Schwierig bei mehreren Personen auf gleicher Tiefe

### Alternative 2: Depth-Priorisierung (näheste 2 Hände)
**Pro:** Sehr einfach, kein Face-Detection nötig
**Contra:** Instabil bei mehreren Personen nahe beieinander

### Alternative 3: Hybrid (Face + Depth Priority)
**Pro:** Robust + Fallback wenn Face nicht erkannt
**Contra:** Komplexer

**Empfehlung:** Start mit Hybrid (Alternative 3), später Config-Flag für Depth-Only

---

## 🚀 Fazit

**Ist das zu kompliziert?**
→ **Nein!** Die Komplexität ist gerechtfertigt für ein stabiles Gaming-System.

**Vorteile:**
- ✅ Stabile Sessions ohne Flickering
- ✅ Intuitive First-Come-First-Serve Logik
- ✅ Game-Engine Integration via Events
- ✅ Konfigurierbar für verschiedene Spiele

**Implementierung:**
- 2-3 Tage Arbeit für Phase 4A-C
- Baut auf bestehender Infrastruktur auf
- Testbar in Isolation

**Nächster Schritt nach Phase 3:**
1. Phase 3 testen (Stereo Depth verifizieren)
2. Play Volume definieren (mit dir abstimmen)
3. Phase 4A implementieren (Volume Filter + Session FSM)

