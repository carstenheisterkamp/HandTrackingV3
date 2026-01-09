# OPTIMAL_WORKFLOW_V3: 3D Hand Controller Architecture

> **Kernphilosophie:** Wir bauen keinen Computer-Vision-Stack. Wir bauen einen **3D-Controller**.

---

## 🎯 Executive Summary

V3 ist eine radikale Vereinfachung: OAK-D wird zum reinen Sensor degradiert, alle Intelligenz wandert auf den Jetson.

| Komponente | V2 (Komplex) | V3 (Simpel) |
|------------|--------------|-------------|
| OAK-D | RGB + ObjectTracker + NN | **Nur Sensoren** (RGB + Mono L/R) |
| Jetson | TensorRT NNs | TensorRT NNs + Stereo + Tracking |
| XLink | Bidirektional (BBox-Rückkanal) | **Unidirektional** (nur Frames) |
| Depth | StereoDepth Node | **Punktuelle Tiefe** (Palm Center) |
| Tracking | ObjectTracker auf VPU | **Kalman Filter** auf CPU |

**Ergebnis:** Keine API-Lücken, keine BBox-Rückkanal-Probleme, volle Kontrolle.

---

## 🏗 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           OAK-D Pro PoE                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  Mono Left   │  │     RGB      │  │  Mono Right  │              │
│  │  THE_400_P   │  │   640×360    │  │  THE_400_P   │              │
│  │    @60fps    │  │    NV12      │  │    @60fps    │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                 │                 │                       │
│         │    Zero-Copy NV12 via dma-buf     │                       │
│         └─────────────────┼─────────────────┘                       │
│                           │                                         │
│                     ┌─────┴─────┐                                   │
│                     │   Sync    │                                   │
│                     │   Node    │                                   │
│                     └─────┬─────┘                                   │
│                           │                                         │
│                      XLink Out                                      │
└───────────────────────────┼─────────────────────────────────────────┘
                            │ GigE PoE (1 Gbps)
                            ▼
┌───────────────────────────────────────────────────────────────────────┐
│                        Jetson Orin Nano                               │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                      InputLoop Thread                            │ │
│  │  ┌──────────────────────────────────────────────────────────┐   │ │
│  │  │  Receive: RGB (NV12) + Mono L/R → Pinned Memory Pool     │   │ │
│  │  └──────────────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────┬───────────────────────────────────┘ │
│                                │                                     │
│                          SPSC Queue                                  │
│                                ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    ProcessingLoop Thread                         │ │
│  │                                                                   │ │
│  │  Step 1: Palm Detection (TensorRT)                               │ │
│  │      └─→ Full Frame 640×360, 1 Detection                        │ │
│  │                                                                   │ │
│  │  Step 2: Hand Landmark (TensorRT)                                │ │
│  │      └─→ ROI from Palm, 21 Landmarks (x,y,z_relative)           │ │
│  │                                                                   │ │
│  │  Step 3: Depth at Palm Center                                    │ │
│  │      └─→ Rectify Mono L/R → 9×9 Window → Median                 │ │
│  │      └─→ Palm 3D Position (Camera Coords)                        │ │
│  │                                                                   │ │
│  │  Step 4: Kalman Filter + Prediction                              │ │
│  │      └─→ State: [x, y, z, vx, vy, vz]                           │ │
│  │      └─→ Predict +1 Frame (Latency Compensation)                │ │
│  │                                                                   │ │
│  │  Step 5: Gesture FSM                                             │ │
│  │      └─→ States: Idle → Palm → Pinch/Grab/Point                 │ │
│  │      └─→ Hysteresis + Debounce                                   │ │
│  │                                                                   │ │
│  └─────────────────────────────┬───────────────────────────────────┘ │
│                                │                                     │
│                          SPSC Queue (Drop-Oldest >50ms)              │
│                                ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                      OSC Thread (30 Hz)                          │ │
│  │  └─→ /hand/palm [x, y, z]                                       │ │
│  │  └─→ /hand/landmarks [21 × 3 floats]                            │ │
│  │  └─→ /hand/gesture [state, confidence]                          │ │
│  │  └─→ /hand/velocity [vx, vy, vz]                                │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Implementation Steps

### Step 1: OAK-D Pipeline auf Sensor-Only reduzieren

**Ziel:** OAK-D als reiner Frame-Lieferant, keine NNs, kein Encoding, kein ObjectTracker.

```cpp
// Pipeline Configuration
auto monoLeft = pipeline.create<dai::node::Camera>()->build(
    dai::CameraBoardSocket::CAM_B,  // Left Mono
    dai::CameraResolution::THE_400_P,
    60  // FPS
);

auto monoRight = pipeline.create<dai::node::Camera>()->build(
    dai::CameraBoardSocket::CAM_C,  // Right Mono
    dai::CameraResolution::THE_400_P,
    60  // FPS
);

auto rgb = pipeline.create<dai::node::Camera>()->build(
    dai::CameraBoardSocket::CAM_A,  // RGB Center
    std::make_pair(1920, 1080),
    60
);

// RGB Preview: 640×360 NV12 (Zero-Copy friendly)
auto rgbPreview = rgb->requestOutput(
    std::make_pair(640, 360),
    dai::ImgFrame::Type::NV12,
    dai::ImgResizeMode::LETTERBOX,
    60
);

// Sync all three streams
auto sync = pipeline.create<dai::node::Sync>();
sync->setTimestampThreshold(std::chrono::milliseconds(10));
rgbPreview->link(sync->inputs["rgb"]);
monoLeft->requestOutput(...)->link(sync->inputs["monoLeft"]);
monoRight->requestOutput(...)->link(sync->inputs["monoRight"]);

// Single output queue
auto outputQueue = sync->out.createOutputQueue(4, false);
```

**Constraints:**
- Keine `NeuralNetwork` Nodes
- Keine `ImageManip` für NN-Preprocessing
- Keine `VideoEncoder` Nodes
- Mono L/R **MÜSSEN** erhalten bleiben für Stereo

---

### Step 2: Hand-NN immer Full Frame, 1 Person

**Ziel:** Einfache, robuste Hand-Detection ohne komplexes ROI-Management.

```cpp
// Palm Detection: Full Frame, Single Detection
class PalmDetector {
    nvinfer1::IExecutionContext* context_;
    
public:
    struct Detection {
        float x, y, w, h;      // Bounding Box (normalized)
        float confidence;
        float rotation;        // Hand rotation in radians
    };
    
    std::optional<Detection> detect(const Frame& frame) {
        // 1. Resize to NN input (e.g., 192×192 or 256×256)
        // 2. Run TensorRT inference
        // 3. NMS → Return single best detection (VIP)
        // 4. Return nullopt if confidence < threshold
    }
};

// Hand Landmark: ROI from Palm Detection
class HandLandmark {
public:
    struct Result {
        std::array<Point3D, 21> landmarks;  // 21 keypoints
        float handedness;                    // 0=Left, 1=Right
        float presence;                      // Confidence
    };
    
    std::optional<Result> infer(const Frame& frame, const PalmDetector::Detection& palm) {
        // 1. Extract ROI based on palm bbox + rotation
        // 2. Apply LETTERBOX padding
        // 3. Run TensorRT inference
        // 4. Unletterbox coordinates back to frame space
    }
};
```

**Key Decisions:**
- **1 Person Only:** Keine Multi-Hand-Komplexität
- **Immer Full Frame:** Palm Detector sieht immer das ganze Bild
- **NV12 Pfad:** Frame bleibt in NV12 bis zur GPU, keine CPU-Konvertierung
- **LETTERBOX:** Beibehalt für NN, Unletterbox auf Host

---

### Step 3: Depth nur am Palm Center

**Ziel:** Keine Full-Frame Depth Map, nur punktuelle Tiefe am Palm Center.

```cpp
class StereoDepth {
    // Rektifizierungs-Matrizen (einmal berechnet)
    cv::Mat R1_, R2_, P1_, P2_, Q_;
    cv::cuda::GpuMat mapL1_, mapL2_, mapR1_, mapR2_;
    
public:
    void init(const dai::CalibrationData& calib) {
        // Lade intrinsics + extrinsics aus OAK-D Kalibrierung
        auto K_left = calib.getCameraIntrinsics(CAM_B);
        auto K_right = calib.getCameraIntrinsics(CAM_C);
        auto R = calib.getExtrinsics(CAM_B, CAM_C).rotation;
        auto T = calib.getExtrinsics(CAM_B, CAM_C).translation;
        
        cv::stereoRectify(K_left, D_left, K_right, D_right, 
                          imageSize, R, T, R1_, R2_, P1_, P2_, Q_);
        
        cv::cuda::buildWarpAffine(...);  // Precompute remap tables
    }
    
    float getDepthAtPoint(const cv::cuda::GpuMat& monoL, 
                          const cv::cuda::GpuMat& monoR,
                          int px, int py) {
        // 1. Rektifiziere nur 9×9 Fenster um (px, py)
        // 2. Berechne Disparität für dieses Fenster
        // 3. Median/Biweight Filter für Robustheit
        // 4. Z = (baseline × focal_length) / disparity
        return depth_mm;
    }
};
```

**Algorithm:**
1. **Input:** 2D Palm Center (px, py) aus Landmark Detection
2. **Rectify:** Nur 9×9 Pixel-Fenster (nicht ganzes Bild!)
3. **Match:** Block Matching im lokalen Fenster
4. **Robust:** Median oder Biweight-Mittelwert
5. **Output:** Z in mm (Kamera-Koordinaten)

**Warum kein Full-Frame Stereo?**
- Full-Frame StereoSGBM: ~15-20ms auf Jetson GPU
- Punktuell (9×9): <1ms
- Wir brauchen nur Palm-Tiefe, nicht jedes Pixel

---

### Step 4: Kalman Filter + Prediction

**Ziel:** Glatte, prädiktive Trajektorien für Low-Latency Controller-Feeling.

```cpp
class HandTracker {
    // State: [x, y, z, vx, vy, vz]
    Eigen::Vector6f state_;
    Eigen::Matrix6f P_;  // Covariance
    
    // Process noise (tuned for hand motion)
    static constexpr float PROCESS_NOISE_POS = 10.0f;   // mm
    static constexpr float PROCESS_NOISE_VEL = 50.0f;   // mm/s
    
    // Measurement noise
    static constexpr float MEASUREMENT_NOISE = 5.0f;    // mm
    
    int consecutiveFrames_ = 0;
    bool vipLocked_ = false;
    
public:
    void predict(float dt) {
        // Constant velocity model
        // x_new = x + vx * dt
        state_[0] += state_[3] * dt;
        state_[1] += state_[4] * dt;
        state_[2] += state_[5] * dt;
        
        // Update covariance
        P_ += Q_ * dt;  // Process noise
    }
    
    void update(const Point3D& measurement) {
        // Standard Kalman update
        auto K = P_ * H_.transpose() * (H_ * P_ * H_.transpose() + R_).inverse();
        state_ += K * (measurement - H_ * state_);
        P_ = (I_ - K * H_) * P_;
        
        consecutiveFrames_++;
        if (consecutiveFrames_ >= 15) {
            vipLocked_ = true;
        }
    }
    
    Point3D getPredicted(float lookahead = 0.033f) const {
        // Predict +1 frame ahead for latency compensation
        return {
            state_[0] + state_[3] * lookahead,
            state_[1] + state_[4] * lookahead,
            state_[2] + state_[5] * lookahead
        };
    }
    
    void handleDropout() {
        // Bei fehlendem Frame: Nur predict(), kein update()
        // Nach 5 Dropouts: Reset
        if (++dropoutCount_ > 5) {
            vipLocked_ = false;
            consecutiveFrames_ = 0;
        }
    }
};

// One-Euro Filter für Rotationen (Landmarks-relative Pose)
class OneEuroFilter {
    float minCutoff_ = 1.0f;
    float beta_ = 0.007f;
    float dCutoff_ = 1.0f;
    
public:
    float filter(float value, float dt) {
        // ... Standard One-Euro implementation
    }
};
```

**Key Parameters:**
- **dt:** 1/FPS (bei 60fps = 16.67ms)
- **VIP Lock:** Nach 15 konsistenten Frames (~250ms @ 60fps)
- **Prediction:** +1 Frame Lookahead für Latenz-Kompensation
- **Dropout Handling:** Pure Prediction bei fehlenden Messungen

---

### Step 5: Gesten als FSM implementieren

**Ziel:** Robuste, hysterese-basierte Gesten-Erkennung.

```cpp
enum class GestureState {
    Idle,       // Keine Hand sichtbar
    Palm,       // Hand offen
    Pinch,      // Daumen + Zeigefinger zusammen
    Grab,       // Faust (alle Finger geschlossen)
    Point       // Nur Zeigefinger ausgestreckt
};

class GestureFSM {
    GestureState state_ = GestureState::Idle;
    int frameCount_ = 0;
    
    // Hysteresis thresholds
    static constexpr float PINCH_ENTER = 0.08f;  // 8% of hand size
    static constexpr float PINCH_EXIT = 0.12f;   // 12% - hysteresis gap
    
    static constexpr int DEBOUNCE_FRAMES = 3;    // ~50ms @ 60fps
    
public:
    GestureState update(const HandLandmark::Result& hand) {
        GestureState detected = detectGesture(hand);
        
        if (detected == pendingState_) {
            frameCount_++;
        } else {
            pendingState_ = detected;
            frameCount_ = 1;
        }
        
        // Debounce: Only transition after N consistent frames
        if (frameCount_ >= DEBOUNCE_FRAMES && pendingState_ != state_) {
            state_ = pendingState_;
            onTransition(state_);
        }
        
        return state_;
    }
    
private:
    GestureState detectGesture(const HandLandmark::Result& hand) {
        float pinchDist = distance(hand.landmarks[4], hand.landmarks[8]);  // Thumb tip ↔ Index tip
        float handSize = distance(hand.landmarks[0], hand.landmarks[9]);   // Wrist ↔ Middle base
        
        float normalizedPinch = pinchDist / handSize;
        
        // Apply hysteresis
        if (state_ == GestureState::Pinch) {
            // Already pinching: use exit threshold
            if (normalizedPinch > PINCH_EXIT) {
                return detectOpenHand(hand);
            }
            return GestureState::Pinch;
        } else {
            // Not pinching: use enter threshold
            if (normalizedPinch < PINCH_ENTER) {
                return GestureState::Pinch;
            }
            return detectOpenHand(hand);
        }
    }
    
    GestureState detectOpenHand(const HandLandmark::Result& hand) {
        // Finger extension checks...
        bool indexExtended = isFingerExtended(hand, 5, 6, 7, 8);
        bool allCurled = !indexExtended && !middleExtended && ...;
        
        if (allCurled) return GestureState::Grab;
        if (indexExtended && !middleExtended) return GestureState::Point;
        return GestureState::Palm;
    }
};
```

**State Machine:**
```
        ┌───────────────────────────────────────┐
        │                                       │
        ▼                                       │
    ┌───────┐    hand detected    ┌──────┐      │
    │ Idle  │ ──────────────────► │ Palm │ ─────┤
    └───────┘                     └──────┘      │
        ▲                            │          │
        │ hand lost                  │          │
        │ (5 frames)                 ▼          │
        │                    ┌───────────┐      │
        │                    │           │      │
        │              ┌─────┤  Pinch    │◄─────┤
        │              │     │           │      │
        │              │     └───────────┘      │
        │              │                        │
        │              │     ┌───────────┐      │
        │              ├─────┤   Grab    │◄─────┤
        │              │     └───────────┘      │
        │              │                        │
        │              │     ┌───────────┐      │
        └──────────────┴─────┤  Point    │◄─────┘
                             └───────────┘
```

---

## 📡 OSC Output Specification

```cpp
class OscSender {
    lo_address target_;
    std::chrono::steady_clock::time_point lastSend_;
    
    static constexpr auto SEND_INTERVAL = std::chrono::milliseconds(33);  // 30 Hz
    static constexpr auto MAX_LATENCY = std::chrono::milliseconds(50);
    
public:
    void sendTrackingResult(const TrackingResult& result) {
        auto now = std::chrono::steady_clock::now();
        
        // Rate limiting: 30 Hz constant
        if (now - lastSend_ < SEND_INTERVAL) return;
        
        // Drop-Oldest: Verwerfe veraltete Daten
        if (now - result.timestamp > MAX_LATENCY) {
            stats_.droppedPackets++;
            return;
        }
        
        // Send Palm Position (predicted)
        lo_send(target_, "/hand/palm", "fff", 
            result.palmPosition.x,
            result.palmPosition.y,
            result.palmPosition.z);
        
        // Send Velocity
        lo_send(target_, "/hand/velocity", "fff",
            result.velocity.x,
            result.velocity.y,
            result.velocity.z);
        
        // Send Gesture State
        lo_send(target_, "/hand/gesture", "if",
            static_cast<int>(result.gesture),
            result.gestureConfidence);
        
        // Send all 21 landmarks as blob (optional, for visualization)
        lo_send(target_, "/hand/landmarks", "b",
            result.landmarks.data(),
            21 * 3 * sizeof(float));
        
        lastSend_ = now;
    }
};
```

**Message Format:**
| Address | Types | Description |
|---------|-------|-------------|
| `/hand/palm` | fff | Palm center (x, y, z) normalized |
| `/hand/velocity` | fff | Velocity (vx, vy, vz) mm/s |
| `/hand/gesture` | if | (state_id, confidence) |
| `/hand/landmarks` | blob | 21×3 floats, optional |
| `/service/status` | si | (state_name, fps) |

---

## 📊 Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| E2E Latency | <60ms | Glass-to-OSC timestamping |
| FPS (Device) | 60 Hz | OAK-D frame output |
| FPS (Processing) | 30-60 Hz | Jetson inference rate |
| OSC Rate | 30 Hz | Constant, decoupled |
| Jitter | <5ms σ | OSC interval std-dev |
| VIP Lock Time | 250ms | 15 frames @ 60fps |

---

## 🔧 Configuration

```cpp
namespace config {
    // Pipeline
    constexpr int CAMERA_FPS = 60;
    constexpr int RGB_WIDTH = 640;
    constexpr int RGB_HEIGHT = 360;
    constexpr int MONO_WIDTH = 640;
    constexpr int MONO_HEIGHT = 400;
    
    // Tracking
    constexpr int VIP_LOCK_FRAMES = 15;
    constexpr int DROPOUT_LIMIT = 5;
    constexpr float KALMAN_PROCESS_NOISE = 10.0f;
    constexpr float KALMAN_MEASUREMENT_NOISE = 5.0f;
    
    // Gestures
    constexpr float PINCH_THRESHOLD_ENTER = 0.08f;
    constexpr float PINCH_THRESHOLD_EXIT = 0.12f;
    constexpr int GESTURE_DEBOUNCE_FRAMES = 3;
    
    // OSC
    constexpr int OSC_RATE_HZ = 30;
    constexpr int OSC_MAX_LATENCY_MS = 50;
    
    // Stereo
    constexpr int STEREO_WINDOW_SIZE = 9;  // 9×9 pixels
    constexpr float STEREO_BASELINE_MM = 75.0f;  // OAK-D Pro baseline
}
```

---

## ✅ Checklist for Implementation

### Phase 1: Sensor-Only Pipeline
- [ ] Remove all NN nodes from PipelineManager
- [ ] Add Mono L/R camera streams
- [ ] Add Sync node for RGB + Mono L/R
- [ ] Verify 60 FPS output from OAK-D
- [ ] Update InputLoop for synced MessageGroup

### Phase 2: TensorRT Hand Detection
- [ ] Convert palm_detection to TensorRT engine
- [ ] Convert hand_landmark to TensorRT engine
- [ ] Implement NV12 → NN input preprocessing (GPU)
- [ ] Implement unletterbox for coordinates
- [ ] Verify 30+ FPS on Jetson

### Phase 3: Stereo Depth at Palm
- [ ] Load calibration from OAK-D
- [ ] Implement rectification (precomputed maps)
- [ ] Implement local stereo matching (9×9 window)
- [ ] Output Z in camera coordinates

### Phase 4: Kalman Tracking
- [ ] Implement 6-state Kalman filter
- [ ] Implement One-Euro for rotations
- [ ] Implement VIP lock logic
- [ ] Implement +1 frame prediction

### Phase 5: Gesture FSM
- [ ] Implement state machine with hysteresis
- [ ] Implement debounce logic
- [ ] Define all gesture states
- [ ] Test transition robustness

### Phase 6: OSC Integration
- [ ] 30 Hz constant rate
- [ ] Drop-Oldest >50ms policy
- [ ] All message types
- [ ] Performance metrics

---

## 📁 File Structure (Target)

```
src/
├── core/
│   ├── PipelineManager.cpp      # OAK-D Sensor-Only Pipeline
│   ├── InputLoop.cpp            # Frame reception + sync
│   ├── ProcessingLoop.cpp       # Main processing orchestrator
│   ├── StereoDepth.cpp          # NEW: Punktuelle Tiefe
│   ├── HandTracker.cpp          # NEW: Kalman + VIP Logic
│   └── GestureFSM.cpp           # NEW: Gesture State Machine
├── inference/
│   ├── TensorRTEngine.cpp       # NEW: TensorRT wrapper
│   ├── PalmDetector.cpp         # NEW: Palm detection
│   └── HandLandmark.cpp         # NEW: Landmark inference
├── net/
│   ├── OscSender.cpp            # OSC output (30 Hz)
│   └── MjpegServer.cpp          # Debug preview
└── main.cpp
```

---

## 🚀 Next Steps

1. **JETZT:** PipelineManager auf Sensor-Only umbauen
2. **DANACH:** TensorRT Engine Wrapper implementieren
3. **DANN:** Kalman + Gesture FSM

> **Erinnerung:** Dies ist ein 3D-Controller, kein CV-System. 
> Jede Entscheidung optimiert für Latenz und Robustheit, nicht für Genauigkeit.

