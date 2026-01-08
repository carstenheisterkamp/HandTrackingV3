# TODO: HandTracking V3 - Production Roadmap

**Basierend auf:** OPTIMAL_WORKFLOW_V2_FINAL.md + Review  
**Status:** 2026-01-08  
**Aktueller Stand:** 18 FPS @ 1080p/30 mit Palm + Landmarks  
**Ziel:** 45 FPS @ 720p mit 2 VIPs + Person Detection  

---

## 📊 Aktueller Status (Commit: 88ae95a)

### ✅ Was funktioniert (70% von SPEC.md)
- [x] Single-Hand Tracking (Palm Detection + Landmarks)
- [x] VIP Locking (15-Frame Konsistenz)
- [x] Kalman + One-Euro Filter (21 Landmarks)
- [x] Gesture Recognition (FIST, FIVE, POINT, OK, THUMB_UP)
- [x] Velocity-Berechnung (3D mit Depth)
- [x] OSC Output (30 Hz mit Backpressure)
- [x] Zero-Copy Pipeline (Pinned Memory + NPP CUDA)
- [x] MJPEG Preview Server
- [x] Stereo Depth Infrastructure (CUDA Kernel)

### ⚠️ Was suboptimal ist
- [ ] FPS: 18 statt 25-30 (SPEC) bzw. 45 (OPTIMAL)
- [ ] MJPEG läuft permanent (auch ohne Clients)
- [ ] Stereo Depth jedes Frame (ineffizient)
- [ ] Preview: 960x540 (zu groß)
- [ ] NN Threads: 2 (sollte 1 sein)

### ❌ Was fehlt für OPTIMAL_WORKFLOW
- [ ] Person Detection (0%)
- [ ] ObjectTracker (0%)
- [ ] Multi-VIP Support (0%)
- [ ] ROI-System (0%)
- [ ] Asynchrone Inference-Raten (0%)
- [ ] Performance Metrics (0%)
- [ ] Config System (0%)

---

## 🚀 PHASE 0: Quick Wins (18 → 30 FPS)

**Ziel:** SPEC.md erfüllen in 1 Tag  
**Status:** 🔴 NOT STARTED  
**Priorität:** 🔥 KRITISCH - SOFORT STARTEN  

### Aufgaben (6-8 Stunden)

#### 1. MJPEG hasClients() Check
**Status:** [ ] TODO  
**Datei:** `src/core/ProcessingLoop.cpp`, `include/net/MjpegServer.hpp`  
**Aufwand:** 1-2 Stunden  
**FPS-Gewinn:** +10 FPS

```cpp
// MjpegServer.hpp - Neue Methode hinzufügen
class MjpegServer {
public:
    bool hasClients() const { return !_clients.empty(); }
    // ...existing code...
};

// ProcessingLoop.cpp - Nur encoden wenn Clients
if (_mjpegServer && _mjpegServer->hasClients()) {
    // Color Conversion + JPEG Encoding
    debugFrame = convertToRGB(frame);
    _mjpegServer->sendFrame(debugFrame);
}
```

**Acceptance Criteria:**
- [ ] MJPEG-Encoding nur bei aktiven Clients
- [ ] FPS ohne Clients: 28-30 (statt 18)
- [ ] Keine Regression bei verbundenen Clients

---

#### 2. Stereo Depth Throttling
**Status:** [ ] TODO  
**Datei:** `src/core/ProcessingLoop.cpp`  
**Aufwand:** 30 Minuten  
**FPS-Gewinn:** +5 FPS

```cpp
// ProcessingLoop.cpp - Stereo nur alle 3 Frames
#ifdef ENABLE_CUDA
static int stereoCounter = 0;
if (++stereoCounter % 3 == 0 && frame->hasStereoData) {
    computeStereoDepth(frame->monoLeftData.get(),
                      frame->monoRightData.get(),
                      (uint16_t*)frame->depthData.get(),
                      frame->monoWidth, frame->monoHeight);
    cudaStreamSynchronize(0);
}
// Verwende gecachte Depth für Frames 1 und 2
#endif
```

**Acceptance Criteria:**
- [ ] Stereo nur alle 3 Frames (alle 66 ms @ 45 FPS)
- [ ] FPS-Gewinn: +3-5 FPS
- [ ] Depth-Qualität visuell unverändert

---

#### 3. Preview-Auflösung reduzieren
**Status:** [ ] TODO  
**Datei:** `src/main.cpp`  
**Aufwand:** 5 Minuten  
**FPS-Gewinn:** +2 FPS

```cpp
// main.cpp - Kleinere Preview
config.previewWidth = 640;   // war: 960
config.previewHeight = 360;  // war: 540
```

**Acceptance Criteria:**
- [ ] Preview: 640x360 (statt 960x540)
- [ ] FPS-Gewinn: +1-2 FPS
- [ ] MJPEG Preview noch lesbar

---

#### 4. NN Threads optimieren
**Status:** [ ] TODO  
**Datei:** `src/core/PipelineManager.cpp`  
**Aufwand:** 5 Minuten  
**FPS-Gewinn:** +3 FPS

```cpp
// PipelineManager.cpp - 1 Thread ist schneller auf Myriad X
palmDetect->setNumInferenceThreads(1);  // war: 2
landmarkNN->setNumInferenceThreads(1);  // war: 2
```

**Acceptance Criteria:**
- [ ] Beide NNs mit 1 Thread
- [ ] FPS-Gewinn: +2-3 FPS
- [ ] Keine Accuracy-Regression

---

#### 5. Sync Threshold reduzieren
**Status:** [ ] TODO  
**Datei:** `src/core/PipelineManager.cpp`  
**Aufwand:** 5 Minuten  
**FPS-Gewinn:** +2 FPS

```cpp
// PipelineManager.cpp - Niedrigerer Threshold
sync->setSyncThreshold(std::chrono::milliseconds(10));  // war: 20
```

**Acceptance Criteria:**
- [ ] Sync Threshold: 10 ms (statt 20 ms)
- [ ] FPS-Gewinn: +1-2 FPS
- [ ] Keine Frame-Drops durch zu enges Timing

---

#### 6. Exposure Limit (garantierte 30 FPS)
**Status:** [ ] TODO  
**Datei:** `src/core/PipelineManager.cpp`  
**Aufwand:** 10 Minuten  
**FPS-Gewinn:** Stabilität (verhindert Drops bei Dunkelheit)

```cpp
// PipelineManager.cpp - Max 20 ms Exposure
cam->initialControl.setAutoExposureLimit(20000);  // 20 ms = 1/50s
cam->initialControl.setAutoExposureCompensation(1);  // Leicht heller
```

**Acceptance Criteria:**
- [ ] Exposure nie > 20 ms (garantiert 30+ FPS)
- [ ] ISO kompensiert (Helligkeit akzeptabel)
- [ ] FPS stabil auch bei schlechtem Licht

---

### Phase 0 - Gesamt Acceptance Criteria

**Must-Have:**
- [ ] **FPS: 28-30** (ohne MJPEG-Clients)
- [ ] **FPS: 25-28** (mit MJPEG-Clients)
- [ ] **Latenz: < 80 ms** (E2E gemessen)
- [ ] **Stabilität:** Keine Frame-Drops über 5 Minuten

**Testing:**
```bash
# 1. Ohne MJPEG-Clients
./HandTrackingV3
# Erwartung: 28-30 FPS

# 2. Mit MJPEG-Client
firefox http://100.101.16.21:8080 &
./HandTrackingV3
# Erwartung: 25-28 FPS

# 3. Stress-Test
./HandTrackingV3 &
# Laufen lassen für 5 Minuten
# Erwartung: Keine Abstürze, FPS stabil ±2
```

**Wenn erfolgreich:** ✅ SPEC.md FPS-Ziel erreicht → **Weiter zu Phase 1**

---

## 🚀 PHASE 1: Person Detection & Tracking

**Ziel:** Multi-Person-Support (2 VIPs)  
**Status:** 🔴 NOT STARTED  
**Priorität:** 🔥 KRITISCH (nach Phase 0)  
**Dauer:** 5-7 Tage  

### 1.1 YOLOv8n-person Model Setup

**Status:** [ ] TODO  
**Aufwand:** 2-3 Tage  

#### 1.1.1 Model Download & Export
**Datei:** `models/`  
**Skript:** Neues `scripts/prepare_yolov8n_person.sh`

```bash
#!/bin/bash
# Download YOLOv8n Pretrained
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt

# Export zu ONNX (Person-only, Class 0)
yolo export model=yolov8n.pt format=onnx simplify=True classes=[0]

# Ausgabe: yolov8n-person.onnx
```

**Acceptance Criteria:**
- [ ] yolov8n.pt heruntergeladen
- [ ] yolov8n-person.onnx exportiert (Person-only)
- [ ] Model-Größe: ~6 MB (ONNX)

---

#### 1.1.2 TensorRT Conversion (INT8)
**Datei:** `models/`  
**Skript:** Neues `scripts/convert_yolov8n_tensorrt.sh`

**WICHTIG:** INT8 benötigt Calibration-Dataset

```bash
#!/bin/bash
# Calibration Dataset vorbereiten (COCO Person-Subset)
# ~1000 Images, mixed lighting

# TensorRT Conversion mit INT8
trtexec --onnx=yolov8n-person.onnx \
        --int8 \
        --workspace=4096 \
        --saveEngine=yolov8n-person-int8.trt \
        --shapes=images:1x3x384x640 \
        --fp16  # Fallback für Layers die kein INT8 können

# Benchmark
trtexec --loadEngine=yolov8n-person-int8.trt \
        --shapes=images:1x3x384x640 \
        --iterations=100
```

**Expected Output:**
```
Latency: mean=8.5ms, median=8.3ms, 99th=10.2ms
Throughput: ~120 FPS (bei kontinuierlicher Nutzung)
```

**Acceptance Criteria:**
- [ ] TensorRT Engine erstellt (INT8)
- [ ] Latenz: < 12 ms @ 640×384
- [ ] VRAM: ~120 MB
- [ ] Benchmark durchgeführt

**Config-Specs (aus Review):**
```
Input:      640×384 (16:9.6 Aspect Ratio)
Precision:  INT8
Classes:    Person only (Class 0)
Target FPS: 12-15 FPS (async)
Latenz:     ~8-10 ms
VRAM:       ~120 MB
```

---

#### 1.1.3 TensorRT Inference Wrapper (C++)
**Status:** [ ] TODO  
**Neue Dateien:**
- `include/detection/PersonDetector.hpp`
- `src/detection/PersonDetector.cpp`

```cpp
// PersonDetector.hpp
#pragma once
#include <string>
#include <vector>
#include <memory>
#include <NvInfer.h>

namespace detection {

struct PersonBBox {
    float x, y, w, h;      // Normalized (0..1)
    float confidence;
    int track_id = -1;     // Wird von ObjectTracker gefüllt
};

class PersonDetector {
public:
    PersonDetector(const std::string& enginePath);
    ~PersonDetector();
    
    std::vector<PersonBBox> detect(const cv::Mat& rgbFrame);
    
private:
    nvinfer1::IRuntime* _runtime;
    nvinfer1::ICudaEngine* _engine;
    nvinfer1::IExecutionContext* _context;
    
    void* _inputBuffer;
    void* _outputBuffer;
    
    void preprocess(const cv::Mat& frame, float* inputBuffer);
    std::vector<PersonBBox> postprocess(float* outputBuffer);
};

} // namespace detection
```

**Acceptance Criteria:**
- [ ] TensorRT Engine geladen
- [ ] Preprocessing: cv::Mat → GPU Tensor (640×384)
- [ ] Postprocessing: NMS (Threshold: 0.35 für Multi-Person)
- [ ] Inference: < 12 ms
- [ ] Memory Leaks: Keine (Valgrind)

---

#### 1.1.4 Async Inference Manager
**Status:** [ ] TODO  
**Neue Datei:** `include/detection/AsyncPersonDetector.hpp`

```cpp
// AsyncPersonDetector.hpp - Läuft @ 12 FPS statt 45 FPS
class AsyncPersonDetector {
public:
    AsyncPersonDetector(std::shared_ptr<PersonDetector> detector);
    
    // Non-blocking: Triggered alle ~4 Frames (bei 45 FPS)
    void submitFrame(const cv::Mat& frame, int frameId);
    
    // Returns cached result wenn kein neues verfügbar
    std::vector<PersonBBox> getLatestDetections();
    
private:
    std::shared_ptr<PersonDetector> _detector;
    std::thread _inferenceThread;
    std::atomic<bool> _running;
    
    std::vector<PersonBBox> _cachedDetections;
    std::mutex _cacheMutex;
    
    int _frameCounter = 0;
    static constexpr int DETECTION_INTERVAL = 4;  // Alle 4 Frames @ 45 FPS = 12 FPS
};
```

**Acceptance Criteria:**
- [ ] Detection nur alle 4 Frames (12 FPS @ 45 FPS System)
- [ ] Cached Results für Frames 1-3
- [ ] Non-blocking für Main-Thread
- [ ] Thread-safe (keine Data-Races)

---

### 1.2 ObjectTracker Integration (OAK-D)

**Status:** [ ] TODO  
**Aufwand:** 1-2 Tage  
**Datei:** `src/core/PipelineManager.cpp`

#### 1.2.1 ObjectTracker Node hinzufügen

```cpp
// PipelineManager.cpp - createPipeline()

// Person Detections vom Jetson → XLinkIn
auto trackerIn = pipeline_->create<dai::node::XLinkIn>();
trackerIn->setStreamName("tracker_in");

// ObjectTracker Node
auto tracker = pipeline_->create<dai::node::ObjectTracker>();
tracker->setTrackerType(dai::TrackerType::SHORT_TERM_KCF);  // Oder ZERO_TERM_COLOR_HISTOGRAM
tracker->setMaxObjectsToTrack(2);  // Nur 2 VIPs
tracker->setTrackerIdAssignmentPolicy(dai::TrackerIdAssignmentPolicy::SMALLEST_ID);

// Config
tracker->inputTrackerFrame.setBlocking(false);
tracker->inputDetectionFrame.setBlocking(false);

// Wiring
rgbOutput->link(tracker->inputTrackerFrame);
trackerIn->out.link(tracker->inputDetections);

// Output Queue
auto trackerQueue = tracker->out.createOutputQueue(4, false);
queues_["tracker"] = trackerQueue;
```

**Acceptance Criteria:**
- [ ] ObjectTracker Node in Pipeline
- [ ] XLinkIn für Detections vom Jetson
- [ ] Tracker Config: 2 VIPs max
- [ ] Output Queue erstellt

---

#### 1.2.2 Detection → Tracker Feed
**Status:** [ ] TODO  
**Datei:** `src/core/InputLoop.cpp`

```cpp
// InputLoop.cpp - Person Detections zum Tracker senden

// 1. Hole Person Detections vom AsyncPersonDetector
auto detections = _personDetector->getLatestDetections();

// 2. Konvertiere zu dai::ImgDetections
dai::ImgDetections daiDetections;
for (const auto& bbox : detections) {
    dai::ImgDetection det;
    det.xmin = bbox.x;
    det.ymin = bbox.y;
    det.xmax = bbox.x + bbox.w;
    det.ymax = bbox.y + bbox.h;
    det.confidence = bbox.confidence;
    det.label = 0;  // Person
    daiDetections.detections.push_back(det);
}

// 3. Sende zu OAK-D Tracker via XLinkIn
auto trackerInQueue = _device->getInputQueue("tracker_in");
trackerInQueue->send(daiDetections);
```

**Acceptance Criteria:**
- [ ] Detections vom Jetson zum OAK-D Tracker
- [ ] Format-Conversion korrekt
- [ ] Latenz: < 5 ms (XLink Transfer)

---

#### 1.2.3 Tracker Output Processing
**Status:** [ ] TODO  
**Neue Datei:** `include/tracking/VIPManager.hpp`

```cpp
// VIPManager.hpp - VIP1/VIP2 Selection & Management
class VIPManager {
public:
    struct VIP {
        int id = -1;
        cv::Rect2f bbox;      // Normalized
        float depth_z = 0.0f; // mm
        float confidence = 0.0f;
        bool locked = false;
    };
    
    void update(const std::vector<dai::Tracklet>& tracklets);
    
    VIP getVIP1() const { return _vip1; }
    VIP getVIP2() const { return _vip2; }
    
private:
    VIP _vip1, _vip2;
    int _vipSwitchCounter = 0;
    static constexpr int SWITCH_HYSTERESIS = 30;  // 0.66s @ 45 FPS
    
    void selectVIPs(std::vector<VIP>& candidates);
};
```

**VIP-Selection-Logik (aus OPTIMAL_WORKFLOW):**
```cpp
void VIPManager::selectVIPs(std::vector<VIP>& candidates) {
    if (candidates.empty()) {
        _vip1.id = _vip2.id = -1;
        return;
    }
    
    // Sort by depth (nearest = VIP1)
    std::sort(candidates.begin(), candidates.end(),
              [](const VIP& a, const VIP& b) { return a.depth_z < b.depth_z; });
    
    int newVIP1 = candidates[0].id;
    int newVIP2 = (candidates.size() > 1) ? candidates[1].id : -1;
    
    // Hysterese: 30 Frames (0.66s) bevor Switch
    if (newVIP1 != _vip1.id) {
        _vipSwitchCounter++;
        if (_vipSwitchCounter > SWITCH_HYSTERESIS) {
            Logger::info("VIP Switch: ", _vip1.id, " → ", newVIP1);
            _vip1 = candidates[0];
            _vip2 = (candidates.size() > 1) ? candidates[1] : VIP{};
            _vipSwitchCounter = 0;
        }
    } else {
        _vipSwitchCounter = 0;
        _vip2 = (candidates.size() > 1) ? candidates[1] : VIP{};
    }
}
```

**Acceptance Criteria:**
- [ ] VIP1/VIP2 Selection (nearest person = VIP1)
- [ ] Hysterese: 30 Frames (verhindert Flackern)
- [ ] Tracking-Confidence checked (> 0.7)
- [ ] Fallback zu Detection bei Track-Loss

---

### 1.3 Testing & Validation

**Status:** [ ] TODO  
**Aufwand:** 1 Tag  

#### Test-Scenarios:

**1. Basic Multi-Person:**
- [ ] 2 Personen im Frame (30s stabil)
- [ ] VIP1/VIP2 IDs bleiben konsistent
- [ ] FPS: > 35 (mit Person Detection)

**2. VIP-Switch:**
- [ ] Person A näher → VIP1
- [ ] Person B kommt näher → Switch nach 30 Frames
- [ ] Kein Flackern während Switch

**3. Track-Loss & Re-ID:**
- [ ] Person hinter Möbel → Track-Loss
- [ ] Person wieder sichtbar → Re-ID
- [ ] Erwartung: Gleiche ID (wenn < 5 Sekunden)

**4. Okklusion:**
- [ ] 2 Personen kreuzen sich (< 50 cm Abstand)
- [ ] Beide IDs bleiben stabil
- [ ] Kein ID-Swap

**5. Edge Cases:**
- [ ] 3+ Personen → nur VIP1/VIP2 tracked
- [ ] 0 Personen → System idle
- [ ] 1 Person → VIP1 only, VIP2 = -1

**Metrics zu erfassen:**
```cpp
struct Phase1Metrics {
    float person_detection_fps;  // Target: 12-15
    float tracking_fps;          // Target: 40-45
    float vip1_uptime;           // Target: > 95%
    int id_switches;             // Target: < 5 in 60s
    float avg_latency;           // Target: < 80 ms
};
```

**Acceptance Criteria Phase 1:**
- [ ] Person Detection: 12-15 FPS
- [ ] Tracking FPS: > 35 (mit Detection-Last)
- [ ] VIP1 Uptime: > 90%
- [ ] ID-Stabilität: > 95% über 30s
- [ ] Latenz: < 80 ms E2E
- [ ] Keine Crashes über 5 Minuten

**Wenn erfolgreich:** ✅ Multi-Person-Support funktioniert → **Weiter zu Phase 2**

---

## 🚀 PHASE 2: ROI-System (Host-side)

**Ziel:** Hand-Tracking nur für VIP1 (Effizienz)  
**Status:** 🔴 NOT STARTED  
**Priorität:** 🔥 HOCH (nach Phase 1)  
**Dauer:** 3-4 Tage  

### 2.1 ROI-Calculation (Upper-Body)

**Status:** [ ] TODO  
**Aufwand:** 1 Tag  
**Neue Datei:** `include/tracking/ROIManager.hpp`

```cpp
// ROIManager.hpp
class ROIManager {
public:
    cv::Rect2f computeHandROI(const VIPManager::VIP& vip) {
        // Person BBox → Hand-ROI (1.5× Armspanne)
        float armSpan = vip.bbox.height * 0.35f;  // Empirisch
        
        cv::Rect2f roi;
        roi.x = vip.bbox.x + vip.bbox.width * 0.5f - armSpan;
        roi.y = vip.bbox.y + vip.bbox.height * 0.3f - armSpan * 0.5f;
        roi.width = armSpan * 2.0f;
        roi.height = armSpan * 1.5f;
        
        // Clamp to [0, 1]
        roi &= cv::Rect2f(0, 0, 1, 1);
        return roi;
    }
};
```

**Acceptance Criteria:**
- [ ] ROI umfasst Hand-Region (Schulter bis Hand)
- [ ] ROI dynamisch angepasst an Person-Größe
- [ ] ROI nie außerhalb Frame-Grenzen

---

### 2.2 Hand-NN auf ROI anwenden

**Status:** [ ] TODO  
**Aufwand:** 2 Tage  
**Datei:** `src/core/ProcessingLoop.cpp`

```cpp
// ProcessingLoop.cpp - Hand-Tracking nur VIP1

// 1. Hole VIP1 ROI
auto vip1 = _vipManager->getVIP1();
if (vip1.id < 0) {
    // Kein VIP1 → kein Hand-Tracking
    return;
}

cv::Rect2f handROI = _roiManager->computeHandROI(vip1);

// 2. Crop RGB auf ROI
cv::Rect pixelROI(
    handROI.x * frame->width,
    handROI.y * frame->height,
    handROI.width * frame->width,
    handROI.height * frame->height
);
cv::Mat rgbROI = rgbFrame(pixelROI);

// 3. Hand Landmarks NN (TensorRT)
auto landmarks = _handNN->infer(rgbROI);

// 4. Transform Landmarks: ROI-Space → Frame-Space
for (auto& lm : landmarks) {
    lm.x = handROI.x + lm.x * handROI.width;
    lm.y = handROI.y + lm.y * handROI.height;
}
```

**Erwarteter Speedup:**
- Hand-ROI = ~25% des Frames
- Inference: 4× schneller als Full-Frame
- FPS-Gewinn: +5-10 FPS

**Acceptance Criteria:**
- [ ] Hand-NN nur auf VIP1-ROI
- [ ] Landmarks korrekt transformiert (ROI → Frame)
- [ ] FPS-Gewinn: > +5 FPS
- [ ] Tracking-Qualität unverändert

---

### 2.3 VIP2: Position only (kein Hand-Tracking)

**Status:** [ ] TODO  
**Aufwand:** 1 Tag  
**Datei:** `src/core/ProcessingLoop.cpp`

```cpp
// ProcessingLoop.cpp - VIP2 Output

auto vip2 = _vipManager->getVIP2();
if (vip2.id >= 0) {
    // VIP2: Nur Torso-Position + Velocity
    TrackingResult vip2Result;
    vip2Result.vipId = 2;
    vip2Result.torsoPosition = {
        vip2.bbox.x + vip2.bbox.width * 0.5f,
        vip2.bbox.y + vip2.bbox.height * 0.3f,
        vip2.depth_z
    };
    vip2Result.velocity = calculateVelocity(vip2.id);
    
    // Kein Hand-Tracking für VIP2
    vip2Result.landmarks.clear();
    
    _oscQueue->push(vip2Result);
}
```

**OSC Format:**
```
/vip/2/position/x <float>
/vip/2/position/y <float>
/vip/2/position/z <float>
/vip/2/velocity/x <float>
/vip/2/velocity/y <float>
/vip/2/velocity/z <float>
/vip/2/hand/status "none"
```

**Acceptance Criteria:**
- [ ] VIP2 sendet nur Position (kein Hand-Tracking)
- [ ] OSC-Client kann VIP2 unterscheiden
- [ ] Kein FPS-Impact durch VIP2

---

### 2.4 Testing & Validation

**Test-Scenarios:**

**1. ROI Accuracy:**
- [ ] Hand immer innerhalb ROI (verschiedene Posen)
- [ ] ROI passt sich an Person-Größe an

**2. Performance:**
- [ ] FPS-Gewinn: > +5 FPS vs. Full-Frame
- [ ] Latenz-Impact: < 5 ms

**3. Multi-VIP:**
- [ ] VIP1: Full Hand-Tracking
- [ ] VIP2: Nur Position
- [ ] Beide @ 45 FPS

**Acceptance Criteria Phase 2:**
- [ ] FPS: 35 → 40-45 (mit ROI)
- [ ] Hand-Tracking nur VIP1
- [ ] VIP2 ohne FPS-Impact
- [ ] Tracking-Qualität unverändert
- [ ] Latenz: < 70 ms E2E

**Wenn erfolgreich:** ✅ ROI-System funktioniert → **Weiter zu Phase 3**

---

## 🚀 PHASE 3: Stereo Depth für Torso

**Ziel:** 3D-Position für VIP1 + VIP2  
**Status:** 🔴 NOT STARTED  
**Priorität:** 🟡 MITTEL (nach Phase 2)  
**Dauer:** 2-3 Tage  

### 3.1 Stereo @ 20 FPS (throttled)

**Status:** [ ] TODO  
**Datei:** `src/core/ProcessingLoop.cpp`

```cpp
// ProcessingLoop.cpp - Stereo alle 2-3 Frames
#ifdef ENABLE_CUDA
static int stereoCounter = 0;
static cv::Mat lastDepthMap;

if (++stereoCounter % 2 == 0 && frame->hasStereoData) {
    computeStereoDepth(frame->monoLeftData.get(),
                      frame->monoRightData.get(),
                      (uint16_t*)frame->depthData.get(),
                      frame->monoWidth, frame->monoHeight);
    cudaStreamSynchronize(0);
    lastDepthMap = cv::Mat(...);  // Cache
} else {
    // Verwende cached Depth
    frame->depthData = lastDepthMap;
}
#endif
```

**Acceptance Criteria:**
- [ ] Stereo @ 20-30 FPS (alle 2 Frames @ 45 FPS)
- [ ] Depth-Qualität visuell unverändert
- [ ] FPS stabil (kein Impact durch Throttling)

---

### 3.2 3D-Position aus Depth + BBox

**Status:** [ ] TODO  
**Aufwand:** 1 Tag  

```cpp
// VIPManager.cpp - Depth aus Stereo-Map extrahieren
float VIPManager::getDepthAtBBox(const cv::Rect2f& bbox, const cv::Mat& depthMap) {
    // Sample Depth im Torso-Bereich (Mitte der BBox, obere Hälfte)
    int cx = (bbox.x + bbox.width * 0.5f) * depthMap.cols;
    int cy = (bbox.y + bbox.height * 0.3f) * depthMap.rows;
    
    // Median-Filter (5×5) für Stabilität
    std::vector<uint16_t> samples;
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            int x = std::clamp(cx + dx, 0, depthMap.cols - 1);
            int y = std::clamp(cy + dy, 0, depthMap.rows - 1);
            uint16_t depth = depthMap.at<uint16_t>(y, x);
            if (depth > 0) samples.push_back(depth);
        }
    }
    
    if (samples.empty()) return 0.0f;  // Invalid
    
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];  // Median
}
```

**Acceptance Criteria:**
- [ ] 3D-Position für VIP1 + VIP2 (Torso)
- [ ] Depth-Jitter: < 50 mm
- [ ] Invalid Depth → Fallback 2D

---

### 3.3 Hand-Position mit Depth

**Status:** [ ] TODO  
**Datei:** `src/core/ProcessingLoop.cpp`

```cpp
// ProcessingLoop.cpp - Hand Landmarks mit echter Depth

for (int i = 0; i < 21; ++i) {
    float x = landmarks[i].x;  // Normalized
    float y = landmarks[i].y;
    
    // Sample Depth an Hand-Position
    int u = x * frame->depthWidth;
    int v = y * frame->depthHeight;
    uint16_t depth = depthMap.at<uint16_t>(v, u);
    
    if (depth > 0) {
        landmarks[i].z = depth;  // mm
    } else {
        // Fallback: Relative Depth vom NN-Model
        landmarks[i].z = nnOutput[i * 3 + 2];
    }
}
```

**Acceptance Criteria:**
- [ ] Hand-Landmarks mit echter Depth (mm)
- [ ] Fallback bei invalid Depth
- [ ] Velocity mit echter Z-Achse

---

**Acceptance Criteria Phase 3:**
- [ ] 3D-Position für VIP1 + VIP2 (Torso + Hand)
- [ ] Depth-Jitter: < 50 mm
- [ ] Stereo @ 20-30 FPS (throttled)
- [ ] FPS: 40-45 (unverändert)
- [ ] OSC sendet mm-Koordinaten

**Wenn erfolgreich:** ✅ Full 3D-Tracking → **Weiter zu Phase 4 (Optional)**

---

## 🚀 PHASE 4: FPS-Optimierung auf 45 FPS

**Ziel:** OPTIMAL_WORKFLOW FPS erreichen  
**Status:** 🔴 NOT STARTED  
**Priorität:** 🟡 OPTIONAL (nach Phase 3)  
**Dauer:** 3-5 Tage  

### 4.1 RGB @ 720p/45 FPS

**Status:** [ ] TODO  
**Datei:** `src/core/PipelineManager.cpp`, `src/main.cpp`

```cpp
// main.cpp
config.fps = 45.0f;  // war: 30

// PipelineManager.cpp
auto cam = pipeline_->create<dai::node::Camera>()->build(
    dai::CameraBoardSocket::CAM_A,
    std::make_pair(1280, 720),  // 720p statt 1080p
    45.0f
);

// Exposure-Limit: 22 ms (für garantierte 45 FPS)
cam->initialControl.setAutoExposureLimit(22000);
```

**Acceptance Criteria:**
- [ ] RGB @ 720p/45 FPS
- [ ] Exposure nie > 22 ms
- [ ] PoE Bandwidth OK (< 1 Gbps)

---

### 4.2 Async Inference-Raten

**Status:** [ ] TODO  

**Person Detection:** 12 FPS (alle ~4 Frames)
```cpp
// Bereits in Phase 1 implementiert
static constexpr int DETECTION_INTERVAL = 4;  // 45 FPS / 4 = ~11 FPS
```

**Gesture:** 15 FPS (alle 3 Frames)
```cpp
// ProcessingLoop.cpp
static int gestureCounter = 0;
if (++gestureCounter % 3 == 0) {
    recognizeGesture(landmarks);
}
```

**Acceptance Criteria:**
- [ ] Person Detection: 12 FPS
- [ ] Gesture: 15 FPS
- [ ] Hand Landmarks: 30 FPS (VIP1 only)
- [ ] Stereo Depth: 20 FPS

---

### 4.3 Pipeline-Tuning

**Status:** [ ] TODO  

```cpp
// Sync Threshold weiter reduzieren
sync->setSyncThreshold(std::chrono::milliseconds(8));  // war: 10

// Queue Sizes reduzieren (weniger Latenz)
auto syncQueue = sync->out.createOutputQueue(3, false);  // war: 4

// ImageManip Threads
manipPalm->setNumThreads(2);
manipLandmark->setNumThreads(2);
```

**Acceptance Criteria:**
- [ ] Sync Threshold: 8 ms
- [ ] Queue Sizes: 3
- [ ] Keine Frame-Drops

---

### 4.4 Profiling & Bottleneck-Analyse

**Status:** [ ] TODO  
**Aufwand:** 1 Tag  

**Tools:**
```bash
# CUDA Profiling
nsys profile -o report.qdrep ./HandTrackingV3

# CPU Profiling
perf record -g ./HandTrackingV3
perf report

# Memory Profiling
valgrind --tool=massif ./HandTrackingV3
```

**Latenz-Breakdown messen:**
```cpp
struct LatencyBreakdown {
    float camera_capture;    // Target: 22 ms
    float rgb_transfer;      // Target: 10 ms
    float person_detection;  // Target: 10 ms
    float object_tracker;    // Target: 2 ms
    float hand_nn;          // Target: 12 ms
    float gesture;          // Target: 3 ms
    float osc_send;         // Target: 1 ms
    // ────────────────────────────
    // Total:                 ~60 ms
};
```

**Acceptance Criteria:**
- [ ] Bottlenecks identifiziert
- [ ] Latenz-Budget eingehalten (< 70 ms)
- [ ] Optimization-Priorities klar

---

**Acceptance Criteria Phase 4:**
- [ ] **FPS: 43-45** (stabil)
- [ ] **Latenz: < 60 ms** (E2E)
- [ ] **Jitter: < 10 ms**
- [ ] **Frame Drops: < 1%**
- [ ] Profiling abgeschlossen

**Wenn erfolgreich:** ✅ OPTIMAL_WORKFLOW FPS erreicht → **Weiter zu Phase 5 (Infrastruktur)**

---

## 🚀 PHASE 5: Production-Infrastruktur

**Ziel:** Messbarkeit, Config, Monitoring  
**Status:** 🔴 NOT STARTED  
**Priorität:** 🟢 OPTIONAL (nach Phase 4)  
**Dauer:** 2-3 Tage  

### 5.1 Config-System (JSON)

**Status:** [ ] TODO  
**Aufwand:** 1 Tag  
**Neue Dateien:**
- `config/settings.json`
- `include/core/ConfigManager.hpp`
- `src/core/ConfigManager.cpp`

```json
// config/settings.json
{
  "camera": {
    "fps": 45.0,
    "resolution": {"width": 1280, "height": 720},
    "exposure_limit_ms": 22,
    "preview": {"width": 640, "height": 360},
    "device_ip": "169.254.1.222"
  },
  "person_detection": {
    "model_path": "models/yolov8n-person-int8.trt",
    "fps": 12,
    "nms_threshold": 0.35,
    "confidence_threshold": 0.6
  },
  "hand_tracking": {
    "model_path": "models/hand_landmark_full_sh4.blob",
    "vip1_only": true
  },
  "osc": {
    "host": "127.0.0.1",
    "port": 9000,
    "rate": 30
  },
  "filters": {
    "kalman": {
      "process_noise": 0.01,
      "measurement_noise": 0.1
    },
    "one_euro": {
      "cutoff": 1.0,
      "beta": 0.007
    }
  },
  "performance": {
    "stereo_interval": 2,
    "gesture_interval": 3,
    "mjpeg_port": 8080
  }
}
```

```cpp
// ConfigManager.hpp
class ConfigManager {
public:
    static ConfigManager& instance();
    void load(const std::string& path);
    
    float getCameraFPS() const;
    std::string getOSCHost() const;
    // ...etc
    
private:
    nlohmann::json _config;
};
```

**Acceptance Criteria:**
- [ ] JSON-Config geladen
- [ ] Alle Magic-Numbers eliminiert
- [ ] Runtime-Parameter änderbar (ohne Recompile)

---

### 5.2 Performance-Metriken & HTTP-Endpoint

**Status:** [ ] TODO  
**Aufwand:** 1 Tag  
**Neue Dateien:**
- `include/core/PerformanceMonitor.hpp`
- `src/core/PerformanceMonitor.cpp`
- `include/net/MetricsServer.hpp`

```cpp
// PerformanceMonitor.hpp
class PerformanceMonitor {
public:
    void recordFrameLatency(float ms);
    void recordOscSendTime(float ms);
    void recordDroppedFrame();
    void recordIDSwitch();
    
    struct Metrics {
        float device_fps;
        float host_fps;
        float osc_fps;
        float e2e_latency_ms;
        float jitter_ms;
        float vip1_uptime;
        float vip2_uptime;
        int id_switches;
        int frames_dropped;
        int osc_drops;
    };
    
    Metrics getMetrics() const;
    std::string getMetricsJSON() const;
};
```

```cpp
// MetricsServer.hpp - HTTP Server für /service/metrics
class MetricsServer {
public:
    MetricsServer(int port);
    void start();
    void stop();
    
    void setMetrics(const PerformanceMonitor::Metrics& metrics);
    
private:
    int _port;
    std::thread _serverThread;
    // ...existing code...
};
```

**HTTP Endpoint:**
```bash
curl http://100.101.16.21:9001/service/metrics

# Response:
{
  "timestamp": "2026-01-08T15:30:45Z",
  "device_fps": 44.8,
  "host_fps": 43.2,
  "osc_fps": 30.0,
  "e2e_latency_ms": 58.3,
  "jitter_ms": 4.2,
  "vip1_uptime": 0.95,
  "vip2_uptime": 0.87,
  "id_switches": 3,
  "frames_dropped": 12,
  "osc_drops": 0
}
```

**Acceptance Criteria:**
- [ ] Metriken erfasst (FPS, Latenz, Jitter, etc.)
- [ ] HTTP-Endpoint funktioniert
- [ ] JSON-Output @ 1 Hz
- [ ] Operierbar (Prometheus/Grafana-ready)

---

### 5.3 Thread-Priorities (SCHED_FIFO)

**Status:** [ ] TODO  
**Aufwand:** 2 Stunden  
**Dateien:** `src/core/InputLoop.cpp`, `src/core/ProcessingLoop.cpp`

```cpp
// InputLoop.cpp - Höchste Priorität
void InputLoop::start() {
    if (_running) return;
    _running = true;
    _thread = std::thread(&InputLoop::loop, this);
    
    // Set SCHED_FIFO Priority 95
    pthread_t handle = _thread.native_handle();
    struct sched_param param;
    param.sched_priority = 95;
    if (pthread_setschedparam(handle, SCHED_FIFO, &param) != 0) {
        Logger::warn("Failed to set InputLoop priority (needs sudo)");
    }
    
    Logger::info("InputLoop started with SCHED_FIFO priority 95.");
}

// ProcessingLoop.cpp - Hohe Priorität
void ProcessingLoop::start() {
    // ...existing code...
    
    // Set SCHED_FIFO Priority 90
    pthread_t handle = _thread.native_handle();
    struct sched_param param;
    param.sched_priority = 90;
    pthread_setschedparam(handle, SCHED_FIFO, &param);
    
    Logger::info("ProcessingLoop started with SCHED_FIFO priority 90.");
}

// OscSender.cpp - Default Priority (keine Änderung)
```

**Acceptance Criteria:**
- [ ] InputLoop: SCHED_FIFO 95
- [ ] ProcessingLoop: SCHED_FIFO 90
- [ ] OscSender: Default
- [ ] Latenz-Jitter reduziert (< 5 ms)

**WICHTIG:** Braucht CAP_SYS_NICE oder sudo:
```bash
sudo setcap cap_sys_nice=eip ./HandTrackingV3
```

---

**Acceptance Criteria Phase 5:**
- [ ] Config-System funktioniert (JSON)
- [ ] Metriken messbar (/service/metrics)
- [ ] Thread-Priorities gesetzt
- [ ] Keine Regression (FPS, Latenz)

**Wenn erfolgreich:** ✅ Production-Ready System → **DEPLOYMENT**

---

## 📊 Erfolgs-Kriterien (Gesamt)

### SPEC.md (Single-Hand) - nach Phase 0
- [ ] FPS: 28-30 (ohne MJPEG-Clients)
- [ ] Latenz: < 80 ms
- [ ] CPU Load: < 20%
- [ ] Single-Hand Tracking stabil

### OPTIMAL_WORKFLOW (2 VIPs) - nach Phase 1-4
- [ ] **Device FPS: 43-45** (stabil)
- [ ] **Host FPS: 40-43** (Processing)
- [ ] **E2E Latenz: 50-70 ms**
- [ ] **Jitter: < 10 ms**
- [ ] **VIP1 Uptime: > 90%**
- [ ] **VIP2 Uptime: > 80%**
- [ ] **ID-Stabilität: > 95%** (30s Test)
- [ ] **Frame Drops: < 1%**

### Production-Ready - nach Phase 5
- [ ] Config via JSON (keine Hardcode-Values)
- [ ] Metriken messbar (HTTP-Endpoint)
- [ ] Thread-Priorities gesetzt
- [ ] Keine Crashes über 1 Stunde

---

## 🔧 Optionale Features (Nice-to-Have)

### A. XLinkIn Camera Control
**Status:** [ ] OPTIONAL  
**Priorität:** 🟢 NIEDRIG  
**Dauer:** 2-3 Tage

- [ ] XLinkIn Queue für CameraControl
- [ ] OSC Input: `/camera/focus/manual <value>`
- [ ] Runtime Focus/Exposure/WB ohne Neustart

### B. Device-side ROI (Phase 2b)
**Status:** [ ] OPTIONAL  
**Priorität:** 🟢 NIEDRIG  
**Dauer:** 2-3 Tage  
**Voraussetzung:** DepthAI Script-Node API stabil

- [ ] Script-Node statt Host-side Crop
- [ ] ImageManip mit dynamischer setCropRect()
- [ ] Latenz-Gewinn: -5 ms

### C. GPU Pre-Processing (YOLOv8n)
**Status:** [ ] OPTIONAL  
**Priorität:** 🟢 NIEDRIG  
**Dauer:** 1 Tag

- [ ] NPP Resize/Normalize (statt CPU cv2)
- [ ] Latenz-Gewinn: -2-3 ms (10 → 7-8 ms)

### D. Gesture-Voting (3-Frame Hysterese)
**Status:** [ ] OPTIONAL  
**Priorität:** 🟢 NIEDRIG  
**Dauer:** 1 Tag

- [ ] Gesture nur nach 3 konsistenten Frames
- [ ] Verhindert Flackern bei unklaren Posen

---

## 📝 Known Issues & Risks

### Critical Risks
1. **YOLOv8n TensorRT Conversion** (Phase 1)
   - Risiko: INT8 Calibration suboptimal → Accuracy-Drop > 5%
   - Mitigation: Gutes Calibration-Dataset (COCO Person-Subset)

2. **ObjectTracker ID-Stability** (Phase 1)
   - Risiko: ID-Swaps bei Okklusion
   - Mitigation: Tracker-Confidence Check + Fallback zu Detection

3. **PoE Bandwidth @ 45 FPS** (Phase 4)
   - Risiko: 720p/45 FPS > 1 Gbps
   - Mitigation: Preview klein halten (640x360)

### Medium Risks
4. **CMX Memory auf OAK-D** (Phase 1)
   - Risiko: Palm + Landmarks + Tracker > 2.5 MB
   - Mitigation: Tracker läuft parallel (kein NN)

5. **Thermal Throttling @ 15W** (Phase 4)
   - Risiko: GPU-Last zu hoch → Throttling
   - Mitigation: Power-Monitoring, ggf. Stereo reduzieren

### Low Risks
6. **Script-Node API instabil** (Optional B)
   - Risiko: Device-side ROI crasht
   - Mitigation: Host-side ROI als Phase 1 (funktioniert garantiert)

---

## 📅 Geschätzter Zeitplan (Best Case)

| Phase | Dauer | Start (ab heute) | Ende |
|-------|-------|------------------|------|
| **Phase 0** | 1 Tag | Tag 1 | Tag 1 |
| **Phase 1** | 5-7 Tage | Tag 2 | Tag 8 |
| **Phase 2** | 3-4 Tage | Tag 9 | Tag 12 |
| **Phase 3** | 2-3 Tage | Tag 13 | Tag 15 |
| **Phase 4** | 3-5 Tage | Tag 16 | Tag 20 |
| **Phase 5** | 2-3 Tage | Tag 21 | Tag 23 |

**Gesamt:** ~3-4 Wochen bis vollständiges OPTIMAL_WORKFLOW + Production-Ready

**Realistischer Zeitplan (mit Puffer):**
- Phase 0: **1 Tag** (Quick Wins)
- Phase 1-2: **2 Wochen** (Person Detection + ROI)
- Phase 3-5: **1-2 Wochen** (Depth + Optimization + Infra)

**Total: 4-5 Wochen** für vollständige Implementierung

---

## 🎯 Nächste Schritte (SOFORT)

### HEUTE starten:
1. ✅ **Phase 0.1: MJPEG hasClients() Check**
   - Datei: `include/net/MjpegServer.hpp`, `src/core/ProcessingLoop.cpp`
   - Dauer: 1-2 Stunden
   - Erwartung: +10 FPS sofort

2. ✅ **Phase 0.2: Stereo Throttling**
   - Datei: `src/core/ProcessingLoop.cpp`
   - Dauer: 30 Minuten
   - Erwartung: +5 FPS

3. ✅ **Phase 0.3-0.6: Restliche Quick Wins**
   - Dauer: 4 Stunden
   - Erwartung: 18 → 30 FPS erreicht

### Morgen:
4. **Phase 0 Testing**
   - Validierung: FPS stabil?
   - Wenn JA → Phase 1 starten

5. **Phase 1 Vorbereitung**
   - YOLOv8n Download
   - TensorRT Installation checken
   - Calibration-Dataset organisieren

---

## 📖 Referenzen

- **Architektur:** `docs/OPTIMAL_WORKFLOW_V2_FINAL.md`
- **Review:** `docs/OPTIMAL_WORKFLOW_V2_REVIEW.md`
- **Spezifikation:** `docs/SPECIFICATION.md`
- **Person Detection Config:** OPTIMAL_WORKFLOW_V2_REVIEW.md (Addendum)
- **Gap-Analyse:** `docs/COMPLETE_GAP_ANALYSIS.md`

---

**Status:** 🔴 Phase 0 NOT STARTED  
**Letzte Aktualisierung:** 2026-01-08  
**Nächste Review:** Nach Phase 0 (morgen)

---

**READY TO START.** 🚀

