# Development Plan & Status

## Phase 1: Foundation & Infrastructure
- [x] **Project Structure Setup**
    - *Why:* Establish standard C++ layout (include/src) and CMake configuration.
    - *Status:* Created basic folder structure (`include/core`, `src/core`, etc.).
- [x] **Core Utilities**
    - *Why:* Need lock-free communication before implementing threads.
    - *Status:* Implemented `SpscQueue.hpp` (Lock-free Ringbuffer) and `Logger.hpp`.
- [x] **Pipeline Manager Skeleton**
    - *Why:* Abstraction for DepthAI v3 device management.
    - *Status:* Implemented `PipelineManager` class. Migrated to correct v3 API pattern (build() and requestOutput()) to fix compilation errors.
- [x] **Logging & Error Handling**
    - *Why:* Essential for debugging on headless Jetson.
    - *Status:* Implemented `Logger` class.
- [ ] **Configuration System**
    - *Task:* Implement JSON config loader (e.g., using `nlohmann/json`).
    - *Goal:* Eliminate magic numbers. Allow runtime tuning of Filter/OSC parameters.

## Phase 2: Data Pipeline (Zero-Copy Focus)
- [x] **OAK-D Pipeline Configuration**
    - *Task:* Configure ColorCamera, ISP Scaling, and NeuralNetwork nodes.
    - *Goal:* Ensure 0% CPU load on Jetson for resizing.
    - *Status:* Implemented `PipelineManager` with `NeuralNetwork` and `Sync` nodes.
    - *Note (2026-01-06):* **StereoDepth removed from OAK-D** due to CMX memory constraints (~2.5MB limit). Running Palm Detection + Hand Landmarks NNs leaves no room for StereoDepth on the Myriad X chip.
- [x] **Memory Management Strategy**
    - *Task:* Implement `AlignedAllocator` and buffer pool.
    - *Goal:* 256-byte alignment for DMA compatibility.
    - *Status:* Implemented `MemoryUtils.hpp`, `Frame.hpp`, and `FramePool.hpp`. Added Mono L/R buffer support for future GPU stereo.
- [x] **Zero-Copy Implementation**
    - *Status:* Implemented GPU-accelerated Color Conversion (NPP) to reduce CPU load. Host->GPU is Zero-Copy via Pinned Memory.
    - *Note:* True Zero-Copy from OAK-D not possible (PoE data arrives in non-pinned memory). Single memcpy to pinned buffer is unavoidable.

## Phase 3: Tracking Engine
- [x] **Hand Landmark Decoding**
    - *Task:* Parse NN output tensors into C++ structures.
    - *Status:* Implemented in `ProcessingLoop`. Fixed FP16/FP32 mismatch using OpenCV.
- [x] **Palm Detection**
    - *Task:* Run Palm Detection NN to filter false positives.
    - *Status:* Implemented (2026-01-06). Runs on OAK-D alongside Landmarks NN.
- [x] **VIP Logic**
    - *Task:* Implement "Locking" mechanism (15 frames consistency check).
    - *Status:* Implemented in `ProcessingLoop`.
- [x] **Gesture Recognition**
    - *Task:* Implement heuristics for FIST, PINCH, FIVE, etc.
    - *Status:* Implemented in `ProcessingLoop` based on finger joint angles/distances.
- [x] **Filtering System**
    - [x] Kalman Filter (Position).
    - [ ] One-Euro Filter (Jitter reduction). *Note: Implemented class, but rotation logic pending.*

## Phase 4: Networking & OSC
- [x] **Liblo Integration**
    - *Task:* Wrap `liblo` in a C++ class (`OscSender`).
    - *Status:* Implemented `OscSender` class.
- [x] **OSC Message Builder**
    - *Task:* Serialize tracking data to OSC blobs.
    - *Status:* Implemented in `OscSender::send`. Includes Gesture ID/Name.
- [x] **Backpressure Handling**
    - *Task:* Implement "Drop-Oldest" logic in the network thread.
    - *Status:* Implemented latency check in `OscSender` and non-blocking push in `ProcessingLoop`.

## Phase 5: Optimization & Metrics
- [x] **Debug Preview (MJPEG)**
    - *Task:* HTTP Stream with visual overlay (Skeleton, BBox, Gestures).
    - *Priority:* **High** (Essential for tuning).
    - *Status:* Implemented MjpegServer and overlay in ProcessingLoop.
- [ ] **Stereo Depth Integration (Jetson GPU)**
    - *Task:* Stream Mono L/R from OAK-D, compute depth on Jetson via CUDA/OpenCV GPU.
    - *Priority:* **Medium** (Enhancement for absolute Z coordinates).
    - *Note (2026-01-06):* **Architecture Change** - StereoDepth cannot run on OAK-D alongside 2 NNs due to CMX memory limits. Depth will be computed on Jetson GPU instead. Mono camera streaming infrastructure prepared in `Frame.hpp` and `FramePool.hpp`.
- [ ] **Performance Monitor**
    - *Task:* Measure Glass-to-OSC latency.
- [ ] **Systemd Integration**
    - *Task:* Create service file with Realtime priorities.
- [ ] **Final Profiling**
    - *Task:* Verify CPU < 10% and Latency < 30ms.
- [ ] **Configuration System**
    - *Task:* Implement JSON config loader.

---

## Change Log / Decisions
- **2026-01-06**: 
  - **Major Architecture Change:** StereoDepth removed from OAK-D pipeline due to E_OUT_OF_MEM errors. The Myriad X chip (~2.5MB CMX) cannot fit StereoDepth + Palm Detection + Hand Landmarks simultaneously.
  - **New Architecture:** 
    - OAK-D: RGB Camera + Palm Detection NN + Hand Landmarks NN (stable)
    - Jetson: GPU-based Stereo Matching (TODO) + Processing + OSC Output
  - Palm Detection re-enabled after testing minimal pipeline stability.
  - Fixed CUDA buffer registration to handle already-registered buffers gracefully.
  - Reduced NN threads to 1 each to maximize stability.
- **2026-01-05**: Initialized project structure. Created `SpscQueue` to ensure thread safety from the start. Defined `PipelineManager` interface to encapsulate DepthAI v3 logic.
