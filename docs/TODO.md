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
- [x] **Memory Management Strategy**
    - *Task:* Implement `AlignedAllocator` and buffer pool.
    - *Goal:* 256-byte alignment for DMA compatibility.
    - *Status:* Implemented `MemoryUtils.hpp`, `Frame.hpp`, and `FramePool.hpp`.
- [ ] **Zero-Copy Implementation**
    - *Status:* **Moved to Phase 5 (Optimization).** Currently using "One-Copy" strategy (Memcpy).

## Phase 3: Tracking Engine
- [x] **Hand Landmark Decoding**
    - *Task:* Parse NN output tensors into C++ structures.
    - *Status:* Implemented in `ProcessingLoop`. Fixed FP16/FP32 mismatch using OpenCV.
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
- [ ] **Debug Preview (MJPEG)**
    - *Task:* HTTP Stream with visual overlay (Skeleton, BBox, Gestures).
    - *Priority:* **High** (Essential for tuning).
- [ ] **Stereo Depth Integration**
    - *Task:* Add StereoDepth node to pipeline and sample Z-depth at landmarks.
    - *Priority:* **Medium** (Enhancement for absolute coordinates).
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
- **2026-01-05**: Initialized project structure. Created `SpscQueue` to ensure thread safety from the start. Defined `PipelineManager` interface to encapsulate DepthAI v3 logic.
