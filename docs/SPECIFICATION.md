# High-Performance Hand Tracking Service (V3) - Technical Specification
*   **Scheduling:** `SCHED_FIFO`.
*   **Nice Level:** -20.
*   **Service:** Systemd unit file.
## 5. Deployment

*   **Metrics:** `/service/metrics` endpoint reporting E2E latency, jitter, and drop counts (1Hz).
*   **Style:** `clang-format` (LLVM), `clang-tidy` (performance-*, readability-*).
*   **Warnings:** `-Wall -Wextra -Werror`.
*   **Compiler:** C++17 standard.
### 4.3 Quality Gates
*   **Compiler:** C++17 standard.
*   **Warnings:** `-Wall -Wextra -Werror`.
*   **Style:** `clang-format` (LLVM), `clang-tidy` (performance-*, readability-*).
*   **Metrics:** `/service/metrics` endpoint reporting E2E latency, jitter, and drop counts (1Hz).
*   **Time Sync:** Synchronize `dai::Clock` timestamps with host `std::chrono::system_clock` to measure accurate "Glass-to-OSC" latency. Use `dai::Message` timestamps as the source of truth.
*   **Configuration:** No magic numbers. All tunable parameters (Filter betas, Timeouts, IP/Ports) must be loaded from a configuration file (e.g., `config/settings.json`) or defined as named constants.

## 5. Deployment

*   **Metrics:** `/service/metrics` endpoint reporting E2E latency, jitter, and drop counts (1Hz).
*   **Style:** `clang-format` (LLVM), `clang-tidy` (performance-*, readability-*).
*   **Warnings:** `-Wall -Wextra -Werror`.
*   **Compiler:** C++17 standard.
### 4.3 Quality Gates

*   **Thread C (Network):** Standard Priority. Handles OSC serialization and sending.
*   **Thread B (Processing):** Priority FIFO 90. Runs filters and logic.
*   **Thread A (Input):** Priority FIFO 95. Handles OAK-D XLink streams.
### 4.2 Threading Model

*   **Format:** NV12/YUV420 preferred for zero-copy.
*   **Allocation:** Use `std::aligned_alloc` with RAII wrappers (`std::unique_ptr` with custom deleters).
*   **Alignment:** All shared buffers must be 64-byte aligned (cache lines) or 256-byte aligned (GPU DMA).
*   **Registration:** Register buffers with `cudaHostRegister` (Mapped | Portable) to enable zero-copy access by CUDA kernels on the Orin Nano (Unified Memory).
### 4.1 Memory Management

## 4. Detailed Requirements

    *   Backpressure: Drop-Oldest policy if latency > 50ms.
    *   Target Rate: 30Hz constant.
    *   Asynchronous transmission via `liblo`.
4.  **Output (OSC):**
        *   **Heuristics:** Velocity and acceleration calculation.
            *   Rotation/Landmarks: One-Euro Filter (Cutoff 1.0Hz, Beta 0.007).
            *   Position/BBox: Kalman Filter.
        *   **Filtering:**
        *   **VIP Locking:** Select primary hand after 15 consistent frames.
    *   **Tracking Engine:**
    *   **CUDA:** Color space conversion (if needed) or direct buffer access.
3.  **Processing (Jetson CPU/GPU):**
    *   Mapping to Jetson `dma-buf`.
    *   PCIe/Ethernet transfer of `dai::Buffer` (NV12).
2.  **Transport:**
    *   Neural Network Inference (Hand Landmarks) on-device.
    *   ISP Scaling to NN Input Resolution (e.g., 640x360) on-device.
    *   1080p Capture.
1.  **Acquisition (OAK-D):**
### 3.2 Pipeline Stages

*   **Thread Isolation:** Dedicated threads for Input (OAK), Processing (Inference/Filter), and Output (OSC) with prioritized scheduling.
*   **Lock-Free Concurrency:** Use of SPSC (Single-Producer-Single-Consumer) ring buffers for inter-thread communication. No mutexes in the hot path.
*   **Zero-Copy Data Path:** Direct transfer from OAK-D to Jetson memory (dma-buf). Use **CUDA Unified Memory** (`cudaHostRegister`) to allow GPU access to CPU-allocated buffers without `cudaMemcpy`.
### 3.1 Core Principles

## 3. Software Architecture

## 2. Hardware & Environment

*   **Compute:** NVIDIA Jetson Orin Nano (8GB RAM).
*   **Sensor:** Luxonis OAK-D Pro PoE (Ethernet).
*   **Network:** Tailscale VPN (Target: MacBook).
*   **OS:** Linux (Jetson JetPack), macOS (Remote Dev).

### 2.1 Models
The service uses the following pre-compiled blobs for the Myriad X VPU:
*   **Hand Landmarks:** `models/hand_landmark_full_sh4.blob`
    *   *Source:* MediaPipe Hand Landmark (Full), compiled with 4 Shaves.
    *   *Input:* 224x224 RGB.
    *   *Output:* 63 floats (21 landmarks x 3 coords).
*   **Palm Detection (Optional/Planned):** `models/palm_detection_sh4.blob`
    *   *Source:* MediaPipe Palm Detection.
    *   *Input:* 128x128 RGB.

# High-Performance Hand Tracking Service (V3) - Technical Specification

This service implements a high-performance, low-latency hand tracking pipeline on the NVIDIA Jetson Orin Nano using the Luxonis OAK-D Pro PoE camera. It leverages the DepthAI v3 API for data-centric processing and outputs tracking data via OSC (Open Sound Control) over a Tailscale VPN.

**Current Performance (2026-01-08):**
- **FPS:** 25-30 FPS @ 15W MAXN Mode
- **Latency Target:** < 50ms Glass-to-OSC (TBD measurement)
- **CPU Load Target:** < 20% (TBD measurement)
- **GPU Utilization:** Optimized for color conversion (NPP) only

## 1. System Overview

### 1.1 OAK-D Pro PoE - Hardware Quick Reference
Kurzreferenz der wichtigsten Hardware- und CV-Funktionen der OAK-D Pro PoE (für Entwickler):

- Prozessor & Kamera
  - RVC2/VPU-like compute (≈4 TOPS allokiert für on-device inference)
  - Stereo pair: OV9282 (1 MP monochrome, global shutter) ×2
  - RGB: Sony IMX378 (12 MP), variant: Auto-Focus or Fixed-Focus
  - Industrial IP65 enclosure, M12 X-coded PoE connector

- Capture & Framerate
  - Stereo Mono: up to 120 FPS (OV9282)
  - RGB: up to 60 FPS (IMX378)
  - Recommended for stereo processing in this project: mono @ 640x400 (THE_400_P) @ 30 FPS

- Depth & IR
  - Active IR dot projector + floodlight available to aid stereo
  - Depth effective range approx. 0.7m - 12m depending on scene and reflectivity

- Networking & Streaming
  - PoE (802.3af), GigE (1 Gbps)
  - On-device H.264/H.265/MJPEG encoding (use with care for CPU)

- I/O & Sensors
  - M8 connector: USB2, GPIO, FSIN, STROBE
  - Integrated 9-axis IMU (BNO085)

## 2. Pipeline & Data Flow

... (rest of file continues)
