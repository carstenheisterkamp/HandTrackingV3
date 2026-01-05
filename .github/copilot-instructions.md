# Copilot Instructions: DepthAI v3 High-Performance Service

## 🎯 Project Context (V3 Focus)

* **API:** DepthAI API Version 3 (v3). **Strict requirement:** Do not use legacy `dai::node` Gen2 structures if v3 equivalents exist.
* **Hardware:** NVIDIA Jetson Orin Nano (8GB), OAK-D Pro PoE (Ethernet/PoE only; USB/libusb path is not used).
* **Low-Level:** Focus on direct buffer handling and zero-copy transfers between OAK and Jetson GPU (CUDA).
* **Dev Workflow:** Build and run on Jetson (User: `nvidia`); macOS host is IDE/remote frontend only. **Network:** Always use Tailscale IP `100.101.16.21`. **Build:** CLion Remote with Ninja.
* **References:** DepthAI v3 C++ API https://docs.luxonis.com/software-v3/depthai/api/cpp/ · depthai-core https://github.com/luxonis/depthai-core

## 🏗 Build Environment & Stability

* **User:** `nvidia` (Home: `/home/nvidia`).
* **Generator:** Use `Ninja` (via CLion Remote Host).
* **Cache Management:**
    * **Stale Cache:** If linking errors occur (especially undefined symbols from `depthai-core` or `spdlog`), aggressively clear the remote CMake cache.
    * **Command:** `rm -rf /home/nvidia/dev/HandTrackingV3/cmake-build-debug-remote-host/*` before re-running CMake.
    * **Library Conflicts:** Ensure no static libraries (`.a`) remain in `/usr/local/lib` if shared libraries (`.so`) are expected.

## 🛠 DepthAI v3 Coding Standards

* **Resource Management:** Utilize the new v3 `Device` and `Pipeline` managers.
* **Data-Centric Flow:** Implement workflows based on v3 `Message` types. Prefer direct metadata manipulation on the device.
* **Pipeline Dynamics:** Leverage v3 capabilities to reconfigure pipeline parts (e.g., VIP switching) at runtime without device resets.

## 🚀 Performance & Real-Time (C++17)

* **Memory Alignment:** Enforce strict memory alignment for Jetson. Use `std::aligned_alloc` for buffers shared between CPU and GPU.
* **OAK ISP v3:** Use advanced v3 ISP functions for hardware cropping and on-device warping (dewarping) to maintain 0% CPU load on the Jetson.
* **Thread-Safety:** Use `std::atomic` for status flags and high-performance `std::mutex` or lock-free structures for data exchange between the Video and OSC threads.

## 🖐 Tracking & VIP Logic

* **Real-Time Heuristics:** Calculate velocity () and acceleration of hand landmarks directly within the C++ loop.
* **Stable VIP-ID:** Implement a Kalman filter or a low-pass filter for bounding box coordinates to prevent "jumping" in OSC output.

## 📡 OSC & Networking

* **Liblo V3 Integration:** Encapsulate OSC transmission to run asynchronously from the tracking loop. Network latency must never block camera inference.

---

## ⚡ Advanced Implementation Rules (DepthAI v3 & Jetson Orin)

### 1. API Clarity & v3 Paradigm

* **DO:** Use `dai::Pipeline` with the v3 resource model. Use `std::shared_ptr<dai::Message>` for data exchange.
* **DON'T:** Use any `dai::node::...` classes from Gen2.
* **Dynamic Reconfig:** Implement adjustments (FPS/Exposure) via `Pipeline::update`.
* *Example:* `pipeline.update(camera_node_id, {{"fps", 60}});` without calling `device.close()`.

### 2. Zero-Copy & Memory Path

* **Target Path:** OAK-D (NV12/YUV420) → PCIe → Jetson dma-buf → CUDA Kernel.
* **Checklist:**
* Use `dai::Buffer` with dma-buf handles for GPU inference.
* **Profiling Target:** GPU Utilization > 90%, CPU Load < 10% (orchestration only).
* Avoid `cv::cvtColor` on CPU; use CUDA kernels for color space conversion.

### 3. Alignment & Allocation

* **Sizes:** Use **64B alignment** for cache-line optimization and **256B** for GPU DMA transfers.
* **Implementation:** Use `void* ptr = std::aligned_alloc(256, size);`.
* **RAII:** Wrap allocations in `std::unique_ptr<T, decltype(&std::free)> myPtr(ptr, &std::free);`.

### 4. Threading & Synchronization

* **Data Flow:** Implement an **SPSC (Single-Producer-Single-Consumer) Ringbuffer** for video frames.
* **Blocking:** Strictly forbid `get()` (blocking) in the inference thread. Use callbacks or `try_get()`.
* **OSC Isolation:** Run OSC transmission in a dedicated `std::thread` with lower priority than the inference thread (Policy: "Drop Oldest" on backpressure).

### 5. VIP-Filtering & Stabilization

* **Sampling:** 30Hz or 60Hz (synchronized with camera FPS).
* **Smoothing:** Kalman filter for 3D position; One-Euro filter for hand rotation (Cutoff 1.0Hz, Beta 0.007).
* **VIP Locking:** A VIP is considered "Locked" only after **15 consistent frames** (~0.5s).

### 6. OSC & Backpressure

* **Latency Limit:** Discard packets older than **50ms**.
* **Target Rate:** Constant 30Hz (decoupled from camera inference fluctuations).
* **Format:** Use compact blobs or bundled messages to reduce overhead.

### 7. Profiling & Metrics (Mandatory)

Implement a `PerformanceMonitor` transmitting the following via `/service/metrics`:

* **E2E Latency:** Glass-to-OSC timestamping.
* **Jitter:** Standard deviation of OSC transmission intervals.
* **Drops:** Count of discarded frames/packets.
* **Tools:** Monitor via `tegrastats` (Power/Thermal) and `perf` (CPU hotspots).

## ✅ Best Practices & Code Quality

* **Tooling:** Run `clang-format` (LLVM style) and `clang-tidy` (enable performance-* and readability-* checks) on changes; compile with `-Wall -Wextra -Werror` on Jetson.
* **RAII & Safety:** No raw new/delete; prefer RAII wrappers and `std::unique_ptr`/`std::shared_ptr`. Guard hot-path pointers with assertions in debug builds.
* **Hot Path Discipline:** Keine Allokationen/Locks im Inferenz-Hotpath; SPSC-Ringbuffer, `try_get`, keine Blocker. Logging im Hotpath minimieren oder auf Trace-Level per Feature-Flag.
* **Determinismus:** Fixe Seeds für Filter/Kalman, konstante Sampling-Raten (30/60 Hz), Drop-Oldest Backpressure strikt durchsetzen.
* **Error Handling:** Früh und klar failen: `Dai`/OSC/Netzwerk-Status prüfen, defensive Checks auf Null/Größe, klare Fehler-Logs; Exceptions im Hotpath vermeiden.
* **Tests (gezielt):** Unit-Tests für Filter (Kalman/One-Euro Parameter), VIP-Locking (15 Frames), und OSC-Queue-Drop-Policy; einfache Benchmark/Timing-Probe für E2E-Latenz.
* **No Magic Numbers:** Vermeide "Lucky Numbers" im Code. Nutze `constexpr` Konstanten für Compile-Time Werte oder eine zentrale `Config`-Klasse/JSON für Runtime-Parameter (z.B. Filter-Koeffizienten, Timeouts).
* **Ordnung:** Keine ad-hoc Skripte oder Tests im Source-Root; Hilfsskripte unter `scripts/`, Tests unter `tests/`, Dokumentation unter `docs/` ablegen.
