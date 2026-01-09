# Copilot Instructions: DepthAI v3 High-Performance Service

## ⚠️ CRITICAL: READ FIRST
**Before making ANY changes, read and follow: `.github/copilot-working-rules.md`**

Key Rules:
1. **NO architecture changes without explicit approval**
2. **NO utility scripts bloat** (CLion handles deployment)
3. **STAY FOCUSED on the current TODO item**
4. **ASK before major changes**
5. **⚠️ NEVER COMMIT WITHOUT USER TESTING** - Wait for explicit "funktioniert" confirmation

---

## 🎯 Project Context (V3 Focus)

* **API:** DepthAI API Version 3 (v3). Strict requirement: do not use legacy `dai::node` Gen2 structures if v3 equivalents exist.
* **Hardware:** NVIDIA Jetson Orin Nano (8GB), OAK-D Pro PoE (Ethernet/PoE only; USB/libusb path is not used).
* **Low-Level:** Focus on direct buffer handling and zero-copy transfers between OAK and Jetson GPU (CUDA).
* **Dev Workflow:** Build and run on Jetson (user: `nvidia`); macOS host is IDE/remote frontend only. **Network:** Always use Tailscale IP `100.101.16.21`. **Build:** CLion Remote with Ninja.
* **References:** DepthAI v3 C++ API https://docs.luxonis.com/software-v3/depthai/api/cpp/ · depthai-core https://github.com/luxonis/depthai-core

---

### OAK-D Pro PoE Quick Reference (use this for decisions & code)

Keep this short reference at hand — it captures the device features and the engineering constraints we must respect when changing pipeline or runtime behaviour.

- Hardware summary
  - Stereo pair: OV9282 (1 MP monochrome, global-shutter) ×2 — high-FPS stereo source.
  - RGB: Sony IMX378 (12 MP) center color sensor; available in Auto-Focus or Fixed-Focus variants.
  - On-device compute: VPU capable of running on-device inference (approx. 4 TOPS total available on the device for models).
  - PoE: 802.3af over M12 X-coded GigE (1 Gbps) — video + power over a single cable.
  - Rugged: IP65 enclosure, M8 IO for GPIO/USB2/FSIN/STROBE.

- Typical, recommended modes for our service
  - Stereo mono capture: use 640x400 (THE_400_P) for left/right mono cameras when doing Jetson-side stereo matching. This is a good performance/accuracy balance.
  - RGB preview: use a small preview (e.g., 640x360) to reduce PoE bandwidth.
  - Neural nets: prefer NV12 image format where possible (zero-copy path) and ImageManip `LETTERBOX` for NN resizing — unwrap with `unletterbox()` on host.
  - NN threads: use 1 inference thread per Myriad NN for stability and performance.
  - ISP on-device: use ImageManip/ISP features for cropping/letterbox when possible to avoid host conversion.

- Stereo & Depth
  - Active IR dot projector + floodlight improve stereo on low-texture scenes.
  - Depth effective range typically ~0.7m to ~12m depending on scene and reflectivity.
  - For accurate depth: stream mono left/right to Jetson, compute stereo with CUDA (our `StereoKernel.cu`) and synchronize with landmark coordinates.
  - DO NOT disable Stereo Depth without explicit sign-off — it is a primary feature for 3D tracking.

- Camera controls & runtime config
  - Use DepthAI v3 `Pipeline::update` / CameraControl via XLinkIn to adjust exposure/focus at runtime (preferred). Do not set manual focus during pipeline construction unless explicitly required.
  - Prefer autofocus (CONTINUOUS_VIDEO) for general use; use manual/fixed focus only when repeatable, stable scene and after team approval.

- Networking & streaming
  - H.264/H.265/MJPEG are supported on-device — prefer not to encode on host if bandwidth permits; for preview use MJPEG only when clients connected (we implemented server-side skip to save CPU).
  - For zero-copy path: prefer `dai::Buffer` dma-buf handles and NV12 transfer.

- Power & thermal
  - Expected power: up to ~7.5W under heavy load (camera + IR + VPU). Respect Jetson MAXN/15W mode for stable performance.

- Practical rules for code changes (short)
  - Always prefer on-device ImageManip/ISP for resizing/cropping before sending to host.
  - Avoid `cv::cvtColor` on CPU in the hot path — use CUDA NPP or device-side formats.
  - Mono L/R must be preserved for stereo depth; do not remove mono streams from the pipeline without approval.
  - When in doubt: propose changes in a short plan and wait for approval.

---

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

## 📝 Documentation Standards

* **NO new .md files without explicit approval:** The `docs/` folder must stay clean and focused. Only create new documentation files if the user explicitly requests it.
* **Existing docs structure:** 
  - `OPTIMAL_WORKFLOW_V2_FINAL.md` - The architecture (frozen)
  - `OPTIMAL_WORKFLOW_V2_REVIEW.md` - The review (frozen)
  - `TODO.md` - Daily work plan (active)
  - `README.md` - Navigation (stable)
  - `OSC_GESTURE_REFERENCE.md` - Protocol reference (stable)
* **Update existing files instead:** If you need to document something, update `TODO.md` or ask where to put it.
* **No analysis/review files:** Don't create GAP_ANALYSIS.md, REVIEW.md, SUMMARY.md, etc. without explicit request.

## 🚀 Performance & Real-Time (C++17)

* **Memory Alignment:** Enforce strict memory alignment for Jetson. Use `std::aligned_alloc` for buffers shared between CPU and GPU.
* **OAK ISP v3:** Use advanced v3 ISP functions for hardware cropping and on-device warping (dewarping) to maintain minimal CPU load on the Jetson.
* **Thread-Safety:** Use `std::atomic` for status flags and high-performance `std::mutex` or lock-free structures for data exchange between the Video and OSC threads.

## 🖐 Tracking & VIP Logic

* **Real-Time Heuristics:** Calculate velocity and acceleration of hand landmarks directly within the C++ loop.
* **Stable VIP-ID:** Implement a Kalman filter or a low-pass filter for bounding box coordinates to prevent "jumping" in OSC output.

## 📡 OSC & Networking

* **Liblo V3 Integration:** Encapsulate OSC transmission to run asynchronously from the tracking loop. Network latency must never block camera inference.

---

## 🤖 TensorRT 10.x API (Jetson Orin - JetPack 6)

**CRITICAL:** The Jetson Orin runs **TensorRT 10.3+** (JetPack 6). The old binding-based APIs are **completely removed** in TensorRT 10.x. Always use the new tensor-based APIs:

### Removed → New API Mapping (TensorRT 10.x)

| ❌ Removed (TRT <10) | ✅ Required (TensorRT 10.x) |
|---------------------|------------------------|
| `createNetworkV2(kEXPLICIT_BATCH)` | `createNetworkV2(0)` - explicit batch is now default |
| `engine->getNbBindings()` | `engine->getNbIOTensors()` |
| `engine->getBindingName(i)` | `engine->getIOTensorName(i)` |
| `engine->bindingIsInput(i)` | `engine->getTensorIOMode(name) == TensorIOMode::kINPUT` |
| `engine->getBindingDimensions(i)` | `engine->getTensorShape(name)` |
| `context->executeV2(bindings)` | `context->setTensorAddress(name, ptr)` + `context->enqueueV3(stream)` |
| `context->enqueueV2(bindings, stream, events)` | `context->setTensorAddress(name, ptr)` + `context->enqueueV3(stream)` |

### Inference Pattern (TensorRT 10.x)

```cpp
// ✅ CORRECT: TensorRT 10.x inference
context_->setTensorAddress(inputInfo_.name.c_str(), d_input_);
context_->setTensorAddress(outputInfo_.name.c_str(), d_output_);
bool success = context_->enqueueV3(cudaStream);
cudaStreamSynchronize(cudaStream);  // CRITICAL: enqueueV3 is async!
```

### Common Pitfalls (TensorRT 10.x)

1. **Missing Stream Sync:** `enqueueV3()` is async - always call `cudaStreamSynchronize()` before reading output
2. **Wrong Tensor Names:** Use exact names from ONNX model, not positional indices
3. **Missing Null Checks:** Check `engine_`, `context_`, tensor names before use
4. **Buffer Size Mismatch:** Verify `inputInfo_.size` matches expected model input

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
* **Hot Path Discipline:** No allocations/locks in the inference hot path; use SPSC ringbuffer and `try_get`. Minimize logging in the hot path or only use trace level under feature flags.
* **Determinism:** Fixed seeds for filters/Kalman, constant sampling rates (30/60 Hz), Drop-Oldest backpressure strictly enforced.
* **Error Handling:** Fail early and clearly: check `dai`/OSC/network status, defensive null/size checks, clear error logs; avoid exceptions in hot path.
* **Tests (targeted):** Unit tests for filters (Kalman/One-Euro parameters), VIP-locking (30 Frames), and OSC queue drop policy; simple benchmarks for E2E latency.
* **No Magic Numbers:** Avoid "lucky numbers" in code. Use `constexpr` constants or a central `Config` JSON for runtime-tunable parameters (filter coefficients, timeouts).
* **Order and Hygiene:** No ad-hoc scripts or tests in the source root; helper scripts under `scripts/`, tests under `tests/`, docs under `docs/`.
