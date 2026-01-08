#include "core/ProcessingLoop.hpp"
#include "core/SystemMonitor.hpp"
#include <chrono>
#include <algorithm> // Added for std::clamp
#include <opencv2/imgproc.hpp> // For drawing

#ifdef ENABLE_CUDA
#include "core/StereoKernel.hpp" // Custom CUDA Kernel
#include <nppi_color_conversion.h>
#include <cuda_runtime.h>
#endif

namespace core {

ProcessingLoop::ProcessingLoop(std::shared_ptr<AppProcessingQueue> inputQueue,
                               std::shared_ptr<AppFramePool> framePool,
                               std::shared_ptr<OscQueue> oscQueue)
    : _inputQueue(std::move(inputQueue)),
      _framePool(std::move(framePool)),
      _oscQueue(std::move(oscQueue)),
      _running(false) {

    // Initialize 21 Kalman filters for hand landmarks
    _landmarkFilters.resize(21);

#ifdef ENABLE_CUDA
    // Initialize StereoBM
    // 64 disparities, blockSize 7 is a good starting point for 640x400
    /*
    try {
        _stereoBM = cv::cuda::createStereoBM(64, 7);
    } catch(const cv::Exception& e) {
        Logger::error("Failed to initialize CUDA StereoBM: ", e.what());
    }
    */
#endif

    // Initialize MJPEG Server
    _mjpegServer = std::make_unique<net::MjpegServer>(8080);

    // Initialize performance cache
    _performanceSummary = SystemMonitor::getPerformanceSummary();
    _lastPerfUpdate = std::chrono::steady_clock::now();
}

ProcessingLoop::~ProcessingLoop() {
    stop();
}

void ProcessingLoop::start() {
    if (_running) return;
    _running = true;
    _mjpegServer->start();
    Logger::info("MJPEG Preview available at http://100.101.16.21:8080");
    _lastFpsTime = std::chrono::steady_clock::now();
    _thread = std::thread(&ProcessingLoop::loop, this);
    Logger::info("ProcessingLoop started.");
}

void ProcessingLoop::stop() {
    if (!_running) return;
    _running = false;
    _mjpegServer->stop();
    if (_thread.joinable()) {
        _thread.join();
    }
    Logger::info("ProcessingLoop stopped.");
}

bool ProcessingLoop::isRunning() const {
    return _running;
}

void ProcessingLoop::loop() {
    while (_running) {
        Frame* frame = nullptr;
        if (_inputQueue->pop_front(frame)) {
            if (frame) {
                processFrame(frame);
                // Return frame to pool after processing
                _framePool->release(frame);
            }
        } else {
            // Busy wait or sleep?
            // For low latency, busy wait might be preferred, but let's be nice to CPU for now if queue is empty
            // Or use a condition variable if we change SpscQueue to support it, but SpscQueue is lock-free.
            // A short sleep is acceptable if we are faster than camera.
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
}

// Configuration for Interaction Box (in mm if using Depth, or relative if using 2D)
// We will assume 2D normalized for X/Y for now, and pseudo-Z.
// If StereoDepth is added later, these should match the mm values.
constexpr float BOX_MIN_X = 0.0f;
constexpr float BOX_MAX_X = 1.0f; // Image width
constexpr float BOX_MIN_Y = 0.0f;
constexpr float BOX_MAX_Y = 1.0f; // Image height
// Z depth logic
constexpr float BOX_MIN_Z = 200.0f; // 20cm
constexpr float BOX_MAX_Z = 1500.0f; // 1.5m

// Helper to map letterboxed NN coordinates back to original frame
void unletterbox(float& x, float& y, float nnSize, float origW, float origH) {
    if (x < 0.001f && y < 0.001f) return; // Ignore 0,0 garbage

    float scale = std::min(nnSize / origW, nnSize / origH);
    float newW = origW * scale;
    float newH = origH * scale;

    float offX = (nnSize - newW) / 2.0f;
    float offY = (nnSize - newH) / 2.0f;

    // Map normalized (0..1) to pixel on NN (0..nnSize)
    float px = x * nnSize;
    float py = y * nnSize;

    // Remove offset
    px -= offX;
    py -= offY;

    // Scale back to original
    x = px / newW; // 0..1 relative to original width
    y = py / newH; // 0..1 relative to original height

    // Clamp
    x = std::clamp(x, 0.0f, 1.0f);
    y = std::clamp(y, 0.0f, 1.0f);
}

float normalize(float value, float min, float max) {
    return std::clamp((value - min) / (max - min), 0.0f, 1.0f);
}

void ProcessingLoop::processFrame(Frame* frame) {
    // FPS Calculation
    _frameCount++;
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - _lastFpsTime).count();
    if (elapsed >= 1000) {
        _currentFps = _frameCount * 1000.0f / elapsed;
        _frameCount = 0;
        _lastFpsTime = now;
    }

    // Access frame->data for pixel data (NV12)
    // Access frame->width, frame->height, etc.

    // Prepare Debug Frame (RGB) - ONLY if MJPEG clients are connected
    // PHASE 0 OPTIMIZATION: Skip expensive color conversion when no viewers
    cv::Mat debugFrame;
    bool shouldRenderDebug = _mjpegServer && _mjpegServer->hasClients();

    if (shouldRenderDebug) {
        // Ensure BGR buffer is allocated and registered
        size_t requiredSize = frame->width * frame->height * 3;
        if (!_bgrBuffer || _bgrBufferSize < requiredSize || _bgrWidth != frame->width || _bgrHeight != frame->height) {
            _bgrBuffer = allocate_aligned<uint8_t>(requiredSize);
            _bgrBufferSize = requiredSize;
            _bgrWidth = frame->width;
            _bgrHeight = frame->height;
            register_buffer_cuda(_bgrBuffer.get(), requiredSize);
        }

#ifdef ENABLE_CUDA
        // Try CUDA Color Conversion
        void* srcDev = get_device_pointer(frame->data.get());
        void* dstDev = get_device_pointer(_bgrBuffer.get());

        if (srcDev && dstDev) {
            NppiSize oSizeROI = {(int)frame->width, (int)frame->height};
            NppStatus status = NPP_ERROR;

            if (frame->type == 22) { // NV12
                const Npp8u* pSrc[2];
                pSrc[0] = (const Npp8u*)srcDev;
                pSrc[1] = (const Npp8u*)srcDev + frame->width * frame->height;
                int nSrcStep = (int)frame->width;
                int nDstStep = (int)frame->width * 3;
                status = nppiNV12ToBGR_8u_P2C3R(pSrc, nSrcStep, (Npp8u*)dstDev, nDstStep, oSizeROI);
            } else if (frame->type == 2) { // YUV420p (I420)
                const Npp8u* pSrc[3];
                size_t ySize = frame->width * frame->height;
                size_t uvSize = ySize / 4;
                pSrc[0] = (const Npp8u*)srcDev;
                pSrc[1] = (const Npp8u*)srcDev + ySize;
                pSrc[2] = (const Npp8u*)srcDev + ySize + uvSize;
                int nSrcStep[3] = {(int)frame->width, (int)frame->width/2, (int)frame->width/2};
                int nDstStep = (int)frame->width * 3;
                status = nppiYUV420ToBGR_8u_P3C3R(pSrc, nSrcStep, (Npp8u*)dstDev, nDstStep, oSizeROI);
            } else {
                // Try NV12 as default if type is unknown (0) or other
                const Npp8u* pSrc[2];
                pSrc[0] = (const Npp8u*)srcDev;
                pSrc[1] = (const Npp8u*)srcDev + frame->width * frame->height;
                int nSrcStep = (int)frame->width;
                int nDstStep = (int)frame->width * 3;
                status = nppiNV12ToBGR_8u_P2C3R(pSrc, nSrcStep, (Npp8u*)dstDev, nDstStep, oSizeROI);
            }

            if (status == NPP_NO_ERROR) {
                cudaStreamSynchronize(0); // Wait for GPU
                debugFrame = cv::Mat((int)frame->height, (int)frame->width, CV_8UC3, _bgrBuffer.get());
            } else {
                Logger::error("NPP Color Conversion failed: ", status, " Type: ", frame->type);
                // Don't return - continue processing without debug overlay
                shouldRenderDebug = false;
            }
        } else {
            Logger::error("CUDA device pointers NULL - check CUDA registration!");
            // Don't return - continue processing without debug overlay
            shouldRenderDebug = false;
        }
#else
        // NO CUDA - Use CPU (will be slow)
        cv::Mat nv12(frame->height * 3 / 2, frame->width, CV_8UC1, frame->data.get());
        cv::cvtColor(nv12, debugFrame, cv::COLOR_YUV2BGR_NV12);
#endif

        if (debugFrame.empty()) {
            Logger::warn("Debug frame empty - skipping overlay");
            shouldRenderDebug = false;
        }
    }

    // --- Palm Detection Gating ---
    // Since we are running the Landmark model on the full frame (without cropping),
    // we MUST gate it with the Palm Detector to avoid false positives.
    bool palmDetected = false;
    if (!frame->palmData.empty()) {
        // MediaPipe Palm Detection output stride is typically 18 per anchor
        size_t stride = 18;
        if (frame->palmData.size() >= stride) {
            int numDetections = static_cast<int>(frame->palmData.size() / stride);
            numDetections = std::min(numDetections, 20); // Cap
            for (int i = 0; i < numDetections; ++i) {
                float rawScore = frame->palmData[i * stride + 0];

                // CRITICAL FIX: Blob outputs RAW LOGITS (negative values like -14.0)
                // Apply sigmoid to convert to probability [0, 1]
                float score = 1.0f / (1.0f + std::exp(-rawScore));

                if (score > 0.8f) { // Confidence threshold increased to 0.8 (was 0.5) to avoid false positives
                    palmDetected = true;
                    break;
                }
            }
        }
    }

    // If no palm detected, discard landmarks (unless we are already locked and tracking?)
    // For now, strict gating to fix "False Positives".
    if (!palmDetected) {
        if (_lockCounter > 0) _lockCounter--;
        if (_lockCounter == 0) {
            _vipLocked = false;
            _hasLastState = false;
             for(auto& kf : _landmarkFilters) kf.reset();
        }

        // Draw debug overlay and return early (only if clients connected)
        if (shouldRenderDebug) {
            drawDebugOverlay(debugFrame, frame, {}, "", 0, 0.0f);
        }
        return;
    }

    // Parse landmarks (21 points * 3 coords = 63 floats)
    if (frame->nnData.size() < 63) {
        static int warnCounter = 0;
        if (warnCounter++ % 300 == 0) {
           Logger::warn("ProcessingLoop: Invalid NN data size: ", frame->nnData.size(), " (Expected 63)");
        }
        // Draw debug overlay and return early (only if clients connected)
        if (shouldRenderDebug) {
            drawDebugOverlay(debugFrame, frame, {}, "", 0, 0.0f);
        }
        return;
    }

    // Prepare result
    TrackingResult result;
    // result.vipLocked will be set at the end
    result.landmarks.reserve(21);

    std::vector<math::KalmanFilter::Point3f> currentLandmarks;
    currentLandmarks.reserve(21);

#ifdef ENABLE_CUDA
    // GPU Stereo Depth Computation - THROTTLED for performance (Phase 0)
    // Depth changes slower than hand position, so compute every 3rd frame
    static int stereoCounter = 0;
    if (++stereoCounter % 3 == 0 && frame->hasStereoData && frame->monoWidth > 0 && frame->monoLeftData && frame->monoRightData) {
       computeStereoDepth(frame->monoLeftData.get(),
                          frame->monoRightData.get(),
                          (uint16_t*)frame->depthData.get(),
                          static_cast<int>(frame->monoWidth),
                          static_cast<int>(frame->monoHeight));

        frame->depthWidth = frame->monoWidth;
        frame->depthHeight = frame->monoHeight;

        // Sync GPU before reading depth
        cudaStreamSynchronize(0);
    }
    // Use cached depth for frames where we didn't compute
#endif

    // Process each landmark
    for (int i = 0; i < 21; ++i) {
        float rawX = frame->nnData[i * 3 + 0];
        float rawY = frame->nnData[i * 3 + 1];
        float rawZ = frame->nnData[i * 3 + 2]; // Relative depth from NN model

        // DEBUG: Log first landmark values once
        static bool landmarkValuesLogged = false;
        if (!landmarkValuesLogged && i == 0) {
            Logger::info("Raw Landmark 0 (Wrist): X=", rawX, " Y=", rawY, " Z=", rawZ);
            Logger::info("Frame size: ", frame->width, "x", frame->height);
            landmarkValuesLogged = true;
        }

        // CRITICAL: Hand Landmark model outputs PIXEL coordinates (0-224)
        // NOT normalized! We need to normalize to (0-1) range
        // BUT: Model was fed LETTERBOXED 224x224 from 1920x1080
        // So we need to map: 224x224 letterbox → 1920x1080 original

        // Step 1: Normalize from pixels to 0-1 relative to 224x224
        if (rawX > 1.0f || rawY > 1.0f) {
            rawX /= 224.0f;
            rawY /= 224.0f;
        }

        // Step 2: Unletterbox - map from 224x224 (letterboxed) back to original aspect ratio
        // ImageManip used LETTERBOX mode to fit 1920x1080 into 224x224
        unletterbox(rawX, rawY, 224.0f, (float)frame->width, (float)frame->height);

        // Use computed depth if available
        if (frame->depthData && frame->depthWidth > 0) {
            int u = std::clamp((int)(rawX * frame->depthWidth), 0, (int)frame->depthWidth - 1);
            int v = std::clamp((int)(rawY * frame->depthHeight), 0, (int)frame->depthHeight - 1);

            uint16_t* depthMap = (uint16_t*)frame->depthData.get();
            uint16_t distMM = depthMap[v * frame->depthWidth + u];

            if (distMM > 0) {
                 rawZ = (float)distMM;
            }
        }

        // Apply One-Euro Filter
        auto smoothed = _landmarkFilters[i].update(rawX, rawY, rawZ);
        currentLandmarks.push_back(smoothed);

        // Normalize and add to result
        TrackingResult::NormalizedPoint np = {0.0f, 0.0f, 0.0f};
        np.x = normalize(smoothed.x, BOX_MIN_X, BOX_MAX_X);
        np.y = normalize(smoothed.y, BOX_MIN_Y, BOX_MAX_Y);
        np.z = normalize(smoothed.z, BOX_MIN_Z, BOX_MAX_Z); // Metric Normalized Z

        result.landmarks.push_back(np);
    }

    // Calculate Velocity (Wrist = index 0)
    if (_hasLastState) {
        auto dt = std::chrono::duration<float>(frame->timestamp - _lastHandState.timestamp).count();
        if (dt > 0.0001f) { // Avoid division by zero
            // Calculate velocity of wrist (index 0)
            float vx = (currentLandmarks[0].x - _lastHandState.landmarks[0].x) / dt;
            float vy = (currentLandmarks[0].y - _lastHandState.landmarks[0].y) / dt;
            float vz = (currentLandmarks[0].z - _lastHandState.landmarks[0].z) / dt;

            // Store for overlay display
            _lastVelocity.x = vx;
            _lastVelocity.y = vy;
            _lastVelocity.z = vz;

            // Log velocity occasionally
            static int logCounter = 0;
            if (++logCounter % 30 == 0) {
                Logger::info("Wrist Velocity: [", vx, ", ", vy, ", ", vz, "]");
            }
        }
    }

    // VIP Locking: Increment counter when hand is consistently tracked
    if (_lockCounter < LOCK_THRESHOLD) {
        _lockCounter++;
        if (_lockCounter >= LOCK_THRESHOLD) {
            _vipLocked = true;
            Logger::info("VIP LOCKED after ", LOCK_THRESHOLD, " consistent frames");
        }
    }

    // Update state
    _lastHandState.landmarks = currentLandmarks;
    _lastHandState.timestamp = frame->timestamp;
    _hasLastState = true;

    // Gesture Recognition
    // Heuristics based on OSC_GESTURE_REFERENCE.md

    float pinchDist = 0.0f;
    float fistDist = 0.0f;
    std::string gestureName = "unknown";
    int gestureId = 0;

    if (currentLandmarks.size() >= 21) {
        auto dist2D = [](const math::KalmanFilter::Point3f& a, const math::KalmanFilter::Point3f& b) {
            return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
        };

        // Key Landmarks (MediaPipe Hand Landmark indices)
        // 0=Wrist, 1-4=Thumb(CMC,MCP,IP,TIP), 5-8=Index(MCP,PIP,DIP,TIP), etc.
        auto wrist = currentLandmarks[0];

        auto thumbCmc = currentLandmarks[1];
        auto thumbMcp = currentLandmarks[2];
        auto thumbIp = currentLandmarks[3];
        auto thumbTip = currentLandmarks[4];

        auto indexMcp = currentLandmarks[5];
        auto indexPip = currentLandmarks[6];
        auto indexDip = currentLandmarks[7];
        auto indexTip = currentLandmarks[8];

        auto middleMcp = currentLandmarks[9];
        auto middlePip = currentLandmarks[10];
        auto middleDip = currentLandmarks[11];
        auto middleTip = currentLandmarks[12];

        auto ringMcp = currentLandmarks[13];
        auto ringPip = currentLandmarks[14];
        auto ringDip = currentLandmarks[15];
        auto ringTip = currentLandmarks[16];

        auto pinkyMcp = currentLandmarks[17];
        auto pinkyPip = currentLandmarks[18];
        auto pinkyDip = currentLandmarks[19];
        auto pinkyTip = currentLandmarks[20];

        // Robust Finger State Detection:
        // A finger is EXTENDED if its TIP is further from the palm center (MCP) than its PIP
        // Using Y-coordinate comparison (smaller Y = higher on screen = extended)
        // Note: In image coordinates, Y increases downward

        // For non-thumb fingers: TIP.y < PIP.y means finger is extended (pointing up relative to palm)
        // But this depends on hand orientation. A simpler check:
        // Extended = TIP is further from wrist than MCP (in 2D)
        bool indexOpen = dist2D(indexTip, wrist) > dist2D(indexMcp, wrist) * 1.15f;  // REDUCED from 1.3
        bool middleOpen = dist2D(middleTip, wrist) > dist2D(middleMcp, wrist) * 1.15f;
        bool ringOpen = dist2D(ringTip, wrist) > dist2D(ringMcp, wrist) * 1.15f;
        bool pinkyOpen = dist2D(pinkyTip, wrist) > dist2D(pinkyMcp, wrist) * 1.15f;

        // Thumb is special: check if TIP is far from index MCP (spread out)
        bool thumbOpen = dist2D(thumbTip, indexMcp) > 0.05f;  // REDUCED from 0.08

        // Pinch detection: Thumb Tip close to Index Tip
        float pinchDist2D = dist2D(thumbTip, indexTip);
        pinchDist = static_cast<float>(pinchDist2D);

        // Fist detection: All tips close to palm center
        auto palmCenter = currentLandmarks[9]; // Middle MCP as palm center
        fistDist = (dist2D(indexTip, palmCenter) + dist2D(middleTip, palmCenter) +
                    dist2D(ringTip, palmCenter) + dist2D(pinkyTip, palmCenter)) / 4.0f;

        // Thresholds (relaxed for better detection)
        const float PINCH_THRESH = 0.08f;  // INCREASED from 0.06
        const float FIST_THRESH = 0.15f;   // INCREASED from 0.12

        // Gesture Logic (priority order)
        if (fistDist < FIST_THRESH && !thumbOpen) {
            gestureId = 2; gestureName = "FIST";
        }
        else if (pinchDist < PINCH_THRESH) {
            if (middleOpen && ringOpen && pinkyOpen) {
                gestureId = 1; gestureName = "OK";
            } else {
                gestureId = 1; gestureName = "PINCH";
            }
        }
        else if (thumbOpen && indexOpen && middleOpen && ringOpen && pinkyOpen) {
            // Check for VULCAN (Gap between Middle and Ring)
            float midRingDist = dist2D(middleTip, ringTip);
            if (midRingDist > 0.1f) {
                 gestureId = 10; gestureName = "VULCAN";
            } else {
                 gestureId = 3; gestureName = "FIVE";
            }
        }
        else if (indexOpen && !middleOpen && !ringOpen && !pinkyOpen) {
            if (thumbOpen) {
                gestureId = 5; gestureName = "GUN"; // L-Shape / Two
            } else {
                gestureId = 4; gestureName = "POINTING";
            }
        }
        else if (indexOpen && middleOpen && !ringOpen && !pinkyOpen) {
            gestureId = 6; gestureName = "PEACE";
        }
        else if (thumbOpen && pinkyOpen && !indexOpen && !middleOpen && !ringOpen) {
            gestureId = 7; gestureName = "CALL_ME";
        }
        else if (indexOpen && pinkyOpen && !middleOpen && !ringOpen) {
            if (thumbOpen) {
                gestureId = 9; gestureName = "LOVE_YOU";
            } else {
                gestureId = 8; gestureName = "METAL";
            }
        }
        else if (thumbOpen && !indexOpen && !middleOpen && !ringOpen && !pinkyOpen) {
            gestureId = 11; gestureName = "THUMBS_UP";
        }
        else if (middleOpen && !indexOpen && !ringOpen && !pinkyOpen) {
            gestureId = 12; gestureName = "MIDDLE_FINGER";
        }
        else {
            gestureId = 0; gestureName = "---";
        }

        // Log Gesture occasionally
        static int gestureLogCounter = 0;
        if (++gestureLogCounter % 90 == 0) {  // REDUCED frequency from 60 to 90
            Logger::info("Gesture: ", gestureName, " | Fingers: T:", thumbOpen,
                         " I:", indexOpen, " M:", middleOpen, " R:", ringOpen, " P:", pinkyOpen,
                         " | Pinch:", pinchDist, " Fist:", fistDist);
        }
    }

    // Draw Debug Overlay (only if clients connected)
    if (shouldRenderDebug) {
        drawDebugOverlay(debugFrame, frame, currentLandmarks, gestureName, gestureId, pinchDist);
    }

    // If no landmarks were found or valid, return now, but after drawing the overlay.
    // This allows the user to see the video and palm boxes even if landmarks fail.
    if (frame->nnData.size() < 63) {
        return;
    }

    // Update result for OSC
    result.vipLocked = _vipLocked;
    result.timestamp = frame->timestamp;
    result.pinchDistance = pinchDist;
    result.gestureId = gestureId;
    result.gestureName = gestureName;
    // Landmarks were already pushed to result.landmarks in the loop above

    if (!_oscQueue->try_push(result)) {
        // Queue full -> Drop newest (Backpressure)
        // Logger::warn("ProcessingLoop: OSC Queue full, dropping result.");
    }
}

void ProcessingLoop::drawDebugOverlay(cv::Mat& debugFrame, Frame* frame,
                                      const std::vector<math::KalmanFilter::Point3f>& currentLandmarks,
                                      const std::string& gestureName, int gestureId, float pinchDist) {
    if (!_mjpegServer || debugFrame.empty()) {
        return;
    }

    // Helper: Draw semi-transparent rectangle
    auto drawTransparentRect = [&](cv::Rect rect, cv::Scalar color, double alpha) {
        cv::Mat overlay = debugFrame.clone();
        cv::rectangle(overlay, rect, color, cv::FILLED);
        cv::addWeighted(overlay, alpha, debugFrame, 1.0 - alpha, 0, debugFrame);
    };

    // === TOP LEFT: Status Info Box ===
    int statusBoxHeight = 130;
    drawTransparentRect(cv::Rect(5, 5, 280, statusBoxHeight), cv::Scalar(0, 0, 0), 0.6);

    int yPos = 22;
    int lineHeight = 20;

    // 1. Date & Time
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time_t);
    char timeStr[64];
    std::strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", &tm);
    cv::putText(debugFrame, timeStr, cv::Point(10, yPos), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1);
    yPos += lineHeight;

    // 2. FPS with color coding
    cv::Scalar fpsColor = (_currentFps >= 28) ? cv::Scalar(0, 255, 0) :
                          (_currentFps >= 20) ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 255);
    char fpsStr[32];
    snprintf(fpsStr, sizeof(fpsStr), "FPS: %.1f", _currentFps);
    cv::putText(debugFrame, fpsStr, cv::Point(10, yPos), cv::FONT_HERSHEY_SIMPLEX, 0.5, fpsColor, 1);
    yPos += lineHeight;

    // 3. System Performance (Update every 5 seconds)
    auto perfNow = std::chrono::steady_clock::now();
    auto perfElapsed = std::chrono::duration_cast<std::chrono::seconds>(perfNow - _lastPerfUpdate).count();
    if (perfElapsed >= 5) {
        _performanceSummary = SystemMonitor::getPerformanceSummary();
        _lastPerfUpdate = perfNow;
    }
    cv::putText(debugFrame, _performanceSummary, cv::Point(10, yPos), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 255), 1);
    yPos += lineHeight;

    // 4. Count Palm Detections (Hands)
    int numHands = 0;
    std::vector<cv::Rect> palmBoxes;

    if (!frame->palmData.empty()) {
        // Log palm data structure once for debugging
        static bool palmDataLogged = false;
        if (!palmDataLogged && frame->palmData.size() >= 18) {
            Logger::info("Palm Detection Raw (first 18): ");
            for (int k = 0; k < 18 && k < (int)frame->palmData.size(); ++k) {
                Logger::info("  [", k, "]=", frame->palmData[k]);
            }
            palmDataLogged = true;
        }

        // MediaPipe Palm Detection output format (896 anchors, each with score + bbox):
        // The blob outputs scores and bboxes separately, then combined.
        // For simplicity, try to find high-confidence detections.
        // Format varies, but commonly: [score, x_center, y_center, width, height, ...]

        size_t stride = 18; // Typical anchor output size
        if (frame->palmData.size() >= stride) {
            int numDetections = static_cast<int>(frame->palmData.size() / stride);
            numDetections = std::min(numDetections, 10); // Cap for sanity

            for (int i = 0; i < numDetections; ++i) {
                float rawScore = frame->palmData[i * stride + 0];

                // CRITICAL FIX: Apply sigmoid to raw logits
                float score = 1.0f / (1.0f + std::exp(-rawScore));

                if (score > 0.5f) {  // Confidence threshold (after sigmoid)
                    numHands++;

                    // Extract normalized bbox (center + size format)
                    float cx = frame->palmData[i * stride + 1];
                    float cy = frame->palmData[i * stride + 2];
                    float w = frame->palmData[i * stride + 3];
                    float h = frame->palmData[i * stride + 4];

                    // Normalize if in pixel coordinates
                    if (cx > 1.0f) cx /= 192.0f;
                    if (cy > 1.0f) cy /= 192.0f;
                    if (w > 1.0f) w /= 192.0f;
                    if (h > 1.0f) h /= 192.0f;

                    // Un-letterbox from 192x192 to original frame
                    float x1 = cx - w / 2.0f;
                    float y1 = cy - h / 2.0f;
                    float x2 = cx + w / 2.0f;
                    float y2 = cy + h / 2.0f;

                    // Apply unletterbox transformation
                    unletterbox(x1, y1, 192.0f, (float)frame->width, (float)frame->height);
                    unletterbox(x2, y2, 192.0f, (float)frame->width, (float)frame->height);

                    // Scale to debug frame
                    int px1 = static_cast<int>(x1 * debugFrame.cols);
                    int py1 = static_cast<int>(y1 * debugFrame.rows);
                    int px2 = static_cast<int>(x2 * debugFrame.cols);
                    int py2 = static_cast<int>(y2 * debugFrame.rows);

                    if (px2 > px1 && py2 > py1) {
                        palmBoxes.emplace_back(cv::Point(px1, py1), cv::Point(px2, py2));
                    }
                }
            }
        }
    }

    std::string handsStr = "Hands: " + std::to_string(numHands);
    cv::putText(debugFrame, handsStr, cv::Point(10, yPos), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
    yPos += lineHeight;

    // 5. VIP Status
    int vipCount = _vipLocked ? 1 : 0;
    std::string vipStr = "VIP: " + std::to_string(vipCount) + (_vipLocked ? " [LOCKED]" : " [Searching...]");
    cv::Scalar vipColor = _vipLocked ? cv::Scalar(0, 255, 0) : cv::Scalar(128, 128, 128);
    cv::putText(debugFrame, vipStr, cv::Point(10, yPos), cv::FONT_HERSHEY_SIMPLEX, 0.5, vipColor, 1);
    yPos += lineHeight;

    // === DRAW PALM BOUNDING BOXES ===
    for (size_t i = 0; i < palmBoxes.size(); ++i) {
        cv::rectangle(debugFrame, palmBoxes[i], cv::Scalar(255, 100, 0), 2);
        cv::putText(debugFrame, "Palm " + std::to_string(i + 1),
                    cv::Point(palmBoxes[i].x, palmBoxes[i].y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 100, 0), 1);
    }

    // === DRAW LANDMARKS & SKELETON ===
    if (currentLandmarks.size() >= 21) {
        cv::Scalar skeletonColor = cv::Scalar(0, 255, 0);
        cv::Scalar jointColor = cv::Scalar(0, 0, 255);

        std::vector<cv::Point> pixelLandmarks;
        pixelLandmarks.reserve(21);

        for (const auto& lm : currentLandmarks) {
            // Landmarks are normalized (0..1), scale to debugFrame dimensions
            int x = static_cast<int>(lm.x * static_cast<float>(debugFrame.cols));
            int y = static_cast<int>(lm.y * static_cast<float>(debugFrame.rows));
            pixelLandmarks.emplace_back(x, y);
        }

        // Draw Skeleton connections
        const std::vector<std::pair<int, int>> connections = {
            {0, 1}, {1, 2}, {2, 3}, {3, 4},         // Thumb
            {0, 5}, {5, 6}, {6, 7}, {7, 8},         // Index
            {0, 9}, {9, 10}, {10, 11}, {11, 12},    // Middle
            {0, 13}, {13, 14}, {14, 15}, {15, 16},  // Ring
            {0, 17}, {17, 18}, {18, 19}, {19, 20},  // Pinky
            {5, 9}, {9, 13}, {13, 17}               // Palm base
        };

        for (const auto& conn : connections) {
            cv::line(debugFrame, pixelLandmarks[conn.first], pixelLandmarks[conn.second], skeletonColor, 2);
        }

        // Draw joints
        for (size_t j = 0; j < pixelLandmarks.size(); ++j) {
            int radius = (j == 0) ? 6 : 4; // Larger wrist
            cv::circle(debugFrame, pixelLandmarks[j], radius, jointColor, -1);
            cv::circle(debugFrame, pixelLandmarks[j], radius, cv::Scalar(255,255,255), 1); // White outline
        }

        // === PER-HAND INFO BOX ===
        auto wrist = currentLandmarks[0];
        int anchorX = static_cast<int>(wrist.x * static_cast<float>(frame->width)) + 25;
        int anchorY = static_cast<int>(wrist.y * static_cast<float>(frame->height)) - 80;

        // Clamp to screen bounds
        anchorX = std::clamp(anchorX, 10, static_cast<int>(frame->width) - 180);
        anchorY = std::clamp(anchorY, 10, static_cast<int>(frame->height) - 100);

        int boxWidth = 170;
        int boxHeight = 90;

        // Semi-transparent background
        drawTransparentRect(cv::Rect(anchorX - 5, anchorY - 5, boxWidth, boxHeight), cv::Scalar(0, 0, 0), 0.7);
        cv::rectangle(debugFrame, cv::Rect(anchorX - 5, anchorY - 5, boxWidth, boxHeight), cv::Scalar(0, 255, 0), 1);

        int handInfoY = anchorY + 12;
        int smallLineHeight = 18;

        // Position: X, Y in normalized (0..1), Z in mm
        char posStr[64];
        snprintf(posStr, sizeof(posStr), "X:%.2f Y:%.2f Z:%.0fmm", wrist.x, wrist.y, wrist.z);
        cv::putText(debugFrame, posStr, cv::Point(anchorX, handInfoY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        handInfoY += smallLineHeight;

        // Velocity (units per second, scaled for display)
        char velStr[64];
        snprintf(velStr, sizeof(velStr), "Vel: %.2f, %.2f, %.0f", _lastVelocity.x, _lastVelocity.y, _lastVelocity.z);
        cv::putText(debugFrame, velStr, cv::Point(anchorX, handInfoY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 255), 1);
        handInfoY += smallLineHeight;

        // Delta (per-frame change)
        float deltaX = 0, deltaY = 0, deltaZ = 0;
        if (_hasLastState && !_lastHandState.landmarks.empty()) {
            deltaX = wrist.x - _lastHandState.landmarks[0].x;
            deltaY = wrist.y - _lastHandState.landmarks[0].y;
            deltaZ = wrist.z - _lastHandState.landmarks[0].z;
        }
        char deltaStr[64];
        snprintf(deltaStr, sizeof(deltaStr), "Dlt: %.3f, %.3f, %.1f", deltaX, deltaY, deltaZ);
        cv::putText(debugFrame, deltaStr, cv::Point(anchorX, handInfoY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 200, 200), 1);
        handInfoY += smallLineHeight;

        // Gesture with color
        cv::Scalar gestureColor = (gestureId > 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(150, 150, 150);
        std::string gestureDisplay = (gestureId > 0) ? gestureName : "---";
        cv::putText(debugFrame, gestureDisplay, cv::Point(anchorX, handInfoY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, gestureColor, 1);
    }

    _mjpegServer->update(debugFrame);
}

} // namespace core

