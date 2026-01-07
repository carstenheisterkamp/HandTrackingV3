#include "core/ProcessingLoop.hpp"
#include "core/SystemMonitor.hpp"
#include <chrono>
#include <algorithm> // Added for std::clamp
#include <opencv2/imgproc.hpp> // For drawing

#ifdef ENABLE_CUDA
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

    // Prepare Debug Frame (RGB)
    cv::Mat debugFrame;
    if (_mjpegServer) {
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
            }
        }
#endif

        if (debugFrame.empty()) {
            // CPU Fallback - FORCE NV12 conversion if frame matches NV12 size
            // This fixes the "Green/Purple" issue if type flag is wrong
            size_t expectedSizeNV12 = frame->width * frame->height * 3 / 2;

             // Default NV12
             cv::Mat nv12(frame->height * 3 / 2, frame->width, CV_8UC1, frame->data.get());
             cv::cvtColor(nv12, debugFrame, cv::COLOR_YUV2BGR_NV12);
        }
    }

    // --- Palm Detection Gating ---
    // Since we are running the Landmark model on the full frame (without cropping),
    // we MUST gate it with the Palm Detector to avoid false positives.
    bool palmDetected = false;
    if (!frame->palmData.empty()) {
        int numDetections = frame->palmData.size() / 7;
        for (int i = 0; i < numDetections; ++i) {
            float conf = frame->palmData[i * 7 + 2];
            if (conf > 0.4f) { // Confidence threshold
                palmDetected = true;
                break;
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

        // Draw debug overlay and return early
        drawDebugOverlay(debugFrame, frame, {}, "", 0, 0.0f);
        return;
    }

    // Parse landmarks (21 points * 3 coords = 63 floats)
    if (frame->nnData.size() < 63) {
        static int warnCounter = 0;
        if (warnCounter++ % 300 == 0) {
           Logger::warn("ProcessingLoop: Invalid NN data size: ", frame->nnData.size(), " (Expected 63)");
        }
        // Draw debug overlay and return early
        drawDebugOverlay(debugFrame, frame, {}, "", 0, 0.0f);
        return;
    }

    // Prepare result
    TrackingResult result;
    // result.vipLocked will be set at the end
    result.landmarks.reserve(21);

    std::vector<math::KalmanFilter::Point3f> currentLandmarks;
    currentLandmarks.reserve(21);

    // Filter Parameters
    constexpr float ONE_EURO_MIN_CUTOFF = 1.0f; // Hz
    constexpr float ONE_EURO_BETA = 0.007f;
    constexpr float ONE_EURO_D_CUTOFF = 1.0f;

    // Process each landmark
    for (int i = 0; i < 21; ++i) {
        float rawX = frame->nnData[i * 3 + 0];
        float rawY = frame->nnData[i * 3 + 1];
        float rawZ = frame->nnData[i * 3 + 2]; // Relative depth from NN model

        // DEBUG: Log first landmark values once
        static bool landmarkValuesLogged = false;
        if (!landmarkValuesLogged && i == 0) {
            Logger::info("Raw Landmark 0 (Wrist): X=", rawX, " Y=", rawY, " Z=", rawZ);
            landmarkValuesLogged = true;
        }

        // Auto-normalize if values are in pixel coordinates (0-224) instead of (0-1)
        // The hand_landmark model outputs in pixel space of the input (224x224)
        if (rawX > 1.0f || rawY > 1.0f) {
            rawX /= 224.0f;
            rawY /= 224.0f;
            // Z bleibt relativ (oft im Bereich -0.5 bis 0.5 oder ähnlich)
        }

        // TODO: Compute metric depth on Jetson GPU using frame->monoLeftData/monoRightData
        // For now, use the relative Z coordinate from the landmark model.
        // GPU Stereo implementation would go here:
        // if (frame->monoLeftData && frame->monoRightData && frame->monoWidth > 0) {
        //     computeStereoDepthCUDA(frame);  // Writes to frame->depthData
        // }

        // Use computed depth if available (after GPU stereo is implemented)
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
        TrackingResult::NormalizedPoint np;
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
        auto dist = [](const math::KalmanFilter::Point3f& a, const math::KalmanFilter::Point3f& b) {
            return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2) + std::pow(a.z - b.z, 2));
        };

        // Key Landmarks
        auto wrist = currentLandmarks[0];
        auto thumbTip = currentLandmarks[4];
        auto indexTip = currentLandmarks[8];
        auto middleTip = currentLandmarks[12];
        auto ringTip = currentLandmarks[16];
        auto pinkyTip = currentLandmarks[20];

        auto indexPip = currentLandmarks[6];
        auto middlePip = currentLandmarks[10];
        auto ringPip = currentLandmarks[14];
        auto pinkyPip = currentLandmarks[18];

        // Finger States (Extended vs Folded)
        // Heuristic: Tip is further from wrist than PIP joint = Extended
        bool thumbOpen = dist(thumbTip, wrist) > dist(currentLandmarks[2], wrist); // Thumb is special (CMC vs IP)
        bool indexOpen = dist(indexTip, wrist) > dist(indexPip, wrist);
        bool middleOpen = dist(middleTip, wrist) > dist(middlePip, wrist);
        bool ringOpen = dist(ringTip, wrist) > dist(ringPip, wrist);
        bool pinkyOpen = dist(pinkyTip, wrist) > dist(pinkyPip, wrist);

        // Pinch (Thumb Tip - Index Tip)
        pinchDist = dist(thumbTip, indexTip);

        // Fist (Average dist of tips to wrist)
        fistDist = (dist(indexTip, wrist) + dist(middleTip, wrist) + dist(ringTip, wrist) + dist(pinkyTip, wrist)) / 4.0f;

        // Thresholds (Normalized [0,1])
        const float PINCH_THRESH = 0.05f;

        // Gesture Logic
        if (!thumbOpen && !indexOpen && !middleOpen && !ringOpen && !pinkyOpen) {
            gestureId = 2; gestureName = "FIST";
        }
        else if (thumbOpen && indexOpen && middleOpen && ringOpen && pinkyOpen) {
            // Check for VULCAN (Gap between Middle and Ring)
            float midRingDist = dist(middleTip, ringTip);
            if (midRingDist > 0.08f) { // Heuristic gap
                 gestureId = 10; gestureName = "VULCAN";
            } else {
                 gestureId = 3; gestureName = "FIVE";
            }
        }
        else if (pinchDist < PINCH_THRESH) {
            // Check if others are open
            if (middleOpen && ringOpen && pinkyOpen) {
                gestureId = 1; gestureName = "OK"; // or PINCH
            } else {
                gestureId = 1; gestureName = "PINCH";
            }
        }
        else if (!thumbOpen && indexOpen && !middleOpen && !ringOpen && !pinkyOpen) {
            gestureId = 4; gestureName = "POINTING";
        }
        else if (thumbOpen && indexOpen && !middleOpen && !ringOpen && !pinkyOpen) {
            gestureId = 5; gestureName = "TWO"; // L-Shape
        }
        else if (!thumbOpen && indexOpen && middleOpen && !ringOpen && !pinkyOpen) {
            gestureId = 6; gestureName = "PEACE";
        }
        else if (thumbOpen && !indexOpen && !middleOpen && !ringOpen && pinkyOpen) {
            gestureId = 7; gestureName = "CALL_ME";
        }
        else if (!thumbOpen && indexOpen && !middleOpen && !ringOpen && pinkyOpen) {
            gestureId = 8; gestureName = "METAL";
        }
        else if (thumbOpen && indexOpen && !middleOpen && !ringOpen && pinkyOpen) {
            gestureId = 9; gestureName = "LOVE_YOU";
        }
        else if (thumbOpen && !indexOpen && !middleOpen && !ringOpen && !pinkyOpen) {
            gestureId = 11; gestureName = "THUMBS_UP"; // Orientation check needed for real thumbs up
        }
        else {
            gestureName = "unknown";
        }

        // Log Gesture occasionally
        static int gestureLogCounter = 0;
        if (++gestureLogCounter % 30 == 0) {
            Logger::info("Gesture: ", gestureName, " (Pinch: ", pinchDist, ")");
            Logger::info("Fingers: T:", thumbOpen, " I:", indexOpen, " M:", middleOpen, " R:", ringOpen, " P:", pinkyOpen);
        }
    }

    // Draw Debug Overlay (always, even if no hands detected)
    drawDebugOverlay(debugFrame, frame, currentLandmarks, gestureName, gestureId, pinchDist);

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

    // === TOP LEFT: Status Info ===
    int yPos = 25;
    int lineHeight = 22;

    // 1. Date & Time
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time_t);
    char timeStr[64];
    std::strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", &tm);
    cv::putText(debugFrame, timeStr, cv::Point(10, yPos), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    yPos += lineHeight;

    // 2. FPS
    std::string fpsStr = "FPS: " + std::to_string((int)_currentFps);
    cv::putText(debugFrame, fpsStr, cv::Point(10, yPos), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
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
        // DEBUG: Log palm data structure once
        static bool palmDataLogged = false;
        if (!palmDataLogged && frame->palmData.size() >= 7) {
            Logger::info("Palm Detection Raw: size=", frame->palmData.size(),
                         " [0]=", frame->palmData[0], " [1]=", frame->palmData[1],
                         " [2]=", frame->palmData[2], " [3]=", frame->palmData[3],
                         " [4]=", frame->palmData[4], " [5]=", frame->palmData[5],
                         " [6]=", frame->palmData[6]);
            palmDataLogged = true;
        }

        // Palm detection output format varies by model
        // Common formats: [x_center, y_center, w, h, conf, ...] or [conf, x1, y1, x2, y2, ...]
        int numDetections = frame->palmData.size() / 7;
        for (int i = 0; i < numDetections; ++i) {
            float conf = frame->palmData[i * 7 + 2];
            if (conf > 0.4f) {  // Confidence threshold
                numHands++;
                float x1 = frame->palmData[i * 7 + 3];
                float y1 = frame->palmData[i * 7 + 4];
                float x2 = frame->palmData[i * 7 + 5];
                float y2 = frame->palmData[i * 7 + 6];

                // Auto-normalize if values are in pixel coordinates (>1.0)
                // Palm model input is 128x128
                if (x1 > 1.0f || y1 > 1.0f || x2 > 1.0f || y2 > 1.0f) {
                    x1 /= 128.0f;
                    y1 /= 128.0f;
                    x2 /= 128.0f;
                    y2 /= 128.0f;
                }

                // Scale to debug frame size
                int px1 = static_cast<int>(x1 * debugFrame.cols);
                int py1 = static_cast<int>(y1 * debugFrame.rows);
                int px2 = static_cast<int>(x2 * debugFrame.cols);
                int py2 = static_cast<int>(y2 * debugFrame.rows);

                palmBoxes.emplace_back(cv::Point(px1, py1), cv::Point(px2, py2));
            }
        }
    }

    std::string handsStr = "Hands: " + std::to_string(numHands);
    cv::putText(debugFrame, handsStr, cv::Point(10, yPos), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
    yPos += lineHeight;

    // 5. VIP Status (Count: 0 when searching, 1 when locked)
    int vipCount = _vipLocked ? 1 : 0;
    std::string vipStr = "VIP: " + std::to_string(vipCount) + (_vipLocked ? " (Locked)" : " (Searching)");
    cv::Scalar vipColor = _vipLocked ? cv::Scalar(0, 255, 0) : cv::Scalar(128, 128, 128);
    cv::putText(debugFrame, vipStr, cv::Point(10, yPos), cv::FONT_HERSHEY_SIMPLEX, 0.5, vipColor, 1);
    yPos += lineHeight;

    // === DRAW PALM BOUNDING BOXES ===
    for (size_t i = 0; i < palmBoxes.size(); ++i) {
        cv::rectangle(debugFrame, palmBoxes[i], cv::Scalar(255, 0, 0), 2);
        cv::putText(debugFrame, "Hand " + std::to_string(i + 1),
                    cv::Point(palmBoxes[i].x, palmBoxes[i].y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
    }

    // === DRAW LANDMARKS & SKELETON + PER-HAND INFO ===
    if (!currentLandmarks.empty()) {
        cv::Scalar skeletonColor = (numHands > 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(100, 100, 100);
        cv::Scalar jointColor = (numHands > 0) ? cv::Scalar(0, 0, 255) : cv::Scalar(100, 100, 100);

        std::vector<cv::Point> pixelLandmarks;
        pixelLandmarks.reserve(21);

        for (const auto& lm : currentLandmarks) {
            int x = static_cast<int>(lm.x * frame->width);
            int y = static_cast<int>(lm.y * frame->height);
            pixelLandmarks.emplace_back(x, y);
        }

        // Draw Skeleton
        const std::vector<std::pair<int, int>> connections = {
            {0, 1}, {1, 2}, {2, 3}, {3, 4},         // Thumb
            {0, 5}, {5, 6}, {6, 7}, {7, 8},         // Index
            {0, 9}, {9, 10}, {10, 11}, {11, 12},    // Middle
            {0, 13}, {13, 14}, {14, 15}, {15, 16},  // Ring
            {0, 17}, {17, 18}, {18, 19}, {19, 20},  // Pinky
            {5, 9}, {9, 13}, {13, 17}, {0, 5}, {0, 17} // Palm
        };

        for (const auto& conn : connections) {
            if (conn.first < (int)pixelLandmarks.size() && conn.second < (int)pixelLandmarks.size()) {
                cv::line(debugFrame, pixelLandmarks[conn.first], pixelLandmarks[conn.second], skeletonColor, 2);
            }
        }

        for (const auto& pt : pixelLandmarks) {
            cv::circle(debugFrame, pt, 4, jointColor, -1);
        }

        // === PER-HAND INFO (drawn next to the hand) ===
        // Use wrist position as anchor point
        auto wrist = currentLandmarks[0];
        int anchorX = static_cast<int>(wrist.x * frame->width) + 20; // Offset to the right of wrist
        int anchorY = static_cast<int>(wrist.y * frame->height) - 60; // Above wrist

        // Clamp to screen bounds
        anchorX = std::clamp(anchorX, 10, (int)frame->width - 150);
        anchorY = std::clamp(anchorY, 20, (int)frame->height - 80);

        int handInfoY = anchorY;
        int smallLineHeight = 16;

        // Background for readability
        cv::rectangle(debugFrame,
                      cv::Point(anchorX - 5, anchorY - 15),
                      cv::Point(anchorX + 140, anchorY + 65),
                      cv::Scalar(0, 0, 0), cv::FILLED);
        cv::rectangle(debugFrame,
                      cv::Point(anchorX - 5, anchorY - 15),
                      cv::Point(anchorX + 140, anchorY + 65),
                      cv::Scalar(0, 255, 0), 1);

        // Position (normalized %)
        std::string posStr = "Pos: " + std::to_string((int)(wrist.x * 100)) + "%, "
                                     + std::to_string((int)(wrist.y * 100)) + "%";
        cv::putText(debugFrame, posStr, cv::Point(anchorX, handInfoY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        handInfoY += smallLineHeight;

        // Velocity
        std::string velStr = "Vel: " + std::to_string((int)(_lastVelocity.x * 100)) + ", "
                                     + std::to_string((int)(_lastVelocity.y * 100)) + ", "
                                     + std::to_string((int)(_lastVelocity.z * 100));
        cv::putText(debugFrame, velStr, cv::Point(anchorX, handInfoY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 255), 1);
        handInfoY += smallLineHeight;

        // Delta (position change from last frame)
        float deltaX = 0, deltaY = 0, deltaZ = 0;
        if (_hasLastState && !_lastHandState.landmarks.empty()) {
            deltaX = wrist.x - _lastHandState.landmarks[0].x;
            deltaY = wrist.y - _lastHandState.landmarks[0].y;
            deltaZ = wrist.z - _lastHandState.landmarks[0].z;
        }
        std::string deltaStr = "Dlt: " + std::to_string((int)(deltaX * 1000)) + ", "
                                       + std::to_string((int)(deltaY * 1000)) + ", "
                                       + std::to_string((int)(deltaZ * 1000));
        cv::putText(debugFrame, deltaStr, cv::Point(anchorX, handInfoY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 200, 200), 1);
        handInfoY += smallLineHeight;

        // Gesture (if detected)
        if (gestureId > 0 && gestureName != "unknown") {
            cv::putText(debugFrame, gestureName, cv::Point(anchorX, handInfoY),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        } else {
            cv::putText(debugFrame, "---", cv::Point(anchorX, handInfoY),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(150, 150, 150), 1);
        }
    }

    _mjpegServer->update(debugFrame);
}

} // namespace core

