/**
 * V3 ProcessingLoop - Simplified for Sensor-Only Pipeline
 *
 * This replaces the old 800+ line ProcessingLoop that expected
 * NN data from OAK-D. In V3, all NNs run on Jetson via TensorRT.
 *
 * Current State (Phase 1-2):
 * - Receives RGB frames from OAK-D
 * - Displays debug preview via MJPEG
 * - Placeholder for TensorRT inference (Phase 2)
 * - Placeholder for stereo depth (Phase 3)
 */

#include "core/ProcessingLoop.hpp"
#include "core/SystemMonitor.hpp"
#include "core/HandTracker.hpp"
#include "core/GestureFSM.hpp"
#include "core/StereoDepth.hpp"

#include <filesystem>
#include <chrono>
#include <algorithm>
#include <opencv2/imgproc.hpp>

#ifdef ENABLE_TENSORRT
#include "inference/PalmDetector.hpp"
#include "inference/HandLandmark.hpp"
#endif

#ifdef ENABLE_CUDA
#include "core/StereoKernel.hpp"
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

    // V3: Initialize tracking components
    _handTracker = std::make_unique<HandTracker>();
    _gestureFSM = std::make_unique<GestureFSM>();
    _stereoDepth = std::make_unique<StereoDepth>();

    // Note: TensorRT initialization moved to initInference()
    // Called lazily to not block startup

    // MJPEG Server for debug preview
    _mjpegServer = std::make_unique<net::MjpegServer>(8080);

    // Performance cache
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
    Logger::info("ProcessingLoop started (V3 Mode).");
}

void ProcessingLoop::stop() {
    if (!_running) return;
    _running = false;
    _mjpegServer->stop();
    if (_thread.joinable()) {
        _thread.join();
    }
    // Wait for TRT init thread if still running
    if (_trtInitThread.joinable()) {
        _trtInitThread.join();
    }
    Logger::info("ProcessingLoop stopped.");
}

bool ProcessingLoop::isRunning() const {
    return _running;
}

void ProcessingLoop::loop() {
    // Lazy TensorRT initialization in BACKGROUND thread (doesn't block frame processing)
#ifdef ENABLE_TENSORRT
    Logger::info("🔧 ENABLE_TENSORRT is defined, checking init state...");
    Logger::info("   _inferenceInitialized: ", _inferenceInitialized ? "true" : "false");
    Logger::info("   _inferenceAttempted: ", _inferenceAttempted ? "true" : "false");

    if (!_inferenceInitialized && !_inferenceAttempted) {
        _inferenceAttempted = true;
        Logger::info("🚀 Starting TensorRT initialization thread...");

        // Start TRT init in background thread
        _trtInitThread = std::thread([this]() {
            Logger::info("🔧 TensorRT init thread STARTED");

            auto palmDetector = std::make_unique<inference::PalmDetector>();
            auto handLandmark = std::make_unique<inference::HandLandmark>();

            inference::PalmDetector::Config palmConfig;
            palmConfig.modelPath = "models/palm_detection.onnx";

            inference::HandLandmark::Config landmarkConfig;
            landmarkConfig.modelPath = "models/hand_landmark.onnx";

            // Check if ONNX files exist
            Logger::info("🔍 Checking for ONNX models...");
            Logger::info("   Palm model path: ", palmConfig.modelPath);
            Logger::info("   Landmark model path: ", landmarkConfig.modelPath);

            bool palmExists = std::filesystem::exists(palmConfig.modelPath);
            bool landmarkExists = std::filesystem::exists(landmarkConfig.modelPath);

            Logger::info("   Palm exists: ", palmExists ? "YES" : "NO");
            Logger::info("   Landmark exists: ", landmarkExists ? "YES" : "NO");

            if (!palmExists || !landmarkExists) {
                Logger::warn("ONNX models not found!");
                if (!palmExists) Logger::warn("  Missing: ", palmConfig.modelPath);
                if (!landmarkExists) Logger::warn("  Missing: ", landmarkConfig.modelPath);
                Logger::warn("Run: python3 scripts/convert_to_onnx.py");
                Logger::info("Running in preview-only mode");
                return;
            }

            Logger::info("🔧 Initializing Palm Detector...");
            bool palmOk = palmDetector->init(palmConfig);
            Logger::info("   Palm Detector init: ", palmOk ? "SUCCESS" : "FAILED");

            bool landmarkOk = false;
            if (palmOk) {
                Logger::info("🔧 Initializing Hand Landmark...");
                landmarkOk = handLandmark->init(landmarkConfig);
                Logger::info("   Hand Landmark init: ", landmarkOk ? "SUCCESS" : "FAILED");
            }

            if (palmOk && landmarkOk) {
                // Transfer ownership to class members (thread-safe)
                std::lock_guard<std::mutex> lock(_trtMutex);
                _palmDetector = std::move(palmDetector);
                _handLandmark = std::move(handLandmark);
                _inferenceInitialized = true;
                Logger::info("✅ TensorRT inference initialized successfully (background)");
                Logger::info("   Ready for palm detection and landmark inference!");
            } else {
                Logger::warn("❌ TensorRT inference initialization failed");
                if (!palmOk) Logger::warn("   Palm detector init failed");
                if (!landmarkOk) Logger::warn("   Hand landmark init failed");
            }
        });
    }
#endif

    while (_running) {
        Frame* frame = nullptr;
        if (_inputQueue->pop_front(frame)) {
            if (frame) {
                processFrame(frame);
                _framePool->release(frame);
            }
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
}

void ProcessingLoop::processFrame(Frame* frame) {
    auto frameStart = std::chrono::high_resolution_clock::now();

    // ═══════════════════════════════════════════════════════════
    // FPS Tracking
    // ═══════════════════════════════════════════════════════════
    _frameCount++;
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - _lastFpsTime).count();
    if (elapsed >= 2000) {
        _currentFps = _frameCount * 1000.0f / elapsed;

        Logger::info("═══ V3 PROCESSING STATS ═══");
        Logger::info("FPS: ", _currentFps);
        Logger::info("Frame: ", frame->width, "x", frame->height);
        Logger::info("Stereo: ", frame->hasStereoData ? "Available" : "Disabled");
        Logger::info("MJPEG: ", (_mjpegServer && _mjpegServer->hasClients()) ? "Clients connected" : "No clients");

        // Hand tracking stats (REQUIRED for visibility)
        Logger::info("🖐 Hands Detected: ", _lastHandCount);
        if (_lastHandCount > 0) {
            Logger::info("   Position: (", _lastPalmX, ", ", _lastPalmY, ", ", _lastPalmZ, ")");
            Logger::info("   Velocity: (", _lastVelX, ", ", _lastVelY, ")");
            Logger::info("   Gesture: ", _lastGesture);
            Logger::info("   VIP Locked: ", _lastVipLocked ? "Yes" : "No");
        }
        Logger::info("TensorRT: ", _inferenceInitialized ? "Ready" : "Not initialized");

        _frameCount = 0;
        _lastFpsTime = now;
    }

    // ═══════════════════════════════════════════════════════════
    // Step 1: Convert NV12 to BGR for visualization
    // ═══════════════════════════════════════════════════════════
    bool shouldRenderDebug = _mjpegServer && _mjpegServer->hasClients();
    cv::Mat debugFrame;

    if (shouldRenderDebug) {
        size_t requiredSize = frame->width * frame->height * 3;
        if (!_bgrBuffer || _bgrBufferSize < requiredSize) {
            _bgrBuffer = allocate_aligned<uint8_t>(requiredSize);
            _bgrBufferSize = requiredSize;
            _bgrWidth = frame->width;
            _bgrHeight = frame->height;
            register_buffer_cuda(_bgrBuffer.get(), requiredSize);
        }

#ifdef ENABLE_CUDA
        void* srcDev = get_device_pointer(frame->data.get());
        void* dstDev = get_device_pointer(_bgrBuffer.get());

        if (srcDev && dstDev) {
            NppiSize oSizeROI = {(int)frame->width, (int)frame->height};
            const Npp8u* pSrc[2];
            pSrc[0] = (const Npp8u*)srcDev;
            pSrc[1] = (const Npp8u*)srcDev + frame->width * frame->height;

            NppStatus status = nppiNV12ToBGR_8u_P2C3R(
                pSrc, (int)frame->width,
                (Npp8u*)dstDev, (int)frame->width * 3,
                oSizeROI
            );

            if (status == NPP_NO_ERROR) {
                cudaStreamSynchronize(0);
                debugFrame = cv::Mat((int)frame->height, (int)frame->width, CV_8UC3, _bgrBuffer.get());
            }
        }
#else
        cv::Mat nv12(frame->height * 3 / 2, frame->width, CV_8UC1, frame->data.get());
        cv::cvtColor(nv12, debugFrame, cv::COLOR_YUV2BGR_NV12);
#endif
    }

    // ═══════════════════════════════════════════════════════════
    // Step 2: TensorRT Inference (Palm + Landmarks)
    // ═══════════════════════════════════════════════════════════

#ifdef ENABLE_TENSORRT
    bool canInfer = false;
    {
        std::lock_guard<std::mutex> lock(_trtMutex);
        canInfer = _inferenceInitialized && _palmDetector && _handLandmark;
    }

    if (canInfer) {
        // Debug: Log that we're attempting inference
        static int inferenceAttempts = 0;
        if (++inferenceAttempts % 60 == 1) {
            Logger::info("🔍 Running Palm Detection inference (attempt ", inferenceAttempts, ")...");
        }

        // Palm Detection
        auto palmDetection = _palmDetector->detect(
            frame->data.get(),
            static_cast<int>(frame->width),
            static_cast<int>(frame->height)
        );

        if (palmDetection) {
            // Debug log (every 30 frames)
            static int detectionCount = 0;
            if (++detectionCount % 30 == 1) {
                Logger::info("🖐 Palm detected! Score: ", palmDetection->score,
                            " Pos: (", palmDetection->x, ", ", palmDetection->y, ")");
            }
        } else {
            // No detection - log periodically
            static int noDetectionCount = 0;
            if (++noDetectionCount % 60 == 1) {
                Logger::info("❌ No palm detected (frame ", noDetectionCount, ")");
            }
        }

        if (palmDetection) {
            // Draw palm BBox
            if (!debugFrame.empty()) {
                int cx = static_cast<int>(palmDetection->x * debugFrame.cols);
                int cy = static_cast<int>(palmDetection->y * debugFrame.rows);
                int w = static_cast<int>(palmDetection->width * debugFrame.cols);
                int h = static_cast<int>(palmDetection->height * debugFrame.rows);
                cv::rectangle(debugFrame, cv::Rect(cx - w/2, cy - h/2, w, h), cv::Scalar(0, 255, 0), 2);
                cv::circle(debugFrame, cv::Point(cx, cy), 5, cv::Scalar(0, 255, 255), -1);
            }

            // Hand Landmark Inference
            auto landmarks = _handLandmark->infer(
                frame->data.get(),
                static_cast<int>(frame->width),
                static_cast<int>(frame->height),
                *palmDetection
            );

            if (landmarks) {
                // Debug log
                static int landmarkCount = 0;
                if (++landmarkCount % 30 == 1) {
                    Logger::info("✋ Landmarks! Presence: ", landmarks->presence);
                }

                // Draw landmarks
                if (!debugFrame.empty()) {
                    for (size_t i = 0; i < landmarks->landmarks.size(); ++i) {
                        int lx = static_cast<int>(landmarks->landmarks[i].x * debugFrame.cols);
                        int ly = static_cast<int>(landmarks->landmarks[i].y * debugFrame.rows);
                        cv::Scalar color = (i == 4 || i == 8 || i == 12 || i == 16 || i == 20)
                            ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
                        cv::circle(debugFrame, cv::Point(lx, ly), 3, color, -1);
                    }

                    // Skeleton
                    const int conns[][2] = {{0,1},{1,2},{2,3},{3,4},{0,5},{5,6},{6,7},{7,8},
                        {0,9},{9,10},{10,11},{11,12},{0,13},{13,14},{14,15},{15,16},
                        {0,17},{17,18},{18,19},{19,20}};
                    for (const auto& c : conns) {
                        if (c[0] < (int)landmarks->landmarks.size() && c[1] < (int)landmarks->landmarks.size()) {
                            cv::line(debugFrame,
                                cv::Point(landmarks->landmarks[c[0]].x * debugFrame.cols,
                                          landmarks->landmarks[c[0]].y * debugFrame.rows),
                                cv::Point(landmarks->landmarks[c[1]].x * debugFrame.cols,
                                          landmarks->landmarks[c[1]].y * debugFrame.rows),
                                cv::Scalar(0, 255, 0), 1);
                        }
                    }
                }

                // Kalman + Gesture + OSC
                static auto lastTime = std::chrono::steady_clock::now();
                auto currentTime = std::chrono::steady_clock::now();
                float dt = std::chrono::duration<float>(currentTime - lastTime).count();
                lastTime = currentTime;

                float palmZ = 500.0f;
                if (frame->hasStereoData && _stereoDepth) {
                    palmZ = _stereoDepth->getDepthAtPoint(
                        frame->monoLeftData.get(), frame->monoRightData.get(),
                        (int)frame->monoWidth, (int)frame->monoHeight,
                        (int)(landmarks->palmCenterX * frame->monoWidth),
                        (int)(landmarks->palmCenterY * frame->monoHeight));
                }

                Point3D palm3D = {landmarks->palmCenterX, landmarks->palmCenterY, palmZ};
                _handTracker->predict(dt);
                _handTracker->update(palm3D);

                std::vector<TrackingResult::NormalizedPoint> lmPoints;
                for (const auto& lm : landmarks->landmarks) lmPoints.push_back(lm);
                auto gesture = _gestureFSM->update(lmPoints);

                TrackingResult result;
                result.palmPosition = _handTracker->getPosition();
                result.velocity = _handTracker->getVelocity();
                result.gesture = gesture;
                result.vipLocked = _handTracker->isLocked();
                result.timestamp = std::chrono::steady_clock::now();
                for (size_t i = 0; i < 21 && i < landmarks->landmarks.size(); ++i)
                    result.landmarks.push_back(landmarks->landmarks[i]);
                _oscQueue->try_push(result);

                // Update tracking state for stats display
                _lastHandCount = 1;
                _lastPalmX = result.palmPosition.x;
                _lastPalmY = result.palmPosition.y;
                _lastPalmZ = result.palmPosition.z;
                _lastVelX = result.velocity.vx;
                _lastVelY = result.velocity.vy;
                _lastGesture = GestureFSM::getStateName(gesture);
                _lastVipLocked = result.vipLocked;
            }
        } else {
            // No palm detected - reset hand count
            _lastHandCount = 0;
        }
    }
#endif

    // ═══════════════════════════════════════════════════════════
    // Step 3: Send to MJPEG (AFTER drawing detections)
    // ═══════════════════════════════════════════════════════════
    if (shouldRenderDebug && !debugFrame.empty()) {
        drawDebugOverlay(debugFrame, frame);
        _mjpegServer->update(debugFrame);
    }

    // ═══════════════════════════════════════════════════════════
    // Timing
    // ═══════════════════════════════════════════════════════════
    auto frameEnd = std::chrono::high_resolution_clock::now();
    auto frameDuration = std::chrono::duration_cast<std::chrono::microseconds>(frameEnd - frameStart).count();

    static int timingCounter = 0;
    static long totalTime = 0;
    totalTime += frameDuration;
    if (++timingCounter >= 60) {
        float avgMs = totalTime / 60000.0f;
        if (avgMs > 30.0f) {
            Logger::warn("Processing slow: ", avgMs, " ms/frame");
        }
        timingCounter = 0;
        totalTime = 0;
    }
}

void ProcessingLoop::drawDebugOverlay(cv::Mat& debugFrame, Frame* frame) {
    // Semi-transparent status box (larger for hand info)
    cv::Mat overlay = debugFrame.clone();
    cv::rectangle(overlay, cv::Rect(5, 5, 280, 160), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::addWeighted(overlay, 0.6, debugFrame, 0.4, 0, debugFrame);

    int y = 22;
    int lineHeight = 18;

    // Date/Time
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time_t);
    char timeStr[64];
    std::strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", &tm);
    cv::putText(debugFrame, timeStr, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    y += lineHeight;

    // FPS
    cv::Scalar fpsColor = (_currentFps >= 28) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    char fpsStr[32];
    snprintf(fpsStr, sizeof(fpsStr), "FPS: %.1f", _currentFps);
    cv::putText(debugFrame, fpsStr, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, fpsColor, 1);
    y += lineHeight;

    // TensorRT Status
    std::string trtStatus = _inferenceInitialized ? "TensorRT: Ready" : "TensorRT: Building...";
    cv::Scalar trtColor = _inferenceInitialized ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
    cv::putText(debugFrame, trtStatus, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.4, trtColor, 1);
    y += lineHeight;

    // Hand Detection Status
    char handStr[64];
    snprintf(handStr, sizeof(handStr), "Hands: %d", _lastHandCount);
    cv::Scalar handColor = (_lastHandCount > 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(128, 128, 128);
    cv::putText(debugFrame, handStr, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, handColor, 1);
    y += lineHeight;

    // If hand detected, show details
    if (_lastHandCount > 0) {
        char posStr[64];
        snprintf(posStr, sizeof(posStr), "Pos: (%.2f, %.2f, %.0f)", _lastPalmX, _lastPalmY, _lastPalmZ);
        cv::putText(debugFrame, posStr, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 255, 200), 1);
        y += lineHeight;

        char velStr[64];
        snprintf(velStr, sizeof(velStr), "Vel: (%.2f, %.2f)", _lastVelX, _lastVelY);
        cv::putText(debugFrame, velStr, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 255), 1);
        y += lineHeight;

        std::string gestureStr = "Gesture: " + _lastGesture;
        cv::Scalar gestureColor = (_lastGesture != "None") ? cv::Scalar(0, 255, 255) : cv::Scalar(200, 200, 200);
        cv::putText(debugFrame, gestureStr, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.4, gestureColor, 1);
        y += lineHeight;

        std::string vipStr = _lastVipLocked ? "VIP: LOCKED" : "VIP: Tracking...";
        cv::Scalar vipColor = _lastVipLocked ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 200, 0);
        cv::putText(debugFrame, vipStr, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.4, vipColor, 1);
    }

    // System Performance (update every 5s) - bottom of frame
    auto perfNow = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(perfNow - _lastPerfUpdate).count() >= 5) {
        _performanceSummary = SystemMonitor::getPerformanceSummary();
        _lastPerfUpdate = perfNow;
    }
    cv::putText(debugFrame, _performanceSummary, cv::Point(10, debugFrame.rows - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(200, 200, 255), 1);
}

} // namespace core

