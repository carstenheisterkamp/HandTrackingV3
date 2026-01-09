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

    // V3: Initialize tracking components for 2 hands
    for (int i = 0; i < MAX_HANDS; ++i) {
        _handTrackers[i] = std::make_unique<HandTracker>();
        _gestureFSMs[i] = std::make_unique<GestureFSM>();
    }
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

        // Hand tracking stats (for both hands)
        Logger::info("🖐 Hands Detected: ", _lastHandCount);
        for (int h = 0; h < _lastHandCount && h < MAX_HANDS; ++h) {
            Logger::info("   Hand ", h, ":");
            Logger::info("     Position: (", _handStates[h].palmX, ", ", _handStates[h].palmY, ", ", _handStates[h].palmZ, ")");
            Logger::info("     Velocity: (", _handStates[h].velX, ", ", _handStates[h].velY, ")");
            Logger::info("     Gesture: ", _handStates[h].gesture);
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
    // Step 2: TensorRT Inference (Palm + Landmarks) - 2 HANDS
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

        // Palm Detection - detect ALL hands (up to 2)
        auto palmDetections = _palmDetector->detectAll(
            frame->data.get(),
            static_cast<int>(frame->width),
            static_cast<int>(frame->height),
            MAX_HANDS
        );

        // Debug log
        static int detectionLogCounter = 0;
        if (++detectionLogCounter % 30 == 1) {
            if (!palmDetections.empty()) {
                Logger::info("🖐 ", palmDetections.size(), " palm(s) detected!");
                for (size_t i = 0; i < palmDetections.size(); ++i) {
                    Logger::info("   Hand ", i, ": Score=", palmDetections[i].score,
                                " Pos=(", palmDetections[i].x, ", ", palmDetections[i].y, ")");
                }
            } else {
                Logger::info("❌ No palms detected");
            }
        }

        // Time delta for Kalman
        static auto lastTime = std::chrono::steady_clock::now();
        auto currentTime = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(currentTime - lastTime).count();
        lastTime = currentTime;

        // Process each detected hand
        int handCount = 0;
        for (size_t h = 0; h < palmDetections.size() && h < MAX_HANDS; ++h) {
            const auto& palmDetection = palmDetections[h];

            // Hand Landmark Inference
            auto landmarks = _handLandmark->infer(
                frame->data.get(),
                static_cast<int>(frame->width),
                static_cast<int>(frame->height),
                palmDetection
            );

            if (landmarks) {
                // Draw bounding box around ENTIRE hand (all landmarks)
                if (!debugFrame.empty()) {
                    cv::Scalar boxColor = (h == 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 165, 0);
                    cv::Scalar pointColor = (h == 0) ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255);
                    cv::Scalar tipColor = (h == 0) ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 255);
                    cv::Scalar lineColor = boxColor;

                    // Calculate bounding box from all landmarks
                    float minX = 1.0f, maxX = 0.0f, minY = 1.0f, maxY = 0.0f;
                    for (const auto& lm : landmarks->landmarks) {
                        minX = std::min(minX, lm.x);
                        maxX = std::max(maxX, lm.x);
                        minY = std::min(minY, lm.y);
                        maxY = std::max(maxY, lm.y);
                    }

                    // Add padding (10%)
                    float padX = (maxX - minX) * 0.1f;
                    float padY = (maxY - minY) * 0.1f;
                    int bx1 = static_cast<int>((minX - padX) * debugFrame.cols);
                    int by1 = static_cast<int>((minY - padY) * debugFrame.rows);
                    int bx2 = static_cast<int>((maxX + padX) * debugFrame.cols);
                    int by2 = static_cast<int>((maxY + padY) * debugFrame.rows);

                    cv::rectangle(debugFrame, cv::Point(bx1, by1), cv::Point(bx2, by2), boxColor, 2);

                    // Draw hand label
                    char labelStr[16];
                    snprintf(labelStr, sizeof(labelStr), "Hand %zu", h);
                    cv::putText(debugFrame, labelStr, cv::Point(bx1, by1 - 5),
                                cv::FONT_HERSHEY_SIMPLEX, 0.4, boxColor, 1);

                    // Draw landmarks
                    for (size_t i = 0; i < landmarks->landmarks.size(); ++i) {
                        int lx = static_cast<int>(landmarks->landmarks[i].x * debugFrame.cols);
                        int ly = static_cast<int>(landmarks->landmarks[i].y * debugFrame.rows);
                        cv::Scalar color = (i == 4 || i == 8 || i == 12 || i == 16 || i == 20)
                            ? tipColor : pointColor;
                        cv::circle(debugFrame, cv::Point(lx, ly), 3, color, -1);
                    }

                    // Skeleton
                    const int conns[][2] = {{0,1},{1,2},{2,3},{3,4},{0,5},{5,6},{6,7},{7,8},
                        {0,9},{9,10},{10,11},{11,12},{0,13},{13,14},{14,15},{15,16},
                        {0,17},{17,18},{18,19},{19,20}};
                    for (const auto& c : conns) {
                        if (c[0] < (int)landmarks->landmarks.size() && c[1] < (int)landmarks->landmarks.size()) {
                            cv::line(debugFrame,
                                cv::Point(static_cast<int>(landmarks->landmarks[c[0]].x * debugFrame.cols),
                                          static_cast<int>(landmarks->landmarks[c[0]].y * debugFrame.rows)),
                                cv::Point(static_cast<int>(landmarks->landmarks[c[1]].x * debugFrame.cols),
                                          static_cast<int>(landmarks->landmarks[c[1]].y * debugFrame.rows)),
                                lineColor, 1);
                        }
                    }
                }

                // Kalman + Gesture (per hand)
                float palmZ = 0.0f;  // TODO: Phase 4 - Stereo Depth

                Point3D palm3D = {landmarks->palmCenterX, landmarks->palmCenterY, palmZ};
                _handTrackers[h]->predict(dt);
                _handTrackers[h]->update(palm3D);

                std::vector<TrackingResult::NormalizedPoint> lmPoints;
                for (const auto& lm : landmarks->landmarks) lmPoints.push_back(lm);
                auto gesture = _gestureFSMs[h]->update(lmPoints);

                // Build TrackingResult for OSC
                TrackingResult result;
                result.handId = static_cast<int>(h);  // Hand ID for OSC routing
                result.palmPosition = _handTrackers[h]->getPosition();
                result.velocity = _handTrackers[h]->getVelocity();
                result.gesture = gesture;
                result.vipLocked = _handTrackers[h]->isLocked();
                result.timestamp = std::chrono::steady_clock::now();
                for (size_t i = 0; i < 21 && i < landmarks->landmarks.size(); ++i)
                    result.landmarks.push_back(landmarks->landmarks[i]);
                _oscQueue->try_push(result);

                // Update tracking state for stats display
                _handStates[h].palmX = result.palmPosition.x;
                _handStates[h].palmY = result.palmPosition.y;
                _handStates[h].palmZ = result.palmPosition.z;
                _handStates[h].velX = result.velocity.vx;
                _handStates[h].velY = result.velocity.vy;
                _handStates[h].gesture = GestureFSM::getStateName(gesture);
                _handStates[h].vipLocked = result.vipLocked;

                handCount++;
            }
        }

        _lastHandCount = handCount;

        // Reset unused hand trackers (prediction only mode)
        for (int h = handCount; h < MAX_HANDS; ++h) {
            _handTrackers[h]->predict(dt);  // Keep predicting to avoid jumps
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
    // Semi-transparent status box (larger for 2-hand info with velocity)
    int boxHeight = 100 + (_lastHandCount > 0 ? _lastHandCount * 75 : 0);
    cv::Mat overlay = debugFrame.clone();
    cv::rectangle(overlay, cv::Rect(5, 5, 280, boxHeight), cv::Scalar(0, 0, 0), cv::FILLED);
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
    snprintf(fpsStr, sizeof(fpsStr), "FPS: %.1f", static_cast<double>(_currentFps));
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

    // Show details for each detected hand
    for (int h = 0; h < _lastHandCount && h < MAX_HANDS; ++h) {
        const auto& state = _handStates[h];
        cv::Scalar labelColor = (h == 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 165, 0);

        char labelStr[32];
        snprintf(labelStr, sizeof(labelStr), "Hand %d:", h);
        cv::putText(debugFrame, labelStr, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.4, labelColor, 1);
        y += lineHeight;

        char posStr[64];
        snprintf(posStr, sizeof(posStr), "  Pos: (%.2f, %.2f)",
                 static_cast<double>(state.palmX), static_cast<double>(state.palmY));
        cv::putText(debugFrame, posStr, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(200, 255, 200), 1);
        y += lineHeight - 4;

        // Velocity display
        char velStr[64];
        snprintf(velStr, sizeof(velStr), "  Vel: (%.2f, %.2f)",
                 static_cast<double>(state.velX), static_cast<double>(state.velY));
        cv::putText(debugFrame, velStr, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(200, 200, 255), 1);
        y += lineHeight - 4;

        std::string gestureStr = "  Gesture: " + state.gesture;
        cv::Scalar gestureColor = (state.gesture != "None" && state.gesture != "Palm")
            ? cv::Scalar(0, 255, 255) : cv::Scalar(200, 200, 200);
        cv::putText(debugFrame, gestureStr, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.35, gestureColor, 1);
        y += lineHeight;
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

