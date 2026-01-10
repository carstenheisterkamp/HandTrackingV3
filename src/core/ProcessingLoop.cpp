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
#include "core/PlayVolume.hpp"
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

    // Phase 4: Initialize Play Volume for standing player at 2m
    // Optimized for 220cm × 125cm display, arm reach coverage
    _playVolume = std::make_unique<PlayVolume>(getGamePlayVolume());
    Logger::info("Play Volume initialized (GAME): ",
                 _playVolume->getWidth() * 100, "% x ",
                 _playVolume->getHeight() * 100, "% coverage, ",
                 "Z: ", _playVolume->minZ, "-", _playVolume->maxZ, "mm (",
                 _playVolume->minZ / 1000.0f, "m-", _playVolume->maxZ / 1000.0f, "m)");

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
            palmConfig.modelPath = _palmModelPath;  // Use configurable path

            inference::HandLandmark::Config landmarkConfig;
            landmarkConfig.modelPath = _landmarkModelPath;  // Use configurable path

            // Check if ONNX files exist
            Logger::info("🔍 Checking for ONNX models...");
            Logger::info("   Palm model path: ", palmConfig.modelPath);
            Logger::info("   Landmark model path: ", landmarkConfig.modelPath);

            bool palmExists = std::filesystem::exists(palmConfig.modelPath);
            bool landmarkExists = std::filesystem::exists(landmarkConfig.modelPath);

            Logger::info("   Palm exists: ", palmExists ? "YES" : "NO");
            Logger::info("   Landmark exists: ", landmarkExists ? "YES" : "NO");

            // Log file sizes to verify model type
            if (palmExists) {
                auto palmSize = std::filesystem::file_size(palmConfig.modelPath);
                Logger::info("   Palm model size: ", palmSize / 1024, " KB");
            }
            if (landmarkExists) {
                auto landmarkSize = std::filesystem::file_size(landmarkConfig.modelPath);
                Logger::info("   Landmark model size: ", landmarkSize / 1024, " KB");
            }

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

                // Initialize StereoDepth with default calibration
                if (_stereoDepth && _stereoDepth->loadFromDevice(nullptr)) {
                    _stereoInitialized = true;
                    Logger::info("✅ StereoDepth initialized (default OAK-D calibration)");
                } else {
                    Logger::warn("⚠️ StereoDepth initialization failed - Z-coordinate disabled");
                }
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
        Logger::info("Models: ", getModelType(), " (", _palmModelPath.find("_full") != std::string::npos ? "full models" : "lite models", ")");
        Logger::info("Stereo: ", frame->hasStereoData ? "Available" : "Disabled");
        Logger::info("MJPEG: ", (_mjpegServer && _mjpegServer->hasClients()) ? "Clients connected" : "No clients");

        // Hand tracking stats (for both hands)
        Logger::info("🖐 Hands Detected: ", _lastHandCount);
        for (int h = 0; h < _lastHandCount && h < MAX_HANDS; ++h) {
            Logger::info("   Hand ", h, ":");
            Logger::info("     Position: (", _handStates[h].palmX, ", ", _handStates[h].palmY, ", ", _handStates[h].palmZ, ")");
            Logger::info("     Velocity: (", _handStates[h].velX, ", ", _handStates[h].velY, ", ", _handStates[h].velZ, ")");
            Logger::info("     Gesture: ", _handStates[h].gesture);
        }
        Logger::info("TensorRT: ", _inferenceInitialized ? "Ready" : "Not initialized");
        Logger::info("StereoDepth: ", _stereoInitialized ? "Ready" : "Not initialized");
        if (frame->hasStereoData) {
            Logger::info("  Mono L/R: ", frame->monoWidth, "x", frame->monoHeight);
        }

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

        // ═══════════════════════════════════════════════════════════
        // Phase 4: Volume-Filtering
        // Filter out palms OUTSIDE the play volume (before landmark inference)
        // This saves GPU time by not processing hands we'll discard anyway
        // ═══════════════════════════════════════════════════════════
        std::vector<inference::PalmDetector::Detection> filteredDetections;
        int rejectedCount = 0;

        for (const auto& palm : palmDetections) {
            // Check if palm center is inside 2D play volume
            // Note: Z-check will be done after stereo depth computation
            if (_playVolume->contains2D(palm.x, palm.y)) {
                filteredDetections.push_back(palm);
            } else {
                rejectedCount++;
                // Debug: Show rejected palms in preview
                if (!debugFrame.empty()) {
                    int px = static_cast<int>(palm.x * debugFrame.cols);
                    int py = static_cast<int>(palm.y * debugFrame.rows);
                    cv::circle(debugFrame, cv::Point(px, py), 10, cv::Scalar(0, 0, 255), 2);  // Red circle
                    cv::putText(debugFrame, "OUT", cv::Point(px - 15, py - 15),
                                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);
                }
            }
        }

        // Log filtering stats (every 60 frames)
        static int filterLogCounter = 0;
        if (++filterLogCounter % 60 == 1 && rejectedCount > 0) {
            Logger::info("🔲 Volume Filter: ", palmDetections.size(), " detected, ",
                        filteredDetections.size(), " in volume, ",
                        rejectedCount, " rejected");
        }

        // Continue with filtered detections only
        palmDetections = std::move(filteredDetections);

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
                    // Note: Text will be mirrored after frame flip, so we draw it normally here
                    // Position needs to account for the upcoming flip
                    char labelStr[16];
                    snprintf(labelStr, sizeof(labelStr), "Hand %zu", h);

                    // Calculate text size to position it correctly after mirror
                    int baseline = 0;
                    cv::Size textSize = cv::getTextSize(labelStr, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);

                    // After flip, bx1 will become (frameWidth - bx1)
                    // We want text at the TOP LEFT of the box, so we draw at TOP RIGHT before flip
                    int textX = bx2 - textSize.width;  // Right side (will be left after flip)
                    int textY = by1 - 5;

                    // Draw mirrored text (flip horizontally so it reads correctly after frame flip)
                    cv::Mat textROI;
                    cv::Mat textImg = cv::Mat::zeros(textSize.height + baseline, textSize.width, CV_8UC3);
                    cv::putText(textImg, labelStr, cv::Point(0, textSize.height),
                                cv::FONT_HERSHEY_SIMPLEX, 0.4, boxColor, 1);
                    cv::flip(textImg, textImg, 1);  // Flip text horizontally

                    // Place flipped text on frame
                    int y1 = std::max(0, textY - textSize.height);
                    int y2 = std::min(debugFrame.rows, textY);
                    int x1 = std::max(0, textX);
                    int x2 = std::min(debugFrame.cols, textX + textSize.width);

                    if (y2 > y1 && x2 > x1) {
                        cv::Mat destROI = debugFrame(cv::Rect(x1, y1, x2 - x1, y2 - y1));
                        cv::Mat srcROI = textImg(cv::Rect(0, 0, x2 - x1, y2 - y1));

                        // Blend text (white pixels from text)
                        for (int row = 0; row < srcROI.rows; ++row) {
                            for (int col = 0; col < srcROI.cols; ++col) {
                                cv::Vec3b pixel = srcROI.at<cv::Vec3b>(row, col);
                                if (pixel[0] > 0 || pixel[1] > 0 || pixel[2] > 0) {
                                    destROI.at<cv::Vec3b>(row, col) = pixel;
                                }
                            }
                        }
                    }

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
                float palmZ = 0.0f;

                // Phase 3: Stereo Depth - compute Z at palm center
                if (_stereoInitialized && frame->hasStereoData &&
                    frame->monoLeftData && frame->monoRightData) {

                    // Convert normalized palm coords to mono image pixel coords
                    // Note: Mono is 640x400, RGB preview is 640x360
                    int palmPxX = static_cast<int>(landmarks->palmCenterX * frame->monoWidth);
                    int palmPxY = static_cast<int>(landmarks->palmCenterY * frame->monoHeight);

                    // Get depth at palm center (returns mm, or -1 if invalid)
                    float depthMm = _stereoDepth->getDepthAtPoint(
                        frame->monoLeftData.get(),
                        frame->monoRightData.get(),
                        static_cast<int>(frame->monoWidth),
                        static_cast<int>(frame->monoHeight),
                        palmPxX, palmPxY
                    );

                    if (depthMm > 0) {
                        // Convert mm to normalized Z (1.2m-2.8m → 0-1) for Game Volume
                        // Game Volume: minZ=1200mm, maxZ=2800mm, range=1600mm
                        //
                        // IMPORTANT: Normalized (0-1) coordinates allow flexible scaling in Game Engine!
                        // Physical 1.6m depth can map to ANY virtual size in UE:
                        //   - 1:1   → 1.6m virtual (realistic)
                        //   - 10:1  → 16m virtual (large world)
                        //   - 100:1 → 160m virtual ("giant mode")
                        //   - Custom asymmetric scaling per axis
                        // See docs/OSC_QUICK_REFERENCE.md for UE mapping examples
                        palmZ = (depthMm - 1200.0f) / 1600.0f;
                        palmZ = std::max(0.0f, std::min(1.0f, palmZ));  // Clamp to [0,1]

                        // Debug log (every 30 frames)
                        static int depthLogCounter = 0;
                        if (++depthLogCounter % 30 == 1) {
                            Logger::info("📐 Hand ", h, " depth: ", depthMm, "mm (",
                                        depthMm / 1000.0f, "m) → Z=", palmZ);
                        }
                    }
                }

                Point3D palm3D = {landmarks->palmCenterX, landmarks->palmCenterY, palmZ};
                _handTrackers[h]->predict(dt);
                _handTrackers[h]->update(palm3D);

                std::vector<TrackingResult::NormalizedPoint> lmPoints;
                for (const auto& lm : landmarks->landmarks) lmPoints.push_back(lm);

                // Determine handedness: Use palm X position as heuristic
                // Right hand typically appears on the left side of the image (mirrored view)
                // Left hand typically appears on the right side
                bool isRightHand = landmarks->palmCenterX < 0.5f;

                auto gesture = _gestureFSMs[h]->update(lmPoints, isRightHand);

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

                // Calculate delta (acceleration) from velocity change
                _handStates[h].deltaX = result.velocity.vx - _handStates[h].prevVelX;
                _handStates[h].deltaY = result.velocity.vy - _handStates[h].prevVelY;
                _handStates[h].deltaZ = result.velocity.vz - _handStates[h].prevVelZ;

                // Store current velocity for next frame's delta
                _handStates[h].prevVelX = result.velocity.vx;
                _handStates[h].prevVelY = result.velocity.vy;
                _handStates[h].prevVelZ = result.velocity.vz;

                _handStates[h].velX = result.velocity.vx;
                _handStates[h].velY = result.velocity.vy;
                _handStates[h].velZ = result.velocity.vz;
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
        // Mirror camera image horizontally BEFORE drawing overlay
        // This makes the camera view act like a mirror (natural)
        // But overlay text remains readable
        cv::flip(debugFrame, debugFrame, 1);  // 1 = horizontal flip

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
    // ═══════════════════════════════════════════════════════════
    // 1. Draw Play Volume (3D Box) - Phase 4 Active Volume
    // ═══════════════════════════════════════════════════════════
    // Use PlayVolume values (16:9 aspect ratio, 0.5m-2.5m depth)
    const float volumeMinX = _playVolume->minX;
    const float volumeMaxX = _playVolume->maxX;
    const float volumeMinY = _playVolume->minY;
    const float volumeMaxY = _playVolume->maxY;

    int vx1 = static_cast<int>(volumeMinX * debugFrame.cols);
    int vx2 = static_cast<int>(volumeMaxX * debugFrame.cols);
    int vy1 = static_cast<int>(volumeMinY * debugFrame.rows);
    int vy2 = static_cast<int>(volumeMaxY * debugFrame.rows);

    cv::Scalar volumeColor = cv::Scalar(100, 200, 100);  // Light green
    cv::rectangle(debugFrame, cv::Point(vx1, vy1), cv::Point(vx2, vy2),
                  volumeColor, 2, cv::LINE_AA);

    // Corner markers for 3D effect
    int markerSize = 20;
    cv::line(debugFrame, cv::Point(vx1, vy1), cv::Point(vx1 + markerSize, vy1), volumeColor, 3);
    cv::line(debugFrame, cv::Point(vx1, vy1), cv::Point(vx1, vy1 + markerSize), volumeColor, 3);
    cv::line(debugFrame, cv::Point(vx2, vy1), cv::Point(vx2 - markerSize, vy1), volumeColor, 3);
    cv::line(debugFrame, cv::Point(vx2, vy1), cv::Point(vx2, vy1 + markerSize), volumeColor, 3);
    cv::line(debugFrame, cv::Point(vx1, vy2), cv::Point(vx1 + markerSize, vy2), volumeColor, 3);
    cv::line(debugFrame, cv::Point(vx1, vy2), cv::Point(vx1, vy2 - markerSize), volumeColor, 3);
    cv::line(debugFrame, cv::Point(vx2, vy2), cv::Point(vx2 - markerSize, vy2), volumeColor, 3);
    cv::line(debugFrame, cv::Point(vx2, vy2), cv::Point(vx2, vy2 - markerSize), volumeColor, 3);

    // Z-Depth indication and filter status
    cv::putText(debugFrame, "GAME VOLUME (FULLSCREEN) - ACTIVE",
                cv::Point(vx1 + 10, vy1 + 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, volumeColor, 1, cv::LINE_AA);

    char depthStr[64];
    snprintf(depthStr, sizeof(depthStr), "Z: %.1fm - %.1fm (Standing @ 2m)",
             _playVolume->minZ / 1000.0f, _playVolume->maxZ / 1000.0f);
    cv::putText(debugFrame, depthStr,
                cv::Point(vx1 + 10, vy1 + 45),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, volumeColor, 1, cv::LINE_AA);

    // ═══════════════════════════════════════════════════════════
    // 2. Info Panel (Status Box)
    // ═══════════════════════════════════════════════════════════
    // Always show 2 hands (even if not detected) to prevent flickering
    // Box height: Base (118 = 6 header lines + margin) + per-hand (95 = 4 lines × 18 + margins)
    int boxHeight = 118 + (MAX_HANDS * 95);  // Fixed height for 2 hands with delta + model info
    cv::Mat overlay = debugFrame.clone();
    cv::rectangle(overlay, cv::Rect(5, 5, 320, boxHeight), cv::Scalar(0, 0, 0), cv::FILLED);
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

    // Model Type (LITE or FULL)
    std::string modelType = getModelType();
    cv::Scalar modelColor = (modelType == "FULL") ? cv::Scalar(255, 165, 0) : cv::Scalar(0, 255, 0);  // Orange for FULL, Green for LITE
    std::string modelText = "Models: " + modelType;
    cv::putText(debugFrame, modelText, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.4, modelColor, 1);
    y += lineHeight;

    // Stereo Status
    if (_stereoInitialized && frame->hasStereoData) {
        cv::putText(debugFrame, "Stereo: Active", cv::Point(10, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
    } else {
        cv::putText(debugFrame, "Stereo: Disabled", cv::Point(10, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(128, 128, 128), 1);
    }
    y += lineHeight;

    // Hand Detection Status
    char handStr[64];
    snprintf(handStr, sizeof(handStr), "Hands Detected: %d / 2", _lastHandCount);
    cv::Scalar handColor = (_lastHandCount > 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(128, 128, 128);
    cv::putText(debugFrame, handStr, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, handColor, 1);
    y += lineHeight + 5;

    // ═══════════════════════════════════════════════════════════
    // 3. Hand Details (ALWAYS show both slots)
    // ═══════════════════════════════════════════════════════════
    for (int h = 0; h < MAX_HANDS; ++h) {
        bool detected = (h < _lastHandCount);
        const auto& state = _handStates[h];

        cv::Scalar labelColor = detected
            ? ((h == 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 165, 0))
            : cv::Scalar(80, 80, 80);  // Gray for undetected

        // Hand label
        char labelStr[32];
        snprintf(labelStr, sizeof(labelStr), "Hand %d: %s", h, detected ? "ACTIVE" : "NOT DETECTED");
        cv::putText(debugFrame, labelStr, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.4, labelColor, 1);
        y += lineHeight;

        // Position (always show, 0,0,0 if not detected)
        char posStr[80];
        if (detected) {
            snprintf(posStr, sizeof(posStr), "  Pos: (%.2f, %.2f, %.2f)",
                     static_cast<double>(state.palmX),
                     static_cast<double>(state.palmY),
                     static_cast<double>(state.palmZ));
        } else {
            snprintf(posStr, sizeof(posStr), "  Pos: (0.00, 0.00, 0.00)");
        }
        cv::putText(debugFrame, posStr, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.35,
                    detected ? cv::Scalar(200, 255, 200) : cv::Scalar(60, 60, 60), 1);
        y += lineHeight - 4;

        // Velocity (always show, 0,0,0 if not detected)
        char velStr[80];
        if (detected) {
            snprintf(velStr, sizeof(velStr), "  Vel: (%.2f, %.2f, %.2f)",
                     static_cast<double>(state.velX),
                     static_cast<double>(state.velY),
                     static_cast<double>(state.velZ));
        } else {
            snprintf(velStr, sizeof(velStr), "  Vel: (0.00, 0.00, 0.00)");
        }
        cv::putText(debugFrame, velStr, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.35,
                    detected ? cv::Scalar(200, 200, 255) : cv::Scalar(60, 60, 60), 1);
        y += lineHeight - 4;

        // Delta/Acceleration (always show, 0,0,0 if not detected)
        char deltaStr[80];
        if (detected) {
            snprintf(deltaStr, sizeof(deltaStr), "  Delta: (%.2f, %.2f, %.2f)",
                     static_cast<double>(state.deltaX),
                     static_cast<double>(state.deltaY),
                     static_cast<double>(state.deltaZ));
        } else {
            snprintf(deltaStr, sizeof(deltaStr), "  Delta: (0.00, 0.00, 0.00)");
        }
        cv::putText(debugFrame, deltaStr, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.35,
                    detected ? cv::Scalar(255, 200, 200) : cv::Scalar(60, 60, 60), 1);
        y += lineHeight - 4;

        // Gesture name (always show)
        char gestureStr[64];
        if (detected) {
            snprintf(gestureStr, sizeof(gestureStr), "  Gesture: %s", state.gesture.c_str());
        } else {
            snprintf(gestureStr, sizeof(gestureStr), "  Gesture: None");
        }
        cv::putText(debugFrame, gestureStr, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                    detected ? cv::Scalar(0, 255, 255) : cv::Scalar(60, 60, 60), 1);
        y += lineHeight + 5;
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

