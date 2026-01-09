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
    Logger::info("ProcessingLoop stopped.");
}

bool ProcessingLoop::isRunning() const {
    return _running;
}

void ProcessingLoop::loop() {
    // Lazy TensorRT initialization in BACKGROUND thread (doesn't block frame processing)
#ifdef ENABLE_TENSORRT
    if (!_inferenceInitialized && !_inferenceAttempted) {
        _inferenceAttempted = true;

        // Start TRT init in background thread
        _trtInitThread = std::thread([this]() {
            Logger::info("Initializing TensorRT inference (background thread)...");

            auto palmDetector = std::make_unique<inference::PalmDetector>();
            auto handLandmark = std::make_unique<inference::HandLandmark>();

            inference::PalmDetector::Config palmConfig;
            palmConfig.modelPath = "models/palm_detection.onnx";

            inference::HandLandmark::Config landmarkConfig;
            landmarkConfig.modelPath = "models/hand_landmark.onnx";

            // Check if ONNX files exist
            bool palmExists = std::filesystem::exists(palmConfig.modelPath);
            bool landmarkExists = std::filesystem::exists(landmarkConfig.modelPath);

            if (!palmExists || !landmarkExists) {
                Logger::warn("ONNX models not found!");
                if (!palmExists) Logger::warn("  Missing: ", palmConfig.modelPath);
                if (!landmarkExists) Logger::warn("  Missing: ", landmarkConfig.modelPath);
                Logger::warn("Run: python3 scripts/convert_to_onnx.py");
                Logger::info("Running in preview-only mode");
                return;
            }

            bool palmOk = palmDetector->init(palmConfig);
            bool landmarkOk = landmarkOk ? handLandmark->init(landmarkConfig) : false;

            if (palmOk && landmarkOk) {
                // Transfer ownership to class members (thread-safe)
                std::lock_guard<std::mutex> lock(_trtMutex);
                _palmDetector = std::move(palmDetector);
                _handLandmark = std::move(handLandmark);
                _inferenceInitialized = true;
                Logger::info("TensorRT inference initialized successfully (background)");
            } else {
                Logger::warn("TensorRT inference initialization failed");
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

        _frameCount = 0;
        _lastFpsTime = now;
    }

    // ═══════════════════════════════════════════════════════════
    // V3 Phase 1-2: Debug Preview Only
    // TensorRT inference will be added in Phase 2
    // ═══════════════════════════════════════════════════════════

    bool shouldRenderDebug = _mjpegServer && _mjpegServer->hasClients();
    cv::Mat debugFrame;

    if (shouldRenderDebug) {
        // Convert NV12 to BGR for MJPEG preview
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

            // NV12 to BGR
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
        // CPU fallback
        cv::Mat nv12(frame->height * 3 / 2, frame->width, CV_8UC1, frame->data.get());
        cv::cvtColor(nv12, debugFrame, cv::COLOR_YUV2BGR_NV12);
#endif

        if (!debugFrame.empty()) {
            drawDebugOverlay(debugFrame, frame);
            _mjpegServer->update(debugFrame);
        }
    }

    // ═══════════════════════════════════════════════════════════
    // V3 Phase 2: TensorRT Palm Detection + Hand Landmarks
    // ═══════════════════════════════════════════════════════════

#ifdef ENABLE_TENSORRT
    if (_inferenceInitialized) {
        // Palm Detection
        auto palmDetection = _palmDetector->detect(
            frame->data.get(),
            static_cast<int>(frame->width),
            static_cast<int>(frame->height)
        );

        if (palmDetection) {
            // Hand Landmark Inference
            auto landmarks = _handLandmark->infer(
                frame->data.get(),
                static_cast<int>(frame->width),
                static_cast<int>(frame->height),
                *palmDetection
            );

            if (landmarks) {
                // Calculate delta time for Kalman filter
                static auto lastTime = std::chrono::steady_clock::now();
                auto currentTime = std::chrono::steady_clock::now();
                float dt = std::chrono::duration<float>(currentTime - lastTime).count();
                lastTime = currentTime;

                // Get palm center as 3D point (Z from depth if available, else estimated)
                float palmZ = 500.0f;  // Default 50cm

                // Phase 3: Get depth at palm center (when stereo enabled)
                if (frame->hasStereoData && _stereoDepth) {
                    int palmU = static_cast<int>(landmarks->palmCenterX * frame->monoWidth);
                    int palmV = static_cast<int>(landmarks->palmCenterY * frame->monoHeight);
                    palmZ = _stereoDepth->getDepthAtPoint(
                        frame->monoLeftData.get(),
                        frame->monoRightData.get(),
                        static_cast<int>(frame->monoWidth),
                        static_cast<int>(frame->monoHeight),
                        palmU, palmV
                    );
                }

                Point3D palm3D = {
                    landmarks->palmCenterX,
                    landmarks->palmCenterY,
                    palmZ
                };

                // Kalman filter update
                _handTracker->predict(dt);
                _handTracker->update(palm3D);

                // Gesture FSM update - convert landmarks to expected format
                std::vector<TrackingResult::NormalizedPoint> landmarkPoints;
                landmarkPoints.reserve(21);
                for (const auto& lm : landmarks->landmarks) {
                    landmarkPoints.push_back(lm);
                }
                auto gesture = _gestureFSM->update(landmarkPoints);

                // Prepare tracking result for OSC
                TrackingResult result;
                result.palmPosition = _handTracker->getPosition();
                result.velocity = _handTracker->getVelocity();
                result.gesture = gesture;
                result.vipLocked = _handTracker->isLocked();
                result.timestamp = std::chrono::steady_clock::now();

                // Copy landmarks
                for (size_t i = 0; i < 21 && i < landmarks->landmarks.size(); ++i) {
                    result.landmarks.push_back(landmarks->landmarks[i]);
                }

                // Send to OSC queue
                _oscQueue->try_push(result);
            }
        }
    }
#endif

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
    // Semi-transparent status box
    cv::Mat overlay = debugFrame.clone();
    cv::rectangle(overlay, cv::Rect(5, 5, 250, 100), cv::Scalar(0, 0, 0), cv::FILLED);
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

    // V3 Status
    cv::putText(debugFrame, "V3 Sensor-Only Mode", cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 200, 0), 1);
    y += lineHeight;

    // System Performance (update every 5s)
    auto perfNow = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(perfNow - _lastPerfUpdate).count() >= 5) {
        _performanceSummary = SystemMonitor::getPerformanceSummary();
        _lastPerfUpdate = perfNow;
    }
    cv::putText(debugFrame, _performanceSummary, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(200, 200, 255), 1);
    y += lineHeight;

    // Stereo Status
    std::string stereoStr = frame->hasStereoData ? "Stereo: Ready" : "Stereo: Disabled";
    cv::putText(debugFrame, stereoStr, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1);
}

} // namespace core

