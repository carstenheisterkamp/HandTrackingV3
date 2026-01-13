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
#include "core/SessionFSM.hpp"
#include "core/StereoDepth.hpp"

#include <filesystem>
#include <chrono>
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <unistd.h>
#include <limits.h>

#ifdef ENABLE_TENSORRT
#include "inference/PalmDetector.hpp"
#include "inference/HandLandmark.hpp"
#endif

#ifdef ENABLE_CUDA
#include "core/StereoKernel.hpp"
#include <nppi_color_conversion.h>
#include <npp.h>
#include <cuda_runtime.h>
#endif

// Hilfsfunktion: robuste Auflösung von Modellpfaden
static std::string resolveModelPath(const std::string& inputPath) {
    namespace fs = std::filesystem;
    try {
        fs::path in(inputPath);
        // Wenn absolut und vorhanden, direkt zurückgeben
        if (in.is_absolute() && fs::exists(in)) return in.string();

        // Kandidatenliste aufbauen
        std::vector<fs::path> candidates;
        fs::path filename = in.filename();

        // 1) relativ zum aktuellen Arbeitsverzeichnis
        candidates.push_back(fs::current_path() / in);
        candidates.push_back(fs::current_path() / "models" / filename);

        // 2) relativ zum Executable-Verzeichnis
        char buf[PATH_MAX];
        ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if (len > 0) {
            buf[len] = '\0';
            fs::path exePath(buf);
            fs::path exeDir = exePath.parent_path();
            candidates.push_back(exeDir / in);
            candidates.push_back(exeDir / "models" / filename);
            candidates.push_back(exeDir / ".." / in);
            candidates.push_back(exeDir / ".." / "models" / filename);
        }

        // 3) Standard-Projektpfad auf Jetson
        candidates.push_back(fs::path("/home/nvidia/dev/HandTrackingV3") / in);
        candidates.push_back(fs::path("/home/nvidia/dev/HandTrackingV3/models") / filename);

        for (const auto& c : candidates) {
            if (fs::exists(c)) {
                core::Logger::info("Resolved model path: '", inputPath, "' -> '", c.string(), "'");
                return c.string();
            }
        }

        // Nichts gefunden, original zurückgeben
        core::Logger::warn("Could not resolve model path: ", inputPath, ". Using as-is.");
        return inputPath;
    } catch (const std::exception& e) {
        core::Logger::warn("resolveModelPath exception: ", e.what());
        return inputPath;
    }
}

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
        _sessionFSMs[i] = std::make_unique<SessionFSM>();  // Phase 4
    }
    _stereoDepth = std::make_unique<StereoDepth>();

    // Phase 4: Initialize Play Volume with 5% margin (90% of image size)
    _playVolume = std::make_unique<PlayVolume>(getDefaultPlayVolume());
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

            // Pfade robust auflösen (WorkingDirectory kann 'build' sein)
            palmConfig.modelPath = resolveModelPath(palmConfig.modelPath);
            landmarkConfig.modelPath = resolveModelPath(landmarkConfig.modelPath);

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

    Frame* lastFrame = nullptr;
    while (_running) {
        Frame* frame = nullptr;

        // OPTIMIZATION: Non-blocking pop - nie warten auf Queue
        // Falls kein Frame verfügbar → verwende letzten Frame (Predictive Tracking)
        bool isNewFrame = _inputQueue->pop_front(frame);

        if (isNewFrame && frame) {
            // Neuer Frame vom OAK-D → Normal processing
            processFrame(frame);
            // Nur neue Frames für FPS-Zählung verwenden!
            // Wird weiter unten in processFrame() gemacht
            if (lastFrame) {
                _framePool->release(lastFrame);
            }
            lastFrame = frame;  // Cache für next iteration
        } else if (lastFrame) {
            // Kein neuer Frame → verwende cached Frame mit Prediction
            // Kalman-Tracker extrapoliert Position
            // WICHTIG: Wird für FPS NICHT gezählt (siehe processFrame)
            processFrame(lastFrame);
        } else {
            // Startup: Noch kein Frame → kurz warten
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    // Cleanup
    if (lastFrame) {
        _framePool->release(lastFrame);
    }
}

void ProcessingLoop::processFrame(Frame* frame) {
    auto frameStart = std::chrono::high_resolution_clock::now();

    // Performance profiling accumulators (static for accumulation between frames)
    static long long totalNV12Time = 0;
    static long long totalPalmTime = 0;
    static long long totalLandmarkTime = 0;
    static long long totalStereoTime = 0;
    static long long totalDrawTime = 0;
    static long long totalJpegTime = 0;
    static long long totalFrameTime = 0;
    static long long totalQueueWaitTime = 0;  // NEW: Queue latency tracking
    static int profileFrameCount = 0;

    // ═══════════════════════════════════════════════════════════
    // FPS Tracking with DETAILED PERFORMANCE BREAKDOWN
    // WICHTIG: Nur echte neue Frames vom OAK-D zählen!
    // (Nicht Predictive-Tracked cached Frames)
    // ═══════════════════════════════════════════════════════════

    // Check ob das ein neuer Frame vom OAK-D ist oder ein Re-Processed cached Frame
    static Frame* lastProcessedFrame = nullptr;
    bool isNewFrameFromCamera = (frame != lastProcessedFrame);
    lastProcessedFrame = frame;

    if (isNewFrameFromCamera) {
        // Nur echte neue Frames zählen!
        _frameCount++;
    }
    profileFrameCount++;

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - _lastFpsTime).count();
    if (elapsed >= 2000 && isNewFrameFromCamera) {
        // Nur bei neuen Frames und mindestens 2 Sekunden vergangen
        _currentFps = _frameCount * 1000.0f / elapsed;

        // Calculate average time per component (in ms)
        float avgNV12 = profileFrameCount > 0 ? totalNV12Time / (float)profileFrameCount / 1000.0f : 0;
        float avgPalm = profileFrameCount > 0 ? totalPalmTime / (float)profileFrameCount / 1000.0f : 0;
        float avgLandmark = profileFrameCount > 0 ? totalLandmarkTime / (float)profileFrameCount / 1000.0f : 0;
        float avgStereo = profileFrameCount > 0 ? totalStereoTime / (float)profileFrameCount / 1000.0f : 0;
        float avgDraw = profileFrameCount > 0 ? totalDrawTime / (float)profileFrameCount / 1000.0f : 0;
        float avgJpeg = profileFrameCount > 0 ? totalJpegTime / (float)profileFrameCount / 1000.0f : 0;
        float avgQueueWait = profileFrameCount > 0 ? totalQueueWaitTime / (float)profileFrameCount / 1000.0f : 0;
        float totalMeasured = avgNV12 + avgPalm + avgLandmark + avgStereo + avgDraw + avgJpeg + avgQueueWait;
        float frameBudget = 1000.0f / _currentFps;
        float avgFrameMs = profileFrameCount > 0 ? totalFrameTime / (float)profileFrameCount / 1000.0f : 0;
        float unaccounted = avgFrameMs - totalMeasured;  // time spent elsewhere (scheduling, queues, etc)

        Logger::info("═══ V3 PERFORMANCE BREAKDOWN ═══");
        Logger::info("FPS: ", _currentFps, " (OSC: 28 Hz fixed)");
        Logger::info("Frame: ", frame->width, "x", frame->height);
        Logger::info("Models: ", getModelType(), " (", _palmModelPath.find("_full") != std::string::npos ? "FULL" : "LITE", ")");
        Logger::info("───────────────────────────────");
        Logger::info("⏱️  Timing per Frame (average ms):");
        Logger::info("  Queue Wait:    ", avgQueueWait, " ms");
        Logger::info("  NV12→BGR:      ", avgNV12, " ms");
        Logger::info("  Palm Detect:   ", avgPalm, " ms");
        Logger::info("  Landmark:      ", avgLandmark, " ms");
        Logger::info("  Stereo Depth:  ", avgStereo, " ms");
        Logger::info("  Draw Overlay:  ", avgDraw, " ms");
        Logger::info("  JPEG Encode:   ", avgJpeg, " ms");
        Logger::info("  ───────────────");
        Logger::info("  Total Measured:", totalMeasured, " ms");
        Logger::info("  Avg Frame Time:", avgFrameMs, " ms");
        Logger::info("  Frame Budget:  ", frameBudget, " ms");
        Logger::info("  Unaccounted:   ", unaccounted, " ms");

        // Reset accumulators
        totalNV12Time = 0;
        totalPalmTime = 0;
        totalLandmarkTime = 0;
        totalStereoTime = 0;
        totalDrawTime = 0;
        totalJpegTime = 0;
        totalFrameTime = 0;
        totalQueueWaitTime = 0;
        profileFrameCount = 0;


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

    // Wenn kein neuer Frame vom OAK-D → skip rest (nur Kalman prediction)
    if (!isNewFrameFromCamera) {
        // Predictive Tracking ohne Inference
        for (int h = 0; h < MAX_HANDS; ++h) {
            _handTrackers[h]->predict(0.033f);  // Predict mit 30ms Delta
        }
        return;  // Nichts mehr zu tun
    }

    // ═══════════════════════════════════════════════════════════
    // Step 1: Convert NV12 to BGR for visualization
    // OPTIMIZATION: Skip komplett wenn kein MJPEG Client connected
    // ═══════════════════════════════════════════════════════════
    static bool headlessMode = std::getenv("HANDTRACKING_HEADLESS") != nullptr;
    bool hasClients = _mjpegServer && _mjpegServer->hasClients();
    bool shouldRenderDebug = !headlessMode && hasClients;
    cv::Mat debugFrame;

    auto nv12Start = std::chrono::high_resolution_clock::now();
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
    auto nv12End = std::chrono::high_resolution_clock::now();
    totalNV12Time += std::chrono::duration_cast<std::chrono::microseconds>(nv12End - nv12Start).count();

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

        // OPTIMIZATION: Run Palm Detection every 3rd frame (skip strategy)
        // Hands don't move significantly between frames, so we can reuse detections
        // This saves ~33% of Palm Inference time (most expensive single operation)
        // Client doesn't know - OSC stays 28 Hz with cached data
        static int palmSkipCounter = 0;
        static std::vector<inference::PalmDetector::Detection> cachedPalmDetections;

        std::vector<inference::PalmDetector::Detection> palmDetections;

        auto palmStart = std::chrono::high_resolution_clock::now();
        if (palmSkipCounter++ % 3 == 0) {
            // Run Palm Detection (every 3rd frame)
            palmDetections = _palmDetector->detectAll(
                frame->data.get(),
                static_cast<int>(frame->width),
                static_cast<int>(frame->height),
                MAX_HANDS
            );
            cachedPalmDetections = palmDetections;  // Cache for next 2 frames
        } else {
            // Skip Palm Detection, reuse cached results
            palmDetections = cachedPalmDetections;
        }
        auto palmEnd = std::chrono::high_resolution_clock::now();
        totalPalmTime += std::chrono::duration_cast<std::chrono::microseconds>(palmEnd - palmStart).count();

        // ═══════════════════════════════════════════════════════════
        // Phase 4: Volume-Filtering
        // Filter out palms OUTSIDE the play volume (before landmark inference)
        // This saves GPU time by not processing hands we'll discard anyway
        // ═══════════════════════════════════════════════════════════
        std::vector<inference::PalmDetector::Detection> filteredDetections;
        int rejectedCount = 0;

        for (auto palm : palmDetections) {
            // ROI Coordinate Denormalization
            // If ROI mode: Palm coords are (0-1) in 1080×1080 ROI quadrat
            // We need to map them back to Full 1920×1080 (0-1) for Kalman tracking
            if (_useROI) {
                // palm.x, palm.y are in ROI space (0-1)
                // Convert to full frame pixel coords:
                // pixel_x = palm.x * _roiSize + _roiOffsetX
                // pixel_y = palm.y * _roiSize + _roiOffsetY
                // Then normalize back to (0-1):
                // norm_x = pixel_x / 1920
                // norm_y = pixel_y / 1080
                palm.x = (palm.x * _roiSize + _roiOffsetX) / 1920.0f;
                palm.y = (palm.y * _roiSize + _roiOffsetY) / 1080.0f;
            }

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

        // OPTIMIZATION: Landmark Skip Strategy (every 2nd frame)
        // Reduces Landmark Inference cost by 50% (~5ms savings)
        static int landmarkSkipCounter = 0;
        static std::vector<std::optional<inference::HandLandmark::Result>> cachedLandmarks(MAX_HANDS);
        bool shouldRunLandmark = (landmarkSkipCounter++ % 2 == 0);

        for (size_t h = 0; h < palmDetections.size() && h < MAX_HANDS; ++h) {
            const auto& palmDetection = palmDetections[h];

            // Hand Landmark Inference - WITH TIMING AND SKIP STRATEGY
            auto landmarkStart = std::chrono::high_resolution_clock::now();
            std::optional<inference::HandLandmark::Result> landmarks;

            if (shouldRunLandmark) {
                // Run Landmark Inference (every 2nd frame)
                landmarks = _handLandmark->infer(
                    frame->data.get(),
                    static_cast<int>(frame->width),
                    static_cast<int>(frame->height),
                    palmDetection
                );
                cachedLandmarks[h] = landmarks;  // Cache for next frame
            } else {
                // Skip Landmark Inference, reuse cached results
                landmarks = cachedLandmarks[h];
            }
            auto landmarkEnd = std::chrono::high_resolution_clock::now();
            totalLandmarkTime += std::chrono::duration_cast<std::chrono::microseconds>(landmarkEnd - landmarkStart).count();

            if (landmarks) {
                // ROI Coordinate Denormalization for Landmarks
                // If ROI mode: All landmark coords are (0-1) in 1080×1080 ROI quadrat
                // Convert back to Full 1920×1080 (0-1) for Kalman tracking
                if (_useROI) {
                    for (auto& lm : landmarks->landmarks) {
                        // Map from ROI space to Full frame space
                        lm.x = (lm.x * _roiSize + _roiOffsetX) / 1920.0f;
                        lm.y = (lm.y * _roiSize + _roiOffsetY) / 1080.0f;
                        // Z coordinate unchanged (depth)
                    }
                    // Also denormalize palm center
                    landmarks->palmCenterX = (landmarks->palmCenterX * _roiSize + _roiOffsetX) / 1920.0f;
                    landmarks->palmCenterY = (landmarks->palmCenterY * _roiSize + _roiOffsetY) / 1080.0f;
                }

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
                float depthMm = -1.0f;  // Store depth for display

                // Phase 3: Stereo Depth - compute Z at palm center - WITH TIMING AND CACHING
                // OPTIMIZATION: Cache depth for 3 frames (depth changes slowly)
                static int stereoSkipCounter = 0;
                static float cachedDepth[MAX_HANDS] = {-1.0f, -1.0f};
                bool shouldRunStereo = (stereoSkipCounter++ % 3 == 0);

                auto stereoStart = std::chrono::high_resolution_clock::now();
                if (_stereoInitialized && frame->hasStereoData &&
                    frame->monoLeftData && frame->monoRightData) {

                    if (shouldRunStereo) {
                        // Compute depth (every 3rd frame)
                        int palmPxX = static_cast<int>(landmarks->palmCenterX * frame->monoWidth);
                        int palmPxY = static_cast<int>(landmarks->palmCenterY * frame->monoHeight);

                        depthMm = _stereoDepth->getDepthAtPoint(
                            frame->monoLeftData.get(),
                            frame->monoRightData.get(),
                            static_cast<int>(frame->monoWidth),
                            static_cast<int>(frame->monoHeight),
                            palmPxX, palmPxY
                        );
                        cachedDepth[h] = depthMm;  // Cache for next 2 frames
                    } else {
                        // Use cached depth
                        depthMm = cachedDepth[h];
                    }

                    if (depthMm > 0) {
                        // Convert mm to normalized Z (0.0 - 1.0) for Game Volume
                        palmZ = (depthMm - Z_MIN_MM) / Z_RANGE_MM;
                        palmZ = std::max(0.0f, std::min(1.0f, palmZ));  // Clamp to [0,1]

                        // Debug log (every 30 frames)
                        static int depthLogCounter = 0;
                        if (++depthLogCounter % 30 == 1) {
                            Logger::info("📐 Hand ", h, " depth: ", depthMm, "mm (",
                                        depthMm / 1000.0f, "m) → Z=", palmZ);
                        }

                        // Display depth at hand in preview (before frame flip)
                        if (!debugFrame.empty()) {
                            // ...existing depth display code...
                        }
                    }
                }
                auto stereoEnd = std::chrono::high_resolution_clock::now();
                totalStereoTime += std::chrono::duration_cast<std::chrono::microseconds>(stereoEnd - stereoStart).count();

                Point3D palm3D = {landmarks->palmCenterX, landmarks->palmCenterY, palmZ};
                _handTrackers[h]->predict(dt);
                _handTrackers[h]->update(palm3D);

                std::vector<Point3D> lmPoints;
                for (const auto& lm : landmarks->landmarks) lmPoints.push_back(lm);

                // Determine handedness: Use palm X position as heuristic
                // Right hand typically appears on the left side of the image (mirrored view)
                // Left hand typically appears on the right side
                bool isRightHand = landmarks->palmCenterX < 0.5f;

                auto gesture = _gestureFSMs[h]->update(lmPoints, isRightHand);

                // ═══════════════════════════════════════════════════════════
                // Phase 4: Session FSM - Update player session state
                // Redundancy reduction: Only emit ENTER/EXIT events
                // (Unreal uses /hand/{id}/palm for continuous tracking + own timeout logic)
                // ═══════════════════════════════════════════════════════════
                bool palmInVolume = _playVolume->contains2D(landmarks->palmCenterX, landmarks->palmCenterY);
                bool sessionStateChanged = _sessionFSMs[h]->update(palmInVolume);
                auto sessionState = _sessionFSMs[h]->getState();

                // Emit OSC events ONLY on state transitions (ENTER/EXIT)
                if (sessionStateChanged) {
                    TrackingResult sessionEvent;
                    sessionEvent.handId = static_cast<int>(h);
                    sessionEvent.timestamp = std::chrono::steady_clock::now();

                    if (sessionState == SessionState::ACTIVE) {
                        sessionEvent.osc_event = "/player/session/enter";  // IDLE → ACTIVE
                        Logger::info("📤 OSC: /player/session/enter (Hand ", h, ")");
                    } else if (sessionState == SessionState::IDLE) {
                        // Only exit on IDLE (skip LOST state event)
                        sessionEvent.osc_event = "/player/session/exit";   // LOST → IDLE (or direct)
                        Logger::info("📤 OSC: /player/session/exit (Hand ", h, ")");
                    }
                    // Note: /player/session/active is redundant with /hand/{id}/palm
                    // Unreal implements own timeout: no Palm > 100ms → Lost

                    _oscQueue->try_push(sessionEvent);
                }


                // Build TrackingResult for OSC
                TrackingResult result;
                result.handId = static_cast<int>(h);  // Hand ID for OSC routing
                result.palmPosition = _handTrackers[h]->getPosition();
                result.velocity = _handTrackers[h]->getVelocity();

                // Calculate delta (acceleration) from velocity change (pre-inversion)
                result.delta.dx = result.velocity.vx - _handStates[h].prevVelX;
                result.delta.dy = result.velocity.vy - _handStates[h].prevVelY;
                result.delta.dz = result.velocity.vz - _handStates[h].prevVelZ;

                // No Y-axis inversion - send raw camera coordinates to OSC
                // (Y=0 at top, Y=1 at bottom - standard image coordinates)

                result.gesture = gesture;
                result.gestureConfidence = _gestureFSMs[h]->getConfidence();  // 0-1 confidence
                result.palmConfidence = palmDetection.score;  // Palm detection score
                result.landmarkPresence = landmarks->presence;  // Landmark presence confidence
                result.vipLocked = _handTrackers[h]->isLocked();
                result.timestamp = std::chrono::steady_clock::now();
                for (size_t i = 0; i < 21 && i < landmarks->landmarks.size(); ++i)
                    result.landmarks.push_back(landmarks->landmarks[i]);
                _oscQueue->try_push(result);

                // Update tracking state for stats display
                _handStates[h].palmX = _handTrackers[h]->getPosition().x;
                _handStates[h].palmY = _handTrackers[h]->getPosition().y;
                _handStates[h].palmZ = _handTrackers[h]->getPosition().z;

                // Store delta for display
                _handStates[h].deltaX = result.delta.dx;
                _handStates[h].deltaY = result.delta.dy;
                _handStates[h].deltaZ = result.delta.dz;

                // Store current velocity for next frame's delta (non-inverted)
                _handStates[h].prevVelX = _handTrackers[h]->getVelocity().vx;
                _handStates[h].prevVelY = _handTrackers[h]->getVelocity().vy;
                _handStates[h].prevVelZ = _handTrackers[h]->getVelocity().vz;

                _handStates[h].velX = _handTrackers[h]->getVelocity().vx;
                _handStates[h].velY = _handTrackers[h]->getVelocity().vy;
                _handStates[h].velZ = _handTrackers[h]->getVelocity().vz;
                _handStates[h].gesture = GestureFSM::getStateName(gesture);
                _handStates[h].vipLocked = result.vipLocked;
                _handStates[h].isRightHand = isRightHand;  // Store handedness for visualization

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
    // Step 3: Send to MJPEG (AFTER drawing detections) - WITH TIMING
    // ═══════════════════════════════════════════════════════════
    auto drawStart = std::chrono::high_resolution_clock::now();
    long long jpegDuration = 0;
    if (shouldRenderDebug && !debugFrame.empty()) {
        // Mirror camera image horizontally BEFORE drawing overlay
        cv::flip(debugFrame, debugFrame, 1);  // 1 = horizontal flip

        drawDebugOverlay(debugFrame, frame);

        auto jpegStart = std::chrono::high_resolution_clock::now();
        _mjpegServer->update(debugFrame);
        auto jpegEnd = std::chrono::high_resolution_clock::now();
        jpegDuration = std::chrono::duration_cast<std::chrono::microseconds>(jpegEnd - jpegStart).count();
        totalJpegTime += jpegDuration;
    }
    auto drawEnd = std::chrono::high_resolution_clock::now();
    long long drawDuration = std::chrono::duration_cast<std::chrono::microseconds>(drawEnd - drawStart).count();
    totalDrawTime += (drawDuration - jpegDuration);

    // ═══════════════════════════════════════════════════════════
    // Timing
    // ═══════════════════════════════════════════════════════════
    auto frameEnd = std::chrono::high_resolution_clock::now();
    auto frameDuration = std::chrono::duration_cast<std::chrono::microseconds>(frameEnd - frameStart).count();

    totalFrameTime += frameDuration;

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
    // Right-align volume text to avoid hand overlay overlap
    char volTitle[64];
    snprintf(volTitle, sizeof(volTitle), "GAME VOLUME (%dx%d) - ACTIVE",
             static_cast<int>(frame->width), static_cast<int>(frame->height));
    int base = 0;
    cv::Size titleSize = cv::getTextSize(volTitle, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
    int titleX = vx2 - titleSize.width - 10;
    cv::putText(debugFrame, volTitle,
                cv::Point(titleX, vy1 + 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, volumeColor, 1, cv::LINE_AA);

    char depthStr[64];
    snprintf(depthStr, sizeof(depthStr), "Z: %.1fm - %.1fm (Standing @ 2m)",
             _playVolume->minZ / 1000.0f, _playVolume->maxZ / 1000.0f);
    cv::Size depthSize = cv::getTextSize(depthStr, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &base);
    int depthX = vx2 - depthSize.width - 10;
    cv::putText(debugFrame, depthStr,
                cv::Point(depthX, vy1 + 45),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, volumeColor, 1, cv::LINE_AA);

    // ═══════════════════════════════════════════════════════════
    // 1b. Face Detection Visualization (Phase 4 - Face-Anchored Tracking)
    // ═══════════════════════════════════════════════════════════
    if (_palmDetector && _palmDetector->isInitialized()) {
        const auto& faceRects = _palmDetector->getFaceRects();
        int faceFrameWidth = _palmDetector->getFaceFrameWidth();
        int faceFrameHeight = _palmDetector->getFaceFrameHeight();

        // Draw detected faces as circles at center
        for (const auto& faceRect : faceRects) {
            int faceCenterX = faceRect.x + faceRect.width / 2;
            int faceCenterY = faceRect.y + faceRect.height / 2;

            // Draw face center as blue circle (person anchor)
            cv::circle(debugFrame, cv::Point(faceCenterX, faceCenterY), 10, cv::Scalar(255, 0, 0), 2);
            cv::putText(debugFrame, "👤", cv::Point(faceCenterX - 8, faceCenterY + 8),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);

            // Draw face bounding box
            cv::rectangle(debugFrame, faceRect, cv::Scalar(200, 100, 255), 1, cv::LINE_AA);
        }

        // Draw lines from faces to nearest hands (if detected)
        // NOTE: Frame is already flipped horizontally BEFORE this function is called.
        // All coordinates here are in flipped space, so we must mirror normalized hand coords and face centers.
        if (_lastHandCount > 0 && !faceRects.empty()) {
            for (const auto& faceRect : faceRects) {
                // Mirror face center because frame is flipped (x' = width - x)
                int faceCenterX = debugFrame.cols - (faceRect.x + faceRect.width / 2);
                int faceCenterY = faceRect.y + faceRect.height / 2;

                // Find nearest hand to this face (mirror hand X as well)
                float minDist = std::numeric_limits<float>::max();
                int nearestHandIdx = -1;

                for (int h = 0; h < _lastHandCount && h < MAX_HANDS; ++h) {
                    int handPixelX = static_cast<int>((1.0f - _handStates[h].palmX) * debugFrame.cols);
                    int handPixelY = static_cast<int>(_handStates[h].palmY * debugFrame.rows);

                    float dist = std::sqrt(std::pow(handPixelX - faceCenterX, 2) +
                                          std::pow(handPixelY - faceCenterY, 2));

                    if (dist < minDist) {
                        minDist = dist;
                        nearestHandIdx = h;
                    }
                }

                // Draw line from face to nearest hand
                if (nearestHandIdx >= 0) {
                    int handPixelX = static_cast<int>((1.0f - _handStates[nearestHandIdx].palmX) * debugFrame.cols);
                    int handPixelY = static_cast<int>(_handStates[nearestHandIdx].palmY * debugFrame.rows);

                    cv::line(debugFrame, cv::Point(faceCenterX, faceCenterY),
                            cv::Point(handPixelX, handPixelY),
                            cv::Scalar(255, 100, 200), 1, cv::LINE_AA);  // Magenta line

                    // Distance label (at midpoint)
                    int midX = (faceCenterX + handPixelX) / 2;
                    int midY = (faceCenterY + handPixelY) / 2;
                    char distStr[32];
                    snprintf(distStr, sizeof(distStr), "%.0f px", minDist);

                    int baseline = 0;
                    cv::Size distTextSize = cv::getTextSize(distStr, cv::FONT_HERSHEY_SIMPLEX, 0.35, 1, &baseline);
                    cv::Mat distTextImg = cv::Mat::zeros(distTextSize.height + baseline, distTextSize.width, CV_8UC3);
                    cv::putText(distTextImg, distStr, cv::Point(0, distTextSize.height),
                               cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(255, 100, 200), 1);
                    // No flip needed here because hand/face are already mirrored above

                    int dtx1 = std::max(0, midX - distTextSize.width / 2);
                    int dtx2 = std::min(debugFrame.cols, dtx1 + distTextSize.width);
                    int dty1 = std::max(0, midY - 5 - distTextSize.height);
                    int dty2 = std::min(debugFrame.rows, dty1 + distTextSize.height);

                    if (dty2 > dty1 && dtx2 > dtx1) {
                        cv::Mat distDestROI = debugFrame(cv::Rect(dtx1, dty1, dtx2 - dtx1, dty2 - dty1));
                        cv::Mat distSrcROI = distTextImg(cv::Rect(0, 0, dtx2 - dtx1, dty2 - dty1));

                        for (int row = 0; row < distSrcROI.rows; ++row) {
                            for (int col = 0; col < distSrcROI.cols; ++col) {
                                cv::Vec3b pixel = distSrcROI.at<cv::Vec3b>(row, col);
                                if (pixel[0] > 0 || pixel[1] > 0 || pixel[2] > 0) {
                                    distDestROI.at<cv::Vec3b>(row, col) = pixel;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════
    // 2. Info Panel (Status Box)
    // ═══════════════════════════════════════════════════════════
    // Always show 2 hands (even if not detected) to prevent flickering
    // Box height: Base + per-hand; reduced size (~1/3 smaller) to avoid covering volume text
    int boxHeight = 90 + (MAX_HANDS * 70);
    int boxWidth = 220;
    cv::Mat overlay = debugFrame.clone();
    cv::rectangle(overlay, cv::Rect(5, 5, boxWidth, boxHeight), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::addWeighted(overlay, 0.6, debugFrame, 0.4, 0, debugFrame);

    int y = 18;
    int lineHeight = 14;

    // Date/Time
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time_t);
    char timeStr[64];
    std::strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", &tm);
    cv::putText(debugFrame, timeStr, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(255, 255, 255), 1);
    y += lineHeight;

    // FPS
    cv::Scalar fpsColor = (_currentFps >= 28) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    char fpsStr[32];
    snprintf(fpsStr, sizeof(fpsStr), "FPS: %.1f", static_cast<double>(_currentFps));
    cv::putText(debugFrame, fpsStr, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.45, fpsColor, 1);
    y += lineHeight;

    // TensorRT Status
    std::string trtStatus = _inferenceInitialized ? "TensorRT: Ready" : "TensorRT: Building...";
    cv::Scalar trtColor = _inferenceInitialized ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
    cv::putText(debugFrame, trtStatus, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.35, trtColor, 1);
    y += lineHeight;

    // Model Type (LITE or FULL)
    std::string modelType = getModelType();
    cv::Scalar modelColor = (modelType == "FULL") ? cv::Scalar(255, 165, 0) : cv::Scalar(0, 255, 0);  // Orange for FULL, Green for LITE
    std::string modelText = "Models: " + modelType;
    cv::putText(debugFrame, modelText, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.35, modelColor, 1);
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

        // Hand label with handedness (Phase 4: Face-Anchored Tracking)
        char labelStr[64];
        if (detected) {
            const char* handedness = state.isRightHand ? "RIGHT" : "LEFT";
            // Check if position matches handedness for validation
            bool positionValid = (state.isRightHand && state.palmX < 0.5f) ||
                                 (!state.isRightHand && state.palmX >= 0.5f);
            const char* validIcon = positionValid ? "✓" : "✗";
            snprintf(labelStr, sizeof(labelStr), "Hand %d: %s %s %s", h,
                     detected ? "ACTIVE" : "NOT DETECTED", handedness, validIcon);
        } else {
            snprintf(labelStr, sizeof(labelStr), "Hand %d: NOT DETECTED", h);
        }
        cv::putText(debugFrame, labelStr, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.35, labelColor, 1);
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
        cv::putText(debugFrame, posStr, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.32,
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
        cv::putText(debugFrame, velStr, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.32,
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
        cv::putText(debugFrame, deltaStr, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.32,
                    detected ? cv::Scalar(255, 200, 200) : cv::Scalar(60, 60, 60), 1);
        y += lineHeight - 4;

        // Gesture name (always show)
        char gestureStr[64];
        if (detected) {
            snprintf(gestureStr, sizeof(gestureStr), "  Gesture: %s", state.gesture.c_str());
        } else {
            snprintf(gestureStr, sizeof(gestureStr), "  Gesture: None");
        }
        cv::putText(debugFrame, gestureStr, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.35,
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

