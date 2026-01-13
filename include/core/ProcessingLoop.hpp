#pragma once

#include <atomic>
#include <thread>
#include <mutex>
#include <memory>
#include <vector>
#include <array>
#include <opencv2/opencv.hpp>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include "Types.hpp"
#include "Logger.hpp"
#include "net/MjpegServer.hpp"
#include "core/MemoryUtils.hpp"

// Forward declarations for V3 components
namespace core {
    class HandTracker;
    class GestureFSM;
    class StereoDepth;
    class SessionFSM;
    struct PlayVolume;
}

namespace inference {
    class PalmDetector;
    class HandLandmark;
}

namespace core {

/**
 * V3 ProcessingLoop - Simplified for Sensor-Only Pipeline
 *
 * Receives RGB frames from OAK-D and:
 * - Phase 1: Debug preview only
 * - Phase 2: TensorRT inference (Palm + Landmarks)
 * - Phase 3: Stereo depth + Kalman + Gesture FSM
 */
class ProcessingLoop {
public:
    ProcessingLoop(std::shared_ptr<AppProcessingQueue> inputQueue,
                   std::shared_ptr<AppFramePool> framePool,
                   std::shared_ptr<OscQueue> oscQueue);
    ~ProcessingLoop();

    void start();
    void stop();
    bool isRunning() const;

    /**
     * Set model paths before starting (for model selection)
     * Call this before start() to use custom models
     */
    void setModelPaths(const std::string& palmModel, const std::string& landmarkModel) {
        _palmModelPath = palmModel;
        _landmarkModelPath = landmarkModel;
    }

    /**
     * Get current model type for display (LITE or FULL)
     */
    std::string getModelType() const {
        if (_palmModelPath.find("_full") != std::string::npos) {
            return "FULL";
        }
        return "LITE";
    }

    /**
     * Set ROI configuration (for coordinate normalization)
     * @param useROI: Whether ROI cropping is enabled
     * @param roiSize: Size of ROI quadrat (e.g., 1080 for 1080×1080)
     */
    void setROIConfig(bool useROI, int roiSize) {
        _useROI = useROI;
        _roiSize = roiSize;
        if (useROI) {
            // Calculate ROI offsets from full 1920×1080 frame
            _roiOffsetX = (1920 - roiSize) / 2;  // Center horizontally
            _roiOffsetY = (1080 - roiSize) / 2;  // Center vertically
        } else {
            _roiOffsetX = 0;
            _roiOffsetY = 0;
        }
    }

private:
    void loop();
    void processFrame(Frame* frame);
    void drawDebugOverlay(cv::Mat& debugFrame, Frame* frame);

    std::shared_ptr<AppProcessingQueue> _inputQueue;
    std::shared_ptr<AppFramePool> _framePool;
    std::shared_ptr<OscQueue> _oscQueue;

    std::atomic<bool> _running;
    std::thread _thread;

    // V3 Components (Phase 2+) - Support for 2 hands
    static constexpr int MAX_HANDS = 2;
    std::array<std::unique_ptr<HandTracker>, MAX_HANDS> _handTrackers;
    std::array<std::unique_ptr<GestureFSM>, MAX_HANDS> _gestureFSMs;
    std::array<std::unique_ptr<SessionFSM>, MAX_HANDS> _sessionFSMs;  // Phase 4
    std::unique_ptr<StereoDepth> _stereoDepth;

    // Phase 4: Play Volume for filtering
    std::unique_ptr<PlayVolume> _playVolume;

    // V3 Inference (TensorRT)
    std::unique_ptr<inference::PalmDetector> _palmDetector;
    std::unique_ptr<inference::HandLandmark> _handLandmark;
    bool _inferenceInitialized = false;
    bool _inferenceAttempted = false;
    bool _stereoInitialized = false;  // Phase 3: Stereo Depth
    std::thread _trtInitThread;
    std::mutex _trtMutex;

    // Debug Preview
    std::unique_ptr<net::MjpegServer> _mjpegServer;

    // BGR Buffer for MJPEG
    std::unique_ptr<uint8_t, AlignedDeleter> _bgrBuffer;
    size_t _bgrBufferSize = 0;
    size_t _bgrWidth = 0;
    size_t _bgrHeight = 0;

    // FPS Counting
    std::chrono::steady_clock::time_point _lastFpsTime;
    int _frameCount = 0;
    float _currentFps = 0.0f;

    // Hand Tracking State (for stats display) - 2 hands
    int _lastHandCount = 0;
    struct HandState {
        float palmX = 0.0f, palmY = 0.0f, palmZ = 0.0f;
        float velX = 0.0f, velY = 0.0f, velZ = 0.0f;
        float deltaX = 0.0f, deltaY = 0.0f, deltaZ = 0.0f;  // Acceleration/Delta
        std::string gesture = "None";
        bool vipLocked = false;
        bool isRightHand = false;  // Phase 4: Handedness for visualization

        // Previous velocity for delta calculation
        float prevVelX = 0.0f, prevVelY = 0.0f, prevVelZ = 0.0f;
    };
    std::array<HandState, MAX_HANDS> _handStates;

    // System Performance (cached)
    std::string _performanceSummary;
    std::chrono::steady_clock::time_point _lastPerfUpdate;

    // Model Paths (configurable for lite vs full models)
    // TensorRT auto-compiles .onnx → .engine (cached in build/models/)
    // CMake preserves .engine files across builds for fast startup
    // OPTIMIZED FOR 2m DISTANCE: FULL model + 1024x576 resolution
    // Trade-off: Accuracy > Speed (FULL is more robust for small hands @ 2m)
    std::string _palmModelPath = "models/palm_detection_full.onnx";
    std::string _landmarkModelPath = "models/hand_landmark_full.onnx";

    // ROI Configuration
    bool _useROI = false;
    int _roiSize = 1080;
    int _roiOffsetX = 0;
    int _roiOffsetY = 0;
};

} // namespace core

