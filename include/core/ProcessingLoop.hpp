#pragma once

#include <atomic>
#include <thread>
#include <mutex>
#include <memory>
#include <vector>
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

private:
    void loop();
    void processFrame(Frame* frame);
    void drawDebugOverlay(cv::Mat& debugFrame, Frame* frame);

    std::shared_ptr<AppProcessingQueue> _inputQueue;
    std::shared_ptr<AppFramePool> _framePool;
    std::shared_ptr<OscQueue> _oscQueue;

    std::atomic<bool> _running;
    std::thread _thread;

    // V3 Components (Phase 2+)
    std::unique_ptr<HandTracker> _handTracker;
    std::unique_ptr<GestureFSM> _gestureFSM;
    std::unique_ptr<StereoDepth> _stereoDepth;

    // V3 Inference (TensorRT)
    std::unique_ptr<inference::PalmDetector> _palmDetector;
    std::unique_ptr<inference::HandLandmark> _handLandmark;
    bool _inferenceInitialized = false;
    bool _inferenceAttempted = false;
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

    // Hand Tracking State (for stats display)
    int _lastHandCount = 0;
    float _lastPalmX = 0.0f, _lastPalmY = 0.0f, _lastPalmZ = 0.0f;
    float _lastVelX = 0.0f, _lastVelY = 0.0f;
    std::string _lastGesture = "None";
    bool _lastVipLocked = false;

    // System Performance (cached)
    std::string _performanceSummary;
    std::chrono::steady_clock::time_point _lastPerfUpdate;
};

} // namespace core

