#pragma once

#include <atomic>
#include <thread>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>

#include "Types.hpp"
#include "Logger.hpp"
#include "math/Filters.hpp"
#include "net/MjpegServer.hpp"
#include "core/MemoryUtils.hpp"

namespace core {

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
    void drawDebugOverlay(cv::Mat& debugFrame, Frame* frame,
                          const std::vector<math::KalmanFilter::Point3f>& currentLandmarks,
                          const std::string& gestureName, int gestureId, float pinchDist);

    struct HandState {
        std::vector<math::KalmanFilter::Point3f> landmarks;
        std::chrono::time_point<std::chrono::steady_clock> timestamp;
    };

    std::shared_ptr<AppProcessingQueue> _inputQueue;
    std::shared_ptr<AppFramePool> _framePool;
    std::shared_ptr<OscQueue> _oscQueue;

    std::atomic<bool> _running;
    std::thread _thread;

    // Tracking state
    HandState _lastHandState;
    bool _hasLastState = false;

    // Last velocity for display
    struct Velocity { float x = 0, y = 0, z = 0; };
    Velocity _lastVelocity;

    // VIP Logic
    int _lockCounter = 0;
    bool _vipLocked = false;
    static constexpr int LOCK_THRESHOLD = 15;

    std::vector<math::KalmanFilter> _landmarkFilters;

    // Debug Preview
    std::unique_ptr<net::MjpegServer> _mjpegServer;

    // BGR Buffer for MJPEG (Aligned/Pinned for CUDA)
    std::unique_ptr<uint8_t, AlignedDeleter> _bgrBuffer;
    size_t _bgrBufferSize = 0;
    size_t _bgrWidth = 0;
    size_t _bgrHeight = 0;

    // FPS Counting
    std::chrono::steady_clock::time_point _lastFpsTime;
    int _frameCount = 0;
    float _currentFps = 0.0f;
};

} // namespace core

