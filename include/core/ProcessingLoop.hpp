#pragma once

#include <atomic>
#include <thread>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
// Note: opencv2/cudastereo.hpp is missing in standard libopencv-dev.
// We disable OpenCV-CUDA features until opencv-contrib is installed.
// #include <opencv2/cudastereo.hpp>
// #include <opencv2/cudaarithm.hpp>
#endif

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

#ifdef ENABLE_CUDA
    // GPU Stereo Matching (Disabled due to missing opencv_cudastereo)
    // cv::Ptr<cv::cuda::StereoBM> _stereoBM;
    // cv::cuda::GpuMat _gmLeft, _gmRight, _gmDisp;
#endif

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
    float _currentFps = 0.0f;// System Performance (cached, updated every 5 seconds)
    std::string _performanceSummary;
    std::chrono::steady_clock::time_point _lastPerfUpdate;
};

} // namespace core

