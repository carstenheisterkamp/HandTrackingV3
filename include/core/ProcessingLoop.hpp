#pragma once

#include <atomic>
#include <thread>
#include <memory>
#include <vector>

#include "Types.hpp"
#include "Logger.hpp"
#include "math/Filters.hpp"

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

    // VIP Logic
    int _lockCounter = 0;
    bool _vipLocked = false;
    static constexpr int LOCK_THRESHOLD = 15;

    std::vector<math::KalmanFilter> _landmarkFilters;
};

} // namespace core

