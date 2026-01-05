#include "core/ProcessingLoop.hpp"
#include <chrono>

namespace core {

ProcessingLoop::ProcessingLoop(std::shared_ptr<AppProcessingQueue> inputQueue,
                               std::shared_ptr<AppFramePool> framePool,
                               std::shared_ptr<OscQueue> oscQueue)
    : _inputQueue(std::move(inputQueue)),
      _framePool(std::move(framePool)),
      _oscQueue(std::move(oscQueue)),
      _running(false) {

    // Initialize 21 Kalman filters for hand landmarks
    _landmarkFilters.resize(21);
}

ProcessingLoop::~ProcessingLoop() {
    stop();
}

void ProcessingLoop::start() {
    if (_running) return;
    _running = true;
    _thread = std::thread(&ProcessingLoop::loop, this);
    Logger::info("ProcessingLoop started.");
}

void ProcessingLoop::stop() {
    if (!_running) return;
    _running = false;
    if (_thread.joinable()) {
        _thread.join();
    }
    Logger::info("ProcessingLoop stopped.");
}

bool ProcessingLoop::isRunning() const {
    return _running;
}

void ProcessingLoop::loop() {
    while (_running) {
        Frame* frame = nullptr;
        if (_inputQueue->pop_front(frame)) {
            if (frame) {
                processFrame(frame);
                // Return frame to pool after processing
                _framePool->release(frame);
            }
        } else {
            // Busy wait or sleep?
            // For low latency, busy wait might be preferred, but let's be nice to CPU for now if queue is empty
            // Or use a condition variable if we change SpscQueue to support it, but SpscQueue is lock-free.
            // A short sleep is acceptable if we are faster than camera.
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
}

void ProcessingLoop::processFrame(Frame* frame) {
    // TODO: Implement tracking logic here
    // Access frame->data for pixel data (NV12)
    // Access frame->width, frame->height, etc.

    if (frame->nnData.empty()) {
        // VIP Logic: Reset lock if no hand detected
        if (_lockCounter > 0) {
            _lockCounter--;
        }
        if (_lockCounter == 0) {
            _vipLocked = false;
            // Reset filters when tracking is lost
            for(auto& kf : _landmarkFilters) {
                kf.reset();
            }
            _hasLastState = false;
        }
        return;
    }

    // Parse landmarks (21 points * 3 coords = 63 floats)
    if (frame->nnData.size() < 63) {
        Logger::warn("ProcessingLoop: Invalid NN data size: ", frame->nnData.size());
        return;
    }

    // VIP Logic: Increment lock counter
    if (_lockCounter < LOCK_THRESHOLD) {
        _lockCounter++;
    } else {
        _vipLocked = true;
    }

    std::vector<math::KalmanFilter::Point3f> currentLandmarks;
    currentLandmarks.reserve(21);

    // Predict step for all filters
    for(auto& kf : _landmarkFilters) {
        kf.predict();
    }

    for (size_t i = 0; i < 21; ++i) {
        float rawX = frame->nnData[i * 3 + 0];
        float rawY = frame->nnData[i * 3 + 1];
        float rawZ = frame->nnData[i * 3 + 2];

        // Apply Kalman Filter
        auto smoothed = _landmarkFilters[i].update(rawX, rawY, rawZ);

        currentLandmarks.push_back(smoothed);
    }

    // Calculate velocity if we have history
    if (_hasLastState) {
        auto dt = std::chrono::duration<float>(frame->timestamp - _lastHandState.timestamp).count();
        if (dt > 0.0001f) { // Avoid division by zero
            // Calculate velocity of wrist (index 0)
            float vx = (currentLandmarks[0].x - _lastHandState.landmarks[0].x) / dt;
            float vy = (currentLandmarks[0].y - _lastHandState.landmarks[0].y) / dt;
            float vz = (currentLandmarks[0].z - _lastHandState.landmarks[0].z) / dt;

            // Log velocity occasionally
            static int logCounter = 0;
            if (++logCounter % 30 == 0) {
                Logger::info("Wrist Velocity: [", vx, ", ", vy, ", ", vz, "]");
            }
        }
    }

    // Update state
    _lastHandState.landmarks = currentLandmarks;
    _lastHandState.timestamp = frame->timestamp;
    _hasLastState = true;

    // Push to OSC Queue
    TrackingResult result;
    result.vipLocked = _vipLocked;
    result.timestamp = frame->timestamp;

    // Copy landmarks to result array
    for (size_t i = 0; i < 21; ++i) {
        result.landmarks[i * 3 + 0] = currentLandmarks[i].x;
        result.landmarks[i * 3 + 1] = currentLandmarks[i].y;
        result.landmarks[i * 3 + 2] = currentLandmarks[i].z;
    }

    if (!_oscQueue->try_push(result)) {
        // Queue full -> Drop newest (Backpressure)
        // Logger::warn("ProcessingLoop: OSC Queue full, dropping result.");
    }
}

} // namespace core

