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

    // Gesture Recognition
    // Heuristics based on OSC_GESTURE_REFERENCE.md

    float pinchDist = 0.0f;
    float fistDist = 0.0f;
    std::string gestureName = "unknown";
    int gestureId = 0;

    if (currentLandmarks.size() >= 21) {
        auto dist = [](const math::KalmanFilter::Point3f& a, const math::KalmanFilter::Point3f& b) {
            return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2) + std::pow(a.z - b.z, 2));
        };

        // Key Landmarks
        auto wrist = currentLandmarks[0];
        auto thumbTip = currentLandmarks[4];
        auto indexTip = currentLandmarks[8];
        auto middleTip = currentLandmarks[12];
        auto ringTip = currentLandmarks[16];
        auto pinkyTip = currentLandmarks[20];

        auto indexPip = currentLandmarks[6];
        auto middlePip = currentLandmarks[10];
        auto ringPip = currentLandmarks[14];
        auto pinkyPip = currentLandmarks[18];

        // Finger States (Extended vs Folded)
        // Heuristic: Tip is further from wrist than PIP joint = Extended
        bool thumbOpen = dist(thumbTip, wrist) > dist(currentLandmarks[2], wrist); // Thumb is special (CMC vs IP)
        bool indexOpen = dist(indexTip, wrist) > dist(indexPip, wrist);
        bool middleOpen = dist(middleTip, wrist) > dist(middlePip, wrist);
        bool ringOpen = dist(ringTip, wrist) > dist(ringPip, wrist);
        bool pinkyOpen = dist(pinkyTip, wrist) > dist(pinkyPip, wrist);

        // Pinch (Thumb Tip - Index Tip)
        pinchDist = dist(thumbTip, indexTip);

        // Fist (Average dist of tips to wrist)
        fistDist = (dist(indexTip, wrist) + dist(middleTip, wrist) + dist(ringTip, wrist) + dist(pinkyTip, wrist)) / 4.0f;

        // Thresholds (Normalized [0,1])
        const float PINCH_THRESH = 0.05f;

        // Gesture Logic
        if (!thumbOpen && !indexOpen && !middleOpen && !ringOpen && !pinkyOpen) {
            gestureId = 2; gestureName = "FIST";
        }
        else if (thumbOpen && indexOpen && middleOpen && ringOpen && pinkyOpen) {
            // Check for VULCAN (Gap between Middle and Ring)
            float midRingDist = dist(middleTip, ringTip);
            if (midRingDist > 0.08f) { // Heuristic gap
                 gestureId = 10; gestureName = "VULCAN";
            } else {
                 gestureId = 3; gestureName = "FIVE";
            }
        }
        else if (pinchDist < PINCH_THRESH) {
            // Check if others are open
            if (middleOpen && ringOpen && pinkyOpen) {
                gestureId = 1; gestureName = "OK"; // or PINCH
            } else {
                gestureId = 1; gestureName = "PINCH";
            }
        }
        else if (!thumbOpen && indexOpen && !middleOpen && !ringOpen && !pinkyOpen) {
            gestureId = 4; gestureName = "POINTING";
        }
        else if (thumbOpen && indexOpen && !middleOpen && !ringOpen && !pinkyOpen) {
            gestureId = 5; gestureName = "TWO"; // L-Shape
        }
        else if (!thumbOpen && indexOpen && middleOpen && !ringOpen && !pinkyOpen) {
            gestureId = 6; gestureName = "PEACE";
        }
        else if (thumbOpen && !indexOpen && !middleOpen && !ringOpen && pinkyOpen) {
            gestureId = 7; gestureName = "CALL_ME";
        }
        else if (!thumbOpen && indexOpen && !middleOpen && !ringOpen && pinkyOpen) {
            gestureId = 8; gestureName = "METAL";
        }
        else if (thumbOpen && indexOpen && !middleOpen && !ringOpen && pinkyOpen) {
            gestureId = 9; gestureName = "LOVE_YOU";
        }
        else if (thumbOpen && !indexOpen && !middleOpen && !ringOpen && !pinkyOpen) {
            gestureId = 11; gestureName = "THUMBS_UP"; // Orientation check needed for real thumbs up
        }
        else {
            gestureName = "unknown";
        }

        // Log Gesture occasionally
        static int gestureLogCounter = 0;
        if (++gestureLogCounter % 30 == 0) {
            Logger::info("Gesture: ", gestureName, " (Pinch: ", pinchDist, ")");
            Logger::info("Fingers: T:", thumbOpen, " I:", indexOpen, " M:", middleOpen, " R:", ringOpen, " P:", pinkyOpen);
        }
    }

    // Push to OSC Queue
    TrackingResult result;
    result.vipLocked = _vipLocked;
    result.timestamp = frame->timestamp;
    result.pinchDistance = pinchDist;
    result.gestureId = gestureId;
    result.gestureName = gestureName;

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

