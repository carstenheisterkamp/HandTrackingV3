#include "core/GestureFSM.hpp"
#include "core/Logger.hpp"
#include <cmath>

namespace core {

GestureFSM::GestureFSM() {
    reset();
}

void GestureFSM::reset() {
    state_ = GestureState::Idle;
    pendingState_ = GestureState::Idle;
    frameCount_ = 0;
    handLostFrames_ = 0;
}

const char* GestureFSM::getStateName(GestureState state) {
    switch (state) {
        case GestureState::Idle:  return "Idle";
        case GestureState::Palm:  return "Palm";
        case GestureState::Pinch: return "Pinch";
        case GestureState::Grab:  return "Grab";
        case GestureState::Point: return "Point";
        default: return "Unknown";
    }
}

float GestureFSM::getConfidence() const {
    if (state_ == pendingState_) {
        return 1.0f;  // Stable state
    }
    // Transitioning - return progress through debounce
    return static_cast<float>(frameCount_) / static_cast<float>(GESTURE_DEBOUNCE_FRAMES);
}

GestureState GestureFSM::update(const std::vector<TrackingResult::NormalizedPoint>& landmarks) {
    if (landmarks.size() < 21) {
        handleHandLost();
        return state_;
    }

    handLostFrames_ = 0;  // Reset hand lost counter

    // Detect current gesture from landmarks
    GestureState detected = detectGesture(landmarks);

    // Debounce logic
    if (detected == pendingState_) {
        frameCount_++;
    } else {
        pendingState_ = detected;
        frameCount_ = 1;
    }

    // Transition after N consistent frames
    if (frameCount_ >= GESTURE_DEBOUNCE_FRAMES && pendingState_ != state_) {
        transitionTo(pendingState_);
    }

    return state_;
}

void GestureFSM::handleHandLost() {
    handLostFrames_++;

    // Transition to Idle after 5 frames of no hand
    if (handLostFrames_ >= 5 && state_ != GestureState::Idle) {
        transitionTo(GestureState::Idle);
    }
}

void GestureFSM::transitionTo(GestureState newState) {
    GestureState oldState = state_;
    state_ = newState;
    pendingState_ = newState;
    frameCount_ = GESTURE_DEBOUNCE_FRAMES;  // Reset debounce

    Logger::info("GestureFSM: ", getStateName(oldState), " → ", getStateName(newState));

    if (transitionCallback_) {
        transitionCallback_(oldState, newState);
    }
}

GestureState GestureFSM::detectGesture(const std::vector<TrackingResult::NormalizedPoint>& landmarks) const {
    float handSize = getHandSize(landmarks);
    if (handSize < 0.001f) return GestureState::Idle;  // Invalid landmarks

    float pinchDist = getPinchDistance(landmarks);
    float normalizedPinch = pinchDist / handSize;

    // Apply hysteresis for Pinch detection
    if (state_ == GestureState::Pinch) {
        // Already pinching: use exit threshold (higher)
        if (normalizedPinch > PINCH_THRESHOLD_EXIT) {
            return detectOpenHand(landmarks);
        }
        return GestureState::Pinch;
    } else {
        // Not pinching: use enter threshold (lower)
        if (normalizedPinch < PINCH_THRESHOLD_ENTER) {
            return GestureState::Pinch;
        }
        return detectOpenHand(landmarks);
    }
}

GestureState GestureFSM::detectOpenHand(const std::vector<TrackingResult::NormalizedPoint>& landmarks) const {
    using LI = LandmarkIndices;

    // Check finger extension
    bool thumbExtended = isFingerExtended(landmarks, LI::THUMB_CMC, LI::THUMB_MCP, LI::THUMB_IP, LI::THUMB_TIP);
    bool indexExtended = isFingerExtended(landmarks, LI::INDEX_MCP, LI::INDEX_PIP, LI::INDEX_DIP, LI::INDEX_TIP);
    bool middleExtended = isFingerExtended(landmarks, LI::MIDDLE_MCP, LI::MIDDLE_PIP, LI::MIDDLE_DIP, LI::MIDDLE_TIP);
    bool ringExtended = isFingerExtended(landmarks, LI::RING_MCP, LI::RING_PIP, LI::RING_DIP, LI::RING_TIP);
    bool pinkyExtended = isFingerExtended(landmarks, LI::PINKY_MCP, LI::PINKY_PIP, LI::PINKY_DIP, LI::PINKY_TIP);

    // Grab: All fingers curled
    if (!indexExtended && !middleExtended && !ringExtended && !pinkyExtended) {
        return GestureState::Grab;
    }

    // Point: Only index extended
    if (indexExtended && !middleExtended && !ringExtended && !pinkyExtended) {
        return GestureState::Point;
    }

    // Default: Palm (hand open or partial)
    return GestureState::Palm;
}

bool GestureFSM::isFingerExtended(const std::vector<TrackingResult::NormalizedPoint>& landmarks,
                                   int mcp, int pip, int dip, int tip) const {
    // Finger is extended if tip is farther from wrist than PIP
    // This works better than angle-based for various hand orientations

    const auto& wrist = landmarks[LandmarkIndices::WRIST];
    const auto& tipPt = landmarks[tip];
    const auto& pipPt = landmarks[pip];

    float tipToWrist = distance(tipPt, wrist);
    float pipToWrist = distance(pipPt, wrist);

    // Tip should be farther from wrist than PIP for extended finger
    // Using 1.1x threshold for some margin
    return tipToWrist > pipToWrist * 1.1f;
}

float GestureFSM::distance(const TrackingResult::NormalizedPoint& a,
                           const TrackingResult::NormalizedPoint& b) const {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

float GestureFSM::getHandSize(const std::vector<TrackingResult::NormalizedPoint>& landmarks) const {
    // Hand size = distance from wrist to middle finger MCP
    // This is stable across different poses
    using LI = LandmarkIndices;
    return distance(landmarks[LI::WRIST], landmarks[LI::MIDDLE_MCP]);
}

float GestureFSM::getPinchDistance(const std::vector<TrackingResult::NormalizedPoint>& landmarks) const {
    // Pinch = distance between thumb tip and index tip
    using LI = LandmarkIndices;
    return distance(landmarks[LI::THUMB_TIP], landmarks[LI::INDEX_TIP]);
}

} // namespace core

