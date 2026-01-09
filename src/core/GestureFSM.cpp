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
        case GestureState::Idle:         return "unknown";
        case GestureState::Five:         return "FIVE";
        case GestureState::Fist:         return "FIST";
        case GestureState::Pointing:     return "POINTING";
        case GestureState::Pinch:        return "PINCH";
        case GestureState::ThumbsUp:     return "THUMBS_UP";
        case GestureState::Two:          return "TWO";
        case GestureState::Three:        return "THREE";
        case GestureState::Four:         return "FOUR";
        case GestureState::Peace:        return "PEACE";
        case GestureState::Metal:        return "METAL";
        case GestureState::LoveYou:      return "LOVE_YOU";
        case GestureState::CallMe:       return "CALL_ME";
        case GestureState::Vulcan:       return "VULCAN";
        case GestureState::MiddleFinger: return "MIDDLE_FINGER";
        default: return "unknown";
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

    // Check finger extension for all 5 fingers
    bool thumbExtended = isThumbExtended(landmarks);
    bool indexExtended = isFingerExtended(landmarks, LI::INDEX_MCP, LI::INDEX_PIP, LI::INDEX_DIP, LI::INDEX_TIP);
    bool middleExtended = isFingerExtended(landmarks, LI::MIDDLE_MCP, LI::MIDDLE_PIP, LI::MIDDLE_DIP, LI::MIDDLE_TIP);
    bool ringExtended = isFingerExtended(landmarks, LI::RING_MCP, LI::RING_PIP, LI::RING_DIP, LI::RING_TIP);
    bool pinkyExtended = isFingerExtended(landmarks, LI::PINKY_MCP, LI::PINKY_PIP, LI::PINKY_DIP, LI::PINKY_TIP);

    // Count extended fingers
    int fingerCount = (thumbExtended ? 1 : 0) + (indexExtended ? 1 : 0) +
                      (middleExtended ? 1 : 0) + (ringExtended ? 1 : 0) + (pinkyExtended ? 1 : 0);

    // Debug log (every 60 frames)
    static int debugCounter = 0;
    if (++debugCounter % 60 == 1) {
        Logger::info("🖐 Gesture Debug: thumb=", thumbExtended, " idx=", indexExtended,
                     " mid=", middleExtended, " ring=", ringExtended, " pinky=", pinkyExtended,
                     " count=", fingerCount);
    }

    // ═══════════════════════════════════════════════════════════
    // Gesture Detection (ordered by specificity - most specific first)
    // ═══════════════════════════════════════════════════════════

    // FIST: All fingers curled ✊
    if (!thumbExtended && !indexExtended && !middleExtended && !ringExtended && !pinkyExtended) {
        return GestureState::Fist;
    }

    // THUMBS_UP: Only thumb extended 👍
    if (thumbExtended && !indexExtended && !middleExtended && !ringExtended && !pinkyExtended) {
        return GestureState::ThumbsUp;
    }

    // POINTING: Only index extended ☝️
    if (!thumbExtended && indexExtended && !middleExtended && !ringExtended && !pinkyExtended) {
        return GestureState::Pointing;
    }

    // MIDDLE_FINGER: Only middle extended 🖕
    if (!thumbExtended && !indexExtended && middleExtended && !ringExtended && !pinkyExtended) {
        return GestureState::MiddleFinger;
    }

    // PEACE: Index + Middle extended ✌️
    if (!thumbExtended && indexExtended && middleExtended && !ringExtended && !pinkyExtended) {
        return GestureState::Peace;
    }

    // METAL: Index + Pinky extended 🤘
    if (!thumbExtended && indexExtended && !middleExtended && !ringExtended && pinkyExtended) {
        return GestureState::Metal;
    }

    // CALL_ME: Thumb + Pinky extended 🤙
    if (thumbExtended && !indexExtended && !middleExtended && !ringExtended && pinkyExtended) {
        return GestureState::CallMe;
    }

    // TWO: Thumb + Index extended
    if (thumbExtended && indexExtended && !middleExtended && !ringExtended && !pinkyExtended) {
        return GestureState::Two;
    }

    // THREE: Thumb + Index + Middle extended
    if (thumbExtended && indexExtended && middleExtended && !ringExtended && !pinkyExtended) {
        return GestureState::Three;
    }

    // LOVE_YOU: Thumb + Index + Pinky extended 🤟
    if (thumbExtended && indexExtended && !middleExtended && !ringExtended && pinkyExtended) {
        return GestureState::LoveYou;
    }

    // FOUR: All except thumb
    if (!thumbExtended && indexExtended && middleExtended && ringExtended && pinkyExtended) {
        return GestureState::Four;
    }

    // FIVE: All 5 fingers extended 🖐️
    if (thumbExtended && indexExtended && middleExtended && ringExtended && pinkyExtended) {
        // Check for VULCAN (spread between middle and ring)
        if (isVulcanSpread(landmarks)) {
            return GestureState::Vulcan;
        }
        return GestureState::Five;
    }

    // Default: FIVE for any partial open hand
    return GestureState::Five;
}

bool GestureFSM::isFingerExtended(const std::vector<TrackingResult::NormalizedPoint>& landmarks,
                                   int mcp, int pip, int dip, int tip) const {
    // Use 2D distance only (Z is often 0 or unreliable without stereo depth)
    //
    // A finger is EXTENDED if:
    // 1. The tip is significantly farther from wrist than PIP (stretched out)
    // 2. The finger is relatively straight (angle check via distances)
    //
    // A finger is CURLED if:
    // - Tip is close to or behind PIP relative to wrist
    // - The DIP->TIP segment folds back towards palm

    const auto& wrist = landmarks[LandmarkIndices::WRIST];
    const auto& tipPt = landmarks[tip];
    const auto& pipPt = landmarks[pip];
    const auto& dipPt = landmarks[dip];
    const auto& mcpPt = landmarks[mcp];

    float tipToWrist = distance2D(tipPt, wrist);
    float pipToWrist = distance2D(pipPt, wrist);

    // Primary check: Tip must be farther from wrist than PIP
    // For a curled finger, tip folds back and gets closer to wrist than PIP
    bool tipFartherThanPip = tipToWrist > pipToWrist * 1.15f;

    // Secondary check: Finger length ratio
    // Extended finger: tip->mcp distance is significant
    // Curled finger: tip comes back close to mcp
    float tipToMcp = distance2D(tipPt, mcpPt);
    float dipToMcp = distance2D(dipPt, mcpPt);

    // For extended finger, tip should be significantly farther from MCP than DIP
    bool straightFinger = tipToMcp > dipToMcp * 1.2f;

    // Both conditions must be met for extended finger
    // This is stricter than before (was OR, now AND)
    return tipFartherThanPip && straightFinger;
}

float GestureFSM::distance(const TrackingResult::NormalizedPoint& a,
                           const TrackingResult::NormalizedPoint& b) const {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

float GestureFSM::distance2D(const TrackingResult::NormalizedPoint& a,
                             const TrackingResult::NormalizedPoint& b) const {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return std::sqrt(dx*dx + dy*dy);
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

bool GestureFSM::isThumbExtended(const std::vector<TrackingResult::NormalizedPoint>& landmarks) const {
    // Thumb is special - check if tip is farther from palm center than IP joint
    // Also check if thumb is spread away from the hand
    using LI = LandmarkIndices;

    // Palm center approximation (between wrist and middle MCP)
    TrackingResult::NormalizedPoint palmCenter;
    palmCenter.x = (landmarks[LI::WRIST].x + landmarks[LI::MIDDLE_MCP].x) / 2.0f;
    palmCenter.y = (landmarks[LI::WRIST].y + landmarks[LI::MIDDLE_MCP].y) / 2.0f;
    palmCenter.z = 0.0f;

    float tipToPalm = distance2D(landmarks[LI::THUMB_TIP], palmCenter);
    float ipToPalm = distance2D(landmarks[LI::THUMB_IP], palmCenter);
    float mcpToPalm = distance2D(landmarks[LI::THUMB_CMC], palmCenter);

    // Thumb extended if tip is significantly farther from palm than IP
    // AND the thumb is spread outward (tip farther than MCP from palm)
    bool extended = tipToPalm > ipToPalm * 1.2f;
    bool spreadOut = tipToPalm > mcpToPalm * 1.3f;

    return extended && spreadOut;
}

bool GestureFSM::isVulcanSpread(const std::vector<TrackingResult::NormalizedPoint>& landmarks) const {
    // Vulcan: Check if there's a significant spread between middle and ring fingers
    // while index-middle and ring-pinky are close together
    using LI = LandmarkIndices;

    float indexMiddleDist = distance2D(landmarks[LI::INDEX_TIP], landmarks[LI::MIDDLE_TIP]);
    float middleRingDist = distance2D(landmarks[LI::MIDDLE_TIP], landmarks[LI::RING_TIP]);
    float ringPinkyDist = distance2D(landmarks[LI::RING_TIP], landmarks[LI::PINKY_TIP]);

    // Vulcan if middle-ring gap is significantly larger than others
    float avgOtherGap = (indexMiddleDist + ringPinkyDist) / 2.0f;
    return middleRingDist > avgOtherGap * 1.8f;
}

} // namespace core

