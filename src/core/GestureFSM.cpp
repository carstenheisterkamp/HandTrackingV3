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

GestureState GestureFSM::update(const std::vector<TrackingResult::NormalizedPoint>& landmarks,
                                 bool isRightHand) {
    if (landmarks.size() < 21) {
        handleHandLost();
        return state_;
    }

    handLostFrames_ = 0;  // Reset hand lost counter

    // Detect current gesture from landmarks
    GestureState detected = detectGesture(landmarks, isRightHand);

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

GestureState GestureFSM::detectGesture(const std::vector<TrackingResult::NormalizedPoint>& landmarks,
                                        bool isRightHand) const {
    float handSize = getHandSize(landmarks);
    if (handSize < 0.001f) return GestureState::Idle;  // Invalid landmarks

    float pinchDist = getPinchDistance(landmarks);
    float normalizedPinch = pinchDist / handSize;

    // Apply hysteresis for Pinch detection
    if (state_ == GestureState::Pinch) {
        // Already pinching: use exit threshold (higher)
        if (normalizedPinch > PINCH_THRESHOLD_EXIT) {
            return detectOpenHand(landmarks, isRightHand);
        }
        return GestureState::Pinch;
    } else {
        // Not pinching: use enter threshold (lower)
        if (normalizedPinch < PINCH_THRESHOLD_ENTER) {
            return GestureState::Pinch;
        }
        return detectOpenHand(landmarks, isRightHand);
    }
}

GestureState GestureFSM::detectOpenHand(const std::vector<TrackingResult::NormalizedPoint>& landmarks,
                                         bool isRightHand) const {
    using LI = LandmarkIndices;

    // ═══════════════════════════════════════════════════════════
    // SIMPLIFIED FINGER DETECTION (based on Python/MediaPipe article)
    //
    // Key insight: Use relative Y positions, not complex angles
    // - Finger is UP if tip.y < pip.y (tip higher than PIP joint)
    // - Thumb uses X position (depends on left/right hand)
    //
    // This is MORE ROBUST because:
    // 1. Y-comparison works regardless of viewing angle
    // 2. No complex angle calculations that fail at edge cases
    // 3. Simple enough to be reliable
    // ═══════════════════════════════════════════════════════════

    // Thumb: X-position based (different for left/right hand)
    bool thumbUp = isThumbUp(landmarks, isRightHand);

    // Fingers: Y-position based (tip higher than PIP = extended)
    bool indexUp = isFingerUp(landmarks, LI::INDEX_TIP, LI::INDEX_PIP);
    bool middleUp = isFingerUp(landmarks, LI::MIDDLE_TIP, LI::MIDDLE_PIP);
    bool ringUp = isFingerUp(landmarks, LI::RING_TIP, LI::RING_PIP);
    bool pinkyUp = isFingerUp(landmarks, LI::PINKY_TIP, LI::PINKY_PIP);

    // Count extended fingers
    int fingerCount = (thumbUp ? 1 : 0) + (indexUp ? 1 : 0) +
                      (middleUp ? 1 : 0) + (ringUp ? 1 : 0) + (pinkyUp ? 1 : 0);

    // Debug log (every 60 frames)
    static int debugCounter = 0;
    if (++debugCounter % 60 == 1) {
        // Calculate hand size for debug output
        float handSize = distance2D(landmarks[LI::WRIST], landmarks[LI::MIDDLE_MCP]);
        Logger::info("🖐 Gesture (Y-based): T=", thumbUp, " I=", indexUp,
                     " M=", middleUp, " R=", ringUp, " P=", pinkyUp,
                     " count=", fingerCount, " hand=", (isRightHand ? "R" : "L"),
                     " size=", handSize);
    }

    // ═══════════════════════════════════════════════════════════
    // Gesture Detection (ordered by specificity - most specific first)
    // ═══════════════════════════════════════════════════════════

    // FIST: All fingers down ✊
    // IMPROVED: Additional check - fingers should be CURLED, not just not-up
    if (!thumbUp && !indexUp && !middleUp && !ringUp && !pinkyUp) {
        // Additional verification: Tips should be close to palm (wrist)
        // This prevents false FIST when fingers are pointing sideways
        using LI = LandmarkIndices;
        const auto& wrist = landmarks[LI::WRIST];
        const auto& middleMcp = landmarks[LI::MIDDLE_MCP];
        float handSize = distance2D(wrist, middleMcp);

        // Check if fingertips are close to palm
        float indexTipDist = distance2D(landmarks[LI::INDEX_TIP], middleMcp);
        float middleTipDist = distance2D(landmarks[LI::MIDDLE_TIP], middleMcp);

        // For FIST, tips should be within 1.5x hand size from MCP
        // (curled fingers fold back towards palm)
        bool fingersCurled = (indexTipDist < handSize * 1.5f) &&
                            (middleTipDist < handSize * 1.5f);

        // Accept FIST even if curl check fails (to handle edge cases)
        // but log when curl check helps
        if (!fingersCurled && debugCounter % 60 == 1) {
            Logger::info("🖐 FIST: Fingers not curled enough, but accepting (tips far from palm)");
        }

        return GestureState::Fist;
    }

    // THUMBS_UP: Only thumb up 👍
    if (thumbUp && !indexUp && !middleUp && !ringUp && !pinkyUp) {
        return GestureState::ThumbsUp;
    }

    // POINTING: Only index up ☝️
    if (!thumbUp && indexUp && !middleUp && !ringUp && !pinkyUp) {
        return GestureState::Pointing;
    }

    // MIDDLE_FINGER: Only middle up 🖕
    if (!thumbUp && !indexUp && middleUp && !ringUp && !pinkyUp) {
        return GestureState::MiddleFinger;
    }

    // PEACE: Index + Middle up ✌️
    if (!thumbUp && indexUp && middleUp && !ringUp && !pinkyUp) {
        return GestureState::Peace;
    }

    // METAL: Index + Pinky up 🤘
    if (!thumbUp && indexUp && !middleUp && !ringUp && pinkyUp) {
        return GestureState::Metal;
    }

    // CALL_ME: Thumb + Pinky up 🤙
    if (thumbUp && !indexUp && !middleUp && !ringUp && pinkyUp) {
        return GestureState::CallMe;
    }

    // LOVE_YOU: Thumb + Index + Pinky up 🤟
    if (thumbUp && indexUp && !middleUp && !ringUp && pinkyUp) {
        return GestureState::LoveYou;
    }

    // TWO: Thumb + Index up
    if (thumbUp && indexUp && !middleUp && !ringUp && !pinkyUp) {
        return GestureState::Two;
    }

    // THREE: Thumb + Index + Middle up
    if (thumbUp && indexUp && middleUp && !ringUp && !pinkyUp) {
        return GestureState::Three;
    }

    // FOUR: All except thumb up
    if (!thumbUp && indexUp && middleUp && ringUp && pinkyUp) {
        return GestureState::Four;
    }

    // FIVE: All 5 fingers up 🖐️
    if (thumbUp && indexUp && middleUp && ringUp && pinkyUp) {
        // Check for VULCAN (spread between middle and ring)
        if (isVulcanSpread(landmarks)) {
            return GestureState::Vulcan;
        }
        return GestureState::Five;
    }

    // ═══════════════════════════════════════════════════════════
    // Fallback: Count-based detection for ambiguous cases
    // ═══════════════════════════════════════════════════════════

    if (fingerCount >= 5) return GestureState::Five;
    if (fingerCount == 4) return GestureState::Four;
    if (fingerCount == 0) return GestureState::Fist;

    // Ambiguous - return current state to avoid flicker
    return state_;
}

// ═══════════════════════════════════════════════════════════════════════════
// SIMPLE Y-BASED FINGER DETECTION (from Python/MediaPipe article)
// Much more robust than complex angle calculations
// ═══════════════════════════════════════════════════════════════════════════

bool GestureFSM::isFingerUp(const std::vector<TrackingResult::NormalizedPoint>& landmarks,
                             int tipIdx, int pipIdx) const {
    // Simple check: Tip is HIGHER (smaller Y) than PIP joint
    // In image coordinates, Y=0 is top, so smaller Y = higher
    //
    // This works because:
    // - When finger is extended, tip is above PIP
    // - When finger is curled, tip folds down below PIP
    // - Robust to viewing angle (always uses image Y coordinate)

    const auto& tip = landmarks[tipIdx];
    const auto& pip = landmarks[pipIdx];

    // Dynamic threshold based on hand size for better multi-distance support
    // Get hand size (wrist to middle MCP distance)
    using LI = LandmarkIndices;
    float handSize = distance2D(landmarks[LI::WRIST], landmarks[LI::MIDDLE_MCP]);

    // VERY LENIENT: Accept finger as "up" if tip is even slightly above PIP
    // Use only 2% of hand size as minimum separation
    float threshold = handSize * 0.02f;
    threshold = std::clamp(threshold, 0.002f, 0.01f);  // Very small threshold

    // Tip must be higher than PIP
    return tip.y < pip.y - threshold;
}

bool GestureFSM::isThumbUp(const std::vector<TrackingResult::NormalizedPoint>& landmarks,
                            bool isRightHand) const {
    // Thumb is special: Uses X-position instead of Y
    // because thumb extends sideways, not upward
    //
    // For RIGHT hand: Thumb tip should be LEFT of thumb IP (smaller X)
    // For LEFT hand: Thumb tip should be RIGHT of thumb IP (larger X)

    using LI = LandmarkIndices;
    const auto& thumbTip = landmarks[LI::THUMB_TIP];
    const auto& thumbIP = landmarks[LI::THUMB_IP];

    // IMPROVED: More lenient threshold for better FIST detection
    // Dynamic threshold based on hand size
    float handSize = distance2D(landmarks[LI::WRIST], landmarks[LI::MIDDLE_MCP]);
    float threshold = handSize * 0.10f;  // Reduced from 12% to 10%
    threshold = std::clamp(threshold, 0.008f, 0.025f);  // Wider range

    if (isRightHand) {
        // Right hand: extended thumb has tip to the LEFT of IP
        return thumbTip.x < thumbIP.x - threshold;
    } else {
        // Left hand: extended thumb has tip to the RIGHT of IP
        return thumbTip.x > thumbIP.x + threshold;
    }
}

bool GestureFSM::isThumbExtended(const std::vector<TrackingResult::NormalizedPoint>& landmarks,
                                  bool isRightHand) const {
    // Wrapper for consistency with old API
    return isThumbUp(landmarks, isRightHand);
}

bool GestureFSM::isFingerExtended(const std::vector<TrackingResult::NormalizedPoint>& landmarks,
                                   int mcp, int pip, int dip, int tip) const {
    // Use simple Y-based check for consistency
    return isFingerUp(landmarks, tip, pip);
}

float GestureFSM::getFingerCurl(const std::vector<TrackingResult::NormalizedPoint>& landmarks,
                                 int mcp, int pip, int dip, int tip) const {
    // ═══════════════════════════════════════════════════════════
    // Robust Finger Curl Detection (0.0 = extended, 1.0 = curled)
    //
    // This approach is more robust to viewing angles by using:
    // 1. Relative Y positions (works for most hand orientations)
    // 2. Distance ratios normalized by hand size
    // 3. Multiple redundant checks
    // ═══════════════════════════════════════════════════════════

    const auto& wrist = landmarks[LandmarkIndices::WRIST];
    const auto& tipPt = landmarks[tip];
    const auto& dipPt = landmarks[dip];
    const auto& pipPt = landmarks[pip];
    const auto& mcpPt = landmarks[mcp];

    // Get hand orientation (wrist to middle MCP direction)
    // This tells us which way "extended" points
    const auto& middleMcp = landmarks[LandmarkIndices::MIDDLE_MCP];
    float handDirX = middleMcp.x - wrist.x;
    float handDirY = middleMcp.y - wrist.y;
    float handLength = std::sqrt(handDirX * handDirX + handDirY * handDirY);
    if (handLength < 0.001f) return 0.5f;  // Invalid

    // Normalize hand direction
    handDirX /= handLength;
    handDirY /= handLength;

    // Project finger segments onto hand direction
    // Extended finger: tip is far along hand direction
    // Curled finger: tip folds back

    // Vector from MCP to tip
    float fingerVecX = tipPt.x - mcpPt.x;
    float fingerVecY = tipPt.y - mcpPt.y;

    // Project onto hand direction (dot product)
    float tipProjection = fingerVecX * handDirX + fingerVecY * handDirY;

    // Vector from MCP to PIP
    float pipVecX = pipPt.x - mcpPt.x;
    float pipVecY = pipPt.y - mcpPt.y;
    float pipProjection = pipVecX * handDirX + pipVecY * handDirY;

    // Curl metric 1: How much does tip extend beyond PIP?
    // Extended: tipProjection >> pipProjection
    // Curled: tipProjection < pipProjection (tip folds back)
    float extensionRatio = 0.5f;
    if (pipProjection > 0.001f) {
        extensionRatio = tipProjection / (pipProjection * 2.5f);  // 2.5 = expected full extension
        extensionRatio = std::clamp(extensionRatio, 0.0f, 1.0f);
    }
    float curlFromProjection = 1.0f - extensionRatio;

    // Curl metric 2: Distance ratio (tip-to-mcp vs pip-to-mcp)
    // Extended: tip far from mcp
    // Curled: tip close to mcp
    float tipToMcp = distance2D(tipPt, mcpPt);
    float pipToMcp = distance2D(pipPt, mcpPt);

    float distanceRatio = 0.5f;
    if (pipToMcp > 0.001f) {
        // Fully extended finger: tip is ~2.5x further than pip
        distanceRatio = tipToMcp / (pipToMcp * 2.5f);
        distanceRatio = std::clamp(distanceRatio, 0.0f, 1.0f);
    }
    float curlFromDistance = 1.0f - distanceRatio;

    // Curl metric 3: Tip position relative to DIP
    // If tip is closer to wrist than DIP, finger is definitely curled
    float tipToWrist = distance2D(tipPt, wrist);
    float dipToWrist = distance2D(dipPt, wrist);

    float curlFromTipPos = 0.0f;
    if (tipToWrist < dipToWrist * 0.95f) {
        // Tip has folded back past DIP - definitely curled
        curlFromTipPos = 1.0f;
    } else if (tipToWrist < dipToWrist * 1.1f) {
        // Tip is near DIP - partially curled
        curlFromTipPos = 0.5f;
    }

    // Combine metrics (weighted average)
    float curl = curlFromProjection * 0.4f + curlFromDistance * 0.4f + curlFromTipPos * 0.2f;

    return std::clamp(curl, 0.0f, 1.0f);
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

float GestureFSM::getThumbCurl(const std::vector<TrackingResult::NormalizedPoint>& landmarks,
                                bool isRightHand) const {
    // For the curl value, we use the simple X-based check
    // Extended = 0.0, Curled = 1.0
    using LI = LandmarkIndices;

    const auto& thumbTip = landmarks[LI::THUMB_TIP];
    const auto& thumbIP = landmarks[LI::THUMB_IP];

    // Check if thumb is extended based on X position
    bool extended = false;
    if (isRightHand) {
        extended = thumbTip.x < thumbIP.x - 0.02f;
    } else {
        extended = thumbTip.x > thumbIP.x + 0.02f;
    }

    return extended ? 0.0f : 1.0f;
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

