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

    // Get curl values for all 5 fingers (0.0 = extended, 1.0 = curled)
    float thumbCurl = getThumbCurl(landmarks);
    float indexCurl = getFingerCurl(landmarks, LI::INDEX_MCP, LI::INDEX_PIP, LI::INDEX_DIP, LI::INDEX_TIP);
    float middleCurl = getFingerCurl(landmarks, LI::MIDDLE_MCP, LI::MIDDLE_PIP, LI::MIDDLE_DIP, LI::MIDDLE_TIP);
    float ringCurl = getFingerCurl(landmarks, LI::RING_MCP, LI::RING_PIP, LI::RING_DIP, LI::RING_TIP);
    float pinkyCurl = getFingerCurl(landmarks, LI::PINKY_MCP, LI::PINKY_PIP, LI::PINKY_DIP, LI::PINKY_TIP);

    // Convert to extended booleans with hysteresis threshold
    // Extended: curl < 0.4 (stricter)
    // Curled: curl > 0.6 (stricter)
    // In between: keep previous state (handled by debounce)
    constexpr float EXTENDED_THRESHOLD = 0.4f;
    constexpr float CURLED_THRESHOLD = 0.6f;

    bool thumbExtended = thumbCurl < EXTENDED_THRESHOLD;
    bool indexExtended = indexCurl < EXTENDED_THRESHOLD;
    bool middleExtended = middleCurl < EXTENDED_THRESHOLD;
    bool ringExtended = ringCurl < EXTENDED_THRESHOLD;
    bool pinkyExtended = pinkyCurl < EXTENDED_THRESHOLD;

    bool thumbCurled = thumbCurl > CURLED_THRESHOLD;
    bool indexCurled = indexCurl > CURLED_THRESHOLD;
    bool middleCurled = middleCurl > CURLED_THRESHOLD;
    bool ringCurled = ringCurl > CURLED_THRESHOLD;
    bool pinkyCurled = pinkyCurl > CURLED_THRESHOLD;

    // Count extended fingers
    int extendedCount = (thumbExtended ? 1 : 0) + (indexExtended ? 1 : 0) +
                        (middleExtended ? 1 : 0) + (ringExtended ? 1 : 0) + (pinkyExtended ? 1 : 0);

    // Debug log (every 60 frames)
    static int debugCounter = 0;
    if (++debugCounter % 60 == 1) {
        Logger::info("🖐 Gesture Debug: T=", thumbCurl, " I=", indexCurl,
                     " M=", middleCurl, " R=", ringCurl, " P=", pinkyCurl,
                     " ext=", extendedCount);
    }

    // ═══════════════════════════════════════════════════════════
    // Gesture Detection (ordered by specificity - most specific first)
    // Using BOTH extended AND curled checks for robustness
    // ═══════════════════════════════════════════════════════════

    // FIST: All fingers curled ✊
    if (thumbCurled && indexCurled && middleCurled && ringCurled && pinkyCurled) {
        return GestureState::Fist;
    }

    // THUMBS_UP: Only thumb extended, all others curled 👍
    if (thumbExtended && indexCurled && middleCurled && ringCurled && pinkyCurled) {
        return GestureState::ThumbsUp;
    }

    // POINTING: Only index extended ☝️
    if (thumbCurled && indexExtended && middleCurled && ringCurled && pinkyCurled) {
        return GestureState::Pointing;
    }

    // MIDDLE_FINGER: Only middle extended 🖕
    if (thumbCurled && indexCurled && middleExtended && ringCurled && pinkyCurled) {
        return GestureState::MiddleFinger;
    }

    // PEACE: Index + Middle extended, others curled ✌️
    if (thumbCurled && indexExtended && middleExtended && ringCurled && pinkyCurled) {
        return GestureState::Peace;
    }

    // METAL: Index + Pinky extended, others curled 🤘
    if (thumbCurled && indexExtended && middleCurled && ringCurled && pinkyExtended) {
        return GestureState::Metal;
    }

    // CALL_ME: Thumb + Pinky extended, middle fingers curled 🤙
    if (thumbExtended && indexCurled && middleCurled && ringCurled && pinkyExtended) {
        return GestureState::CallMe;
    }

    // LOVE_YOU: Thumb + Index + Pinky extended 🤟
    if (thumbExtended && indexExtended && middleCurled && ringCurled && pinkyExtended) {
        return GestureState::LoveYou;
    }

    // TWO: Thumb + Index extended (V sign with thumb)
    if (thumbExtended && indexExtended && middleCurled && ringCurled && pinkyCurled) {
        return GestureState::Two;
    }

    // THREE: Thumb + Index + Middle extended
    if (thumbExtended && indexExtended && middleExtended && ringCurled && pinkyCurled) {
        return GestureState::Three;
    }

    // FOUR: All except thumb extended
    if (thumbCurled && indexExtended && middleExtended && ringExtended && pinkyExtended) {
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

    // ═══════════════════════════════════════════════════════════
    // Fallback: Count-based detection for ambiguous cases
    // ═══════════════════════════════════════════════════════════

    if (extendedCount >= 5) return GestureState::Five;
    if (extendedCount == 4) return GestureState::Four;
    if (extendedCount == 0) return GestureState::Fist;

    // Ambiguous - return current state to avoid flicker
    return state_;
}

bool GestureFSM::isFingerExtended(const std::vector<TrackingResult::NormalizedPoint>& landmarks,
                                   int mcp, int pip, int dip, int tip) const {
    // Use curl factor - extended if curl < 0.5
    float curl = getFingerCurl(landmarks, mcp, pip, dip, tip);
    return curl < 0.5f;
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

bool GestureFSM::isThumbExtended(const std::vector<TrackingResult::NormalizedPoint>& landmarks) const {
    float curl = getThumbCurl(landmarks);
    return curl < 0.5f;
}

float GestureFSM::getThumbCurl(const std::vector<TrackingResult::NormalizedPoint>& landmarks) const {
    // Thumb curl is special because thumb moves perpendicular to other fingers
    using LI = LandmarkIndices;

    const auto& thumbTip = landmarks[LI::THUMB_TIP];
    const auto& thumbIP = landmarks[LI::THUMB_IP];
    const auto& thumbMCP = landmarks[LI::THUMB_MCP];
    const auto& thumbCMC = landmarks[LI::THUMB_CMC];
    const auto& indexMCP = landmarks[LI::INDEX_MCP];
    const auto& wrist = landmarks[LI::WRIST];

    // Metric 1: Distance from index finger base
    // Extended thumb is spread away from other fingers
    float thumbTipToIndex = distance2D(thumbTip, indexMCP);
    float thumbMCPToIndex = distance2D(thumbMCP, indexMCP);

    float spreadRatio = 0.5f;
    if (thumbMCPToIndex > 0.001f) {
        // Extended: tip is ~2x further from index than MCP
        spreadRatio = thumbTipToIndex / (thumbMCPToIndex * 2.0f);
        spreadRatio = std::clamp(spreadRatio, 0.0f, 1.0f);
    }
    float curlFromSpread = 1.0f - spreadRatio;

    // Metric 2: Extension along thumb direction
    float thumbDirX = thumbMCP.x - thumbCMC.x;
    float thumbDirY = thumbMCP.y - thumbCMC.y;
    float thumbDirLen = std::sqrt(thumbDirX * thumbDirX + thumbDirY * thumbDirY);

    float curlFromExtension = 0.5f;
    if (thumbDirLen > 0.001f) {
        thumbDirX /= thumbDirLen;
        thumbDirY /= thumbDirLen;

        // Project tip onto thumb direction
        float tipVecX = thumbTip.x - thumbMCP.x;
        float tipVecY = thumbTip.y - thumbMCP.y;
        float tipProjection = tipVecX * thumbDirX + tipVecY * thumbDirY;

        // Extended thumb: positive projection
        // Curled thumb: negative or small projection
        if (tipProjection > thumbDirLen * 0.5f) {
            curlFromExtension = 0.0f;  // Extended
        } else if (tipProjection < thumbDirLen * 0.1f) {
            curlFromExtension = 1.0f;  // Curled
        } else {
            curlFromExtension = 0.5f;  // Ambiguous
        }
    }

    // Metric 3: Distance from wrist
    float thumbTipToWrist = distance2D(thumbTip, wrist);
    float thumbIPToWrist = distance2D(thumbIP, wrist);

    float curlFromWrist = 0.5f;
    if (thumbTipToWrist > thumbIPToWrist * 1.3f) {
        curlFromWrist = 0.0f;  // Tip is far from wrist - extended
    } else if (thumbTipToWrist < thumbIPToWrist * 1.05f) {
        curlFromWrist = 1.0f;  // Tip is close to wrist - curled
    }

    // Combine metrics
    float curl = curlFromSpread * 0.4f + curlFromExtension * 0.3f + curlFromWrist * 0.3f;
    return std::clamp(curl, 0.0f, 1.0f);
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

