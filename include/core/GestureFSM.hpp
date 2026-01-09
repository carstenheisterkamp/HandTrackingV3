#pragma once

#include "Types.hpp"
#include <array>
#include <functional>

namespace core {

/**
 * V3 GestureFSM: Finite State Machine for robust gesture recognition
 *
 * States: Idle → Palm → Pinch/Grab/Point
 *
 * Features:
 * - Hysteresis thresholds (enter ≠ exit)
 * - Debounce (N consecutive frames required)
 * - Finger extension checks
 */
class GestureFSM {
public:
    /**
     * Landmark indices for gesture detection
     * Based on MediaPipe Hand Landmark model
     */
    struct LandmarkIndices {
        // Thumb
        static constexpr int THUMB_TIP = 4;
        static constexpr int THUMB_IP = 3;
        static constexpr int THUMB_MCP = 2;
        static constexpr int THUMB_CMC = 1;

        // Index finger
        static constexpr int INDEX_TIP = 8;
        static constexpr int INDEX_DIP = 7;
        static constexpr int INDEX_PIP = 6;
        static constexpr int INDEX_MCP = 5;

        // Middle finger
        static constexpr int MIDDLE_TIP = 12;
        static constexpr int MIDDLE_DIP = 11;
        static constexpr int MIDDLE_PIP = 10;
        static constexpr int MIDDLE_MCP = 9;

        // Ring finger
        static constexpr int RING_TIP = 16;
        static constexpr int RING_DIP = 15;
        static constexpr int RING_PIP = 14;
        static constexpr int RING_MCP = 13;

        // Pinky
        static constexpr int PINKY_TIP = 20;
        static constexpr int PINKY_DIP = 19;
        static constexpr int PINKY_PIP = 18;
        static constexpr int PINKY_MCP = 17;

        // Wrist
        static constexpr int WRIST = 0;
    };

    using TransitionCallback = std::function<void(GestureState from, GestureState to)>;

    GestureFSM();

    /**
     * Update FSM with new landmark data
     * @param landmarks 21 3D landmarks (normalized coordinates)
     * @param isRightHand true if right hand, false if left (for thumb detection)
     * @return Current gesture state (may be unchanged due to debounce)
     */
    GestureState update(const std::vector<TrackingResult::NormalizedPoint>& landmarks,
                        bool isRightHand = true);

    /**
     * Handle hand lost (no detection)
     * Transitions to Idle after timeout
     */
    void handleHandLost();

    /**
     * Get current state
     */
    [[nodiscard]] GestureState getState() const { return state_; }

    /**
     * Get state name as string
     */
    [[nodiscard]] static const char* getStateName(GestureState state);

    /**
     * Get confidence (0-1) based on debounce progress
     */
    [[nodiscard]] float getConfidence() const;

    /**
     * Set callback for state transitions
     */
    void setTransitionCallback(TransitionCallback callback) { transitionCallback_ = std::move(callback); }

    /**
     * Reset to Idle state
     */
    void reset();

private:
    GestureState state_ = GestureState::Idle;
    GestureState pendingState_ = GestureState::Idle;
    int frameCount_ = 0;
    int handLostFrames_ = 0;

    TransitionCallback transitionCallback_;

    // Detection helpers
    [[nodiscard]] GestureState detectGesture(const std::vector<TrackingResult::NormalizedPoint>& landmarks,
                                              bool isRightHand) const;
    [[nodiscard]] GestureState detectOpenHand(const std::vector<TrackingResult::NormalizedPoint>& landmarks,
                                               bool isRightHand) const;

    // Finger curl: 0.0 = fully extended, 1.0 = fully curled
    [[nodiscard]] float getFingerCurl(const std::vector<TrackingResult::NormalizedPoint>& landmarks,
                                       int mcp, int pip, int dip, int tip) const;

    [[nodiscard]] float getThumbCurl(const std::vector<TrackingResult::NormalizedPoint>& landmarks,
                                      bool isRightHand) const;

    // Simple Y-based finger detection (robust to viewing angle)
    [[nodiscard]] bool isFingerUp(const std::vector<TrackingResult::NormalizedPoint>& landmarks,
                                   int tipIdx, int pipIdx) const;

    [[nodiscard]] bool isThumbUp(const std::vector<TrackingResult::NormalizedPoint>& landmarks,
                                  bool isRightHand) const;

    [[nodiscard]] bool isFingerExtended(const std::vector<TrackingResult::NormalizedPoint>& landmarks,
                                        int mcp, int pip, int dip, int tip) const;

    [[nodiscard]] bool isThumbExtended(const std::vector<TrackingResult::NormalizedPoint>& landmarks,
                                        bool isRightHand) const;

    [[nodiscard]] bool isVulcanSpread(const std::vector<TrackingResult::NormalizedPoint>& landmarks) const;

    [[nodiscard]] float distance(const TrackingResult::NormalizedPoint& a,
                                 const TrackingResult::NormalizedPoint& b) const;

    [[nodiscard]] float distance2D(const TrackingResult::NormalizedPoint& a,
                                   const TrackingResult::NormalizedPoint& b) const;

    [[nodiscard]] float getHandSize(const std::vector<TrackingResult::NormalizedPoint>& landmarks) const;
    [[nodiscard]] float getPinchDistance(const std::vector<TrackingResult::NormalizedPoint>& landmarks) const;

    void transitionTo(GestureState newState);
};

} // namespace core

