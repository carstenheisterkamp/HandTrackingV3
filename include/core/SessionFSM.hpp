#pragma once

#include <cstdint>
#include <array>

namespace core {

/**
 * SessionFSM - Player Session State Machine
 *
 * Manages per-hand session state:
 * - IDLE: No hand in volume
 * - ACTIVE: Hand stable in volume (≥15 frames)
 * - LOST: Was ACTIVE, hand exited volume (≤3 frames, then → IDLE)
 *
 * Emits OSC events on state transitions:
 * - /player/session/enter  (IDLE → ACTIVE)
 * - /player/session/active (every frame while ACTIVE)
 * - /player/session/lost   (ACTIVE → LOST)
 * - /player/session/exit   (LOST → IDLE)
 */

enum class SessionState {
    IDLE,    ///< No hand detected or lost for >3 frames
    ACTIVE,  ///< Hand stable in volume for ≥15 frames
    LOST     ///< Was ACTIVE, now outside volume, waiting for timeout
};

class SessionFSM {
public:
    SessionFSM();
    ~SessionFSM() = default;

    /**
     * Update session state based on palm-in-volume status
     * @param palmInVolume True if palm center is inside play volume
     * @return True if state changed (caller should emit OSC event)
     */
    bool update(bool palmInVolume);

    /**
     * Get current session state
     */
    SessionState getState() const { return _state; }

    /**
     * Check if session is active (hand is stable in volume)
     */
    bool isActive() const { return _state == SessionState::ACTIVE; }

    /**
     * Get frame count in current state
     */
    int getFrameCount() const { return _frameCount; }

    /**
     * Reset FSM to IDLE
     */
    void reset();

private:
    SessionState _state;
    int _frameCount;

    // Constants for state transitions
    static constexpr int STABLE_FRAMES = 15;  ///< Frames to transition IDLE → ACTIVE
    static constexpr int LOST_TIMEOUT = 3;    ///< Frames to transition LOST → IDLE
};

} // namespace core

