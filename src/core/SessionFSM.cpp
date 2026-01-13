#include "core/SessionFSM.hpp"
#include "core/Logger.hpp"

namespace core {

SessionFSM::SessionFSM()
    : _state(SessionState::IDLE), _frameCount(0) {
}

bool SessionFSM::update(bool palmInVolume) {
    bool stateChanged = false;

    switch (_state) {
        case SessionState::IDLE:
            if (palmInVolume) {
                _frameCount++;
                if (_frameCount >= STABLE_FRAMES) {
                    // Transition: IDLE → ACTIVE
                    _state = SessionState::ACTIVE;
                    _frameCount = 0;
                    stateChanged = true;
                    Logger::info("📍 Session IDLE → ACTIVE");
                }
            } else {
                _frameCount = 0;
            }
            break;

        case SessionState::ACTIVE:
            if (palmInVolume) {
                _frameCount++;
                // Stay ACTIVE, emit signal for OSC (caller will send /player/session/active)
            } else {
                // Transition: ACTIVE → LOST
                _state = SessionState::LOST;
                _frameCount = 0;
                stateChanged = true;
                Logger::info("📍 Session ACTIVE → LOST");
            }
            break;

        case SessionState::LOST:
            _frameCount++;
            if (_frameCount >= LOST_TIMEOUT) {
                // Transition: LOST → IDLE
                _state = SessionState::IDLE;
                _frameCount = 0;
                stateChanged = true;
                Logger::info("📍 Session LOST → IDLE (timeout)");
            }
            break;
    }

    return stateChanged;
}

void SessionFSM::reset() {
    _state = SessionState::IDLE;
    _frameCount = 0;
}

} // namespace core

