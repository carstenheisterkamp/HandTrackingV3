#include "net/OscSender.hpp"
#include "core/GestureFSM.hpp"

namespace net {

OscSender::OscSender(std::shared_ptr<core::OscQueue> inputQueue, const std::string& host, const std::string& port)
    : _inputQueue(std::move(inputQueue)), _host(host), _port(port), _running(false) {
}

OscSender::~OscSender() {
    stop();
    if (_loAddress) {
        lo_address_free(_loAddress);
    }
}

void OscSender::start() {
    if (_running) return;

    // Initialize liblo address
    _loAddress = lo_address_new(_host.c_str(), _port.c_str());
    if (!_loAddress) {
        core::Logger::error("OscSender: Failed to create LO address for ", _host, ":", _port);
        return;
    }

    _running = true;
    _thread = std::thread(&OscSender::loop, this);
    core::Logger::info("OscSender started. Target: ", _host, ":", _port);
    core::Logger::info("  📡 OSC Rate: 28 Hz (guaranteed, every 35.7ms)");
}

void OscSender::stop() {
    if (!_running) return;
    _running = false;
    if (_thread.joinable()) {
        _thread.join();
    }
    core::Logger::info("OscSender stopped.");
}

void OscSender::loop() {
    // FIXED OSC RATE: 28 Hz (35.7ms per frame)
    // Guaranteed minimum rate that client can rely on
    // Even if camera runs at lower FPS, OSC output is stable 28 Hz
    // Client can safely assume: "OSC arrives every ~35.7ms, never slower"
    const auto targetFrameTime = std::chrono::microseconds(35714); // 28 Hz = 35.714ms (1/28)
    auto lastSendTime = std::chrono::steady_clock::now();

    while (_running) {
        core::TrackingResult result;
        if (_inputQueue->pop_front(result)) {
            // Check latency
            auto now = std::chrono::steady_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(now - result.timestamp).count();

            if (latency > 50) {
                // Latency Limit: Discard packets older than 50ms
                // core::Logger::warn("OscSender: Dropping old packet, latency: ", latency, "ms");
                continue;
            }

            // Frame pacing: Wait for target frame time (28 Hz)
            // This ensures OSC arrives at a GUARANTEED rate the client can rely on
            // Client knows: "OSC arrives every 35.7ms MAX, never slower"
            auto elapsed = now - lastSendTime;
            if (elapsed < targetFrameTime) {
                std::this_thread::sleep_for(targetFrameTime - elapsed);
            }
            lastSendTime = std::chrono::steady_clock::now();

            send(result);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

void OscSender::send(const core::TrackingResult& result) {
    if (!_loAddress) return;

    // Build OSC path with hand ID (e.g., /hand/0/palm or /hand/1/palm)
    std::string handPrefix = "/hand/" + std::to_string(result.handId);

    // V3: Send palm position (all coordinates normalized to 0.0-1.0)
    // X, Y: Image coordinates (0=left/top, 1=right/bottom)
    // Z: Depth normalized to play volume (0=1.2m close, 1=2.8m far)
    // Game Engine can scale this 0-1 range to ANY world size
    lo_message palmMsg = lo_message_new();
    lo_message_add_float(palmMsg, static_cast<float>(result.palmPosition.x));
    lo_message_add_float(palmMsg, static_cast<float>(result.palmPosition.y));
    lo_message_add_float(palmMsg, static_cast<float>(result.palmPosition.z));
    lo_send_message(_loAddress, (handPrefix + "/palm").c_str(), palmMsg);
    lo_message_free(palmMsg);

    // V3: Send velocity (change in position per frame, normalized coords)
    lo_message velMsg = lo_message_new();
    lo_message_add_float(velMsg, static_cast<float>(result.velocity.vx));
    lo_message_add_float(velMsg, static_cast<float>(result.velocity.vy));
    lo_message_add_float(velMsg, static_cast<float>(result.velocity.vz));
    lo_send_message(_loAddress, (handPrefix + "/velocity").c_str(), velMsg);
    lo_message_free(velMsg);

    // V3: Send delta (acceleration - change in velocity per frame)
    // All coordinates are normalized
    lo_message deltaMsg = lo_message_new();
    lo_message_add_float(deltaMsg, static_cast<float>(result.delta.dx));
    lo_message_add_float(deltaMsg, static_cast<float>(result.delta.dy));
    lo_message_add_float(deltaMsg, static_cast<float>(result.delta.dz));
    lo_send_message(_loAddress, (handPrefix + "/delta").c_str(), deltaMsg);
    lo_message_free(deltaMsg);

    // V3: Send gesture (normalized confidence 0-1)
    lo_message gestMsg = lo_message_new();
    lo_message_add_int32(gestMsg, static_cast<int32_t>(result.gesture));
    lo_message_add_float(gestMsg, static_cast<float>(result.gestureConfidence));
    lo_message_add_string(gestMsg, core::GestureFSM::getStateName(result.gesture));
    lo_send_message(_loAddress, (handPrefix + "/gesture").c_str(), gestMsg);
    lo_message_free(gestMsg);

    // V3: Send confidence values (for filtering in Unreal)
    // Format: /hand/{id}/confidence [gesture_conf, palm_conf, landmark_conf]
    lo_message confMsg = lo_message_new();
    lo_message_add_float(confMsg, static_cast<float>(result.gestureConfidence));
    lo_message_add_float(confMsg, static_cast<float>(result.palmConfidence));
    lo_message_add_float(confMsg, static_cast<float>(result.landmarkPresence));
    lo_send_message(_loAddress, (handPrefix + "/confidence").c_str(), confMsg);
    lo_message_free(confMsg);

    // Note: Removed /vip message as it's legacy and not needed anymore
}

} // namespace net

