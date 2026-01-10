#include "net/OscSender.hpp"
#include "core/GestureFSM.hpp"
#include <iostream>

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

    // V3: Send palm position (explicit float cast to ensure correct type)
    lo_message palmMsg = lo_message_new();
    lo_message_add_float(palmMsg, static_cast<float>(result.palmPosition.x));
    lo_message_add_float(palmMsg, static_cast<float>(result.palmPosition.y));
    lo_message_add_float(palmMsg, static_cast<float>(result.palmPosition.z));
    lo_send_message(_loAddress, (handPrefix + "/palm").c_str(), palmMsg);
    lo_message_free(palmMsg);

    // V3: Send velocity (explicit float cast)
    lo_message velMsg = lo_message_new();
    lo_message_add_float(velMsg, static_cast<float>(result.velocity.vx));
    lo_message_add_float(velMsg, static_cast<float>(result.velocity.vy));
    lo_message_add_float(velMsg, static_cast<float>(result.velocity.vz));
    lo_send_message(_loAddress, (handPrefix + "/velocity").c_str(), velMsg);
    lo_message_free(velMsg);

    // V3: Send delta (acceleration/change in velocity)
    lo_message deltaMsg = lo_message_new();
    lo_message_add_float(deltaMsg, static_cast<float>(result.delta.dx));
    lo_message_add_float(deltaMsg, static_cast<float>(result.delta.dy));
    lo_message_add_float(deltaMsg, static_cast<float>(result.delta.dz));
    lo_send_message(_loAddress, (handPrefix + "/delta").c_str(), deltaMsg);
    lo_message_free(deltaMsg);

    // V3: Send gesture (int, float, string)
    lo_message gestMsg = lo_message_new();
    lo_message_add_int32(gestMsg, static_cast<int32_t>(result.gesture));
    lo_message_add_float(gestMsg, static_cast<float>(result.gestureConfidence));
    lo_message_add_string(gestMsg, core::GestureFSM::getStateName(result.gesture));
    lo_send_message(_loAddress, (handPrefix + "/gesture").c_str(), gestMsg);
    lo_message_free(gestMsg);

    // Note: Removed /vip message as it's legacy and not needed anymore
}

} // namespace net

