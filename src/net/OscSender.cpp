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

    // V3: Send palm position
    lo_message palmMsg = lo_message_new();
    lo_message_add_float(palmMsg, result.palmPosition.x);
    lo_message_add_float(palmMsg, result.palmPosition.y);
    lo_message_add_float(palmMsg, result.palmPosition.z);
    lo_send_message(_loAddress, "/hand/palm", palmMsg);
    lo_message_free(palmMsg);

    // V3: Send velocity
    lo_message velMsg = lo_message_new();
    lo_message_add_float(velMsg, result.velocity.vx);
    lo_message_add_float(velMsg, result.velocity.vy);
    lo_message_add_float(velMsg, result.velocity.vz);
    lo_send_message(_loAddress, "/hand/velocity", velMsg);
    lo_message_free(velMsg);

    // V3: Send gesture
    lo_message gestMsg = lo_message_new();
    lo_message_add_int32(gestMsg, static_cast<int32_t>(result.gesture));
    lo_message_add_float(gestMsg, result.gestureConfidence);
    lo_message_add_string(gestMsg, core::GestureFSM::getStateName(result.gesture));
    lo_send_message(_loAddress, "/hand/gesture", gestMsg);
    lo_message_free(gestMsg);

    // V3: Send VIP status
    lo_message vipMsg = lo_message_new();
    lo_message_add_int32(vipMsg, result.vipLocked ? 1 : 0);
    lo_send_message(_loAddress, "/hand/vip", vipMsg);
    lo_message_free(vipMsg);

    // Legacy: Full tracking message (for backward compatibility)
    lo_message msg = lo_message_new();

    // Add VIP Locked status
    lo_message_add_int32(msg, result.vipLocked ? 1 : 0);

    // Add Landmarks (Blob)
    // Vector of struct {float,float,float} is contiguous in memory
    size_t dataSize = result.landmarks.size() * sizeof(core::TrackingResult::NormalizedPoint);
    lo_blob blob = lo_blob_new(dataSize, result.landmarks.data());
    lo_message_add_blob(msg, blob);

    // Add Gesture Data
    lo_message_add_float(msg, result.pinchDistance);
    lo_message_add_int32(msg, result.gestureId);
    lo_message_add_string(msg, result.gestureName.c_str());

    // Send message to /hand/tracking
    int ret = lo_send_message(_loAddress, "/hand/tracking", msg);
    if (ret == -1) {
        core::Logger::error("OscSender: Failed to send message.");
    }

    lo_blob_free(blob);
    lo_message_free(msg);
}

} // namespace net

