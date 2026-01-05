#include "core/OscSender.hpp"
#include <iostream>

namespace core {

OscSender::OscSender(std::shared_ptr<OscQueue> inputQueue, const std::string& host, const std::string& port)
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
        Logger::error("OscSender: Failed to create LO address for ", _host, ":", _port);
        return;
    }

    _running = true;
    _thread = std::thread(&OscSender::loop, this);
    Logger::info("OscSender started. Target: ", _host, ":", _port);
}

void OscSender::stop() {
    if (!_running) return;
    _running = false;
    if (_thread.joinable()) {
        _thread.join();
    }
    Logger::info("OscSender stopped.");
}

void OscSender::loop() {
    while (_running) {
        TrackingResult result;
        if (_inputQueue->pop_front(result)) {
            // Check latency
            auto now = std::chrono::steady_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(now - result.timestamp).count();

            if (latency > 50) {
                // Latency Limit: Discard packets older than 50ms
                // Logger::warn("OscSender: Dropping old packet, latency: ", latency, "ms");
                continue;
            }

            send(result);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

void OscSender::send(const TrackingResult& result) {
    if (!_loAddress) return;

    lo_message msg = lo_message_new();

    // Add VIP Locked status
    lo_message_add_int32(msg, result.vipLocked ? 1 : 0);

    // Add Landmarks (Blob or individual floats?)
    // Instructions: "Format: Use compact blobs or bundled messages to reduce overhead."
    // Let's use a blob for the 63 floats.
    // 63 * 4 bytes = 252 bytes.

    lo_blob blob = lo_blob_new(result.landmarks.size() * sizeof(float), result.landmarks.data());
    lo_message_add_blob(msg, blob);

    // Send message to /hand/tracking
    int ret = lo_send_message(_loAddress, "/hand/tracking", msg);
    if (ret == -1) {
        Logger::error("OscSender: Failed to send message.");
    }

    lo_blob_free(blob);
    lo_message_free(msg);
}

} // namespace core

