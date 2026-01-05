#pragma once

#include <thread>
#include <atomic>
#include <memory>
#include <lo/lo.h>
// Force remote sync
#include "core/Types.hpp"
#include "core/Logger.hpp"

namespace net {

class OscSender {
public:
    OscSender(std::shared_ptr<core::OscQueue> inputQueue, const std::string& host, const std::string& port);
    ~OscSender();

    void start();
    void stop();

private:
    void loop();
    void send(const core::TrackingResult& result);

    std::shared_ptr<core::OscQueue> _inputQueue;
    std::string _host;
    std::string _port;

    lo_address _loAddress = nullptr;

    std::atomic<bool> _running;
    std::thread _thread;
};

} // namespace net

