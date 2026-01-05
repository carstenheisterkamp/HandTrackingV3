#pragma once

#include <thread>
#include <atomic>
#include <memory>
#include <lo/lo.h>
#include "Types.hpp"
#include "Logger.hpp"

namespace core {

class OscSender {
public:
    OscSender(std::shared_ptr<OscQueue> inputQueue, const std::string& host, const std::string& port);
    ~OscSender();

    void start();
    void stop();

private:
    void loop();
    void send(const TrackingResult& result);

    std::shared_ptr<OscQueue> _inputQueue;
    std::string _host;
    std::string _port;

    lo_address _loAddress = nullptr;

    std::atomic<bool> _running;
    std::thread _thread;
};

} // namespace core

