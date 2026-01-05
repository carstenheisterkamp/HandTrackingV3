#pragma once

#include <thread>
#include <atomic>
#include <memory>
#include <string>

#include "core/PipelineManager.hpp"
#include "core/Types.hpp"

namespace core {

/**
 * Dedicated thread for receiving frames from OAK-D.
 * Copies data from DepthAI messages into our pre-allocated, CUDA-registered FramePool.
 */
class InputLoop {
public:
    InputLoop(std::shared_ptr<PipelineManager> pipelineMgr,
              std::shared_ptr<AppFramePool> framePool,
              std::shared_ptr<AppProcessingQueue> outputQueue);

    ~InputLoop();

    void start();
    void stop();

private:
    void loop();

    std::shared_ptr<PipelineManager> pipelineMgr_;
    std::shared_ptr<AppFramePool> framePool_;
    std::shared_ptr<AppProcessingQueue> outputQueue_; // To Processing Thread

    std::thread thread_;
    std::atomic<bool> running_{false};

    // Queue name to fetch from
    std::string queueName_ = "rgb";
};

} // namespace core

