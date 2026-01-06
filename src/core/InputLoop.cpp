#include "core/InputLoop.hpp"
#include "core/Logger.hpp"
#include <cstring> // for memcpy
#include <chrono>
#include <algorithm> // for std::fill
#include <depthai/pipeline/datatype/MessageGroup.hpp>
#include <depthai/pipeline/datatype/NNData.hpp>
#include <depthai/pipeline/datatype/ImgFrame.hpp>

namespace core {

InputLoop::InputLoop(std::shared_ptr<PipelineManager> pipelineMgr,
                     std::shared_ptr<AppFramePool> framePool,
                     std::shared_ptr<AppProcessingQueue> outputQueue)
    : pipelineMgr_(std::move(pipelineMgr)), framePool_(std::move(framePool)), outputQueue_(std::move(outputQueue)) {
}

InputLoop::~InputLoop() {
    stop();
}

void InputLoop::start() {
    if (running_) return;
    running_ = true;
    thread_ = std::thread(&InputLoop::loop, this);

    // Set thread priority (if possible on this OS)
    // On Linux/Jetson we would use pthread_setschedparam for SCHED_FIFO
    // TODO: Implement Realtime priority setting
    Logger::info("InputLoop started.");
}

void InputLoop::stop() {
    if (!running_) return;
    running_ = false;
    if (thread_.joinable()) {
        thread_.join();
    }
    Logger::info("InputLoop stopped.");
}

void InputLoop::loop() {
    // Try to get 'sync' queue first, then 'rgb'
    auto queue = pipelineMgr_->getOutputQueue("sync");
    bool isSync = true;
    if (!queue) {
        queue = pipelineMgr_->getOutputQueue("rgb");
        isSync = false;
    }

    if (!queue) {
        Logger::error("InputLoop: No suitable queue found (checked 'sync' and 'rgb')!");
        return;
    }

    Logger::debug("InputLoop: Waiting for frames from queue: ", queue->getName());

    while (running_) {
        try {
            std::shared_ptr<dai::ImgFrame> imgFrame;
            std::shared_ptr<dai::NNData> landmarkData;
            std::shared_ptr<dai::NNData> palmData;

            if (isSync) {
                auto msgGroup = queue->tryGet<dai::MessageGroup>();
                if (!msgGroup) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
                imgFrame = msgGroup->get<dai::ImgFrame>("rgb");
                landmarkData = msgGroup->get<dai::NNData>("landmarks");
                palmData = msgGroup->get<dai::NNData>("palm");
            } else {
                imgFrame = queue->tryGet<dai::ImgFrame>();
                if (!imgFrame) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
            }

            if (!imgFrame) {
             // Should not happen if queue returned message, but safety check
             continue;
            }

            // 1. Acquire Frame from Pool
            Frame* frame = framePool_->acquire();
            if (!frame) {
                // Pool empty (Backpressure) -> Drop frame
                Logger::warn("InputLoop: FramePool empty, dropping frame seq=", imgFrame->getSequenceNum());
                continue;
            }

            // 2. Copy Data to Pinned Memory
            // Verify size
            size_t dataSize = imgFrame->getData().size();

            static bool typeLogged = false;
            if (!typeLogged) {
                Logger::info("InputLoop: Received frame type: ", (int)imgFrame->getType(),
                             " Size: ", imgFrame->getWidth(), "x", imgFrame->getHeight(),
                             " DataSize: ", dataSize);
                typeLogged = true;
            }

            if (dataSize > frame->size) {
                Logger::error("InputLoop: Frame too large (", dataSize, " > ", frame->size, ")");
                framePool_->release(frame);
                continue;
            }

            // Copy to pinned buffer (CPU Copy).
            // This is necessary because OAK-PoE data arrives in non-pinned memory.
            // The destination 'frame->data' is registered with CUDA, allowing the GPU
            // to read it directly (Zero-Copy Host->Device).
            std::memcpy(frame->data.get(), imgFrame->getData().data(), dataSize);

            // Remove reference since we own the data now
            frame->daiFrame = nullptr;

            // NOTE: Mono L/R stereo frames disabled in minimal pipeline
            // Will be re-enabled when stereo is added back

            // 3. Fill Metadata
            frame->width = imgFrame->getWidth();
            frame->height = imgFrame->getHeight();
            frame->type = (int)imgFrame->getType();
            frame->sequenceNum = imgFrame->getSequenceNum();
            frame->timestamp = std::chrono::steady_clock::now(); // Host arrival
            // frame->captureTimestamp = ... // TODO: Convert imgFrame->getTimestamp()

            // 4. Copy NN Data if available
            if (landmarkData) {
                try {
                    // Robust Layer Selection:
                    // The model should output 63 floats (21 landmarks * 3).
                    // We iterate through all layers to find the one with the correct size.
                    auto layerNames = landmarkData->getAllLayerNames();
                    bool found = false;

                    for (const auto& name : layerNames) {
                        auto tensor = landmarkData->getTensor<float>(name);
                        if (tensor.size() == 63) {
                            frame->nnData.assign(tensor.begin(), tensor.end());
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        // Fallback: Use the first tensor but log a warning with details
                        if (!layerNames.empty()) {
                            auto tensor = landmarkData->getFirstTensor<float>();
                            frame->nnData.assign(tensor.begin(), tensor.end());

                            Logger::warn("InputLoop: NNData size mismatch (Expected 63). Available layers:");
                            for (const auto& name : layerNames) {
                                 auto t = landmarkData->getTensor<float>(name);
                                 Logger::warn(" - Layer '", name, "': size=", t.size());
                            }
                        } else {
                            Logger::warn("InputLoop: No layers in NNData");
                            frame->nnData.clear();
                        }
                    }
                } catch (const std::exception& e) {
                    Logger::warn("InputLoop: Failed to get NN data: ", e.what());
                }
            } else {
                frame->nnData.clear();
            }

            // 4b. Copy Palm Data if available
            if (palmData) {
                try {
                    auto layerNames = palmData->getAllLayerNames();
                    if (!layerNames.empty()) {
                        auto tensor = palmData->getTensor<float>(layerNames[0]);
                        frame->palmData.assign(tensor.begin(), tensor.end());
                    } else {
                        frame->palmData.clear();
                    }
                } catch (const std::exception& e) {
                    Logger::warn("InputLoop: Failed to get Palm data: ", e.what());
                    frame->palmData.clear();
                }
            } else {
                frame->palmData.clear();
            }

            // 5. Push to Processing Queue
            if (!outputQueue_->try_push(frame)) {
                // Queue full -> Drop and release
                Logger::warn("InputLoop: OutputQueue full, dropping frame seq=", frame->sequenceNum);
                framePool_->release(frame);
            }
        } catch (const std::exception& e) {
            Logger::error("InputLoop Critical Error (Connection lost?): ", e.what());
            hasError_ = true;
            running_ = false; // Stop loop so main can detect and restart
        }
    }
}

} // namespace core

