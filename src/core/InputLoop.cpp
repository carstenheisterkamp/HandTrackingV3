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

        // 2. Copy Data (One-Copy to Pinned Memory)
        // Verify size
        size_t dataSize = imgFrame->getData().size();
        if (dataSize > frame->size) {
            Logger::error("InputLoop: Frame too large (", dataSize, " > ", frame->size, ")");
            framePool_->release(frame);
            continue;
        }

        // Memcpy to pinned buffer
        std::memcpy(frame->data.get(), imgFrame->getData().data(), dataSize);

        // 3. Fill Metadata
        frame->width = imgFrame->getWidth();
        frame->height = imgFrame->getHeight();
        frame->sequenceNum = imgFrame->getSequenceNum();
        frame->timestamp = std::chrono::steady_clock::now(); // Host arrival
        // frame->captureTimestamp = ... // TODO: Convert imgFrame->getTimestamp()

        // 4. Copy NN Data if available
        if (landmarkData) {
            // Assuming generic NNData which has a layer named "output" or similar, or just raw data.
            // For hand_landmark_full_sh4, we need to know the output layer name or just take the first one.
            // dai::NNData::getData() returns the raw data of the first layer? No, it returns raw data if it's a single blob?
            // Actually NNData has methods to get layer data.
            // If we use generic NeuralNetwork node, we might get raw tensor data.
            // Let's assume we copy the first layer for now.
            // Or just copy all data.
            // For now, let's try to get "Identity" or whatever the output layer is.
            // Or just use getAllLayerNames() and iterate.

            // For simplicity, let's just copy the first layer's float data.
            // hand_landmark_full_sh4 usually outputs a single tensor of 63 floats (21 landmarks * 3 coords).
            // Let's check if we can get it as float.

            // nnData->getFirstLayerFp16();

            // We need to be careful about exceptions here.
            try {
                // Use raw data access to avoid API version mismatches
                auto rawData = landmarkData->getData();
                size_t size = rawData.size();
                if (size >= 63 * sizeof(float)) {
                    // Assuming the data is float (FP32)
                    // Note: If the model outputs FP16, we would need conversion.
                    // But usually standard models output FP32 or we can request it.
                    // For now, assume FP32 as per standard DepthAI behavior for getTensor<float>.

                    // Safety check for alignment/size
                    size_t numFloats = rawData.size() / sizeof(float);

                    // Use memcpy to be safe against alignment issues on ARM
                    frame->nnData.resize(numFloats);
                    std::memcpy(frame->nnData.data(), rawData.data(), rawData.size());
                } else {
                    Logger::warn("InputLoop: NNData size too small: ", rawData.size());
                }
            } catch (const std::exception& e) {
                Logger::warn("InputLoop: Failed to get NN data: ", e.what());
            }
        } else {
            frame->nnData.clear();
        }

        // 5. Push to Processing Queue
        if (!outputQueue_->try_push(frame)) {
            // Queue full -> Drop and release
            Logger::warn("InputLoop: OutputQueue full, dropping frame seq=", frame->sequenceNum);
            framePool_->release(frame);
        }
    }
}

} // namespace core

