#include "core/InputLoop.hpp"
#include "core/Logger.hpp"
#include <cstring> // for memcpy
#include <chrono>
#include <algorithm> // for std::fill
#include <depthai/pipeline/datatype/MessageGroup.hpp>
#include <depthai/pipeline/datatype/ImgFrame.hpp>
// Note: NNData.hpp removed - NNs run on Jetson per OPTIMAL_WORKFLOW_V2_FINAL.md

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
    // Device FPS measurement
    auto lastDeviceFpsTime = std::chrono::steady_clock::now();
    int deviceFrameCount = 0;

    // ========================================
    // V3 SENSOR-ONLY PIPELINE
    // Per OPTIMAL_WORKFLOW_V3.md:
    // - OAK-D: RGB + optional Mono L/R synchronized
    // - All NNs run on Jetson (TensorRT)
    // ========================================

    // Try sync queue first (stereo enabled), fallback to rgb (stereo disabled)
    auto queue = pipelineMgr_->getOutputQueue("sync");
    bool isSync = (queue != nullptr);

    if (!queue) {
        queue = pipelineMgr_->getOutputQueue("rgb");
        isSync = false;
    }

    if (!queue) {
        Logger::error("InputLoop: No queue found (checked 'sync' and 'rgb')!");
        hasError_ = true;
        return;
    }

    Logger::info("═══════════════════════════════════════════════════════════");
    Logger::info("V3 InputLoop Started");
    Logger::info("  Queue: ", queue->getName(), " (Sync: ", isSync ? "YES" : "NO", ")");
    Logger::info("  NNs: Will run on Jetson (TensorRT) - Phase 2");
    if (isSync) {
        Logger::info("  Stereo: Will compute on Jetson GPU - Phase 3");
    } else {
        Logger::info("  Stereo: DISABLED (RGB-only mode)");
    }
    Logger::info("═══════════════════════════════════════════════════════════");

    while (running_) {
        try {
            std::shared_ptr<dai::ImgFrame> imgFrame;
            std::shared_ptr<dai::ImgFrame> monoLeftFrame;
            std::shared_ptr<dai::ImgFrame> monoRightFrame;

            if (isSync) {
                // Sync mode: RGB + Mono L/R
                auto msgGroup = queue->tryGet<dai::MessageGroup>();
                if (!msgGroup) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }

                imgFrame = msgGroup->get<dai::ImgFrame>("rgb");
                monoLeftFrame = msgGroup->get<dai::ImgFrame>("monoLeft");
                monoRightFrame = msgGroup->get<dai::ImgFrame>("monoRight");
            } else {
                // RGB-only mode
                imgFrame = queue->tryGet<dai::ImgFrame>();
                if (!imgFrame) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
            }

            if (!imgFrame) {
                Logger::warn("InputLoop: Missing 'rgb' frame");
                continue;
            }

            // DEVICE FPS TRACKING
            deviceFrameCount++;
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastDeviceFpsTime).count();
            if (elapsed >= 2000) {
                float deviceFps = static_cast<float>(deviceFrameCount) * 1000.0f / static_cast<float>(elapsed);
                Logger::info("📹 DEVICE FPS: ", deviceFps, " (OAK-D ", isSync ? "Synced" : "RGB", " Stream)");
                if (deviceFps < 25.0f) {
                    Logger::warn("  ⚠️ Device FPS below target!");
                } else {
                    Logger::info("  ✅ Device FPS OK");
                }

                if (isSync && monoLeftFrame && monoRightFrame) {
                    Logger::info("  📐 Stereo: L+R active (", monoLeftFrame->getWidth(), "x", monoLeftFrame->getHeight(), ")");
                }

                deviceFrameCount = 0;
                lastDeviceFpsTime = now;
            }

            // 1. Acquire Frame from Pool
            Frame* frame = framePool_->acquire();
            if (!frame) {
                Logger::warn("InputLoop: FramePool empty, dropping frame seq=", imgFrame->getSequenceNum());
                continue;
            }

            // 2. Copy RGB Data to Pinned Memory
            size_t dataSize = imgFrame->getData().size();

            static bool typeLogged = false;
            if (!typeLogged) {
                Logger::info("InputLoop: RGB frame type: ", (int)imgFrame->getType(),
                             " Size: ", imgFrame->getWidth(), "x", imgFrame->getHeight(),
                             " DataSize: ", dataSize);
                typeLogged = true;
            }

            if (dataSize > frame->size) {
                Logger::error("InputLoop: RGB frame too large (", dataSize, " > ", frame->size, ")");
                framePool_->release(frame);
                continue;
            }

            std::memcpy(frame->data.get(), imgFrame->getData().data(), dataSize);
            frame->daiFrame = nullptr;

            // 3. Copy Mono L/R for GPU Stereo Depth (if available)
            frame->hasStereoData = false;
            if (monoLeftFrame && monoRightFrame) {
                size_t monoLeftSize = monoLeftFrame->getData().size();
                size_t monoRightSize = monoRightFrame->getData().size();

                if (monoLeftSize <= frame->monoSize && monoRightSize <= frame->monoSize) {
                    std::memcpy(frame->monoLeftData.get(), monoLeftFrame->getData().data(), monoLeftSize);
                    std::memcpy(frame->monoRightData.get(), monoRightFrame->getData().data(), monoRightSize);

                    frame->monoWidth = monoLeftFrame->getWidth();
                    frame->monoHeight = monoLeftFrame->getHeight();
                    frame->hasStereoData = true;
                }
            }

            // 4. Fill Metadata
            frame->width = imgFrame->getWidth();
            frame->height = imgFrame->getHeight();
            frame->type = (int)imgFrame->getType();
            frame->sequenceNum = imgFrame->getSequenceNum();
            frame->timestamp = std::chrono::steady_clock::now();
            frame->captureTimestamp = imgFrame->getTimestamp();

            // V3: No NN data from OAK-D
            frame->nnData.clear();
            frame->palmData.clear();

            // 5. Push to Processing Queue
            if (!outputQueue_->try_push(frame)) {
                Logger::warn("InputLoop: OutputQueue full, dropping frame seq=", frame->sequenceNum);
                framePool_->release(frame);
            }
        } catch (const std::exception& e) {
            Logger::error("InputLoop Critical Error (Connection lost?): ", e.what());
            hasError_ = true;
            running_ = false;
        }
    }
}

} // namespace core

