#pragma once

#include "core/Frame.hpp"
#include "core/SpscQueue.hpp"
#include "core/Logger.hpp"
#include <vector>
#include <memory>

namespace core {

template<size_t PoolSize, size_t BufferSize, size_t DepthBufferSize, size_t MonoBufferSize = 0>
class FramePool {
public:
    FramePool() {
        // Allocate frames and buffers
        for (size_t i = 0; i < PoolSize; ++i) {
            auto frame = std::make_unique<Frame>();

            // Allocate aligned buffer for Zero-Copy (Host -> Device)
            frame->data = allocate_aligned<uint8_t>(BufferSize);
            frame->size = BufferSize;

            // Allocate Depth Buffer (for computed depth on Jetson)
            frame->depthData = allocate_aligned<uint8_t>(DepthBufferSize);
            frame->depthSize = DepthBufferSize;

            // Allocate Mono Left/Right Buffers (for stereo input to Jetson GPU)
            if constexpr (MonoBufferSize > 0) {
                frame->monoLeftData = allocate_aligned<uint8_t>(MonoBufferSize);
                frame->monoRightData = allocate_aligned<uint8_t>(MonoBufferSize);
                frame->monoSize = MonoBufferSize;
                register_buffer_cuda(frame->monoLeftData.get(), MonoBufferSize);
                register_buffer_cuda(frame->monoRightData.get(), MonoBufferSize);
            }

            // Register with CUDA
            register_buffer_cuda(frame->data.get(), BufferSize);
            register_buffer_cuda(frame->depthData.get(), DepthBufferSize);

            // Add to storage and free queue
            if (!freeQueue_.try_push(frame.get())) {
                Logger::error("Failed to push initial frame to free queue");
            }

            storage_.push_back(std::move(frame));
        }

        Logger::info("FramePool initialized with ", PoolSize, " pinned frames");
    }

    /**
     * Acquires a free frame from the pool.
     * Returns nullptr if no frames are available (backpressure/starvation).
     */
    Frame* acquire() {
        auto frameOpt = freeQueue_.try_pop();
        if (!frameOpt) {
            return nullptr;
        }
        return *frameOpt;
    }

    /**
     * Releases a frame back to the pool.
     */
    void release(Frame* frame) {
        if (!frame) return;
        if (!freeQueue_.try_push(frame)) {
            Logger::error("Failed to release frame to pool (queue full?)");
        }
    }

private:
    // Storage to keep frames alive
    std::vector<std::unique_ptr<Frame>> storage_;

    // Queue of free frame pointers
    // Capacity + 1 because SpscQueue wastes one slot
    SpscQueue<Frame*, PoolSize + 1> freeQueue_;
};

} // namespace core

