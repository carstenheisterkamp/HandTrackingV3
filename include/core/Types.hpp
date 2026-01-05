#pragma once

#include <array>
#include <chrono>
#include <memory>
#include "Frame.hpp"
#include "FramePool.hpp"
#include "SpscQueue.hpp"

namespace core {

// Constants
constexpr size_t FRAME_WIDTH = 1920;
constexpr size_t FRAME_HEIGHT = 1080;
constexpr size_t FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * 3 / 2; // NV12
constexpr size_t QUEUE_SIZE = 16;
constexpr size_t POOL_SIZE = 16;

// Data Structures
struct TrackingResult {
    std::array<float, 63> landmarks; // 21 points * 3 coords (x, y, z)
    bool vipLocked;
    std::chrono::steady_clock::time_point timestamp;
};

// Type Aliases
using AppFramePool = FramePool<POOL_SIZE, FRAME_SIZE>;
using AppProcessingQueue = SpscQueue<Frame*, QUEUE_SIZE>;
using OscQueue = SpscQueue<TrackingResult, QUEUE_SIZE>;

} // namespace core

