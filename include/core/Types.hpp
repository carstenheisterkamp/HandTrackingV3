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

// Stereo frames for GPU-based depth (computed on Jetson, not OAK-D)
constexpr size_t MONO_WIDTH = 640;
constexpr size_t MONO_HEIGHT = 400;
constexpr size_t MONO_FRAME_SIZE = MONO_WIDTH * MONO_HEIGHT; // GRAY8
constexpr size_t DEPTH_FRAME_SIZE = MONO_WIDTH * MONO_HEIGHT * 2; // UINT16 depth map

constexpr size_t QUEUE_SIZE = 16;
constexpr size_t POOL_SIZE = 16;

// Data Structures
struct TrackingResult {
    struct NormalizedPoint {
        float x, y, z;
    };

    std::vector<NormalizedPoint> landmarks; // 21 points * 3 coords
    bool vipLocked;
    std::chrono::steady_clock::time_point timestamp;

    // New fields for gestures
    float pinchDistance = 0.0f;
    int gestureId = 0; // 0=None, 1=Pinch, 2=Fist
    std::string gestureName = "unknown"; // For logging/debug
};

// Type Aliases
using AppFramePool = FramePool<POOL_SIZE, FRAME_SIZE, DEPTH_FRAME_SIZE, MONO_FRAME_SIZE>;
using AppProcessingQueue = SpscQueue<Frame*, QUEUE_SIZE>;
using OscQueue = SpscQueue<TrackingResult, QUEUE_SIZE>;

} // namespace core

