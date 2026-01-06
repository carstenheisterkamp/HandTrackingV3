#pragma once

#include <memory>
#include <cstdint>
#include <chrono>
#include <vector>
#include "core/MemoryUtils.hpp"

// Forward declaration
namespace dai { class ImgFrame; }

namespace core {

/**
 * Represents a video frame in the processing pipeline.
 * Uses aligned memory for GPU compatibility.
 */
struct Frame {
    // Pinned Memory (owned by Frame)
    std::unique_ptr<uint8_t, AlignedDeleter> data; // RGB/NV12
    std::unique_ptr<uint8_t, AlignedDeleter> depthData; // Depth (UINT16) - computed on Jetson

    // Stereo frames for GPU-based depth computation
    std::unique_ptr<uint8_t, AlignedDeleter> monoLeftData;  // GRAY8
    std::unique_ptr<uint8_t, AlignedDeleter> monoRightData; // GRAY8

    // Optional: Keep reference if we were to support zero-copy from lib (not used with pinned copy)
    std::shared_ptr<dai::ImgFrame> daiFrame;


    // Metadata
    size_t size = 0;
    size_t depthSize = 0;
    size_t monoSize = 0;      // Size of mono frames (640*400)
    size_t width = 0;
    size_t height = 0;
    size_t depthWidth = 0;
    size_t depthHeight = 0;
    size_t monoWidth = 0;     // 640
    size_t monoHeight = 0;    // 400
    int type = 0; // dai::ImgFrame::Type
    uint32_t sequenceNum = 0;

    // NN Results (Landmarks)
    std::vector<float> nnData;
    std::vector<float> palmData; // Palm Detection results

    // Timestamps
    std::chrono::time_point<std::chrono::steady_clock> timestamp; // Host arrival time
    std::chrono::time_point<std::chrono::steady_clock> captureTimestamp; // Camera capture time (synced)

    Frame() = default;

    // Move-only
    Frame(Frame&&) = default;
    Frame& operator=(Frame&&) = default;
    Frame(const Frame&) = delete;
    Frame& operator=(const Frame&) = delete;
};

} // namespace core

