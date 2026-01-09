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
 * V3 Frame - Simplified for Sensor-Only Pipeline
 *
 * Contains:
 * - RGB data (NV12 from OAK-D)
 * - Mono L/R data (optional, for stereo depth)
 * - Metadata (timestamps, dimensions)
 *
 * NN results are computed on Jetson (TensorRT), not stored here.
 */
struct Frame {
    // ═══════════════════════════════════════════════════════════
    // Pinned Memory Buffers
    // ═══════════════════════════════════════════════════════════

    // RGB Frame (NV12 format)
    std::unique_ptr<uint8_t, AlignedDeleter> data;

    // Depth Map (computed on Jetson, UINT16)
    std::unique_ptr<uint8_t, AlignedDeleter> depthData;

    // Stereo frames for GPU-based depth computation (Phase 3)
    std::unique_ptr<uint8_t, AlignedDeleter> monoLeftData;   // GRAY8
    std::unique_ptr<uint8_t, AlignedDeleter> monoRightData;  // GRAY8

    // Optional: Keep reference to original frame (for debugging)
    std::shared_ptr<dai::ImgFrame> daiFrame;

    // ═══════════════════════════════════════════════════════════
    // Size Information
    // ═══════════════════════════════════════════════════════════

    size_t size = 0;           // RGB buffer size
    size_t depthSize = 0;      // Depth buffer size
    size_t monoSize = 0;       // Mono buffer size (640*400)

    size_t width = 0;          // RGB width
    size_t height = 0;         // RGB height
    size_t depthWidth = 0;     // Depth width
    size_t depthHeight = 0;    // Depth height
    size_t monoWidth = 0;      // Mono width (640)
    size_t monoHeight = 0;     // Mono height (400)

    bool hasStereoData = false; // True when monoLeft/Right are valid
    int type = 0;              // dai::ImgFrame::Type (22 = NV12)
    uint32_t sequenceNum = 0;

    // ═══════════════════════════════════════════════════════════
    // Timestamps
    // ═══════════════════════════════════════════════════════════

    std::chrono::time_point<std::chrono::steady_clock> timestamp;        // Host arrival
    std::chrono::time_point<std::chrono::steady_clock> captureTimestamp; // Camera capture

    // ═══════════════════════════════════════════════════════════
    // Legacy fields (kept for compatibility, will be removed)
    // ═══════════════════════════════════════════════════════════
    std::vector<float> nnData;    // DEPRECATED: V3 NNs run on Jetson
    std::vector<float> palmData;  // DEPRECATED: V3 NNs run on Jetson

    Frame() = default;
    Frame(Frame&&) = default;
    Frame& operator=(Frame&&) = default;
    Frame(const Frame&) = delete;
    Frame& operator=(const Frame&) = delete;
};

} // namespace core

