#pragma once

#include <memory>
#include <cstdint>
#include <chrono>
#include <vector>
#include "core/MemoryUtils.hpp"

namespace core {

/**
 * Represents a video frame in the processing pipeline.
 * Uses aligned memory for GPU compatibility.
 */
struct Frame {
    // Pointer to aligned memory
    std::unique_ptr<uint8_t, AlignedDeleter> data;

    // Metadata
    size_t size = 0;
    size_t width = 0;
    size_t height = 0;
    uint32_t sequenceNum = 0;

    // NN Results (Landmarks)
    std::vector<float> nnData;

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

