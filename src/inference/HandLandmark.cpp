/**
 * V3 Hand Landmark Implementation
 *
 * MediaPipe Hand Landmark model inference on Jetson via TensorRT.
 * Extracts ROI based on palm detection, runs landmark inference.
 */

#include "inference/HandLandmark.hpp"
#include "core/Logger.hpp"

#include <algorithm>
#include <cmath>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace inference {

HandLandmark::HandLandmark() = default;

HandLandmark::~HandLandmark() = default;

bool HandLandmark::init(const Config& config) {
    config_ = config;

    // Initialize TensorRT engine
    engine_ = std::make_unique<TensorRTEngine>();

    TensorRTEngine::Config trtConfig;
    trtConfig.modelPath = config.modelPath;
    trtConfig.fp16 = true;

    if (!engine_->load(trtConfig)) {
        core::Logger::error("HandLandmark: Failed to load TensorRT engine");
        return false;
    }

    // Allocate buffers
    inputBuffer_.resize(3 * config_.inputWidth * config_.inputHeight);
    outputBuffer_.resize(engine_->getOutputInfo().size);

    initialized_ = true;
    core::Logger::info("HandLandmark initialized");
    core::Logger::info("  Input: ", config_.inputWidth, "x", config_.inputHeight);
    core::Logger::info("  Output size: ", engine_->getOutputInfo().size);

    return true;
}

std::optional<HandLandmark::Result> HandLandmark::infer(
    const uint8_t* nv12Data,
    int frameWidth, int frameHeight,
    const PalmDetector::Detection& palm) {

    if (!initialized_) {
        core::Logger::error("HandLandmark not initialized");
        return std::nullopt;
    }

    // Extract ROI based on palm detection
    extractROI(nv12Data, frameWidth, frameHeight, palm);

    // Run inference
    if (!engine_->infer(inputBuffer_.data(), outputBuffer_.data())) {
        core::Logger::error("HandLandmark inference failed");
        return std::nullopt;
    }

    // Parse output
    auto result = parseOutput(outputBuffer_.data(), palm, frameWidth, frameHeight);

    // Check presence
    if (result.presence < config_.presenceThreshold) {
        return std::nullopt;
    }

    // Transform landmarks to frame coordinates
    transformToFrameCoords(result, palm, frameWidth, frameHeight);

    return result;
}

void HandLandmark::extractROI(const uint8_t* nv12Data,
                               int frameWidth, int frameHeight,
                               const PalmDetector::Detection& palm) {
    // Calculate ROI from palm detection
    // Expand palm box by 2.5x for full hand
    float scale = 2.5f;
    float roiSize = std::max(palm.width, palm.height) * scale;

    float roiCenterX = palm.x;
    float roiCenterY = palm.y;

    // Account for rotation
    float rotation = palm.rotation;
    lastRotation_ = rotation;
    lastScale_ = roiSize;

    // Calculate ROI corners in frame coordinates
    float halfSize = roiSize / 2.0f;
    float x1 = (roiCenterX - halfSize) * frameWidth;
    float y1 = (roiCenterY - halfSize) * frameHeight;
    float x2 = (roiCenterX + halfSize) * frameWidth;
    float y2 = (roiCenterY + halfSize) * frameHeight;

    // Clamp to frame bounds
    x1 = std::clamp(x1, 0.0f, static_cast<float>(frameWidth - 1));
    y1 = std::clamp(y1, 0.0f, static_cast<float>(frameHeight - 1));
    x2 = std::clamp(x2, 0.0f, static_cast<float>(frameWidth - 1));
    y2 = std::clamp(y2, 0.0f, static_cast<float>(frameHeight - 1));

    int roiX = static_cast<int>(x1);
    int roiY = static_cast<int>(y1);
    int roiW = static_cast<int>(x2 - x1);
    int roiH = static_cast<int>(y2 - y1);

    if (roiW <= 0 || roiH <= 0) {
        // Invalid ROI, fill with zeros
        std::fill(inputBuffer_.begin(), inputBuffer_.end(), 0.0f);
        return;
    }

    // Extract and resize ROI
    // For now, simple CPU extraction (will optimize with CUDA later)

    // Convert NV12 to RGB for the ROI region
    // NV12 format: Y plane (W*H) followed by interleaved UV plane (W*H/2)

    int planeSize = config_.inputWidth * config_.inputHeight;

    for (int y = 0; y < config_.inputHeight; ++y) {
        for (int x = 0; x < config_.inputWidth; ++x) {
            // Map to ROI coordinates
            float srcX = roiX + (x / static_cast<float>(config_.inputWidth)) * roiW;
            float srcY = roiY + (y / static_cast<float>(config_.inputHeight)) * roiH;

            int sx = static_cast<int>(srcX);
            int sy = static_cast<int>(srcY);

            if (sx < 0 || sx >= frameWidth || sy < 0 || sy >= frameHeight) {
                inputBuffer_[0 * planeSize + y * config_.inputWidth + x] = 0.5f;
                inputBuffer_[1 * planeSize + y * config_.inputWidth + x] = 0.5f;
                inputBuffer_[2 * planeSize + y * config_.inputWidth + x] = 0.5f;
                continue;
            }

            // Get Y value
            uint8_t Y = nv12Data[sy * frameWidth + sx];

            // Get UV values (subsampled 2x2)
            int uvY = sy / 2;
            int uvX = (sx / 2) * 2;  // UV is interleaved
            int uvOffset = frameWidth * frameHeight + uvY * frameWidth + uvX;

            uint8_t U = nv12Data[uvOffset];
            uint8_t V = nv12Data[uvOffset + 1];

            // YUV to RGB conversion
            int C = Y - 16;
            int D = U - 128;
            int E = V - 128;

            int R = std::clamp((298 * C + 409 * E + 128) >> 8, 0, 255);
            int G = std::clamp((298 * C - 100 * D - 208 * E + 128) >> 8, 0, 255);
            int B = std::clamp((298 * C + 516 * D + 128) >> 8, 0, 255);

            // Normalize to [0, 1] and store in CHW format
            int idx = y * config_.inputWidth + x;
            inputBuffer_[0 * planeSize + idx] = R / 255.0f;
            inputBuffer_[1 * planeSize + idx] = G / 255.0f;
            inputBuffer_[2 * planeSize + idx] = B / 255.0f;
        }
    }
}

HandLandmark::Result HandLandmark::parseOutput(
    const float* output,
    const PalmDetector::Detection& palm,
    int frameWidth, int frameHeight) {

    Result result;

    // MediaPipe Hand Landmark output format:
    // - 21 landmarks × 3 coords = 63 floats (in pixel coords of 224x224)
    // - Handedness: 1 float
    // - Presence: 1 float

    // Parse landmarks (in ROI coordinates, 0-224)
    for (int i = 0; i < 21; ++i) {
        // Landmarks are in pixel coordinates of the input image (224x224)
        float x = output[i * 3 + 0] / config_.inputWidth;   // Normalize to 0-1
        float y = output[i * 3 + 1] / config_.inputHeight;
        float z = output[i * 3 + 2];  // Relative depth

        result.landmarks[i].x = x;
        result.landmarks[i].y = y;
        result.landmarks[i].z = z;
    }

    // Parse handedness (index 63)
    result.handedness = output[63];

    // Parse presence (usually sigmoid of raw score)
    float rawPresence = output[64];
    result.presence = 1.0f / (1.0f + std::exp(-rawPresence));

    // Calculate palm center from landmarks (average of wrist and middle finger base)
    result.palmCenterX = (result.landmarks[0].x + result.landmarks[9].x) / 2.0f;
    result.palmCenterY = (result.landmarks[0].y + result.landmarks[9].y) / 2.0f;

    return result;
}

void HandLandmark::transformToFrameCoords(
    Result& result,
    const PalmDetector::Detection& palm,
    int frameWidth, int frameHeight) {

    // Transform from ROI coordinates (0-1) to frame coordinates (0-1)
    float scale = 2.5f;  // Same as in extractROI
    float roiSize = std::max(palm.width, palm.height) * scale;
    float halfSize = roiSize / 2.0f;

    for (int i = 0; i < 21; ++i) {
        // Map from ROI [0,1] to frame coordinates
        float roiX = result.landmarks[i].x;
        float roiY = result.landmarks[i].y;

        // Transform to frame coordinates
        result.landmarks[i].x = palm.x - halfSize + roiX * roiSize;
        result.landmarks[i].y = palm.y - halfSize + roiY * roiSize;

        // Clamp to [0, 1]
        result.landmarks[i].x = std::clamp(result.landmarks[i].x, 0.0f, 1.0f);
        result.landmarks[i].y = std::clamp(result.landmarks[i].y, 0.0f, 1.0f);
    }

    // Update palm center
    result.palmCenterX = palm.x;
    result.palmCenterY = palm.y;
}

} // namespace inference

