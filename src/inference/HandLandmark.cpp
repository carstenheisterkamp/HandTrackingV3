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
    // Model has 4 outputs:
    // Output 0 (Identity): [1, 63] - landmarks (21 × 3)
    // Output 1 (Identity_1): [1, 1] - handedness
    // Output 2 (Identity_2): [1, 1] - presence
    // Output 3 (Identity_3): [1, 63] - world landmarks
    auto& outputs = engine_->getOutputInfos();
    core::Logger::info("HandLandmark: ", outputs.size(), " outputs");
    for (size_t i = 0; i < outputs.size(); ++i) {
        core::Logger::info("  Output ", i, ": ", outputs[i].name, " size=", outputs[i].size);
    }

    // Allocate buffers - NHWC format for input
    inputBuffer_.resize(config_.inputWidth * config_.inputHeight * 3);  // HWC

    // Allocate output buffers
    landmarksBuffer_.resize(63);    // 21 landmarks × 3 coords
    handednessBuffer_.resize(1);
    presenceBuffer_.resize(1);
    worldLandmarksBuffer_.resize(63);

    initialized_ = true;
    core::Logger::info("HandLandmark initialized");
    core::Logger::info("  Input: ", config_.inputWidth, "x", config_.inputHeight, " (NHWC)");

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

    // Run inference with multiple outputs
    std::vector<void*> outputPtrs = {
        landmarksBuffer_.data(),
        handednessBuffer_.data(),
        presenceBuffer_.data(),
        worldLandmarksBuffer_.data()
    };

    if (!engine_->inferMultiOutput(inputBuffer_.data(), outputPtrs)) {
        core::Logger::error("HandLandmark inference failed");
        return std::nullopt;
    }

    // Parse output
    auto result = parseOutput(palm, frameWidth, frameHeight);

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

    for (int y = 0; y < config_.inputHeight; ++y) {
        for (int x = 0; x < config_.inputWidth; ++x) {
            // Map to ROI coordinates
            float srcX = roiX + (x / static_cast<float>(config_.inputWidth)) * roiW;
            float srcY = roiY + (y / static_cast<float>(config_.inputHeight)) * roiH;

            int sx = static_cast<int>(srcX);
            int sy = static_cast<int>(srcY);

            // NHWC format: index = (y * W + x) * 3 + channel
            int idx = (y * config_.inputWidth + x) * 3;

            if (sx < 0 || sx >= frameWidth || sy < 0 || sy >= frameHeight) {
                inputBuffer_[idx + 0] = 0.5f;
                inputBuffer_[idx + 1] = 0.5f;
                inputBuffer_[idx + 2] = 0.5f;
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

            // Normalize to [0, 1] and store in NHWC format (H, W, C)
            // idx already computed above
            inputBuffer_[idx + 0] = R / 255.0f;
            inputBuffer_[idx + 1] = G / 255.0f;
            inputBuffer_[idx + 2] = B / 255.0f;
        }
    }
}

HandLandmark::Result HandLandmark::parseOutput(
    const PalmDetector::Detection& palm,
    int frameWidth, int frameHeight) {

    Result result;

    // Model outputs:
    // landmarksBuffer_: [63] - 21 landmarks × 3 coords (x, y, z in pixel coords of 224x224)
    // handednessBuffer_: [1] - left/right hand probability
    // presenceBuffer_: [1] - hand presence raw score
    // worldLandmarksBuffer_: [63] - world coordinates

    // Parse landmarks (in ROI coordinates, 0-224)
    for (int i = 0; i < 21; ++i) {
        // Landmarks are in pixel coordinates of the input image (224x224)
        float x = landmarksBuffer_[i * 3 + 0] / static_cast<float>(config_.inputWidth);   // Normalize to 0-1
        float y = landmarksBuffer_[i * 3 + 1] / static_cast<float>(config_.inputHeight);
        float z = landmarksBuffer_[i * 3 + 2];  // Relative depth

        result.landmarks[i].x = x;
        result.landmarks[i].y = y;
        result.landmarks[i].z = z;
    }

    // Parse handedness
    result.handedness = handednessBuffer_[0];

    // Parse presence (sigmoid of raw score)
    float rawPresence = presenceBuffer_[0];
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

