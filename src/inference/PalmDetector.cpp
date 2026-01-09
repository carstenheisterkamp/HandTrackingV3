/**
 * V3 Palm Detector Implementation
 *
 * MediaPipe Palm Detection model inference on Jetson via TensorRT.
 * Handles NV12 → RGB preprocessing and anchor-based decoding.
 */

#include "inference/PalmDetector.hpp"
#include "core/Logger.hpp"

#include <algorithm>
#include <cmath>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <nppi_color_conversion.h>
#include <nppi_geometry_transforms.h>
#endif

namespace inference {

PalmDetector::PalmDetector() = default;

PalmDetector::~PalmDetector() {
#ifdef ENABLE_CUDA
    if (d_nv12_) cudaFree(d_nv12_);
    if (d_rgb_) cudaFree(d_rgb_);
    if (d_resized_) cudaFree(d_resized_);
#endif
}

bool PalmDetector::init(const Config& config) {
    config_ = config;

    // Initialize TensorRT engine
    engine_ = std::make_unique<TensorRTEngine>();

    TensorRTEngine::Config trtConfig;
    trtConfig.modelPath = config.modelPath;
    trtConfig.fp16 = true;

    if (!engine_->load(trtConfig)) {
        core::Logger::error("PalmDetector: Failed to load TensorRT engine");
        return false;
    }

    // Allocate output buffer
    outputBuffer_.resize(engine_->getOutputInfo().size);

    // Allocate input buffer (3 x H x W)
    inputBuffer_.resize(3 * config_.inputWidth * config_.inputHeight);

    // Generate anchors for decoding
    generateAnchors();

#ifdef ENABLE_CUDA
    // Allocate GPU buffers for preprocessing
    // NV12: 1.5 * W * H bytes
    // RGB: 3 * W * H bytes
    size_t maxFrameSize = 1920 * 1080;  // Max input frame
    cudaMalloc(&d_nv12_, maxFrameSize * 3 / 2);
    cudaMalloc(&d_rgb_, maxFrameSize * 3);
    cudaMalloc(&d_resized_, config_.inputWidth * config_.inputHeight * 3 * sizeof(float));
#endif

    initialized_ = true;
    core::Logger::info("PalmDetector initialized");
    core::Logger::info("  Input: ", config_.inputWidth, "x", config_.inputHeight);
    core::Logger::info("  Anchors: ", anchors_.size());

    return true;
}

void PalmDetector::generateAnchors() {
    // MediaPipe Palm Detection uses SSD-style anchors
    // Two feature map layers: 24x24 and 12x12

    struct AnchorConfig {
        int gridSize;
        int numAnchors;
        float scale;
    };

    std::vector<AnchorConfig> configs = {
        {24, 2, 0.1f},   // Layer 1: 24x24, 2 anchors per cell
        {12, 6, 0.2f}    // Layer 2: 12x12, 6 anchors per cell
    };

    anchors_.clear();

    for (const auto& cfg : configs) {
        float step = 1.0f / cfg.gridSize;
        for (int y = 0; y < cfg.gridSize; ++y) {
            for (int x = 0; x < cfg.gridSize; ++x) {
                for (int a = 0; a < cfg.numAnchors; ++a) {
                    float cx = (x + 0.5f) * step;
                    float cy = (y + 0.5f) * step;
                    anchors_.push_back({cx, cy, cfg.scale, cfg.scale});
                }
            }
        }
    }

    core::Logger::info("Generated ", anchors_.size(), " anchors");
}

std::optional<PalmDetector::Detection> PalmDetector::detect(
    const uint8_t* nv12Data, int frameWidth, int frameHeight) {

    if (!initialized_) {
        core::Logger::error("PalmDetector not initialized");
        return std::nullopt;
    }

    // Preprocess: NV12 → RGB → Resize → Normalize
    preprocessNV12(nv12Data, frameWidth, frameHeight);

    // Run inference
    if (!engine_->infer(inputBuffer_.data(), outputBuffer_.data())) {
        core::Logger::error("PalmDetector inference failed");
        return std::nullopt;
    }

    // Decode output
    auto detections = decodeOutput(outputBuffer_.data());

    if (detections.empty()) {
        return std::nullopt;
    }

    // NMS and get best detection
    auto best = nms(detections);

    // Unletterbox coordinates to original frame
    unletterbox(best, frameWidth, frameHeight);

    return best;
}

std::optional<PalmDetector::Detection> PalmDetector::detectFromRGB(const float* rgbData) {
    if (!initialized_) {
        return std::nullopt;
    }

    // Direct inference on preprocessed RGB
    if (!engine_->infer(rgbData, outputBuffer_.data())) {
        return std::nullopt;
    }

    auto detections = decodeOutput(outputBuffer_.data());
    if (detections.empty()) {
        return std::nullopt;
    }

    return nms(detections);
}

void PalmDetector::preprocessNV12(const uint8_t* nv12Data, int width, int height) {
#ifdef ENABLE_CUDA
    // Upload NV12 to GPU
    size_t nv12Size = width * height * 3 / 2;
    cudaMemcpy(d_nv12_, nv12Data, nv12Size, cudaMemcpyHostToDevice);

    // NV12 to RGB using NPP
    NppiSize srcSize = {width, height};
    const Npp8u* pSrc[2] = {
        static_cast<const Npp8u*>(d_nv12_),
        static_cast<const Npp8u*>(d_nv12_) + width * height
    };

    nppiNV12ToRGB_8u_P2C3R(pSrc, width,
                           static_cast<Npp8u*>(d_rgb_), width * 3, srcSize);

    // Resize with letterboxing
    // For simplicity, we do letterbox on CPU for now
    // TODO: Implement GPU letterbox resize

    // Copy RGB to host for now (will optimize later)
    std::vector<uint8_t> rgbHost(width * height * 3);
    cudaMemcpy(rgbHost.data(), d_rgb_, rgbHost.size(), cudaMemcpyDeviceToHost);

    // CPU letterbox resize and normalize
    // Calculate scale and offsets
    float scale = std::min(
        static_cast<float>(config_.inputWidth) / width,
        static_cast<float>(config_.inputHeight) / height
    );

    int newW = static_cast<int>(width * scale);
    int newH = static_cast<int>(height * scale);
    int offX = (config_.inputWidth - newW) / 2;
    int offY = (config_.inputHeight - newH) / 2;

    // Fill with gray (128)
    std::fill(inputBuffer_.begin(), inputBuffer_.end(), 0.5f);

    // Simple bilinear resize + normalize to [-1, 1]
    for (int y = 0; y < newH; ++y) {
        for (int x = 0; x < newW; ++x) {
            float srcX = x / scale;
            float srcY = y / scale;

            int x0 = static_cast<int>(srcX);
            int y0 = static_cast<int>(srcY);

            if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
                int srcIdx = (y0 * width + x0) * 3;
                int dstX = x + offX;
                int dstY = y + offY;

                if (dstX >= 0 && dstX < config_.inputWidth &&
                    dstY >= 0 && dstY < config_.inputHeight) {

                    // CHW format, normalized to [0, 1]
                    int dstIdx = dstY * config_.inputWidth + dstX;
                    int planeSize = config_.inputWidth * config_.inputHeight;

                    inputBuffer_[0 * planeSize + dstIdx] = rgbHost[srcIdx + 0] / 255.0f;
                    inputBuffer_[1 * planeSize + dstIdx] = rgbHost[srcIdx + 1] / 255.0f;
                    inputBuffer_[2 * planeSize + dstIdx] = rgbHost[srcIdx + 2] / 255.0f;
                }
            }
        }
    }
#else
    core::Logger::error("CUDA required for preprocessing");
#endif
}

std::vector<PalmDetector::Detection> PalmDetector::decodeOutput(const float* output) {
    std::vector<Detection> detections;

    // MediaPipe Palm Detection output format:
    // For each anchor: [score, x_offset, y_offset, w, h, kp0_x, kp0_y, ..., kp6_x, kp6_y]
    // Total: 1 + 4 + 14 = 19 values per anchor

    const int stride = 19;

    for (size_t i = 0; i < anchors_.size(); ++i) {
        float rawScore = output[i * stride + 0];

        // Sigmoid to get probability
        float score = 1.0f / (1.0f + std::exp(-rawScore));

        if (score < config_.scoreThreshold) {
            continue;
        }

        Detection det;
        det.score = score;

        // Decode box (offsets from anchor)
        float dx = output[i * stride + 1];
        float dy = output[i * stride + 2];
        float dw = output[i * stride + 3];
        float dh = output[i * stride + 4];

        det.x = anchors_[i][0] + dx * anchors_[i][2];
        det.y = anchors_[i][1] + dy * anchors_[i][3];
        det.width = anchors_[i][2] * std::exp(dw);
        det.height = anchors_[i][3] * std::exp(dh);

        // Decode keypoints
        for (int k = 0; k < 7; ++k) {
            det.keypoints[k * 2 + 0] = anchors_[i][0] + output[i * stride + 5 + k * 2 + 0] * anchors_[i][2];
            det.keypoints[k * 2 + 1] = anchors_[i][1] + output[i * stride + 5 + k * 2 + 1] * anchors_[i][3];
        }

        // Calculate rotation from keypoints (wrist to middle finger)
        float kp0_x = det.keypoints[0];  // Wrist
        float kp0_y = det.keypoints[1];
        float kp2_x = det.keypoints[4];  // Middle finger base
        float kp2_y = det.keypoints[5];

        det.rotation = std::atan2(kp2_y - kp0_y, kp2_x - kp0_x);

        detections.push_back(det);
    }

    return detections;
}

PalmDetector::Detection PalmDetector::nms(const std::vector<Detection>& detections) {
    // Simple NMS: just return highest score for now (VIP single hand)
    auto best = std::max_element(detections.begin(), detections.end(),
        [](const Detection& a, const Detection& b) {
            return a.score < b.score;
        });

    return *best;
}

void PalmDetector::unletterbox(Detection& det, int origWidth, int origHeight) {
    float scale = std::min(
        static_cast<float>(config_.inputWidth) / origWidth,
        static_cast<float>(config_.inputHeight) / origHeight
    );

    int newW = static_cast<int>(origWidth * scale);
    int newH = static_cast<int>(origHeight * scale);
    float offX = (config_.inputWidth - newW) / 2.0f / config_.inputWidth;
    float offY = (config_.inputHeight - newH) / 2.0f / config_.inputHeight;
    float scaleX = static_cast<float>(newW) / config_.inputWidth;
    float scaleY = static_cast<float>(newH) / config_.inputHeight;

    // Transform center
    det.x = (det.x - offX) / scaleX;
    det.y = (det.y - offY) / scaleY;
    det.width /= scaleX;
    det.height /= scaleY;

    // Transform keypoints
    for (int k = 0; k < 7; ++k) {
        det.keypoints[k * 2 + 0] = (det.keypoints[k * 2 + 0] - offX) / scaleX;
        det.keypoints[k * 2 + 1] = (det.keypoints[k * 2 + 1] - offY) / scaleY;
    }

    // Clamp to [0, 1]
    det.x = std::clamp(det.x, 0.0f, 1.0f);
    det.y = std::clamp(det.y, 0.0f, 1.0f);
}

} // namespace inference

