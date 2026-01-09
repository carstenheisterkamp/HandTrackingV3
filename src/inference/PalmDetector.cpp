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

    core::Logger::info("PalmDetector: TensorRT engine loaded");

    // Model has 2 outputs:
    // Output 0 (Identity): [1, 2016, 18] - box regressors
    // Output 1 (Identity_1): [1, 2016, 1] - scores
    auto& outputs = engine_->getOutputInfos();
    if (outputs.size() < 2) {
        core::Logger::error("PalmDetector: Expected 2 outputs, got ", outputs.size());
        return false;
    }

    core::Logger::info("  Output 0: ", outputs[0].name, " size=", outputs[0].size);
    core::Logger::info("  Output 1: ", outputs[1].name, " size=", outputs[1].size);

    // Allocate output buffers
    outputBuffer_.resize(outputs[0].size);  // [2016, 18] = 36288
    scoresBuffer_.resize(outputs[1].size);  // [2016, 1] = 2016
    core::Logger::info("  Boxes buffer: ", outputBuffer_.size(), " floats");
    core::Logger::info("  Scores buffer: ", scoresBuffer_.size(), " floats");

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
    // MediaPipe Palm Detection Lite uses 2016 anchors
    // Feature maps: 24x24 (2 anchors) + 12x12 (6 anchors) = 1152 + 864 = 2016
    // But actually it's: 48x48 (2 anchors) + 24x24 (6 anchors) = 4608 + 3456 = 8064?
    // Let's use the actual count: 2016 = 24x24x2 + 12x12x6 = 1152 + 864 = 2016 ✓

    struct AnchorConfig {
        int gridSize;
        int numAnchors;
    };

    // MediaPipe Palm Detection Lite anchor configuration for 192x192 input
    std::vector<AnchorConfig> configs = {
        {24, 2},   // Layer 1: 24x24, 2 anchors per cell = 1152
        {12, 6}    // Layer 2: 12x12, 6 anchors per cell = 864
    };                                                    // Total = 2016 ✓

    anchors_.clear();

    for (const auto& cfg : configs) {
        float step = 1.0f / static_cast<float>(cfg.gridSize);
        for (int y = 0; y < cfg.gridSize; ++y) {
            for (int x = 0; x < cfg.gridSize; ++x) {
                for (int a = 0; a < cfg.numAnchors; ++a) {
                    float cx = (static_cast<float>(x) + 0.5f) * step;
                    float cy = (static_cast<float>(y) + 0.5f) * step;
                    // Anchor size is 1.0 (normalized), offsets are relative
                    anchors_.push_back({cx, cy, 1.0f, 1.0f});
                }
            }
        }
    }

    core::Logger::info("Generated ", anchors_.size(), " anchors (expected 2016)");
}

std::optional<PalmDetector::Detection> PalmDetector::detect(
    const uint8_t* nv12Data, int frameWidth, int frameHeight) {

    if (!initialized_) {
        core::Logger::error("PalmDetector not initialized");
        return std::nullopt;
    }

    // Preprocess: NV12 → RGB → Resize → Normalize
    preprocessNV12(nv12Data, frameWidth, frameHeight);

    // Run inference with multiple outputs
    std::vector<void*> outputPtrs = {outputBuffer_.data(), scoresBuffer_.data()};
    if (!engine_->inferMultiOutput(inputBuffer_.data(), outputPtrs)) {
        core::Logger::error("PalmDetector inference failed");
        return std::nullopt;
    }

    // Debug: Log raw output stats (every 60 frames to find detections)
    static int debugCounter = 0;
    if (++debugCounter % 60 == 1) {
        // Analyze scores buffer
        float minScore = scoresBuffer_[0], maxScore = scoresBuffer_[0];
        int positiveScores = 0;

        for (size_t i = 0; i < scoresBuffer_.size(); ++i) {
            float rawScore = scoresBuffer_[i];
            float score = 1.0f / (1.0f + std::exp(-rawScore));  // Sigmoid
            minScore = std::min(minScore, score);
            maxScore = std::max(maxScore, score);
            if (score > 0.3f) positiveScores++;
        }

        core::Logger::info("🔍 PalmDetector output analysis:");
        core::Logger::info("   Boxes buffer: ", outputBuffer_.size(), " floats");
        core::Logger::info("   Scores buffer: ", scoresBuffer_.size(), " floats");
        core::Logger::info("   Score range (sigmoid): [", minScore, " to ", maxScore, "]");
        core::Logger::info("   Scores above 0.3: ", positiveScores, " / ", scoresBuffer_.size());
        core::Logger::info("   Threshold: ", config_.scoreThreshold);
    }

    // Decode output (boxes + scores are separate)
    auto detections = decodeOutput(outputBuffer_.data(), scoresBuffer_.data());

    // Debug: Log detection count
    if (debugCounter % 60 == 1) {
        core::Logger::info("   Detections found: ", detections.size());
    }

    if (detections.empty()) {
        return std::nullopt;
    }

    // NMS and get best detection
    auto best = nms(detections);

    // Unletterbox coordinates to original frame
    unletterbox(best, frameWidth, frameHeight);

    return best;
}

std::vector<PalmDetector::Detection> PalmDetector::detectAll(
    const uint8_t* nv12Data, int frameWidth, int frameHeight, int maxHands) {

    if (!initialized_) {
        core::Logger::error("PalmDetector not initialized");
        return {};
    }

    // Preprocess: NV12 → RGB → Resize → Normalize
    preprocessNV12(nv12Data, frameWidth, frameHeight);

    // Run inference with multiple outputs
    std::vector<void*> outputPtrs = {outputBuffer_.data(), scoresBuffer_.data()};
    if (!engine_->inferMultiOutput(inputBuffer_.data(), outputPtrs)) {
        core::Logger::error("PalmDetector inference failed");
        return {};
    }

    // Decode output
    auto detections = decodeOutput(outputBuffer_.data(), scoresBuffer_.data());

    // Debug: Log detection stats
    static int detectAllCounter = 0;
    if (++detectAllCounter % 60 == 1) {
        core::Logger::info("🔍 detectAll: raw detections=", detections.size());
    }

    if (detections.empty()) {
        return {};
    }

    // Multi-hand NMS
    auto results = nmsMulti(detections, maxHands);

    if (detectAllCounter % 60 == 1) {
        core::Logger::info("   After NMS: ", results.size(), " hands");
    }

    // Unletterbox all detections
    for (auto& det : results) {
        unletterbox(det, frameWidth, frameHeight);
    }

    return results;
}

std::optional<PalmDetector::Detection> PalmDetector::detectFromRGB(const float* rgbData) {
    if (!initialized_) {
        return std::nullopt;
    }

    // Direct inference on preprocessed RGB with multiple outputs
    std::vector<void*> outputPtrs = {outputBuffer_.data(), scoresBuffer_.data()};
    if (!engine_->inferMultiOutput(rgbData, outputPtrs)) {
        return std::nullopt;
    }

    auto detections = decodeOutput(outputBuffer_.data(), scoresBuffer_.data());
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

    // Simple bilinear resize + normalize to [0, 1]
    // Model expects NHWC format: [1, 192, 192, 3]
    for (int y = 0; y < newH; ++y) {
        for (int x = 0; x < newW; ++x) {
            float srcX = static_cast<float>(x) / scale;
            float srcY = static_cast<float>(y) / scale;

            int x0 = static_cast<int>(srcX);
            int y0 = static_cast<int>(srcY);

            if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
                int srcIdx = (y0 * width + x0) * 3;
                int dstX = x + offX;
                int dstY = y + offY;

                if (dstX >= 0 && dstX < config_.inputWidth &&
                    dstY >= 0 && dstY < config_.inputHeight) {

                    // NHWC format: [batch, height, width, channels]
                    int dstIdx = (dstY * config_.inputWidth + dstX) * 3;

                    inputBuffer_[dstIdx + 0] = static_cast<float>(rgbHost[srcIdx + 0]) / 255.0f;
                    inputBuffer_[dstIdx + 1] = static_cast<float>(rgbHost[srcIdx + 1]) / 255.0f;
                    inputBuffer_[dstIdx + 2] = static_cast<float>(rgbHost[srcIdx + 2]) / 255.0f;
                }
            }
        }
    }
#else
    core::Logger::error("CUDA required for preprocessing");
#endif
}

std::vector<PalmDetector::Detection> PalmDetector::decodeOutput(const float* boxes, const float* scores) {
    std::vector<Detection> detections;

    // MediaPipe Palm Detection Lite output format:
    // boxes: [2016, 18] - for each anchor: [x_center, y_center, w, h, kp0_x, kp0_y, ..., kp6_x, kp6_y]
    // scores: [2016, 1] - raw logit scores (need sigmoid)

    const int boxStride = 18;  // 4 box coords + 7 keypoints * 2 = 18

    for (size_t i = 0; i < anchors_.size() && i < scoresBuffer_.size(); ++i) {
        float rawScore = scores[i];

        // Sigmoid to get probability
        float score = 1.0f / (1.0f + std::exp(-rawScore));

        if (score < config_.scoreThreshold) {
            continue;
        }

        Detection det;
        det.score = score;

        // Box data starts at boxes[i * 18]
        const float* box = &boxes[i * boxStride];

        // Decode box: offsets are relative to anchor, scaled by 192 (input size)
        // MediaPipe uses: center_x = anchor_x + offset_x / 192
        float dx = box[0] / static_cast<float>(config_.inputWidth);
        float dy = box[1] / static_cast<float>(config_.inputHeight);
        float dw = box[2] / static_cast<float>(config_.inputWidth);
        float dh = box[3] / static_cast<float>(config_.inputHeight);

        det.x = anchors_[i][0] + dx;
        det.y = anchors_[i][1] + dy;
        det.width = dw;
        det.height = dh;

        // ═══════════════════════════════════════════════════════════
        // FALSE POSITIVE FILTER: Reject face-like detections
        // ═══════════════════════════════════════════════════════════

        // Filter 1: Aspect ratio - palms are roughly square
        // Very loose filter to avoid rejecting rotated hands
        float aspectRatio = (det.height > 0.001f) ? det.width / det.height : 1.0f;
        if (aspectRatio < 0.3f || aspectRatio > 3.0f) {
            // Debug: Log rejected detections
            static int rejectCount1 = 0;
            if (++rejectCount1 % 100 == 1) {
                core::Logger::info("🚫 Filter1 reject: aspect=", aspectRatio);
            }
            continue;  // Too elongated - likely not a palm
        }

        // Filter 2: Size sanity check - palm shouldn't be too small or too large
        float area = det.width * det.height;
        if (area < 0.002f || area > 0.8f) {
            static int rejectCount2 = 0;
            if (++rejectCount2 % 100 == 1) {
                core::Logger::info("🚫 Filter2 reject: area=", area);
            }
            continue;  // Unrealistic size
        }

        // Filter 3: Position check - faces are usually in upper third of frame
        // Only reject if BOTH upper position AND very low score
        if (det.y < 0.25f && det.score < 0.5f) {
            static int rejectCount3 = 0;
            if (++rejectCount3 % 100 == 1) {
                core::Logger::info("🚫 Filter3 reject: y=", det.y, " score=", det.score);
            }
            continue;  // Likely face in upper frame with weak score
        }

        // Decode keypoints (7 keypoints, each x,y relative to anchor)
        for (int k = 0; k < 7; ++k) {
            float kpx = box[4 + k * 2] / static_cast<float>(config_.inputWidth);
            float kpy = box[4 + k * 2 + 1] / static_cast<float>(config_.inputHeight);
            det.keypoints[k * 2 + 0] = anchors_[i][0] + kpx;
            det.keypoints[k * 2 + 1] = anchors_[i][1] + kpy;
        }

        // ═══════════════════════════════════════════════════════════
        // Filter 4: Keypoint consistency check for palm detection
        // Real palms have a specific keypoint pattern:
        // - KP0 = wrist, KP2 = middle finger base
        // - These should be separated by a reasonable distance
        // - Faces have random/clustered keypoints
        // ═══════════════════════════════════════════════════════════
        float kp0_x = det.keypoints[0];  // Wrist
        float kp0_y = det.keypoints[1];
        float kp2_x = det.keypoints[4];  // Middle finger base
        float kp2_y = det.keypoints[5];

        float keypointDist = std::sqrt(
            (kp2_x - kp0_x) * (kp2_x - kp0_x) +
            (kp2_y - kp0_y) * (kp2_y - kp0_y)
        );

        // Keypoint distance should be proportional to palm size
        // For a real palm, wrist-to-middle-base is roughly 30-60% of palm width
        float expectedMinDist = det.width * 0.15f;
        float expectedMaxDist = det.width * 1.5f;

        if (keypointDist < expectedMinDist || keypointDist > expectedMaxDist) {
            static int rejectCount4 = 0;
            if (++rejectCount4 % 100 == 1) {
                core::Logger::info("🚫 Filter4 reject: kp_dist=", keypointDist,
                                   " expected=[", expectedMinDist, ", ", expectedMaxDist, "]");
            }
            continue;  // Keypoints don't match palm pattern
        }

        // Calculate rotation from keypoints (wrist to middle finger base)
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

float PalmDetector::computeIoU(const Detection& a, const Detection& b) {
    // Convert center-size to corners
    float a_x1 = a.x - a.width / 2.0f;
    float a_y1 = a.y - a.height / 2.0f;
    float a_x2 = a.x + a.width / 2.0f;
    float a_y2 = a.y + a.height / 2.0f;

    float b_x1 = b.x - b.width / 2.0f;
    float b_y1 = b.y - b.height / 2.0f;
    float b_x2 = b.x + b.width / 2.0f;
    float b_y2 = b.y + b.height / 2.0f;

    // Intersection
    float inter_x1 = std::max(a_x1, b_x1);
    float inter_y1 = std::max(a_y1, b_y1);
    float inter_x2 = std::min(a_x2, b_x2);
    float inter_y2 = std::min(a_y2, b_y2);

    float interWidth = std::max(0.0f, inter_x2 - inter_x1);
    float interHeight = std::max(0.0f, inter_y2 - inter_y1);
    float interArea = interWidth * interHeight;

    // Union
    float a_area = a.width * a.height;
    float b_area = b.width * b.height;
    float unionArea = a_area + b_area - interArea;

    if (unionArea <= 0.0f) return 0.0f;
    return interArea / unionArea;
}

std::vector<PalmDetector::Detection> PalmDetector::nmsMulti(
    const std::vector<Detection>& detections, int maxHands) {

    if (detections.empty()) return {};

    // Sort by score descending
    std::vector<Detection> sorted = detections;
    std::sort(sorted.begin(), sorted.end(),
        [](const Detection& a, const Detection& b) {
            return a.score > b.score;
        });

    std::vector<Detection> results;
    std::vector<bool> suppressed(sorted.size(), false);

    for (size_t i = 0; i < sorted.size() && results.size() < static_cast<size_t>(maxHands); ++i) {
        if (suppressed[i]) continue;

        results.push_back(sorted[i]);

        // Suppress overlapping detections
        for (size_t j = i + 1; j < sorted.size(); ++j) {
            if (suppressed[j]) continue;

            float iou = computeIoU(sorted[i], sorted[j]);
            if (iou > config_.nmsThreshold) {
                suppressed[j] = true;
            }
        }
    }

    return results;
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

