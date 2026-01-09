#pragma once

#include "inference/TensorRTEngine.hpp"
#include <optional>
#include <array>

namespace inference {

/**
 * V3 Palm Detector - MediaPipe Palm Detection on Jetson
 *
 * Uses TensorRT for inference.
 * Input: NV12 frame from OAK-D
 * Output: Single best palm detection (VIP)
 */
class PalmDetector {
public:
    struct Detection {
        float x, y;             // Center (normalized 0-1)
        float width, height;    // Size (normalized)
        float score;            // Confidence
        float rotation;         // Hand rotation in radians

        // Keypoints for ROI extraction
        std::array<float, 14> keypoints;  // 7 keypoints × 2 coords
    };

    struct Config {
        std::string modelPath = "models/palm_detection.onnx";
        int inputWidth = 192;
        int inputHeight = 192;
        float scoreThreshold = 0.3f;  // Lowered from 0.5 for better detection
        float nmsThreshold = 0.3f;
    };

    PalmDetector();
    ~PalmDetector();

    /**
     * Initialize with config
     */
    bool init(const Config& config);

    /**
     * Detect palm in NV12 frame
     * @param nv12Data NV12 frame data
     * @param frameWidth Original frame width
     * @param frameHeight Original frame height
     * @return Best detection or nullopt if none found
     */
    std::optional<Detection> detect(const uint8_t* nv12Data,
                                    int frameWidth, int frameHeight);

    /**
     * Detect palm from RGB float buffer (already preprocessed)
     */
    std::optional<Detection> detectFromRGB(const float* rgbData);

    /**
     * Check if initialized
     */
    [[nodiscard]] bool isInitialized() const { return initialized_; }

private:
    Config config_;
    bool initialized_ = false;

    std::unique_ptr<TensorRTEngine> engine_;

    // Preprocessing buffers (GPU)
    void* d_nv12_ = nullptr;
    void* d_rgb_ = nullptr;
    void* d_resized_ = nullptr;

    // Output buffers - Model has 2 outputs:
    // outputBuffer_: [2016, 18] - box regressors (x, y, w, h, + 14 keypoint coords)
    // scoresBuffer_: [2016, 1] - confidence scores
    std::vector<float> outputBuffer_;   // Boxes: 2016 * 18 = 36288
    std::vector<float> scoresBuffer_;   // Scores: 2016 * 1 = 2016

    // Input buffer (host, for preprocessing)
    std::vector<float> inputBuffer_;

    // Anchor boxes for decoding
    std::vector<std::array<float, 4>> anchors_;

    // Helpers
    void generateAnchors();
    void preprocessNV12(const uint8_t* nv12Data, int width, int height);
    std::vector<Detection> decodeOutput(const float* boxes, const float* scores);
    Detection nms(const std::vector<Detection>& detections);
    void unletterbox(Detection& det, int origWidth, int origHeight);
};

} // namespace inference

