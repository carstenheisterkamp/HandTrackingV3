#pragma once

#include "inference/TensorRTEngine.hpp"
#include <optional>
#include <array>
#include <opencv2/objdetect.hpp>

namespace inference {

/**
 * V3 Palm Detector - MediaPipe Palm Detection on Jetson
 *
 * Uses TensorRT for inference.
 * Input: NV12 frame from OAK-D
 * Output: Single best palm detection (VIP)
 *
 * Face Removal: Uses Haar Cascade to detect and exclude face regions
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
        float scoreThreshold = 0.75f;  // Increased to 0.75 to strongly reduce false positives
        float nmsThreshold = 0.4f;
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
     * Detect ALL palms in NV12 frame (up to maxHands)
     * @param nv12Data NV12 frame data
     * @param frameWidth Original frame width
     * @param frameHeight Original frame height
     * @param maxHands Maximum number of hands to return (default 2)
     * @return Vector of detections, sorted by score (best first)
     */
    std::vector<Detection> detectAll(const uint8_t* nv12Data,
                                     int frameWidth, int frameHeight,
                                     int maxHands = 2);

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
    std::vector<Detection> nmsMulti(const std::vector<Detection>& detections, int maxHands);
    void unletterbox(Detection& det, int origWidth, int origHeight);
    float computeIoU(const Detection& a, const Detection& b);

    // Face detection for removing false positives
    cv::CascadeClassifier faceDetector_;
    bool faceDetectorLoaded_ = false;
    std::vector<cv::Rect> lastFaceRects_;  // Cache face rects for current frame
    int lastFaceFrameWidth_ = 0;
    int lastFaceFrameHeight_ = 0;

    void detectFaces(const uint8_t* nv12Data, int width, int height);
    bool isInFaceRegion(float x, float y, float w, float h) const;
};

} // namespace inference

