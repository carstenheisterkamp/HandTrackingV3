#pragma once

#include "inference/TensorRTEngine.hpp"
#include "inference/PalmDetector.hpp"
#include "core/Types.hpp"
#include <optional>
#include <array>

namespace inference {

/**
 * V3 Hand Landmark - MediaPipe Hand Landmark on Jetson
 *
 * Uses TensorRT for inference.
 * Input: Cropped hand ROI based on palm detection
 * Output: 21 3D landmarks + handedness + presence score
 */
class HandLandmark {
public:
    struct Result {
        std::array<core::TrackingResult::NormalizedPoint, 21> landmarks;
        float handedness;       // 0 = Left, 1 = Right
        float presence;         // Hand presence confidence
        float palmCenterX;      // Palm center in original frame coords
        float palmCenterY;
    };

    struct Config {
        std::string modelPath = "models/hand_landmark.onnx";
        int inputWidth = 224;   // MediaPipe uses 224x224
        int inputHeight = 224;
        float presenceThreshold = 0.5f;
    };

    HandLandmark();
    ~HandLandmark();

    /**
     * Initialize with config
     */
    bool init(const Config& config);

    /**
     * Infer landmarks from NV12 frame using palm detection
     * @param nv12Data Full frame in NV12 format
     * @param frameWidth Frame width
     * @param frameHeight Frame height
     * @param palm Palm detection result (defines ROI)
     * @return Landmarks or nullopt if hand not present
     */
    std::optional<Result> infer(const uint8_t* nv12Data,
                                int frameWidth, int frameHeight,
                                const PalmDetector::Detection& palm);

    /**
     * Check if initialized
     */
    [[nodiscard]] bool isInitialized() const { return initialized_; }

private:
    Config config_;
    bool initialized_ = false;

    std::unique_ptr<TensorRTEngine> engine_;

    // Input/output buffers
    std::vector<float> inputBuffer_;
    std::vector<float> outputBuffer_;

    // ROI extraction parameters
    float lastRotation_ = 0.0f;
    float lastScale_ = 1.0f;

    // Helpers
    void extractROI(const uint8_t* nv12Data, int frameWidth, int frameHeight,
                    const PalmDetector::Detection& palm);
    Result parseOutput(const float* output,
                       const PalmDetector::Detection& palm,
                       int frameWidth, int frameHeight);
    void transformToFrameCoords(Result& result,
                                const PalmDetector::Detection& palm,
                                int frameWidth, int frameHeight);
};

} // namespace inference

