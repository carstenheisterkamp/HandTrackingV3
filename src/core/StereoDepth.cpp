#include "core/StereoDepth.hpp"
#include "core/Logger.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace core {

// Internal structure for rectification maps
struct StereoDepth::RectifyMaps {
    // For now, we use a simplified approach without full rectification
    // Full implementation would use OpenCV remap tables
    bool valid = false;
};

StereoDepth::StereoDepth()
    : rectMaps_(std::make_unique<RectifyMaps>()) {
}

StereoDepth::~StereoDepth() = default;

bool StereoDepth::init(const Calibration& calib) {
    calib_ = calib;

    // Compute focal length from intrinsics (use fx from left camera)
    calib_.focalLength = calib_.K_left[0];  // K[0,0] = fx

    if (calib_.focalLength < 1.0f) {
        Logger::error("StereoDepth: Invalid focal length in calibration");
        return false;
    }

    // Compute baseline from translation vector
    // T = [tx, ty, tz] in mm, baseline = |tx|
    calib_.baseline = std::abs(calib_.T[0]);

    if (calib_.baseline < 10.0f) {
        Logger::warn("StereoDepth: Baseline seems too small (", calib_.baseline, " mm). Using default 75mm.");
        calib_.baseline = STEREO_BASELINE_MM;
    }

    Logger::info("StereoDepth initialized:");
    Logger::info("  Image size: ", calib_.width, "x", calib_.height);
    Logger::info("  Baseline: ", calib_.baseline, " mm");
    Logger::info("  Focal length: ", calib_.focalLength, " px");

    // TODO: Precompute rectification maps using OpenCV
    // For V3 Phase 3, we use a simplified approach assuming cameras are already
    // roughly aligned (OAK-D cameras are factory-calibrated)
    rectMaps_->valid = true;

    initialized_ = true;
    return true;
}

bool StereoDepth::loadFromDevice(void* daiDevice) {
    // This would load calibration from dai::Device
    // For now, use default OAK-D Pro PoE calibration

    Logger::info("StereoDepth: Loading calibration from device...");

    // Default OAK-D Pro PoE calibration (approximate values)
    // Real implementation should use device->readCalibration()
    Calibration calib;
    calib.width = MONO_WIDTH;
    calib.height = MONO_HEIGHT;

    // Approximate intrinsics for OAK-D at THE_400_P (640x400)
    // fx, fy ≈ 400-450 pixels at this resolution
    float fx = 420.0f;
    float fy = 420.0f;
    float cx = 320.0f;  // width/2
    float cy = 200.0f;  // height/2

    calib.K_left = {fx, 0, cx, 0, fy, cy, 0, 0, 1};
    calib.K_right = {fx, 0, cx, 0, fy, cy, 0, 0, 1};

    // Zero distortion (assuming rectified output)
    calib.D_left = {0, 0, 0, 0, 0};
    calib.D_right = {0, 0, 0, 0, 0};

    // Identity rotation (cameras parallel)
    calib.R = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    // Translation: baseline 75mm in X direction
    calib.T = {-75.0f, 0, 0};

    calib.baseline = 75.0f;
    calib.focalLength = fx;

    return init(calib);
}

float StereoDepth::getDepthAtPoint(const uint8_t* monoLeft,
                                   const uint8_t* monoRight,
                                   int monoWidth, int monoHeight,
                                   int px, int py) const {
    if (!initialized_) {
        Logger::warn("StereoDepth: Not initialized");
        return -1.0f;
    }

    // Validate point is within image bounds with margin for window
    int halfWindow = STEREO_WINDOW_SIZE / 2;
    if (px < halfWindow || px >= monoWidth - halfWindow ||
        py < halfWindow || py >= monoHeight - halfWindow) {
        return -1.0f;
    }

    // Search range for disparity (in pixels)
    // At 75mm baseline, 420px focal length:
    // depth = baseline * focal / disparity
    // For 0.5m-3m range: disparity ≈ 10-63 pixels
    constexpr int SEARCH_RANGE = 64;

    // Collect disparities from window for robust estimation
    std::vector<float> disparities;
    disparities.reserve(STEREO_WINDOW_SIZE * STEREO_WINDOW_SIZE);

    // For each pixel in the window, compute disparity
    for (int dy = -halfWindow; dy <= halfWindow; dy++) {
        for (int dx = -halfWindow; dx <= halfWindow; dx++) {
            int x = px + dx;
            int y = py + dy;

            // Get reference pixel from left image
            int leftIdx = y * monoWidth + x;
            uint8_t leftPixel = monoLeft[leftIdx];

            // Search for best match in right image (along epipolar line = same row)
            int bestOffset = 0;
            int bestSAD = INT32_MAX;

            for (int d = 0; d < SEARCH_RANGE && x - d >= halfWindow; d++) {
                int rightIdx = y * monoWidth + (x - d);
                uint8_t rightPixel = monoRight[rightIdx];

                int sad = std::abs(static_cast<int>(leftPixel) - static_cast<int>(rightPixel));

                if (sad < bestSAD) {
                    bestSAD = sad;
                    bestOffset = d;
                }
            }

            // Only add if match quality is good
            if (bestSAD < 30 && bestOffset > 0) {
                disparities.push_back(static_cast<float>(bestOffset));
            }
        }
    }

    if (disparities.empty()) {
        return -1.0f;  // No valid matches
    }

    // Use robust median for disparity
    float disparity = robustMedian(disparities);

    if (disparity < 1.0f) {
        return -1.0f;  // Too far or invalid
    }

    // Convert disparity to depth: Z = baseline * focal / disparity
    float depth = (calib_.baseline * calib_.focalLength) / disparity;

    // Validate depth range (0.3m - 5m)
    if (depth < 300.0f || depth > 5000.0f) {
        return -1.0f;
    }

    return depth;
}

Point3D StereoDepth::pixelTo3D(int px, int py, float depth) const {
    if (!initialized_ || depth <= 0) {
        return {0, 0, 0};
    }

    // Unproject pixel to 3D using pinhole camera model
    // X = (px - cx) * Z / fx
    // Y = (py - cy) * Z / fy
    // Z = depth

    float cx = calib_.K_left[2];  // Principal point x
    float cy = calib_.K_left[5];  // Principal point y
    float fx = calib_.K_left[0];  // Focal length x
    float fy = calib_.K_left[4];  // Focal length y

    float X = (static_cast<float>(px) - cx) * depth / fx;
    float Y = (static_cast<float>(py) - cy) * depth / fy;
    float Z = depth;

    return {X, Y, Z};
}

float StereoDepth::computeDisparity(const uint8_t* leftWindow,
                                    const uint8_t* rightWindow,
                                    int windowSize,
                                    int searchRange) const {
    // SAD (Sum of Absolute Differences) matching
    int bestDisparity = 0;
    int bestSAD = INT32_MAX;

    for (int d = 0; d < searchRange; d++) {
        int sad = 0;
        for (int i = 0; i < windowSize * windowSize; i++) {
            sad += std::abs(static_cast<int>(leftWindow[i]) - static_cast<int>(rightWindow[i]));
        }

        if (sad < bestSAD) {
            bestSAD = sad;
            bestDisparity = d;
        }
    }

    return static_cast<float>(bestDisparity);
}

float StereoDepth::robustMedian(const std::vector<float>& values) const {
    if (values.empty()) return 0.0f;

    std::vector<float> sorted = values;
    std::sort(sorted.begin(), sorted.end());

    size_t n = sorted.size();
    if (n % 2 == 0) {
        return (sorted[n/2 - 1] + sorted[n/2]) / 2.0f;
    } else {
        return sorted[n/2];
    }
}

} // namespace core

