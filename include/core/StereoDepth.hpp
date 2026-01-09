#pragma once

#include "Types.hpp"
#include <array>
#include <memory>

// Forward declarations
namespace cv {
    class Mat;
}

namespace core {

/**
 * V3 StereoDepth: Punktuelle Tiefenmessung am Palm Center
 *
 * Keine Full-Frame Depth Map - nur lokales 9x9 Fenster.
 * Verwendet OAK-D Kalibrierungsdaten für Rektifizierung.
 *
 * Performance: <1ms pro Punkt (vs 15-20ms für Full-Frame SGBM)
 */
class StereoDepth {
public:
    /**
     * Stereo calibration data
     */
    struct Calibration {
        // Intrinsics Left (3x3 flattened)
        std::array<float, 9> K_left{};

        // Intrinsics Right (3x3 flattened)
        std::array<float, 9> K_right{};

        // Distortion Left (5 coefficients: k1, k2, p1, p2, k3)
        std::array<float, 5> D_left{};

        // Distortion Right
        std::array<float, 5> D_right{};

        // Rotation (3x3 flattened) - Left to Right
        std::array<float, 9> R{};

        // Translation (3 elements) - Left to Right [mm]
        std::array<float, 3> T{};

        // Image size
        int width = 640;
        int height = 400;

        // Derived: Baseline in mm
        float baseline = 75.0f;

        // Derived: Focal length in pixels (from K_left)
        float focalLength = 0.0f;
    };

    StereoDepth();
    ~StereoDepth();

    /**
     * Initialize with calibration data
     * Precomputes rectification maps
     */
    bool init(const Calibration& calib);

    /**
     * Load calibration from OAK-D device
     * Call after device is connected
     */
    bool loadFromDevice(void* daiDevice);  // dai::Device*

    /**
     * Get depth at a specific 2D point
     *
     * @param monoLeft Left mono image (GRAY8)
     * @param monoRight Right mono image (GRAY8)
     * @param monoWidth Image width
     * @param monoHeight Image height
     * @param px X coordinate (pixel)
     * @param py Y coordinate (pixel)
     * @return Depth in mm, or -1 if invalid
     */
    [[nodiscard]] float getDepthAtPoint(const uint8_t* monoLeft,
                                        const uint8_t* monoRight,
                                        int monoWidth, int monoHeight,
                                        int px, int py) const;

    /**
     * Get 3D position in camera coordinates
     *
     * @param px X coordinate (pixel)
     * @param py Y coordinate (pixel)
     * @param depth Depth in mm
     * @return 3D point in camera frame [mm]
     */
    [[nodiscard]] Point3D pixelTo3D(int px, int py, float depth) const;

    /**
     * Check if initialized
     */
    [[nodiscard]] bool isInitialized() const { return initialized_; }

    /**
     * Get baseline in mm
     */
    [[nodiscard]] float getBaseline() const { return calib_.baseline; }

private:
    Calibration calib_;
    bool initialized_ = false;

    // Rectification maps (precomputed)
    struct RectifyMaps;
    std::unique_ptr<RectifyMaps> rectMaps_;

    // Local stereo matching
    [[nodiscard]] float computeDisparity(const uint8_t* leftWindow,
                                         const uint8_t* rightWindow,
                                         int windowSize,
                                         int searchRange) const;

    // Robust median filter
    [[nodiscard]] float robustMedian(const std::vector<float>& values) const;
};

} // namespace core

