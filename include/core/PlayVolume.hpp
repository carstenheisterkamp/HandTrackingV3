#pragma once

#include <cstddef>

namespace core {

/**
 * PlayVolume: Defines the 3D volume where hands are tracked
 *
 * Phase 4 Component - implements Volume-Filtering for Player Lock System
 * Coordinates are normalized (0-1), matching camera and OSC format
 */
struct PlayVolume {
    // 2D boundaries (normalized coordinates, 0-1)
    float minX = 0.1f;   // 10% margin left
    float maxX = 0.9f;   // 10% margin right
    float minY = 0.1f;   // 10% margin top
    float maxY = 0.9f;   // 10% margin bottom

    // 3D depth boundaries (absolute values in mm)
    float minZ = 500.0f;   // 50cm minimum distance
    float maxZ = 2500.0f;  // 2.5m maximum distance

    /**
     * Check if a 2D point is inside the volume (X, Y only)
     * Used for palm detection filtering before stereo depth
     */
    bool contains2D(float x, float y) const {
        return x >= minX && x <= maxX &&
               y >= minY && y <= maxY;
    }

    /**
     * Check if a 3D point is inside the volume (X, Y, Z)
     * Used after stereo depth is computed
     */
    bool contains3D(float x, float y, float z_mm) const {
        return contains2D(x, y) &&
               z_mm >= minZ && z_mm <= maxZ;
    }

    /**
     * Get volume dimensions for logging/debug
     */
    float getWidth() const { return maxX - minX; }
    float getHeight() const { return maxY - minY; }
    float getDepth() const { return maxZ - minZ; }

    /**
     * Check if volume is 16:9 aspect ratio (within tolerance)
     */
    bool is16x9() const {
        float width = getWidth();
        float height = getHeight();
        float aspectRatio = width / height;
        constexpr float target = 16.0f / 9.0f;
        constexpr float tolerance = 0.1f;
        return (aspectRatio >= target - tolerance) &&
               (aspectRatio <= target + tolerance);
    }
};

/**
 * Default Play Volume configuration
 * 16:9 aspect ratio matching camera (640x360)
 * Optimized for gaming at 0.5m - 2.5m range
 */
inline PlayVolume getDefaultPlayVolume() {
    PlayVolume volume;
    volume.minX = 0.1f;
    volume.maxX = 0.9f;
    volume.minY = 0.1f;
    volume.maxY = 0.9f;
    volume.minZ = 500.0f;   // 0.5m
    volume.maxZ = 2500.0f;  // 2.5m
    return volume;
}

} // namespace core

