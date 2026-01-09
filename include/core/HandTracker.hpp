#pragma once

#include "Types.hpp"
#include <array>
#include <chrono>

namespace core {

/**
 * V3 HandTracker: 6-state Kalman Filter for smooth 3D hand tracking
 *
 * State: [x, y, z, vx, vy, vz]
 * Model: Constant velocity with process noise
 *
 * Features:
 * - VIP lock after 15 consecutive frames
 * - +1 frame prediction for latency compensation
 * - Dropout handling (pure prediction)
 */
class HandTracker {
public:
    HandTracker();

    /**
     * Predict state to current time (call every frame)
     * @param dt Time delta in seconds (1/FPS)
     */
    void predict(float dt);

    /**
     * Update state with new measurement
     * @param measurement 3D position from stereo depth
     */
    void update(const Point3D& measurement);

    /**
     * Get predicted position with lookahead for latency compensation
     * @param lookahead Seconds to predict ahead (default: +1 frame)
     * @return Predicted 3D position
     */
    [[nodiscard]] Point3D getPredicted(float lookahead = KALMAN_LOOKAHEAD_S) const;

    /**
     * Get current velocity estimate from Kalman state
     */
    [[nodiscard]] Velocity3D getVelocity() const;

    /**
     * Handle frame dropout (no detection this frame)
     * Continues prediction, resets VIP after DROPOUT_LIMIT frames
     */
    void handleDropout();

    /**
     * Reset tracker to initial state
     */
    void reset();

    /**
     * Check if VIP is locked (15+ consecutive frames)
     */
    [[nodiscard]] bool isVipLocked() const { return vipLocked_; }

    /**
     * Get number of consecutive tracked frames
     */
    [[nodiscard]] int getConsecutiveFrames() const { return consecutiveFrames_; }

private:
    // State vector: [x, y, z, vx, vy, vz]
    std::array<float, 6> state_{};

    // Covariance matrix (6x6, stored as flat array)
    std::array<float, 36> P_{};

    // Process noise covariance
    static constexpr float PROCESS_NOISE_POS = 10.0f;   // mm
    static constexpr float PROCESS_NOISE_VEL = 50.0f;   // mm/s

    // Measurement noise
    static constexpr float MEASUREMENT_NOISE = 5.0f;    // mm

    // Tracking state
    int consecutiveFrames_ = 0;
    int dropoutCount_ = 0;
    bool vipLocked_ = false;
    bool initialized_ = false;

    // Initialize covariance matrix
    void initCovariance();

    // Matrix helpers
    void updateCovariancePredict(float dt);
    void updateCovarianceMeasure(const std::array<float, 6>& K);
};

} // namespace core

