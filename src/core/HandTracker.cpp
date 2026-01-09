#include "core/HandTracker.hpp"
#include "core/Logger.hpp"
#include <cmath>

namespace core {

HandTracker::HandTracker() {
    reset();
}

void HandTracker::reset() {
    state_.fill(0.0f);
    initCovariance();
    consecutiveFrames_ = 0;
    dropoutCount_ = 0;
    vipLocked_ = false;
    initialized_ = false;
}

void HandTracker::initCovariance() {
    // Initialize P as diagonal matrix with high uncertainty
    P_.fill(0.0f);

    // Position uncertainty: 100mm
    P_[0 * 6 + 0] = 10000.0f;  // x
    P_[1 * 6 + 1] = 10000.0f;  // y
    P_[2 * 6 + 2] = 10000.0f;  // z

    // Velocity uncertainty: 500mm/s
    P_[3 * 6 + 3] = 250000.0f; // vx
    P_[4 * 6 + 4] = 250000.0f; // vy
    P_[5 * 6 + 5] = 250000.0f; // vz
}

void HandTracker::predict(float dt) {
    if (!initialized_) return;

    // Constant velocity model: x_new = x + v * dt
    // State transition:
    // [x]     [1 0 0 dt 0  0 ] [x ]
    // [y]     [0 1 0 0  dt 0 ] [y ]
    // [z]  =  [0 0 1 0  0  dt] [z ]
    // [vx]    [0 0 0 1  0  0 ] [vx]
    // [vy]    [0 0 0 0  1  0 ] [vy]
    // [vz]    [0 0 0 0  0  1 ] [vz]

    state_[0] += state_[3] * dt;  // x += vx * dt
    state_[1] += state_[4] * dt;  // y += vy * dt
    state_[2] += state_[5] * dt;  // z += vz * dt
    // Velocities stay the same in constant velocity model

    // Update covariance: P = F * P * F^T + Q
    updateCovariancePredict(dt);
}

void HandTracker::updateCovariancePredict(float dt) {
    // Simplified covariance update for constant velocity model
    // This is an approximation - full implementation would use proper matrix ops

    float dt2 = dt * dt;

    // Add process noise
    float qPos = PROCESS_NOISE_POS * PROCESS_NOISE_POS * dt;
    float qVel = PROCESS_NOISE_VEL * PROCESS_NOISE_VEL * dt;

    // Propagate position uncertainty with velocity coupling
    P_[0 * 6 + 0] += 2.0f * dt * P_[0 * 6 + 3] + dt2 * P_[3 * 6 + 3] + qPos;
    P_[1 * 6 + 1] += 2.0f * dt * P_[1 * 6 + 4] + dt2 * P_[4 * 6 + 4] + qPos;
    P_[2 * 6 + 2] += 2.0f * dt * P_[2 * 6 + 5] + dt2 * P_[5 * 6 + 5] + qPos;

    // Add process noise to velocity
    P_[3 * 6 + 3] += qVel;
    P_[4 * 6 + 4] += qVel;
    P_[5 * 6 + 5] += qVel;

    // Update cross-covariances (position-velocity)
    P_[0 * 6 + 3] += dt * P_[3 * 6 + 3];
    P_[3 * 6 + 0] = P_[0 * 6 + 3];
    P_[1 * 6 + 4] += dt * P_[4 * 6 + 4];
    P_[4 * 6 + 1] = P_[1 * 6 + 4];
    P_[2 * 6 + 5] += dt * P_[5 * 6 + 5];
    P_[5 * 6 + 2] = P_[2 * 6 + 5];
}

void HandTracker::update(const Point3D& measurement) {
    dropoutCount_ = 0;  // Reset dropout counter

    if (!initialized_) {
        // First measurement - initialize state directly
        state_[0] = measurement.x;
        state_[1] = measurement.y;
        state_[2] = measurement.z;
        state_[3] = 0.0f;  // vx
        state_[4] = 0.0f;  // vy
        state_[5] = 0.0f;  // vz
        initialized_ = true;
        consecutiveFrames_ = 1;
        return;
    }

    // Kalman update for position-only measurement
    // Measurement model: z = H * x, where H = [I_3x3 | 0_3x3]

    // Innovation (measurement residual)
    float y0 = measurement.x - state_[0];
    float y1 = measurement.y - state_[1];
    float y2 = measurement.z - state_[2];

    // Innovation covariance: S = H * P * H^T + R
    float R = MEASUREMENT_NOISE * MEASUREMENT_NOISE;
    float S0 = P_[0 * 6 + 0] + R;
    float S1 = P_[1 * 6 + 1] + R;
    float S2 = P_[2 * 6 + 2] + R;

    // Kalman gain: K = P * H^T * S^-1
    // For our measurement model, K is 6x3 but we compute column by column
    std::array<float, 6> K{};
    K[0] = P_[0 * 6 + 0] / S0;
    K[1] = P_[1 * 6 + 1] / S1;
    K[2] = P_[2 * 6 + 2] / S2;
    K[3] = P_[3 * 6 + 0] / S0;  // vx gain from x measurement
    K[4] = P_[4 * 6 + 1] / S1;  // vy gain from y measurement
    K[5] = P_[5 * 6 + 2] / S2;  // vz gain from z measurement

    // State update: x = x + K * y
    state_[0] += K[0] * y0;
    state_[1] += K[1] * y1;
    state_[2] += K[2] * y2;
    state_[3] += K[3] * y0;
    state_[4] += K[4] * y1;
    state_[5] += K[5] * y2;

    // Covariance update: P = (I - K * H) * P
    updateCovarianceMeasure(K);

    // Update VIP tracking
    consecutiveFrames_++;
    if (consecutiveFrames_ >= VIP_LOCK_FRAMES && !vipLocked_) {
        vipLocked_ = true;
        Logger::info("HandTracker: VIP LOCKED after ", consecutiveFrames_, " frames");
    }
}

void HandTracker::updateCovarianceMeasure(const std::array<float, 6>& K) {
    // Simplified Joseph form update for numerical stability
    // P = (I - K*H) * P * (I - K*H)^T + K * R * K^T

    // For efficiency, we use the simplified form:
    // P = (I - K*H) * P

    // Update diagonal elements
    P_[0 * 6 + 0] *= (1.0f - K[0]);
    P_[1 * 6 + 1] *= (1.0f - K[1]);
    P_[2 * 6 + 2] *= (1.0f - K[2]);

    // Update position-velocity cross terms
    P_[0 * 6 + 3] *= (1.0f - K[0]);
    P_[3 * 6 + 0] = P_[0 * 6 + 3];
    P_[1 * 6 + 4] *= (1.0f - K[1]);
    P_[4 * 6 + 1] = P_[1 * 6 + 4];
    P_[2 * 6 + 5] *= (1.0f - K[2]);
    P_[5 * 6 + 2] = P_[2 * 6 + 5];

    // Velocity covariance updated based on position measurement
    P_[3 * 6 + 3] -= K[3] * P_[0 * 6 + 3];
    P_[4 * 6 + 4] -= K[4] * P_[1 * 6 + 4];
    P_[5 * 6 + 5] -= K[5] * P_[2 * 6 + 5];
}

Point3D HandTracker::getPredicted(float lookahead) const {
    if (!initialized_) {
        return {0.0f, 0.0f, 0.0f};
    }

    // Predict position with lookahead for latency compensation
    return {
        state_[0] + state_[3] * lookahead,
        state_[1] + state_[4] * lookahead,
        state_[2] + state_[5] * lookahead
    };
}

Velocity3D HandTracker::getVelocity() const {
    if (!initialized_) {
        return {0.0f, 0.0f, 0.0f};
    }

    return {state_[3], state_[4], state_[5]};
}

void HandTracker::handleDropout() {
    if (!initialized_) return;

    dropoutCount_++;

    if (dropoutCount_ > DROPOUT_LIMIT) {
        Logger::warn("HandTracker: Lost track after ", dropoutCount_, " dropouts. Resetting VIP.");
        vipLocked_ = false;
        consecutiveFrames_ = 0;
        // Don't reset completely - keep last known state for recovery
    }

    // Continue predicting (no update) to maintain trajectory estimate
    // The predict() call in the processing loop will handle this
}

} // namespace core

