#pragma once

#include <vector>
#include <cmath>

namespace math {

// Force update
class KalmanFilter {
public:
    KalmanFilter();

    struct Point3f { float x, y, z; };

    Point3f update(float x, float y, float z);
    /**
     * Predict the next state.
     * Must be called before update() in each cycle if using control input,
     * or just to propagate state.
     */
    void predict();
    void reset();

private:
    struct AxisFilter {
        float x = 0.0f; // State estimate
        float p = 1.0f; // Estimation error covariance
        float q = 0.01f; // Process noise covariance
        float r = 0.1f; // Measurement noise covariance
        float k = 0.0f; // Kalman gain

        void predict() {
            // Prediction update
            p = p + q;
        }

        float update(float measurement) {
            // Measurement update
            k = p / (p + r);
            x = x + k * (measurement - x);
            p = (1 - k) * p;

            return x;
        }
    };

    AxisFilter fx, fy, fz;
    bool initialized = false;
};

class OneEuroFilter {
public:
    OneEuroFilter(double minCutoff = 1.0, double beta = 0.007, double dCutoff = 1.0);
    double filter(double value, double timestamp);
    void reset();

private:
    struct LowPassFilter {
        double y = 0.0;
        double s = 0.0;
        bool initialized = false;

        double filter(double value, double alpha) {
            if (!initialized) {
                y = value;
                s = value;
                initialized = true;
                return value;
            }
            double result = alpha * value + (1.0 - alpha) * s;
            s = result;
            return result;
        }

        void reset() { initialized = false; }
    };

    double _minCutoff;
    double _beta;
    double _dCutoff;
    LowPassFilter _xFilter;
    LowPassFilter _dxFilter;
    double _lastTimestamp = -1.0;

    double alpha(double cutoff, double dt) {
        double tau = 1.0 / (2 * M_PI * cutoff);
        return 1.0 / (1.0 + tau / dt);
    }
};

} // namespace math

