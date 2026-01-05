#include "core/Filters.hpp"
#include <cmath>

namespace core {

KalmanFilter::KalmanFilter() {
    reset();
}

KalmanFilter::Point3f KalmanFilter::update(float x, float y, float z) {
    if (!initialized) {
        fx.x = x; fy.x = y; fz.x = z;
        initialized = true;
        return {x, y, z};
    }
    return {
        fx.update(x),
        fy.update(y),
        fz.update(z)
    };
}

void KalmanFilter::predict() {
    // Propagate state for all axes
    if (!initialized) return;
    fx.predict();
    fy.predict();
    fz.predict();
}

void KalmanFilter::reset() {
    fx = AxisFilter();
    fy = AxisFilter();
    fz = AxisFilter();
    initialized = false;
}

OneEuroFilter::OneEuroFilter(double minCutoff, double beta, double dCutoff)
    : _minCutoff(minCutoff), _beta(beta), _dCutoff(dCutoff) {
    reset();
}

double OneEuroFilter::filter(double value, double timestamp) {
    if (_lastTimestamp != -1.0 && timestamp != -1.0) {
        double dt = timestamp - _lastTimestamp;
        if (dt > 0) {
            // Compute the filtered derivative of the signal.
            double dx = (value - _xFilter.y) / dt;
            double edx = _dxFilter.filter(dx, alpha(_dCutoff, dt));

            // Use the result to update the cutoff frequency for the main filter.
            double cutoff = _minCutoff + _beta * std::abs(edx);

            // Filter the signal using the variable cutoff frequency.
            double result = _xFilter.filter(value, alpha(cutoff, dt));

            _lastTimestamp = timestamp;
            return result;
        }
    }

    _lastTimestamp = timestamp;
    // If first time or invalid dt, just set the value
    return _xFilter.filter(value, 1.0);
}

void OneEuroFilter::reset() {
    _xFilter.reset();
    _dxFilter.reset();
    _lastTimestamp = -1.0;
}

} // namespace core

