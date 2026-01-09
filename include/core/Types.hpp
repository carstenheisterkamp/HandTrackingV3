#pragma once

#include <array>
#include <chrono>
#include <memory>
#include "Frame.hpp"
#include "FramePool.hpp"
#include "SpscQueue.hpp"

namespace core {

// ============================================================
// V3 Constants - 3D Hand Controller Configuration
// ============================================================

// Camera Configuration
constexpr size_t FRAME_WIDTH = 1920;
constexpr size_t FRAME_HEIGHT = 1080;
constexpr size_t FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * 3 / 2; // NV12

// V3: RGB Preview (for NN input)
constexpr size_t RGB_PREVIEW_WIDTH = 640;
constexpr size_t RGB_PREVIEW_HEIGHT = 360;

// Stereo frames for GPU-based depth (computed on Jetson, not OAK-D)
constexpr size_t MONO_WIDTH = 640;
constexpr size_t MONO_HEIGHT = 400;
constexpr size_t MONO_FRAME_SIZE = MONO_WIDTH * MONO_HEIGHT; // GRAY8
constexpr size_t DEPTH_FRAME_SIZE = MONO_WIDTH * MONO_HEIGHT * 2; // UINT16 depth map

// V3: Target FPS
constexpr float CAMERA_FPS = 60.0f;

// Queue/Pool sizing
constexpr size_t QUEUE_SIZE = 16;
constexpr size_t POOL_SIZE = 16;

// V3: Tracking Configuration
constexpr int VIP_LOCK_FRAMES = 15;        // Frames until VIP is locked (~250ms @ 60fps)
constexpr int DROPOUT_LIMIT = 5;           // Max dropouts before VIP reset
constexpr float KALMAN_LOOKAHEAD_S = 0.033f; // +1 frame prediction (~33ms)

// V3: Gesture Configuration
constexpr float PINCH_THRESHOLD_ENTER = 0.08f;  // 8% of hand size
constexpr float PINCH_THRESHOLD_EXIT = 0.12f;   // 12% - hysteresis gap
constexpr int GESTURE_DEBOUNCE_FRAMES = 3;      // ~50ms @ 60fps

// V3: OSC Configuration
constexpr int OSC_RATE_HZ = 30;
constexpr int OSC_MAX_LATENCY_MS = 50;

// V3: Stereo Configuration
constexpr int STEREO_WINDOW_SIZE = 9;      // 9×9 pixel window for point depth
constexpr float STEREO_BASELINE_MM = 75.0f; // OAK-D Pro baseline

// ============================================================
// Data Structures
// ============================================================

// V3: Gesture States
enum class GestureState {
    Idle = 0,   // No hand visible
    Palm = 1,   // Hand open
    Pinch = 2,  // Thumb + Index together
    Grab = 3,   // Fist
    Point = 4   // Only index extended
};

// V3: 3D Point with velocity
struct Point3D {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct Velocity3D {
    float vx = 0.0f;
    float vy = 0.0f;
    float vz = 0.0f;
};

struct TrackingResult {
    struct NormalizedPoint {
        float x, y, z;
    };

    std::vector<NormalizedPoint> landmarks; // 21 points * 3 coords
    bool vipLocked = false;
    std::chrono::steady_clock::time_point timestamp;

    // V3: Palm center (Kalman filtered, predicted)
    Point3D palmPosition;

    // V3: Velocity from Kalman state
    Velocity3D velocity;

    // V3: Gesture state (FSM output)
    GestureState gesture = GestureState::Idle;
    float gestureConfidence = 0.0f;

    // Legacy fields (for compatibility)
    float pinchDistance = 0.0f;
    int gestureId = 0; // 0=None, 1=Pinch, 2=Fist
    std::string gestureName = "unknown"; // For logging/debug
};

// Type Aliases
using AppFramePool = FramePool<POOL_SIZE, FRAME_SIZE, DEPTH_FRAME_SIZE, MONO_FRAME_SIZE>;
using AppProcessingQueue = SpscQueue<Frame*, QUEUE_SIZE>;
using OscQueue = SpscQueue<TrackingResult, QUEUE_SIZE>;

} // namespace core

