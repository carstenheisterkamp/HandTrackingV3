#pragma once

#include <array>
#include <chrono>
#include <memory>
#include <vector>
#include <string>
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
constexpr int GESTURE_DEBOUNCE_FRAMES = 1;      // Instant recognition (no debounce)

// V3: OSC Configuration
constexpr int OSC_RATE_HZ = 30;
constexpr int OSC_MAX_LATENCY_MS = 50;

// V3: Z-Axis (Depth) Normalization for OSC
// All coordinates sent to OSC are normalized [0.0, 1.0]
// X, Y are image coordinates, Z is physical depth
// This allows flexible scaling in Game Engine (see docs/COORDINATE_SYSTEM.md)
constexpr float Z_MIN_MM = 1200.0f;     // 1.2m - minimum tracking depth (close to player)
constexpr float Z_MAX_MM = 2800.0f;     // 2.8m - maximum tracking depth (far from player)
constexpr float Z_RANGE_MM = Z_MAX_MM - Z_MIN_MM;  // 1600mm total range

// V3: Stereo Configuration
constexpr int STEREO_WINDOW_SIZE = 9;      // 9×9 pixel window for point depth
constexpr float STEREO_BASELINE_MM = 75.0f; // OAK-D Pro baseline

// ============================================================
// Data Structures
// ============================================================

// V3: Gesture States (matching OSC_GESTURE_REFERENCE.md)
enum class GestureState {
    Idle = 0,       // No hand visible / unknown

    // Basic gestures
    Five = 1,       // 🖐️ All 5 fingers open (was "Palm")
    Fist = 2,       // ✊ All fingers closed (was "Grab")
    Pointing = 3,   // ☝️ Only index extended (was "Point")
    Pinch = 4,      // Thumb + Index together

    // Number gestures
    ThumbsUp = 5,   // 👍 Only thumb extended
    Two = 6,        // Thumb + Index extended
    Three = 7,      // Thumb + Index + Middle
    Four = 8,       // All except thumb

    // Symbol gestures
    Peace = 9,      // ✌️ Index + Middle
    Metal = 10,     // 🤘 Index + Pinky
    LoveYou = 11,   // 🤟 Thumb + Index + Pinky
    CallMe = 12,    // 🤙 Thumb + Pinky
    Vulcan = 13,    // 🖖 All 5, spread between Middle and Ring
    MiddleFinger = 14, // 🖕 Only middle

    // Legacy aliases (for backward compatibility)
    Palm = Five,
    Grab = Fist,
    Point = Pointing
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

struct Delta3D {
    float dx = 0.0f;  // Change in velocity X (acceleration)
    float dy = 0.0f;  // Change in velocity Y
    float dz = 0.0f;  // Change in velocity Z
};

struct TrackingResult {
    // Hand identification
    int handId = 0;  // Hand ID (0 or 1)
    bool vipLocked = false;  // VIP lock status
    std::chrono::steady_clock::time_point timestamp;

    // V3: Palm center (Kalman filtered, predicted)
    Point3D palmPosition;

    // V3: Velocity from Kalman state
    Velocity3D velocity;

    // V3: Delta (Acceleration/Change in velocity)
    Delta3D delta;

    // V3: Gesture state (FSM output)
    GestureState gesture = GestureState::Idle;
    float gestureConfidence = 0.0f;  // Gesture FSM confidence (0-1)
    float palmConfidence = 0.0f;     // Palm detection score (0-1)
    float landmarkPresence = 0.0f;   // Landmark presence confidence (0-1)

    // Hand landmarks (21 points)
    std::vector<Point3D> landmarks;

    // Phase 4: OSC Event (e.g., "/player/session/enter")
    std::string osc_event;  // Empty if no event, otherwise OSC path

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

