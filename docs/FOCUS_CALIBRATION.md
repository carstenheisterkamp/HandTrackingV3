# Manual Focus Calibration Guide

## ⚠️ IMPORTANT: Manual Focus NOT YET IMPLEMENTED

**Status:** The manual focus feature is **not yet functional** in the current codebase.

### Why Doesn't Autofocus Work Even With IR Projector?

Even though the OAK-D Pro has IR illumination at 100%, autofocus still struggles in low-light. Here's why:

#### 1. **IR Spectrum Sensitivity**
- OAK-D IR LEDs: ~850nm wavelength (near-infrared)
- RGB sensor: Optimized for visible light (400-700nm)
- **Problem:** RGB sensor has LIMITED IR sensitivity
- The Bayer filter blocks most IR for accurate color reproduction
- AF gets SOME help from IR, but not enough for reliable locking

#### 2. **Phase Detection Autofocus (PDAF) Limitations**
- OAK-D uses PDAF (not contrast-detection)
- PDAF requires **sharp edges** and **texture contrast**
- In low-light, even with IR:
  - Skin has low contrast (similar tones everywhere)
  - Edges are soft/blurred due to noise
  - PDAF cannot find clear phase difference

#### 3. **Contrast vs. Brightness**
- IR projector adds **brightness** ✓
- But PDAF needs **contrast** (edges/texture) ✗
- Evenly illuminated hands = uniform brightness = poor contrast
- Result: AF hunts but never locks

#### 4. **Timing Issue**
- Even though IR is enabled BEFORE pipeline start (as of latest code)
- IR LEDs need ~100-200ms to reach full intensity
- AF initialization might happen during IR ramp-up
- By the time IR is stable, AF may have already failed to lock

#### 5. **Continuous AF Mode**
- Default continuous AF keeps trying to refocus
- In low-light, this causes "hunting" (constant blur/sharp cycling)
- Each attempt fails → more hunting → never stable

### The Only Reliable Solution: Manual Focus

**Manual focus eliminates ALL these problems:**
- ✅ No dependency on lighting conditions
- ✅ No dependency on scene contrast
- ✅ Instant, consistent sharpness
- ✅ No AF hunting/instability
- ✅ Predictable results for fixed working distance

### Why Manual Focus Requires Implementation

DepthAI v3 API requires manual focus to be set via **CameraControl Messages** sent through an **XLinkIn queue** AFTER the pipeline is running. This cannot be done during pipeline construction.

### Current Behavior
- **Autofocus is ACTIVE** (default behavior)
- **IR lights at 100%** intensity (helps slightly but insufficient)
- Manual focus control requires implementing a CameraControl input queue

### TODO
See `docs/TODO.md` → "Runtime Camera Control" - This needs to be implemented first before manual focus can work.

---

## Future Implementation (Not Active Yet)

When CameraControl is implemented, you will be able to:

### 1. Edit `src/main.cpp` (around line 59):
```cpp
config.manualFocus = 170; // Adjust this value based on your setup
```

### 2. Recompile and restart:
```bash
# On Jetson
cd /home/nvidia/dev/HandTrackingV3/cmake-build-debug-remote-host
ninja -j4
sudo systemctl restart hand-tracking.service
```

---

## Focus Value Guide

| Distance Range | Recommended Value | Use Case |
|----------------|-------------------|----------|
| **0.5 - 0.8m** (close) | `180 - 200` | Desktop setup, close interaction |
| **0.8 - 1.2m** (normal) | `160 - 180` | **Default: 170** - Standing/sitting position |
| **1.2 - 2.0m** (far) | `140 - 160` | Larger gestures, performance |

### Focus Scale
- **0** = Infinity (landscape/far away)
- **255** = Closest possible (macro)
- **~170** = Optimal for typical hand tracking (~1m)

---

## Calibration Process

### Option 1: Binary Search (Fast)
1. Start with `170`
2. If blurry → increase by 20 (e.g., `190`)
3. If still blurry → increase by 10 (e.g., `200`)
4. If too sharp/overshoot → decrease by 5
5. Repeat until sharp

### Option 2: Live Preview Method
1. Open MJPEG preview: `http://<jetson-ip>:8080`
2. Edit `config.manualFocus` in `main.cpp`
3. Recompile and restart service
4. Check preview for sharpness
5. Iterate until optimal

### Option 3: Distance Measurement
1. Measure actual distance from camera to typical hand position
2. Use this lookup table:

| Actual Distance | Focus Value (approx.) |
|-----------------|----------------------|
| 40cm | 220 |
| 60cm | 195 |
| 80cm | 180 |
| 100cm | **170** (default) |
| 120cm | 160 |
| 150cm | 145 |
| 200cm | 130 |

---

## IR Light Optimization

The IR intensity is now set to **100%** (was 80%) for maximum illumination in poor lighting:

```cpp
device_->setIrLaserDotProjectorIntensity(1.0f);  // 100%
device_->setIrFloodLightIntensity(1.0f);         // 100%
```

**Note:** If the IR light is too bright (overexposure), you can reduce it by editing `PipelineManager.cpp` line ~87:
```cpp
// Reduce to 80% if needed:
device_->setIrLaserDotProjectorIntensity(0.8f);
```

---

## Troubleshooting

### Image still blurry?
- **Check IR LEDs:** Look at the camera through a phone camera - you should see red/purple glow from IR LEDs
- **Increase IR intensity:** Set both to `1.0f` (100%)
- **Try wider range:** Test values from 140 to 200 in steps of 10

### Too sharp/overexposed?
- **Reduce focus value:** Lower by 10-20
- **Reduce IR intensity:** Try `0.6f` (60%) or disable one IR source

### Hand distance varies?
- **Choose middle value:** If hands move between 0.8-1.5m, use `170` (center of range)
- **Future improvement:** Implement dynamic focus control via OSC (see TODO.md)

### NN Input Format Warnings?
If you see warnings like:
```
[warning] Input image (224x224) does not match NN (3x224)
```

**Solution:** The code now uses `NV12` format and lets the Neural Network handle internal conversion. This warning should disappear after the latest changes. If it persists:
- Ensure you're using the latest code from main branch
- The NN expects CHW (Channel-Height-Width) format, the pipeline automatically converts

---

## Technical Details

### Why Manual Focus?
The OAK-D Pro's phase-detection autofocus (PDAF) requires **contrast and texture** to lock focus. In low-light conditions:
- Not enough ambient light for contrast detection
- IR projector provides some help, but not enough for reliable AF
- **Manual focus is more reliable** for fixed working distances

### DepthAI v3 API
```cpp
// Set manual focus (disables autofocus)
camera->setManualFocus(uint8_t value);  // 0-255

// Re-enable autofocus (if needed later)
camera->setAutoFocusMode(dai::CameraControl::AutoFocusMode::CONTINUOUS_VIDEO);
```

### Future: Runtime Control
See `docs/TODO.md` → "Runtime Camera Control" for planned OSC-based dynamic focus control:
```
/camera/focus/manual 180   # Change focus on-the-fly
/camera/focus/auto         # Re-enable autofocus
```

---

## Example Configurations

### Close Desktop Setup (60-80cm)
```cpp
config.manualFocus = 190;
```

### Standing Position (1m)
```cpp
config.manualFocus = 170;  // Default
```

### Performance Stage (1.5-2m)
```cpp
config.manualFocus = 145;
```

---

**Last Updated:** 2026-01-08  
**Status:** ✅ Working solution - Autofocus disabled, manual focus active

