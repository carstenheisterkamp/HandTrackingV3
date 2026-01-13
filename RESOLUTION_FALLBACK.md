9# Resolution Fallback Guide (Phase 4)

## 🎯 Current Test: 1920×1080 Full HD

**Target:** Maximum hand detection quality @ 2m distance  
**Expected FPS:** 25-30 FPS (monitor closely!)

---

## 📊 Fallback Plan

| Resolution | Hand Size @ 2m | Expected FPS | Use Case |
|------------|---------------|--------------|----------|
| **1920×1080** | ~300px | 20-30 FPS | **TESTING NOW** - Max quality |
| **1536×864** | ~240px | 25-30 FPS | Primary fallback if FPS <25 |
| **1280×720** | ~200px | 28-30 FPS | Proven stable (previous test) |
| **960×540** | ~150px | 30 FPS | Emergency fallback |

---

## 🔧 How to Change Resolution

### Option 1: Edit Header (Recommended)
```bash
# Edit on macOS, then remote build
vim include/core/PipelineManager.hpp

# Line ~25-26:
int previewWidth = 1536;   // Change this
int previewHeight = 864;   // Change this
```

### Option 2: Runtime Config (Future)
```cpp
// main.cpp - Override defaults
config.previewWidth = 1536;
config.previewHeight = 864;
```

---

## 📈 FPS Monitoring

System auto-warns if FPS <25 for >10 seconds:

```
⚠️ FPS DEGRADATION: 22.3 FPS < 25 target
   Current: 1920x1080 (Full HD - Maximum Quality)
   Fallback: 1536x864 (Recommended for stable 30 FPS)
   → Change PipelineManager.hpp previewWidth/Height and rebuild
```

---

## ✅ Test Results (Updated 2026-01-11)

### 1920×1080 Test ❌
- [x] FPS stable (≥25)?  **12.5 FPS** ❌ (Too low!)
- [x] Hand tracking robust @ 2m?  ✅ (Better recognition for fist & angles)
- [x] TensorRT inference time: **~80ms** (bottleneck!)
- [x] Decision: ☑ Fallback to 1536×864

**Analysis:**
- Hand size excellent (~300px)
- Palm Detection + Landmark inference too slow
- TensorRT 10.x on Orin Nano can't handle Full HD @ 30 FPS
- Quality improvement noted but not worth 60% FPS loss

### 1536×864 Test (CURRENT)
- [ ] FPS stable (≥25)?  ___ FPS average
- [ ] Hand tracking quality acceptable?
- [ ] Decision: ☐ Keep  ☐ Fallback to 1280×720

**Expected:** 25-28 FPS (33% less pixels than 1920×1080)

---

## 🎮 Recommendation Matrix

| Scenario | Resolution | Reason |
|----------|-----------|--------|
| **Competitive/Fast gameplay** | 1280×720 | Guaranteed 30 FPS, low latency |
| **Balanced (Default)** | 1536×864 | Good quality + stable FPS |
| **Maximum accuracy** | 1920×1080 | Best hand detection (if FPS allows) |
| **Fallback/Emergency** | 960×540 | Always stable |

---

**Last Updated:** 2026-01-11  
**Current Status:** Testing 1920×1080 Full HD

