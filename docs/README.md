# Documentation Overview

**Last Updated:** 2026-01-08

---

## 📚 Active Documents (Production)

### 1. **OPTIMAL_WORKFLOW_V2_FINAL.md** 
**The definitive architecture and implementation plan**

- Status: ✅ **APPROVED** - Production-Ready
- Contains:
  - Complete system architecture (2 VIPs + Person Detection)
  - Device/Host split (OAK-D vs. Jetson)
  - Production targets (45 FPS @ 720p, 60 ms latency)
  - 5 implementation phases with acceptance criteria
  - YOLOv8n-person config (640×384, INT8, 12 FPS)
  - VIP-Management, ROI-System, Failure-Handling
- **This is the blueprint for implementation.**

---

### 2. **OPTIMAL_WORKFLOW_V2_REVIEW.md**
**Technical review and validation of the architecture**

- Status: ✅ **5/5 Stars** - Approved
- Contains:
  - Critical analysis of OPTIMAL_WORKFLOW_V2_FINAL
  - Hardware reality checks (CMX Memory, PoE Bandwidth, etc.)
  - YOLOv8n-person detailed spec review (Addendum)
  - Risk assessment and mitigations
  - Performance predictions
- **Use this for understanding the rationale behind decisions.**

---

### 3. **TODO.md**
**Complete implementation roadmap (Phase 0 → 5)**

- Status: 🔴 **Phase 0 NOT STARTED** (Ready to begin)
- Contains:
  - Current status snapshot (18 FPS @ 1080p)
  - Phase 0: Quick Wins (18 → 30 FPS in 1 day)
  - Phase 1: Person Detection & Tracking (1 week)
  - Phase 2: ROI-System (3-4 days)
  - Phase 3: Stereo Depth (2-3 days)
  - Phase 4: 45 FPS Optimization (3-5 days)
  - Phase 5: Production Infrastructure (2-3 days)
  - Concrete code snippets, file paths, acceptance criteria
- **This is your daily work plan.**

---

### 4. **OSC_GESTURE_REFERENCE.md**
**OSC protocol specification for gesture output**

- Status: ✅ Active
- Contains:
  - OSC message format
  - Gesture IDs and names
  - Landmark numbering
  - Client integration guide

---

### 5. **OAKDPROPOEFF.pdf**
**Hardware datasheet for OAK-D Pro PoE**

- Official Luxonis documentation
- Camera specs, depth range, connectivity

---

## 🗑️ Removed Documents (Obsolete)

The following documents were **deleted** on 2026-01-08 as they are now obsolete:

- ❌ `COORDINATE_SYSTEM.md` - Superseded by OPTIMAL_WORKFLOW_V2_FINAL
- ❌ `FOCUS_CALIBRATION.md` - Integrated into TODO Phase 1
- ❌ `SERVICE_MANUAL.md` - Superseded by TODO.md
- ❌ `SPECIFICATION.md` - Superseded by OPTIMAL_WORKFLOW_V2_FINAL
- ❌ `FPS_OPTIMIZATION_LOG.md` - Obsolete (old experiments)
- ❌ `ROI_API_FALLBACK.md` - Obsolete (Script-Node experiments)
- ❌ `GAP_ANALYSIS.md` - Superseded by OPTIMAL_WORKFLOW_V2_REVIEW
- ❌ `COMPLETE_GAP_ANALYSIS.md` - Superseded by TODO.md
- ❌ `OPTIMAL_WORKFLOW.md` - Old version (replaced by V2_FINAL)
- ❌ `OPTIMAL_WORKFLOW_NEW.md` - Old version
- ❌ `OPTIMAL_WORKFLOW_REVIEW.md` - Old version (replaced by V2_REVIEW)

---

## 📖 Reading Order (for New Contributors)

1. **Start here:** `OPTIMAL_WORKFLOW_V2_FINAL.md`  
   → Understand the target architecture

2. **Deep dive:** `OPTIMAL_WORKFLOW_V2_REVIEW.md`  
   → Learn why decisions were made

3. **Get to work:** `TODO.md`  
   → Pick a task and implement

4. **Reference:** `OSC_GESTURE_REFERENCE.md`  
   → When integrating OSC output

---

## 🎯 Quick Start (Phase 0)

**Ready to code?** Start with **Phase 0 in TODO.md**:

```bash
# 1. MJPEG hasClients() Check → +10 FPS (1-2 hours)
# 2. Stereo Throttling → +5 FPS (30 minutes)
# 3. Preview reduction → +2 FPS (5 minutes)
# 4. NN Threads: 1 → +3 FPS (5 minutes)
# 5. Sync: 10ms → +2 FPS (5 minutes)

# Result: 18 → 30 FPS in 6-8 hours ✅
```

---

## 🔄 Document Lifecycle

| Document | Status | Update Frequency |
|----------|--------|------------------|
| OPTIMAL_WORKFLOW_V2_FINAL.md | Frozen | Only for architecture changes |
| OPTIMAL_WORKFLOW_V2_REVIEW.md | Frozen | Reference only |
| TODO.md | Active | Daily (as phases complete) |
| OSC_GESTURE_REFERENCE.md | Stable | Only if protocol changes |

---

## 📊 Version History

- **v3.0** (2026-01-08): Complete rewrite based on OPTIMAL_WORKFLOW_V2
  - New TODO with 5 phases
  - Deleted obsolete documents
  - Added .gitignore

- **v2.0** (2026-01-06-08): Architecture refinement
  - Created OPTIMAL_WORKFLOW_V2_FINAL
  - Added comprehensive review
  - Removed old Gap Analysis

- **v1.0** (2025-XX-XX): Initial documentation
  - Basic SPEC, TODO, SERVICE_MANUAL

---

**For questions or clarifications, refer to the Architecture Decision Records in OPTIMAL_WORKFLOW_V2_REVIEW.md**

