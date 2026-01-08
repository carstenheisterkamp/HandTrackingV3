# COPILOT WORKING RULES - STRICT ENFORCEMENT

## 🚨 ABSOLUTE RULES (NEVER VIOLATE)

### 1. NO ARCHITECTURE CHANGES WITHOUT EXPLICIT APPROVAL
```
BEFORE changing any architectural decision:
1. STOP
2. EXPLAIN the problem
3. PROPOSE solution with pros/cons
4. WAIT for user approval
5. THEN implement

DO NOT:
- Roll back to "stable" versions
- Remove features without asking
- Change pipeline structure without discussion
```

### 2. NO UTILITY BLOAT
```
FORBIDDEN:
- Creating "helper" scripts for one-time use
- Creating "deployment" scripts (CLion handles this)
- Creating "convenience" wrappers
- Creating multiple documentation files for same topic

ALLOWED:
- Essential build/diagnostic tools (already in scripts/)
- Single comprehensive docs (TODO.md, SPECIFICATION.md)
```

### 3. FOCUS ON CURRENT GOAL
```
CURRENT GOAL: Dynamic ROI Cropping Working
- Fix Script-Node Python syntax
- Verify pipeline runs
- Measure performance

DO NOT:
- Optimize unrelated code
- Add "nice to have" features
- Create extensive documentation
- Build deployment infrastructure
```

---

## ✅ WORKFLOW FOR PROBLEMS

### Step 1: Diagnose
```
1. Get error logs from Jetson
2. Identify root cause
3. Check if it's a NEW problem or pre-existing
```

### Step 2: Propose Solution
```
ASK USER:
"Problem: [X]
Root Cause: [Y]
Proposed Fix: [Z]

This will affect: [files/architecture]
Do you want me to proceed?"
```

### Step 3: Implement ONLY After Approval
```
- Make minimal changes
- Test immediately
- No "while I'm here" improvements
```

---

## 🎯 CURRENT PROJECT STATUS

**Architecture:** v3 Dynamic ROI with Script-Node
**Problem:** Script-Node Python syntax errors
**Next Step:** Fix Python API calls (setResize vs setOutputSize)

**DO NOT:**
- Revert to old architecture
- Create deployment scripts
- Write extensive documentation

**DO:**
- Fix the immediate bug
- Verify it works
- Move to next TODO item

---

## 📋 TODO WORKFLOW

**ONLY work on items from docs/TODO.md**

When a TODO is complete:
1. Update TODO.md with [x] and brief note
2. Commit changes
3. Ask: "What's next?"

DO NOT:
- Work on items not in TODO.md
- Add "future improvements" to TODO.md
- Create sub-tasks without approval

---

## 🗂️ FILE ORGANIZATION RULES

**Core Implementation:**
- `src/` - C++ implementation ONLY
- `include/` - Headers ONLY
- `scripts/` - Essential build/diagnostic tools ONLY (no one-off helpers)

**Documentation:**
- `docs/TODO.md` - Master TODO list
- `docs/SPECIFICATION.md` - Technical spec
- `docs/*.md` - Specific technical docs (OSC format, coordinate system, etc.)
- NO: workflow guides, deployment guides, architecture diagrams

**Scripts:**
- Keep ONLY what's in copilot-instructions.md
- Remove anything used once

---

## 💬 COMMUNICATION RULES

### When User is Frustrated:
```
1. STOP immediately
2. Acknowledge mistake
3. Ask for clarification
4. Wait for direction
5. NO defensive explanations
```

### Before Major Changes:
```
ALWAYS ASK:
"I need to change [X] to fix [Y].
This affects [Z].
Shall I proceed?"

WAIT for answer.
```

### During Implementation:
```
- Brief status updates
- No verbose explanations unless asked
- Focus on results, not process
```

---

## 🔧 EMERGENCY PROTOCOL

If user says "STOP" or shows frustration:
1. STOP all changes immediately
2. Show what was changed (file list only)
3. Offer to revert
4. Wait for instructions

---

## ✅ SELF-CHECK BEFORE EVERY ACTION

Ask yourself:
1. Did user explicitly ask for this?
2. Is this in TODO.md or copilot-instructions.md?
3. Am I creating a new file? (If yes, WHY?)
4. Am I changing architecture? (If yes, STOP and ASK)
5. Will this file be used more than once?

If ANY answer is unclear → ASK USER FIRST.

---

**Last Updated:** 2026-01-08
**Enforcement:** MANDATORY for all Copilot interactions

