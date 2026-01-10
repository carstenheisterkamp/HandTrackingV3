# OSC Referenz - OAK-D Hand Tracking Service

**Version:** 2.0 (V3 Architecture)  
**Datum:** 10. Januar 2026  
**Port:** 9000 (127.0.0.1 auf Jetson)  
**Preview:** http://100.101.16.21:8080 (via Tailscale)

## Architektur: Non-Blocking OSC

Das OSC-Sending ist **vollständig non-blocking** und blockiert niemals die Hauptpipeline.  
**Performance-Garantie**: Das OSC-Subsystem hat **null** Einfluss auf die Pipeline-Framerate.

### Bewegungsglättung
- **Kalman Filter (6-State):** Position und Velocity werden geglättet `[x, y, z, vx, vy, vz]`
- **Latenz-Kompensation:** +1 Frame Prediction
- **Drop-Oldest Policy:** Pakete älter als 50ms werden verworfen

## Implementierte Features ✅

### OSC-Adressen (Aktuell Implementiert)

| Adresse | Typ | Beschreibung | Status |
|---------|-----|--------------|--------|
| `/hand/{id}/palm` | [x, y, z] | Palm-Position normalisiert (0-1) | ✅ |
| `/hand/{id}/velocity` | [vx, vy, vz] | Geschwindigkeit (geglättet via Kalman) | ✅ |
| `/hand/{id}/gesture` | [int, float, string] | [State-ID, Confidence, Name] | ✅ |
| `/hand/{id}/vip` | int | VIP-Lock Status (legacy, 1=locked) | ✅ |

**Hinweis:** `{id}` ist 0 oder 1 (max. 2 Hände)

### Multi-Person Handling

**Phase 3 (Aktuell):** Top-2 Selection nach Score

**Verhalten bei >2 Händen im Bild:**
1. Palm Detection erkennt ALLE Hände im Frame
2. **NMS (Non-Maximum Suppression)** mit IoU-Threshold (0.3)
3. **Top-2 Selection:** Die 2 Hände mit höchstem Confidence-Score
4. Restliche Hände werden ignoriert

**Limitation:** Hand-IDs können zwischen Personen wechseln wenn Score sich ändert.

---

**Phase 4 (Geplant):** Player Lock System 🎮

**Siehe:** `PLAYER_LOCK_DESIGN.md`

**Stabiles Single-User Gaming:**
1. **Play Volume:** 3D-Bereich im Kameraraum definiert
2. **Face Anchoring:** Haar Cascade ordnet Hände einer Person zu
3. **First-Come-First-Serve:** Erste Person im Volume wird "Owner"
4. **Session Lock:** Hand-IDs bleiben stabil bis Player Volume verlässt

**Neue OSC Events:**
```
/player/enter          → Player betritt Volume
/player/calibrating    → Warte auf stabile Detection
/player/active         → Session aktiv, Gameplay enabled
/player/lost           → Player temporär verloren (3s Grace Period)
/player/exit           → Session beendet
```

**Vorteile für Gaming:**
- ✅ Keine Hand-ID Wechsel während Gameplay
- ✅ Ignoriert Zuschauer/andere Personen
- ✅ Event-basiert (Spawn/Despawn von Player-Objekten)
- ✅ Konfigurierbare Play-Zone
- ✅ **Debug Visualization** - Gesicht, Hände, Volume im MJPEG Preview

**Performance Impact:**
- Player Lock System: ~0.8ms Overhead
- Face Detection (cached): ~0.5ms avg
- Debug Overlay: ~0.5ms
- **Gesamt: <2ms → FPS-Impact vernachlässigbar** ✅

**Debug Visualization (MJPEG Preview):**
- 3D Play Volume (grüner/grauer Rahmen)
- Face Detection (grünes Rechteck wenn locked)
- Hand-to-Face Verbindungen (grüne Linien)
- Session State Banner (oben, farbcodiert)
- Volume Violations (rote/magenta Markierungen für ignorierte Detections)
- Aktivierbar via Config-Flags

---

### Statische Gesten (Implementiert) ✅

Alle regelbasiert auf 21 Hand-Landmarks (MCP + Angle Erkennung).

| Geste | OSC String | Finger | Emoji | Status |
|-------|------------|--------|-------|--------|
| FIVE | "FIVE" | Alle 5 offen | 🖐️ | ✅ |
| FIST | "FIST" | Alle geschlossen | ✊ | ✅ |
| THUMBS_UP | "THUMBS_UP" | Nur Daumen | 👍 | ✅ |
| PEACE | "PEACE" | Zeige + Mittel | ✌️ | ✅ |
| POINTING | "POINTING" | Nur Zeigefinger | ☝️ | ✅ |
| TWO | "TWO" | Daumen + Zeige | | ✅ |
| THREE | "THREE" | Daumen + Zeige + Mittel | | ✅ |
| FOUR | "FOUR" | Alle außer Daumen | | ✅ |
| METAL | "METAL" | Zeige + Kleiner | 🤘 | ✅ |
| LOVE_YOU | "LOVE_YOU" | Daumen + Zeige + Kleiner | 🤟 | ✅ |
| VULCAN | "VULCAN" | Alle 5 offen, V-Spreizung | 🖖 | ✅ |
| CALL_ME | "CALL_ME" | Daumen + Kleiner | 🤙 | ✅ |
| MIDDLE_FINGER | "MIDDLE_FINGER" | Nur Mittelfinger | 🖕 | ✅ |
| PALM | "PALM" | Hand erkannt, keine Geste | | ✅ |
| UNKNOWN | "UNKNOWN" | Nicht erkannt | | ✅ |

## Geplante Features ⬜

### Dynamische Gesten (Phase 4)

Velocity-basiert, nutzt die bereits vorhandene Velocity aus dem Kalman Filter.

| Geste | Bedingung | Status |
|-------|-----------|--------|
| SWIPE_LEFT | FIVE + vx < -0.4 | ⬜ |
| SWIPE_RIGHT | FIVE + vx > 0.4 | ⬜ |
| SWIPE_UP | FIVE + vy < -0.4 | ⬜ |
| SWIPE_DOWN | FIVE + vy > 0.4 | ⬜ |
| PUSH | FIVE + vz > 0.3 | ⬜ |
| PUNCH | FIST + vz > 0.4 | ⬜ |

### Zweihändige Gesten (Phase 5)

Erfordert simultanes Tracking beider Hände + Abstandsberechnung.

| Geste | OSC String | Beschreibung | Status |
|-------|------------|--------------|--------|
| HEART | "HEART" | Beide Hände formen Herz | ⬜ |
| FRAME | "FRAME" | Rechteck mit Fingern | ⬜ |
| CLAP | "CLAP" | Handflächen zusammen | ⬜ |
| TIMEOUT | "TIMEOUT" | T-Form | ⬜ |
| NAMASTE | "NAMASTE" | Handflächen aneinander | ⬜ |

### Weitere geplante OSC-Adressen

| Adresse | Typ | Beschreibung | Status |
|---------|-----|--------------|--------|
| `/service/heartbeat` | float | Unix timestamp | ⬜ |
| `/service/fps` | float | Aktuelle FPS | ⬜ |
| `/hand/{id}/depth` | int | Tiefe in mm (raw) | ⬜ |
| `/hand/{id}/finger_count` | int | Gestreckte Finger (0-5) | ⬜ |
| `/hand/{id}/dynamic_gesture` | string | Dynamische Geste | ⬜ |
| `/hands/gesture` | string | Zweihändige Geste | ⬜ |
| `/hands/distance` | float | Abstand zwischen Händen | ⬜ |

## Koordinatensystem

### OSC Output (Normalisiert)
- **X:** 0.0 (links) → 1.0 (rechts)
- **Y:** 0.0 (oben) → 1.0 (unten)
- **Z:** 0.0 (0.5m nah) → 1.0 (3m fern)

### Velocity (Normalisiert pro Sekunde)
- **vx, vy, vz:** Änderung in normalisierten Einheiten pro Sekunde
- Typischer Bereich: -1.0 bis +1.0

### Unreal Engine Mapping (Empfehlung)
```
OSC X → Unreal Y (horizontal)
OSC Y → Unreal Z (vertikal, invertiert: 1-Y)
OSC Z → Unreal X (Tiefe)
```

**Hinweis:** MJPEG-Preview ist gespiegelt, OSC-Koordinaten sind nicht gespiegelt.

## Erkennungs-Architektur

```
OAK-D Pro PoE (Sensor-Only)
    │
    ├── RGB 640×360 NV12
    ├── Mono Left 640×400 GRAY8
    └── Mono Right 640×400 GRAY8
            │
            ▼
Jetson Orin Nano (TensorRT)
    │
    ├── Palm Detection → BBox
    ├── Hand Landmarks → 21 Points
    ├── Stereo Depth → Z-Coordinate
    │
    ▼
Tracking & Gesture
    │
    ├── Kalman Filter → Position + Velocity (geglättet)
    ├── Gesture FSM → Statische Gesten
    └── Haar Cascade → Face Filter
            │
            ▼
OSC Output (30 Hz, Non-Blocking)
    │
    └── /hand/{0,1}/{palm,velocity,gesture}
```
