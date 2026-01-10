# OSC Referenz - OAK-D Hand Tracking Service

**Version:** 2.0 (V3 Architecture)  
**Datum:** 10. Januar 2026  
**Port:** 9000 (localhost)

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

**Aktuelle Limitierung:** Max. 2 Hände (unabhängig von Anzahl Personen)

**Verhalten bei >2 Händen im Bild:**
1. Palm Detection erkennt ALLE Hände im Frame
2. **NMS (Non-Maximum Suppression)** mit IoU-Threshold (0.3)
   - Unterdrückt überlappende Detections
3. **Top-2 Selection:** Die 2 Hände mit höchstem Confidence-Score werden gewählt
4. Restliche Hände werden ignoriert

**Beispiel-Szenarien:**
- **2 Personen, je 2 Hände:** Tracking wählt die 2 mit höchstem Score (meist die nächsten/deutlichsten)
- **1 Person, beide Hände:** ✅ Beide werden getrackt
- **3+ Hände gleichzeitig:** Nur die 2 besten werden verfolgt

**Empfehlung für Multi-User Games:**
- Nutze räumliche Trennung (z.B. linke/rechte Bildhälfte)
- Oder implementiere zusätzliche Filterung basierend auf Z-Tiefe (näheste 2 Hände)

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
