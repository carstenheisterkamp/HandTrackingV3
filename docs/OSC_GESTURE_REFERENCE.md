# OSC Referenz - OAK-D Hand Tracking Service

**Version:** 1.1  
**Datum:** 28. Dezember 2025  
**Port:** 9000 (localhost)

## Architektur: Non-Blocking OSC

Das OSC-Sending ist **vollständig non-blocking** und blockiert niemals die Hauptpipeline.  
**Performance-Garantie**: Das OSC-Subsystem hat **null** Einfluss auf die Pipeline-Framerate.

## Implementierte Features

### Statische Gesten

Gesendet an `/vip/{n}/hand/{side}/gesture`. Modell-unabhängig, regelbasiert auf 21 Hand-Landmarks.

| Geste | OSC String | Finger | Emoji |
|-------|------------|--------|-------|
| FIVE | "FIVE" | Alle 5 offen | 🖐️ |
| FIST | "FIST" | Alle geschlossen | ✊ |
| THUMBS_UP | "THUMBS_UP" | Nur Daumen | 👍 |
| PEACE | "PEACE" | Zeige + Mittel | ✌️ |
| POINTING | "POINTING" | Nur Zeigefinger | ☝️ |
| TWO | "TWO" | Daumen + Zeige | |
| THREE | "THREE" | Daumen + Zeige + Mittel | |
| FOUR | "FOUR" | Alle außer Daumen | |
| METAL | "METAL" | Zeige + Kleiner | 🤘 |
| LOVE_YOU | "LOVE_YOU" | Daumen + Zeige + Kleiner | 🤟 |
| VULCAN | "VULCAN" | Alle 5 offen, V-Spreizung Mittel↔Ring | 🖖 |
| CALL_ME | "CALL_ME" | Daumen + Kleiner | 🤙 |
| MIDDLE_FINGER | "MIDDLE_FINGER" | Nur Mittelfinger | 🖕 |
| unknown | "unknown" | Nicht erkannt | |

### OSC-Adressen (Implementiert)

| Adresse | Typ | Beschreibung |
|---------|-----|--------------|
| `/service/started` | int | 1 beim Start |
| `/service/heartbeat` | float | Unix timestamp (jede Sekunde) |
| `/service/fps` | float | Aktuelle Pipeline-FPS |
| `/vip/{n}/hand/{side}/position` | [x, y, z] | Wrist-Position normalisiert |
| `/vip/{n}/hand/{side}/velocity` | [vx, vy, vz] | Geschwindigkeit (norm/s) |
| `/vip/{n}/hand/{side}/gesture` | string | Statische Geste |
| `/vip/{n}/hand/{side}/landmarks` | [63 floats] | 21 × (x, y, z) normalisiert |

## Geplante Features

### Zweihändige Gesten

Gesendet an `/vip/{n}/hands/gesture`. Erfordert simultanes Tracking beider Hände.

| Geste | OSC String | Beschreibung | Emoji |
|-------|------------|--------------|-------|
| HEART | "HEART" | Beide Hände formen Herz | 🫶 |
| FRAME | "FRAME" | Rechteck mit Fingern | 📷 |
| CLAP | "CLAP" | Handflächen zusammen | 👏 |
| TIMEOUT | "TIMEOUT" | T-Form | 🇹 |
| NAMASTE | "NAMASTE" | Handflächen aneinander | 🙏 |

### Dynamische Gesten

Velocity-basiert.

| Geste | Bedingung |
|-------|-----------|
| SWIPE_LEFT | Offene Hand + vx < -0.4 norm/s |
| SWIPE_RIGHT | Offene Hand + vx > 0.4 norm/s |
| SWIPE_UP | Offene Hand + vy < -0.4 norm/s |
| SWIPE_DOWN | Offene Hand + vy > 0.4 norm/s |
| PUSH | Offene Hand + vz > 150 mm/s |
| PUNCH | Faust + vz > 225 mm/s |

### Weitere geplante OSC-Adressen

| Adresse | Typ | Beschreibung |
|---------|-----|--------------|
| `/service/stopped` | int | 1 beim Stop |
| `/service/uptime` | int | Sekunden seit Start |
| `/vip/{n}/hand/{side}/depth` | int | Tiefe in mm |
| `/vip/{n}/hand/{side}/finger_count` | int | Anzahl gestreckter Finger (0-5) |
| `/vip/{n}/hand/{side}/dynamic_gesture` | string | Dynamische Geste |
| `/vip/{n}/hands/gesture` | string | Zweihändige Geste |
| `/vip/{n}/hands/distance` | float | Abstand zwischen Händen (0.0-1.0) |

## Unreal Engine Blueprint Beispiele

### Hand-Cursor
- Auf `/vip/0/hand/right/gesture` == "FIST": Trigger Click.
- Auf `/vip/0/hand/right/landmarks`: Update Cursor Position.
- Auf `/vip/0/hand/right/depth` < 500: Trigger "Close".

### Gesten-Aktionen
- Switch auf `/hand/gesture`: "FIST" → Click, "THUMBS_UP" → Confirm, etc.

### Swipe-Navigation
- Switch auf `/hand/dynamic_gesture`: "SWIPE_LEFT" → Previous, etc.

## Koordinatensystem

### OAK-D (OSC)
- X: 0 (links) → 1 (rechts)
- Y: 0 (oben) → 1 (unten)
- Z: 200mm (nah) → 5000mm (fern)

### Unreal Engine Mapping
- OSC X → Unreal Y
- OSC Y → Unreal Z (invertiert)
- OSC Z → Unreal X

**Hinweis:** Preview gespiegelt, OSC-Koordinaten nicht.

## Gesten-Übersicht

### Einhändig
| Kategorie | Gesten | Adresse |
|-----------|--------|---------|
| Zahlen | FIST, TWO-FIVE | `/hand/gesture` |
| Zeigen | POINTING, THUMBS_UP, MIDDLE_FINGER | `/hand/gesture` |
| Symbole | PEACE, METAL, LOVE_YOU, VULCAN, CALL_ME | `/hand/gesture` |
| Bewegung | SWIPE_*, PUSH, PUNCH | `/hand/dynamic_gesture` |

### Zweihändig
| Kategorie | Gesten | Adresse |
|-----------|--------|---------|
| Symbole | HEART, NAMASTE | `/hands/gesture` |
| Aktionen | CLAP, FRAME, TIMEOUT | `/hands/gesture` |

## Erkennungs-Architektur

```
Hand Tracking Model → 21 Landmarks
    ↓
Gesture Recognition (Heuristik)
    ↓
OSC Output (/hand/gesture, /hands/gesture, etc.)
