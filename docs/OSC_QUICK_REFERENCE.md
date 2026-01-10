# OSC Pfade - Quick Reference für Game Engine

## 🎮 Aktuell Implementiert (Live)

### Hand Tracking (Pro Hand)
```
/hand/0/palm           [x, y, z]        # Palm Position (0-1 normalized)
/hand/0/velocity       [vx, vy, vz]     # Velocity (mm/s)
/hand/0/gesture        [id, conf, name] # Gesture State

/hand/1/palm           [x, y, z]        # Zweite Hand (wenn erkannt)
/hand/1/velocity       [vx, vy, vz]
/hand/1/gesture        [id, conf, name]
```

**Rate:** 30 Hz konstant  
**Latenz:** <60ms Glass-to-OSC

---

## 🎯 Gesten (Statisch - Implementiert)

```
FIVE            # 🖐️  Alle 5 Finger offen
FIST            # ✊  Faust
THUMBS_UP       # 👍  Nur Daumen
POINTING        # ☝️  Nur Zeigefinger
PEACE           # ✌️  Zeige + Mittel
METAL           # 🤘  Zeige + Kleiner
LOVE_YOU        # 🤟  Daumen + Zeige + Kleiner
VULCAN          # 🖖  V-Spreizung
CALL_ME         # 🤙  Daumen + Kleiner
TWO/THREE/FOUR  # 2-4 Finger
MIDDLE_FINGER   # 🖕  Nur Mittelfinger
PALM            # Erkannt, keine Geste
```

---

## 🚀 Geplant (Phase 4+)

### Dynamische Gesten
```
/hand/0/dynamic_gesture  [string]  # SWIPE_LEFT/RIGHT, PUSH, PUNCH
```

### Player Lock Events
```
/player/enter            [id]      # Spieler betritt Spielfeld
/player/active           [id]      # Session aktiv
/player/lost             [time]    # Spieler temporär verloren
/player/exit             [id]      # Session beendet
```

### Zweihändige Gesten
```
/hands/gesture           [string]  # HEART, CLAP, FRAME, etc.
/hands/distance          [float]   # Abstand zwischen Händen
```

### Service Metrics
```
/service/fps             [float]   # Current FPS
/service/heartbeat       [float]   # Timestamp
```

---

## 📐 Koordinatensystem

| Achse | Range | Bedeutung |
|-------|-------|-----------|
| X | 0.0-1.0 | Links (0) → Rechts (1) |
| Y | 0.0-1.0 | Oben (0) → Unten (1) |
| Z | 0.0-1.0 | Nah 0.5m (0) → Fern 3m (1) |

**Velocity:** mm/s (Kalman gefiltert)

---

## 🎮 Unreal Engine Mapping (Empfohlen)

```cpp
// OSC → UE Koordinaten
Hand.Location.X = OSC_Z * 300;      // Tiefe
Hand.Location.Y = OSC_X * 800;      // Horizontal
Hand.Location.Z = (1 - OSC_Y) * 600; // Vertikal (invertiert)

Hand.Velocity = OSC_Velocity * scale;
```

---

## 📊 Message Format Beispiel

```
/hand/0/palm
  Type: fff
  Data: [0.45, 0.52, 0.34]  # (x, y, z)

/hand/0/gesture
  Type: ifs
  Data: [2, 0.95, "FIST"]    # (id, confidence, name)

/hand/0/velocity
  Type: fff
  Data: [12.3, -8.5, 45.2]   # (vx, vy, vz) mm/s
```

---

## ⚙️ Verbindung

- **IP:** 100.101.16.21 (via Tailscale)
- **Port:** 9000
- **Protocol:** OSC/UDP
- **Rate:** 30 Hz @ 33ms intervals

---

## 🔄 Versionsverlauf

| Version | Datum | Status |
|---------|-------|--------|
| 1.0 | 2025-12 | V2 (deprecated) |
| 2.0 | 2026-01 | V3 (live) |
| 2.1 | 2026-02 | Phase 4 (planned) |


