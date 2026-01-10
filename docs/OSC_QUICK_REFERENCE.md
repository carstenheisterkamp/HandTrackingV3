# OSC Referenz - Hand Tracking Service

**Version:** 2.0 (V3 Architecture)  
**Datum:** 10. Januar 2026  
**Port:** 9000 (127.0.0.1 auf Jetson)  
**Preview:** http://100.101.16.21:8080 (via Tailscale)

---

## 🎮 Aktuell Implementiert (Live) ✅

### Hand Tracking (Pro Hand)
```
/hand/0/palm           [x, y, z]        # Palm Position (0-1 normalized)
/hand/0/velocity       [vx, vy, vz]     # Velocity (Kalman gefiltert)
/hand/0/gesture        [id, conf, name] # [State-ID, Confidence, Name]

/hand/1/palm           [x, y, z]        # Zweite Hand (wenn erkannt)
/hand/1/velocity       [vx, vy, vz]
/hand/1/gesture        [id, conf, name]
```

**Rate:** 30 Hz @ 33ms intervals  
**Latenz:** <60ms Glass-to-OSC  
**Non-Blocking:** OSC hat null Einfluss auf Pipeline-FPS

---

## 🎯 Gesten - Implementiert ✅

### Statische Gesten (13 Stück)

Regelbasiert auf 21 Hand-Landmarks (MCP + Angle Erkennung).

| Geste | OSC String | Beschreibung | Emoji |
|-------|------------|--------------|-------|
| FIVE | "FIVE" | Alle 5 Finger offen | 🖐️ |
| FIST | "FIST" | Alle Finger geschlossen | ✊ |
| THUMBS_UP | "THUMBS_UP" | Nur Daumen | 👍 |
| POINTING | "POINTING" | Nur Zeigefinger | ☝️ |
| PEACE | "PEACE" | Zeige + Mittel | ✌️ |
| METAL | "METAL" | Zeige + Kleiner | 🤘 |
| LOVE_YOU | "LOVE_YOU" | Daumen + Zeige + Kleiner | 🤟 |
| VULCAN | "VULCAN" | Alle 5, V-Spreizung | 🖖 |
| CALL_ME | "CALL_ME" | Daumen + Kleiner | 🤙 |
| TWO | "TWO" | Daumen + Zeige | |
| THREE | "THREE" | Daumen + Zeige + Mittel | |
| FOUR | "FOUR" | Alle außer Daumen | |
| MIDDLE_FINGER | "MIDDLE_FINGER" | Nur Mittelfinger | 🖕 |
| PALM | "PALM" | Hand erkannt, keine Geste | |

**Erkennung:**
- Y-basierte Finger Detection: `tip.y < pip.y` = Finger oben
- X-basierte Daumen Detection: Links/Rechts-Hand unterschieden
- Debounce: 3 Frames (~100ms @ 30 FPS)
- Face Filter: Haar Cascade (0 False Positives)

---

## 🚀 Geplant (Phase 4+) ⬜

### Dynamische Gesten

Velocity-basiert, nutzt Kalman Filter Velocity.

| Geste | Bedingung | OSC Pfad |
|-------|-----------|----------|
| SWIPE_LEFT | FIVE + vx < -0.4 | `/hand/{id}/dynamic_gesture` |
| SWIPE_RIGHT | FIVE + vx > 0.4 | `/hand/{id}/dynamic_gesture` |
| SWIPE_UP | FIVE + vy < -0.4 | `/hand/{id}/dynamic_gesture` |
| SWIPE_DOWN | FIVE + vy > 0.4 | `/hand/{id}/dynamic_gesture` |
| PUSH | FIVE + vz > 0.3 | `/hand/{id}/dynamic_gesture` |
| PUNCH | FIST + vz > 0.4 | `/hand/{id}/dynamic_gesture` |

### Zweihändige Gesten

Erfordert simultanes Tracking beider Hände + Abstandsberechnung.

| Geste | Beschreibung | OSC Pfad |
|-------|--------------|----------|
| HEART | Beide Hände formen Herz | `/hands/gesture` |
| FRAME | Rechteck mit Fingern | `/hands/gesture` |
| CLAP | Handflächen zusammen | `/hands/gesture` |
| TIMEOUT | T-Form | `/hands/gesture` |
| NAMASTE | Handflächen aneinander | `/hands/gesture` |

**Zusätzlich:**
```
/hands/distance          [float]   # Abstand zwischen Händen (0-1)
```

### Player Lock System (Phase 4)

**Siehe:** `PLAYER_LOCK_DESIGN.md`

Stabiles Single-User-Tracking für Gaming:

```
/player/enter            [id]      # Spieler betritt Play Volume
/player/calibrating      [progress]# Warte auf stabile Detection (0-1)
/player/active           [id]      # Session aktiv, Gameplay enabled
/player/lost             [time]    # Player temporär verloren (Grace Period)
/player/exit             [id]      # Session beendet
```

**Features:**
- 3D Play Volume (16:9, konfigurierbar)
- Face-Anchored Hand-Zuordnung
- First-Come-First-Serve Lock
- Hand-IDs bleiben stabil während Session
- Grace Period: 3s wenn Player temporär verloren

**Performance:** <2ms Overhead (~2% @ 30 FPS)

### Service Metrics

```
/service/fps             [float]   # Current FPS
/service/heartbeat       [float]   # Unix timestamp
```

---

## 📐 Koordinatensystem

| Achse | Range | Bedeutung | Einheit |
|-------|-------|-----------|---------|
| X | 0.0-1.0 | Links → Rechts | Normalized |
| Y | 0.0-1.0 | Oben → Unten | Normalized |
| Z | 0.0-1.0 | 0.5m nah → 3m fern | Normalized |

**Velocity:** 
- mm/s (millimeter pro Sekunde)
- Kalman gefiltert (6-State Filter)
- Latenz-Kompensation: +1 Frame Prediction

**Wichtig:** 
- MJPEG Preview ist gespiegelt (Mirror-View)
- OSC Koordinaten sind NICHT gespiegelt
- X=0 ist links im echten Raum (auch wenn rechts im Preview)

---

## 🎮 Game Engine Integration

### Unreal Engine Mapping

```cpp
// OSC → UE Koordinaten
Hand.Location.X = OSC_Z * 300.0f;      // Tiefe (cm)
Hand.Location.Y = OSC_X * 800.0f;      // Horizontal (cm)
Hand.Location.Z = (1.0f - OSC_Y) * 600.0f; // Vertikal invertiert (cm)

// Velocity mit Scaling
Hand.Velocity.X = OSC_VZ * velocityScale;
Hand.Velocity.Y = OSC_VX * velocityScale;
Hand.Velocity.Z = -OSC_VY * velocityScale;
```

### Unity Mapping

```csharp
// OSC → Unity Koordinaten
transform.position = new Vector3(
    OSC_X * 8f - 4f,     // -4 bis +4 (center)
    (1f - OSC_Y) * 6f,   // 0 bis +6 (invertiert)
    OSC_Z * 3f + 0.5f    // +0.5 bis +3.5
);

// Velocity (mm/s → m/s)
velocity = new Vector3(OSC_VX, -OSC_VY, OSC_VZ) / 1000f;
```

---

## 📊 Message Format Beispiele

### Palm Position
```
Address: /hand/0/palm
Type: fff
Data: [0.45, 0.52, 0.34]  # (x, y, z)
```

### Gesture
```
Address: /hand/0/gesture
Type: ifs
Data: [2, 0.95, "FIST"]    # (id, confidence, name)
```

### Velocity
```
Address: /hand/0/velocity
Type: fff
Data: [12.3, -8.5, 45.2]   # (vx, vy, vz) in mm/s
```

### Player Event (geplant)
```
Address: /player/active
Type: i
Data: [0]  # player_id
```

---

## 🏗️ Multi-Person Handling

### Aktuell (Phase 3)

**Top-2 Selection:**
1. Palm Detection erkennt ALLE Hände
2. NMS (Non-Maximum Suppression, IoU < 0.3)
3. Top-2 nach Confidence Score
4. Ignoriert restliche Hände

**Limitation:** Hand-IDs können zwischen Personen wechseln

### Geplant (Phase 4)

**Player Lock System:**
- Play Volume Filter (nur Hände im 3D Volume)
- Face-Anchored (Haar Cascade ordnet Hände Person zu)
- First-Come-First-Serve (erste Person im Volume = Owner)
- Stable IDs bis Player Volume verlässt

**Debug Visualization:**
- 3D Volume Box im Preview (grün)
- Face Detection (grünes Rechteck)
- Hand-to-Face Verbindungen (grüne Linien)
- Session State Banner (farbcodiert)

---

## ⚙️ Verbindung & Setup

### Connection Details
- **IP:** 100.101.16.21 (via Tailscale)
- **Port:** 9000
- **Protocol:** OSC/UDP
- **Rate:** 30 Hz konstant
- **Latenz:** <60ms Glass-to-OSC

### Client Libraries

**Python:**
```python
from pythonosc import udp_client

client = udp_client.SimpleUDPClient("100.101.16.21", 9000)

# Subscribe to hand data (not needed, just listen)
```

**Unity (OSC Jack):**
```csharp
using OscJack;

OscPropertySender sender;
void Start() {
    sender = new OscPropertySender("100.101.16.21", 9000);
}
```

**Unreal (OSC Plugin):**
```
OSC Settings:
  Receive From: 100.101.16.21:9000
  Send Targets: (optional)
  
Blueprint: Add OSC Server Component
  Bind: /hand/0/palm → OnHandPalmReceived
```

---

## 🔄 Architektur & Performance

### Pipeline
```
OAK-D Pro PoE (Sensor-Only)
    │
    ├─ RGB 640×360 NV12 @ 30 FPS
    ├─ Mono Left 640×400 GRAY8
    └─ Mono Right 640×400 GRAY8
         ↓
Jetson Orin Nano (TensorRT + CUDA)
    │
    ├─ Palm Detection (FULL model: ~15ms)
    ├─ Hand Landmark (FULL model: ~15ms)
    ├─ Stereo Depth (CUDA: <1ms)
    │
    ├─ Kalman Filter [x,y,z,vx,vy,vz]
    ├─ Gesture FSM (MCP+Angle)
    └─ Haar Cascade Face Filter
         ↓
OSC Output (Non-Blocking, 30 Hz)
    │
    └─ /hand/{0,1}/{palm,velocity,gesture}
```

### Performance Garantien
- **FPS:** 25-30 konstant (mit FULL models)
- **OSC Overhead:** <0.2ms (non-blocking)
- **Drop Policy:** Pakete >50ms alt werden verworfen
- **Bewegungsglättung:** Kalman Filter (kein Jitter)

---

## 🔄 Versions-Historie

| Version | Datum | Changes | Status |
|---------|-------|---------|--------|
| 1.0 | 2025-12 | V2 Architecture | Deprecated |
| 2.0 | 2026-01-09 | V3 Architecture, 2-Hand Tracking | Live ✅ |
| 2.1 | 2026-01-10 | FULL Models, 3D Stereo Depth | Live ✅ |
| 2.2 | 2026-Q1 | Player Lock System (Phase 4) | Planned ⬜ |
| 2.3 | 2026-Q1 | Dynamic Gestures (Phase 5) | Planned ⬜ |

---

## 📚 Weitere Dokumentation

- **Vollständige Architektur:** `OPTIMAL_WORKFLOW_V3.md`
- **Player Lock Design:** `PLAYER_LOCK_DESIGN.md`
- **Model Testing:** `MODEL_TESTING.md`
- **TODO & Roadmap:** `TODO.md`



