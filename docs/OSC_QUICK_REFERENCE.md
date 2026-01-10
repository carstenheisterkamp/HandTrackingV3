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

## 📏 Physisches Setup & Play Volume

### Kamera-Setup
- **Position:** Auf Stativ unter Display/Projektionsfläche
- **Höhe:** Ca. gleiche Distanz zum Player wie Display
- **Ausrichtung:** Frontal auf Spieler gerichtet

### Play Volume (3D Bereich)
**Default-Konfiguration (90%):**
- **Horizontal (X):** 90% der Kamera-Breite (5% Margin links/rechts)
- **Vertikal (Y):** 90% der Kamera-Höhe (5% Margin oben/unten)
- **Tiefe (Z):** 0.5m - 2.5m von der Kamera

### Boden-Markierung für Play Volume

**Berechnung der Bodenfläche:**

Bei **127° FoV** (OAK-D Pro) und Kamera auf Display-Höhe:

| Abstand | Breite (ca.) | Höhe (ca.) | Play Volume 90% |
|---------|--------------|------------|-----------------|
| 0.5m | 0.8m | 0.45m | 0.72m × 0.40m |
| 1.0m | 1.6m | 0.9m | 1.44m × 0.81m |
| 1.5m | 2.4m | 1.35m | 2.16m × 1.22m |
| 2.0m | 3.2m | 1.8m | 2.88m × 1.62m |
| 2.5m | 4.0m | 2.25m | 3.60m × 2.03m |

**Empfohlene Markierung auf dem Boden:**
```
Nähere Linie (0.5m):  ~0.7m × 0.4m Rechteck
Fernere Linie (2.5m): ~3.6m × 2.0m Rechteck

         Kamera/Display
              ↓
    ┌─────────────────────┐
    │ ← 0.7m breit →     │  0.5m Entfernung
    └─────────────────────┘
           ▼ ▼ ▼
    ┌─────────────────────────────┐
    │  ← 3.6m breit →            │  2.5m Entfernung
    └─────────────────────────────┘
```

**Praktischer Tipp:**
- Markiere mit Klebeband ein **Rechteck 2m × 1.5m** auf dem Boden
- Zentrumslinie bei ca. 1.5m von der Kamera
- Das gibt Spielern visuelles Feedback für optimale Position
- Entspricht dem **Sweet Spot** für beste Tracking-Qualität

---

## 🎮 Game Engine Integration (Unreal Engine)

### Koordinaten-Transformation: OSC → Unreal Engine

```cpp
// OSC → UE World Space (cm)
Hand.Location.X = OSC_Z * 300.0f;          // Tiefe: 0-3m → 0-300cm
Hand.Location.Y = OSC_X * 800.0f;          // Horizontal: 0-1 → 0-800cm
Hand.Location.Z = (1.0f - OSC_Y) * 600.0f; // Vertikal: 0-1 → 600-0cm (invertiert)

// Velocity Transformation (mm/s → cm/s)
Hand.Velocity.X = OSC_VZ * 0.1f;  // Tiefe
Hand.Velocity.Y = OSC_VX * 0.1f;  // Horizontal
Hand.Velocity.Z = -OSC_VY * 0.1f; // Vertikal (invertiert)
```

**Warum Z invertiert?**
- OSC Y-Koordinate: `0.0 = oben, 1.0 = unten` (Bildschirm-Koordinaten)
- Unreal Z-Koordinate: `0 = unten, höher = oben` (World Space)
- `(1.0f - OSC_Y)` spiegelt die Achse: oben (0) → oben (600), unten (1) → unten (0)

### OSC Empfang in Unreal (C++)

**1. OSC Plugin aktivieren:**
- Plugins → OSC → Enable
- Project Settings → Plugins → OSC

**2. OSC Server Component hinzufügen:**

```cpp
// YourActor.h
#include "OSCServer.h"
#include "OSCMessage.h"

UCLASS()
class YOURGAME_API AHandTracker : public AActor
{
    GENERATED_BODY()

public:
    AHandTracker();
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OSC")
    FString OSCAddress = TEXT("100.101.16.21");
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OSC")
    int32 OSCPort = 9000;
    
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Hand Tracking")
    FVector Hand0Position;
    
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Hand Tracking")
    FVector Hand0Velocity;
    
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Hand Tracking")
    FString Hand0Gesture;

protected:
    virtual void BeginPlay() override;
    
private:
    UPROPERTY()
    UOSCServer* OSCServer;
    
    UFUNCTION()
    void OnPalmReceived(const FOSCMessage& Message, const FString& IPAddress, int32 Port);
    
    UFUNCTION()
    void OnVelocityReceived(const FOSCMessage& Message, const FString& IPAddress, int32 Port);
    
    UFUNCTION()
    void OnGestureReceived(const FOSCMessage& Message, const FString& IPAddress, int32 Port);
};
```

**3. OSC Message Handler implementieren:**

```cpp
// YourActor.cpp
void AHandTracker::BeginPlay()
{
    Super::BeginPlay();
    
    // Create OSC Server
    OSCServer = NewObject<UOSCServer>(this);
    OSCServer->Listen(OSCAddress, OSCPort);
    
    // Bind OSC Addresses
    FOSCAddress PalmAddress;
    PalmAddress.PushContainer("hand");
    PalmAddress.PushContainer("0");
    PalmAddress.PushMethod("palm");
    OSCServer->BindEventToOnOSCAddressPatternMatchesPath(PalmAddress, 
        FOnOSCMessageReceived::CreateUObject(this, &AHandTracker::OnPalmReceived));
    
    // Repeat for velocity and gesture...
}

void AHandTracker::OnPalmReceived(const FOSCMessage& Message, const FString& IPAddress, int32 Port)
{
    if (Message.GetArguments().Num() >= 3)
    {
        float OSC_X = Message.GetArguments()[0].GetFloat();
        float OSC_Y = Message.GetArguments()[1].GetFloat();
        float OSC_Z = Message.GetArguments()[2].GetFloat();
        
        // Transform to Unreal coordinates
        Hand0Position.X = OSC_Z * 300.0f;
        Hand0Position.Y = OSC_X * 800.0f;
        Hand0Position.Z = (1.0f - OSC_Y) * 600.0f;
        
        UE_LOG(LogTemp, Log, TEXT("Hand 0 Position: %s"), *Hand0Position.ToString());
    }
}

void AHandTracker::OnVelocityReceived(const FOSCMessage& Message, const FString& IPAddress, int32 Port)
{
    if (Message.GetArguments().Num() >= 3)
    {
        float OSC_VX = Message.GetArguments()[0].GetFloat();
        float OSC_VY = Message.GetArguments()[1].GetFloat();
        float OSC_VZ = Message.GetArguments()[2].GetFloat();
        
        // Transform velocity (mm/s → cm/s)
        Hand0Velocity.X = OSC_VZ * 0.1f;
        Hand0Velocity.Y = OSC_VX * 0.1f;
        Hand0Velocity.Z = -OSC_VY * 0.1f;
    }
}

void AHandTracker::OnGestureReceived(const FOSCMessage& Message, const FString& IPAddress, int32 Port)
{
    if (Message.GetArguments().Num() >= 3)
    {
        // int32 GestureID = Message.GetArguments()[0].GetInt();
        // float Confidence = Message.GetArguments()[1].GetFloat();
        FString GestureName = Message.GetArguments()[2].GetString();
        
        Hand0Gesture = GestureName;
        
        // Trigger gameplay events based on gesture
        if (GestureName == TEXT("FIST"))
        {
            // Grab action
        }
        else if (GestureName == TEXT("FIVE"))
        {
            // Release action
        }
    }
}
```

### Blueprint-freundliche Variante

```cpp
// Event Dispatcher in Header
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnHandGestureChanged, FString, GestureName);

UPROPERTY(BlueprintAssignable, Category = "Hand Tracking")
FOnHandGestureChanged OnGestureChanged;

// In OnGestureReceived
OnGestureChanged.Broadcast(GestureName);
```

**In Blueprint dann:**
- Event: On Gesture Changed → Switch on String → Execute Actions

### Performance-Tipips

- OSC läuft @ 30 Hz (33ms intervals)
- Nutze Interpolation für smooth 60 FPS Rendering:
  ```cpp
  FVector SmoothedPosition = FMath::VInterpTo(
      CurrentPosition, 
      Hand0Position, 
      DeltaTime, 
      10.0f  // Interp Speed
  );
  ```
- Cache Gesture-States, feuere Events nur bei Änderungen
- Nutze `Hand0Velocity` für Prediction/Motion Blur

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
- **IP:** 100.86.141.97 (MacBook via Tailscale - Testing)
- **Port:** 9000
- **Protocol:** OSC/UDP
- **Rate:** 30 Hz konstant
- **Latenz:** <60ms Glass-to-OSC

**Test Setup:**
- Jetson sendet OSC an MacBook für Testing
- Später: Ändern zu Unreal Engine IP oder zurück zu localhost
- Keine Authentifizierung nötig
- Fire-and-Forget (keine ACKs)

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



