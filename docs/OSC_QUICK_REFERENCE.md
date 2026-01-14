# OSC Referenz - Hand Tracking Service

**Version:** 2.0 (V3 Architecture)  
**Datum:** 10. Januar 2026  
**Port:** 9000 (127.0.0.1 auf Jetson)  
**Preview:** http://100.101.16.21:8080 (via Tailscale)

---

## 🎮 Aktuell Implementiert (Live) ✅

### Hand Tracking (Pro Hand)

**OSC-Reihenfolge (optimiert für Client-Effizienz):**
```
1️⃣ /hand/0/confidence     [palm, gesture, landmark]  ⬅️ ZUERST (Early Rejection)
2️⃣ /hand/0/velocity       [vx, vy, vz]               (Prediction während Palm-Warten)
3️⃣ /hand/0/palm           [x, y, z]                  ⬅️ Hauptdaten (0-1 normalized)
4️⃣ /hand/0/delta          [dx, dy, dz]               (Acceleration)
5️⃣ /hand/0/gesture        [id, conf, name]           (State-ID, Confidence, Name)

# Zweite Hand (wenn erkannt)
/hand/1/confidence     [palm, gesture, landmark]
/hand/1/velocity       [vx, vy, vz]
/hand/1/palm           [x, y, z]
/hand/1/delta          [dx, dy, dz]
/hand/1/gesture        [id, conf, name]
```

**Warum diese Reihenfolge?**
- **Confidence zuerst:** Client kann Low-Quality-Daten sofort verwerfen (Performance!)
- **Velocity vor Palm:** Client kann Prediction starten während Palm-Daten ankommen
- **Delta & Gesture nachher:** Sekundäre Daten, Reihenfolge egal

## 🎯 Performance-Garantie für Clients

**GUARANTEED OSC RATE: 28 Hz (35.7ms fixed interval)**

Der Service garantiert **stabile 28 Hz OSC-Output**, unabhängig von Camera-FPS-Schwankungen:

| Parameter | Wert | Beschreibung |
|-----------|------|--------------|
| **OSC Rate** | **28 Hz** | Garantierte Minimum-Rate (frame-paced) |
| **Interval** | **35.7ms** | Maximales Intervall zwischen OSC-Paketen |
| **Latency** | <60ms | Glass-to-OSC (95th percentile) |
| **Jitter** | <2ms | OSC Timing-Varianz (frame-pacing) |
| **Drop Policy** | >50ms | Alte Pakete werden verworfen |

**Client-Implementierung:**
```cpp
// Client kann sich auf diese Werte verlassen:
const float OSC_RATE = 28.0f;           // Hz (guaranteed minimum)
const float OSC_INTERVAL = 35.7f;       // ms (max interval)
const float SMOOTHING_FACTOR = 0.3f;    // Für 28 Hz optimal
const float PREDICTION_TIME = 35.7f;    // ms (1 frame ahead)
```

**Warum 28 Hz und nicht 30 Hz?**
- Camera liefert 28-30 FPS (je nach TensorRT-Last)
- 28 Hz ist das **garantierte Minimum** unter Last
- Frame-Pacing im OSC-Sender eliminiert Jitter
- Client kann Smoothing/Prediction auf **stabile 28 Hz** optimieren

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

## 📐 Koordinatensystem & Physisches Referenz-Volumen

### 🎯 Normalisierungs-Basis: Game Volume (Physische Referenz)

**WICHTIG:** Die Normalisierung (0.0-1.0) bezieht sich auf ein **definiertes physisches Volumen**!

**Unser Game Volume (Code: `include/core/PlayVolume.hpp`):**
```cpp
// Physisches Referenz-Volumen für Normalisierung
PlayVolume {
    // X/Y: Kamera Field-of-View @ 2m Spieler-Abstand
    minX = 0.0f;     // 100% Kamera-Breite
    maxX = 1.0f;     // → ~3.2m physisch @ 2m
    minY = 0.0f;     // 100% Kamera-Höhe  
    maxY = 1.0f;     // → ~1.8m physisch @ 2m
    
    // Z: Definierter Tiefenbereich (Stereo Depth)
    minZ = 1200mm;   // 1.2m von Kamera (Arm ausgestreckt)
    maxZ = 2800mm;   // 2.8m von Kamera (Arm am Körper)
}
```

**Das bedeutet konkret:**

| Achse | OSC 0-1 | Physisches Referenz-Volumen | Bei 2m Spieler-Abstand |
|-------|---------|----------------------------|------------------------|
| **X** | 0.0-1.0 | 100% Kamera FoV horizontal | ~3.2m Breite (127° FoV) |
| **Y** | 0.0-1.0 | 100% Kamera FoV vertikal | ~1.8m Höhe (nach 16:9) |
| **Z** | 0.0-1.0 | 1.2m - 2.8m absolut | 1.6m Tiefenbereich |

### Normalisierungs-Formeln

```cpp
// X/Y: Kamera-basiert (abhängig von FoV und Abstand)
X_normalized = X_pixel / ImageWidth;   // 0-640px → 0-1
Y_normalized = Y_pixel / ImageHeight;  // 0-360px → 0-1

// Z: Tiefenbereich-basiert (fest definiert)
Z_normalized = (Z_mm - 1200) / (2800 - 1200)
             = (Z_mm - 1200) / 1600

// Beispiele:
  1.2m (1200mm) → Z = 0.0 (minZ, Arm ausgestreckt)
  2.0m (2000mm) → Z = 0.5 (Spieler-Position)
  2.8m (2800mm) → Z = 1.0 (maxZ, Arm am Körper)
```

### OSC Output (Normalisiert auf Referenz-Volumen)

| Achse | Range | Bedeutung | Physische Referenz |
|-------|-------|-----------|-------------------|
| X | 0.0-1.0 | Links → Rechts | 0 = linker Bildrand, 1 = rechter Bildrand |
| Y | 0.0-1.0 | Oben → Unten | 0 = oberer Bildrand, 1 = unterer Bildrand |
| Z | 0.0-1.0 | Nah → Fern | 0 = 1.2m (minZ), 1 = 2.8m (maxZ) |

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

### 🎯 Konzept: Zwei-Stufen-Mapping

**Stufe 1: Physisch → Normalisiert (auf Jetson)**
```
Physisches Game Volume (fest definiert):
├─ X: 0-3.2m Breite (Kamera FoV @ 2m)
├─ Y: 0-1.8m Höhe (Kamera FoV @ 2m)  
└─ Z: 1.2m-2.8m Tiefe (definierter Range)

        ↓ Normalisierung (0-1)

OSC Output (dimensionslos):
├─ X: 0.0 - 1.0
├─ Y: 0.0 - 1.0
└─ Z: 0.0 - 1.0
```

**Stufe 2: Normalisiert → Virtuell (in Unreal Engine)**
```
OSC (dimensionslos):
├─ X: 0.0 - 1.0
├─ Y: 0.0 - 1.0
└─ Z: 0.0 - 1.0

        ↓ Skalierung (beliebig!)

Virtuelles UE Volume (frei wählbar):
├─ X: 0 - 10m   (oder 100m, oder 1cm...)
├─ Y: 0 - 20m   (oder 200m, oder 2cm...)
└─ Z: 0 - 5m    (oder 50m, oder 5cm...)
```

**Warum zwei Stufen?**

✅ **Jetson-seitig:**
- Definiertes physisches Referenz-Volumen (Game Volume)
- Stabile Kalibrierung (1.2m-2.8m bleibt konstant)
- Unabhängig von Game Engine

✅ **Game Engine-seitig:**
- Freie Skalierung ohne Re-Kalibrierung
- Gleiche OSC-Daten für verschiedene Spiele
- 1m physisch = X m virtuell (X frei wählbar!)

**Beispiel:**
```
Physisch (Jetson):
  Hand bewegt sich von 1.2m zu 2.8m (1.6m Bewegung)
  → OSC sendet Z: 0.0 → 1.0

Virtuell (UE Game A):
  VolumeSize = 160cm → Hand bewegt sich 1.6m (1:1)

Virtuell (UE Game B):
  VolumeSize = 1600cm → Hand bewegt sich 16m (10:1)

Virtuell (UE Game C):
  VolumeSize = 16cm → Hand bewegt sich 16cm (1:10)
```

### Koordinaten-Transformation: Flexibles Volume-Mapping

**OSC sendet normalisierte Koordinaten (0.0 - 1.0):**
- Bezogen auf **physisches Game Volume** (Jetson-seitig definiert)
- Unabhängig von virtuellen Dimensionen
- Flexibles Mapping auf **beliebige virtuelle Größen**
- **1m physisch kann 100m virtuell sein** oder jede andere Größe!

**Mapping-Formel:**
```cpp
// OSC (0-1) → Physisches Referenz → Virtuelles Volume
Virtual_Position = VolumeOrigin + (OSC_Value * VirtualVolumeSize)

// Beispiel Z-Achse:
// OSC Z=0.5 → 50% von 1.6m physisch = 0.8m + 1.2m = 2.0m real
// → 50% von VirtualVolumeSize in UE
```

### Koordinaten-Transformation: Flexibles Volume-Mapping

**Methode 1: Direkte Skalierung (Einfach)**

```cpp
// Define your virtual play volume size (in UE units, usually cm)
FVector VolumeSize(1000.0f, 2000.0f, 500.0f);  // 10m × 20m × 5m virtuell
FVector VolumeOrigin(0.0f, 0.0f, 0.0f);        // Startpunkt

// OSC → UE World Space
Hand.Location.X = VolumeOrigin.X + (OSC_Z * VolumeSize.X);          // Tiefe
Hand.Location.Y = VolumeOrigin.Y + (OSC_X * VolumeSize.Y);          // Horizontal
Hand.Location.Z = VolumeOrigin.Z + ((1.0f - OSC_Y) * VolumeSize.Z); // Vertikal (invertiert)

// Velocity: Skaliert mit Volume-Größe
// OSC Velocity in mm/s → UE Velocity in cm/s
float VelocityScaleX = VolumeSize.X / 1600.0f;  // 1600mm = physische Z-Range
float VelocityScaleY = VolumeSize.Y / 3200.0f;  // ~3200mm = physische X-Range @ 2m
float VelocityScaleZ = VolumeSize.Z / 1800.0f;  // ~1800mm = physische Y-Range @ 2m

Hand.Velocity.X = OSC_VZ * VelocityScaleX * 0.1f;  // Tiefe
Hand.Velocity.Y = OSC_VX * VelocityScaleY * 0.1f;  // Horizontal
Hand.Velocity.Z = -OSC_VY * VelocityScaleZ * 0.1f; // Vertikal (invertiert)
```

**Methode 2: Box Component als Referenz (Empfohlen)**

```cpp
// In Unreal: Erstelle Box Component "PlayVolumeBox" im Level
// Größe: Beliebig! (z.B. 1000×2000×500 für 10m×20m×5m)

UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Hand Tracking")
UBoxComponent* PlayVolumeBox;

void AHandTracker::OnPalmReceived(const FOSCMessage& Message, const FString& IPAddress, int32 Port)
{
    if (Message.GetArguments().Num() >= 3 && PlayVolumeBox)
    {
        float OSC_X = Message.GetArguments()[0].GetFloat();
        float OSC_Y = Message.GetArguments()[1].GetFloat();
        float OSC_Z = Message.GetArguments()[2].GetFloat();
        
        // Get box bounds (automatisch aus Editor-Einstellungen)
        FVector BoxExtent = PlayVolumeBox->GetScaledBoxExtent();
        FVector BoxOrigin = PlayVolumeBox->GetComponentLocation();
        
        // Map OSC (0-1) auf Box-Volumen
        Hand0Position.X = BoxOrigin.X + (OSC_Z * 2.0f - 1.0f) * BoxExtent.X;
        Hand0Position.Y = BoxOrigin.Y + (OSC_X * 2.0f - 1.0f) * BoxExtent.Y;
        Hand0Position.Z = BoxOrigin.Z + ((1.0f - OSC_Y) * 2.0f - 1.0f) * BoxExtent.Z;
        
        UE_LOG(LogTemp, Log, TEXT("Hand Position: %s"), *Hand0Position.ToString());
    }
}
```

**Warum Y invertiert?**
- OSC Y-Koordinate: `0.0 = oben, 1.0 = unten` (Kamera/Bildschirm)
- Unreal Z-Koordinate: `0 = unten, höher = oben` (World Space)
- `(1.0f - OSC_Y)` spiegelt die Achse

### Praktische Skalierungs-Beispiele

**Beispiel 1: Realistisches 1:1 Mapping**
```cpp
// Physisches Game Volume: 1.6m Tiefe × 3.2m Breite × 1.8m Höhe
// Virtuelles Volume:       1.6m Tiefe × 3.2m Breite × 1.8m Höhe (1:1)

FVector VolumeSize(160.0f, 320.0f, 180.0f);  // in cm, exakt physisch

Hand.Location = VolumeOrigin + FVector(
    OSC_Z * 160.0f,
    OSC_X * 320.0f,
    (1.0f - OSC_Y) * 180.0f
);

// → Spieler-Hand bewegt sich 1:1 mit virtueller Hand
```

**Beispiel 2: "Giant Mode" - 100× Skalierung**
```cpp
// Physisches Game Volume: 1.6m Tiefe
// Virtuelles Volume:       160m Tiefe (100× größer!)

FVector VolumeSize(16000.0f, 32000.0f, 18000.0f);  // 160m × 320m × 180m

Hand.Location = VolumeOrigin + FVector(
    OSC_Z * 16000.0f,
    OSC_X * 32000.0f,
    (1.0f - OSC_Y) * 18000.0f
);

// → 1cm Handbewegung = 1m virtuelle Bewegung!
// → Perfekt für riesige Welten, präzise Kontrolle
```

**Beispiel 3: "Microscope Mode" - 0.01× Skalierung**
```cpp
// Physisches Game Volume: 1.6m Tiefe
// Virtuelles Volume:       1.6cm Tiefe (100× kleiner!)

FVector VolumeSize(1.6f, 3.2f, 1.8f);  // 1.6cm × 3.2cm × 1.8cm

Hand.Location = VolumeOrigin + FVector(
    OSC_Z * 1.6f,
    OSC_X * 3.2f,
    (1.0f - OSC_Y) * 1.8f
);

// → 1m Handbewegung = 1cm virtuelle Bewegung
// → Perfekt für Mikroskop-Simulation, Präzisions-Arbeit
```

**Beispiel 4: Asymmetrische Skalierung**
```cpp
// Tiefe: 10× größer (16m)
// Breite: 5× größer (16m)
// Höhe: 1:1 (1.8m)

FVector VolumeSize(1600.0f, 1600.0f, 180.0f);

Hand.Location = VolumeOrigin + FVector(
    OSC_Z * 1600.0f,   // 10× Tiefe
    OSC_X * 1600.0f,   // 5× Breite
    (1.0f - OSC_Y) * 180.0f  // 1:1 Höhe
);

// → Verschiedene Achsen unterschiedlich skaliert
// → Nützlich für nicht-kubische Spielwelten
```

### Volume-Visualisierung in Unreal Editor

**Schritt 1: Box Component erstellen**
```cpp
// In BeginPlay() oder Constructor
PlayVolumeBox = CreateDefaultSubobject<UBoxComponent>(TEXT("PlayVolume"));
PlayVolumeBox->SetBoxExtent(FVector(500.0f, 1000.0f, 250.0f));  // Halbe Größe!
PlayVolumeBox->SetCollisionEnabled(ECollisionEnabled::NoCollision);
PlayVolumeBox->SetHiddenInGame(false);  // Im Editor sichtbar
PlayVolumeBox->ShapeColor = FColor::Green;
```

**Schritt 2: Im Editor anpassen**
- Select "PlayVolumeBox" Component
- Adjust Scale/Size im Details Panel
- Move/Rotate nach Bedarf
- **Größe ist flexibel!** (1m bis 1000m)

**Schritt 3: Debug-Visualisierung**
```cpp
void AHandTracker::DrawDebugVolume()
{
    if (PlayVolumeBox)
    {
        FVector Extent = PlayVolumeBox->GetScaledBoxExtent();
        FVector Origin = PlayVolumeBox->GetComponentLocation();
        
        // Draw box outline
        DrawDebugBox(
            GetWorld(),
            Origin,
            Extent,
            FColor::Green,
            false,  // Persistent
            -1.0f,  // Lifetime
            0,      // Depth priority
            2.0f    // Thickness
        );
        
        // Draw current hand position
        if (Hand0Detected)
        {
            DrawDebugSphere(
                GetWorld(),
                Hand0Position,
                10.0f,
                12,
                FColor::Red,
                false,
                -1.0f
            );
        }
    }
}
```

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

### Delta (Acceleration)
```
Address: /hand/0/delta
Type: fff
Data: [5.2, -3.1, 12.8]    # (dx, dy, dz) in mm/s²
Info: Change in velocity (acceleration)
      Useful for: Impact detection, sudden stops, momentum-based interactions
      Positive = speeding up, Negative = slowing down
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



