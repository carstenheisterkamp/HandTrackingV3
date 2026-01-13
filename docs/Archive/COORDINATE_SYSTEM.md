# Koordinatensystem & Play Volume - Übersicht

**Stand:** 2026-01-10  
**Code-Referenz:** `include/core/PlayVolume.hpp`

---

## 🎯 Koordinatensystem-Eichung

### OSC Output (Normalized 0-1)

| Achse | OSC Range | Reale Entfernung | Mapping |
|-------|-----------|------------------|---------|
| **X** | 0.0 - 1.0 | Links → Rechts | Bildabhängig |
| **Y** | 0.0 - 1.0 | Oben → Unten | Bildabhängig |
| **Z** | 0.0 - 1.0 | **1.2m → 2.8m** | Linear gemappt |

### Z-Koordinate Berechnung (Game Volume für 2m Spieler)

```
Z_normalized = (Z_mm - 1200) / (2800 - 1200)
             = (Z_mm - 1200) / 1600

Beispiele:
  1.2m (1200mm) → Z = 0.0  (Arm voll ausgestreckt)
  1.6m (1600mm) → Z = 0.25
  2.0m (2000mm) → Z = 0.5  (Spieler-Position)
  2.4m (2400mm) → Z = 0.75
  2.8m (2800mm) → Z = 1.0  (Arm am Körper)
```

**Wichtig:** 
- Werte < 1.2m werden als 0.0 geclampt
- Werte > 2.8m werden als 1.0 geclampt

---

## 📷 Kamera-Montage & Höhe

### Optimale Kamerahöhe für 2m Spieler

**Setup-Anforderungen:**
- Display: 220cm × 125cm
- Spieler: 165-185cm Körpergröße, steht 2m entfernt
- Hand-Tracking: Von Hüfte (100cm) bis über Kopf (220cm)

**Kamera FoV @ 2m Entfernung:**
```
OAK-D Pro: 127° horizontal FoV
          ~90° vertikal FoV (nach 16:9 Crop)

@ 2m Abstand sieht Kamera:
├─ Horizontal: 3.2m Breite
└─ Vertikal:   1.8m Höhe
```

**Kamerahöhen-Berechnung:**

| Kamerahöhe | Sichtbereich (@ 2m) | Eignung |
|------------|---------------------|---------|
| 80cm | 0cm - 170cm | ⚠️ Zu niedrig, schneidet Kopf ab |
| 95cm | 5cm - 185cm | ✅✅ **OPTIMAL für Display-Unterkante** |
| 110cm | 20cm - 200cm | ✅ Gut, aber höher als Display |
| 140cm | 50cm - 230cm | ✅ Ideal, aber nicht möglich (Display) |

**Constraint: Kamera MUSS unter Display (max. 95cm)**

```
Bei 95cm Kamerahöhe (Display-Unterkante):
                       
    230cm ┐           
    220cm │  ↑ Hand über Kopf (knapp außerhalb)
    200cm │  
    185cm ├─ Oberkante Sichtfeld (mit 10° Neigung)
    180cm │  ↑ Kopf ✅
    160cm │  
    140cm │  ↑ Schulter ✅
    120cm │  
    100cm │  ↑ Hüfte / Brust ✅
     95cm ├─ KAMERA ◄─── Hier montieren (unter Display)
     80cm │
     60cm │
     40cm │
     20cm │
      5cm ├─ Unterkante Sichtfeld (mit 10° Neigung)
      0cm ┘  Boden
```

**WICHTIG bei 95cm Höhe:**
- **10-15° nach oben neigen** erforderlich!
- Sonst wird Hand-über-Kopf abgeschnitten
- Mit Neigung: Oberkante bei ~185-200cm → ausreichend!

### Montage-Optionen (bei 95cm Display-Unterkante)

**Option A: Unter Display @ 95cm (EINZIGE OPTION)**
```
Vorteile:
✅ Passt unter Display
✅ Kamera versteckt hinter Display-Rahmen
✅ Einfache Montage am Stativ/Halterung

Nachteile:
⚠️ Niedriger als ideal (95cm statt 140cm)
⚠️ Starke Neigung nötig (10-15°)
⚠️ Hand-über-Kopf am oberen Bildrand

CRITICAL Setup:
├─ Höhe: 95cm über Boden (Display-Unterkante)
├─ Position: Zentriert unter Display
├─ Neigung: 10-15° nach oben ⚠️ WICHTIG!
└─ Abstand: 2.0m zum Spieler

Mit 15° Neigung @ 2m:
├─ Unterkante: ~5cm (fast Bodenhöhe)
├─ Zentrum:    ~95cm (Brust/Bauch Höhe)
└─ Oberkante:  ~185-200cm (knapp über Kopf)
```

**Warum 10-15° Neigung kritisch ist:**
```
Ohne Neigung (0°):
  95cm ± 90cm = 5cm - 185cm
  → Hand über Kopf NICHT sichtbar ❌

Mit 10° Neigung:
  Oberkante verschiebt sich zu ~195cm
  → Hand über Kopf gerade so sichtbar ⚠️

Mit 15° Neigung:
  Oberkante verschiebt sich zu ~205cm
  → Hand über Kopf gut sichtbar ✅
```

### Praktische Montage-Anleitung (für 95cm Kamerahöhe)

**Benötigt:**
- Stativ oder Wandhalterung mit Neigungsverstellung
- Winkelmesser oder Smartphone-App (z.B. "Wasserwaage")
- Maßband
- Klebeband für Markierungen

**Schritt-für-Schritt:**

1. **Höhe einstellen**
   ```
   Maßband von Boden: 95cm markieren
   Kamera-Zentrum GENAU auf diese Höhe
   (Tipp: Display-Unterkante als Referenz nutzen)
   ```

2. **Horizontale Position**
   ```
   Kamera zentriert vor Spieler-Mittellinie
   2.0m Abstand mit Maßband von Kamera zur Spieler-Position
   ```

3. **CRITICAL: Neigung einstellen (10-15°)**
   ```
   Methode A - Mit Winkelmesser/App:
   ├─ Smartphone auf Kamera legen
   ├─ Wasserwaage-App öffnen
   ├─ Kamera kippen bis 12-15° angezeigt wird
   └─ Festschrauben
   
   Methode B - Visuell (ohne Werkzeug):
   ├─ Spieler @ 2m Position stellen
   ├─ Spieler Hand über Kopf heben (~210cm)
   ├─ Kamera kippen bis Hand im oberen Bildrand sichtbar
   └─ Preview prüfen: http://100.101.16.21:8080
   
   Methode C - Zielpunkt (genau):
   ├─ Markiere 210cm Höhe @ 2m Entfernung
   ├─ Oberkante des Kamera-Sichtfelds sollte dorthin zeigen
   └─ Berechnung: tan(α) = (210-95) / 200 → α ≈ 30°/2 ≈ 12°
   ```

4. **Verifikation im Preview**
   ```
   http://100.101.16.21:8080 öffnen
   
   Spieler stellt sich @ 2m Position:
   ├─ Kopf (180cm) sollte bei ~40-50% von oben sein
   ├─ Schulter (140cm) bei ~55-65% von oben
   ├─ Hüfte (100cm) bei ~75-85% von oben
   └─ Hand über Kopf (210cm) bei ~5-15% von oben ✅ WICHTIG!
   
   Wenn Hand-über-Kopf abgeschnitten → Neigung auf 15° erhöhen!
   Wenn zu viel Boden sichtbar → Neigung auf 10° reduzieren
   ```

### Feinabstimmung (bei 95cm Kamerahöhe)

**Wenn Hand-über-Kopf abgeschnitten wird:**
```
→ Neigung auf 15° erhöhen (statt 10°) ⚠️ WICHTIG!
→ Alternativ: Spieler 20cm weiter zurück (2.2m statt 2m)
→ NICHT Kamera niedriger! (95cm ist schon Minimum)
```

**Wenn zu viel Boden/Füße sichtbar sind:**
```
→ Neigung auf 10° reduzieren (von 15°)
→ Das ist OK - Boden stört nicht beim Hand-Tracking
```

**Wenn Spieler zu groß für Frame (>185cm):**
```
→ Spieler weiter zurück (2.2-2.5m)
→ Neigung auf 15° erhöhen
→ Akzeptieren: Sehr große Personen (>190cm) schwierig bei 95cm Kamerahöhe
```

**Wenn Spieler zu klein für Frame (<165cm):**
```
→ Perfekt! Bei 95cm Höhe ideal für kleinere Personen
→ Neigung kann auf 10° bleiben
→ Mehr Platz über Kopf = besser für Hand-Tracking
```

### Kritische Formel für 95cm Montage

```
Sichtfeld-Oberkante bei Neigung α:
  H_top = H_camera + tan(α + 45°) × Distance
  
Bei 95cm Kamera, 2m Entfernung:
├─ α = 10°: H_top = 95 + tan(55°) × 200 = 95 + 286 = ~195cm ⚠️ Knapp!
├─ α = 12°: H_top = 95 + tan(57°) × 200 = 95 + 308 = ~200cm ✅ Gut
└─ α = 15°: H_top = 95 + tan(60°) × 200 = 95 + 346 = ~205cm ✅✅ Optimal

Empfehlung: 12-15° Neigung für sichere Hand-über-Kopf Erkennung!
```

### Empfohlenes Setup (Final - für 95cm Constraint)

```
📷 KAMERA-POSITION:
   Höhe:     95cm über Boden (Display-Unterkante)
   Abstand:  2.0m vom Spieler
   Neigung:  12-15° nach oben ⚠️ KRITISCH!
   Position: Zentriert unter Display

✅ Tracking-Bereich @ 2m (mit 12° Neigung):
   Oben:  ~200cm (Hand über Kopf) ✅
   Mitte: ~110cm (Brust/Bauch) ← Kamera-Zentrum
   Unten: ~20cm (fast Bodenhöhe)

✅ Spieler-Coverage:
   180cm Person: Kopf bis Füße sichtbar
   165cm Person: Komplett sichtbar
   185cm Person: Knapp über Kopf bei 15° Neigung

⚠️ WICHTIG:
   Ohne 10-15° Neigung → Hand-über-Kopf nicht sichtbar!
   Mit korrekter Neigung → Perfektes Hand-Tracking ✅
```

**Vergleich: Ideal vs. Constraint**

| Parameter | Ideal (140cm) | Dein Setup (95cm) | Lösung |
|-----------|---------------|-------------------|---------|
| Höhe | 140cm | 95cm | ⚠️ 45cm niedriger |
| Neigung | 5° | 12-15° | ✅ Kompensiert durch Neigung |
| Oberkante | 230cm | 200cm | ✅ Ausreichend für 185cm |
| Tracking | Perfekt | Sehr gut | ✅ Kein Qualitätsverlust |

**Fazit:** 95cm Höhe ist OK mit korrekter Neigung! ✅

---

## 📦 Play Volume (3D Spielbereich)

### Aktuelle Konfiguration (Game Volume für 2m Spieler)

```cpp
// Code: include/core/PlayVolume.hpp
PlayVolume getGamePlayVolume() {
    minX = 0.0f;     // 100% horizontal (Fullscreen)
    maxX = 1.0f;
    minY = 0.0f;     // 100% vertikal (Fullscreen)
    maxY = 1.0f;
    minZ = 1200.0f;  // 1.2m - Arm voll ausgestreckt
    maxZ = 2800.0f;  // 2.8m - Arm am Körper
}
```

**Aktiv in:** `src/core/ProcessingLoop.cpp` (Konstruktor ruft `getGamePlayVolume()`)

### Visualisierung (für Game Volume @ 2m Spieler)

```
Draufsicht (von oben):

         Kamera/Display
              ↓
    ┌─────────────────────┐
    │ ← 1.6m breit →     │  @ 1.2m (Vorderkante, Z=0.0)
    └─────────────────────┘
           ▼ ▼ ▼
    ┌───────────────────────────┐
    │  ← 1.9m breit →          │  @ 1.6m (Z=0.25)
    └───────────────────────────┘
           ▼ ▼ ▼
    ┌─────────────────────────────────┐
    │   ← 2.2m breit →               │  @ 2.0m (Spieler, Z=0.5)
    └─────────────────────────────────┘
           ▼ ▼ ▼
    ┌───────────────────────────────────────┐
    │    ← 2.5m breit →                    │  @ 2.4m (Z=0.75)
    └───────────────────────────────────────┘
           ▼ ▼ ▼
    ┌─────────────────────────────────────────────┐
    │     ← 2.8m breit →                         │  @ 2.8m (Hinterkante, Z=1.0)
    └─────────────────────────────────────────────┘
```

### Physische Größe bei verschiedenen Abständen (Game Volume)

| Abstand | Breite (100%) | Höhe (100%) | Fläche | OSC Z |
|---------|---------------|-------------|--------|-------|
| 1.2m | ~1.6m | ~0.9m | ~1.4 m² | 0.0 |
| 1.6m | ~1.9m | ~1.1m | ~2.1 m² | 0.25 |
| 2.0m | ~2.2m | ~1.3m | ~2.9 m² | 0.5 |
| 2.4m | ~2.5m | ~1.4m | ~3.5 m² | 0.75 |
| 2.8m | ~2.8m | ~1.6m | ~4.5 m² | 1.0 |

**Sweet Spot:** 1.8m - 2.2m Entfernung (OSC Z: 0.375 - 0.625)

---

## 🏗️ Boden-Markierung für 2m Spieler-Position

**SETUP:** Spieler steht 2.0m vor Display/Kamera

### Exakte Berechnung

Bei **127° FoV** (OAK-D Pro) und **2m Entfernung**:

```
Kamera FOV @ 2m:
- Breite: 3.2m (640px camera = 3.2m real)
- Höhe: 1.8m (360px camera = 1.8m real)

Spieler Arm-Reichweite (Durchschnitt):
- Horizontal: ±80cm von Körpermitte
- Vertikal: Von Fußboden bis 30cm über Kopf
```

### Boden-Markierungs-Box (Empfohlen)

```
Rechteck mit Klebeband markieren:

                Kamera/Display (2m entfernt)
                        ↓
    ┌───────────────────────────────────┐
    │                                   │
    │      Vorderkante (1.2m)           │ ← Arm voll ausgestreckt
    │         1.6m breit                │   Nach vorne = Z=0.0 (OSC)
    │                                   │
    │          |████|                   │
    │          |Player|                 │ ← Spieler @ 2.0m
    │          |████|                   │   Neutral = Z=0.5 (OSC)
    │                                   │
    │      Hinterkante (2.8m)           │ ← Arm am Körper
    │         2.2m breit                │   Zurück = Z=1.0 (OSC)
    │                                   │
    └───────────────────────────────────┘
```

### Detaillierte Markierung (mit Messband)

**Benötigte Materialien:**
- 25m Stahllineal oder Messstab
- Klebeband (orange/weiß)
- Kreide oder Marker

**Schritt 1: Spieler-Mittellinie markieren**
```
Von Kamera direkt nach vorne 2.0m Linie ziehen
Dies ist die Achse, auf der der Spieler steht
```

**Schritt 2: Vorderkante (Arm ausgestreckt, Z=0.0)**
```
Messung von Kamera:     1.2m
Breite der Box:         1.6m (±0.8m von Mittellinie)
Klebeband-Linie:
  ├─ Punkt A: 1.2m von Kamera, 0.8m links
  ├─ Punkt B: 1.2m von Kamera, 0.8m rechts
  └─ Verbinde A-B parallel zur Kamera
```

**Schritt 3: Hinterkante (Arm am Körper, Z=1.0)**
```
Messung von Kamera:     2.8m
Breite der Box:         2.2m (±1.1m von Mittellinie)
Klebeband-Linie:
  ├─ Punkt C: 2.8m von Kamera, 1.1m links
  ├─ Punkt D: 2.8m von Kamera, 1.1m rechts
  └─ Verbinde C-D parallel zur Kamera
```

**Schritt 4: Seitenkanten (Arm seitlich)**
```
Linke Kante:    2.8m-1.2m Linie @ 0.8m-1.1m = breiter nach hinten
Rechte Kante:   2.8m-1.2m Linie @ -0.8m-(-1.1m) = breiter nach hinten
```

### Endresultat: Trapez

```
                 Kamera
                   ↓

        ← 1.6m breit →   @ 1.2m (vorne, Z=0)
         _______________
        /               \
       /                 \
      /                   \
     /                     \
    /                       \
   /_________________________\
    ← 2.2m breit →          @ 2.8m (hinten, Z=1.0)

SPIELER POSITION:
    Y
    ↑
    │      Oben (arm up)
    │         Y=0.9
    │      
    │    ┌────────────┐
    │    │            │
    │    │  Spieler   │  1.6m - 2.2m
    │    │   @ 2.0m   │  Spielfeld
    │    │            │
    │    └────────────┘
    │         Y=0.1
    │      (Unten, arm down)
    │
    └─────────────────→ X (horizontal)
      -0.8m  0  +0.8m
```

### Praktische Vermessung (vereinfacht)

**Wenn du kein Messstab hast:**

1. **Spieler mit ausgestrecktem Arm stellen** → Markiere diese Linie (1.2m)
2. **Spieler zurückgehen bis Arm am Körper** → Markiere diese Linie (2.8m)
3. **Arm nach links ausstrecken** → Markiere rechte Breite (±0.8m @ vorne, ±1.1m @ hinten)
4. **Arm nach rechts ausstrecken** → Markiere linke Breite
5. **Verbinde mit Klebeband** → Fertig!

### Im MJPEG Preview sichtbar

```
Das grüne Rechteck zeigt die 2D Projektion des Spielfeldes:

┌─────────────────────────┐
│   GAME VOLUME           │
│   (FULLSCREEN)          │
│                         │
│  Z: 1.2m - 2.8m        │
│  (Standing @ 2m)        │
│                         │
│  [Grüne Box = Spielfeld]│
│                         │
└─────────────────────────┘
```

### Z-Werte Verifizierung (mit OSC Monitor)

Nachdem die Box markiert ist, teste die OSC-Werte:

```
Spieler @ vordere Linie (1.2m, Arm ausgestreckt):
  /hand/0/palm: (0.5, 0.5, 0.0)    ← Z sollte ≈ 0.0 sein

Spieler @ mittlere Position (2.0m, neutral):
  /hand/0/palm: (0.5, 0.5, 0.5)    ← Z sollte ≈ 0.5 sein

Spieler @ hintere Linie (2.8m, Arm am Körper):
  /hand/0/palm: (0.5, 0.5, 1.0)    ← Z sollte ≈ 1.0 sein
```

Wenn die Werte matchen → **Perfekt kalibriert!** ✅

---

## 🎮 Unreal Engine Mapping (Flexibles Volume-Scaling)

### 🎯 Konzept: Normalisierung ermöglicht flexible Skalierung

**Wichtig:** OSC sendet **normalisierte Koordinaten (0.0 - 1.0)**, die **unabhängig von physischen Dimensionen** sind!

```
Physisches Game Volume (fest):
├─ 1.6m Tiefe  (1200mm - 2800mm)
├─ 3.2m Breite (@ 2m Kamera-Abstand)
└─ 1.8m Höhe   (@ 2m Kamera-Abstand)

Virtuelles UE Volume (FLEXIBEL skalierbar):
├─ 1.6m → 16m → 160m → 1600m (beliebig!)
├─ 3.2m → 32m → 320m → 3200m
└─ 1.8m → 18m → 180m → 1800m

→ 1m physische Bewegung = X m virtuelle Bewegung
  X ist frei konfigurierbar in Unreal Engine!
```

### Koordinaten-Transformation (flexibel)

**Basis-Formel:**
```cpp
// Definiere dein virtuelles Volume (beliebige Größe!)
FVector VolumeSize(X, Y, Z);        // in cm (UE Standard)
FVector VolumeOrigin(0, 0, 0);      // Startpunkt im Level

// OSC (0-1) → UE World Space
Hand.Location.X = VolumeOrigin.X + OSC_Z * VolumeSize.X;          // Tiefe
Hand.Location.Y = VolumeOrigin.Y + OSC_X * VolumeSize.Y;          // Horizontal
Hand.Location.Z = VolumeOrigin.Z + (1.0f - OSC_Y) * VolumeSize.Z; // Vertikal (invertiert)
```

### Skalierungs-Beispiele

**1:1 Realistisch (1.6m physisch = 1.6m virtuell)**
```cpp
FVector VolumeSize(160.0f, 320.0f, 180.0f);  // 1.6m × 3.2m × 1.8m
// → Natürliche 1:1 Bewegung
```

**10:1 "Large World" (1.6m physisch = 16m virtuell)**
```cpp
FVector VolumeSize(1600.0f, 3200.0f, 1800.0f);  // 16m × 32m × 18m
// → 10cm Hand-Bewegung = 1m virtuelle Bewegung
// → Gut für große Level mit präziser Kontrolle
```

**100:1 "Giant Mode" (1.6m physisch = 160m virtuell)**
```cpp
FVector VolumeSize(16000.0f, 32000.0f, 18000.0f);  // 160m × 320m × 180m
// → 1cm Hand-Bewegung = 1m virtuelle Bewegung
// → Perfekt für riesige Welten (Flight Simulator, God Games)
```

**0.1:1 "Precision Mode" (1.6m physisch = 16cm virtuell)**
```cpp
FVector VolumeSize(16.0f, 32.0f, 18.0f);  // 16cm × 32cm × 18cm
// → 10cm Hand-Bewegung = 1cm virtuelle Bewegung
// → Perfekt für Mikroskop, Chirurgie-Simulation, Feinmechanik
```

**Asymmetrische Skalierung (verschiedene Achsen)**
```cpp
// Tiefe: 20× größer, Breite: 10×, Höhe: 1:1
FVector VolumeSize(3200.0f, 3200.0f, 180.0f);  // 32m × 32m × 1.8m
// → Tiefe und Breite scaled, Höhe realistisch
// → Nützlich für flache, breite Level (Racing, Platformer)
```

### Box Component als Referenz (Empfohlen)

### Box Component als Referenz (Empfohlen)

**Warum Box Component?**
- ✅ Größe direkt im Editor einstellbar (keine Code-Änderung)
- ✅ Visualisierung im Level sichtbar
- ✅ Automatische Bounds-Berechnung
- ✅ Move/Rotate/Scale wie jedes andere Actor

**Implementation:**
```cpp
UCLASS()
class YOURGAME_API AHandTracker : public AActor
{
    GENERATED_BODY()

public:
    // Box Component für Play Volume (im Editor sichtbar)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Hand Tracking")
    UBoxComponent* PlayVolumeBox;

    AHandTracker()
    {
        // Create box component
        PlayVolumeBox = CreateDefaultSubobject<UBoxComponent>(TEXT("PlayVolume"));
        RootComponent = PlayVolumeBox;
        
        // Default size: 10m × 20m × 5m (anpassbar im Editor!)
        PlayVolumeBox->SetBoxExtent(FVector(500.0f, 1000.0f, 250.0f));  // Halbe Größe
        PlayVolumeBox->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        PlayVolumeBox->ShapeColor = FColor::Green;
    }

    void OnPalmReceived(const FOSCMessage& Message, const FString& IPAddress, int32 Port)
    {
        if (Message.GetArguments().Num() >= 3)
        {
            float OSC_X = Message.GetArguments()[0].GetFloat();
            float OSC_Y = Message.GetArguments()[1].GetFloat();
            float OSC_Z = Message.GetArguments()[2].GetFloat();
            
            // Get box bounds (from Editor settings)
            FVector BoxExtent = PlayVolumeBox->GetScaledBoxExtent();
            FVector BoxOrigin = PlayVolumeBox->GetComponentLocation();
            
            // Map OSC (0-1) to Box Volume
            // Note: Extent is HALF size, so multiply by 2
            Hand0Position.X = BoxOrigin.X + (OSC_Z * 2.0f - 1.0f) * BoxExtent.X;
            Hand0Position.Y = BoxOrigin.Y + (OSC_X * 2.0f - 1.0f) * BoxExtent.Y;
            Hand0Position.Z = BoxOrigin.Z + ((1.0f - OSC_Y) * 2.0f - 1.0f) * BoxExtent.Z;
        }
    }
};
```

**Im Unreal Editor:**
1. Place "AHandTracker" Actor im Level
2. Select Actor → Details Panel
3. **PlayVolumeBox → Box Extent:**
   - X: 500 = 10m Tiefe (500cm × 2 = 1000cm)
   - Y: 1000 = 20m Breite (1000cm × 2 = 2000cm)
   - Z: 250 = 5m Höhe (250cm × 2 = 500cm)
4. **Ändern nach Bedarf!** (z.B. 5000, 10000, 25000 für Giant Mode)

### Velocity Skalierung

**Problem:** Velocity kommt in mm/s, muss aber mit Volume-Größe skaliert werden!

**Lösung:**
```cpp
void OnVelocityReceived(const FOSCMessage& Message, const FString& IPAddress, int32 Port)
{
    if (Message.GetArguments().Num() >= 3 && PlayVolumeBox)
    {
        float OSC_VX = Message.GetArguments()[0].GetFloat();  // mm/s
        float OSC_VY = Message.GetArguments()[1].GetFloat();
        float OSC_VZ = Message.GetArguments()[2].GetFloat();
        
        FVector BoxExtent = PlayVolumeBox->GetScaledBoxExtent();
        
        // Physische Ranges (mm):
        // Z: 1600mm (1.2m-2.8m)
        // X: ~3200mm (@ 2m Kamera-Abstand)
        // Y: ~1800mm (@ 2m Kamera-Abstand)
        
        // Scale factors: Virtual size / Physical size
        float ScaleX = (BoxExtent.X * 2.0f) / 1600.0f;  // Tiefe
        float ScaleY = (BoxExtent.Y * 2.0f) / 3200.0f;  // Breite
        float ScaleZ = (BoxExtent.Z * 2.0f) / 1800.0f;  // Höhe
        
        // Apply scaling (mm/s → cm/s)
        Hand0Velocity.X = OSC_VZ * ScaleX * 0.1f;
        Hand0Velocity.Y = OSC_VX * ScaleY * 0.1f;
        Hand0Velocity.Z = -OSC_VY * ScaleZ * 0.1f;  // Invertiert
    }
}
```

**Beispiel:**
- Volume: 10m × 20m × 5m
- Hand bewegt sich mit 100mm/s physisch (nach vorne)
- Scale: (500cm × 2) / 160cm = 6.25×
- Virtual Velocity: 100 × 6.25 × 0.1 = 62.5 cm/s ✅

---

## 🔧 Konfiguration

### Aktuell aktiv: GAME VOLUME

**Setup:** Spieler steht 2m vor 220cm × 125cm Display

```cpp
// Code: include/core/PlayVolume.hpp
PlayVolume getGamePlayVolume() {
    minX = 0.0f;     // Fullscreen (100% horizontal)
    maxX = 1.0f;     // Player uses ~50% center
    minY = 0.0f;     // Fullscreen (100% vertikal)
    maxY = 1.0f;     // Player uses ~80%
    minZ = 1200.0f;  // 1.2m - Arm vollständig ausgestreckt
    maxZ = 2800.0f;  // 2.8m - Arm am Körper + Margin
}
```

**Aktivierung:** `src/core/ProcessingLoop.cpp` Konstruktor ruft `getGamePlayVolume()`

### Andere Presets verfügbar

```cpp
getDefaultPlayVolume()       // 90% Coverage (0.5m-2.5m)
getConservativePlayVolume()  // 80% Coverage (0.5m-2.5m)
getFullscreenPlayVolume()    // 100% Coverage (0.5m-2.5m)
```

### Preset wechseln

In `src/core/ProcessingLoop.cpp` Konstruktor (Zeile ~52):

```cpp
// Aktuell:
_playVolume = std::make_unique<PlayVolume>(getGamePlayVolume());

// Ändern zu:
_playVolume = std::make_unique<PlayVolume>(getDefaultPlayVolume());
// oder andere Presets...
```

---

## 📊 Tiefengenauigkeit (2m Standing Player)

### Stereo-Matching Accuracy @ 1.2m-2.8m Range

| Abstand | Genauigkeit | Tracking-Qualität | Use Case |
|---------|-------------|-------------------|----------|
| 1.2m | ±2cm | ⚠️ Grenzbereich | Arm voll ausgestreckt |
| 1.5m | ±1.5cm | ✅ Gut | Arm 50% ausgestreckt |
| 2.0m | ±1cm | ✅✅ Sehr gut | Spieler neutral Position |
| 2.4m | ±1.5cm | ✅ Gut | Arm nah am Körper |
| 2.8m | ±2cm | ⚠️ Grenzbereich | Arm am Körper |

**Optimal:** 1.8m - 2.2m Entfernung (±1cm Genauigkeit)

### Faktoren für Z-Genauigkeit @ 2m

- **Baseline:** OAK-D Stereo-Baseline ≈ 7.5cm
- **Beleuchtung:** Sehr wichtig! Gut beleuchtete Szene → ±1cm
- **Textur:** Hände haben ausreichend Textur → gut für Stereo
- **Okklusion:** Verdeckte Finger → schlechtere Depth
- **Bewegung:** Schnelle Bewegungen → Motion Blur → weniger Matches

### Z-Resolution für 1.2m-2.8m Range

```
Total Tiefenbereich: 1.6m (1600mm)
OSC Z-Auflösung: 0.0 - 1.0 (normalized)

Pro 0.01 OSC-Schritte:
  0.01 × 1600mm = 16mm = 1.6cm Auflösung

Praktisch erreichbar: ±1cm @ 2m optimal
                      ±2cm @ Grenzen (1.2m, 2.8m)
```

---

## 🔍 Debug & Verifikation

### Im MJPEG Preview (http://100.101.16.21:8080)

Zeigt die grüne Play Volume Box mit:
- "GAME VOLUME (FULLSCREEN) - ACTIVE"
- "Z: 1.2m - 2.8m (Standing @ 2m)"
- Hände außerhalb: Roter Kreis + "OUT" Label

### Im Log

```bash
journalctl -u hand-tracking -f | grep "Play Volume"
```

Sollte zeigen:
```
Play Volume initialized (GAME): 100% x 100% coverage, Z: 1200-2800mm (1.2m-2.8m)
```

### OSC Werte prüfen

Bei bekannter Entfernung (z.B. 2.0m mit Maßband):
```python
# Python OSC Monitor
/hand/0/palm: (0.5, 0.5, 0.5)  # Z=0.5 → 2.0m ✅
```

Erwarteter Z-Wert: `(Entfernung_mm - 1200) / 1600`

---

## 📝 Zusammenfassung (2m Standing Player)

### Schnell-Referenz

| Was | Wert | Einheit | Notiz |
|-----|------|---------|-------|
| **Z-Range (absolut)** | 1.2 - 2.8m | Meter | Arm aus bis am Körper |
| **Z-Range (OSC)** | 0.0 - 1.0 | Normalized | Linear gemappt |
| **XY-Coverage** | 100% (Full) | % | 3.2m × 1.8m @ 2m |
| **Play Volume** | Trapez | Shape | Breiter nach hinten |
| **Front Box** | 1.6m breit @ 1.2m | Meter | Arm voll ausgestreckt |
| **Back Box** | 2.2m breit @ 2.8m | Meter | Arm am Körper |
| **Sweet Spot** | 1.8 - 2.2m | Meter | ±1cm Z-Genauigkeit |
| **Spieler Position** | 2.0m | Meter | Ideal-Entfernung |
| **Arm-Reichweite** | ±80cm | Meter | Horizontal |

### Code-Referenzen

- **Definition:** `include/core/PlayVolume.hpp`
- **Initialisierung:** `src/core/ProcessingLoop.cpp` (Konstruktor)
- **Filtering:** `src/core/ProcessingLoop.cpp` (processFrame)
- **Dokumentation:** `docs/OSC_QUICK_REFERENCE.md`, `docs/PLAYER_LOCK_DESIGN.md`

