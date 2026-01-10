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
| **Z** | 0.0 - 1.0 | **0.5m → 2.5m** | Linear gemappt |

### Z-Koordinate Berechnung

```
Z_normalized = (Z_mm - 500) / (2500 - 500)
             = (Z_mm - 500) / 2000

Beispiele:
  0.5m (500mm)  → Z = 0.0
  1.0m (1000mm) → Z = 0.25
  1.5m (1500mm) → Z = 0.5
  2.0m (2000mm) → Z = 0.75
  2.5m (2500mm) → Z = 1.0
```

**Wichtig:** 
- Werte < 0.5m werden als 0.0 geclampt
- Werte > 2.5m werden als 1.0 geclampt

---

## 📦 Play Volume (3D Spielbereich)

### Aktuelle Konfiguration (90% Default)

```cpp
// Code: include/core/PlayVolume.hpp
PlayVolume {
    minX = 0.05f;    // 5% Margin links
    maxX = 0.95f;    // 5% Margin rechts
    minY = 0.05f;    // 5% Margin oben
    maxY = 0.95f;    // 5% Margin unten
    minZ = 500.0f;   // 0.5m minimum
    maxZ = 2500.0f;  // 2.5m maximum
}
```

### Visualisierung

```
Draufsicht (von oben):

         Kamera/Display
              ↓
    ┌─────────────────────┐
    │ ← 0.7m breit →     │  @ 0.5m Entfernung
    └─────────────────────┘
           ▼ ▼ ▼
    ┌─────────────────────────┐
    │  ← 1.4m breit →        │  @ 1.0m
    └─────────────────────────┘
           ▼ ▼ ▼
    ┌───────────────────────────────┐
    │   ← 2.2m breit →             │  @ 1.5m (Sweet Spot)
    └───────────────────────────────┘
           ▼ ▼ ▼
    ┌─────────────────────────────────────┐
    │    ← 2.9m breit →                  │  @ 2.0m
    └─────────────────────────────────────┘
           ▼ ▼ ▼
    ┌───────────────────────────────────────────┐
    │     ← 3.6m breit →                       │  @ 2.5m
    └───────────────────────────────────────────┘
```

### Physische Größe bei verschiedenen Abständen

| Abstand | Breite (90%) | Höhe (90%) | Fläche |
|---------|--------------|------------|--------|
| 0.5m | ~0.7m | ~0.4m | ~0.3 m² |
| 1.0m | ~1.4m | ~0.8m | ~1.1 m² |
| 1.5m | ~2.2m | ~1.2m | ~2.6 m² |
| 2.0m | ~2.9m | ~1.6m | ~4.6 m² |
| 2.5m | ~3.6m | ~2.0m | ~7.2 m² |

**Sweet Spot:** 1.5m - 2.0m Entfernung (optimale Balance aus Tracking-Qualität und Bewegungsfreiheit)

---

## 🏗️ Boden-Markierung (Empfohlen)

Für dein Setup (Kamera auf Stativ unter Display):

### Empfohlene Markierung

```
Rechteck auf dem Boden:
- Breite: 2.5m
- Tiefe: 2.0m (von 0.5m - 2.5m zur Kamera)
- Zentrum: 1.5m von der Kamera

Mit Klebeband markieren:
┌────────────────────────────────┐
│       Nähere Linie (0.5m)      │ ← Mindestabstand
│         ca. 1m breit           │
│                                │
│     [Sweet Spot Zone]          │ ← 1-2m optimal
│       ca. 2-3m breit           │
│                                │
│       Fernere Linie (2.5m)     │ ← Maximaler Abstand
│         ca. 3.5m breit         │
└────────────────────────────────┘
```

### Praktische Markierung (vereinfacht)

**Option 1: Single Center Box (Empfohlen für Tests)**
- **2m × 1.5m Rechteck**
- Zentriert bei 1.5m von Kamera
- Markiert den optimalen Spielbereich

**Option 2: Multi-Zone (für große Installation)**
- Innere Zone (grün): 1-2m optimal
- Mittlere Zone (gelb): 0.5-1m und 2-2.5m akzeptabel
- Außerhalb: Nicht getrackt

---

## 🎮 Unreal Engine Mapping

### Koordinaten-Transformation

```cpp
// OSC → Unreal World Space
Hand.Location.X = OSC_Z * 300.0f;          // Tiefe: 0-2m → 0-300cm
Hand.Location.Y = OSC_X * 800.0f;          // Horizontal: 0-1 → 0-800cm
Hand.Location.Z = (1.0f - OSC_Y) * 600.0f; // Vertikal: invertiert

// Beispiel:
// OSC: (x=0.5, y=0.5, z=0.5) → 1.5m Entfernung, Bildmitte
// UE:  (X=150cm, Y=400cm, Z=300cm)
```

### Play Volume in Unreal

Wenn du das Play Volume in UE visualisieren willst:

```cpp
// Volume Bounds (in cm, UE-Koordinaten)
FVector VolumeMin(0, 0, 0);              // Nahe Ecke
FVector VolumeMax(200, 800, 600);        // Ferne Ecke

// @ OSC_Z = 0.0 (0.5m): X = 0
// @ OSC_Z = 1.0 (2.5m): X = 200
```

---

## 🔧 Konfiguration

### Aktuell aktiv

- **Preset:** 90% Coverage (Default)
- **Datei:** `include/core/PlayVolume.hpp`
- **Funktion:** `getDefaultPlayVolume()`

### Andere Presets verfügbar

```cpp
// Conservative (80% Coverage)
getConservativePlayVolume()  // minX=0.1, maxX=0.9

// Fullscreen (100% Coverage)
getFullscreenPlayVolume()    // minX=0.0, maxX=1.0
```

### Preset wechseln

In `src/core/ProcessingLoop.cpp` Konstruktor:

```cpp
// Aktuell:
_playVolume = std::make_unique<PlayVolume>(getDefaultPlayVolume());  // 90%

// Ändern zu:
_playVolume = std::make_unique<PlayVolume>(getConservativePlayVolume());  // 80%
// oder
_playVolume = std::make_unique<PlayVolume>(getFullscreenPlayVolume());    // 100%
```

---

## 📊 Tiefengenauigkeit

### Stereo-Matching Accuracy

| Abstand | Genauigkeit | Tracking-Qualität |
|---------|-------------|-------------------|
| 0.5m | ±2cm | ⚠️ Grenzbereich |
| 1.0m | ±1cm | ✅ Gut |
| 1.5m | ±1.5cm | ✅ Sehr gut |
| 2.0m | ±2cm | ✅ Gut |
| 2.5m | ±3cm | ⚠️ Akzeptabel |

**Optimal:** 1.0m - 2.0m Entfernung

### Faktoren für Z-Genauigkeit

- **Beleuchtung:** Gut beleuchtete Szene → bessere Stereo-Matches
- **Textur:** Hände haben gute Textur → gut für Stereo
- **Okklusion:** Verdeckte Finger → schlechtere Depth
- **Bewegung:** Schnelle Bewegungen → Motion Blur → weniger Matches

---

## 🔍 Debug & Verifikation

### Im MJPEG Preview (http://100.101.16.21:8080)

Zeigt die grüne Play Volume Box mit:
- "PLAY VOLUME (16:9) - ACTIVE"
- "Z: 0.5m - 2.5m (Filtering ON)"
- Hände außerhalb: Roter Kreis + "OUT" Label

### Im Log

```bash
journalctl -u hand-tracking -f | grep "Play Volume"
```

Sollte zeigen:
```
Play Volume initialized: 90% x 90% (16:9), Z: 500-2500mm
```

### OSC Werte prüfen

Bei bekannter Entfernung (z.B. 1.0m mit Maßband):
```python
# Python OSC Monitor
/hand/0/palm: (0.5, 0.5, 0.25)  # Z=0.25 → 1.0m ✅
```

Erwarteter Z-Wert: `(Entfernung_mm - 500) / 2000`

---

## 📝 Zusammenfassung

### Schnell-Referenz

| Was | Wert | Einheit |
|-----|------|---------|
| Z-Range (absolut) | 0.5 - 2.5m | Meter |
| Z-Range (OSC) | 0.0 - 1.0 | Normalized |
| Play Volume (XY) | 90% (5% Margin) | % |
| Play Volume (Z) | 0.5 - 2.5m | Meter |
| Sweet Spot | 1.0 - 2.0m | Meter |
| Boden-Markierung | 2m × 1.5m @ 1.5m | Meter |

### Code-Referenzen

- **Definition:** `include/core/PlayVolume.hpp`
- **Initialisierung:** `src/core/ProcessingLoop.cpp` (Konstruktor)
- **Filtering:** `src/core/ProcessingLoop.cpp` (processFrame)
- **Dokumentation:** `docs/OSC_QUICK_REFERENCE.md`, `docs/PLAYER_LOCK_DESIGN.md`

