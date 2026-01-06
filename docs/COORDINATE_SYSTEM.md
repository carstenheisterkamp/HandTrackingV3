Das Mapping von physischen Sensordaten (Oak-D Pro PoE) auf eine virtuelle Umgebung (Unreal Engine 5.4) erfordert eine Entkopplung der Sensor-Logik von der Actor-Logik. Da du Personen und Hände vor einem Screen trackst, befinden wir uns im Bereich der **Natural User Interfaces (NUI)**.

Hier ist der Best-Practice-Ansatz unter Berücksichtigung deiner Spezifikationen und der JetBrains Rider Arbeitsweise.

---

## 1. Analyse & Lösungsansatz

### Das Problem mit absoluten Werten

Absolute Werte der Oak-D (in mm) sind direkt abhängig von der Position der Kamera und dem Sichtfeld (FOV). Wenn du die Kamera nur 10 cm verschiebst, bricht dein gesamtes UE5-Setup, wenn du absolute Werte nutzt.

### Die Lösung: Ein virtuelles Interaktionsvolumen

Wir definieren eine **Interaction Box** in der physischen Welt. Alles innerhalb dieser Box wird auf das **Blocking Volume** in UE5 gemappt.

**Best Practice:** **Normalisierte Koordinaten (0.0 bis 1.0)** für die Übertragung, umgerechnet in **lokale Koordinaten** innerhalb des Zielvolumens in UE5.

---

## 2. Strategie: Das Mapping-Modell

### A. Physische Definition (Calibration Space)

Du definierst in deinem C#- oder Python-Service (der die Oak-D ausliest) ein Clipping-Volumen:

* **X (Breite):** z.B. -500mm bis +500mm (zentriert zur Kamera).
* **Y (Höhe):** z.B. -400mm bis +400mm.
* **Z (Tiefe):** z.B. 500mm bis 1500mm (der Bereich vor dem Screen).

### B. Normalisierung (Der "Contract")

Jeder Punkt  wird in einen normalisierten Vektor  transformiert:


### C. UE5 Mapping (Target Space)

In Unreal Engine nimmst du die Dimensionen deines **Blocking Volumes** (z.B. ).
Die Actor-Position berechnet sich dann simpel:
`ActorLocation = VolumeOrigin + (NormalizedInput * VolumeExtent)`

---

## 3. Implementierungs-Architektur (Clean Code)

Gemäß deiner Regeln nutzen wir ein sauberes Interface-Design. In Rider solltest du die Logik für die Transformation in eine eigene Utility-Klasse auslagern.

### Datenstruktur (C++ / ProcessingLoop)

```cpp
struct TrackingResult {
    // Normalisierte Koordinaten (0.0 - 1.0)
    struct NormalizedPoint {
        float x, y, z;
    };
    
    std::vector<NormalizedPoint> landmarks; 
    bool isTracked;
    // ... velocity etc.
};
```

### Warum keine Deltas?

Deltas (relative Bewegung) neigen zum **Drift**. Wenn ein Paket verloren geht, ist der Actor permanent versetzt.

* **Absolute (normalisierte) Werte** sind zustandslos und stabil.
* **Deltas** nutzt man nur für Gesten (z.B. "Swipe"), aber nicht für die Positionierung eines Actors in einem begrenzten Volumen.

---

## 4. Best Practice Workflow (Service-Implementation)

1. **Deadzone & Smoothing:** Der Service implementiert einen **One-Euro Filter** und **Kalman Filter** zur Glättung der Rohdaten vor der Normalisierung.
2. **Clamping:** Die Werte werden im `ProcessingLoop` strikt auf `[0.0, 1.0]` geclampt, bezogen auf die konfigurierte Interaction Box.
3. **Visual Debugging:** Der MJPEG-Stream visualisiert die Interaction Box im Kamerabild.

---

## 5. Konfiguration (Interaction Box)

Die Interaction Box wird im Service konfiguriert (in mm, relativ zur Kamera):

```cpp
constexpr float BOX_MIN_X = -500.0f;
constexpr float BOX_MAX_X =  500.0f;
constexpr float BOX_MIN_Y = -400.0f;
constexpr float BOX_MAX_Y =  400.0f;
constexpr float BOX_MIN_Z =  500.0f; // Start interaction 50cm from camera
constexpr float BOX_MAX_Z = 1500.0f; // End interaction 1.5m from camera
```

**Empfehlung:** Sende ein OSC-Paket mit den normalisierten Werten der Gelenke (Hand-Center, Handgelenk, etc.) an `/hand/0/pos x y z`.

Möchtest du, dass ich dir einen konkreten C++-Entwurf für die Normalisierungs-Logik der Oak-D Daten erstelle?

