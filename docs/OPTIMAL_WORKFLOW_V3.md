# OPTIMAL_WORKFLOW_V3

## Überblick
- Hardware: Jetson Orin Nano (8GB) + OAK-D Pro PoE
- API: DepthAI v3 (keine Gen2 `dai::node` mehr)
- Netzwerk: Tailscale IP `100.101.16.21`
- Preview: MJPEG, gespiegelt (wie Spiegel); nur Kamerabild/Bounding Boxes/Skelette gespiegelt, Overlay-Text bleibt lesbar

## Datenpfad
1. OAK-D → NV12 RGB Preview (640×360)
2. Jetson: Palm Detection (TensorRT) → HandLandmark (TensorRT)
3. StereoDepth (Mono L/R 640×400) → Z an Palm Center
4. Kalman-Tracking (Position + Velocity + Delta)
5. Volume-Filtering (2D vor Inferenz, 3D nach Stereo)
6. OSC Sender (30 Hz, Drop > 50ms)

## Koordinaten & Normalisierung
- X, Y: 0.0 – 1.0 (Bildkoordinaten)
- Z: 0.0 – 1.0 (normalisiert aus physischer Tiefe)
  - Konstanten: `Z_MIN_MM=1200`, `Z_MAX_MM=2800`, `Z_RANGE_MM=1600`
- Y-Achse invertiert für OSC: 0=oben, 1=unten (Unreal Engine)

## OSC Pfade
- `/hand/{id}/palm` → `[x, y, z]` (alle 0-1)
- `/hand/{id}/velocity` → `[vx, vy, vz]` (in 0-1 Space)
- `/hand/{id}/delta` → `[dx, dy, dz]`
- `/hand/{id}/gesture` → `[id, confidence, name]`
- Legacy entfernt (`/vip`)

## Gesten-Erkennung (FSM)
- Primär: MCP-basierte Heuristiken (tip.y < mcp.y)
- Fallback: Winkel-basierte Checks in Edge-Cases
- Hysterese und Debounce (`GESTURE_DEBOUNCE_FRAMES=1`)
- Bekannte schwierige Fälle: FIVE vs FOUR (Daumen), FIST vs THUMB_UP bei Winkeln

## Volume-Filtering
- 2D: Palm im PlayVolume? → sonst verwerfen (GPU spart Zeit)
- 3D: Nach Z-Bestimmung → outside Volume verwerfen
- Ziel: Stabil zwei Hände, First-Come-First-Serve, Session Events (Spawn/Despawn) – Phase 4

## Preview
- Mirrored (horizontal flip) nur Kamera/Boxes/Skelette
- Text-Overlay bleibt lesbar (flipped Text-Rendering für Labels berücksichtigt)
- Anzeige: FPS, Model (LITE/FULL), Stereo-Status, Hand-Details (Pos/Vel/Delta/Geste)

## Phasenstatus (2026-01-10)
- Phase 2 (2D): abgeschlossen
- Phase 3 (Stereo): implementiert, Tests offen
- Phase 4 (Player Lock): Design fertig, Implementierung nach Stereo-Validierung

## Regeln & Qualität
- Keine Architekturänderungen ohne Freigabe
- Immer zuerst testen, dann committen/pushen
- Niemals ungefragt revertieren
- `-Wall -Wextra -Werror` (auf Jetson), clang-tidy performance/readability
