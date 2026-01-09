# Setup Hand-Tracking Service (Jetson Orin Nano)

Diese Anleitung beschreibt die Installation der Abhängigkeiten und die Konfiguration für ein C++ Projekt mit **DepthAI v3**, **OpenCV**, **OSC** und **Tailscale** unter Ubuntu 22.04.

---

## 🎯 Aktueller Status (2026-01-09)

**Phase 2 - 2D Hand Tracking: ABGESCHLOSSEN ✅**
- ✅ **TensorRT Inference** - Palm Detection + Hand Landmark auf Jetson
- ✅ **2-Hand Tracking** - Beide Hände gleichzeitig erkannt
- ✅ **Y-basierte Gesten** - Robuste Erkennung (FIST, THUMBS_UP, PEACE, FIVE, etc.)
- ✅ **Haar Cascade Face Filter** - Keine False Positives mehr im Gesicht
- ✅ **25-30 FPS** stabil mit voller Inference Pipeline

**Dokumentation:**
- [TODO.md](docs/TODO.md) - Aktueller Projekt-Status
- [OPTIMAL_WORKFLOW_V3.md](docs/OPTIMAL_WORKFLOW_V3.md) - V3 Architektur
- [OSC_GESTURE_REFERENCE.md](docs/OSC_GESTURE_REFERENCE.md) - OSC Protokoll

**Nächste Phase:**
- ⬜ Phase 3: Stereo Depth (Z-Koordinaten für 3D Position)

---


## 1. System-Voraussetzungen & Netz

Stellen Sie sicher, dass der Jetson erreichbar ist.

* **Tailscale:** Zur stabilen Remote-Verbindung (CLion Remote Host).
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

```

## 2. System-Abhängigkeiten installieren

Bevor die DepthAI-Library gebaut wird, müssen die Basispakete vorhanden sein:

```bash
sudo apt update
sudo apt install -y build-essential cmake git libopencv-dev libusb-1.0-0-dev liblo-dev libyaml-cpp-dev

```

## 3. DepthAI-Core manuell bauen (Statisch)

Da die Paketmanager auf dem Jetson oft Konflikte verursachen, bauen wir die Library direkt aus den Quellen:

```bash
cd ~
git clone --recursive https://github.com/luxonis/depthai-core.git
cd depthai-core

# USB-Regeln für OAK-Kameras setzen
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# Build & Installation (Erzeugt libdepthai-core.a)
cmake -S . -B build -D DEPTHAI_BUILD_EXAMPLES=OFF -D DEPTHAI_BUILD_TESTS=OFF
cmake --build build --parallel $(nproc)
sudo cmake --install build
sudo ldconfig

```

## 4. CLion Remote-Host Konfiguration

Um vom PC aus auf dem Jetson zu entwickeln:

1. **Toolchain:** `Settings -> Build, Execution, Deployment -> Toolchains`.
* Neues **Remote Host** Profil erstellen.
* Verbindung via Tailscale-IP des Jetsons herstellen.


2. **CMake-Profil:** `Settings -> Build, Execution, Deployment -> CMake`.
* Sicherstellen, dass die oben erstellte Remote-Toolchain ausgewählt ist.
* Bei Fehlern: `Zahnrad-Icon -> Reset Cache and Reload Project`.



## 5. CMakeLists.txt (Funktionierende Version)

Die Konfiguration nutzt direkte Pfade, um Probleme mit fehlerhaften Config-Dateien zu umgehen:

```cmake
cmake_minimum_required(VERSION 3.22)
project(HandTrackingService LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# 1. Pakete suchen
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBLO REQUIRED liblo)

# 2. DepthAI Pfade manuell setzen (Statisch)
find_path(DEPTHAI_INCLUDE_DIR NAMES depthai/depthai.hpp PATHS /usr/local/include)
find_library(DEPTHAI_LIBRARY NAMES libdepthai-core.a PATHS /usr/local/lib)

add_executable(HandTrackingService src/main.cpp)

if (DEPTHAI_INCLUDE_DIR AND DEPTHAI_LIBRARY)
    target_include_directories(HandTrackingService PRIVATE ${DEPTHAI_INCLUDE_DIR} ${LIBLO_INCLUDE_DIRS})

    target_link_libraries(HandTrackingService PRIVATE
            ${DEPTHAI_LIBRARY}
            ${OpenCV_LIBS}
            ${LIBLO_LIBRARIES}
            pthread
            dl
    )
    message(STATUS "DepthAI Static Lib: ${DEPTHAI_LIBRARY}")
else ()
    message(FATAL_ERROR "Konnte libdepthai-core.a nicht finden.")
endif ()

```

## 6. Performance-Setup (WICHTIG!)

Damit der Jetson Orin Nano die volle Leistung (35-50 FPS) erreicht, muss er im MAXN-Modus laufen.
**Hinweis für Orin Nano:** Der "MAXN" Modus heißt technisch oft **"15W"**. Das ist das physikalische Maximum (Mode 0).

**Installation (einmalig):**
Führen Sie dieses Skript aus, damit der **Hochleistungsmodus automatisch beim Booten** aktiviert wird:

```bash
cd ~/dev/HandTrackingV3/scripts
sudo bash setup_performance_autostart.sh
```

**Ergebnis:**
- ✅ **Automatischer Start:** Der Jetson bootet immer im MAXN/15W-Modus.
- ✅ **Kein Passwort:** Es ist keine manuelle Eingabe mehr nötig.
- ✅ **GPU-Lock:** GPU wird auf maximalen Takt (624 MHz) festgenagelt.

**Hardware-Details (15W Mode):**
- **CPU:** 6x ARM Cortex-A78AE @ **1.51 GHz**
- **GPU:** 1024 CUDA Cores (Ampere) @ **624 MHz**
- **EMC (RAM):** LPDDR5 @ **2.1 GHz**

**System prüfen:**
```bash
# Zeigt CPU/GPU Clocks und Power Mode
bash ~/dev/HandTrackingV3/scripts/diagnose_jetson.sh
```
Das Diagnose-Skript bestätigt "✅ SUCCESS" wenn der 15W Modus aktiv ist.

## 7. Auto-Start der App (Optional: Produktion)

Wenn die App automatisch beim Booten starten soll (nachdem der Performance-Modus aktiv ist), nutzen Sie den `hand-tracking.service`.

**Service-Datei (`scripts/hand-tracking.service`):**
```ini
[Unit]
Description=Ultra-Low-Latency Hand Tracking (C++)
After=network-online.target jetson-performance.service
Wants=network-online.target

[Service]
Type=simple
User=nvidia
Group=nvidia
WorkingDirectory=/home/nvidia/dev/HandTrackingV3
# Pfad zum Binary anpassen (Release oder Debug)
ExecStart=/home/nvidia/dev/HandTrackingV3/cmake-build-debug-remote-host/HandTrackingService

# PERFORMANCE TUNING
CPUSchedulingPolicy=fifo
CPUSchedulingPriority=90
Nice=-20

# UMGEBUNGSVARIABLEN
Environment=LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/local/lib
Environment=OAKD_DEVICE_IP=169.254.1.222
Environment=QT_QPA_PLATFORM=offscreen

Restart=on-failure
RestartSec=2s

[Install]
WantedBy=multi-user.target
```

**Installation:**
```bash
sudo cp scripts/hand-tracking.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable hand-tracking.service
```

## 8. Modelle installieren

Die Modelle sind zwingend erforderlich.

```bash
cd models/

# Option A: Vorcompilierte 4-SHAVE Blobs (Legacy)
wget -nc https://github.com/geaxgx/depthai_hand_tracker/raw/main/models/palm_detection_sh4.blob
wget -nc https://github.com/geaxgx/depthai_hand_tracker/raw/main/models/hand_landmark_full_sh4.blob

# Option B: Optimierte 6-SHAVE Blobs (Empfohlen für V3)
# Siehe scripts/compile_hand_landmark.py
```

### SHAVE-Konfiguration (Performance)

Die OAK-D Pro hat 16 SHAVE-Kerne (RVC3) bzw. 12 (RVC2).
Die V3 Pipeline nutzt:
- **6 SHAVEs** für Palm Detection
- **6 SHAVEs** für Hand Landmarks
- Das ermöglicht **35-40 FPS**.
```