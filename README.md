# Setup Hand-Tracking Service (Jetson Orin Nano)

Diese Anleitung beschreibt die Installation der Abhängigkeiten und die Konfiguration für ein C++ Projekt mit **DepthAI**, **OpenCV**, **OSC** und **Tailscale** unter Ubuntu 22.04.

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

## 6. Build & Ausführung

Dein optimiertes Service-Fileerstellt unter `/etc/systemd/system/hand-tracking.service`:
```yaml
[Unit]
Description=Ultra-Low-Latency Hand Tracking (C++)
# nv-l4t-boot-complete sorgt dafür, dass die NVIDIA-Treiber voll geladen sind
After=network-online.target nv-l4t-boot-complete.service tailscaled.service
Wants=network-online.target

[Service]
Type=simple
User=nvidia
Group=nvidia
# Während der Entwicklung auf den CLion-Pfad zeigen:
WorkingDirectory=/home/nvidia/dev/HandTrackingV3
ExecStart=/home/nvidia/dev/HandTrackingV3/cmake-build-release/HandTrackingService

# PERFORMANCE TUNING
# FIFO 90 ist top! Verhindert, dass der Kernel den Prozess unterbricht.
CPUSchedulingPolicy=fifo
CPUSchedulingPriority=90
Nice=-20

# UMGEBUNGSVARIABLEN
# Falls du CUDA in C++ nutzt, ist der Pfad wichtig
Environment=LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/local/lib
Environment=OAKD_DEVICE_IP=169.254.1.222
# Verhindert Fehlermeldungen von OpenCV/QT bei Headless-Betrieb
Environment=QT_QPA_PLATFORM=offscreen

# STABILITÄT
Restart=on-failure
RestartSec=2s

[Install]
WantedBy=multi-user.target
~                                                                                                                                                                                          
~                                                                                                                                                                                          
"hand-tracking.service" 33L, 1020B                                                                                                                                       16,26       Alles
[Unit]
Description=Ultra-Low-Latency Hand Tracking (C++)
# nv-l4t-boot-complete sorgt dafür, dass die NVIDIA-Treiber voll geladen sind
After=network-online.target nv-l4t-boot-complete.service tailscaled.service
Wants=network-online.target

[Service]
Type=simple
User=nvidia
Group=nvidia
# Während der Entwicklung auf den CLion-Pfad zeigen:
# Production:
# WorkingDirectory=/opt/hand-tracking-service
# Dev
WorkingDirectory=/home/nvidia/dev/HandTrackingV3
ExecStart=/home/nvidia/dev/HandTrackingV3/cmake-build-release/HandTrackingService

# PERFORMANCE TUNING
# FIFO 90 ist top! Verhindert, dass der Kernel den Prozess unterbricht.
CPUSchedulingPolicy=fifo
CPUSchedulingPriority=90
Nice=-20

# UMGEBUNGSVARIABLEN
# Falls du CUDA in C++ nutzt, ist der Pfad wichtig
Environment=LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/local/lib
Environment=OAKD_DEVICE_IP=169.254.1.222
# Verhindert Fehlermeldungen von OpenCV/QT bei Headless-Betrieb
Environment=QT_QPA_PLATFORM=offscreen

# STABILITÄT
Restart=on-failure
RestartSec=2s                                                                                                                                        14,6       Anfang

[Install]
WantedBy=multi-user.target

```
In den models ordner wechseln und die Modelle herunterladen:

```bash
# Palm Detection (Findet die Hand)
wget https://github.com/geaxgx/depthai_hand_tracker/raw/main/models/palm_detection_sh4.blob

# Hand Landmarks (21 Punkte Erkennung)
wget https://github.com/geaxgx/depthai_hand_tracker/raw/main/models/hand_landmark_sh4.blob

Hinweis: sh4 steht für die Anzahl der verwendeten "Shave Cores" auf dem Myriad-Chip der Kamera. 4 Kerne sind ein guter Mittelweg zwischen Speed und Hitze.)

2. Die 21 Gelenkpunkte (Landmarks)
   Damit du weißt, welche Daten du später per OSC sendest, hier eine Übersicht der Indizes, die das Modell liefert:

0: Handgelenk (Wrist)

4, 8, 12, 16, 20: Die Fingerspitzen (Thumb, Index, Middle, Ring, Pinky)
3, 7, 11, 15, 19: Die mittleren Gelenke der Finger
2, 6, 10, 14, 18: Die unteren Gelenke der Finger
1, 5, 9, 13, 17: Die Basisgelenke der Finger
Diese Punkte kannst du nutzen, um die Position und Bewegung der Hand im Raum zu verfolgen und entsprechende OSC-Nachrichten zu generieren.
```