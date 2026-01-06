# Service Management Instructions

## 1. Manuelles Starten (Debugging)

Um den Service direkt im Terminal zu starten (um Logs direkt zu sehen):

```bash
cd /home/nvidia/dev/HandTrackingV3/cmake-build-debug-remote-host
./HandTrackingService
```

Beenden mit `Ctrl+C`.

## 2. Automatische Installation (Systemd)

Damit der Service beim Booten automatisch startet und im Hintergrund läuft:

### Installation

1. Kopiere die Service-Datei nach `/etc/systemd/system/`:
   ```bash
   sudo cp /home/nvidia/dev/HandTrackingV3/scripts/hand-tracking.service /etc/systemd/system/
   ```

2. Lade den Systemd-Daemon neu:
   ```bash
   sudo systemctl daemon-reload
   ```

3. Aktiviere den Autostart:
   ```bash
   sudo systemctl enable hand-tracking.service
   ```

### Steuerung

*   **Starten:** `sudo systemctl start hand-tracking.service`
*   **Stoppen:** `sudo systemctl stop hand-tracking.service`
*   **Neustarten:** `sudo systemctl restart hand-tracking.service`
*   **Status prüfen:** `sudo systemctl status hand-tracking.service`

### Logs ansehen

Die Logs des Hintergrund-Services kannst du mit `journalctl` ansehen:

*   **Live-Logs (Tail):**
    ```bash
    sudo journalctl -u hand-tracking.service -f
    ```
*   **Alle Logs:**
    ```bash
    sudo journalctl -u hand-tracking.service
    ```

