
#!/bin/bash
# Setup script - Run ONCE with: sudo bash setup_performance_autostart.sh

set -e

if [ "$EUID" -ne 0 ]; then
    echo "Run with: sudo bash $0"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Installing systemd service..."
cp "$SCRIPT_DIR/jetson-performance.service" /etc/systemd/system/
chmod 644 /etc/systemd/system/jetson-performance.service

echo "Enabling service..."
systemctl daemon-reload
systemctl enable jetson-performance.service

echo "Starting service now..."
systemctl start jetson-performance.service

echo ""
echo "=== DONE ==="
echo "Jetson will now boot in MAXN mode automatically."
echo "Check status: systemctl status jetson-performance"

