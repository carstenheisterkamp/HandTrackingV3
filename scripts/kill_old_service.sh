#!/bin/bash
# Kill old HandTrackingService process and free port 8080

echo "Stopping old HandTrackingService processes..."

# Stop systemd service
sudo systemctl stop hand-tracking.service 2>/dev/null || true

# Kill any remaining processes
sudo pkill -9 HandTrackingService 2>/dev/null || true

# Wait a moment
sleep 1

# Check if port 8080 is still in use
if sudo lsof -i :8080 -t >/dev/null 2>&1; then
    echo "Port 8080 still in use, killing process..."
    sudo kill -9 $(sudo lsof -i :8080 -t) 2>/dev/null || true
    sleep 1
fi

# Verify port is free
if sudo lsof -i :8080 >/dev/null 2>&1; then
    echo "ERROR: Port 8080 still blocked!"
    sudo lsof -i :8080
    exit 1
else
    echo "✓ Port 8080 is free"
fi

echo "✓ Ready for new service start"

