#!/bin/bash
# Kill old HandTrackingService process and reset OAK-D connection
# Run this before starting a new instance after a crash/force-kill

echo "=== Stopping HandTrackingService and resetting OAK-D ==="

# Stop systemd service
sudo systemctl stop hand-tracking.service 2>/dev/null || true

# Kill any remaining processes (gracefully first, then force)
pkill -SIGTERM HandTrackingService 2>/dev/null
sleep 1
pkill -9 HandTrackingService 2>/dev/null || true

# Wait for process to fully terminate
sleep 2

# Check if port 8080 is still in use
if lsof -i :8080 -t >/dev/null 2>&1; then
    echo "Port 8080 still in use, killing process..."
    kill -9 $(lsof -i :8080 -t) 2>/dev/null || true
    sleep 1
fi

# Reset XLink by reloading network interface (for PoE)
# This forces the OAK-D to re-establish connection
echo "Resetting network interface for OAK-D PoE..."
IFACE=$(ip route get 169.254.1.222 2>/dev/null | grep -oP 'dev \K\S+')
if [ -n "$IFACE" ]; then
    sudo ip link set $IFACE down 2>/dev/null || true
    sleep 1
    sudo ip link set $IFACE up 2>/dev/null || true
    sleep 2
    echo "✓ Network interface $IFACE reset"
fi

# Alternative: Ping the OAK-D to ensure it's responsive
echo "Checking OAK-D connection..."
if ping -c 1 -W 2 169.254.1.222 >/dev/null 2>&1; then
    echo "✓ OAK-D Pro PoE is reachable"
else
    echo "⚠ OAK-D not responding - may need physical power cycle"
fi

# Verify port is free
if lsof -i :8080 >/dev/null 2>&1; then
    echo "ERROR: Port 8080 still blocked!"
    lsof -i :8080
    exit 1
else
    echo "✓ Port 8080 is free"
fi

echo "=== Ready for new service start ==="

