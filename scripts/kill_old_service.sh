#!/bin/bash
# Kill old HandTrackingService process and reset OAK-D connection
# Run this before starting a new instance after a crash/force-kill

OAK_IP="169.254.1.222"

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

# ══════════════════════════════════════════════════════════════
# CRITICAL: Clear ARP cache for OAK-D IP
# This fixes "device not found" after hard process kill
# ══════════════════════════════════════════════════════════════
echo "Clearing ARP cache for OAK-D ($OAK_IP)..."
sudo arp -d $OAK_IP 2>/dev/null || true
sudo ip neigh flush dev eth0 2>/dev/null || true
sudo ip neigh flush dev eth1 2>/dev/null || true

# Reset XLink by reloading network interface (for PoE)
# This forces the OAK-D to re-establish connection
echo "Resetting network interface for OAK-D PoE..."
IFACE=$(ip route get $OAK_IP 2>/dev/null | grep -oP 'dev \K\S+')
if [ -n "$IFACE" ]; then
    echo "  Interface: $IFACE"
    sudo ip link set $IFACE down 2>/dev/null || true
    sleep 1
    sudo ip link set $IFACE up 2>/dev/null || true
    sleep 2
    echo "✓ Network interface $IFACE reset"
fi

# Re-ping to refresh ARP cache with correct MAC
echo "Refreshing ARP cache..."
ping -c 2 -W 1 $OAK_IP >/dev/null 2>&1 || true

# Verify connection
echo "Checking OAK-D connection..."
if ping -c 1 -W 2 $OAK_IP >/dev/null 2>&1; then
    echo "✓ OAK-D Pro PoE is reachable at $OAK_IP"
    arp -n $OAK_IP 2>/dev/null || true
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

