#!/bin/bash
# OAK-D PoE Reconnect Helper
# Fixes connection issues without rebooting Jetson

set -e

OAK_IP="${1:-169.254.1.222}"

echo "═══════════════════════════════════════════════"
echo "OAK-D PoE Reconnect Helper"
echo "═══════════════════════════════════════════════"
echo ""

# 1. Kill any running HandTrackingV3 instances
echo "[1/5] Killing existing HandTrackingV3 processes..."
if pgrep -x "HandTrackingV3" > /dev/null; then
    sudo pkill -9 HandTrackingV3 || true
    echo "  ✓ Killed HandTrackingV3"
    sleep 2
else
    echo "  ✓ No HandTrackingV3 running"
fi

# 2. Clear ARP cache for OAK-D IP
echo ""
echo "[2/5] Clearing ARP cache for ${OAK_IP}..."
if arp -n | grep -q "${OAK_IP}"; then
    sudo arp -d "${OAK_IP}" 2>/dev/null || true
    echo "  ✓ ARP cache cleared"
else
    echo "  ⚠ IP not in ARP cache (might be good)"
fi

# 3. Wait for network stack to settle
echo ""
echo "[3/5] Waiting 3 seconds for network stack..."
sleep 3
echo "  ✓ Done"

# 4. Test connectivity
echo ""
echo "[4/5] Testing connectivity to ${OAK_IP}..."
if ping -c 2 -W 2 "${OAK_IP}" > /dev/null 2>&1; then
    echo "  ✓ OAK-D is reachable!"
else
    echo "  ✗ OAK-D is NOT reachable"
    echo ""
    echo "═══ TROUBLESHOOTING ═══"
    echo "1. Check OAK-D LED - is it on?"
    echo "2. Check PoE cable connection"
    echo "3. Check PoE switch/injector power"
    echo "4. Try: sudo systemctl restart NetworkManager"
    echo "5. Last resort: Power cycle OAK-D (unplug 10s)"
    echo "═══════════════════════"
    exit 1
fi

# 5. Show current network state
echo ""
echo "[5/5] Current network state:"
echo "  ARP entry:"
arp -n | grep "${OAK_IP}" || echo "    (none - will be created on first connect)"
echo ""
echo "  Network interface status:"
ip addr show | grep -A 2 "inet.*${OAK_IP%.*}" || echo "    (no matching interface)"

echo ""
echo "═══════════════════════════════════════════════"
echo "✓ READY TO RECONNECT"
echo "═══════════════════════════════════════════════"
echo ""
echo "You can now start HandTrackingV3:"
echo "  cd ~/dev/HandTrackingV3/cmake-build-debug-remote-host"
echo "  ./HandTrackingV3"
echo ""
echo "If connection still fails:"
echo "  1. Run: sudo systemctl restart NetworkManager"
echo "  2. Power cycle OAK-D (unplug PoE for 10 seconds)"
echo "  3. As LAST resort: sudo reboot"

