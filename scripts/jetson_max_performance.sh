#!/bin/bash
# Jetson Orin Nano - Maximum Performance
# Run ONCE with: sudo bash jetson_max_performance.sh

set -e

echo "=== Setting Jetson to Maximum Performance ==="

# Must be root
if [ "$EUID" -ne 0 ]; then
    echo "Run with: sudo bash $0"
    exit 1
fi

# 1. Set power mode to MAXN (highest performance)
echo "Setting power mode to MAXN..."
nvpmodel -m 0

# 2. Lock clocks to maximum
echo "Locking clocks to maximum..."
jetson_clocks

# 3. Verify
echo ""
echo "=== Verification ==="
nvpmodel -q
echo ""
jetson_clocks --show | head -20

echo ""
echo "=== DONE ==="
echo "Jetson is now at maximum performance."
echo "To make this permanent, run: sudo systemctl enable nvpmodel"
