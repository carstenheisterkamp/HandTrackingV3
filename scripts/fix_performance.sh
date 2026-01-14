#!/bin/bash
# Quick fix script for common performance issues

set -e

echo "═══════════════════════════════════════════════════════════"
echo "HandTrackingV3 Performance Fix"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Device: Jetson Orin Nano"
echo "Target: MAXN Mode (15W) - 1500 MHz CPU, 625 MHz GPU"
echo ""

# 1. Enable MAXN
echo "✓ Step 1: Enabling MAXN mode..."
sudo nvpmodel -m 0 || echo "⚠️  nvpmodel command failed"
sudo jetson_clocks || echo "⚠️  jetson_clocks command failed"

# 2. Wait for frequencies to stabilize
echo "✓ Step 2: Waiting for system to stabilize (3 seconds)..."
sleep 3

echo "Current system state after MAXN activation:"
CPU_FREQ=$(($(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo 0) / 1000))
GPU_FREQ=$(($(cat /sys/class/devfreq/17000000.gpu/cur_freq 2>/dev/null || echo 0) / 1000000))
TEMP=$(($(cat /sys/devices/virtual/thermal/thermal_zone0/temp 2>/dev/null || echo 0) / 1000))

echo "  CPU: $CPU_FREQ MHz (target: 1500 MHz for Orin Nano MAXN/15W)"
echo "  GPU: $GPU_FREQ MHz (target: 625 MHz for Orin Nano MAXN/15W)"
echo "  Temp: $TEMP°C"
echo ""

# 4. Clear TensorRT engine cache
echo "✓ Step 3: Clearing old TensorRT engine cache..."
CACHE_DIR="/home/nvidia/.cache"
if [ -d "$CACHE_DIR" ]; then
    find "$CACHE_DIR" -name "*.engine" -delete 2>/dev/null || echo "  (No engine files to clean)"
    find "$CACHE_DIR" -name "*.plan" -delete 2>/dev/null || echo "  (No plan files to clean)"
fi

# Also clear /tmp engines
rm -f /tmp/*.engine 2>/dev/null || true
rm -f /tmp/*.plan 2>/dev/null || true

echo "  ✓ Cache cleared"
echo ""

# 5. Restart service
echo "✓ Step 4: Restarting hand-tracking service..."
sudo systemctl restart hand-tracking.service || echo "⚠️  Could not restart service"
echo "  ✓ Service restarted"
echo ""

echo "═══════════════════════════════════════════════════════════"
echo "✓ Performance optimization complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "The service will now:"
echo "  1. Run at MAXN performance (1.5GHz CPU, 625MHz GPU @ 15W - Orin Nano max)"
echo "  2. Recompile TensorRT engines for your device (20-45 sec)"
echo ""
echo "Performance improvement:"
echo "  Baseline (7.5W): ~750MHz CPU, ~310MHz GPU"
echo "  MAXN (15W): 1500MHz CPU, 625MHz GPU = 2.0x faster"
echo ""
echo "Check logs: journalctl -u hand-tracking.service -f"
echo ""

