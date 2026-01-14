n#!/bin/bash
# Diagnose performance issues on Jetson Orin Nano

echo "═══════════════════════════════════════════════════════════"
echo "HandTrackingV3 Performance Diagnostic"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check MAXN mode
echo "🔍 1. MAXN Mode Status"
echo "───────────────────────────────────────────────────────────"
nvpmodel -q 2>/dev/null || echo "⚠️  nvpmodel not available"
echo ""

# Check CPU/GPU frequencies
echo "🔍 2. CPU/GPU Frequencies"
echo "───────────────────────────────────────────────────────────"
CPU_FREQ=$(($(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo 0) / 1000))
CPU_MAX=$(($(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || echo 0) / 1000))
GPU_FREQ=$(($(cat /sys/class/devfreq/17000000.gpu/cur_freq 2>/dev/null || echo 0) / 1000000))
GPU_MAX=$(($(cat /sys/class/devfreq/17000000.gpu/max_freq 2>/dev/null || echo 0) / 1000000))

echo "CPU: $CPU_FREQ / $CPU_MAX MHz"
echo "GPU: $GPU_FREQ / $GPU_MAX MHz"
if [ "$CPU_FREQ" -lt 1400 ] || [ "$GPU_FREQ" -lt 600 ]; then
    echo "⚠️  WARNING: Frequencies are LOW! MAXN mode may not be active"
    echo "   (Orin Nano MAXN: 1500 MHz CPU, 625 MHz GPU)"
    echo "   Run: sudo nvpmodel -m 0 && sudo jetson_clocks"
fi
echo ""

# Check temperature
echo "🔍 3. Temperature"
echo "───────────────────────────────────────────────────────────"
TEMP=$(($(cat /sys/devices/virtual/thermal/thermal_zone0/temp 2>/dev/null || echo 0) / 1000))
echo "GPU Temperature: $TEMP°C"
if [ "$TEMP" -gt 70 ]; then
    echo "⚠️  WARNING: Temperature is HIGH! Thermal throttling may be active"
fi
echo ""

# Check TensorRT engines
echo "🔍 4. TensorRT Engine Cache"
echo "───────────────────────────────────────────────────────────"
CACHE_DIR="/home/nvidia/.cache"
ENGINE_COUNT=$(find "$CACHE_DIR" -name "*.engine" -o -name "*.plan" 2>/dev/null | wc -l)
echo "Found $ENGINE_COUNT cached engine files"

if [ "$ENGINE_COUNT" -gt 0 ]; then
    echo "Cached engines:"
    find "$CACHE_DIR" -name "*.engine" -o -name "*.plan" 2>/dev/null | head -10
    echo ""
    echo "⚠️  IMPORTANT: If switching from another Jetson device, clear cache:"
    echo "   rm -f /home/nvidia/.cache/**/*.engine"
    echo "   rm -f /home/nvidia/.cache/**/*.plan"
fi
echo ""

# Check model files
echo "🔍 5. Model Files"
echo "───────────────────────────────────────────────────────────"
MODEL_DIR="$HOME/Developer/HandTrackingV3/models"
if [ -d "$MODEL_DIR" ]; then
    echo "Models directory: $MODEL_DIR"
    ls -lh "$MODEL_DIR"/*.onnx 2>/dev/null || echo "⚠️  No ONNX models found"
else
    echo "⚠️  Models directory not found: $MODEL_DIR"
fi
echo ""

# Check service status
echo "🔍 6. Service Status"
echo "───────────────────────────────────────────────────────────"
systemctl is-active hand-tracking.service >/dev/null 2>&1 && echo "✓ hand-tracking service is RUNNING" || echo "✗ hand-tracking service is NOT running"
systemctl is-active jetson_clocks.service >/dev/null 2>&1 && echo "✓ jetson_clocks service is enabled" || echo "✗ jetson_clocks service is NOT enabled"
systemctl is-active nvpmodel.service >/dev/null 2>&1 && echo "✓ nvpmodel service is enabled" || echo "✗ nvpmodel service is NOT enabled"
echo ""

echo "═══════════════════════════════════════════════════════════"
echo "Recommendations:"
echo "═══════════════════════════════════════════════════════════"
if [ "$CPU_FREQ" -lt 1400 ]; then
    echo "1. Enable MAXN mode:"
    echo "   sudo nvpmodel -m 0"
    echo "   sudo jetson_clocks"
fi

if [ "$ENGINE_COUNT" -gt 0 ]; then
    echo "2. Clear old TensorRT engine cache (if from different device):"
    echo "   rm -rf /home/nvidia/.cache"
    echo "   Then restart: systemctl restart hand-tracking.service"
fi

echo "3. Monitor real-time performance:"
echo "   tegrastats"
echo ""

