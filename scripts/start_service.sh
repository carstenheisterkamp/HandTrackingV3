#!/bin/bash
# Start HandTrackingService with MAXN power mode for optimal performance
# Power modes:
#   - 15W (default):  20 FPS @ 1280x720 FULL models
#   - MAXN (15W):     28-30 FPS @ 1280x720 FULL models (all cores active)

SERVICE_DIR="/home/nvidia/dev/HandTrackingV3/cmake-build-debug-remote-host"

if [ ! -f "$SERVICE_DIR/HandTrackingService" ]; then
    echo "❌ ERROR: HandTrackingService executable not found at $SERVICE_DIR/HandTrackingService"
    echo "   Build first: cd $SERVICE_DIR && cmake .. && ninja"
    exit 1
fi

echo "✅ Activating MAXN power mode for optimal GPU performance..."
sudo nvpmodel -m 0 2>/dev/null || echo "⚠️  Could not set MAXN mode (need sudo)"
sudo jetson_clocks 2>/dev/null || echo "⚠️  Could not set jetson_clocks (need sudo)"

echo "✅ Starting HandTrackingService from: $SERVICE_DIR"
echo "   Models: $SERVICE_DIR/models/"
echo "   .engine files: Pre-compiled (fast loading)"
echo ""
echo "   Expected performance:"
echo "   - FPS: 28-30 (with MAXN mode)"
echo "   - GPU: 80-90% utilization"
echo ""

# Set working directory to ensure relative path "models/" works correctly
cd "$SERVICE_DIR"

# Start the service
./HandTrackingService


