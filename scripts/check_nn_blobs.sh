#!/bin/bash
# NN Blob Checker Script
# Überprüft die Hand-Tracking Blobs auf Größe und Validität

set -e

MODELS_DIR="${1:-models}"

echo "═══════════════════════════════════════════════"
echo "NN Blob Checker - Hand Tracking Models"
echo "═══════════════════════════════════════════════"
echo ""

# Farben für Output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_blob() {
    local blob_path="$1"
    local expected_name="$2"
    local max_size_kb="$3"

    if [ ! -f "$blob_path" ]; then
        echo -e "${RED}✗ MISSING:${NC} $expected_name"
        echo "  Path: $blob_path"
        return 1
    fi

    local size_bytes=$(stat -f%z "$blob_path" 2>/dev/null || stat -c%s "$blob_path" 2>/dev/null)
    local size_kb=$((size_bytes / 1024))
    local size_mb=$(echo "scale=2; $size_bytes / 1048576" | bc)

    echo -e "${GREEN}✓ FOUND:${NC} $expected_name"
    echo "  Size: ${size_kb} KB (${size_mb} MB)"

    if [ $size_kb -gt $max_size_kb ]; then
        echo -e "  ${YELLOW}⚠ WARNING:${NC} Blob larger than expected (>${max_size_kb} KB)"
        echo "  → Large blobs may slow down Myriad X inference"
    fi

    # Check if blob is valid (should start with magic bytes)
    if command -v xxd &> /dev/null; then
        local header=$(xxd -l 16 -p "$blob_path")
        if [[ $header == "" ]]; then
            echo -e "  ${RED}✗ ERROR:${NC} Blob file is empty!"
            return 1
        fi
    fi

    echo ""
    return 0
}

echo "[1/2] Checking Palm Detection Blob..."
check_blob "$MODELS_DIR/palm_detection_full_sh4.blob" \
           "Palm Detection (SH4)" \
           2048  # Max 2 MB expected

echo "[2/2] Checking Hand Landmark Blob..."
check_blob "$MODELS_DIR/hand_landmark_full_sh4.blob" \
           "Hand Landmarks (SH4)" \
           4096  # Max 4 MB expected

echo "═══════════════════════════════════════════════"
echo "Summary"
echo "═══════════════════════════════════════════════"

# Check total CMX memory usage estimate
total_size=0
if [ -f "$MODELS_DIR/palm_detection_full_sh4.blob" ]; then
    palm_size=$(stat -f%z "$MODELS_DIR/palm_detection_full_sh4.blob" 2>/dev/null || stat -c%s "$MODELS_DIR/palm_detection_full_sh4.blob" 2>/dev/null)
    total_size=$((total_size + palm_size))
fi

if [ -f "$MODELS_DIR/hand_landmark_full_sh4.blob" ]; then
    landmark_size=$(stat -f%z "$MODELS_DIR/hand_landmark_full_sh4.blob" 2>/dev/null || stat -c%s "$MODELS_DIR/hand_landmark_full_sh4.blob" 2>/dev/null)
    total_size=$((total_size + landmark_size))
fi

total_mb=$(echo "scale=2; $total_size / 1048576" | bc)
echo "Total Blob Size: ${total_mb} MB"
echo ""

# CMX Memory warning (Myriad X has ~2.5 MB CMX)
if (( $(echo "$total_mb > 2.0" | bc -l) )); then
    echo -e "${YELLOW}⚠ WARNING: Combined blob size > 2 MB${NC}"
    echo "  Myriad X CMX Memory: ~2.5 MB total"
    echo "  Your blobs: ${total_mb} MB"
    echo "  → This may cause slow inference or OOM errors!"
    echo ""
    echo -e "${YELLOW}Recommendation:${NC}"
    echo "  1. Use smaller models (e.g., lite versions)"
    echo "  2. Run one NN at a time instead of parallel"
    echo "  3. Consider moving NNs to Jetson (TensorRT)"
else
    echo -e "${GREEN}✓ Total size OK for Myriad X CMX memory${NC}"
fi

echo ""
echo "═══════════════════════════════════════════════"
echo "Performance Tips"
echo "═══════════════════════════════════════════════"
echo ""
echo "If Device FPS is low (< 25):"
echo "  1. Reduce preview resolution (320x180 vs 640x360)"
echo "  2. Use NN threads = 1 (not 2)"
echo "  3. Check if blobs are sh4 optimized (for RVC2)"
echo "  4. Consider moving Hand NN to Jetson TensorRT"
echo ""

# Optional: Check model format
if command -v file &> /dev/null; then
    echo "Blob File Types:"
    file "$MODELS_DIR"/*.blob 2>/dev/null | grep -E "palm|hand" || echo "  (file command not available)"
fi

