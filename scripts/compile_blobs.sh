#!/bin/bash
# Compile Neural Network Blobs with optimal SHAVE configuration
# Target: 6 SHAVEs per network (2 NNs × 6 = 12 total, matching Myriad X capacity)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

# Configuration
SHAVES=6
OPENVINO_VERSION="2022.1"

echo "=== Blob Compiler for Hand Tracking Service ==="
echo "Target: $SHAVES SHAVEs per network"
echo "Models directory: $MODELS_DIR"
echo ""

# Check if blobconverter is installed
if ! command -v blobconverter &> /dev/null; then
    echo "Installing blobconverter..."
    pip3 install blobconverter
fi

# Palm Detection Model
echo ">>> Compiling Palm Detection (128x128, $SHAVES SHAVEs)..."
blobconverter --zoo-name palm_detection_128x128 \
    --zoo-type depthai \
    --shaves $SHAVES \
    --output-dir "$MODELS_DIR" \
    --compile-params "-ip U8" \
    2>/dev/null || {
    echo "Zoo model not found. Trying MediaPipe source..."

    # Download from MediaPipe if not in zoo
    PALM_TFLITE="$MODELS_DIR/palm_detection.tflite"
    if [ ! -f "$PALM_TFLITE" ]; then
        echo "Downloading palm_detection.tflite..."
        curl -L "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" \
            -o "$MODELS_DIR/hand_landmarker.task" 2>/dev/null || true
    fi

    # Use blobconverter with local file
    if [ -f "$PALM_TFLITE" ]; then
        blobconverter --tflite-path "$PALM_TFLITE" \
            --shaves $SHAVES \
            --output-dir "$MODELS_DIR" \
            --name "palm_detection_sh${SHAVES}"
    fi
}

# Hand Landmark Model
echo ">>> Compiling Hand Landmarks (224x224, $SHAVES SHAVEs)..."
blobconverter --zoo-name hand_landmark_full \
    --zoo-type depthai \
    --shaves $SHAVES \
    --output-dir "$MODELS_DIR" \
    --compile-params "-ip U8" \
    2>/dev/null || {
    echo "Trying alternative model source..."

    # Try hand_landmark from depthai model zoo
    blobconverter --zoo-name hand-landmark-lite \
        --zoo-type depthai \
        --shaves $SHAVES \
        --output-dir "$MODELS_DIR" \
        2>/dev/null || echo "Manual blob compilation required."
}

echo ""
echo "=== Compilation Complete ==="
echo "Blobs in $MODELS_DIR:"
ls -la "$MODELS_DIR"/*.blob 2>/dev/null || echo "No blobs found."

echo ""
echo "Update PipelineManager.cpp with new blob paths if filenames changed."

