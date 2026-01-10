#!/bin/bash
# Quick check: Which models are available and converted?
# Run on Jetson: ./check_models.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/models"

echo "═══════════════════════════════════════════════════════"
echo "MODEL STATUS CHECK"
echo "═══════════════════════════════════════════════════════"
echo ""

# Check TFLite models
echo "📦 TFLite Models (Source):"
echo ""

check_file() {
    local file="$1"
    if [ -f "$MODELS_DIR/$file" ]; then
        local size=$(du -h "$MODELS_DIR/$file" | cut -f1)
        echo "  ✅ $file ($size)"
        return 0
    else
        echo "  ❌ $file (missing)"
        return 1
    fi
}

check_file "palm_detection_lite.tflite"
PALM_LITE_TFLITE=$?

check_file "hand_landmark_lite.tflite"
LANDMARK_LITE_TFLITE=$?

check_file "palm_detection_full.tflite"
PALM_FULL_TFLITE=$?

check_file "hand_landmark_full.tflite"
LANDMARK_FULL_TFLITE=$?

echo ""
echo "🔄 ONNX Models (Converted):"
echo ""

check_file "palm_detection.onnx"
PALM_LITE_ONNX=$?

check_file "hand_landmark.onnx"
LANDMARK_LITE_ONNX=$?

check_file "palm_detection_full.onnx"
PALM_FULL_ONNX=$?

check_file "hand_landmark_full.onnx"
LANDMARK_FULL_ONNX=$?

echo ""
echo "⚡ TensorRT Engines (Cached):"
echo ""

check_file "palm_detection.onnx.engine" || true
check_file "hand_landmark.onnx.engine" || true
check_file "palm_detection_full.onnx.engine" || true
check_file "hand_landmark_full.onnx.engine" || true

echo ""
echo "═══════════════════════════════════════════════════════"
echo "SUMMARY"
echo "═══════════════════════════════════════════════════════"
echo ""

# Check what needs to be done
NEED_TFLITE=false
NEED_CONVERT=false

if [ $PALM_FULL_TFLITE -ne 0 ] || [ $LANDMARK_FULL_TFLITE -ne 0 ]; then
    NEED_TFLITE=true
fi

if [ $PALM_LITE_ONNX -ne 0 ] || [ $LANDMARK_LITE_ONNX -ne 0 ] ||
   [ $PALM_FULL_ONNX -ne 0 ] || [ $LANDMARK_FULL_ONNX -ne 0 ]; then
    NEED_CONVERT=true
fi

if [ "$NEED_TFLITE" = true ]; then
    echo "⚠️  Full models missing!"
    echo "   → Run: python3 scripts/download_tflite_models.py"
    echo ""
fi

if [ "$NEED_CONVERT" = true ]; then
    echo "⚠️  ONNX conversion needed!"
    echo "   → Run: python3 scripts/convert_to_onnx.py"
    echo ""
fi

if [ "$NEED_TFLITE" = false ] && [ "$NEED_CONVERT" = false ]; then
    echo "✅ All models ready!"
    echo ""
    echo "Current mode in src/main.cpp:"
    grep "const bool USE_FULL_MODELS" "$PROJECT_ROOT/src/main.cpp" || echo "   (not found)"
    echo ""
    echo "To switch models:"
    echo "   → ./scripts/switch_models.sh [lite|full]"
fi

echo ""
echo "═══════════════════════════════════════════════════════"

