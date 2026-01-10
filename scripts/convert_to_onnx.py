#!/usr/bin/env python3
"""
Convert TFLite models to ONNX for TensorRT.

Run this on the Jetson (where TensorFlow is installed).

Usage:
    python3 convert_to_onnx.py

Prerequisites:
    pip3 install tf2onnx onnx
"""

import os
import subprocess
import sys

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

CONVERSIONS = [
    # Lite models (default, fast)
    {
        "tflite": "palm_detection_lite.tflite",
        "onnx": "palm_detection.onnx",
        "opset": 13,
    },
    {
        "tflite": "hand_landmark_lite.tflite",
        "onnx": "hand_landmark.onnx",
        "opset": 13,
    },
    # Full models (higher accuracy, slower - for testing)
    {
        "tflite": "palm_detection_full.tflite",
        "onnx": "palm_detection_full.onnx",
        "opset": 13,
    },
    {
        "tflite": "hand_landmark_full.tflite",
        "onnx": "hand_landmark_full.onnx",
        "opset": 13,
    },
]


def main():
    print("Converting TFLite to ONNX for TensorRT")
    print("=" * 50)

    # Check dependencies
    try:
        import tf2onnx
        print("✓ tf2onnx found")
    except ImportError:
        print("Installing tf2onnx...")
        subprocess.run([sys.executable, "-m", "pip", "install", "tf2onnx", "onnx"])

    for conv in CONVERSIONS:
        tflite_path = os.path.join(MODELS_DIR, conv["tflite"])
        onnx_path = os.path.join(MODELS_DIR, conv["onnx"])

        if not os.path.exists(tflite_path):
            print(f"\n✗ {conv['tflite']} not found!")
            print("  Run download_tflite_models.py first")
            continue

        if os.path.exists(onnx_path):
            print(f"\n✓ {conv['onnx']} already exists")
            continue

        print(f"\nConverting {conv['tflite']} → {conv['onnx']}...")

        result = subprocess.run(
            [sys.executable, "-m", "tf2onnx.convert",
             "--tflite", tflite_path,
             "--output", onnx_path,
             "--opset", str(conv["opset"])],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            size_kb = os.path.getsize(onnx_path) / 1024
            print(f"  ✓ Converted ({size_kb:.1f} KB)")
        else:
            print(f"  ✗ Failed:")
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)

    # Verify
    print("\n" + "=" * 50)
    print("Verifying ONNX models...")

    try:
        import onnx
        for conv in CONVERSIONS:
            onnx_path = os.path.join(MODELS_DIR, conv["onnx"])
            if os.path.exists(onnx_path):
                model = onnx.load(onnx_path)
                onnx.checker.check_model(model)
                print(f"✓ {conv['onnx']} is valid")
    except Exception as e:
        print(f"Verification error: {e}")

    print("\n" + "=" * 50)
    print("Done! TensorRT will build .engine files on first run.")
    print("=" * 50)


if __name__ == "__main__":
    main()

