#!/usr/bin/env python3
"""
Kompiliert hand_landmark_full.tflite zu einem 6-SHAVE Blob für OAK-D.

Ablauf:
1. TFLite -> ONNX (via tf2onnx)
2. ONNX -> Blob (via blobconverter)
"""
import subprocess
import sys
from pathlib import Path

MODELS_DIR = Path("/home/nvidia/dev/HandTrackingV3/models")
TFLITE = MODELS_DIR / "hand_landmark_full.tflite"
ONNX = MODELS_DIR / "hand_landmark_full.onnx"
SHAVES = 6

def install_if_missing(package):
    try:
        __import__(package.replace("-", "_"))
    except ImportError:
        print(f"Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package, "--user"], check=True)

def main():
    # Prüfe TFLite
    if not TFLITE.exists():
        print(f"ERROR: {TFLITE} nicht gefunden!")
        sys.exit(1)
    print(f"TFLite: {TFLITE} ({TFLITE.stat().st_size} bytes)")

    # Step 1: TFLite -> ONNX
    if not ONNX.exists():
        print("Konvertiere TFLite -> ONNX...")
        install_if_missing("tf2onnx")
        subprocess.run([
            sys.executable, "-m", "tf2onnx.convert",
            "--tflite", str(TFLITE),
            "--output", str(ONNX),
            "--opset", "12"
        ], check=True)
    print(f"ONNX: {ONNX} ({ONNX.stat().st_size} bytes)")

    # Step 2: ONNX -> Blob
    print(f"Kompiliere zu {SHAVES}-SHAVE Blob...")
    import blobconverter
    blob = blobconverter.from_onnx(
        model=str(ONNX),
        shaves=SHAVES,
        output_dir=str(MODELS_DIR),
        data_type="FP16"
    )
    print(f"SUCCESS: {blob}")

if __name__ == "__main__":
    main()

