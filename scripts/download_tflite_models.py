#!/usr/bin/env python3
"""
Download MediaPipe Hand TFLite models.

Run this on any machine with internet access.
Then run convert_to_onnx.py on the Jetson to convert to ONNX.

Usage:
    python3 download_tflite_models.py
"""

import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

MODELS = {
    "palm_detection_lite.tflite":
        "https://storage.googleapis.com/mediapipe-assets/palm_detection_lite.tflite",
    "hand_landmark_lite.tflite":
        "https://storage.googleapis.com/mediapipe-assets/hand_landmark_lite.tflite",
}

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Downloading MediaPipe TFLite models...")
    print("=" * 50)

    for filename, url in MODELS.items():
        output_path = os.path.join(MODELS_DIR, filename)

        if os.path.exists(output_path):
            print(f"✓ {filename} (already exists)")
            continue

        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, output_path)
            size_kb = os.path.getsize(output_path) / 1024
            print(f"  ✓ Downloaded ({size_kb:.1f} KB)")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print("\n" + "=" * 50)
    print("Next: Run convert_to_onnx.py on Jetson")
    print("=" * 50)


if __name__ == "__main__":
    main()

