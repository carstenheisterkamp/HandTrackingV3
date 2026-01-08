import sys
import subprocess
import os
from pathlib import Path

# Ensure dependencies
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", package])

try:
    import blobconverter
except ImportError:
    install("blobconverter")
    try:
        import blobconverter
    except ImportError:
        print("Failed to install blobconverter")
        sys.exit(1)

# Install tf2onnx for conversion
try:
    import tf2onnx
    import onnx
except ImportError:
    install("tf2onnx")
    install("onnx")

MODELS_DIR = Path("/home/nvidia/dev/HandTrackingV3/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Sources
PALM_URL = "https://storage.googleapis.com/mediapipe-assets/palm_detection_full.tflite"
HAND_URL = "https://storage.googleapis.com/mediapipe-assets/hand_landmark_full.tflite"

SHAVES = 4 # Revert to 4 for Stability

def compile_model(url, name_base):
    print(f"Processing {name_base}...")

    # 1. Download TFLite
    import urllib.request
    tflite_path = MODELS_DIR / f"{name_base}.tflite"
    if not tflite_path.exists():
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, tflite_path)

    # 2. Convert to ONNX
    onnx_path = MODELS_DIR / f"{name_base}.onnx"
    # Always regenerate ONNX to be safe
    if onnx_path.exists():
        onnx_path.unlink()

    print(f"Converting {tflite_path} -> {onnx_path}...")
    subprocess.check_call([
        sys.executable, "-m", "tf2onnx.convert",
        "--tflite", str(tflite_path),
        "--output", str(onnx_path),
        "--opset", "12"
    ])

    # 3. Compile Blob
    blob_name = f"{name_base}_sh{SHAVES}.blob"
    blob_path = MODELS_DIR / blob_name
    print(f"Compiling {onnx_path} -> {blob_path} (SHAVES={SHAVES})...")

    opt_params = [
        "--reverse_input_channels",
        "--mean_values=[127.5,127.5,127.5]",
        "--scale_values=[127.5,127.5,127.5]"
    ]

    blob_path_result = blobconverter.from_onnx(
        model=str(onnx_path),
        shaves=SHAVES,
        version="2022.1",
        output_dir=str(MODELS_DIR),
        data_type="FP16",
        use_cache=True,
        optimizer_params=opt_params
    )

    # Rename/Verify
    p_result = Path(blob_path_result)
    if p_result.name != blob_name:
        if blob_path.exists():
            blob_path.unlink()
        p_result.rename(blob_path)

    print(f"Success: {blob_path} ({blob_path.stat().st_size} bytes)")

if __name__ == "__main__":
    compile_model(PALM_URL, "palm_detection_full")
    compile_model(HAND_URL, "hand_landmark_full")

