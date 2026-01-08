#!/usr/bin/env python3
"""
Blob Converter for Hand Tracking Models
Converts models to SHAVE 5 or 6 variants for OAK-D Pro PoE
Target: Myriad X with 16 SHAVEs total (2 models × 6-8 SHAVEs each)
"""

import sys
import subprocess
import argparse
from pathlib import Path

def install_blobconverter():
    """Ensure blobconverter is installed"""
    try:
        import blobconverter
        print(f"✓ blobconverter {blobconverter.__version__} already installed")
    except ImportError:
        print("Installing blobconverter...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "blobconverter"])
        import blobconverter
        print(f"✓ Installed blobconverter {blobconverter.__version__}")

def convert_model(model_name: str, shaves: int, output_dir: Path, zoo_type: str = "depthai"):
    """
    Convert a model from DepthAI zoo to blob with specified SHAVE count

    Args:
        model_name: Name of model in zoo (e.g., 'palm_detection_128x128')
        shaves: Number of SHAVEs to use (4, 5, 6, or 8)
        output_dir: Directory to save blob
        zoo_type: Type of zoo ('depthai', 'intel')
    """
    import blobconverter

    output_name = f"{model_name}_sh{shaves}.blob"
    output_path = output_dir / output_name

    print(f"\n>>> Converting {model_name} with {shaves} SHAVEs...")
    print(f"    Output: {output_path}")

    try:
        blob_path = blobconverter.from_zoo(
            name=model_name,
            zoo_type=zoo_type,
            shaves=shaves,
            output_dir=str(output_dir),
            use_cache=False  # Always download fresh to avoid corrupted cache
        )

        # Rename to include SHAVE count
        blob_path_obj = Path(blob_path)
        if blob_path_obj.exists():
            blob_path_obj.rename(output_path)
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✓ Created {output_name} ({size_mb:.2f} MB)")
            return output_path
        else:
            print(f"✗ Conversion failed for {model_name}")
            return None

    except Exception as e:
        print(f"✗ Error converting {model_name}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Convert hand tracking models to blob format for OAK-D Pro PoE"
    )
    parser.add_argument(
        "--shaves",
        type=int,
        choices=[4, 5, 6, 8],
        default=6,
        help="Number of SHAVEs per model (default: 6, optimal for dual-model pipeline)"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(__file__).parent.parent / "models",
        help="Output directory for blob files"
    )
    parser.add_argument(
        "--palm-only",
        action="store_true",
        help="Only convert palm detection model"
    )
    parser.add_argument(
        "--landmark-only",
        action="store_true",
        help="Only convert hand landmark model"
    )

    args = parser.parse_args()

    print("=== DepthAI Blob Converter for Hand Tracking ===")
    print(f"Target: {args.shaves} SHAVEs per model")
    print(f"Output: {args.models_dir}")
    print()

    # Ensure output directory exists
    args.models_dir.mkdir(parents=True, exist_ok=True)

    # Install blobconverter if needed
    install_blobconverter()

    success_count = 0

    # Convert palm detection
    if not args.landmark_only:
        # Try multiple model names (zoo naming can vary)
        palm_models = [
            "palm_detection_128x128",
            "palm-detection-128x128",
            "palmdet-128x128"
        ]

        palm_success = False
        for model_name in palm_models:
            try:
                result = convert_model(model_name, args.shaves, args.models_dir)
                if result:
                    palm_success = True
                    success_count += 1
                    break
            except Exception as e:
                print(f"  Trying next palm detection variant... ({e})")
                continue

        if not palm_success:
            print("✗ Could not find palm detection model in zoo")
            print("  Try manually downloading from MediaPipe or use existing sh4 blob")

    # Convert hand landmark
    if not args.palm_only:
        landmark_models = [
            "hand_landmark_full",
            "hand-landmark-full",
            "hand_landmark_lite",
            "hand-landmark-lite"
        ]

        landmark_success = False
        for model_name in landmark_models:
            try:
                result = convert_model(model_name, args.shaves, args.models_dir)
                if result:
                    landmark_success = True
                    success_count += 1
                    break
            except Exception as e:
                print(f"  Trying next hand landmark variant... ({e})")
                continue

        if not landmark_success:
            print("✗ Could not find hand landmark model in zoo")
            print("  Try manually downloading from MediaPipe or use existing sh4 blob")

    print("\n=== Conversion Summary ===")
    print(f"Successfully converted: {success_count} model(s)")
    print(f"\nAvailable blobs in {args.models_dir}:")

    for blob_file in sorted(args.models_dir.glob("*.blob")):
        size_mb = blob_file.stat().st_size / (1024 * 1024)
        print(f"  {blob_file.name:40s} {size_mb:8.2f} MB")

    print("\n💡 Tip: Update PipelineManager.cpp to use the new blob files")
    print(f"    e.g., 'models/palm_detection_sh{args.shaves}.blob'")

    return 0 if success_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())

