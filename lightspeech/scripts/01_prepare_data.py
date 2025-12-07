"""
Script 01: Prepare RAVDESS dataset
"""
# Temporary workaround for OpenMP runtime conflicts (unsafe): allow duplicate OpenMP libs.
# This avoids the "Initializing libomp.dll, but found libiomp5md.dll already initialized"
# error that aborts execution. Prefer fixing the environment long-term (see README).
import os
import sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Ensure project root is on sys.path so `lightspeech` package imports work when
# the script is executed directly (e.g. python lightspeech/scripts/01_prepare_data.py)
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import argparse
from lightspeech.code.data.preprocess import AudioPreprocessor
from lightspeech.code.data.loader import create_splits


def main(args):
    print("\n=== STEP 1: Preparing dataset ===")

    # Create output directory if missing
    os.makedirs(args.output, exist_ok=True)

    print(f"[INFO] Raw data: {args.raw_data}")
    print(f"[INFO] Output dir: {args.output}")
    print(f"[INFO] Augmentation: {args.augment}\n")

    # Preprocessor
    processor = AudioPreprocessor(
        sample_rate=16000,
        segment_length=3.0,
        n_mels=64,
        augment=args.augment
    )

    # Feature extraction
    print("[INFO] Extracting features...")
    processor.process_dataset(
        raw_dataset_path=args.raw_data,
        output_path=args.output
    )

    # Dataset splits
    print("[INFO] Creating train/val/test splits...")
    create_splits(
        processed_dir=args.output,
        seed=42,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15
    )

    print("\n[SUCCESS] Dataset prepared successfully!")
    print(f"Processed files saved to: {args.output}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare audio dataset for training.")
    parser.add_argument("--raw_data", type=str, required=True, help="Path to raw RAVDESS files")
    parser.add_argument("--output", type=str, required=True, help="Directory for processed dataset")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")

    args = parser.parse_args()
    main(args)
