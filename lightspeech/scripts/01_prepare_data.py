#!/usr/bin/env python3
"""
Step 1: Data Preparation Script
Prepares datasets for training by downloading, preprocessing, and extracting features.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from code.data.loader import load_dataset
from code.data.preprocess import preprocess_audio
from code.data.augment import apply_augmentation


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for training')
    parser.add_argument('--dataset', type=str, default='CREMA-D',
                       choices=['CREMA-D', 'RAVDESS', 'Emo-DB'],
                       help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Directory containing raw data')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--sample_rate', type=int, default=16000,
                       help='Target sample rate')
    parser.add_argument('--segment_length', type=float, default=3.0,
                       help='Segment length in seconds')
    
    args = parser.parse_args()
    
    print(f"Preparing {args.dataset} dataset...")
    print(f"Input: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    
    # Load dataset
    print("\n[1/3] Loading dataset...")
    data = load_dataset(args.dataset, args.data_dir)
    
    # Preprocess audio
    print("\n[2/3] Preprocessing audio...")
    processed_data = preprocess_audio(
        data,
        sample_rate=args.sample_rate,
        segment_length=args.segment_length
    )
    
    # Apply augmentation (optional)
    print("\n[3/3] Applying data augmentation...")
    augmented_data = apply_augmentation(processed_data)
    
    # Save processed data
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    # TODO: Save processed data
    
    print(f"\n✓ Data preparation complete!")
    print(f"Processed data saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

