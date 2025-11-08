#!/usr/bin/env python3
"""
Step 2: Baseline Model Training
Trains the full-size baseline emotion recognition model.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from code.models.baseline import BaselineModel
from code.training.trainer import Trainer
import yaml


def main():
    parser = argparse.ArgumentParser(description='Train baseline model')
    parser.add_argument('--config', type=str, default='configs/baseline.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Directory containing processed data')
    parser.add_argument('--output_dir', type=str, default='results/models',
                       help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Training Baseline Model")
    print(f"Config: {args.config}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    
    # Initialize model
    print("\n[1/4] Initializing model...")
    model = BaselineModel(config['model'])
    
    # Initialize trainer
    print("\n[2/4] Setting up trainer...")
    trainer = Trainer(model, config['training'], args.data_dir)
    
    # Train model
    print("\n[3/4] Training model...")
    trainer.train()
    
    # Save model
    print("\n[4/4] Saving model...")
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_path / 'baseline.pth')
    
    print(f"\n✓ Training complete!")
    print(f"Model saved to: {output_path / 'baseline.pth'}")


if __name__ == '__main__':
    main()

