"""
Script 02: Train baseline teacher model
"""

import os
import sys
# Temporary workaround for OpenMP runtime conflicts (unsafe): allow duplicate OpenMP libs.
# This avoids the "Initializing libomp.dll, but found libiomp5md.dll already initialized"
# error that aborts execution. Prefer fixing the environment long-term (see README).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse


proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from lightspeech.code.training.trainer import BaselineTrainer
from lightspeech.code.models.compression import load_teacher_model
from lightspeech.code.data.loader import get_dataloaders
from lightspeech.code.evaluation.visualization import plot_training_curves
import json
import os

def main(args):
    print("=== STEP 2: Training baseline model ===")

    train_loader, val_loader, _ = get_dataloaders(
        args.data,
        batch_size=args.batch_size,
        arch=args.arch
    )

    model = load_teacher_model(num_classes=args.num_classes, arch=args.arch)

    trainer = BaselineTrainer(
        model=model,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device
    )

    history = trainer.train(train_loader, val_loader)
    # save history JSON
    os.makedirs(os.path.join(args.output, '..', 'logs'), exist_ok=True)
    logs_dir = os.path.abspath(os.path.join(args.output, '..', 'logs'))
    with open(os.path.join(logs_dir, 'baseline_training.json'), 'w') as f:
        json.dump(history, f)

    # generate plot
    os.makedirs(os.path.join(args.output, '..', 'plots'), exist_ok=True)
    plots_dir = os.path.abspath(os.path.join(args.output, '..', 'plots'))
    # convert history list of dicts to dict of lists expected by plot_training_curves
    hist_dict = {"train_loss": [h['train_loss'] for h in history],
                 "val_loss": [h['val_loss'] for h in history],
                 "train_acc": [h['train_acc'] for h in history],
                 "val_acc": [h['val_acc'] for h in history]}
    plot_training_curves(hist_dict, os.path.join(plots_dir, 'baseline_training_curves.png'))

    trainer.save(args.output)

    print(f"Baseline model saved at: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arch", default="cnn2d", help="Model architecture: cnn2d|mobilenet|efficientnet|transformer")
    args = parser.parse_args()

    main(args)
