"""
Script 03: Compress model (distillation + quantization)
"""

import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import torch

# ensure project root is importable when running the script directly
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from lightspeech.code.training.distillation import DistillationTrainer
from lightspeech.code.training.quantization import QuantizationTrainer, QATTrainer
from lightspeech.code.training.pruning import PruningSpec, prune_and_report, fine_tune_after_pruning
from lightspeech.code.models.student import StudentModel
from lightspeech.code.models.compression import load_teacher_model
from lightspeech.code.data.loader import get_dataloaders
from lightspeech.code.evaluation.visualization import plot_training_curves, plot_compression_results
import json
import os
import tempfile
import numpy as np
from pathlib import Path
import subprocess


def _is_valid_onnx(path: str) -> bool:
    """Lightweight ONNX validation; returns False if onnx is missing or model is invalid."""
    try:
        import onnx
        m = onnx.load(path)
        onnx.checker.check_model(m)
        return True
    except Exception as e:
        print(f"[WARN] ONNX validation failed for {path}: {e}")
        return False

def main(args):
    print("=== STEP 3: Compressing model ===")

    train_loader, val_loader, _ = get_dataloaders(
        args.data,
        batch_size=args.batch_size
    )

    teacher = load_teacher_model(ckpt_path=args.teacher_ckpt)
    student = StudentModel(num_classes=args.num_classes)

    # ---- DISTILLATION ----
    if args.distill:
        print("Running knowledge distillation...")
        distiller = DistillationTrainer(
            teacher=teacher,
            student=student,
            temperature=args.temperature,
            alpha=args.alpha,
            lr=args.lr,
            epochs=args.epochs
        )
        dist_history = distiller.train(train_loader, val_loader)
        # save distillation history
        plots_dir = Path(args.output).parent / 'plots'
        logs_dir = Path(args.output).parent / 'logs'
        results_dir = Path(args.output).parent / 'compression'
        plots_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        with open(logs_dir / 'distillation_training.json', 'w') as f:
            json.dump(dist_history, f)

        # convert dist_history to plot dict if val_acc present
        hist = {"train_loss": [d.get('loss') for d in dist_history],
                "val_loss": [None]*len(dist_history),
                "train_acc": [None]*len(dist_history),
                "val_acc": [d.get('val_acc') if d.get('val_acc') is not None else 0.0 for d in dist_history]}
        plot_training_curves(hist, str(plots_dir / 'distillation_training_curves.png'))

        distilled_path = Path(args.output) / (Path(args.output).name + "_distilled.pt")
        distiller.save(str(distilled_path))

    # ---- PRUNING ----
    if args.prune:
        print("Running pruning...")
        spec = PruningSpec(
            amount=args.prune_amount,
            structured=args.prune_structured,
            norm=args.prune_norm,
            bias=args.prune_bias,
        )

        overall_sparsity, per_layer = prune_and_report(student, spec)
        print(f"[INFO] Overall sparsity after pruning: {overall_sparsity:.3f}")

        # optional light fine-tuning to recover accuracy
        val_acc = None
        if args.prune_ft_epochs > 0:
            print(f"[INFO] Fine-tuning pruned model for {args.prune_ft_epochs} epochs...")
            val_loss, val_acc = fine_tune_after_pruning(
                student,
                train_loader,
                val_loader,
                epochs=args.prune_ft_epochs,
                lr=args.prune_ft_lr,
            )
            print(f"[INFO] Post-prune val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        # save pruned model
        pruned_path = Path(args.output) / (Path(args.output).name + "_pruned.pt")
        torch.save(student.state_dict(), pruned_path)

        # log compression metrics
        results_dir = Path(args.output).parent / 'compression'
        results_dir.mkdir(parents=True, exist_ok=True)
        size_mb = pruned_path.stat().st_size / (1024*1024)
        csv_path = results_dir / 'compression_results.csv'
        header = 'name,model_path,model_size_mb,latency_ms,accuracy\n'
        row = f"{Path(args.output).name}_pruned,{pruned_path.as_posix()},{size_mb:.3f},0,{val_acc if val_acc is not None else ''}\n"
        if not csv_path.exists():
            csv_path.write_text(header + row)
        else:
            csv_path.write_text(csv_path.read_text() + row)

    # ---- QUANTIZATION ----
    if args.quantize:
        print("Running quantization...")
        # If we have a distilled checkpoint from earlier in the pipeline, prefer
        # to export a float ONNX and quantize that using the export helper. This
        # avoids calling ExportDB on already-quantized PyTorch objects which can
        # fail for packed params. We'll call the repository's `export_onnx.py`.
        exported_onnx = None
        try:
            if 'distilled_path' in locals() and Path(distilled_path).exists():
                print(f"[INFO] Exporting float ONNX from distilled checkpoint: {distilled_path}")
                # call export script; it writes to results/models/... by default
                # use same batch size as this compress run to keep shapes consistent
                cmd = [sys.executable, str(Path(__file__).parent / 'export_onnx.py'), '--batch_size', str(args.batch_size), '--ckpt', str(distilled_path)]
                subprocess.run(cmd, check=False)
                # Common export locations used by export_onnx.py
                candidate_quant = Path('results/models/models_quantized.onnx')
                candidate_float = Path('results/models/model_float.onnx')
                if candidate_quant.exists():
                    exported_onnx = str(candidate_quant)
                elif candidate_float.exists():
                    exported_onnx = str(candidate_float)
                if exported_onnx:
                    print(f"[INFO] Using exported ONNX model: {exported_onnx}")
        except Exception as e:
            print(f"[WARN] Export helper failed: {e}")
        quant = QuantizationTrainer(
            model=student,
            backend=args.backend
        )
        qmodel = quant.quantize()
        # evaluate quantized model accuracy on test set
        _, _, test_dl = get_dataloaders(args.data, batch_size=32)
        from sklearn.metrics import accuracy_score

        def eval_model(m):
            m.eval()
            ys, yps = [], []
            with torch.no_grad():
                for xb, yb in test_dl:
                    logits = m(xb)
                    preds = logits.argmax(dim=1).cpu().numpy()
                    yps.extend(preds.tolist())
                    ys.extend(yb.cpu().numpy().tolist())
            return accuracy_score(ys, yps)

        # if dynamic quant returned a new model object
        model_to_eval = qmodel if qmodel is not None else student
        acc = eval_model(model_to_eval)

        # ensure results dir exists (in case distillation step was skipped)
        results_dir = Path(args.output).parent / 'compression'
        results_dir.mkdir(parents=True, exist_ok=True)

        # estimate model size by saving temp state_dict (for CSV baseline)
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
        try:
            torch.save(model_to_eval.state_dict(), tmpf.name)
            size_mb = Path(tmpf.name).stat().st_size / (1024*1024)
        finally:
            tmpf.close()

        # If we got an ONNX from the export helper, prefer that as the quantized artifact.
        if exported_onnx:
            saved_path = exported_onnx
            print(f"[INFO] Skipping QuantizationTrainer.save because ONNX exported: {saved_path}")
        else:
            # save quantized model (the QuantizationTrainer.save now returns the saved path)
            out_name = Path(args.output).name + '_quantized.onnx'
            target_path = Path(args.output) / out_name
            saved_path = quant.save(str(target_path))
            if saved_path is None:
                saved_path = str(target_path)

        # Validate ONNX; if corrupt, fall back to float ONNX if available, else to .pt
        if str(saved_path).lower().endswith('.onnx') and not _is_valid_onnx(str(saved_path)):
            float_candidate = Path('results/models/model_float.onnx')
            if float_candidate.exists() and _is_valid_onnx(float_candidate):
                print(f"[INFO] Falling back to float ONNX: {float_candidate}")
                saved_path = str(float_candidate)
            else:
                fallback_pt = Path(args.output) / (Path(args.output).name + '_quantized_fallback.pt')
                torch.save(model_to_eval.state_dict(), fallback_pt)
                saved_path = str(fallback_pt)
                print(f"[INFO] Saved fallback PyTorch checkpoint: {saved_path}")

        # if the saver wrote a fallback .pt, compute size from that file instead
        try:
            size_mb = Path(saved_path).stat().st_size / (1024*1024)
        except Exception:
            # keep previous estimate
            pass

        # append to compression results CSV
        csv_path = results_dir / 'compression_results.csv'
        header = 'name,model_path,model_size_mb,latency_ms,accuracy\n'
        row = f"{Path(args.output).name}_quantized,{Path(saved_path).as_posix()},{size_mb:.3f},0,{acc:.4f}\n"
        if not csv_path.exists():
            csv_path.write_text(header + row)
        else:
            csv_path.write_text(csv_path.read_text() + row)

        print(f"[INFO] Quantized model saved to: {saved_path}")

    print("Compression pipeline complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--distill", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--prune_amount", type=float, default=0.3, help="Fraction of weights to prune")
    parser.add_argument("--prune_structured", action="store_true", help="Use structured LN pruning instead of unstructured")
    parser.add_argument("--prune_norm", type=int, default=2, help="Norm for structured pruning (LN)")
    parser.add_argument("--prune_bias", action="store_true", help="Also prune bias terms")
    parser.add_argument("--prune_ft_epochs", type=int, default=20, help="Fine-tune epochs after pruning (0 to skip)")
    parser.add_argument("--prune_ft_lr", type=float, default=1e-4, help="Fine-tune learning rate after pruning")
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--backend", default="qnnpack")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_classes", type=int, default=8)

    args = parser.parse_args()
    main(args)
