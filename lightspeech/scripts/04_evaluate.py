"""
Script 04: Evaluate models
"""

import os
import sys
# Temporary workaround for OpenMP runtime conflicts (unsafe): allow duplicate OpenMP libs.
# This avoids the "Initializing libomp.dll, but found libiomp5md.dll already initialized"
# error that aborts execution. Prefer fixing the environment long-term.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse

# ensure project root is importable when running the script directly
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from lightspeech.code.evaluation.evaluator import evaluate_model
from lightspeech.code.models.compression import load_teacher_model
from lightspeech.code.data.loader import get_dataloaders
from lightspeech.code.evaluation.visualization import plot_confusion_matrix
from lightspeech.code.data.loader import EMOTIONS
import json
import numpy as np
from pathlib import Path
import time
import torch

def main(args):
    print("=== STEP 4: Model evaluation ===")

    _, _, test_loader = get_dataloaders(
        args.data,
        batch_size=args.batch_size
    )
    out_dir = Path(args.output or 'results')
    plots_dir = out_dir / 'plots'
    eval_dir = out_dir / 'evaluation'
    comp_dir = out_dir / 'compression'
    logs_dir = out_dir / 'logs'
    for d in (plots_dir, eval_dir, comp_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Prepare class names mapping
    class_names = [EMOTIONS[k] for k in sorted(EMOTIONS.keys())]

    results_summary = {}

    def _save_confusion(cm, name):
        np.save(eval_dir / f"{name}_confusion.npy", cm)
        plot_confusion_matrix(cm, class_names, title=f"Confusion: {name}", save_path=str(plots_dir / f"{name}_confusion.png"))

    # normalize device argument to torch.device
    try:
        device = args.device if isinstance(args.device, torch.device) else torch.device(args.device)
    except Exception:
        device = torch.device('cpu')

    # 1) Teacher / baseline
    if args.teacher_ckpt:
        print('[INFO] Evaluating teacher model...')
        teacher = load_teacher_model(ckpt_path=args.teacher_ckpt, device=device)
        metrics = evaluate_model(teacher, test_loader, device=device)
        _save_confusion(metrics['confusion_matrix'], 'teacher')
        # save metrics
        with open(logs_dir / 'teacher_metrics.json', 'w') as f:
            json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metrics.items()}, f)
        # size & latency
        try:
            size_mb = float(np.round(np.float64(__import__('lightspeech.code.evaluation.evaluator', fromlist=['']).model_size_mb(teacher)), 4))
        except Exception:
            size_mb = None
        try:
            latency_ms = float(np.round(__import__('lightspeech.code.evaluation.evaluator', fromlist=['']).model_latency(teacher, device=device), 4))
        except Exception:
            latency_ms = None
        results_summary['teacher'] = {'accuracy': metrics['accuracy'], 'size_mb': size_mb, 'latency_ms': latency_ms}

    # 2) Distilled student
    if args.student_ckpt:
        print('[INFO] Evaluating distilled student model...')
        # instantiate student architecture and load weights
        from lightspeech.code.models.student import StudentModel
        student = StudentModel(num_classes=args.num_classes)
        try:
            state = __import__('torch').load(args.student_ckpt, map_location=device)
            if isinstance(state, dict) and 'model_state' in state:
                state = state['model_state']
            student.load_state_dict(state)
        except Exception as e:
            print(f"[WARN] Could not load student checkpoint: {e}")
        metrics = evaluate_model(student, test_loader, device=device)
        _save_confusion(metrics['confusion_matrix'], 'student')
        with open(logs_dir / 'student_metrics.json', 'w') as f:
            json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metrics.items()}, f)
        # size & latency
        try:
            size_mb = float(np.round(__import__('lightspeech.code.evaluation.evaluator', fromlist=['']).model_size_mb(student), 4))
        except Exception:
            size_mb = None
        try:
            latency_ms = float(np.round(__import__('lightspeech.code.evaluation.evaluator', fromlist=['']).model_latency(student, device=device), 4))
        except Exception:
            latency_ms = None
        results_summary['student'] = {'accuracy': metrics['accuracy'], 'size_mb': size_mb, 'latency_ms': latency_ms}

    # 3) Quantized model (if provided)
    if args.quantized:
        print('[INFO] Evaluating quantized model...')
        qpath = Path(args.quantized)
        # If it's a torch state dict, load into StudentModel; otherwise, we measure file size.
        q_accuracy = None
        q_size = None
        q_latency = None
        try:
            from lightspeech.code.models.student import StudentModel
            qmodel = StudentModel(num_classes=args.num_classes)
            st = None
            try:
                st = __import__('torch').load(str(qpath), map_location=device)
                if isinstance(st, dict) and 'model_state' in st:
                    st = st['model_state']
                qmodel.load_state_dict(st)
                metrics = evaluate_model(qmodel, test_loader, device=device)
                _save_confusion(metrics['confusion_matrix'], 'quantized')
                q_accuracy = metrics['accuracy']
                try:
                    q_size = float(np.round(__import__('lightspeech.code.evaluation.evaluator', fromlist=['']).model_size_mb(qmodel), 4))
                except Exception:
                    q_size = qpath.stat().st_size / 1e6
                try:
                    q_latency = float(np.round(__import__('lightspeech.code.evaluation.evaluator', fromlist=['']).model_latency(qmodel, device=device), 4))
                except Exception:
                    q_latency = None
            except Exception:
                # not a torch model - use file size as size proxy
                q_size = qpath.stat().st_size / 1e6
                # If it's an ONNX model, try to evaluate with onnxruntime
                if qpath.suffix.lower() == '.onnx':
                    try:
                        import onnxruntime as ort
                        sess = ort.InferenceSession(str(qpath), providers=['CPUExecutionProvider'])
                        input_meta = sess.get_inputs()[0]
                        input_name = input_meta.name
                        # normalize expected shape: convert ints where possible, else None
                        exp_shape = []
                        for d in input_meta.shape:
                            try:
                                exp_shape.append(int(d))
                            except Exception:
                                exp_shape.append(None)

                        preds_all = []
                        labels_all = []

                        import torch.nn.functional as F

                        def _run_batch_inference(x_tensor):
                            # x_tensor: torch.Tensor (B,C,H,W)
                            B, C, H, W = x_tensor.shape
                            target_c = exp_shape[1] if len(exp_shape) > 1 else C
                            target_h = exp_shape[2] if len(exp_shape) > 2 else H
                            target_w = exp_shape[3] if len(exp_shape) > 3 else W
                            xt = x_tensor
                            if (target_h is not None and target_w is not None) and (H != target_h or W != target_w):
                                xt = F.interpolate(xt, size=(target_h, target_w), mode='bilinear', align_corners=False)
                            # if model expects batch size 1, run per-sample
                            if exp_shape[0] == 1 and B > 1:
                                preds = []
                                for i in range(B):
                                    xi = xt[i:i+1].cpu().numpy().astype('float32')
                                    out = sess.run(None, {input_name: xi})[0]
                                    preds.append(int(np.argmax(out, axis=1)[0]))
                                return preds
                            else:
                                x_np = xt.cpu().numpy().astype('float32')
                                out = sess.run(None, {input_name: x_np})[0]
                                return [int(p) for p in np.argmax(out, axis=1)]

                        # collect preds/labels
                        for x, y in test_loader:
                            preds = _run_batch_inference(x)
                            preds_all.extend(preds)
                            labels_all.extend(y.cpu().numpy().tolist())

                        if len(labels_all) > 0:
                            from sklearn.metrics import accuracy_score
                            q_accuracy = float(accuracy_score(labels_all, preds_all))

                        # measure ONNX latency (avg ms per inference unit)
                        try:
                            # warmup: run a few batches/samples
                            for _ in range(3):
                                for x, _ in test_loader:
                                    _run_batch_inference(x)
                                    break
                            # timed runs
                            times = []
                            n_runs = 20
                            for _ in range(n_runs):
                                start = time.time()
                                for x, _ in test_loader:
                                    _run_batch_inference(x)
                                    break
                                times.append(time.time() - start)
                            q_latency = float(np.round((np.mean(times) * 1000), 4))
                        except Exception as e:
                            print(f"[WARN] ONNX latency measurement failed: {e}")
                    except Exception as e:
                        print(f"[WARN] ONNX evaluation failed: {e}")
        except Exception as e:
            print(f"[WARN] Could not evaluate quantized model directly: {e}")

        results_summary['quantized'] = {'accuracy': q_accuracy, 'size_mb': q_size, 'latency_ms': q_latency}

    # Write compression summary CSV
    csv_path = comp_dir / 'compression_results.csv'
    if not csv_path.exists():
        csv_path.write_text('model,model_size_mb,latency_ms,accuracy\n')
    for name, info in results_summary.items():
        row = f"{name},{info.get('size_mb', '')},{info.get('latency_ms', '')},{info.get('accuracy', '')}\n"
        csv_path.write_text(csv_path.read_text() + row)

    # Generate bar plots (three models) for size, latency and accuracy
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Ensure consistent order and only include available models
        ordered = ['teacher', 'student', 'quantized']
        model_names = []
        sizes = []
        latencies = []
        accs = []
        for name in ordered:
            if name in results_summary:
                info = results_summary[name]
                model_names.append(name)
                sizes.append(info.get('size_mb') if info.get('size_mb') is not None else np.nan)
                latencies.append(info.get('latency_ms') if info.get('latency_ms') is not None else np.nan)
                a = info.get('accuracy')
                if a is None:
                    accs.append(np.nan)
                else:
                    accs.append(float(a) * 100.0 if a <= 1.0 else float(a))

        if len(model_names) == 0:
            raise RuntimeError("No models to plot")

        x = np.arange(len(model_names))
        width = 0.6

        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

        # Utility to plot bars and annotate n/a where appropriate
        def _bar_plot(ax, vals, title, ylabel, fmt="{:.2f}"):
            vals_arr = np.array(vals, dtype=float)
            nan_mask = np.isnan(vals_arr)
            filled = np.where(nan_mask, 0.0, vals_arr)  # plot zeros for missing, annotate as "n/a"
            bars = ax.bar(x, filled, width, color="#4C72B0")
            ax.set_xticks(x)
            ax.set_xticklabels(model_names)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            # annotate values or "n/a"
            for i, (b, v, is_nan) in enumerate(zip(bars, vals_arr, nan_mask)):
                h = b.get_height()
                if is_nan:
                    ax.text(b.get_x() + b.get_width() / 2, h + (0.02 * (max(1.0, np.nanmax(np.where(nan_mask, 0, vals_arr))) or 1.0)),
                            "n/a", ha='center', va='bottom', color='red', fontsize=9)
                else:
                    ax.text(b.get_x() + b.get_width() / 2, h + (0.01 * max(1.0, np.abs(v))),
                            fmt.format(v), ha='center', va='bottom', fontsize=9)

        _bar_plot(axes[0], sizes, "Model size (MB)", "Size (MB)")
        _bar_plot(axes[1], latencies, "Latency (ms)", "Latency (ms)")
        _bar_plot(axes[2], accs, "Accuracy (%)", "Accuracy (%)", fmt="{:.1f}%")

        plot_path = plots_dir / 'compression_tradeoff_bars.png'
        fig.suptitle("Compression trade-offs by model", fontsize=14)
        fig.savefig(str(plot_path), dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] Could not create compression bar plots: {e}")

    print('\n=== Evaluation Results Summaries ===')
    for k, v in results_summary.items():
        print(k, v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to processed feature folder")
    parser.add_argument("--teacher_ckpt", required=False, help="Path to teacher/baseline checkpoint (.pth)")
    parser.add_argument("--student_ckpt", required=False, help="Path to distilled student checkpoint (.pth)")
    parser.add_argument("--quantized", required=False, help="Path to quantized model file (torch state_dict or other)")
    parser.add_argument("--output", required=False, default="results", help="Base output directory for plots/logs/compression")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_classes", type=int, default=8, help="Number of target classes")
    args = parser.parse_args()

    main(args)
