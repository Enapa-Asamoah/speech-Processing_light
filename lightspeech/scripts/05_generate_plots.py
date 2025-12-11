"""
05_generate_plots.py
Generates all training, evaluation, and compression visualizations.

Run:
    python scripts/05_generate_plots.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Ensure package imports work when running script directly
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from lightspeech.code.evaluation.visualization import (
    plot_confusion_matrix,
    plot_training_curves,
    plot_compression_results
)

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
EVAL_DIR = os.path.join(RESULTS_DIR, "evaluation")
COMPRESS_DIR = os.path.join(RESULTS_DIR, "compression")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(COMPRESS_DIR, exist_ok=True)


# ---------------------------------------------------------
# 1. TRAINING CURVES (Loss + Accuracy)
# ---------------------------------------------------------

def generate_training_curves():
    history_path = os.path.join(LOGS_DIR, "baseline_training.json")

    if not os.path.exists(history_path):
        print("[WARN] No baseline_training.json found. Skipping training curves.")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    print("[INFO] Generating training curves...")
    save_path = os.path.join(PLOTS_DIR, "baseline_training_curves.png")
    plot_training_curves(history, save_path)


# ---------------------------------------------------------
# 2. CONFUSION MATRIX
# ---------------------------------------------------------

def generate_confusion_matrix():
    cm_path = os.path.join(EVAL_DIR, "confusion_matrix.npy")
    labels_path = os.path.join(EVAL_DIR, "label_mapping.json")

    if not os.path.exists(cm_path) or not os.path.exists(labels_path):
        print("[WARN] Confusion matrix or label mapping missing. Skipping CM plot.")
        return

    cm = np.load(cm_path)

    with open(labels_path, "r") as f:
        label_map = json.load(f)

    class_names = [label_map[str(i)] for i in range(len(label_map))]

    print("[INFO] Generating confusion matrix...")
    save_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, save_path=save_path)


# ---------------------------------------------------------
# 3. COMPRESSION TRADE-OFF PLOT
# ---------------------------------------------------------

def generate_compression_plots():
    comp_results_path = os.path.join(COMPRESS_DIR, "compression_results.csv")

    if not os.path.exists(comp_results_path):
        print("[WARN] No compression_results.csv found. Skipping compression plots.")
        return

    try:
        df = pd.read_csv(comp_results_path)
    except Exception as e:
        # Fallback: handle malformed CSVs by parsing manually with csv.reader
        import csv
        print(f"[WARN] Failed to read compression CSV with pandas: {e}. Falling back to manual parse.")
        with open(comp_results_path, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = [r for r in reader if any(cell.strip() for cell in r)]

        if not rows:
            print('[WARN] compression_results.csv empty after manual parse. Skipping.')
            return

        # normalize rows to same length
        max_cols = max(len(r) for r in rows)
        rows = [r + [''] * (max_cols - len(r)) for r in rows]

        header = rows[0]
        data_rows = rows[1:]

        # If header looks like data (no non-numeric tokens), synthesize likely header
        if all(h.strip().replace('.', '', 1).isdigit() for h in header if h.strip()):
            # common variants: 4 columns (name,size,latency,accuracy) or 5 with model_path
            if max_cols == 4:
                header = ['name', 'model_size_mb', 'latency_ms', 'accuracy']
            elif max_cols == 5:
                header = ['name', 'model_path', 'model_size_mb', 'latency_ms', 'accuracy']
            else:
                # fallback generic names
                header = [f'col{i}' for i in range(max_cols)]

        # build columns
        cols = {h: [] for h in header}
        for r in data_rows:
            for i, h in enumerate(header):
                cols[h].append(r[i].strip())

        # try to coerce numeric columns
        def to_float_list(key):
            out = []
            for v in cols.get(key, []):
                try:
                    out.append(float(v))
                except Exception:
                    out.append(float('nan'))
            return out

        model_sizes = to_float_list('model_size_mb')
        latencies = to_float_list('latency_ms')
        accuracies = to_float_list('accuracy')
        model_names = cols.get('name', cols.get('model', [f'model_{i}' for i in range(len(model_sizes))]))
    else:
        model_sizes = df["model_size_mb"].tolist()
        latencies = df["latency_ms"].tolist()
        accuracies = df["accuracy"].tolist()
        model_names = df.get("model", df.get("name", [f"model_{i}" for i in range(len(model_sizes))])).tolist()

    # Filter and ensure pruned model is included if present
    # Expected model order: teacher, student, pruned, quantized
    print(f"[INFO] Found {len(model_names)} models in compression results: {model_names}")
    
    print("[INFO] Generating compression trade-off plot...")
    save_path = os.path.join(PLOTS_DIR, "compression_tradeoff.png")

    plot_compression_results(model_sizes, latencies, accuracies, save_path)


# ---------------------------------------------------------
# 4. MAIN EXECUTION
# ---------------------------------------------------------

def main():
    print("\n=== Generating All Plots ===\n")
    generate_training_curves()
    generate_confusion_matrix()
    generate_compression_plots()
    print("\n[DONE] All plots generated in results/plots/\n")


if __name__ == "__main__":
    main()
