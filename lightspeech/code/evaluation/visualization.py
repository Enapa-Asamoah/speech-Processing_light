import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_training_curves(history_dict, save_path=None):
    """
    history_dict = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }
    """

    # Accept either a dict of lists or a list of per-epoch dicts
    if isinstance(history_dict, list):
        # convert list of epoch-records to dict-of-lists
        epochs = history_dict
        train_loss = [e.get('loss') or e.get('train_loss') for e in epochs]
        val_loss = [e.get('val_loss') or e.get('val_loss') for e in epochs]
        train_acc = [e.get('train_acc') or e.get('acc') or None for e in epochs]
        val_acc = [e.get('val_acc') or e.get('val_accuracy') or None for e in epochs]
        history = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        }
    else:
        history = history_dict

    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.get("train_loss", []), label="Train Loss")
    plt.plot(history.get("val_loss", []), label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.get("train_acc", []), label="Train Acc")
    plt.plot(history.get("val_acc", []), label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_compression_results(model_sizes, latencies, accuracies, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.scatter(model_sizes, accuracies, s=100)

    for i, size in enumerate(model_sizes):
        plt.annotate(f"{latencies[i]}ms", (model_sizes[i], accuracies[i]))

    plt.xlabel("Model Size (MB)")
    plt.ylabel("Accuracy (%)")
    plt.title("Compression Trade-off: Size vs Accuracy")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
