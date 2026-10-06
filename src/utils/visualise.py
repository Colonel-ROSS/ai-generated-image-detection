"""
utils/visualise.py

Visualisation utilities:
- Learnable spectral mask overlay
- Confusion matrix
- ROC curve
- Training history plot
- PRNU residual heatmap
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import roc_curve, confusion_matrix


def plot_spectral_mask(
    mask_np: np.ndarray,
    save_path: str | Path | None = None,
    title: str = "Learnable Spectral Mask",
) -> plt.Figure:
    """Plot the learnable FFT mask as a heatmap."""
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(mask_np, cmap="hot", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Frequency u")
    ax.set_ylabel("Frequency v")
    plt.colorbar(im, ax=ax, label="Mask weight")
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list[str] | None = None,
    save_path: str | Path | None = None,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Plot a normalised confusion matrix."""
    if class_names is None:
        class_names = ["Real", "Fake"]
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontsize=12,
            )

    plt.colorbar(im, ax=ax, label="Normalised count")
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    save_path: str | Path | None = None,
    label: str = "Model",
) -> plt.Figure:
    """Plot ROC curve."""
    from sklearn.metrics import roc_auc_score
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.4f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


def plot_training_history(
    history: list[dict],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot train/val loss and accuracy over epochs."""
    epochs = [r["epoch"] for r in history]
    train_loss = [r.get("train_loss", 0) for r in history]
    val_loss = [r.get("val_loss", 0) for r in history]
    train_acc = [r.get("train_acc", 0) for r in history]
    val_acc = [r.get("val_acc", 0) for r in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_loss, label="Val Loss", linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_acc, label="Train Acc", linewidth=2)
    ax2.plot(epochs, val_acc, label="Val Acc", linewidth=2, linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


def plot_prnu_residual(
    residual_np: np.ndarray,
    save_path: str | Path | None = None,
    title: str = "PRNU Noise Residual",
) -> plt.Figure:
    """Visualise a PRNU noise residual as a heatmap."""
    fig, ax = plt.subplots(figsize=(6, 6))
    # Scale to [0, 1] for display
    r_min, r_max = residual_np.min(), residual_np.max()
    if r_max > r_min:
        disp = (residual_np - r_min) / (r_max - r_min)
    else:
        disp = residual_np
    im = ax.imshow(disp, cmap="RdBu_r", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, label="Residual intensity")
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


def plot_ablation_bar(
    results: dict[str, dict],
    metric: str = "auc",
    save_path: str | Path | None = None,
    title: str = "Ablation Study",
) -> plt.Figure:
    """Bar chart comparing ablation variants."""
    names = list(results.keys())
    values = [results[n].get(metric, 0) for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, values, color="steelblue", edgecolor="black")
    ax.set_xlabel("Model Variant")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig
