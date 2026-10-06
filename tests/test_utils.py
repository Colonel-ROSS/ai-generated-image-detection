"""
tests/test_utils.py

Unit tests for utility modules: metrics, visualise, checkpoint.
"""

import pytest
import json
import numpy as np
import torch
from pathlib import Path
import tempfile
import os

from src.utils.metrics import compute_metrics, compute_optimal_threshold, per_generator_metrics
from src.utils.visualise import (
    plot_spectral_mask,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_history,
    plot_prnu_residual,
)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_perfect_classification(self):
        labels = np.array([0, 0, 1, 1])
        probs = np.array([0.1, 0.2, 0.8, 0.9])
        m = compute_metrics(labels, probs)
        assert m["auc"] == 1.0
        assert m["acc"] == 1.0

    def test_all_wrong(self):
        labels = np.array([0, 0, 1, 1])
        probs = np.array([0.9, 0.8, 0.1, 0.2])
        m = compute_metrics(labels, probs)
        assert m["acc"] == 0.0

    def test_metric_keys(self):
        labels = np.array([0, 1, 0, 1])
        probs = np.array([0.3, 0.7, 0.4, 0.6])
        m = compute_metrics(labels, probs)
        for key in ["auc", "acc", "ap", "precision", "recall", "f1", "tp", "fp", "tn", "fn"]:
            assert key in m

    def test_single_class_labels(self):
        labels = np.array([1, 1, 1, 1])
        probs = np.array([0.9, 0.8, 0.7, 0.6])
        m = compute_metrics(labels, probs)
        assert m["auc"] == 0.0  # undefined, returns 0

    def test_optimal_threshold(self):
        labels = np.array([0, 0, 1, 1])
        probs = np.array([0.1, 0.2, 0.8, 0.9])
        thresh, m = compute_optimal_threshold(labels, probs)
        assert 0 <= thresh <= 1
        assert m["acc"] > 0

    def test_per_generator(self):
        labels = np.array([0, 1, 0, 1])
        probs = np.array([0.2, 0.8, 0.3, 0.7])
        gens = ["real", "stylegan", "real", "stylegan"]
        results = per_generator_metrics(labels, probs, gens)
        assert "real" in results
        assert "stylegan" in results


# ──────────────────────────────────────────────────────────────────────────────
# Visualise
# ──────────────────────────────────────────────────────────────────────────────

class TestVisualisations:
    def test_plot_spectral_mask(self):
        mask_np = np.random.rand(32, 32)
        fig = plot_spectral_mask(mask_np)
        assert fig is not None

    def test_plot_confusion_matrix(self):
        labels = np.array([0, 1, 0, 1])
        preds = np.array([0, 1, 1, 1])
        fig = plot_confusion_matrix(labels, preds)
        assert fig is not None

    def test_plot_roc_curve(self):
        labels = np.array([0, 0, 1, 1])
        probs = np.array([0.1, 0.4, 0.6, 0.9])
        fig = plot_roc_curve(labels, probs)
        assert fig is not None

    def test_plot_training_history(self):
        history = [
            {"epoch": 1, "train_loss": 0.5, "val_loss": 0.6, "train_acc": 0.7, "val_acc": 0.65},
            {"epoch": 2, "train_loss": 0.4, "val_loss": 0.5, "train_acc": 0.8, "val_acc": 0.75},
        ]
        fig = plot_training_history(history)
        assert fig is not None

    def test_plot_prnu_residual(self):
        residual = np.random.randn(64, 64)
        fig = plot_prnu_residual(residual)
        assert fig is not None

    def test_save_figure(self):
        mask_np = np.random.rand(16, 16)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "mask.png")
            plot_spectral_mask(mask_np, save_path=save_path)
            assert os.path.exists(save_path)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint utilities
# ──────────────────────────────────────────────────────────────────────────────

class TestCheckpoint:
    def test_load_save_json(self):
        from src.utils.checkpoint import save_checkpoint_json, load_checkpoint_json
        with tempfile.TemporaryDirectory() as tmpdir:
            import src.utils.checkpoint as ckpt_mod
            orig_path = ckpt_mod.CHECKPOINT_JSON
            orig_logs = ckpt_mod.LOGS_DIR

            ckpt_mod.CHECKPOINT_JSON = Path(tmpdir) / "CHECKPOINT.json"
            ckpt_mod.LOGS_DIR = Path(tmpdir) / "logs"

            state = {"current_task": "T4", "status": "IN_PROGRESS"}
            save_checkpoint_json(state)
            loaded = load_checkpoint_json()
            assert loaded["current_task"] == "T4"

            ckpt_mod.CHECKPOINT_JSON = orig_path
            ckpt_mod.LOGS_DIR = orig_logs
