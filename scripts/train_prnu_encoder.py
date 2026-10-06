#!/usr/bin/env python3
"""
scripts/train_prnu_encoder.py

Phase 3: Train PRNUEncoder end-to-end with binary supervision on noise residuals.

- Real images: data/real/coco/train2017/  (label=0)
- Fake images: data/fake/stable_diffusion/images/ + data/fake/stylegan3/images/ (label=1)
- DnCNN: extract noise residual at FULL RESOLUTION, then resize residual to RESIDUAL_SIZE
- PRNUEncoder: Conv(1->32)->Conv(32->64)->GlobalAvgPool->FC(64->128) -> BCE
- 20 epochs, Adam lr=1e-4, batch=32, balanced classes
- Saves best model to models/pretrained/prnu_encoder_trained.pth

Usage:
    python3 scripts/train_prnu_encoder.py [--n_real 5000] [--n_fake 5000] [--epochs 20]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stream1_prnu.denoiser import DnCNNWrapper
from src.stream1_prnu.encoder import PRNUEncoder

IMG_DNCNN_SIZE = 128  # resize grayscale image to this BEFORE DnCNN (trained encoder; not NCC approach)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

log_lines: list[str] = []


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_lines.append(line)


class ResidualDataset(Dataset):
    """Resizes image to img_size, extracts DnCNN noise residual, returns [1,S,S] + label."""

    def __init__(self, paths: list[tuple[str, int]], denoiser: DnCNNWrapper,
                 img_size: int = IMG_DNCNN_SIZE):
        self.paths = paths
        self.denoiser = denoiser
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path, label = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            t = TF.to_tensor(img)
            gray = TF.rgb_to_grayscale(t)                                         # [1, H, W]
            # Resize grayscale to fixed size before DnCNN (makes batching tractable on CPU)
            gray_rs = TF.resize(gray, [self.img_size, self.img_size],
                                 interpolation=TF.InterpolationMode.BILINEAR)     # [1, S, S]
            residual = self.denoiser.noise_residual(gray_rs)                      # [1, S, S]
            return residual.float(), torch.tensor(label, dtype=torch.float32)
        except Exception:
            return torch.zeros(1, self.img_size, self.img_size), \
                   torch.tensor(float(label))


def collect_paths(dirs: list[str], limit: int) -> list[str]:
    paths: list[str] = []
    for d in dirs:
        for p in Path(d).rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                paths.append(str(p))
    random.shuffle(paths)
    return paths[:limit]


def evaluate(model: PRNUEncoder, loader: DataLoader, criterion: nn.BCEWithLogitsLoss,
             device: torch.device) -> tuple[float, float, float]:
    model.eval()
    losses, preds, labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x).squeeze(1)
            losses.append(criterion(logits, y).item())
            preds.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())
    loss = float(np.mean(losses))
    try:
        auc = float(roc_auc_score(labels, preds))
    except Exception:
        auc = 0.5
    acc = float(np.mean((np.array(preds) >= 0.5) == np.array(labels)))
    return loss, auc, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_real",  type=int, default=5000)
    parser.add_argument("--n_fake",  type=int, default=5000)
    parser.add_argument("--epochs",  type=int, default=20)
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--lr",      type=float, default=1e-4)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--img_size",     type=int, default=IMG_DNCNN_SIZE,
                        help="Resize grayscale image to this before DnCNN (128 = fast, 224 = better)")
    parser.add_argument("--val_frac", type=float, default=0.15)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    log(f"n_real={args.n_real}  n_fake={args.n_fake}  epochs={args.epochs}  "
        f"batch={args.batch}  lr={args.lr}  img_size={args.img_size}")

    # --- Phase 2 check: verify DnCNN produces non-zero residuals ---
    denoiser = DnCNNWrapper(device=str(device))
    test_arr = np.random.randint(0, 255, (args.img_size, args.img_size, 3), dtype=np.uint8)
    test_pil = Image.fromarray(test_arr)
    t_test = TF.to_tensor(test_pil)
    g_test = TF.rgb_to_grayscale(t_test)
    r_test = denoiser.noise_residual(g_test)
    log(f"DnCNN diagnostic: residual std={r_test.std().item():.6f} "
        f"(>0.001 = OK, ≈0 = weights not loaded)")
    if r_test.std().item() < 1e-4:
        log("ERROR: DnCNN produces near-zero residuals — check models/pretrained/dncnn.pth")
        sys.exit(1)

    # --- Collect image paths ---
    real_dirs = [str(ROOT / "data" / "real" / "coco" / "train2017")]
    fake_dirs = [
        str(ROOT / "data" / "fake" / "stable_diffusion" / "images"),
        str(ROOT / "data" / "fake" / "stylegan3" / "images"),
    ]

    log("Collecting real image paths...")
    real_paths = collect_paths(real_dirs, args.n_real)
    log(f"  Real paths: {len(real_paths)}")

    log("Collecting fake image paths...")
    fake_paths = collect_paths(fake_dirs, args.n_fake)
    log(f"  Fake paths: {len(fake_paths)}")

    if len(real_paths) < 100 or len(fake_paths) < 100:
        log("ERROR: Too few images found. Check data directories.")
        sys.exit(1)

    # --- Train/val split ---
    all_items = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
    random.shuffle(all_items)
    n_val = int(len(all_items) * args.val_frac)
    val_items  = all_items[:n_val]
    train_items = all_items[n_val:]
    log(f"Split: {len(train_items)} train / {len(val_items)} val")

    train_ds = ResidualDataset(train_items, denoiser, args.img_size)
    val_ds   = ResidualDataset(val_items,   denoiser, args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=False)

    # --- Model ---
    model = PRNUEncoder(out_dim=128).to(device)
    # BCE with logits: remove final ReLU concern by attaching a linear head
    # We'll wrap: model -> linear(128,1) for binary output
    head = nn.Linear(128, 1).to(device)
    params = list(model.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    class FullModel(nn.Module):
        def __init__(self, enc, h):
            super().__init__()
            self.enc = enc
            self.head = h
        def forward(self, x):
            return self.head(self.enc(x))

    full_model = FullModel(model, head).to(device)

    best_auc   = 0.0
    best_epoch = 0
    results    = []
    out_path   = ROOT / "models" / "pretrained" / "prnu_encoder_trained.pth"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log(f"Starting training for {args.epochs} epochs...")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        full_model.train()
        epoch_loss = []
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = full_model(x).squeeze(1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

            if (batch_idx + 1) % 50 == 0:
                log(f"  Ep{epoch} batch {batch_idx+1}/{len(train_loader)} "
                    f"loss={np.mean(epoch_loss):.4f}")

        scheduler.step()
        train_loss = float(np.mean(epoch_loss))

        # Evaluate on validation set
        val_loss, val_auc, val_acc = evaluate(full_model, val_loader, criterion, device)
        elapsed = time.time() - t0
        log(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_AUC={val_auc:.4f}  val_ACC={val_acc:.4f}  "
            f"elapsed={elapsed:.0f}s")

        results.append({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4), "val_auc": round(val_auc, 4),
            "val_acc": round(val_acc, 4),
        })

        if val_auc > best_auc:
            best_auc   = val_auc
            best_epoch = epoch
            torch.save({
                "epoch":            epoch,
                "encoder_state":    model.state_dict(),
                "head_state":       head.state_dict(),
                "val_auc":          val_auc,
                "val_acc":          val_acc,
                "img_size":         args.img_size,
                "args":             vars(args),
            }, str(out_path))
            log(f"  *** New best AUC={val_auc:.4f} at epoch {epoch} — saved to {out_path}")

    total_time = time.time() - t0
    log(f"\nTraining complete in {total_time:.0f}s")
    log(f"Best val AUC: {best_auc:.4f} at epoch {best_epoch}")

    # Decision gate
    if best_auc > 0.55:
        log("PASS: AUC > 0.55 — proceed to Phase 4 (full feature extraction)")
        gate_status = "PASS"
    else:
        log(f"FAIL: AUC={best_auc:.4f} <= 0.55 — PRNU encoder not viable, stop here")
        gate_status = "FAIL"

    # Save results JSON
    results_path = ROOT / "experiments" / "prnu_encoder_training.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "timestamp":    datetime.now().isoformat(),
            "best_auc":     round(best_auc, 4),
            "best_epoch":   best_epoch,
            "gate_status":  gate_status,
            "total_time_s": round(total_time, 1),
            "args":         vars(args),
            "epoch_results": results,
        }, f, indent=2)
    log(f"Results saved → {results_path}")


if __name__ == "__main__":
    main()
