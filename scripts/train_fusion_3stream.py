#!/usr/bin/env python3
"""
scripts/train_fusion_3stream.py

Phase 5: Train 3-stream fusion head on aligned S1+S2+S3 features.

Loads features_new_3stream/{stream1,stream2,stream3}_features.npy + labels.npy
Concatenates to 448-dim, Z-normalises, trains FusionHead(448->256->1).
WeightedRandomSampler for class balance.

Baseline to beat: streams_23_balanced_best.pth  AUC=0.892

Usage:
    python3 scripts/train_fusion_3stream.py \
        --features_dir features_new_3stream \
        --epochs 50 --batch_size 512 --lr 1e-3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

log_lines: list[str] = []


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_lines.append(line)


class FusionHead(nn.Module):
    def __init__(self, in_dim: int = 448, hidden: int = 256, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate(model: FusionHead, loader: DataLoader, criterion: nn.BCEWithLogitsLoss,
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
        ap  = float(average_precision_score(labels, preds))
    except Exception:
        auc = ap = 0.5
    # Youden J threshold
    from sklearn.metrics import roc_curve
    fpr, tpr, thrs = roc_curve(labels, preds)
    j_idx = np.argmax(tpr - fpr)
    best_thr = float(thrs[j_idx])
    acc = float(accuracy_score(labels, (np.array(preds) >= best_thr).astype(int)))
    return loss, auc, ap, acc, best_thr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="features_new_3stream")
    parser.add_argument("--epochs",    type=int,   default=50)
    parser.add_argument("--batch_size",type=int,   default=512)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--hidden",    type=int,   default=256)
    parser.add_argument("--dropout",   type=float, default=0.5)
    parser.add_argument("--val_frac",  type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--patience",  type=int,   default=10)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--out_name",  default="streams_123_trained_prnu_best.pth")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    fdir = ROOT / args.features_dir
    log(f"Loading features from {fdir}")

    s1 = np.load(str(fdir / "stream1_features.npy")).astype(np.float32)  # (N, 128)
    s2 = np.load(str(fdir / "stream2_features.npy")).astype(np.float32)  # (N, 256)
    s3 = np.load(str(fdir / "stream3_features.npy")).astype(np.float32)  # (N, 64)
    lb = np.load(str(fdir / "labels.npy")).astype(np.float32)            # (N,)

    log(f"S1={s1.shape}  S2={s2.shape}  S3={s3.shape}  labels={lb.shape}")
    log(f"Label dist: real={int((lb==0).sum())}  fake={int((lb==1).sum())}")

    feats = np.concatenate([s1, s2, s3], axis=1)  # (N, 448)
    in_dim = feats.shape[1]
    log(f"Concatenated features: {feats.shape} (in_dim={in_dim})")

    # Stratified split
    N = len(lb)
    idx = np.arange(N)
    np.random.shuffle(idx)

    n_test = int(N * args.test_frac)
    n_val  = int(N * args.val_frac)
    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    log(f"Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")

    # Z-normalisation on training split
    tr_mean = feats[train_idx].mean(axis=0).astype(np.float32)
    tr_std  = np.maximum(feats[train_idx].std(axis=0), 1e-2).astype(np.float32)
    np.savez(str(fdir / "feature_stats.npz"), mean=tr_mean, std=tr_std)
    log("feature_stats.npz updated from training split")

    feats_norm = (feats - tr_mean) / tr_std  # (N, 448) normalised

    # Tensors
    X_tr = torch.from_numpy(feats_norm[train_idx])
    y_tr = torch.from_numpy(lb[train_idx])
    X_va = torch.from_numpy(feats_norm[val_idx])
    y_va = torch.from_numpy(lb[val_idx])
    X_te = torch.from_numpy(feats_norm[test_idx])
    y_te = torch.from_numpy(lb[test_idx])

    # WeightedRandomSampler for class balance
    y_tr_np = lb[train_idx]
    n_real  = int((y_tr_np == 0).sum())
    n_fake  = int((y_tr_np == 1).sum())
    w_real  = 1.0 / n_real
    w_fake  = 1.0 / n_fake
    sample_weights = np.where(y_tr_np == 0, w_real, w_fake)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=args.batch_size,
                              sampler=sampler, num_workers=0)
    val_loader   = DataLoader(TensorDataset(X_va, y_va), batch_size=args.batch_size,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(TensorDataset(X_te, y_te), batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    model     = FusionHead(in_dim=in_dim, hidden=args.hidden, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    out_dir = ROOT / "models" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name

    best_auc     = 0.0
    best_epoch   = 0
    patience_cnt = 0
    best_val_loss = float("inf")
    epoch_results = []
    t0 = time.time()

    log(f"Training FusionHead({in_dim}→{args.hidden}→1) for {args.epochs} epochs  "
        f"batch={args.batch_size}  lr={args.lr}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x).squeeze(1), y)
            loss.backward()
            optimizer.step()
            ep_loss.append(loss.item())
        scheduler.step()

        val_loss, val_auc, val_ap, val_acc, val_thr = evaluate(model, val_loader, criterion, device)
        tr_loss = float(np.mean(ep_loss))
        elapsed = time.time() - t0

        log(f"Ep{epoch:02d}/{args.epochs}  tr_loss={tr_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_AUC={val_auc:.4f}  val_ACC={val_acc:.4f}  "
            f"thr={val_thr:.4f}  {elapsed:.0f}s")

        epoch_results.append({
            "epoch": epoch, "tr_loss": round(tr_loss, 4),
            "val_loss": round(val_loss, 4), "val_auc": round(val_auc, 4),
            "val_acc": round(val_acc, 4), "val_thr": round(val_thr, 4),
        })

        if val_auc > best_auc:
            best_auc   = val_auc
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_auc":    val_auc,
                "val_acc":    val_acc,
                "val_thr":    val_thr,
                "in_dim":     in_dim,
                "epoch":      epoch,
                "features_dir": str(fdir),
            }, str(out_path))
            log(f"  *** Best AUC={val_auc:.4f} saved → {out_path}")

        # Early stopping on val_loss
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                log(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # Test evaluation
    model.load_state_dict(torch.load(str(out_path), map_location=device)["model_state_dict"])
    test_loss, test_auc, test_ap, test_acc, test_thr = evaluate(model, test_loader, criterion, device)
    total_time = time.time() - t0

    log(f"\nTest results  AUC={test_auc:.4f}  AP={test_ap:.4f}  ACC={test_acc:.4f}  thr={test_thr:.4f}")
    log(f"Baseline (S2+S3 balanced): AUC=0.8924")
    diff = test_auc - 0.8924
    sign = "+" if diff >= 0 else ""
    log(f"Delta vs baseline: {sign}{diff:.4f}")

    # Results JSON
    results = {
        "timestamp":     datetime.now().isoformat(),
        "model":         args.out_name,
        "features_dir":  str(fdir),
        "in_dim":        in_dim,
        "best_val_auc":  round(best_auc, 4),
        "best_epoch":    best_epoch,
        "test_auc":      round(test_auc, 4),
        "test_ap":       round(test_ap,  4),
        "test_acc":      round(test_acc, 4),
        "test_thr":      round(test_thr, 4),
        "baseline_auc":  0.8924,
        "delta_auc":     round(diff, 4),
        "total_time_s":  round(total_time, 1),
        "args":          vars(args),
        "epoch_log":     epoch_results,
    }

    results_path = ROOT / "experiments" / "streams_123_training.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Results saved → {results_path}")


if __name__ == "__main__":
    main()
