"""
fusion/train.py

Training loop with early stopping on validation loss.
Saves model checkpoints every epoch to models/checkpoints/.
Logs metrics to experiments/ as JSON.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class EarlyStopping:
    """Stop training when val_loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state: dict | None = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_residual: bool = False,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        if use_residual:
            images, residuals, labels = batch
            residuals = residuals.to(device)
        else:
            images, labels = batch
            residuals = None

        images = images.to(device)
        labels = labels.float().to(device).unsqueeze(1)  # [B, 1]

        optimiser.zero_grad()
        preds = model(images, residual=residuals)  # [B, 1]
        loss = criterion(preds, labels)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * images.size(0)
        predicted = (preds.detach() > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += images.size(0)

    return {
        "loss": total_loss / total,
        "acc": correct / total,
    }


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_residual: bool = False,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="Eval", leave=False):
        if use_residual:
            images, residuals, labels = batch
            residuals = residuals.to(device)
        else:
            images, labels = batch
            residuals = None

        images = images.to(device)
        labels = labels.float().to(device).unsqueeze(1)

        preds = model(images, residual=residuals)
        loss = criterion(preds, labels)

        total_loss += loss.item() * images.size(0)
        predicted = (preds > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += images.size(0)

        all_preds.extend(preds.cpu().squeeze().tolist())
        all_labels.extend(labels.cpu().squeeze().tolist())

    metrics = {
        "loss": total_loss / total,
        "acc": correct / total,
        "all_preds": all_preds,
        "all_labels": all_labels,
    }
    return metrics


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    experiment_name: str = "fusion",
) -> dict[str, Any]:
    """
    Full training loop.

    config keys:
        epochs:       int (default 50)
        lr:           float (default 1e-4)
        weight_decay: float (default 1e-5)
        patience:     int (default 10)
        device:       str (default 'cuda' if available)
        use_residual: bool (default False)
    """
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)

    epochs = config.get("epochs", 50)
    lr = config.get("lr", 1e-4)
    wd = config.get("weight_decay", 1e-5)
    patience = config.get("patience", 10)
    use_residual = config.get("use_residual", False)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.BCELoss()
    early_stop = EarlyStopping(patience=patience)

    # Checkpoint and experiment dirs
    ckpt_dir = PROJECT_ROOT / "models" / "checkpoints"
    exp_dir = PROJECT_ROOT / "experiments" / "ablation"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_acc = 0.0
    ts_start = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(model, train_loader, optimiser, criterion, device, use_residual)
        val_metrics = eval_epoch(model, val_loader, criterion, device, use_residual)

        elapsed = time.time() - t0
        record = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_acc": round(train_metrics["acc"], 4),
            "val_loss": round(val_metrics["loss"], 6),
            "val_acc": round(val_metrics["acc"], 4),
            "elapsed_s": round(elapsed, 2),
        }
        history.append(record)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train loss={record['train_loss']:.4f} acc={record['train_acc']:.4f} | "
            f"val loss={record['val_loss']:.4f} acc={record['val_acc']:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Save checkpoint every epoch
        ckpt_path = ckpt_dir / f"{experiment_name}_epoch{epoch:03d}_{ts_start}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "config": config,
            },
            ckpt_path,
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_path = ckpt_dir / f"{experiment_name}_best_{ts_start}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_metrics["acc"],
                    "config": config,
                },
                best_path,
            )

        if early_stop(val_metrics["loss"], model):
            print(f"[EarlyStopping] Triggered at epoch {epoch}.")
            early_stop.restore_best(model)
            break

    # Save training history
    hist_path = exp_dir / f"{experiment_name}_history_{ts_start}.json"
    with open(hist_path, "w") as f:
        # Remove large lists before saving
        clean_history = [
            {k: v for k, v in r.items() if k not in ("all_preds", "all_labels")}
            for r in history
        ]
        json.dump(
            {"experiment": experiment_name, "config": config, "history": clean_history},
            f,
            indent=2,
        )
    print(f"[Train] History saved to {hist_path}")

    return {
        "history": history,
        "best_val_acc": best_val_acc,
        "model": model,
    }


if __name__ == "__main__":
    """
    Two-phase CPU-tractable training pipeline:
      Phase 1 — Offline feature extraction on a subset of data/
                 Streams 2+3 take ~10 min; Stream 1 (PRNU) adds ~1-2 h on CPU.
      Phase 2 — Train FusionHead MLP on concatenated feature vectors (fast).

    Usage:
      python3 src/fusion/train.py                          # default: streams 23, 2000/class
      python3 src/fusion/train.py --streams 123            # all three streams
      python3 src/fusion/train.py --skip_extract           # reuse features/ from prior run
      python3 src/fusion/train.py --subset_size 5000       # larger subset
    """
    import argparse
    import json
    import random
    import shutil
    import sys

    import numpy as np

    _ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_ROOT))

    parser = argparse.ArgumentParser(description="Train AI image detector (extract → head)")
    parser.add_argument("--data_root", default="data/")
    parser.add_argument("--features_dir", default="features/")
    parser.add_argument(
        "--streams", default="23",
        help="Streams to extract/train: '23' fast on CPU, '123' adds PRNU (~1-2 h extra)",
    )
    parser.add_argument("--subset_size", type=int, default=2000,
                        help="Images per class for feature extraction")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--skip_extract", action="store_true",
                        help="Skip phase 1 and reuse existing features/")
    parser.add_argument("--full", action="store_true",
                        help="Full training: streams 123, subset_size=5000, pretrained DnCNN")
    parser.add_argument("--subset_dir", default="",
                        help="Custom subset directory (default: data/.train_subset)")
    parser.add_argument("--pretrained_backbone", action="store_true",
                        help="Use pretrained ImageNet ResNet-18 for Stream 2 FFT encoder (v5+)")
    parser.add_argument("--balanced", action="store_true",
                        help="Use WeightedRandomSampler to balance real/fake classes during training")
    parser.add_argument("--experiment", default="")
    parser.add_argument("--exp_name", default="",
                        help="Alias for --experiment (takes precedence if set)")
    args = parser.parse_args()
    if args.exp_name:
        args.experiment = args.exp_name

    if args.full:
        args.streams = "123"
        if args.subset_size == 2000:
            args.subset_size = 5000

    if not args.experiment:
        args.experiment = f"streams_{''.join(sorted(args.streams))}"

    data_root    = _ROOT / args.data_root
    features_dir = _ROOT / args.features_dir
    ckpt_dir     = _ROOT / "models" / "checkpoints"
    exp_dir      = _ROOT / "experiments" / "ablation"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Offline feature extraction ──────────────────────────────────
    if not args.skip_extract:
        from src.data.dataset import RealFakeImageFolder
        from src.data.preprocess import extract_features

        print(f"[Phase 1] Discovering images under {data_root} ...")
        full_ds = RealFakeImageFolder(data_root)
        real_paths = [s for s in full_ds.samples if s[1] == 0]
        fake_paths = [s for s in full_ds.samples if s[1] == 1]
        print(f"[Phase 1] Found {len(real_paths)} real, {len(fake_paths)} fake images total.")

        random.seed(42)
        n = args.subset_size
        real_sample = random.sample(real_paths, min(n, len(real_paths)))
        fake_sample = random.sample(fake_paths, min(n, len(fake_paths)))
        print(f"[Phase 1] Using subset: {len(real_sample)} real, {len(fake_sample)} fake.")

        # Build temp directory with symlinks (fallback to copy if symlinks unsupported)
        _subset_name = args.subset_dir if args.subset_dir else ".train_subset"
        subset_dir = _ROOT / "data" / _subset_name
        if subset_dir.exists():
            shutil.rmtree(subset_dir)
        (subset_dir / "real").mkdir(parents=True, exist_ok=True)
        (subset_dir / "fake").mkdir(parents=True, exist_ok=True)

        def _link_or_copy(src: Path, dst_dir: Path) -> None:
            dst = dst_dir / src.name
            if dst.exists():
                dst = dst_dir / f"{src.stem}_{src.parent.name}{src.suffix}"
            if dst.exists():
                # Still collides — use hash suffix
                dst = dst_dir / f"{src.stem}_{abs(hash(str(src))) % 100000}{src.suffix}"
            try:
                dst.symlink_to(src.resolve())
            except (OSError, NotImplementedError, FileExistsError):
                if not dst.exists():
                    shutil.copy2(src, dst)

        for p, _, _ in real_sample:
            _link_or_copy(p, subset_dir / "real")
        for p, _, _ in fake_sample:
            _link_or_copy(p, subset_dir / "fake")

        stream_eta = {"1": "~60-120 min", "2": "~5 min", "3": "~5 min"}
        for s in sorted(args.streams):
            if s in stream_eta:
                print(f"[Phase 1]   Stream {s}: {stream_eta[s]} on CPU for {len(real_sample)+len(fake_sample)} images")

        _dncnn_path = _ROOT / "models" / "pretrained" / "dncnn.pth"
        extract_features(
            data_root=subset_dir,
            out_dir=features_dir,
            streams=args.streams,
            img_size=args.img_size,
            batch_size=16,
            device="cpu",
            num_workers=0,
            dncnn_weights=str(_dncnn_path) if _dncnn_path.exists() else None,
            pretrained_backbone=args.pretrained_backbone,
        )
        shutil.rmtree(subset_dir)
        print("[Phase 1] Feature extraction complete.")

    # ── Phase 2: Load features and train FusionHead ───────────────────────────
    # Discover which streams have extracted features (infer dims from actual arrays)
    active_streams = sorted(
        s for s in "123"
        if s in args.streams and (features_dir / f"stream{s}" / "features.npy").exists()
    )

    print(f"\n[Phase 2] Loading features for streams {active_streams} ...")
    labels = np.load(str(features_dir / "labels.npy")).astype(np.float32)
    feat_parts = []
    for s in active_streams:
        arr = np.load(str(features_dir / f"stream{s}" / "features.npy")).astype(np.float32)
        feat_parts.append(arr)
        print(f"  stream{s}: {arr.shape}")
    X = np.concatenate(feat_parts, axis=1)   # [N, total_dim]
    total_dim = X.shape[1]
    print(f"[Phase 2] Concatenated: {X.shape}, labels: {labels.shape}")

    import torch
    from torch.utils.data import DataLoader, TensorDataset, random_split

    # ── Feature Z-normalisation ───────────────────────────────────────────────
    # Compute stats from training split only; save for inference-time normalisation.
    n_total = len(X)
    n_val   = max(1, int(0.15 * n_total))
    n_test  = max(1, int(0.15 * n_total))
    n_train = n_total - n_val - n_test

    # Determine train indices (same generator/seed as the split below)
    _gen_idx = torch.Generator().manual_seed(42)
    all_idx = torch.randperm(n_total, generator=_gen_idx).tolist()
    train_idx = all_idx[:n_train]

    feat_mean = X[train_idx].mean(axis=0).astype(np.float32)
    # Floor std at 1e-2 to avoid division-by-zero on near-constant features
    feat_std  = np.maximum(X[train_idx].std(axis=0), 1e-2).astype(np.float32)
    X = (X - feat_mean) / feat_std    # normalise entire matrix

    stats_path = features_dir / "feature_stats.npz"
    np.savez(str(stats_path), mean=feat_mean, std=feat_std)
    print(f"[Phase 2] Feature stats saved → {stats_path}  (mean|{feat_mean.mean():.4f} std|{feat_std.mean():.4f})")

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(labels).unsqueeze(1)
    ds = TensorDataset(X_t, y_t)

    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test], generator=gen)

    if args.balanced:
        from torch.utils.data import WeightedRandomSampler
        train_labels_arr = labels[train_ds.indices]
        n_real_tr = (train_labels_arr == 0).sum()
        n_fake_tr = (train_labels_arr == 1).sum()
        sample_weights = np.where(train_labels_arr == 0,
                                  1.0 / n_real_tr, 1.0 / n_fake_tr).astype(np.float32)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
        print(f"[Phase 2] Balanced sampler: {n_real_tr} real / {n_fake_tr} fake "
              f"(ratio {n_real_tr/n_fake_tr:.2f}x → equalised)")
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)
    print(f"[Phase 2] Split: {n_train} train / {n_val} val / {n_test} test")

    from src.fusion.network import FusionHead
    model = FusionHead(in_dim=total_dim).to("cpu")
    optimiser  = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion  = torch.nn.BCELoss()
    stopper    = EarlyStopping(patience=args.patience)
    history    = []
    best_val_acc = 0.0

    print(f"[Phase 2] Training FusionHead for up to {args.epochs} epochs ...")
    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        t_loss = t_correct = t_total = 0
        for feats, lbls in train_loader:
            optimiser.zero_grad()
            preds = model(feats)
            loss  = criterion(preds, lbls)
            loss.backward()
            optimiser.step()
            t_loss    += loss.item() * feats.size(0)
            t_correct += ((preds > 0.5) == lbls.bool()).sum().item()
            t_total   += feats.size(0)

        # --- val ---
        model.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for feats, lbls in val_loader:
                preds  = model(feats)
                v_loss += criterion(preds, lbls).item() * feats.size(0)
                v_correct += ((preds > 0.5) == lbls.bool()).sum().item()
                v_total   += feats.size(0)

        train_acc = t_correct / t_total
        val_acc   = v_correct / v_total
        val_loss  = v_loss / v_total

        record = {
            "epoch": epoch,
            "train_loss": round(t_loss / t_total, 6),
            "train_acc":  round(train_acc, 4),
            "val_loss":   round(val_loss, 6),
            "val_acc":    round(val_acc, 4),
        }
        history.append(record)
        print(
            f"  Epoch {epoch:3d}/{args.epochs} | "
            f"train loss={record['train_loss']:.4f} acc={train_acc:.3f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_dir / f"{args.experiment}_best.pth")

        if stopper(val_loss, model):
            print(f"  [EarlyStopping] No improvement for {args.patience} epochs — stopping.")
            stopper.restore_best(model)
            break

    print(f"\n[Phase 2] Best val acc: {best_val_acc:.4f}")

    # --- test ---
    model.eval()
    ts_correct = ts_total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for feats, lbls in test_loader:
            preds = model(feats)
            ts_correct += ((preds > 0.5) == lbls.bool()).sum().item()
            ts_total   += feats.size(0)
            all_preds.append(preds.squeeze(1))
            all_labels.append(lbls.squeeze(1))
    test_acc = ts_correct / ts_total

    from sklearn.metrics import roc_auc_score, average_precision_score
    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    try:
        auc = roc_auc_score(all_labels, all_preds)
        ap  = average_precision_score(all_labels, all_preds)
    except Exception:
        auc = ap = float("nan")

    print(f"[Phase 2] Test  acc={test_acc:.4f}  AUC={auc:.4f}  AP={ap:.4f}")

    # Save results
    results = {
        "experiment":    args.experiment,
        "streams":       args.streams,
        "subset_size":   args.subset_size,
        "best_val_acc":  best_val_acc,
        "test_acc":      test_acc,
        "test_auc":      auc,
        "test_ap":       ap,
        "history":       history,
    }
    hist_path = exp_dir / f"{args.experiment}_results.json"
    with open(hist_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[Phase 2] Results saved → {hist_path}")
