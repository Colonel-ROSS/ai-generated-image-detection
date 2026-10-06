"""
fusion/evaluate.py

Evaluation utilities: AUC, ACC, AP, per-generator breakdown.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device | str = "cpu",
    use_residual: bool = False,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Run inference and compute AUC, ACC, AP, confusion matrix.
    Returns a metrics dict.
    """
    model.eval()
    device = torch.device(device)
    model.to(device)

    all_probs = []
    all_labels = []

    for batch in tqdm(loader, desc="Evaluate"):
        if use_residual:
            images, residuals, labels = batch
            residuals = residuals.to(device)
        else:
            images, labels = batch
            residuals = None

        images = images.to(device)
        probs = model(images, residual=residuals)  # [B, 1]
        all_probs.extend(probs.cpu().squeeze(1).tolist())
        all_labels.extend(labels.tolist())

    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_labels)
    preds_arr = (probs_arr >= threshold).astype(int)

    auc = float(roc_auc_score(labels_arr, probs_arr)) if len(np.unique(labels_arr)) > 1 else 0.0
    acc = float(accuracy_score(labels_arr, preds_arr))
    ap = float(average_precision_score(labels_arr, probs_arr)) if len(np.unique(labels_arr)) > 1 else 0.0
    cm = confusion_matrix(labels_arr, preds_arr).tolist()

    return {
        "auc": round(auc, 4),
        "acc": round(acc, 4),
        "ap": round(ap, 4),
        "confusion_matrix": cm,
        "n_samples": len(labels_arr),
        "n_real": int((labels_arr == 0).sum()),
        "n_fake": int((labels_arr == 1).sum()),
        "threshold": threshold,
    }


@torch.no_grad()
def evaluate_per_generator(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device | str = "cpu",
    use_residual: bool = False,
    threshold: float = 0.5,
) -> dict[str, dict]:
    """
    Evaluate per generator source.
    Requires loader to yield (images, labels, generator_names) tuples.
    """
    model.eval()
    device = torch.device(device)
    model.to(device)

    generator_data: dict[str, dict] = {}

    for batch in tqdm(loader, desc="Eval per-generator"):
        if use_residual and len(batch) == 4:
            images, residuals, labels, generators = batch
            residuals = residuals.to(device)
        elif len(batch) == 3:
            images, labels, generators = batch
            residuals = None
        else:
            continue

        images = images.to(device)
        probs = model(images, residual=residuals).cpu().squeeze(1)

        for prob, label, gen in zip(probs.tolist(), labels.tolist(), generators):
            if gen not in generator_data:
                generator_data[gen] = {"probs": [], "labels": []}
            generator_data[gen]["probs"].append(prob)
            generator_data[gen]["labels"].append(label)

    results = {}
    for gen, data in generator_data.items():
        probs_arr = np.array(data["probs"])
        labels_arr = np.array(data["labels"])
        preds_arr = (probs_arr >= threshold).astype(int)
        auc = float(roc_auc_score(labels_arr, probs_arr)) if len(np.unique(labels_arr)) > 1 else 0.0
        acc = float(accuracy_score(labels_arr, preds_arr))
        ap = float(average_precision_score(labels_arr, probs_arr)) if len(np.unique(labels_arr)) > 1 else 0.0
        results[gen] = {"auc": round(auc, 4), "acc": round(acc, 4), "ap": round(ap, 4), "n": len(labels_arr)}

    return results


def save_eval_results(results: dict, name: str, subdir: str = "ablation"):
    """Save evaluation results JSON to experiments/."""
    from datetime import datetime
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "experiments" / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Evaluate] Results saved to {path}")
    return path


def robustness_sweep(
    model: torch.nn.Module,
    get_loader_fn,
    perturbations: dict[str, list],
    device: str = "cpu",
    experiment_name: str = "robustness",
) -> dict:
    """
    Run evaluation over a grid of perturbation parameters.
    get_loader_fn(perturbation_type, value) -> DataLoader
    perturbations: e.g. {"jpeg": [95, 75, 50, 30], "blur": [1, 2, 3, 4]}
    """
    results = {}
    for ptype, values in perturbations.items():
        results[ptype] = {}
        for val in values:
            loader = get_loader_fn(ptype, val)
            metrics = evaluate(model, loader, device=device)
            results[ptype][str(val)] = metrics
            print(f"[Robustness] {ptype}={val}: AUC={metrics['auc']:.4f} ACC={metrics['acc']:.4f}")

    save_eval_results(results, experiment_name, subdir="robustness")
    return results


if __name__ == "__main__":
    """
    CLI: evaluate a trained FusionHead checkpoint on pre-extracted features.

    Usage:
        # Single checkpoint evaluation:
        python3 src/fusion/evaluate.py --checkpoint models/checkpoints/streams_23_best.pth \\
                                        --streams 23 --features_dir features/

        # Full evaluation suite (finds best checkpoint automatically):
        python3 src/fusion/evaluate.py --full_eval
        python3 src/fusion/evaluate.py --full_eval --features_dir features/ --streams 23
    """
    import argparse
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split

    parser = argparse.ArgumentParser(description="Evaluate FusionHead on pre-extracted features")
    parser.add_argument("--checkpoint", default="", help="Path to FusionHead .pth checkpoint")
    parser.add_argument("--streams", default="", help="Which streams were used (e.g. '23', '123')")
    parser.add_argument("--features_dir", default="features/", help="Directory with stream*.npy files")
    parser.add_argument("--split", choices=["test", "all"], default="test")
    parser.add_argument("--name", default="", help="Name for saved results (default: checkpoint stem)")
    parser.add_argument("--full_eval", action="store_true",
                        help="Run full evaluation suite: all checkpoints + robustness sweep")
    args = parser.parse_args()

    # ── Full evaluation suite ──────────────────────────────────────────────────
    if args.full_eval:
        import io, random
        from PIL import Image, ImageFilter
        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

        ckpt_dir = PROJECT_ROOT / "models" / "checkpoints"
        features_dir = PROJECT_ROOT / args.features_dir

        # Auto-detect streams — support both sub-dir and flat layouts
        if not args.streams:
            available = sorted(s for s in "123"
                               if (features_dir / f"stream{s}" / "features.npy").exists()
                               or (features_dir / f"stream{s}_features.npy").exists())
            if not available:
                print("[Evaluate] ERROR: No feature files found. Run training first.")
                sys.exit(1)
            args.streams = "".join(available)
            print(f"[Evaluate] Auto-detected streams: {args.streams}")

        # Auto-detect checkpoint
        if not args.checkpoint:
            exp_name = f"streams_{''.join(sorted(args.streams))}"
            candidates = sorted(ckpt_dir.glob(f"{exp_name}*_best*.pth"))
            if not candidates:
                candidates = sorted(ckpt_dir.glob(f"{exp_name}*.pth"))
            if not candidates:
                print(f"[Evaluate] ERROR: No checkpoint found for {exp_name} in {ckpt_dir}")
                sys.exit(1)
            args.checkpoint = str(candidates[-1])
            print(f"[Evaluate] Auto-selected checkpoint: {args.checkpoint}")

        # Load features — support both stream{s}/features.npy and stream{s}_features.npy layouts
        labels_arr = np.load(str(features_dir / "labels.npy")).astype(np.float32)
        feat_parts = []
        for s in sorted(args.streams):
            p_sub  = features_dir / f"stream{s}" / "features.npy"
            p_flat = features_dir / f"stream{s}_features.npy"
            if p_sub.exists():
                arr = np.load(str(p_sub)).astype(np.float32)
            elif p_flat.exists():
                arr = np.load(str(p_flat)).astype(np.float32)
            else:
                print(f"[Evaluate] ERROR: no features for stream{s} in {features_dir}")
                sys.exit(1)
            feat_parts.append(arr)
            print(f"[Evaluate] stream{s}: {arr.shape}")
        X_full = np.concatenate(feat_parts, axis=1)
        in_dim = X_full.shape[1]
        print(f"[Evaluate] Feature matrix: {X_full.shape}")

        # Apply feature normalisation if stats exist (saved by train.py)
        feat_mean_arr = feat_std_arr = None
        stats_path = features_dir / "feature_stats.npz"
        if stats_path.exists():
            stats = np.load(str(stats_path))
            feat_mean_arr = stats["mean"].astype(np.float32)
            feat_std_arr  = np.maximum(stats["std"].astype(np.float32), 1e-2)
            X_full = ((X_full - feat_mean_arr) / feat_std_arr).astype(np.float32)
            print(f"[Evaluate] Applied feature normalisation from {stats_path}")

        # Build FusionHead and load weights
        class _FH(nn.Module):
            def __init__(self, d, h=256):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d, h), nn.ReLU(inplace=True),
                    nn.Dropout(0.5), nn.Linear(h, 1), nn.Sigmoid())
            def forward(self, x): return self.net(x)

        model = _FH(in_dim)
        state = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.eval()

        # ── Test-split evaluation ──────────────────────────────────────────────
        X_t = torch.from_numpy(X_full)
        y_t = torch.from_numpy(labels_arr).unsqueeze(1)
        ds  = TensorDataset(X_t, y_t)
        n   = len(ds)
        n_val  = max(1, int(0.15 * n))
        n_test = max(1, int(0.15 * n))
        n_tr   = n - n_val - n_test
        g = torch.Generator().manual_seed(42)
        _, _, test_ds = random_split(ds, [n_tr, n_val, n_test], generator=g)
        loader = DataLoader(test_ds, batch_size=256, shuffle=False)

        all_p, all_l = [], []
        with torch.no_grad():
            for feats, lbls in loader:
                all_p.append(model(feats).squeeze(1))
                all_l.append(lbls.squeeze(1))
        all_p = torch.cat(all_p).numpy()
        all_l = torch.cat(all_l).numpy()

        auc = float(roc_auc_score(all_l, all_p))
        ap  = float(average_precision_score(all_l, all_p))
        acc = float(accuracy_score(all_l.astype(int), (all_p >= 0.5).astype(int)))
        from sklearn.metrics import confusion_matrix as _cm
        cm  = _cm(all_l.astype(int), (all_p >= 0.5).astype(int)).tolist()

        print(f"\n[Evaluate] === Test split ({len(all_l)} samples) ===")
        print(f"  ACC={acc:.4f}  AUC={auc:.4f}  AP={ap:.4f}")
        print(f"  Confusion matrix: {cm}")

        test_results = {
            "checkpoint": args.checkpoint, "streams": args.streams,
            "n_samples": len(all_l), "acc": round(acc, 4),
            "auc": round(auc, 4), "ap": round(ap, 4), "confusion_matrix": cm,
        }

        # ── Robustness sweep ───────────────────────────────────────────────────
        from src.data.dataset import RealFakeImageFolder
        from src.stream2_fft.encoder import Stream2
        from src.stream3_nss.encoder import NSSEncoder
        from src.stream3_nss.statistics import extract_nss_features, NSS_DIM
        import torchvision.transforms as T

        data_root = PROJECT_ROOT / "data"
        try:
            full_ds = RealFakeImageFolder(data_root)
        except Exception as e:
            print(f"[Evaluate] Cannot load dataset for robustness: {e}")
            full_ds = None

        rob_results = {}
        if full_ds is not None and ("2" in args.streams or "1" in args.streams):
            print("\n[Evaluate] === Robustness sweep (using trained model) ===")

            denoiser_rob = prnu_enc_rob = None
            if "1" in args.streams:
                from src.stream1_prnu.denoiser import DnCNNWrapper
                from src.stream1_prnu.encoder import PRNUEncoder
                denoiser_rob = DnCNNWrapper()
                prnu_enc_rob = PRNUEncoder(out_dim=128).eval()
                prnu_w = PROJECT_ROOT / "models" / "pretrained" / "prnu_encoder_trained.pth"
                if prnu_w.exists():
                    _st = torch.load(str(prnu_w), map_location="cpu")
                    prnu_enc_rob.load_state_dict(_st["encoder_state"])
                    print(f"[Evaluate] Loaded PRNUEncoder from {prnu_w}")

            torch.manual_seed(0)
            s2_enc = Stream2(out_dim=256, spectral_size=224).eval()
            torch.manual_seed(0)
            nss_enc = NSSEncoder(in_dim=NSS_DIM, out_dim=64).eval()
            # Load saved encoder weights — try features_dir, fall back to features_combined
            _enc_bases = [features_dir, PROJECT_ROOT / "features_combined"]
            for _base in _enc_bases:
                s2_w = _base / "stream2" / "stream2_encoder.pth"
                if s2_w.exists():
                    s2_enc.load_state_dict(torch.load(str(s2_w), map_location="cpu"))
                    print(f"[Evaluate] Loaded Stream2 encoder from {s2_w}")
                    break
            for _base in _enc_bases:
                nss_w = _base / "stream3" / "nss_encoder.pth"
                if nss_w.exists():
                    nss_enc.load_state_dict(torch.load(str(nss_w), map_location="cpu"))
                    print(f"[Evaluate] Loaded NSSEncoder from {nss_w}")
                    break

            # Load normalisation stats for inference (if trained with normalisation)
            feat_mean_arr = feat_std_arr = None
            _rob_stats = features_dir / "feature_stats.npz"
            if _rob_stats.exists():
                _s = np.load(str(_rob_stats))
                feat_mean_arr = _s["mean"].astype(np.float32)
                feat_std_arr  = np.maximum(_s["std"].astype(np.float32), 1e-2)
                print(f"[Evaluate] Loaded normalisation stats from {_rob_stats}")

            norm = T.Compose([
                T.Resize((224, 224)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            real_s = [s for s in full_ds.samples if s[1] == 0]
            fake_s = [s for s in full_ds.samples if s[1] == 1]
            random.seed(123)
            n_rob = 200
            rob_samples = (random.sample(real_s, min(n_rob, len(real_s))) +
                           random.sample(fake_s, min(n_rob, len(fake_s))))
            rob_paths  = [Path(p) for p, _, _ in rob_samples]
            rob_labels = np.array([0]*min(n_rob, len(real_s)) + [1]*min(n_rob, len(fake_s)), dtype=np.float32)

            def _apply_pert(img, ptype, val):
                if ptype == "jpeg":
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=int(val)); buf.seek(0)
                    return Image.open(buf).copy()
                elif ptype == "blur":
                    return img.filter(ImageFilter.GaussianBlur(radius=float(val)))
                elif ptype == "resize":
                    w, h = img.size
                    img = img.resize((max(1,int(w*val)), max(1,int(h*val))), Image.BICUBIC)
                    return img.resize((w, h), Image.BICUBIC)
                return img

            def _extract_rob(paths, ptype="none", val=None):
                """Extract (and optionally normalise) features for robustness eval. Order: S1+S2+S3."""
                import torchvision.transforms.functional as TF
                parts = []
                if "1" in args.streams:
                    feats_s1 = []
                    for p in paths:
                        img = Image.open(p).convert("RGB")
                        if ptype != "none":
                            img = _apply_pert(img, ptype, val)
                        t_raw = TF.to_tensor(img)
                        g = TF.rgb_to_grayscale(t_raw)
                        g_s = TF.resize(g, [128, 128], interpolation=TF.InterpolationMode.BILINEAR)
                        with torch.no_grad():
                            res = denoiser_rob.noise_residual(g_s)
                            feats_s1.append(prnu_enc_rob(res.unsqueeze(0)).squeeze(0).numpy())
                    parts.append(np.stack(feats_s1))
                if "2" in args.streams:
                    feats_s2 = []
                    for p in paths:
                        img = Image.open(p).convert("RGB")
                        if ptype != "none":
                            img = _apply_pert(img, ptype, val)
                        t = norm(img).unsqueeze(0)
                        with torch.no_grad():
                            feats_s2.append(s2_enc(t).squeeze(0).numpy())
                    parts.append(np.stack(feats_s2))
                if "3" in args.streams:
                    import torchvision.transforms.functional as _TF
                    feats_s3 = []
                    for p in paths:
                        img = Image.open(p).convert("RGB")
                        if ptype != "none":
                            img = _apply_pert(img, ptype, val)
                        # Un-normalized [0,1] — matches extract_all_streams_80k.py
                        t_01 = _TF.to_tensor(img)
                        with torch.no_grad():
                            raw = extract_nss_features(t_01)
                            feats_s3.append(nss_enc(torch.from_numpy(raw).unsqueeze(0).float()).squeeze(0).numpy())
                    parts.append(np.stack(feats_s3))
                X_r = np.concatenate(parts, axis=1).astype(np.float32)
                if feat_mean_arr is not None:
                    X_r = ((X_r - feat_mean_arr) / feat_std_arr).astype(np.float32)
                return X_r

            # Evaluate TRAINED model on clean features as baseline
            print("  Extracting clean robustness features...")
            X_rob_clean = _extract_rob(rob_paths)
            with torch.no_grad():
                base_p = model(torch.from_numpy(X_rob_clean)).squeeze(1).numpy()
            base_auc = float(roc_auc_score(rob_labels, base_p))
            base_acc = float(accuracy_score(rob_labels.astype(int), (base_p>=0.5).astype(int)))
            print(f"  Baseline clean (trained model): acc={base_acc:.4f} AUC={base_auc:.4f}")
            rob_results["baseline_clean"] = {"acc": round(base_acc,4), "auc": round(base_auc,4)}

            # Sweep perturbations — evaluate TRAINED model on perturbed features
            for ptype, vals in [("jpeg",[95,75,50,30]),("blur",[1,2,3,4]),("resize",[0.5,0.75,1.5])]:
                rob_results[ptype] = {}
                for val in vals:
                    X_pert = torch.from_numpy(_extract_rob(rob_paths, ptype, val))
                    with torch.no_grad():
                        pp = model(X_pert).squeeze(1).numpy()
                    pauc = float(roc_auc_score(rob_labels, pp))
                    pacc = float(accuracy_score(rob_labels.astype(int), (pp>=0.5).astype(int)))
                    rob_results[ptype][str(val)] = {
                        "acc": round(pacc,4), "auc": round(pauc,4),
                        "acc_drop": round(base_acc-pacc,4), "auc_drop": round(base_auc-pauc,4),
                    }
                    print(f"  {ptype}={val}: acc={pacc:.4f} AUC={pauc:.4f} (Δ={base_auc-pauc:+.4f})")

        # ── Save all results ───────────────────────────────────────────────────
        from datetime import datetime
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        exp_name = f"streams_{''.join(sorted(args.streams))}"
        full_results = {
            "experiment": exp_name,
            "timestamp": ts,
            "test_evaluation": test_results,
            "robustness": rob_results,
        }
        out_dir = PROJECT_ROOT / "experiments" / "ablation"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{exp_name}_full_eval_{ts}.json"
        with open(out_path, "w") as f:
            json.dump(full_results, f, indent=2)
        print(f"\n[Evaluate] Full results saved → {out_path}")

        # Also save robustness separately
        if rob_results:
            rob_dir = PROJECT_ROOT / "experiments" / "robustness"
            rob_dir.mkdir(parents=True, exist_ok=True)
            rob_path = rob_dir / f"robustness_{exp_name}_{ts}.json"
            with open(rob_path, "w") as f:
                json.dump({"experiment": exp_name, **rob_results}, f, indent=2)
            print(f"[Evaluate] Robustness results saved → {rob_path}")

        sys.exit(0)

    # ── Single checkpoint evaluation (original mode) ──────────────────────────
    if not args.checkpoint:
        print("[Evaluate] ERROR: --checkpoint required (or use --full_eval)")
        sys.exit(1)
    if not args.streams:
        print("[Evaluate] ERROR: --streams required (or use --full_eval)")
        sys.exit(1)

    features_dir = PROJECT_ROOT / args.features_dir
    labels = np.load(str(features_dir / "labels.npy")).astype(np.float32)

    feat_parts = []
    for s in sorted(args.streams):
        p_sub  = features_dir / f"stream{s}" / "features.npy"
        p_flat = features_dir / f"stream{s}_features.npy"
        if p_sub.exists():
            arr_path = p_sub
        elif p_flat.exists():
            arr_path = p_flat
        else:
            print(f"[Evaluate] ERROR: no features for stream{s} in {features_dir}")
            sys.exit(1)
        arr = np.load(str(arr_path)).astype(np.float32)
        feat_parts.append(arr)
        print(f"[Evaluate] stream{s}: {arr.shape}")

    X = np.concatenate(feat_parts, axis=1)
    in_dim = X.shape[1]
    print(f"[Evaluate] Feature matrix: {X.shape}, in_dim={in_dim}")

    # Apply feature normalisation if stats exist
    _stats_path = features_dir / "feature_stats.npz"
    if _stats_path.exists():
        _stats = np.load(str(_stats_path))
        _std = np.maximum(_stats["std"].astype(np.float32), 1e-2)
        X = ((X - _stats["mean"]) / _std).astype(np.float32)
        print(f"[Evaluate] Applied feature normalisation from {_stats_path}")

    # Load FusionHead
    class _FusionHead(nn.Module):
        def __init__(self, in_dim, hidden=256, dropout=0.5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
                nn.Dropout(dropout), nn.Linear(hidden, 1), nn.Sigmoid(),
            )
        def forward(self, x):
            return self.net(x)

    model = _FusionHead(in_dim)
    ckpt_path = PROJECT_ROOT / args.checkpoint
    if not ckpt_path.exists():
        print(f"[Evaluate] ERROR: checkpoint {ckpt_path} not found")
        sys.exit(1)
    state = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    print(f"[Evaluate] Loaded checkpoint: {ckpt_path}")

    # Build dataset with same 70/15/15 split as training
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(labels).unsqueeze(1)
    ds = TensorDataset(X_t, y_t)
    n_total = len(ds)
    n_val  = max(1, int(0.15 * n_total))
    n_test = max(1, int(0.15 * n_total))
    n_train = n_total - n_val - n_test
    gen = torch.Generator().manual_seed(42)
    _, _, test_ds = random_split(ds, [n_train, n_val, n_test], generator=gen)

    if args.split == "test":
        eval_ds = test_ds
    else:
        eval_ds = ds

    loader = DataLoader(eval_ds, batch_size=128, shuffle=False)

    all_preds, all_labels_out = [], []
    with torch.no_grad():
        for feats, lbls in loader:
            preds = model(feats).squeeze(1)
            all_preds.append(preds)
            all_labels_out.append(lbls.squeeze(1))

    all_preds  = torch.cat(all_preds).numpy()
    all_labels_out = torch.cat(all_labels_out).numpy()
    pred_labels = (all_preds >= 0.5).astype(int)

    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix
    auc = float(roc_auc_score(all_labels_out, all_preds))
    ap  = float(average_precision_score(all_labels_out, all_preds))
    acc = float(accuracy_score(all_labels_out.astype(int), pred_labels))
    cm  = confusion_matrix(all_labels_out.astype(int), pred_labels).tolist()

    results = {
        "checkpoint":     str(ckpt_path),
        "streams":        args.streams,
        "split":          args.split,
        "n_samples":      int(len(all_labels_out)),
        "acc":            round(acc, 4),
        "auc":            round(auc, 4),
        "ap":             round(ap, 4),
        "confusion_matrix": cm,
    }

    print(f"\n[Evaluate] Results on {args.split} split ({len(all_labels_out)} samples):")
    print(f"  ACC = {acc:.4f}")
    print(f"  AUC = {auc:.4f}")
    print(f"  AP  = {ap:.4f}")
    print(f"  Confusion matrix: {cm}")

    name = args.name or ckpt_path.stem
    save_eval_results(results, name, subdir="ablation")
