#!/usr/bin/env python3
"""
scripts/extract_all_streams_80k.py

Phase 4: Extract aligned S1+S2+S3 features for all on-disk images.

Sources:
  Real: data/real/coco/train2017/      (label=0)
  Fake: data/fake/stable_diffusion/images/  (label=1)
        data/fake/stylegan3/images/         (label=1)

Output (features_new_3stream/):
  stream1_features.npy  (N, 128)  — DnCNN residual + PRNUEncoder
  stream2_features.npy  (N, 256)  — FFT SpectralTransform + ResNet-18
  stream3_features.npy  (N, 64)   — NSS statistics + NSSEncoder
  labels.npy            (N,)
  paths.txt             N lines of absolute paths

Also saves feature_stats.npz (mean/std over combined 448-dim vector) for fusion normalisation.

Usage:
    python3 scripts/extract_all_streams_80k.py \
        --prnu_checkpoint models/pretrained/prnu_encoder_trained.pth \
        --out_dir features_new_3stream \
        --batch_size 32
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stream1_prnu.denoiser import DnCNNWrapper
from src.stream1_prnu.encoder import PRNUEncoder
from src.stream2_fft.encoder import Stream2
from src.stream3_nss.encoder import NSSEncoder
from src.stream3_nss.statistics import extract_nss_features, NSS_DIM

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

_NORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

log_lines: list[str] = []


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_lines.append(line)


def collect_all_paths(real_dirs: list[str], fake_dirs: list[str]) -> list[tuple[str, int]]:
    items: list[tuple[str, int]] = []
    for d in real_dirs:
        for p in sorted(Path(d).rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                items.append((str(p), 0))
    for d in fake_dirs:
        for p in sorted(Path(d).rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                items.append((str(p), 1))
    return items


def load_prnu_encoder(checkpoint: str, device: torch.device) -> tuple[DnCNNWrapper, PRNUEncoder, torch.nn.Linear]:
    denoiser = DnCNNWrapper(device=str(device))
    encoder  = PRNUEncoder(out_dim=128).to(device).eval()
    head     = torch.nn.Linear(128, 1).to(device).eval()

    ckpt = torch.load(checkpoint, map_location=device)
    if "encoder_state" in ckpt:
        encoder.load_state_dict(ckpt["encoder_state"])
        head.load_state_dict(ckpt["head_state"])
        log(f"PRNUEncoder loaded ← {checkpoint}  (val_AUC={ckpt.get('val_auc', '?')})")
    else:
        log(f"WARNING: could not parse PRNU checkpoint at {checkpoint} — random weights")
    return denoiser, encoder, head


def load_stream2(features_dir: Path, device: torch.device) -> Stream2:
    s2 = Stream2(out_dim=256, spectral_size=224).to(device).eval()
    w  = features_dir / "stream2" / "stream2_encoder.pth"
    if w.exists():
        s2.load_state_dict(torch.load(str(w), map_location=device))
        log(f"Stream2 loaded ← {w}")
    else:
        log(f"WARNING: {w} not found — random Stream2 weights")
    return s2


def load_stream3(features_dir: Path, device: torch.device) -> NSSEncoder:
    nss = NSSEncoder(in_dim=NSS_DIM, out_dim=64).to(device).eval()
    w   = features_dir / "stream3" / "nss_encoder.pth"
    if w.exists():
        nss.load_state_dict(torch.load(str(w), map_location=device))
        log(f"NSSEncoder loaded ← {w}")
    else:
        log(f"WARNING: {w} not found — random NSSEncoder weights")
    return nss


@torch.no_grad()
def extract_single(
    path: str,
    label: int,
    denoiser: DnCNNWrapper,
    prnu_enc: PRNUEncoder,
    s2: Stream2,
    nss_enc: NSSEncoder,
    device: torch.device,
    img_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (f1[128], f2[256], f3[64]) numpy arrays or zeros on error."""
    img = Image.open(path).convert("RGB")

    # --- Stream 1 (PRNU) ---
    t   = TF.to_tensor(img)                                   # [3, H, W]
    g   = TF.rgb_to_grayscale(t)                              # [1, H, W]
    g_s = TF.resize(g, [img_size, img_size],
                    interpolation=TF.InterpolationMode.BILINEAR)
    res = denoiser.noise_residual(g_s)                        # [1, S, S]
    f1  = prnu_enc(res.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()  # [128]

    # --- Stream 2 (FFT) ---
    t_norm = _NORM(img).unsqueeze(0).to(device)               # [1, 3, 224, 224]
    f2     = s2(t_norm).squeeze(0).cpu().numpy()              # [256]

    # --- Stream 3 (NSS) ---
    t_01   = TF.to_tensor(img)                                # [3, H, W] in [0,1]
    nss_v  = extract_nss_features(t_01)                       # [NSS_DIM]
    f3     = nss_enc(torch.from_numpy(nss_v).unsqueeze(0)
                     .float().to(device)).squeeze(0).cpu().numpy()  # [64]

    return f1, f2, f3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prnu_checkpoint", default="models/pretrained/prnu_encoder_trained.pth")
    parser.add_argument("--features_dir",    default="features_combined")
    parser.add_argument("--out_dir",         default="features_new_3stream")
    parser.add_argument("--img_size",        type=int, default=128,
                        help="DnCNN input size (same as training)")
    parser.add_argument("--save_every",      type=int, default=5000,
                        help="Save partial results every N images")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    real_dirs = [str(ROOT / "data" / "real" / "coco" / "train2017")]
    fake_dirs = [
        str(ROOT / "data" / "fake" / "stable_diffusion" / "images"),
        str(ROOT / "data" / "fake" / "stylegan3" / "images"),
    ]

    log("Collecting image paths (sorted for reproducibility)...")
    all_items = collect_all_paths(real_dirs, fake_dirs)
    log(f"Total images: {len(all_items)} "
        f"(real={sum(l==0 for _,l in all_items)}, fake={sum(l==1 for _,l in all_items)})")

    n = len(all_items)
    n_real = sum(l == 0 for _, l in all_items)
    n_fake = sum(l == 1 for _, l in all_items)
    # Time estimate
    secs_per_img = 0.35   # S1(0.16) + S2(0.10) + S3(0.09) on CPU, rough
    est_hours = n * secs_per_img / 3600
    log(f"Estimated extraction time: {est_hours:.1f} hours at ~{secs_per_img}s/image on CPU")

    feat_dir = ROOT / args.out_dir
    feat_dir.mkdir(parents=True, exist_ok=True)

    # Check if already partially done
    s1_path = feat_dir / "stream1_features.npy"
    s2_path = feat_dir / "stream2_features.npy"
    s3_path = feat_dir / "stream3_features.npy"
    lb_path = feat_dir / "labels.npy"
    pt_path = feat_dir / "paths.txt"

    start_idx = 0
    f1_buf: list[np.ndarray] = []
    f2_buf: list[np.ndarray] = []
    f3_buf: list[np.ndarray] = []
    lb_buf: list[float]      = []
    pt_buf: list[str]        = []

    if s1_path.exists() and lb_path.exists() and pt_path.exists():
        existing_labels = np.load(str(lb_path))
        start_idx = len(existing_labels)
        if start_idx >= n:
            log(f"Extraction already complete ({start_idx}/{n}). Recomputing stats only.")
        else:
            log(f"Resuming from index {start_idx}/{n}")
            f1_buf = list(np.load(str(s1_path)))
            f2_buf = list(np.load(str(s2_path)))
            f3_buf = list(np.load(str(s3_path)))
            lb_buf = list(existing_labels)
            with open(pt_path) as fh:
                pt_buf = [l.rstrip() for l in fh]

    # Load models
    feat_dir_combined = ROOT / args.features_dir
    denoiser, prnu_enc, _ = load_prnu_encoder(str(ROOT / args.prnu_checkpoint), device)
    s2_model  = load_stream2(feat_dir_combined, device)
    nss_model = load_stream3(feat_dir_combined, device)

    # Main extraction loop
    errors = 0
    t0 = time.time()
    for i, (path, label) in enumerate(all_items[start_idx:], start=start_idx):
        try:
            f1, f2, f3 = extract_single(
                path, label, denoiser, prnu_enc, s2_model, nss_model, device, args.img_size
            )
            f1_buf.append(f1)
            f2_buf.append(f2)
            f3_buf.append(f3)
            lb_buf.append(float(label))
            pt_buf.append(path)
        except Exception as e:
            log(f"  ERROR at {os.path.basename(path)}: {e}")
            errors += 1
            continue

        # Progress every 1000 images
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1 - start_idx) / elapsed
            remaining = (n - i - 1) / rate / 3600
            log(f"  {i+1}/{n} images  errors={errors}  "
                f"{rate:.1f} img/s  ETA {remaining:.1f}h")

        # Periodic save
        if (i + 1) % args.save_every == 0 or (i + 1) == n:
            np.save(str(s1_path), np.array(f1_buf, dtype=np.float32))
            np.save(str(s2_path), np.array(f2_buf, dtype=np.float32))
            np.save(str(s3_path), np.array(f3_buf, dtype=np.float32))
            np.save(str(lb_path), np.array(lb_buf, dtype=np.float32))
            with open(pt_path, "w") as fh:
                fh.write("\n".join(pt_buf) + "\n")
            log(f"  Saved checkpoint at {i+1}/{n} images → {feat_dir}")

    total_time = time.time() - t0
    log(f"\nExtraction complete: {len(f1_buf)} images in {total_time/3600:.2f}h  errors={errors}")

    # Final save
    f1_arr = np.array(f1_buf, dtype=np.float32)  # (N, 128)
    f2_arr = np.array(f2_buf, dtype=np.float32)  # (N, 256)
    f3_arr = np.array(f3_buf, dtype=np.float32)  # (N, 64)
    lb_arr = np.array(lb_buf, dtype=np.float32)  # (N,)
    np.save(str(s1_path), f1_arr)
    np.save(str(s2_path), f2_arr)
    np.save(str(s3_path), f3_arr)
    np.save(str(lb_path), lb_arr)
    with open(pt_path, "w") as fh:
        fh.write("\n".join(pt_buf) + "\n")

    # Compute normalisation stats on full 448-dim concatenated vector
    combined = np.concatenate([f1_arr, f2_arr, f3_arr], axis=1)  # (N, 448)
    stats_mean = combined.mean(axis=0).astype(np.float32)
    stats_std  = np.maximum(combined.std(axis=0), 1e-2).astype(np.float32)
    np.savez(str(feat_dir / "feature_stats.npz"), mean=stats_mean, std=stats_std)

    log(f"feature_stats.npz saved  mean.shape={stats_mean.shape}")
    log(f"Shapes: S1={f1_arr.shape}  S2={f2_arr.shape}  S3={f3_arr.shape}  labels={lb_arr.shape}")
    log(f"Label dist: real={int((lb_arr==0).sum())}  fake={int((lb_arr==1).sum())}")

    # Summary JSON
    summary = {
        "timestamp":    datetime.now().isoformat(),
        "n_total":      len(lb_arr),
        "n_real":       int((lb_arr == 0).sum()),
        "n_fake":       int((lb_arr == 1).sum()),
        "errors":       errors,
        "total_time_h": round(total_time / 3600, 2),
        "out_dir":      str(feat_dir),
        "args":         vars(args),
    }
    with open(feat_dir / "extraction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Summary saved → {feat_dir}/extraction_summary.json")


if __name__ == "__main__":
    main()
