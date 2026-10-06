#!/usr/bin/env python3
"""
scripts/prnu_ncc_test.py

PRNU NCC discriminability test — memory-safe version.

Builds a camera fingerprint from RAISE-1k D7000 TIFFs (10 images).
Crops each image to CROP_SIZE before DnCNN to avoid OOM on large TIFFs.
Tests whether the fingerprint NCC discriminates real COCO vs fake SD/SG3.

Thesis target: AUC > 0.60 to justify PRNU stream inclusion.
"""
from __future__ import annotations
import sys, json, random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torchvision.transforms.functional as TF
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stream1_prnu.denoiser import DnCNNWrapper

RAISE_DIR = ROOT / "data/real/raise1k/D7000_TIFF"
COCO_DIR  = ROOT / "data/real/coco/train2017"
SD_DIR    = ROOT / "data/fake/stable_diffusion/images"
SG3_DIR   = ROOT / "data/fake/stylegan3/images"
OUT_JSON  = ROOT / "experiments/prnu_ncc_discriminability.json"

N_RAISE_FP = 10    # fewer TIFFs = safe memory; 10 still gives stable mean fingerprint
N_REAL     = 100
N_SD       = 100
N_SG3      = 50
CROP_SIZE  = (256, 256)   # crop before DnCNN — mandatory for TIFF memory safety


def centre_crop_pil(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Centre-crop PIL image to (H, W). Resizes first if smaller."""
    w, h = img.size
    th, tw = size
    if h < th or w < tw:
        img = img.resize((max(w, tw), max(h, th)), Image.BILINEAR)
        w, h = img.size
    left = (w - tw) // 2
    top  = (h - th) // 2
    return img.crop((left, top, left + tw, top + th))


def extract_residual_cropped(img: Image.Image, denoiser: DnCNNWrapper) -> np.ndarray:
    """
    Crop to CROP_SIZE then compute noise residual.
    Returns (H, W) float32 numpy array.
    NOTE: Cropping before DnCNN is intentional here — this is a discriminability
    test only, not production PRNU extraction.
    """
    cropped = centre_crop_pil(img.convert("L"), CROP_SIZE)
    t = TF.to_tensor(cropped)           # [1, H, W] in [0,1]
    r = denoiser.noise_residual(t)      # [1, H, W]
    return r.squeeze().cpu().numpy()


def compute_ncc(residual: np.ndarray, fingerprint: np.ndarray) -> float:
    w = residual.flatten()
    r = fingerprint.flatten()
    if w.std() < 1e-8 or r.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(w, r)[0, 1])


def main():
    print("=" * 60)
    print("PRNU NCC Discriminability Test  (memory-safe)")
    print(f"Crop size: {CROP_SIZE}, N_FP images: {N_RAISE_FP}")
    print("=" * 60)

    denoiser = DnCNNWrapper()   # auto-loads models/pretrained/dncnn.pth

    # ── Step 1: Build fingerprint from RAISE-1k TIFFs ───────────────
    tif_paths = sorted(RAISE_DIR.glob("*.TIF"))[:N_RAISE_FP]
    if not tif_paths:
        tif_paths = sorted(RAISE_DIR.glob("*.tif"))[:N_RAISE_FP]
    print(f"\n[1] Building fingerprint from {len(tif_paths)} D7000 TIFFs ...")

    accumulator = np.zeros(CROP_SIZE, dtype=np.float64)
    count = 0
    for p in tqdm(tif_paths, desc="FP build", ncols=70):
        try:
            img = Image.open(str(p)).convert("RGB")
            residual = extract_residual_cropped(img, denoiser)
            accumulator += residual
            count += 1
        except Exception as e:
            print(f"  Skipping {p.name}: {e}")

    if count == 0:
        raise RuntimeError("No RAISE-1k TIFFs processed.")
    fingerprint = (accumulator / count).astype(np.float32)
    print(f"[1] Fingerprint built from {count} images. "
          f"mean={fingerprint.mean():.5f} std={fingerprint.std():.5f}")

    # ── Step 2: Sample test images ───────────────────────────────────
    random.seed(42)
    real_paths = random.sample(
        sorted(COCO_DIR.glob("*.jpg")),
        min(N_REAL, len(list(COCO_DIR.glob("*.jpg"))))
    )
    sd_all = sorted(SD_DIR.glob("*.webp")) + sorted(SD_DIR.glob("*.jpg")) + sorted(SD_DIR.glob("*.png"))
    sd_paths = random.sample(sd_all, min(N_SD, len(sd_all)))
    sg3_all  = sorted(SG3_DIR.glob("*.jpg")) + sorted(SG3_DIR.glob("*.png")) + sorted(SG3_DIR.glob("*.webp"))
    sg3_paths = random.sample(sg3_all, min(N_SG3, len(sg3_all)))

    print(f"\n[2] Sampled: {len(real_paths)} real (COCO), "
          f"{len(sd_paths)} fake (SD1.5), {len(sg3_paths)} fake (StyleGAN3)")

    # ── Step 3: Compute NCC for each image ──────────────────────────
    def run_ncc(paths, label):
        nccs = []
        for p in tqdm(paths, desc=f"NCC {label}", ncols=70):
            try:
                img = Image.open(str(p)).convert("RGB")
                r = extract_residual_cropped(img, denoiser)
                nccs.append(compute_ncc(r, fingerprint))
            except Exception:
                nccs.append(0.0)
        return np.array(nccs, dtype=np.float32)

    real_nccs = run_ncc(real_paths, "real/COCO")
    sd_nccs   = run_ncc(sd_paths,   "fake/SD1.5")
    sg3_nccs  = run_ncc(sg3_paths,  "fake/StyleGAN3")

    fake_nccs  = np.concatenate([sd_nccs, sg3_nccs])
    all_nccs   = np.concatenate([real_nccs, fake_nccs])
    all_labels = np.array([1] * len(real_nccs) + [0] * len(fake_nccs))

    # ── Step 4: Report ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Real (COCO)   NCC: mean={real_nccs.mean():.5f}  std={real_nccs.std():.5f}")
    print(f"  Fake (SD1.5)  NCC: mean={sd_nccs.mean():.5f}  std={sd_nccs.std():.5f}")
    print(f"  Fake (SG3)    NCC: mean={sg3_nccs.mean():.5f}  std={sg3_nccs.std():.5f}")

    if len(np.unique(all_labels)) > 1:
        auc = float(roc_auc_score(all_labels, all_nccs))
        print(f"\n  ROC AUC (real=1 vs fake=0):  {auc:.4f}")
        if auc > 0.70:
            verdict = "STRONG"
            print("  *** STRONG signal — PRNU discriminates real vs fake well ***")
            print("  RECOMMENDATION: Proceed with Stream 1 feature extraction.")
        elif auc > 0.60:
            verdict = "MODERATE"
            print("  Moderate signal — PRNU provides some value.")
            print("  RECOMMENDATION: Include Stream 1 with caution.")
        elif auc > 0.55:
            verdict = "WEAK"
            print("  Weak signal — PRNU barely above random.")
            print("  RECOMMENDATION: Skip Stream 1.")
        else:
            verdict = "NO_SIGNAL"
            print("  No signal — PRNU indistinguishable from random.")
            print("  RECOMMENDATION: DO NOT add Stream 1.")
    else:
        auc = 0.5
        verdict = "NO_SIGNAL"
        print("  Could not compute AUC — all labels same class.")

    gap = float(real_nccs.mean() - fake_nccs.mean())
    print(f"\n  NCC gap (real - fake mean): {gap:.5f}")

    # ── Step 5: Save ─────────────────────────────────────────────────
    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "fingerprint_source": "RAISE-1k D7000 TIFFs",
        "n_raise_images": count,
        "crop_size": list(CROP_SIZE),
        "n_real_test": len(real_paths),
        "n_fake_sd": len(sd_paths),
        "n_fake_sg3": len(sg3_paths),
        "real_ncc":     {"mean": float(real_nccs.mean()), "std": float(real_nccs.std())},
        "fake_sd_ncc":  {"mean": float(sd_nccs.mean()),  "std": float(sd_nccs.std())},
        "fake_sg3_ncc": {"mean": float(sg3_nccs.mean()), "std": float(sg3_nccs.std())},
        "auc_real_vs_fake":       auc,
        "ncc_gap_real_minus_fake": gap,
        "verdict": verdict,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[5] Saved to {OUT_JSON}")
    print("=" * 60)


if __name__ == "__main__":
    main()
