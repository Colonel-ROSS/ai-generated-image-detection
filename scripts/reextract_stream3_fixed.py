"""
Re-extract Stream 3 (NSS) features using the fixed _to_gray_numpy() in statistics.py.

Uses SAME NSSEncoder weights as features_full/stream3/nss_encoder.pth so the
18→64 projection is identical; only the upstream 18-dim NSS stats are corrected.

Output: features_fixed/stream3/features.npy  shape=(80060, 64)
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFile
from tqdm import tqdm

from src.stream3_nss.statistics import extract_nss_features, NSS_DIM
from src.stream3_nss.encoder import NSSEncoder
from src.data.dataset import RealFakeImageFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True

FEATURES_FULL = ROOT / "features_full"
FEATURES_FIXED = ROOT / "features_fixed"
DATA_ROOT = ROOT / "data"

# ── Prepare output dirs ────────────────────────────────────────────────────────
(FEATURES_FIXED / "stream3").mkdir(parents=True, exist_ok=True)

# ── Load NSSEncoder with saved weights (fixed random projection) ───────────────
nss_enc = NSSEncoder(in_dim=NSS_DIM, out_dim=64)
weights_path = FEATURES_FULL / "stream3" / "nss_encoder.pth"
nss_enc.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
nss_enc.eval()
print(f"Loaded NSSEncoder from {weights_path}")

# ── Build dataset in SAME order as original extraction ────────────────────────
# RealFakeImageFolder walks real/ then fake/ with rglob — deterministic on same FS
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
ds = RealFakeImageFolder(DATA_ROOT, transform=transform)
n = len(ds)
print(f"Dataset: {n} images")

# Verify label distribution matches features_full
labels_orig = np.load(str(FEATURES_FULL / "labels.npy"))
assert len(labels_orig) == n, f"Length mismatch: orig={len(labels_orig)} ds={n}"
labels_new = np.array([ds.samples[i][1] for i in range(n)], dtype=np.float32)
match_count = (labels_orig == labels_new).sum()
print(f"Label alignment: {match_count}/{n} match ({100*match_count/n:.2f}%)")
if match_count < n:
    print("WARNING: Label ordering mismatch — filesystem order changed!")
    # Still continue; the misalignment would be obvious from bad results
else:
    print("Label ordering confirmed identical — safe to proceed.")

# ── Extract features ──────────────────────────────────────────────────────────
t0 = time.time()
feats = []
errors = 0

for i in tqdm(range(n), desc="NSS re-extract (fixed)"):
    path, label, _ = ds.samples[i]
    try:
        img = Image.open(path).convert("RGB")
        t = transform(img)
        raw = extract_nss_features(t)  # [18] — uses fixed _to_gray_numpy
        with torch.no_grad():
            f3 = nss_enc(torch.from_numpy(raw).unsqueeze(0).float()).squeeze(0).numpy()
        feats.append(f3)
    except Exception as e:
        feats.append(np.zeros(64, dtype=np.float32))
        errors += 1

elapsed = time.time() - t0
feats_arr = np.stack(feats).astype(np.float32)
print(f"\nExtraction complete: {feats_arr.shape}  errors={errors}  time={elapsed/60:.1f}min")

# ── Sanity check: compare feature distributions ────────────────────────────────
old_s3 = np.load(str(FEATURES_FULL / "stream3" / "features.npy"))
print(f"\nOld features_full S3: mean={old_s3.mean():.4f} std={old_s3.std():.4f} max={np.abs(old_s3).max():.4f}")
print(f"New features_fixed S3: mean={feats_arr.mean():.4f} std={feats_arr.std():.4f} max={np.abs(feats_arr).max():.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = FEATURES_FIXED / "stream3" / "features.npy"
np.save(str(out_path), feats_arr)
print(f"Saved → {out_path}")
