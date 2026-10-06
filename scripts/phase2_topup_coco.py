"""
Phase 2.1 — Get remaining COCO images and extract features via HuggingFace streaming.

Streams detection-datasets/coco train split, skips images already in
data/real/coco/train2017/ (identified by image_id), extracts Stream2+3
features for new images, saves to features_new/coco_extra/.
No raw images saved to disk — features extracted on the fly.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

COCO_DIR      = ROOT / "data" / "real" / "coco" / "train2017"
OUT_DIR       = ROOT / "features_new" / "coco_extra"
FEATURES_FULL = ROOT / "features_full"
MIN_FREE_GB   = 20

def check_disk():
    import shutil as _sh
    free = _sh.disk_usage("/mnt/d").free / (1024**3)
    print(f"  Disk free: {free:.1f} GB")
    return free

print("=== Phase 2.1: COCO top-up via HF streaming ===")
check_disk()

# Build set of existing COCO image_ids from filenames
existing_ids = set()
for f in COCO_DIR.glob("*.jpg"):
    try:
        existing_ids.add(int(f.stem))
    except ValueError:
        pass
print(f"Existing COCO images: {len(existing_ids)}")

# Load encoders with same weights as features_full
from src.stream2_fft.encoder import Stream2
from src.stream3_nss.encoder import NSSEncoder
from src.stream3_nss.statistics import extract_nss_features, NSS_DIM

torch.manual_seed(0)
s2_enc = Stream2(out_dim=256, spectral_size=224, pretrained_backbone=True).eval()
s2_enc.load_state_dict(torch.load(
    str(FEATURES_FULL / "stream2" / "stream2_encoder.pth"), map_location="cpu"))
print("Loaded Stream2 encoder")

nss_enc = NSSEncoder(in_dim=NSS_DIM, out_dim=64).eval()
nss_enc.load_state_dict(torch.load(
    str(FEATURES_FULL / "stream3" / "nss_encoder.pth"), map_location="cpu"))
print("Loaded NSSEncoder")

norm = T.Compose([
    T.Resize((224, 224)), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Stream COCO and extract features for new images only
from datasets import load_dataset
ds = load_dataset("detection-datasets/coco", split="train", streaming=True)

feats_s2, feats_s3 = [], []
errors = 0
processed = 0
skipped = 0

print("Streaming COCO train split...")
for sample in ds:
    img_id = sample["image_id"]

    if img_id in existing_ids:
        skipped += 1
        continue

    free = check_disk() if processed % 10000 == 0 and processed > 0 else None
    if free is not None and free < MIN_FREE_GB:
        print(f"Disk low ({free:.1f} GB), stopping.")
        break

    try:
        img = sample["image"].convert("RGB")
        t = norm(img).unsqueeze(0)
        with torch.no_grad():
            f2 = s2_enc(t).squeeze(0).numpy()
            raw = extract_nss_features(t.squeeze(0))
            f3 = nss_enc(torch.from_numpy(raw).unsqueeze(0).float()).squeeze(0).numpy()
        feats_s2.append(f2)
        feats_s3.append(f3)
        processed += 1
    except Exception as e:
        feats_s2.append(np.zeros(256, dtype=np.float32))
        feats_s3.append(np.zeros(64, dtype=np.float32))
        errors += 1
        processed += 1

    if processed % 1000 == 0:
        print(f"  processed={processed} skipped={skipped} errors={errors}", flush=True)

if not feats_s2:
    print("No new COCO images found.")
else:
    feats_s2 = np.stack(feats_s2).astype(np.float32)
    feats_s3 = np.stack(feats_s3).astype(np.float32)
    labels = np.zeros(len(feats_s2), dtype=np.float32)  # all real

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "stream2").mkdir(exist_ok=True)
    (OUT_DIR / "stream3").mkdir(exist_ok=True)
    np.save(str(OUT_DIR / "stream2" / "features.npy"), feats_s2)
    np.save(str(OUT_DIR / "stream3" / "features.npy"), feats_s3)
    np.save(str(OUT_DIR / "labels.npy"), labels)
    print(f"\nSaved {len(feats_s2)} new COCO features (errors={errors}) → {OUT_DIR}")

check_disk()
print("\n=== Phase 2.1 complete ===")
