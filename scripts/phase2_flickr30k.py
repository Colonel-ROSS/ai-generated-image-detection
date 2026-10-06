"""
Phase 2.3 — Download Flickr30k via HuggingFace and extract features.

Dataset: nlphuji/flickr30k (~31,783 images, all real)
Saves features to features_new/flickr30k/ with labels.npy (all 0 = real).
Deletes raw images after feature extraction.
"""
import sys
import gc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

OUT_DIR = ROOT / "features_new" / "flickr30k"
FEATURES_FULL = ROOT / "features_full"
CACHE_DIR = ROOT / "data" / "real" / "flickr30k" / "_hf_cache"

MIN_FREE_GB = 15

def check_disk():
    import shutil as _sh
    free = _sh.disk_usage("/mnt/d").free / (1024**3)
    print(f"  Disk free: {free:.1f} GB")
    return free

print("=== Phase 2.3: Flickr30k via HuggingFace ===")
free = check_disk()
if free < MIN_FREE_GB + 5:
    print(f"Need at least {MIN_FREE_GB+5} GB free. Aborting.")
    sys.exit(1)

from datasets import load_dataset
print("Loading nlphuji/flickr30k (streaming) ...")
ds = load_dataset("nlphuji/flickr30k", split="test", streaming=True)
check_disk()

# Load encoders (same weights as features_full)
from src.stream2_fft.encoder import Stream2
from src.stream3_nss.encoder import NSSEncoder
from src.stream3_nss.statistics import extract_nss_features, NSS_DIM

torch.manual_seed(0)
s2_enc = Stream2(out_dim=256, spectral_size=224, pretrained_backbone=True).eval()
s2_enc.load_state_dict(torch.load(
    str(FEATURES_FULL / "stream2" / "stream2_encoder.pth"), map_location="cpu"))
print(f"Loaded Stream2 encoder")

nss_enc = NSSEncoder(in_dim=NSS_DIM, out_dim=64).eval()
nss_enc.load_state_dict(torch.load(
    str(FEATURES_FULL / "stream3" / "nss_encoder.pth"), map_location="cpu"))
print(f"Loaded NSSEncoder")

norm = T.Compose([
    T.Resize((224, 224)), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

feats_s2, feats_s3 = [], []
errors = 0

for i, sample in enumerate(ds):
    if i % 1000 == 0:
        print(f"  processed={i} errors={errors}", flush=True)
    try:
        img = sample["image"].convert("RGB")
        t = norm(img).unsqueeze(0)
        with torch.no_grad():
            f2 = s2_enc(t).squeeze(0).numpy()
            raw_nss = extract_nss_features(t.squeeze(0))
            f3 = nss_enc(torch.from_numpy(raw_nss).unsqueeze(0).float()).squeeze(0).numpy()
        feats_s2.append(f2)
        feats_s3.append(f3)
    except Exception as e:
        feats_s2.append(np.zeros(256, dtype=np.float32))
        feats_s3.append(np.zeros(64, dtype=np.float32))
        errors += 1

    # Periodic disk check every 5000 images
    if i % 5000 == 0 and i > 0:
        free = check_disk()
        if free < MIN_FREE_GB:
            print(f"Disk low ({free:.1f} GB), stopping.")
            break

feats_s2 = np.stack(feats_s2).astype(np.float32)
feats_s3 = np.stack(feats_s3).astype(np.float32)
labels = np.zeros(len(feats_s2), dtype=np.float32)  # all real
print(f"Features: S2={feats_s2.shape} S3={feats_s3.shape} errors={errors}")

# Save
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "stream2").mkdir(exist_ok=True)
(OUT_DIR / "stream3").mkdir(exist_ok=True)
np.save(str(OUT_DIR / "stream2" / "features.npy"), feats_s2)
np.save(str(OUT_DIR / "stream3" / "features.npy"), feats_s3)
np.save(str(OUT_DIR / "labels.npy"), labels)
print(f"Saved to {OUT_DIR}")

check_disk()

print("\n=== Phase 2.3 Flickr30k complete ===")
