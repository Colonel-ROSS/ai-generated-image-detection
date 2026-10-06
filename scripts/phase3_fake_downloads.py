"""
Phase 3 — Download additional fake images and extract features.

Tries in order:
  3.1 StyleGAN3 via HuggingFace (Styyx/stylegan3-dataset)
  3.2 DiffusionDB SD2.x samples (poloclub/diffusiondb 2m_first_10k)
  3.3 OpenImages/LAION-Aesthetics-based SD1.5 (stabilityai/stable-diffusion-2 samples)

Features saved to features_new/{source}/ with labels=1 (fake).
Raw images deleted after extraction.
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

FEATURES_FULL = ROOT / "features_full"
MIN_FREE_GB = 20

def check_disk():
    import shutil as _sh
    free = _sh.disk_usage("/mnt/d").free / (1024**3)
    print(f"  Disk free: {free:.1f} GB")
    return free

def load_encoders():
    from src.stream2_fft.encoder import Stream2
    from src.stream3_nss.encoder import NSSEncoder
    from src.stream3_nss.statistics import NSS_DIM

    torch.manual_seed(0)
    s2_enc = Stream2(out_dim=256, spectral_size=224, pretrained_backbone=True).eval()
    s2_enc.load_state_dict(torch.load(
        str(FEATURES_FULL / "stream2" / "stream2_encoder.pth"), map_location="cpu"))

    nss_enc = NSSEncoder(in_dim=NSS_DIM, out_dim=64).eval()
    nss_enc.load_state_dict(torch.load(
        str(FEATURES_FULL / "stream3" / "nss_encoder.pth"), map_location="cpu"))
    return s2_enc, nss_enc, NSS_DIM

def extract_and_save(image_iter, out_dir: Path, label: int, s2_enc, nss_enc, NSS_DIM):
    from src.stream3_nss.statistics import extract_nss_features

    norm = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    feats_s2, feats_s3 = [], []
    errors = 0

    for i, img in enumerate(tqdm(image_iter, desc=f"Extracting {out_dir.name}")):
        try:
            if not isinstance(img, Image.Image):
                img = img.convert("RGB")
            else:
                img = img.convert("RGB")
            t = norm(img).unsqueeze(0)
            with torch.no_grad():
                f2 = s2_enc(t).squeeze(0).numpy()
                raw = extract_nss_features(t.squeeze(0))
                f3 = nss_enc(torch.from_numpy(raw).unsqueeze(0).float()).squeeze(0).numpy()
            feats_s2.append(f2)
            feats_s3.append(f3)
        except Exception as e:
            feats_s2.append(np.zeros(256, dtype=np.float32))
            feats_s3.append(np.zeros(64, dtype=np.float32))
            errors += 1

        if i % 2000 == 0 and i > 0:
            free = check_disk()
            if free < MIN_FREE_GB:
                print(f"WARN: disk low ({free:.1f} GB), stopping at {i}")
                break

    if not feats_s2:
        print("No features extracted.")
        return

    feats_s2 = np.stack(feats_s2).astype(np.float32)
    feats_s3 = np.stack(feats_s3).astype(np.float32)
    labels = np.full(len(feats_s2), label, dtype=np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stream2").mkdir(exist_ok=True)
    (out_dir / "stream3").mkdir(exist_ok=True)
    np.save(str(out_dir / "stream2" / "features.npy"), feats_s2)
    np.save(str(out_dir / "stream3" / "features.npy"), feats_s3)
    np.save(str(out_dir / "labels.npy"), labels)
    print(f"Saved {len(feats_s2)} samples (errors={errors}) → {out_dir}")


print("=== Phase 3: Additional fake images ===")
check_disk()

s2_enc, nss_enc, NSS_DIM = load_encoders()

# ── 3.1 StyleGAN3 via HuggingFace ─────────────────────────────────────────────
print("\n--- 3.1 StyleGAN3 (Styyx/stylegan3-dataset) ---")
try:
    from datasets import load_dataset
    import shutil
    cache = ROOT / "data" / "fake" / "stylegan3" / "_hf_cache"
    ds_sgan = load_dataset("Styyx/stylegan3-dataset", split="train",
                           cache_dir=str(cache))
    print(f"StyleGAN3 dataset: {len(ds_sgan)} samples")
    check_disk()
    extract_and_save(
        (s["image"] for s in ds_sgan),
        ROOT / "features_new" / "stylegan3_hf",
        label=1, s2_enc=s2_enc, nss_enc=nss_enc, NSS_DIM=NSS_DIM
    )
    shutil.rmtree(str(cache), ignore_errors=True)
    check_disk()
except Exception as e:
    print(f"StyleGAN3 failed: {e}")

# ── 3.2 DiffusionDB (SD2 generated) ───────────────────────────────────────────
print("\n--- 3.2 DiffusionDB SD2 (poloclub/diffusiondb 2m_first_10k) ---")
try:
    import shutil
    cache = ROOT / "data" / "fake" / "sdxl" / "_hf_cache"
    ds_diff = load_dataset("poloclub/diffusiondb", "2m_first_10k",
                           split="train", cache_dir=str(cache))
    print(f"DiffusionDB: {len(ds_diff)} samples")
    check_disk()
    extract_and_save(
        (s["image"] for s in ds_diff),
        ROOT / "features_new" / "diffusiondb",
        label=1, s2_enc=s2_enc, nss_enc=nss_enc, NSS_DIM=NSS_DIM
    )
    shutil.rmtree(str(cache), ignore_errors=True)
    check_disk()
except Exception as e:
    print(f"DiffusionDB failed: {e}")

# ── 3.3 DALL-E 3 / GenImage ───────────────────────────────────────────────────
print("\n--- 3.3 GenImage (shizhe95/GenImage) ---")
try:
    import shutil
    cache = ROOT / "data" / "fake" / "dalle3" / "_hf_cache"
    ds_gen = load_dataset("shizhe95/GenImage", split="test",
                          cache_dir=str(cache))
    print(f"GenImage: {len(ds_gen)} samples")
    check_disk()
    extract_and_save(
        (s["image"] if "image" in s else Image.open(s["path"]) for s in ds_gen),
        ROOT / "features_new" / "genimage",
        label=1, s2_enc=s2_enc, nss_enc=nss_enc, NSS_DIM=NSS_DIM
    )
    shutil.rmtree(str(cache), ignore_errors=True)
    check_disk()
except Exception as e:
    print(f"GenImage failed: {e}")

print("\n=== Phase 3 complete ===")
check_disk()
