"""
data/preprocess.py

Offline feature extraction to .npy / .h5 files.
Run this script once before training to pre-extract all features.

Usage:
    python -m src.data.preprocess \
        --data_root data/ \
        --out_dir features/ \
        --streams 123 \
        --img_size 224 \
        --batch_size 16 \
        --device cuda
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def extract_features(
    data_root: str | Path,
    out_dir: str | Path,
    streams: str = "123",
    img_size: int = 224,
    batch_size: int = 16,
    device: str = "cpu",
    num_workers: int = 4,
    dncnn_weights: str | None = None,
    pretrained_backbone: bool = False,
):
    """
    Extract features for all three streams and save as .npy files.

    Saves to:
        out_dir/stream1/features.npy  (if '1' in streams)
        out_dir/stream2/features.npy  (if '2' in streams)
        out_dir/stream3/features.npy  (if '3' in streams)
        out_dir/labels.npy
    """
    from src.data.dataset import RealFakeImageFolder, get_val_transform, get_prnu_transform
    from src.stream1_prnu.denoiser import DnCNNWrapper
    from src.stream1_prnu.extractor import PRNUExtractor
    from src.stream1_prnu.encoder import PRNUEncoder
    from src.stream2_fft.encoder import Stream2
    from src.stream3_nss.statistics import extract_nss_features, NSS_DIM

    device_t = torch.device(device)
    out_dir = Path(out_dir)

    # Load dataset WITHOUT resize transform (for PRNU, full-res is needed)
    dataset_noresize = RealFakeImageFolder(
        data_root,
        transform=get_prnu_transform(),  # no resize
    )
    dataset_resized = RealFakeImageFolder(
        data_root,
        transform=get_val_transform(img_size),  # with resize for FFT/NSS
    )

    # Align samples
    assert len(dataset_noresize) == len(dataset_resized)

    n = len(dataset_noresize)
    labels = np.array([dataset_noresize.samples[i][1] for i in range(n)], dtype=np.float32)

    # Save labels
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir / "labels.npy"), labels)
    print(f"[Preprocess] Saved labels: {labels.shape}")

    # ── Stream 1: PRNU ────────────────────────────────────────────────────────
    if "1" in streams:
        print(f"[Preprocess] Extracting Stream 1 (PRNU, resize to {img_size}px)...")
        denoiser = DnCNNWrapper(weights_path=dncnn_weights, device=device)
        extractor = PRNUExtractor(denoiser=denoiser)
        encoder = PRNUEncoder(out_dim=128).to(device_t).eval()

        # Resize before PRNU for speed on CPU (still captures noise structure)
        import torchvision.transforms.functional as TF
        loader = DataLoader(dataset_noresize, batch_size=1, shuffle=False, num_workers=0)
        feats = []

        with torch.no_grad():
            for img_t, _ in tqdm(loader, desc="Stream1"):
                # Resize to img_size for faster DnCNN processing
                img_t = TF.resize(img_t, [img_size, img_size], antialias=True)
                # img_t: [1, C, H, W]
                gray = img_t[:, :1, :, :] if img_t.shape[1] == 1 else \
                    (0.299 * img_t[:, 0:1] + 0.587 * img_t[:, 1:2] + 0.114 * img_t[:, 2:3])
                residual = denoiser.noise_residual(gray.to(device_t))  # [1, 1, H, W]
                feat = encoder(residual)  # [1, 128]
                feats.append(feat.cpu().numpy())

        feats_arr = np.vstack(feats)
        s1_dir = out_dir / "stream1"
        s1_dir.mkdir(exist_ok=True)
        np.save(str(s1_dir / "features.npy"), feats_arr)
        print(f"[Preprocess] Stream1 features: {feats_arr.shape}")

    # ── Stream 2: FFT ────────────────────────────────────────────────────────
    if "2" in streams:
        print("[Preprocess] Extracting Stream 2 (FFT)...")
        torch.manual_seed(0)   # fixed seed for reproducible spectral mask weights
        stream2 = Stream2(out_dim=256, spectral_size=img_size, pretrained_backbone=pretrained_backbone).to(device_t).eval()

        loader = DataLoader(
            dataset_resized, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        feats = []

        with torch.no_grad():
            for img_t, _ in tqdm(loader, desc="Stream2"):
                img_t = img_t.to(device_t)
                feat = stream2(img_t)  # [B, 256]
                feats.append(feat.cpu().numpy())

        feats_arr = np.vstack(feats)
        s2_dir = out_dir / "stream2"
        s2_dir.mkdir(exist_ok=True)
        np.save(str(s2_dir / "features.npy"), feats_arr)
        # Save encoder weights for reproducible inference on new images
        torch.save(stream2.to("cpu").state_dict(), str(s2_dir / "stream2_encoder.pth"))
        print(f"[Preprocess] Stream2 features: {feats_arr.shape}")

    # ── Stream 3: NSS ────────────────────────────────────────────────────────
    if "3" in streams:
        print("[Preprocess] Extracting Stream 3 (NSS → NSSEncoder → 64-dim)...")
        from src.stream3_nss.encoder import NSSEncoder
        from src.stream3_nss.statistics import NSS_DIM
        torch.manual_seed(0)   # fixed seed for reproducible NSS encoder weights
        nss_encoder = NSSEncoder(in_dim=NSS_DIM, out_dim=64).to(device_t).eval()

        loader = DataLoader(dataset_resized, batch_size=1, shuffle=False, num_workers=0)
        feats = []

        with torch.no_grad():
            for img_t, _ in tqdm(loader, desc="Stream3"):
                nss = extract_nss_features(img_t.squeeze(0))         # [18]
                nss_t = torch.from_numpy(nss).unsqueeze(0).float().to(device_t)  # [1, 18]
                encoded = nss_encoder(nss_t)                          # [1, 64]
                feats.append(encoded.cpu().numpy())

        feats_arr = np.vstack(feats)   # [N, 64]
        s3_dir = out_dir / "stream3"
        s3_dir.mkdir(exist_ok=True)
        np.save(str(s3_dir / "features.npy"), feats_arr)
        # Save encoder weights so evaluation can reproduce the same projection
        torch.save(nss_encoder.state_dict(), str(s3_dir / "nss_encoder.pth"))
        print(f"[Preprocess] Stream3 features: {feats_arr.shape}")

    print("[Preprocess] Done.")


if __name__ == "__main__":
    import random
    import shutil
    import sys
    from PIL import ImageFile

    # Tolerate truncated images (e.g. incomplete COCO downloads) rather than crashing
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Ensure project root is on sys.path so 'src.*' imports resolve
    sys.path.insert(0, str(PROJECT_ROOT))

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/")
    parser.add_argument("--data_dir", default="",
                        help="Alias for --data_root (takes precedence if set)")
    parser.add_argument("--out_dir", default="features/")
    parser.add_argument("--streams", default="123")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dncnn_weights", default=None)
    parser.add_argument("--pretrained_backbone", action="store_true",
                        help="Use pretrained ImageNet ResNet-18 for Stream 2 FFT encoder")
    parser.add_argument("--subset_size", type=int, default=0,
                        help="Max images per class (0 = use all images, N > 0 = cap per class)")
    args = parser.parse_args()

    data_root = args.data_dir if args.data_dir else args.data_root

    # If subset_size > 0, build a temporary subset directory with symlinks
    subset_dir = None
    if args.subset_size > 0:
        from src.data.dataset import RealFakeImageFolder
        print(f"[Preprocess] Discovering images under {data_root} ...")
        full_ds = RealFakeImageFolder(data_root)
        real_paths = [s for s in full_ds.samples if s[1] == 0]
        fake_paths = [s for s in full_ds.samples if s[1] == 1]
        print(f"[Preprocess] Found {len(real_paths)} real, {len(fake_paths)} fake.")

        random.seed(42)
        n = args.subset_size
        real_sample = random.sample(real_paths, min(n, len(real_paths)))
        fake_sample = random.sample(fake_paths, min(n, len(fake_paths)))
        print(f"[Preprocess] Subset: {len(real_sample)} real + {len(fake_sample)} fake")

        subset_dir = Path(data_root).parent / ".preprocess_subset"
        if subset_dir.exists():
            shutil.rmtree(subset_dir)
        (subset_dir / "real").mkdir(parents=True)
        (subset_dir / "fake").mkdir(parents=True)

        def _link(src: Path, dst_dir: Path):
            dst = dst_dir / src.name
            if dst.exists():
                dst = dst_dir / f"{src.stem}_{abs(hash(str(src))) % 100000}{src.suffix}"
            try:
                dst.symlink_to(src.resolve())
            except (OSError, NotImplementedError):
                if not dst.exists():
                    shutil.copy2(src, dst)

        for p, _, _ in real_sample:
            _link(p, subset_dir / "real")
        for p, _, _ in fake_sample:
            _link(p, subset_dir / "fake")
        data_root = str(subset_dir)

    extract_features(
        data_root=data_root,
        out_dir=args.out_dir,
        streams=args.streams,
        img_size=args.img_size,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        dncnn_weights=args.dncnn_weights,
        pretrained_backbone=args.pretrained_backbone,
    )

    if subset_dir is not None and subset_dir.exists():
        shutil.rmtree(subset_dir)
