"""
data/dataset.py

PyTorch Dataset wrappers for the AI image detector.

Label convention: real=0, fake=1
"""

from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image, ImageFile
from typing import Callable, Optional

# Allow loading slightly truncated images without crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ──────────────────────────────────────────────────────────────────────────────
# Standard image transforms
# ──────────────────────────────────────────────────────────────────────────────

def get_train_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_prnu_transform() -> transforms.Compose:
    """Transform for PRNU residual extraction - no resize, just to tensor."""
    return transforms.Compose([
        transforms.ToTensor(),   # preserves full resolution
    ])


# ──────────────────────────────────────────────────────────────────────────────
# ImageFolder-based dataset
# ──────────────────────────────────────────────────────────────────────────────

class RealFakeImageFolder(Dataset):
    """
    Dataset from a directory with 'real/' and 'fake/' subdirectories.
    Labels: real=0, fake=1

    structure:
        root/
            real/  ...
            fake/  ...
    """

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
        include_generator_name: bool = False,
    ):
        self.root = Path(root)
        self.transform = transform
        self.include_generator_name = include_generator_name

        self.samples: list[tuple[Path, int, str]] = []

        for label, cls in [(0, "real"), (1, "fake")]:
            cls_dir = self.root / cls
            if not cls_dir.exists():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.webp"):
                for p in cls_dir.rglob(ext):
                    if cls == "fake":
                        # Use the top-level subdirectory under fake/ as generator name
                        rel = p.relative_to(cls_dir)
                        generator = rel.parts[0] if len(rel.parts) > 1 else p.parent.name
                    else:
                        generator = "real"
                    self.samples.append((p, label, generator))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root}/real or {root}/fake")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, generator = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Corrupted or unreadable file — return a black placeholder image
            img = Image.new("RGB", (224, 224), color=0)
        if self.transform is not None:
            img = self.transform(img)
        if self.include_generator_name:
            return img, label, generator
        return img, label


# ──────────────────────────────────────────────────────────────────────────────
# Pre-extracted feature dataset
# ──────────────────────────────────────────────────────────────────────────────

class FeatureDataset(Dataset):
    """
    Load pre-extracted features from .npy files.
    Expects:
        features_dir/stream1/*.npy  (PRNU 128-dim)
        features_dir/stream2/*.npy  (FFT 256-dim)
        features_dir/stream3/*.npy  (NSS 18-dim)
        features_dir/labels.npy     (int labels)
    """

    def __init__(
        self,
        features_dir: str | Path,
        streams: str = "123",
    ):
        self.features_dir = Path(features_dir)
        self.streams = streams

        labels_path = self.features_dir / "labels.npy"
        if not labels_path.exists():
            raise FileNotFoundError(f"labels.npy not found in {features_dir}")
        self.labels = np.load(str(labels_path)).astype(np.float32)

        self.feat_data = {}
        for s in streams:
            arr_path = self.features_dir / f"stream{s}" / "features.npy"
            if not arr_path.exists():
                raise FileNotFoundError(f"features.npy not found in {arr_path.parent}")
            self.feat_data[s] = np.load(str(arr_path)).astype(np.float32)

        # Validate lengths
        for s, arr in self.feat_data.items():
            assert len(arr) == len(self.labels), f"Length mismatch for stream {s}"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        feats = {s: torch.from_numpy(arr[idx]) for s, arr in self.feat_data.items()}
        label = torch.tensor(self.labels[idx])
        return feats, label


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader builders
# ──────────────────────────────────────────────────────────────────────────────

def make_dataloaders(
    data_root: str | Path,
    img_size: int = 224,
    batch_size: int = 32,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from a real/fake directory.
    Splits: 70% train, 15% val, 15% test.
    """
    dataset = RealFakeImageFolder(data_root)
    N = len(dataset)
    n_train = int(N * train_frac)
    n_val = int(N * val_frac)
    n_test = N - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    # Apply transforms after split via wrapping
    train_ds.dataset = RealFakeImageFolder(data_root, transform=get_train_transform(img_size))
    val_ds.dataset = RealFakeImageFolder(data_root, transform=get_val_transform(img_size))
    test_ds.dataset = RealFakeImageFolder(data_root, transform=get_val_transform(img_size))

    # Rebuild with transforms - use SimpleSubset approach
    train_loader = DataLoader(
        _TransformedSubset(train_ds.dataset, train_ds.indices, get_train_transform(img_size)),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        _TransformedSubset(val_ds.dataset, val_ds.indices, get_val_transform(img_size)),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        _TransformedSubset(test_ds.dataset, test_ds.indices, get_val_transform(img_size)),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


class _TransformedSubset(Dataset):
    """Dataset with transform applied to a subset of indices."""

    def __init__(self, base: RealFakeImageFolder, indices: list[int], transform: Callable):
        self.base = base
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        path, label, _ = self.base.samples[self.indices[idx]]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label
