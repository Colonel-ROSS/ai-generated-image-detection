"""
data/augmentation.py

Augmentation pipeline for robustness training:
- JPEG compression (Q=65-95)
- Gaussian blur (sigma=0.1-2.0)
- Bicubic resize (0.75x - 1.25x)

Also provides perturbation functions for the robustness test matrix.
"""

from __future__ import annotations

import io
import random
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter


# ──────────────────────────────────────────────────────────────────────────────
# Primitive augmentations
# ──────────────────────────────────────────────────────────────────────────────

class RandomJPEGCompression:
    """Apply JPEG compression at a random quality in [q_min, q_max]."""

    def __init__(self, q_min: int = 65, q_max: int = 95):
        self.q_min = q_min
        self.q_max = q_max

    def __call__(self, img: Image.Image) -> Image.Image:
        q = random.randint(self.q_min, self.q_max)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).copy()


class RandomGaussianBlur:
    """Apply Gaussian blur with sigma uniformly sampled from [sigma_min, sigma_max]."""

    def __init__(self, sigma_min: float = 0.1, sigma_max: float = 2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img: Image.Image) -> Image.Image:
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))


class RandomBicubicResize:
    """Randomly resize by a scale factor and then back to original size."""

    def __init__(self, scale_min: float = 0.75, scale_max: float = 1.25):
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = random.uniform(self.scale_min, self.scale_max)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.BICUBIC)
        return img.resize((w, h), Image.BICUBIC)


# ──────────────────────────────────────────────────────────────────────────────
# Training augmentation pipeline
# ──────────────────────────────────────────────────────────────────────────────

def get_augmented_train_transform(
    img_size: int = 224,
    jpeg_prob: float = 0.5,
    blur_prob: float = 0.3,
    resize_prob: float = 0.3,
) -> T.Compose:
    """
    Augmentation pipeline for training robustness.
    Applies JPEG, blur, and bicubic resize with given probabilities.
    """
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.RandomApply([RandomJPEGCompression(65, 95)], p=jpeg_prob),
        T.RandomApply([RandomGaussianBlur(0.1, 2.0)], p=blur_prob),
        T.RandomApply([RandomBicubicResize(0.75, 1.25)], p=resize_prob),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Robustness perturbation functions (for the test matrix)
# ──────────────────────────────────────────────────────────────────────────────

def apply_jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def apply_gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_bicubic_resize(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.BICUBIC)


def get_perturbation_transform(
    perturbation_type: str,
    value: float | int,
    img_size: int = 224,
) -> T.Compose:
    """
    Build a transform that applies a specific perturbation at a fixed level.
    Used for the robustness sweep.

    perturbation_type: 'jpeg', 'blur', 'resize'
    value: quality for jpeg; sigma for blur; scale for resize
    """
    pil_transforms = []
    if perturbation_type == "jpeg":
        pil_transforms.append(lambda img: apply_jpeg(img, int(value)))
    elif perturbation_type == "blur":
        pil_transforms.append(lambda img: apply_gaussian_blur(img, float(value)))
    elif perturbation_type == "resize":
        pil_transforms.append(lambda img: apply_bicubic_resize(img, float(value)))

    steps = (
        [T.Resize((img_size, img_size))]
        + [T.Lambda(fn) for fn in pil_transforms]
        + [T.Resize((img_size, img_size))]  # ensure consistent output size after perturbation
        + [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    return T.Compose(steps)
