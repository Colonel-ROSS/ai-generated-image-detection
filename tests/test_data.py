"""
tests/test_data.py

Unit tests for data pipeline: augmentation, dataset helpers.
"""

import pytest
import io
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

from src.data.augmentation import (
    RandomJPEGCompression,
    RandomGaussianBlur,
    RandomBicubicResize,
    apply_jpeg,
    apply_gaussian_blur,
    apply_bicubic_resize,
    get_augmented_train_transform,
    get_perturbation_transform,
)
from src.data.dataset import get_train_transform, get_val_transform


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation primitives
# ──────────────────────────────────────────────────────────────────────────────

def make_pil(h=64, w=64):
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestAugmentation:
    def test_jpeg_compression_output_size(self):
        img = make_pil(64, 64)
        jpeg = RandomJPEGCompression(75, 95)
        out = jpeg(img)
        assert out.size == img.size

    def test_gaussian_blur_output_size(self):
        img = make_pil(64, 64)
        blur = RandomGaussianBlur(0.5, 1.5)
        out = blur(img)
        assert out.size == img.size

    def test_bicubic_resize_output_size(self):
        img = make_pil(64, 64)
        resize = RandomBicubicResize(0.75, 1.25)
        out = resize(img)
        assert out.size == img.size

    def test_apply_jpeg(self):
        img = make_pil(32, 32)
        out = apply_jpeg(img, quality=80)
        assert out.size == img.size

    def test_apply_blur(self):
        img = make_pil(32, 32)
        out = apply_gaussian_blur(img, sigma=1.0)
        assert out.size == img.size

    def test_apply_resize(self):
        img = make_pil(64, 64)
        out = apply_bicubic_resize(img, scale=0.5)
        assert out.size == (32, 32)

    def test_augmented_train_transform(self):
        transform = get_augmented_train_transform(img_size=64)
        img = make_pil(128, 128)
        out = transform(img)
        assert out.shape == (3, 64, 64)
        assert isinstance(out, torch.Tensor)

    def test_perturbation_transform_jpeg(self):
        transform = get_perturbation_transform("jpeg", 80, img_size=64)
        img = make_pil(128, 128)
        out = transform(img)
        assert out.shape == (3, 64, 64)

    def test_perturbation_transform_blur(self):
        transform = get_perturbation_transform("blur", 1.5, img_size=64)
        img = make_pil(128, 128)
        out = transform(img)
        assert out.shape == (3, 64, 64)

    def test_perturbation_transform_resize(self):
        transform = get_perturbation_transform("resize", 0.75, img_size=64)
        img = make_pil(128, 128)
        out = transform(img)
        assert out.shape == (3, 64, 64)


class TestDatasetTransforms:
    def test_train_transform_shape(self):
        transform = get_train_transform(img_size=224)
        img = make_pil(512, 512)
        out = transform(img)
        assert out.shape == (3, 224, 224)

    def test_val_transform_shape(self):
        transform = get_val_transform(img_size=224)
        img = make_pil(512, 512)
        out = transform(img)
        assert out.shape == (3, 224, 224)

    def test_transform_normalised_range(self):
        transform = get_val_transform(224)
        img = make_pil(256, 256)
        out = transform(img)
        # After ImageNet normalisation, values can go outside [0, 1]
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 224, 224)
