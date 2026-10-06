"""
tests/test_stream1_prnu.py

Unit tests for Stream 1: PRNU denoiser, extractor, fingerprint, encoder.
"""

import pytest
import numpy as np
import torch
from PIL import Image

from src.stream1_prnu.denoiser import DnCNN, DnCNNWrapper
from src.stream1_prnu.extractor import PRNUExtractor, pil_to_tensor, tensor_to_gray
from src.stream1_prnu.encoder import PRNUEncoder, Stream1


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_random_pil(h=128, w=128):
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def make_random_tensor(b=2, c=1, h=64, w=64):
    return torch.rand(b, c, h, w)


# ──────────────────────────────────────────────────────────────────────────────
# DnCNN
# ──────────────────────────────────────────────────────────────────────────────

class TestDnCNN:
    def test_forward_shape(self):
        model = DnCNN(num_channels=1)
        x = torch.rand(2, 1, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)

    def test_output_clamped(self):
        wrapper = DnCNNWrapper()
        x = torch.rand(1, 64, 64)
        out = wrapper.denoise(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_residual_shape(self):
        wrapper = DnCNNWrapper()
        x = torch.rand(1, 1, 48, 48)
        residual = wrapper.noise_residual(x)
        assert residual.shape == x.shape

    def test_batch_denoise(self):
        wrapper = DnCNNWrapper()
        x = torch.rand(4, 1, 32, 32)
        out = wrapper.denoise(x)
        assert out.shape == (4, 1, 32, 32)


# ──────────────────────────────────────────────────────────────────────────────
# PRNUExtractor
# ──────────────────────────────────────────────────────────────────────────────

class TestPRNUExtractor:
    def test_extract_residual_pil(self):
        extractor = PRNUExtractor()
        img = make_random_pil(96, 96)
        residual = extractor.extract_residual(img)
        assert residual.shape == (1, 96, 96)
        assert residual.dtype == torch.float32

    def test_extract_residual_tensor(self):
        extractor = PRNUExtractor()
        t = torch.rand(3, 80, 80)
        residual = extractor.extract_residual(t)
        assert residual.shape == (1, 80, 80)

    def test_extract_features_keys(self):
        extractor = PRNUExtractor()
        img = make_random_pil(64, 64)
        feats = extractor.extract_features(img)
        assert "mean" in feats
        assert "kurtosis" in feats
        assert "psd_mean" in feats

    def test_correlation_same_image(self):
        extractor = PRNUExtractor()
        img = make_random_pil(64, 64)
        r = extractor.extract_residual(img)
        corr = extractor.compute_correlation(r, r)
        assert abs(corr - 1.0) < 1e-4

    def test_correlation_different_images(self):
        extractor = PRNUExtractor()
        r1 = extractor.extract_residual(make_random_pil(64, 64))
        r2 = extractor.extract_residual(make_random_pil(64, 64))
        corr = extractor.compute_correlation(r1, r2)
        assert -1.1 < corr < 1.1


# ──────────────────────────────────────────────────────────────────────────────
# PRNUEncoder
# ──────────────────────────────────────────────────────────────────────────────

class TestPRNUEncoder:
    def test_output_dim(self):
        encoder = PRNUEncoder(out_dim=128)
        x = torch.rand(4, 1, 64, 64)
        out = encoder(x)
        assert out.shape == (4, 128)

    def test_gradient_flows(self):
        encoder = PRNUEncoder(out_dim=128)
        x = torch.rand(2, 1, 64, 64, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_stream1_without_fingerprint(self):
        stream = Stream1(out_dim=128)
        residual = torch.rand(3, 1, 64, 64)
        out = stream(residual)
        assert out.shape == (3, 128)

    def test_stream1_with_fingerprint(self):
        fp = torch.rand(1, 64, 64)
        stream = Stream1(out_dim=128, fingerprint_tensor=fp)
        residual = torch.rand(2, 1, 64, 64)
        out = stream(residual)
        assert out.shape == (2, 128)
