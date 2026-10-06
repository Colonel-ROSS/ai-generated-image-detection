"""
tests/test_stream2_fft.py

Unit tests for Stream 2: FFT transform, learnable mask, encoder.
"""

import pytest
import numpy as np
import torch
from PIL import Image

from src.stream2_fft.transform import (
    compute_fft_magnitude,
    log_compress,
    SpectralTransform,
    extract_spectral_features_numpy,
)
from src.stream2_fft.mask import LearnableSpectralMask, MaskedSpectralModule
from src.stream2_fft.encoder import FFTEncoder, Stream2


# ──────────────────────────────────────────────────────────────────────────────
# SpectralTransform
# ──────────────────────────────────────────────────────────────────────────────

class TestSpectralTransform:
    def test_grayscale_input(self):
        transform = SpectralTransform()
        x = torch.rand(2, 1, 64, 64)
        out = transform(x)
        assert out.shape == (2, 1, 64, 64)
        assert out.min() >= 0

    def test_rgb_input(self):
        transform = SpectralTransform()
        x = torch.rand(3, 3, 64, 64)
        out = transform(x)
        assert out.shape == (3, 1, 64, 64)

    def test_log_compress_nonnegative(self):
        mag = torch.rand(2, 1, 32, 32).abs()
        compressed = log_compress(mag)
        assert (compressed >= 0).all()

    def test_fft_magnitude_shape(self):
        x = torch.rand(4, 1, 32, 32)
        mag = compute_fft_magnitude(x, shift=True)
        assert mag.shape == (4, 1, 32, 32)

    def test_numpy_features(self):
        gray = np.random.rand(64, 64)
        feats = extract_spectral_features_numpy(gray)
        assert "psd_slope" in feats
        assert "log_mag_mean" in feats
        assert isinstance(feats["psd_slope"], float)


# ──────────────────────────────────────────────────────────────────────────────
# LearnableSpectralMask
# ──────────────────────────────────────────────────────────────────────────────

class TestLearnableSpectralMask:
    def test_init_ones(self):
        mask = LearnableSpectralMask(h=32, w=32, init="ones")
        x = torch.rand(2, 1, 32, 32)
        out = mask(x)
        assert out.shape == x.shape
        # With ones init, mask weights ~ 1.0, output ~ input
        assert torch.allclose(out, x, atol=0.01)

    def test_mask_values_in_range(self):
        mask = LearnableSpectralMask(h=32, w=32)
        m = torch.sigmoid(mask.raw_mask)
        assert (m >= 0).all() and (m <= 1).all()

    def test_gradient_flows_through_mask(self):
        mask = LearnableSpectralMask(h=32, w=32)
        x = torch.rand(2, 1, 32, 32)
        out = mask(x)
        out.sum().backward()
        assert mask.raw_mask.grad is not None

    def test_masked_spectral_module(self):
        msm = MaskedSpectralModule(h=64, w=64)
        spectral = torch.rand(2, 1, 128, 128)
        out = msm(spectral)
        assert out.shape == (2, 1, 64, 64)

    def test_get_mask_numpy(self):
        mask = LearnableSpectralMask(h=16, w=16)
        np_mask = mask.get_mask_numpy()
        assert np_mask.shape == (16, 16)
        assert np_mask.min() >= 0 and np_mask.max() <= 1


# ──────────────────────────────────────────────────────────────────────────────
# FFTEncoder / Stream2
# ──────────────────────────────────────────────────────────────────────────────

class TestFFTEncoder:
    def test_output_dim(self):
        encoder = FFTEncoder(out_dim=256, pretrained=False)
        x = torch.rand(2, 1, 224, 224)
        out = encoder(x)
        assert out.shape == (2, 256)

    def test_stream2_end_to_end(self):
        stream = Stream2(out_dim=256, spectral_size=64, pretrained_backbone=False)
        x = torch.rand(2, 3, 128, 128)
        out = stream(x)
        assert out.shape == (2, 256)

    def test_stream2_gradient(self):
        stream = Stream2(out_dim=256, spectral_size=32, pretrained_backbone=False)
        x = torch.rand(2, 3, 64, 64)
        out = stream(x)
        out.sum().backward()
        # Mask should have gradient
        assert stream.masked_spectral.mask.raw_mask.grad is not None

    def test_stream2_grayscale_input(self):
        stream = Stream2(out_dim=256, spectral_size=32, pretrained_backbone=False)
        stream.eval()
        x = torch.rand(1, 1, 64, 64)
        with torch.no_grad():
            out = stream(x)
        assert out.shape == (1, 256)
