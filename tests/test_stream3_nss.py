"""
tests/test_stream3_nss.py

Unit tests for Stream 3: NSS feature extraction and encoder.
"""

import pytest
import numpy as np
import torch
from PIL import Image

from src.stream3_nss.statistics import extract_nss_features, NSS_DIM
from src.stream3_nss.encoder import NSSEncoder, Stream3


# ──────────────────────────────────────────────────────────────────────────────
# NSS feature extraction
# ──────────────────────────────────────────────────────────────────────────────

class TestNSSStatistics:
    def test_pil_input(self):
        img = Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))
        feats = extract_nss_features(img)
        assert feats.shape == (NSS_DIM,)
        assert feats.dtype == np.float32

    def test_numpy_input(self):
        arr = np.random.rand(64, 64, 3).astype(np.float64)
        feats = extract_nss_features(arr)
        assert feats.shape == (NSS_DIM,)

    def test_tensor_input(self):
        t = torch.rand(3, 64, 64)
        feats = extract_nss_features(t)
        assert feats.shape == (NSS_DIM,)

    def test_no_nans(self):
        img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        feats = extract_nss_features(img)
        assert not np.any(np.isnan(feats))
        assert not np.any(np.isinf(feats))

    def test_small_image(self):
        img = Image.fromarray(np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8))
        feats = extract_nss_features(img)
        assert feats.shape == (NSS_DIM,)

    def test_feature_dim_constant(self):
        assert NSS_DIM == 18


# ──────────────────────────────────────────────────────────────────────────────
# NSSEncoder
# ──────────────────────────────────────────────────────────────────────────────

class TestNSSEncoder:
    def test_output_shape(self):
        encoder = NSSEncoder(in_dim=NSS_DIM, hidden_dim=64, out_dim=64)
        x = torch.rand(4, NSS_DIM)
        out = encoder(x)
        assert out.shape == (4, 64)

    def test_gradient_flows(self):
        encoder = NSSEncoder(in_dim=NSS_DIM, out_dim=64)
        x = torch.rand(2, NSS_DIM, requires_grad=True)
        out = encoder(x)
        out.sum().backward()
        assert x.grad is not None

    def test_batch_size_1(self):
        encoder = NSSEncoder(in_dim=NSS_DIM, out_dim=64)
        # BatchNorm1d requires B>=2 in train mode; switch to eval
        encoder.eval()
        x = torch.rand(1, NSS_DIM)
        out = encoder(x)
        assert out.shape == (1, 64)


# ──────────────────────────────────────────────────────────────────────────────
# Stream3 end-to-end
# ──────────────────────────────────────────────────────────────────────────────

class TestStream3:
    def test_forward_tensor(self):
        stream = Stream3(out_dim=64)
        feats = torch.rand(4, NSS_DIM)
        out = stream.forward_tensor(feats)
        assert out.shape == (4, 64)

    def test_forward_image_batch(self):
        stream = Stream3(out_dim=64)
        stream.eval()
        x = torch.rand(3, 3, 64, 64)
        out = stream(x)
        assert out.shape == (3, 64)

    def test_forward_single_image(self):
        stream = Stream3(out_dim=64)
        stream.eval()
        x = torch.rand(1, 3, 32, 32)
        out = stream(x)
        assert out.shape == (1, 64)
