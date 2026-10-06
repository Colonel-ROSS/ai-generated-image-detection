"""
tests/test_fusion.py

Unit tests for the fusion network.
"""

import pytest
import torch
from src.fusion.network import AIImageDetector, FusionHead, build_ablation_model


class TestFusionHead:
    def test_output_shape(self):
        head = FusionHead(in_dim=448, hidden_dim=256, dropout=0.0)
        x = torch.rand(4, 448)
        out = head(x)
        assert out.shape == (4, 1)

    def test_output_in_01(self):
        head = FusionHead(in_dim=448, hidden_dim=256, dropout=0.0)
        head.eval()
        x = torch.rand(8, 448)
        out = head(x)
        assert (out >= 0).all() and (out <= 1).all()


class TestAIImageDetector:
    def _make_batch(self, B=2, img_size=32):
        image = torch.rand(B, 3, img_size, img_size)
        residual = torch.rand(B, 1, img_size, img_size)
        return image, residual

    def test_full_model_output_shape(self):
        model = AIImageDetector(
            enable_stream1=True,
            enable_stream2=True,
            enable_stream3=True,
            spectral_size=32,
        )
        model.eval()
        image, residual = self._make_batch(2, 64)
        out = model(image, residual=residual)
        assert out.shape == (2, 1)
        assert (out >= 0).all() and (out <= 1).all()

    def test_stream2_only(self):
        model = AIImageDetector(
            enable_stream1=False,
            enable_stream2=True,
            enable_stream3=False,
            spectral_size=32,
        )
        model.eval()
        image = torch.rand(2, 3, 64, 64)
        out = model(image, residual=None)
        assert out.shape == (2, 1)

    def test_stream3_only(self):
        model = AIImageDetector(
            enable_stream1=False,
            enable_stream2=False,
            enable_stream3=True,
        )
        model.eval()
        image = torch.rand(2, 3, 64, 64)
        out = model(image, residual=None)
        assert out.shape == (2, 1)

    def test_stream1_only(self):
        model = AIImageDetector(
            enable_stream1=True,
            enable_stream2=False,
            enable_stream3=False,
            spectral_size=32,
        )
        model.eval()
        image = torch.rand(2, 3, 64, 64)
        residual = torch.rand(2, 1, 64, 64)
        out = model(image, residual=residual)
        assert out.shape == (2, 1)

    def test_stream1_requires_residual(self):
        model = AIImageDetector(enable_stream1=True, enable_stream2=False, enable_stream3=False)
        image = torch.rand(2, 3, 64, 64)
        with pytest.raises(ValueError):
            model(image, residual=None)

    def test_forward_features(self):
        model = AIImageDetector(
            enable_stream1=True,
            enable_stream2=True,
            enable_stream3=True,
            spectral_size=32,
        )
        model.eval()
        image, residual = self._make_batch(2, 64)
        feat_dict = model.forward_features(image, residual=residual)
        assert "stream1" in feat_dict
        assert "stream2" in feat_dict
        assert "stream3" in feat_dict
        assert "probability" in feat_dict
        assert feat_dict["stream1"].shape == (2, 128)
        assert feat_dict["stream2"].shape == (2, 256)
        assert feat_dict["stream3"].shape == (2, 64)
        assert feat_dict["probability"].shape == (2, 1)

    def test_gradient_flows_full_model(self):
        model = AIImageDetector(
            enable_stream1=False,
            enable_stream2=True,
            enable_stream3=True,
            spectral_size=32,
        )
        image = torch.rand(2, 3, 64, 64)
        out = model(image, residual=None)
        out.sum().backward()
        # Check gradients exist in mask
        assert model.stream2.masked_spectral.mask.raw_mask.grad is not None

    def test_ablation_model_12(self):
        model = build_ablation_model("12", spectral_size=32)
        model.eval()
        image = torch.rand(2, 3, 64, 64)
        residual = torch.rand(2, 1, 64, 64)
        out = model(image, residual=residual)
        assert out.shape == (2, 1)

    def test_ablation_model_23(self):
        model = build_ablation_model("23", spectral_size=32)
        model.eval()
        image = torch.rand(2, 3, 64, 64)
        out = model(image)
        assert out.shape == (2, 1)

    def test_all_ablation_configs(self):
        configs = ["1", "2", "3", "12", "13", "23", "123"]
        for config in configs:
            model = build_ablation_model(config, spectral_size=16)
            model.eval()
            image = torch.rand(2, 3, 32, 32)
            residual = torch.rand(2, 1, 32, 32)
            if "1" in config:
                out = model(image, residual=residual)
            else:
                out = model(image)
            assert out.shape == (2, 1), f"Failed for config={config}"
