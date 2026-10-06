"""
stream1_prnu/extractor.py

PRNU noise residual extraction.

CRITICAL: PRNU extraction happens BEFORE any resize operation.
Call extract_residual() on the full-resolution image.
"""

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
from typing import Union

from .denoiser import DnCNNWrapper


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to float32 tensor [C, H, W] in [0, 1]."""
    return TF.to_tensor(img.convert("RGB"))


def tensor_to_gray(t: torch.Tensor) -> torch.Tensor:
    """[C, H, W] RGB -> [1, H, W] grayscale."""
    return TF.rgb_to_grayscale(t)


class PRNUExtractor:
    """
    Extracts per-image noise residuals using a DnCNN denoiser.

    Usage:
        extractor = PRNUExtractor(denoiser=DnCNNWrapper())
        residual = extractor.extract_residual(pil_image)   # [1, H, W] float32
        features = extractor.extract_features(pil_image)   # dict of scalar stats
    """

    def __init__(self, denoiser: DnCNNWrapper | None = None):
        self.denoiser = denoiser or DnCNNWrapper()

    def extract_residual(self, img: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Compute noise residual W = Y - F(Y) on full-resolution image.
        Input: PIL Image or [C, H, W] float32 tensor.
        Output: [1, H, W] float32 residual (grayscale).

        IMPORTANT: Do NOT resize before calling this method.
        """
        if isinstance(img, Image.Image):
            t = pil_to_tensor(img)
        else:
            t = img.float()
            if t.max() > 1.0:
                t = t / 255.0

        gray = tensor_to_gray(t)          # [1, H, W]
        residual = self.denoiser.noise_residual(gray)   # [1, H, W]
        return residual

    def extract_features(self, img: Union[Image.Image, torch.Tensor]) -> dict:
        """
        Compute statistical features of the noise residual:
        - spatial_coherence: mean absolute value of residual
        - psd_peak: peak of the power spectral density
        - psd_mean: mean PSD
        - mean, std, kurtosis, skewness of residual
        """
        residual = self.extract_residual(img)
        w = residual.squeeze().cpu().numpy()

        # Basic statistics
        mean_val = float(np.mean(w))
        std_val = float(np.std(w))
        spatial_coherence = float(np.mean(np.abs(w)))

        # 4th and 3rd standardised moments
        if std_val > 1e-8:
            kurt = float(np.mean(((w - mean_val) / std_val) ** 4) - 3.0)
            skew = float(np.mean(((w - mean_val) / std_val) ** 3))
        else:
            kurt, skew = 0.0, 0.0

        # Power spectral density
        F = np.fft.fft2(w)
        psd = np.abs(F) ** 2
        psd_peak = float(np.max(psd))
        psd_mean = float(np.mean(psd))

        return {
            "mean": mean_val,
            "std": std_val,
            "kurtosis": kurt,
            "skewness": skew,
            "spatial_coherence": spatial_coherence,
            "psd_peak": psd_peak,
            "psd_mean": psd_mean,
        }

    def compute_correlation(
        self, residual: torch.Tensor, reference: torch.Tensor
    ) -> float:
        """
        Pearson correlation between a noise residual and a reference PRNU fingerprint.
        Inputs: [1, H, W] tensors (will be cropped to smaller if sizes differ).
        """
        w = residual.squeeze().cpu().numpy()
        r = reference.squeeze().cpu().numpy()

        # Crop to common size
        h = min(w.shape[0], r.shape[0])
        ww = min(w.shape[1], r.shape[1])
        w = w[:h, :ww]
        r = r[:h, :ww]

        w_flat = w.flatten()
        r_flat = r.flatten()

        if w_flat.std() < 1e-8 or r_flat.std() < 1e-8:
            return 0.0

        corr = float(np.corrcoef(w_flat, r_flat)[0, 1])
        return corr
