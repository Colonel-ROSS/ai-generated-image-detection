"""
stream2_fft/mask.py

Learnable spectral mask M(u,v) trained end-to-end.

The mask is initialised to ones (identity) and learns to suppress or amplify
specific frequency regions. Target signals:
- CFA demosaicing peaks at ~0.5 cycles/pixel (real cameras)
- GAN upsampling grid artefacts (periodic spikes in fake images)

Inspired by MaskSim (Li et al., CVPR 2024).
"""

import torch
import torch.nn as nn
import numpy as np


class LearnableSpectralMask(nn.Module):
    """
    Element-wise learnable mask applied to the log-compressed FFT magnitude.

    M(u,v) is a [1, 1, H, W] parameter initialised to 1.
    The mask is sigmoid-activated to keep values in (0, 1] and multiplied
    element-wise with the spectral map.

    Usage:
        mask = LearnableSpectralMask(h=224, w=224)
        out = mask(spectral_map)   # [B, 1, 224, 224]
    """

    def __init__(self, h: int = 224, w: int = 224, init: str = "ones"):
        super().__init__()
        self.h = h
        self.w = w

        if init == "ones":
            data = torch.ones(1, 1, h, w)
        elif init == "zeros":
            data = torch.zeros(1, 1, h, w)
        elif init == "random":
            data = torch.randn(1, 1, h, w) * 0.1
        else:
            raise ValueError(f"Unknown init: {init}")

        # Logit of ones = large positive -> sigmoid ~= 1.0
        # We store the raw logit and apply sigmoid in forward
        self.raw_mask = nn.Parameter(data * 5.0 if init == "ones" else data)

    def forward(self, spectral: torch.Tensor) -> torch.Tensor:
        """
        spectral: [B, 1, H, W] log-compressed magnitude
        Returns: [B, 1, H, W] masked magnitude
        """
        B, C, H, W = spectral.shape

        # Resize mask if input size differs (enables variable-res inference)
        if H != self.h or W != self.w:
            mask = torch.sigmoid(
                nn.functional.interpolate(
                    self.raw_mask, size=(H, W), mode="bilinear", align_corners=False
                )
            )
        else:
            mask = torch.sigmoid(self.raw_mask)

        return spectral * mask

    def get_mask_numpy(self) -> np.ndarray:
        """Return the mask as a [H, W] numpy array for visualisation."""
        with torch.no_grad():
            return torch.sigmoid(self.raw_mask).squeeze().cpu().numpy()

    def visualise_cfa_peaks(self) -> np.ndarray:
        """
        Return the mask with CFA peak locations highlighted.
        CFA peaks appear at (0.5, 0), (0, 0.5), (0.5, 0.5) in normalised freq.
        """
        mask_np = self.get_mask_numpy()
        H, W = mask_np.shape
        overlay = mask_np.copy()

        # Mark CFA peak bands
        cy, cx = H // 2, W // 2
        for frac in [0.5]:
            iy = int(cy + frac * H // 2)
            ix = int(cx + frac * W // 2)
            overlay[iy, :] = 1.0
            overlay[:, ix] = 1.0

        return overlay


class MaskedSpectralModule(nn.Module):
    """
    Combines SpectralTransform + LearnableSpectralMask into one module.
    Resizes the spectral map to (h, w) before masking.
    """

    def __init__(self, h: int = 224, w: int = 224):
        super().__init__()
        self.target_h = h
        self.target_w = w
        self.mask = LearnableSpectralMask(h=h, w=w)

    def forward(self, spectral: torch.Tensor) -> torch.Tensor:
        """
        spectral: [B, 1, H, W] log-compressed magnitude (any spatial size)
        Returns: [B, 1, target_h, target_w] masked spectral map
        """
        # Resize to target spatial resolution
        if spectral.shape[-2] != self.target_h or spectral.shape[-1] != self.target_w:
            spectral = nn.functional.interpolate(
                spectral,
                size=(self.target_h, self.target_w),
                mode="bilinear",
                align_corners=False,
            )
        return self.mask(spectral)
