"""
stream2_fft/transform.py

2D FFT magnitude spectrum with log-compression.

Output: log-compressed spectral map S(u,v) = log(1 + |F(u,v)|)
with an optional learnable mask M(u,v) applied as element-wise multiplication.
"""

import torch
import torch.nn as nn
import numpy as np


def compute_fft_magnitude(x: torch.Tensor, shift: bool = True) -> torch.Tensor:
    """
    Compute 2D FFT magnitude of a grayscale image tensor.

    x: [B, 1, H, W] float32 in [0, 1]
    Returns: [B, 1, H, W] magnitude spectrum (not log-compressed)
    """
    # rfft2 returns complex tensor; take abs for magnitude
    F = torch.fft.rfft2(x, norm="ortho")
    mag = torch.abs(F)  # [B, 1, H, W//2+1]

    # Reconstruct full-size symmetric magnitude via irfft trick - use full fft2
    F_full = torch.fft.fft2(x, norm="ortho")
    mag_full = torch.abs(F_full)  # [B, 1, H, W]

    if shift:
        # Shift zero frequency to centre
        mag_full = torch.fft.fftshift(mag_full, dim=(-2, -1))

    return mag_full


def log_compress(mag: torch.Tensor) -> torch.Tensor:
    """Apply log(1 + |F|) compression."""
    return torch.log1p(mag)


class SpectralTransform(nn.Module):
    """
    Converts a batch of grayscale images to log-compressed FFT magnitude maps.

    Pipeline:
        [B, 1, H, W] -> FFT2 -> |·| -> log(1+|·|) -> [B, 1, H, W]
    """

    def __init__(self, shift: bool = True):
        super().__init__()
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, H, W] grayscale float32 in [0, 1]
        Returns: [B, 1, H, W] log-compressed magnitude
        """
        if x.shape[1] == 3:
            # Convert RGB to grayscale: 0.299R + 0.587G + 0.114B
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        mag = compute_fft_magnitude(x, shift=self.shift)
        return log_compress(mag)


def extract_spectral_features_numpy(img_gray: np.ndarray) -> dict:
    """
    Extract scalar spectral features from a 2D grayscale numpy array [H, W].
    Useful for NSS-style analysis.
    """
    F = np.fft.fft2(img_gray)
    F_shifted = np.fft.fftshift(F)
    mag = np.abs(F_shifted)
    log_mag = np.log1p(mag)

    H, W = mag.shape
    cy, cx = H // 2, W // 2

    # Azimuthal average (radial profile)
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    r_max = min(cy, cx)
    radial_mean = np.array(
        [mag[r == i].mean() if (r == i).any() else 0 for i in range(r_max)]
    )

    # PSD slope (fit log-log line)
    eps = 1e-8
    freqs = np.arange(1, len(radial_mean))
    vals = radial_mean[1:] + eps
    if len(freqs) > 2 and vals.min() > 0:
        log_freqs = np.log(freqs)
        log_vals = np.log(vals)
        coeffs = np.polyfit(log_freqs, log_vals, 1)
        psd_slope = float(coeffs[0])
    else:
        psd_slope = 0.0

    return {
        "psd_slope": psd_slope,
        "log_mag_mean": float(log_mag.mean()),
        "log_mag_std": float(log_mag.std()),
        "mag_peak": float(mag.max()),
        "mag_mean": float(mag.mean()),
    }
