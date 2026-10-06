"""
stream3_nss/statistics.py

Natural Scene Statistics (NSS) feature extraction.

Features computed:
1. PSD slope alpha (1/f law fit in log-log space)
2. Noise residual kurtosis
3. Noise residual skewness
4. Gradient statistics (mean, std of gradient magnitude)
5. Wavelet coefficient distributions (std per subband)
6. Spatial autocorrelation lag-1 in x and y

Output: fixed-length numpy feature vector of length 18.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from scipy import stats as scipy_stats
from typing import Union


# Feature vector dimension
NSS_DIM = 18


def _to_gray_numpy(img: Union[Image.Image, np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert various input types to float64 2D grayscale [H, W] in [0, 1].

    Handles three tensor regimes:
      - uint8-as-float (max > 5): divide by 255 then grayscale
      - already-normalized float (max in [-5, 5], e.g. ImageNet range [-2.1, 2.6]):
        convert to grayscale directly, then min-max rescale to [0, 1] to preserve
        all local spatial structure for gradient/wavelet/autocorr features
    """
    if isinstance(img, torch.Tensor):
        t = img.float()
        if t.max() > 5.0:
            # Genuine uint8-as-float tensor: scale to [0, 1] before grayscale
            t = t / 255.0
        # else: already normalized (ImageNet range ≈ [-2.1, 2.6]) — use as-is
        if t.dim() == 3:
            t = TF.rgb_to_grayscale(t).squeeze(0)
        gray = t.cpu().numpy().astype(np.float64)
        # Min-max rescale to [0, 1] to preserve local structure for all NSS features
        g_min, g_max = gray.min(), gray.max()
        if g_max > g_min:
            gray = (gray - g_min) / (g_max - g_min)
        else:
            gray = np.zeros_like(gray)
        return gray
    if isinstance(img, Image.Image):
        return np.array(img.convert("L")).astype(np.float64) / 255.0
    if isinstance(img, np.ndarray):
        if img.ndim == 3:
            img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        return img.astype(np.float64) / (img.max() + 1e-8)
    raise TypeError(f"Unsupported type: {type(img)}")


def _psd_slope(gray: np.ndarray) -> float:
    """Fit alpha in PSD ~ f^(-alpha) via log-log regression on radial profile."""
    F = np.fft.fft2(gray)
    mag = np.abs(np.fft.fftshift(F))
    H, W = mag.shape
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r = np.round(np.sqrt((x - cx) ** 2 + (y - cy) ** 2)).astype(int)
    r_max = min(cy, cx) - 1
    radial = np.array(
        [mag[r == i].mean() if (r == i).any() else 1e-10 for i in range(1, r_max)]
    )
    freqs = np.arange(1, len(radial) + 1, dtype=np.float64)
    valid = radial > 0
    if valid.sum() < 3:
        return 0.0
    coeffs = np.polyfit(np.log(freqs[valid]), np.log(radial[valid]), 1)
    return float(coeffs[0])


def _noise_residual_stats(gray: np.ndarray) -> tuple[float, float]:
    """
    Simple noise residual via local mean subtraction.
    Returns (kurtosis, skewness) of the residual.
    """
    from scipy.ndimage import uniform_filter
    smoothed = uniform_filter(gray, size=3)
    residual = gray - smoothed
    flat = residual.flatten()
    if flat.std() < 1e-10:
        return 0.0, 0.0
    kurt = float(scipy_stats.kurtosis(flat, fisher=True))
    skew = float(scipy_stats.skew(flat))
    return kurt, skew


def _gradient_stats(gray: np.ndarray) -> tuple[float, float, float, float]:
    """Gradient magnitude mean and std, plus anisotropy ratio."""
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mean_mag = float(mag.mean())
    std_mag = float(mag.std())
    mean_gx = float(np.abs(gx).mean())
    mean_gy = float(np.abs(gy).mean())
    aniso = mean_gx / (mean_gy + 1e-8)
    return mean_mag, std_mag, aniso, float(mag.max())


def _wavelet_stats(gray: np.ndarray, levels: int = 3) -> list[float]:
    """
    Compute approximation and detail coefficients via Haar wavelet decomposition.
    Returns std of horizontal, vertical, diagonal subbands at each level.
    """
    stds = []
    img = gray.copy()
    for _ in range(levels):
        H, W = img.shape
        if H < 2 or W < 2:
            stds.extend([0.0, 0.0, 0.0])
            continue
        h_even, h_odd = img[0::2, :], img[1::2, :]
        min_h = min(h_even.shape[0], h_odd.shape[0])
        h_even, h_odd = h_even[:min_h, :], h_odd[:min_h, :]
        # Haar rows
        LL_r = (h_even + h_odd) / 2
        LH_r = (h_even - h_odd) / 2
        # Haar cols on each
        LW, LH2 = LL_r.shape
        min_w = min(LL_r[:, 0::2].shape[1], LL_r[:, 1::2].shape[1])
        LL = (LL_r[:, 0::2][:, :min_w] + LL_r[:, 1::2][:, :min_w]) / 2
        LH = (LH_r[:, 0::2][:, :min_w] - LH_r[:, 1::2][:, :min_w]) / 2
        HL = (LL_r[:, 0::2][:, :min_w] - LL_r[:, 1::2][:, :min_w]) / 2
        HH = (LH_r[:, 0::2][:, :min_w] + LH_r[:, 1::2][:, :min_w]) / 2
        stds.extend([float(LH.std()), float(HL.std()), float(HH.std())])
        img = LL
    return stds


def _spatial_autocorrelation(gray: np.ndarray) -> tuple[float, float]:
    """Lag-1 spatial autocorrelation in row and column directions."""
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return 0.0, 0.0
    flat_x = gray[:, :-1].flatten()
    flat_x1 = gray[:, 1:].flatten()
    flat_y = gray[:-1, :].flatten()
    flat_y1 = gray[1:, :].flatten()

    def safe_corr(a, b):
        if a.std() < 1e-10 or b.std() < 1e-10:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    return safe_corr(flat_x, flat_x1), safe_corr(flat_y, flat_y1)


def extract_nss_features(
    img: Union[Image.Image, np.ndarray, torch.Tensor]
) -> np.ndarray:
    """
    Extract NSS feature vector of length NSS_DIM=18 from an image.

    Features (18 total):
        [0]     psd_slope
        [1]     kurtosis
        [2]     skewness
        [3]     gradient_mean
        [4]     gradient_std
        [5]     gradient_anisotropy
        [6]     gradient_max
        [7-15]  wavelet stds (3 levels x 3 subbands)
        [16]    autocorr_x
        [17]    autocorr_y
    """
    gray = _to_gray_numpy(img)

    psd_slope = _psd_slope(gray)
    kurt, skew = _noise_residual_stats(gray)
    g_mean, g_std, g_aniso, g_max = _gradient_stats(gray)
    wav_stds = _wavelet_stats(gray, levels=3)  # 9 values
    ac_x, ac_y = _spatial_autocorrelation(gray)

    features = np.array(
        [psd_slope, kurt, skew, g_mean, g_std, g_aniso, g_max]
        + wav_stds
        + [ac_x, ac_y],
        dtype=np.float32,
    )

    # Clip extreme values to prevent NaN propagation
    features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
    features = np.clip(features, -100.0, 100.0)

    assert len(features) == NSS_DIM, f"Expected {NSS_DIM}, got {len(features)}"
    return features
