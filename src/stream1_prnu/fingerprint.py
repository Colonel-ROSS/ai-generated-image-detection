"""
stream1_prnu/fingerprint.py

Reference PRNU fingerprint builder from RAISE-1k RAW images.

RAISE-1k is used EXCLUSIVELY here - never for training labels.
The fingerprint is the average noise residual across all reference images.
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Iterable
from PIL import Image
import json
from datetime import datetime

try:
    import rawpy
    RAWPY_AVAILABLE = True
except ImportError:
    RAWPY_AVAILABLE = False

from .extractor import PRNUExtractor


class FingerprintBuilder:
    """
    Builds a camera PRNU reference fingerprint from a set of RAW/TIFF images.

    The fingerprint K is:
        K = (1/N) * sum_i W_i
    where W_i is the noise residual of image i.

    All images are processed at full resolution (no resize).
    The fingerprint is stored as a fixed-size 2D numpy array.
    """

    def __init__(
        self,
        extractor: PRNUExtractor | None = None,
        target_size: tuple[int, int] = (512, 512),
        fingerprint_path: str | Path | None = None,
    ):
        self.extractor = extractor or PRNUExtractor()
        self.target_size = target_size  # (H, W) for centre-crop of residuals
        self.fingerprint: np.ndarray | None = None
        self.n_images: int = 0

        self.fingerprint_path = Path(fingerprint_path) if fingerprint_path else \
            Path(__file__).resolve().parents[3] / "features" / "stream1" / "prnu_fingerprint.npy"

        if self.fingerprint_path.exists():
            self._load()

    # ------------------------------------------------------------------
    def build(self, image_paths: Iterable[str | Path], save: bool = True) -> np.ndarray:
        """
        Build the fingerprint from a list of image file paths.
        Accepts RAW (.NEF, .CR2, .ARW), TIFF, and JPEG.
        """
        accumulator: np.ndarray | None = None
        count = 0

        for path in image_paths:
            path = Path(path)
            try:
                img = self._load_image(path)
                residual = self.extractor.extract_residual(img)  # full-res, no resize
                residual_np = residual.squeeze().cpu().numpy()

                # Centre-crop to target_size for accumulation
                cropped = self._centre_crop(residual_np, self.target_size)

                if accumulator is None:
                    accumulator = np.zeros_like(cropped, dtype=np.float64)
                accumulator += cropped
                count += 1

                if count % 50 == 0:
                    print(f"[Fingerprint] Processed {count} images...")

            except Exception as e:
                print(f"[Fingerprint] Skipping {path.name}: {e}")
                continue

        if count == 0:
            raise RuntimeError("No images were successfully processed.")

        self.fingerprint = (accumulator / count).astype(np.float32)
        self.n_images = count

        if save:
            self._save()

        print(f"[Fingerprint] Built from {count} images. Shape: {self.fingerprint.shape}")
        return self.fingerprint

    # ------------------------------------------------------------------
    def get(self) -> np.ndarray | None:
        return self.fingerprint

    def as_tensor(self) -> torch.Tensor | None:
        if self.fingerprint is None:
            return None
        return torch.from_numpy(self.fingerprint).unsqueeze(0)  # [1, H, W]

    def correlate(self, residual: torch.Tensor) -> float:
        """PCE-style correlation of a residual with the stored fingerprint."""
        if self.fingerprint is None:
            raise RuntimeError("No fingerprint loaded. Call build() first.")
        ref = self.as_tensor()
        from .extractor import PRNUExtractor
        e = PRNUExtractor()
        return e.compute_correlation(residual, ref)

    # ------------------------------------------------------------------
    def _load_image(self, path: Path) -> Image.Image:
        suffix = path.suffix.lower()
        if suffix in (".nef", ".cr2", ".arw", ".dng", ".raw"):
            if not RAWPY_AVAILABLE:
                raise ImportError("rawpy not installed; cannot read RAW files.")
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
            return Image.fromarray(rgb)
        else:
            return Image.open(path).convert("RGB")

    def _centre_crop(self, arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        H, W = arr.shape
        th, tw = size
        if H < th or W < tw:
            # Pad with zeros if image is smaller than target
            out = np.zeros(size, dtype=arr.dtype)
            h_ = min(H, th)
            w_ = min(W, tw)
            out[:h_, :w_] = arr[:h_, :w_]
            return out
        top = (H - th) // 2
        left = (W - tw) // 2
        return arr[top:top + th, left:left + tw]

    def _save(self):
        self.fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(self.fingerprint_path), self.fingerprint)
        meta = {
            "n_images": self.n_images,
            "target_size": list(self.target_size),
            "built_at": datetime.utcnow().isoformat(),
        }
        meta_path = self.fingerprint_path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[Fingerprint] Saved to {self.fingerprint_path}")

    def _load(self):
        self.fingerprint = np.load(str(self.fingerprint_path)).astype(np.float32)
        meta_path = self.fingerprint_path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.n_images = meta.get("n_images", 0)
        print(f"[Fingerprint] Loaded from {self.fingerprint_path} ({self.n_images} images)")
