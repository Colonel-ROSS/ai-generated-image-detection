"""
fusion/network.py

Three-stream fusion detector.

Stream 1 (PRNU):    128-dim
Stream 2 (FFT):     256-dim
Stream 3 (NSS):      64-dim
Concatenated:        448-dim

Fusion head: FC(448->256) -> ReLU -> Dropout(0.5) -> FC(256->1) -> Sigmoid
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.stream1_prnu.encoder import Stream1
from src.stream2_fft.encoder import Stream2
from src.stream3_nss.encoder import Stream3


class FusionHead(nn.Module):
    """
    Fusion MLP: [B, 448] -> [B, 1] probability.
    """

    def __init__(
        self,
        in_dim: int = 448,
        hidden_dim: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_dim] -> [B, 1]"""
        return self.net(x)


class AIImageDetector(nn.Module):
    """
    Full 3-stream AI image detector.

    Inputs:
        image:    [B, C, H, W] float32 in [0, 1]  — for streams 2 and 3
        residual: [B, 1, H, W] float32             — pre-extracted PRNU residual (stream 1)
                  or None to skip stream 1

    Output: [B, 1] fake probability in (0, 1)
    """

    S1_DIM = 128
    S2_DIM = 256
    S3_DIM = 64
    FUSED_DIM = S1_DIM + S2_DIM + S3_DIM  # 448

    def __init__(
        self,
        enable_stream1: bool = True,
        enable_stream2: bool = True,
        enable_stream3: bool = True,
        spectral_size: int = 224,
        fusion_hidden: int = 256,
        dropout: float = 0.5,
        pretrained_backbone: bool = False,
        fingerprint_tensor: torch.Tensor | None = None,
    ):
        super().__init__()
        self.enable_s1 = enable_stream1
        self.enable_s2 = enable_stream2
        self.enable_s3 = enable_stream3

        active_dim = 0

        if enable_stream1:
            self.stream1 = Stream1(
                out_dim=self.S1_DIM,
                fingerprint_tensor=fingerprint_tensor,
            )
            active_dim += self.S1_DIM

        if enable_stream2:
            self.stream2 = Stream2(
                out_dim=self.S2_DIM,
                spectral_size=spectral_size,
                pretrained_backbone=pretrained_backbone,
            )
            active_dim += self.S2_DIM

        if enable_stream3:
            self.stream3 = Stream3(out_dim=self.S3_DIM)
            active_dim += self.S3_DIM

        self.fusion = FusionHead(
            in_dim=active_dim,
            hidden_dim=fusion_hidden,
            dropout=dropout,
        )
        self.active_dim = active_dim

    def forward(
        self,
        image: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        image:    [B, C, H, W] — full-res image for streams 2 and 3
        residual: [B, 1, H, W] — pre-extracted noise residual for stream 1

        Returns: [B, 1] fake probability
        """
        parts = []

        if self.enable_s1:
            if residual is None:
                raise ValueError("Stream 1 is enabled but residual=None was passed.")
            f1 = self.stream1(residual)  # [B, 128]
            parts.append(f1)

        if self.enable_s2:
            f2 = self.stream2(image)     # [B, 256]
            parts.append(f2)

        if self.enable_s3:
            f3 = self.stream3(image)     # [B, 64]
            parts.append(f3)

        fused = torch.cat(parts, dim=1)  # [B, active_dim]
        return self.fusion(fused)        # [B, 1]

    def forward_features(
        self,
        image: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Return per-stream feature vectors (for Gradio per-stream confidence).
        """
        out = {}
        parts = []

        if self.enable_s1:
            if residual is None:
                raise ValueError("Stream 1 is enabled but residual=None.")
            f1 = self.stream1(residual)
            out["stream1"] = f1
            parts.append(f1)

        if self.enable_s2:
            f2 = self.stream2(image)
            out["stream2"] = f2
            parts.append(f2)

        if self.enable_s3:
            f3 = self.stream3(image)
            out["stream3"] = f3
            parts.append(f3)

        fused = torch.cat(parts, dim=1)
        prob = self.fusion(fused)
        out["probability"] = prob
        return out


def build_ablation_model(streams: str = "123", **kwargs) -> AIImageDetector:
    """
    Build an ablation model with the specified streams enabled.
    streams: string containing '1', '2', '3' for each enabled stream.
    E.g. '12' enables streams 1 and 2 only.
    """
    return AIImageDetector(
        enable_stream1="1" in streams,
        enable_stream2="2" in streams,
        enable_stream3="3" in streams,
        **kwargs,
    )
