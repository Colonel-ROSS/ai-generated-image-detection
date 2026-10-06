"""
stream1_prnu/encoder.py

2-layer CNN encoder: noise residual [1, H, W] -> 128-dim feature vector.
Accepts variable spatial resolution; global average pooling before FC layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PRNUEncoder(nn.Module):
    """
    Lightweight 2-layer CNN that maps a single-channel noise residual to a
    128-dimensional feature vector.

    Architecture:
        Conv(1->32, 3x3, BN, ReLU) -> MaxPool(2)
        Conv(32->64, 3x3, BN, ReLU) -> MaxPool(2)
        GlobalAvgPool
        FC(64 -> 128) -> ReLU
    """

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.out_dim = out_dim

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # -> [B, 64, 1, 1]
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, H, W] noise residual
        Returns: [B, 128] feature vector
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.global_pool(x)
        x = self.fc(x)
        return x


class Stream1(nn.Module):
    """
    Full Stream 1 pipeline: residual -> PRNUEncoder -> 128-dim vector.
    In practice the residual computation happens outside this module
    (in PRNUExtractor) because it must occur BEFORE any resize.
    This module accepts the pre-computed [B, 1, H, W] residual tensor.
    """

    def __init__(self, out_dim: int = 128, fingerprint_tensor: torch.Tensor | None = None):
        super().__init__()
        self.encoder = PRNUEncoder(out_dim=out_dim)

        # Optional learnable correlation weight with fingerprint
        if fingerprint_tensor is not None:
            self.register_buffer("fingerprint", fingerprint_tensor.float())
            self.use_fingerprint = True
        else:
            self.fingerprint = None
            self.use_fingerprint = False

        # Optional correlation scalar -> extra feature
        if self.use_fingerprint:
            self.corr_fc = nn.Linear(out_dim + 1, out_dim)
        else:
            self.corr_fc = None

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        """
        residual: [B, 1, H, W] pre-extracted noise residual
        Returns: [B, 128]
        """
        feat = self.encoder(residual)  # [B, 128]

        if self.use_fingerprint and self.fingerprint is not None:
            # Compute batch-wise correlation with fingerprint
            B = residual.shape[0]
            fp = self.fingerprint.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 1, H, W]
            # Crop to common spatial size
            h = min(residual.shape[2], fp.shape[2])
            w = min(residual.shape[3], fp.shape[3])
            r_ = residual[:, :, :h, :w].reshape(B, -1)
            f_ = fp[:, :, :h, :w].reshape(B, -1)
            r_norm = F.normalize(r_, dim=1)
            f_norm = F.normalize(f_, dim=1)
            corr = (r_norm * f_norm).sum(dim=1, keepdim=True)  # [B, 1]
            feat = self.corr_fc(torch.cat([feat, corr], dim=1))

        return feat
