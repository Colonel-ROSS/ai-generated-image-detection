"""
stream3_nss/encoder.py

MLP encoder: NSS feature vector (18-dim) -> 64-dim feature vector.

Architecture: FC(18->128) -> BN -> ReLU -> Dropout(0.3) -> FC(128->64) -> ReLU
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Union

from .statistics import extract_nss_features, NSS_DIM


class NSSEncoder(nn.Module):
    """
    Compact 2-layer MLP that maps an 18-dim NSS feature vector to 64-dim.
    """

    def __init__(self, in_dim: int = NSS_DIM, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_dim] NSS feature vector
        Returns: [B, 64]
        """
        return self.net(x)


class Stream3(nn.Module):
    """
    Full Stream 3 pipeline: image -> NSS features -> NSSEncoder -> 64-dim vector.

    Feature extraction is performed on-the-fly for each image in the batch.
    For large-scale training, pre-extract features to .npy files instead.
    """

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.encoder = NSSEncoder(in_dim=NSS_DIM, out_dim=out_dim)
        self.out_dim = out_dim

    def forward_tensor(self, features: torch.Tensor) -> torch.Tensor:
        """
        Accept pre-extracted [B, NSS_DIM] feature tensor.
        Returns: [B, 64]
        """
        return self.encoder(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] image batch, float32 in [0, 1]
        Extracts NSS features on CPU then encodes.
        Returns: [B, 64]
        """
        B = x.shape[0]
        feats_list = []
        x_cpu = x.detach().cpu()
        for i in range(B):
            nss = extract_nss_features(x_cpu[i])  # [18]
            feats_list.append(nss)
        feats_np = np.stack(feats_list, axis=0)  # [B, 18]
        feats_t = torch.from_numpy(feats_np).float().to(x.device)
        return self.encoder(feats_t)  # [B, 64]
