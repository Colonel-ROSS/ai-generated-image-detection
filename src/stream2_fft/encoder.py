"""
stream2_fft/encoder.py

ResNet-18 encoder for the FFT spectral stream.
Input: [B, 1, 224, 224] masked spectral map
Output: [B, 256] feature vector
"""

import torch
import torch.nn as nn
import torchvision.models as models
from .transform import SpectralTransform
from .mask import MaskedSpectralModule


class FFTEncoder(nn.Module):
    """
    ResNet-18 adapted for single-channel spectral input -> 256-dim feature vector.

    The first conv layer is replaced to accept 1 channel instead of 3.
    The final FC layer outputs 256-dim.
    """

    def __init__(self, out_dim: int = 256, pretrained: bool = False):
        super().__init__()
        self.out_dim = out_dim

        # Load ResNet-18 backbone
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Replace first conv: 3-channel -> 1-channel
        resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Replace final FC: 512 -> out_dim
        resnet.fc = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.ReLU(inplace=True),
        )

        self.backbone = resnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, H, W] masked spectral map
        Returns: [B, 256]
        """
        return self.backbone(x)


class Stream2(nn.Module):
    """
    Full Stream 2 pipeline:
        [B, C, H, W] image -> SpectralTransform -> MaskedSpectralModule -> FFTEncoder -> [B, 256]

    The full pipeline from raw image to 256-dim vector.
    """

    def __init__(
        self,
        out_dim: int = 256,
        spectral_size: int = 224,
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        self.spectral_transform = SpectralTransform(shift=True)
        self.masked_spectral = MaskedSpectralModule(h=spectral_size, w=spectral_size)
        self.encoder = FFTEncoder(out_dim=out_dim, pretrained=pretrained_backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] input image (RGB or grayscale), float32 in [0,1]
        Returns: [B, 256]
        """
        spectral = self.spectral_transform(x)    # [B, 1, H, W]
        masked = self.masked_spectral(spectral)  # [B, 1, 224, 224]
        feat = self.encoder(masked)              # [B, 256]
        return feat

    def get_mask(self):
        """Return the learnable spectral mask for visualisation."""
        return self.masked_spectral.mask
