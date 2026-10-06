"""
stream1_prnu/denoiser.py

DnCNN denoiser for PRNU noise residual extraction.
Architecture: 17-layer CNN with batch norm, trained for blind Gaussian denoising.
"""

import torch
import torch.nn as nn
from pathlib import Path


class DnCNN(nn.Module):
    """17-layer blind DnCNN (Zhang et al. 2017).

    Architecture matches the pretrained weights in models/pretrained/dncnn.pth:
    flat Conv→ReLU×16 then Conv, all with bias, no BatchNorm, keys model.{0,2,...,32}.
    """

    def __init__(self, num_layers: int = 17, num_features: int = 64, num_channels: int = 1):
        super().__init__()
        layers = [
            nn.Conv2d(num_channels, num_features, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(num_features, num_features, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(num_features, num_channels, 3, padding=1, bias=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns denoised image: x - predicted_noise."""
        return x - self.model(x)


class DnCNNWrapper:
    """
    Loads (or initialises) a DnCNN model and provides a denoise() call.

    If a pretrained .pth is supplied it is loaded; otherwise the model runs
    with random weights (useful for structural testing without weights).
    """

    def __init__(
        self,
        weights_path: str | None = None,
        device: str | None = None,
        num_channels: int = 1,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = DnCNN(num_channels=num_channels).to(self.device)
        self.model.eval()

        # Auto-detect pretrained weights if none specified
        if weights_path is None:
            _default = Path(__file__).resolve().parents[2] / "models" / "pretrained" / "dncnn.pth"
            if _default.exists():
                weights_path = str(_default)

        if weights_path is not None:
            path = Path(weights_path)
            if path.exists():
                state = torch.load(path, map_location=self.device)
                if isinstance(state, dict) and "model_state_dict" in state:
                    state = state["model_state_dict"]
                self.model.load_state_dict(state, strict=False)
                print(f"[DnCNN] Loaded weights from {path}")
            else:
                print(f"[DnCNN] Warning: weights not found at {path}, using random init")

    @torch.no_grad()
    def denoise(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Denoise a [B, C, H, W] or [C, H, W] float tensor in [0, 1].
        Returns denoised tensor of the same shape.
        """
        squeeze = img_tensor.dim() == 3
        if squeeze:
            img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        out = self.model(img_tensor)
        if squeeze:
            out = out.squeeze(0)
        return out.clamp(0.0, 1.0)

    @torch.no_grad()
    def noise_residual(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Returns the noise residual W = Y - F(Y).
        Same shape as input.
        """
        squeeze = img_tensor.dim() == 3
        if squeeze:
            img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        denoised = self.model(img_tensor)
        residual = img_tensor - denoised
        if squeeze:
            residual = residual.squeeze(0)
        return residual
