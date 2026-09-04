import torch
import torch.nn as nn


class SensorFiLM(nn.Module):
    """Sensor-conditioned Feature-wise Linear Modulation (FiLM) layer.

    Applies a residual affine correction to feature maps conditioned on a binary
    sensor flag (0.0 = OLI, 1.0 = TM). Zero-initialized so the correction starts
    as identity for both sensors — the pretrained model is exactly preserved at
    the start of fine-tuning.

    For OLI (sensor_flag=0): output is mathematically identical to input,
    regardless of learned parameters. OLI performance is guaranteed by construction.

    Args:
        num_channels: Number of feature channels C to modulate.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(num_channels))
        self.beta  = nn.Parameter(torch.zeros(num_channels))

    def forward(self, feat: torch.Tensor, sensor_flag: torch.Tensor) -> torch.Tensor:
        """Apply sensor-conditioned affine modulation.

        Args:
            feat:        (B, T, C, H, W) — feature map from encoder block.
            sensor_flag: (B,) — per-item sensor identity, 0.0=OLI, 1.0=TM.

        Returns:
            Modulated feature map, same shape as feat.
        """
        B, T, C, H, W = feat.shape
        g = self.gamma.to(feat.dtype).view(1, 1, C, 1, 1)
        b = self.beta.to(feat.dtype).view(1, 1, C, 1, 1)
        s = sensor_flag.to(feat.dtype).view(B, 1, 1, 1, 1)
        # Residual form: OLI (s=0) → exact identity; TM (s=1) → feat*(1+γ) + β
        return feat + s * (feat * g + b)
