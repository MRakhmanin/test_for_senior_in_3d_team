from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _make_norm_3d(num_channels: int, norm: str = "group") -> nn.Module:
    if norm == "group":
        groups = 8 if num_channels >= 8 else 1
        return nn.GroupNorm(groups, num_channels)
    if norm == "instance":
        return nn.InstanceNorm3d(num_channels, affine=True)
    if norm == "batch":
        return nn.BatchNorm3d(num_channels)
    raise ValueError(f"Unknown norm type: {norm}")


class ResidualBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm: str = "group",
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = _make_norm_3d(out_channels, norm=norm)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = _make_norm_3d(out_channels, norm=norm)
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout_p) if dropout_p > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                _make_norm_3d(out_channels, norm=norm),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + identity
        out = self.act(out)
        return out


class AngleRegressor3D(nn.Module):
    """
    Regresss roll/pitch/yaw angles from a 3D volume.
    Input tensor shape: [B, 1, D, H, W]
    Output tensor shape: [B, 3] in degrees.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        norm: str = "group",
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c1, kernel_size=3, stride=2, padding=1, bias=False),
            _make_norm_3d(c1, norm=norm),
            nn.SiLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            ResidualBlock3D(c1, c1, stride=1, norm=norm, dropout_p=dropout_p),
            ResidualBlock3D(c1, c1, stride=1, norm=norm, dropout_p=dropout_p),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock3D(c1, c2, stride=2, norm=norm, dropout_p=dropout_p),
            ResidualBlock3D(c2, c2, stride=1, norm=norm, dropout_p=dropout_p),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock3D(c2, c3, stride=2, norm=norm, dropout_p=dropout_p),
            ResidualBlock3D(c3, c3, stride=1, norm=norm, dropout_p=dropout_p),
        )
        self.stage4 = nn.Sequential(
            ResidualBlock3D(c3, c4, stride=2, norm=norm, dropout_p=dropout_p),
            ResidualBlock3D(c4, c4, stride=1, norm=norm, dropout_p=dropout_p),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(c4, c4 // 2),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(c4 // 2, 3),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


@dataclass
class Batch:
    volumes: Tensor  # [B, 1, D, H, W]
    angles: Tensor   # [B, 3]


def angle_regression_loss(pred_deg: Tensor, target_deg: Tensor) -> Tensor:
    return F.smooth_l1_loss(pred_deg, target_deg)


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0
    for batch in loader:
        x = batch.volumes.to(device, non_blocking=True)
        y = batch.angles.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = angle_regression_loss(pred, y)
        loss.backward()
        optimizer.step()

        bsz = int(x.shape[0])
        total_loss += float(loss.detach().cpu()) * bsz
        n_samples += bsz
    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable[Batch],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    n_samples = 0
    for batch in loader:
        x = batch.volumes.to(device, non_blocking=True)
        y = batch.angles.to(device, non_blocking=True)
        pred = model(x)
        loss = angle_regression_loss(pred, y)
        mae = (pred - y).abs().mean()

        bsz = int(x.shape[0])
        total_loss += float(loss.detach().cpu()) * bsz
        total_mae += float(mae.detach().cpu()) * bsz
        n_samples += bsz

    denom = max(n_samples, 1)
    return {
        "loss": total_loss / denom,
        "mae_deg": total_mae / denom,
    }


@torch.no_grad()
def infer_angles(
    model: nn.Module,
    volume_4d: Tensor,
    device: torch.device,
) -> Tensor:
    """
    volume_4d: [1, D, H, W] or [B, D, H, W]
    returns: [B, 3] predicted angles in degrees
    """
    model.eval()
    if volume_4d.ndim == 4:
        volume_5d = volume_4d.unsqueeze(0)  # [1, 1, D, H, W]
    elif volume_4d.ndim == 5:
        volume_5d = volume_4d
    else:
        raise ValueError("Expected [1,D,H,W] or [B,1,D,H,W] tensor.")
    volume_5d = volume_5d.to(device, non_blocking=True)
    return model(volume_5d).cpu()

