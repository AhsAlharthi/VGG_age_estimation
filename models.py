from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activate: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if activate:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VGGStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_convs: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_in = in_channels
        for _ in range(num_convs):
            layers.append(ConvBNReLU(current_in, out_channels, activate=True))
            current_in = out_channels
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualizedVGGStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_convs: int) -> None:
        super().__init__()
        if num_convs < 1:
            raise ValueError("num_convs must be >= 1")

        layers: list[nn.Module] = []
        current_in = in_channels
        for _ in range(num_convs - 1):
            layers.append(ConvBNReLU(current_in, out_channels, activate=True))
            current_in = out_channels
        layers.append(ConvBNReLU(current_in, out_channels, activate=False))

        self.body = nn.Sequential(*layers)
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(x) + self.skip(x))


class VGGEncoder(nn.Module):
    def __init__(
        self,
        channels: Sequence[int],
        blocks_per_stage: Sequence[int],
        residualize: bool = False,
    ) -> None:
        super().__init__()
        if len(channels) != len(blocks_per_stage):
            raise ValueError("channels and blocks_per_stage must have the same length")

        stage_cls = ResidualizedVGGStage if residualize else VGGStage
        self.stages = nn.ModuleList()
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(len(channels) - 1)])

        in_channels = 3
        for out_channels, num_convs in zip(channels, blocks_per_stage):
            self.stages.append(stage_cls(in_channels, out_channels, num_convs))
            in_channels = out_channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        for stage_index, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
            if stage_index < len(self.pools):
                x = self.pools[stage_index](x)
        return features


class RegressionHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(1)


class VGGRegressor(nn.Module):
    def __init__(
        self,
        channels: Sequence[int],
        blocks_per_stage: Sequence[int],
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = VGGEncoder(channels=channels, blocks_per_stage=blocks_per_stage, residualize=False)
        self.regressor = RegressionHead(in_channels=channels[-1], dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.regressor(features[-1])

    def get_gradcam_target_layer(self) -> nn.Module:
        return self.encoder.stages[-1]


class ResidualizedVGGRegressor(nn.Module):
    def __init__(
        self,
        channels: Sequence[int],
        blocks_per_stage: Sequence[int],
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = VGGEncoder(channels=channels, blocks_per_stage=blocks_per_stage, residualize=True)
        self.regressor = RegressionHead(in_channels=channels[-1], dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.regressor(features[-1])

    def get_gradcam_target_layer(self) -> nn.Module:
        return self.encoder.stages[-1]


class UNetVGGRegressor(nn.Module):
    def __init__(
        self,
        channels: Sequence[int],
        blocks_per_stage: Sequence[int],
        dropout: float = 0.3,
        decoder_scale: float = 0.12,
        decoder_min_channels: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = VGGEncoder(channels=channels, blocks_per_stage=blocks_per_stage, residualize=False)
        self.decoder_channels = [max(decoder_min_channels, int(ch * decoder_scale)) for ch in channels[-2::-1]]

        self.decoder_fusions = nn.ModuleList()
        current_channels = channels[-1]
        for skip_channels, decoder_channels in zip(reversed(channels[:-1]), self.decoder_channels):
            self.decoder_fusions.append(
                nn.Sequential(
                    ConvBNReLU(current_channels + skip_channels, decoder_channels, activate=True),
                    ConvBNReLU(decoder_channels, decoder_channels, activate=True),
                )
            )
            current_channels = decoder_channels

        head_channels = max(current_channels, decoder_min_channels)
        self.regressor = nn.Sequential(
            nn.Conv2d(current_channels, head_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(head_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        x = features[-1]
        skip_features = list(reversed(features[:-1]))

        for fusion_block, skip in zip(self.decoder_fusions, skip_features):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = fusion_block(x)

        return self.regressor(x).squeeze(1)

    def get_gradcam_target_layer(self) -> nn.Module:
        return self.encoder.stages[-1]


class ResNet50Regressor(nn.Module):
    def __init__(self, dropout: float = 0.0, pretrained: bool = False) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        if dropout > 0:
            self.backbone.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, 1))
        else:
            self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(1)

    def get_gradcam_target_layer(self) -> nn.Module:
        return self.backbone.layer4[-1]


def build_model(
    model_name: str,
    channels: Sequence[int] = (45, 90, 180, 270, 360),
    blocks_per_stage: Sequence[int] = (2, 2, 3, 3, 3),
    dropout: float = 0.3,
    unet_decoder_scale: float = 0.12,
    resnet_pretrained: bool = False,
) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "vgg":
        return VGGRegressor(channels=channels, blocks_per_stage=blocks_per_stage, dropout=dropout)
    if model_name in {"resvgg", "residual_vgg", "residualized_vgg"}:
        return ResidualizedVGGRegressor(
            channels=channels,
            blocks_per_stage=blocks_per_stage,
            dropout=dropout,
        )
    if model_name in {"unet", "vgg_unet", "unet_vgg"}:
        return UNetVGGRegressor(
            channels=channels,
            blocks_per_stage=blocks_per_stage,
            dropout=dropout,
            decoder_scale=unet_decoder_scale,
        )
    if model_name in {"resnet50", "resnet"}:
        return ResNet50Regressor(dropout=dropout, pretrained=resnet_pretrained)
    raise ValueError(f"Unsupported model name: {model_name}")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def format_parameter_count(num_parameters: int) -> str:
    return f"{num_parameters / 1_000_000:.3f}M"
