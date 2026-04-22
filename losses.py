from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(predictions, targets)
        return torch.sqrt(mse + self.eps)


def build_loss(loss_name: str, huber_delta: float = 5.0) -> nn.Module:
    loss_name = loss_name.lower()
    if loss_name == "rmse":
        return RMSELoss()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name in {"mae", "l1"}:
        return nn.L1Loss()
    if loss_name == "huber":
        return nn.HuberLoss(delta=huber_delta)
    raise ValueError(f"Unsupported loss: {loss_name}")
