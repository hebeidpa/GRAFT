from __future__ import annotations

import torch
from torch import nn


class FusionMultiTaskHead(nn.Module):
    """A compact two-logit head for recurrence and complication."""

    def __init__(
        self,
        input_dim: int,
        head_type: str = "linear",
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if head_type == "linear":
            self.net = nn.Linear(input_dim, 2)
        elif head_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
        else:
            raise ValueError(f"head_type must be 'linear' or 'mlp', got {head_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
