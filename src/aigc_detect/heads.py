"""Classifier heads trained on top of frozen-backbone pooled embeddings.

Per the "Simplicity Prevails" recipe (arXiv:2602.01738), the paper's own
head is a single linear layer; MLPHead is provided as an optional richer
alternative for comparison, not part of the paper's recipe.
"""

from __future__ import annotations

import torch.nn as nn


class LinearHead(nn.Module):
    """The paper's head: a single linear layer to one logit."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.linear(x)


class MLPHead(nn.Module):
    """Optional richer head: Linear -> ReLU -> Dropout -> Linear."""

    def __init__(self, in_dim: int, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def build_head(kind: str, in_dim: int, **kwargs) -> nn.Module:
    if kind == "linear":
        return LinearHead(in_dim)
    elif kind == "mlp":
        return MLPHead(in_dim, **kwargs)
    else:
        raise ValueError(f"Unknown head kind '{kind}'. Expected 'linear' or 'mlp'.")
