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
    """Optional richer head: [Linear -> ReLU -> Dropout] x N -> Linear.

    `hidden` takes an int for one hidden layer (the original behaviour) or a
    sequence for a deeper stack, e.g. (1024, 512) for two.

    Depth is a genuine question here rather than a free upgrade. The frozen
    backbone has already done the representation learning; the head's only job
    is to find a decision boundary in that space. Capacity spent past that goes
    into fitting the TRAINING generators, and the thing we actually need is
    generalization to generators nobody has seen. UniversalFakeDetect (Ojha et
    al., CVPR 2023) is the canonical result: on frozen CLIP features, nearest-
    neighbour and linear probing transfer to unseen generators far better than
    a trained deep classifier does, and the deep classifier's advantage on the
    training distribution inverts off it. "Simplicity Prevails"
    (arXiv:2602.01738), the recipe this project follows, likewise uses a single
    linear layer.

    So measure any MLP on the UNSEEN-generator tier, not on val. See
    FINDINGS "head depth ablation" for what we actually got.
    """

    def __init__(self, in_dim: int, hidden: int | tuple[int, ...] = 512, dropout: float = 0.2):
        super().__init__()
        dims = (hidden,) if isinstance(hidden, int) else tuple(hidden)
        layers: list[nn.Module] = []
        prev = in_dim
        for h in dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_head(kind: str, in_dim: int, **kwargs) -> nn.Module:
    if kind == "linear":
        return LinearHead(in_dim)
    elif kind == "mlp":
        return MLPHead(in_dim, **kwargs)
    elif kind == "mlp2":
        kwargs.setdefault("hidden", (1024, 512))
        return MLPHead(in_dim, **kwargs)
    else:
        raise ValueError(f"Unknown head kind '{kind}'. Expected 'linear', 'mlp' or 'mlp2'.")
