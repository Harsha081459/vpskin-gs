"""[vpskin] Blend field beta(x): canonical position -> logit of the LBS<->DQS blend weight (C2).

A small coordinate MLP (same positional encoding as GauHuman's LBS-offset decoder).
Being a smooth field, it needs no explicit smoothness term and survives densification/
pruning untouched (no per-Gaussian parameter to split or prune).
The output bias sets the initial blend: init_logit=0 -> beta=0.5 (default; the sigmoid has its
largest gradient there, so the regularisers can push either way). init_logit=-4 -> beta~0.018,
i.e. start at LBS. The GPU smoke test showed beta barely moves from there (sigmoid slope ~0.018).
"""
import torch
import torch.nn as nn

from nets.mlp_delta_weight_lbs import get_embedder


class BetaField(nn.Module):
    def __init__(self, multires: int = 6, width: int = 64, depth: int = 3, init_logit: float = 0.0):
        super().__init__()
        self.embed, in_ch = get_embedder(multires, 3)
        layers, c = [], in_ch
        for _ in range(depth):
            layers += [nn.Linear(c, width), nn.ReLU(inplace=True)]
            c = width
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(width, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, init_logit)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz (B, N, 3) -> logits (B, N, 1)."""
        return self.out(self.mlp(self.embed(xyz)))
