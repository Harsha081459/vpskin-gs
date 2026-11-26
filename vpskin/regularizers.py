"""Image-free regularisers (C3) and the matching animation metrics (E5).

Training uses knn_graph(k=8); evaluation uses k=16 on disjoint motions (PHASE2_plan §4).
"""
from __future__ import annotations

import torch

__all__ = ["knn_graph", "iso_loss", "vol_loss", "beta_smooth_loss",
           "neighbour_drift", "part_volume_ratio"]


@torch.no_grad()
def knn_graph(xyz: torch.Tensor, k: int = 8, radius: float | None = None, chunk: int = 2048):
    """Undirected k-NN edge list on canonical centres.

    Returns edges (E, 2) long with i < k (deduplicated) and rest lengths (E,).
    Chunked brute force: O(N^2) time, O(chunk*N) memory. Fine for N <= ~200k on GPU.
    """
    N = xyz.shape[0]
    k_eff = min(k, N - 1)
    nbrs = []
    for s in range(0, N, chunk):
        d = torch.cdist(xyz[s:s + chunk], xyz)                     # (c, N)
        d[torch.arange(d.shape[0], device=xyz.device), torch.arange(s, s + d.shape[0], device=xyz.device)] = float("inf")
        nbrs.append(d.topk(k_eff, dim=1, largest=False).indices)
    nbr = torch.cat(nbrs, 0)                                         # (N, k)
    i = torch.arange(N, device=xyz.device).repeat_interleave(k_eff)
    j = nbr.reshape(-1)
    e = torch.stack([torch.minimum(i, j), torch.maximum(i, j)], 1)
    e = torch.unique(e, dim=0)
    rest = (xyz[e[:, 0]] - xyz[e[:, 1]]).norm(dim=1)
    if radius is not None:
        keep = rest <= radius
        e, rest = e[keep], rest[keep]
    return e, rest


def _rel_stretch(xyz_posed: torch.Tensor, edges: torch.Tensor, rest: torch.Tensor, eps: float = 1e-6):
    cur = (xyz_posed[edges[:, 0]] - xyz_posed[edges[:, 1]]).norm(dim=1)
    return (cur - rest) / (rest + eps)


def iso_loss(xyz_posed: torch.Tensor, edges: torch.Tensor, rest: torch.Tensor, delta: float = 0.05):
    """L_iso: Charbonnier penalty on relative edge-length change vs canonical.
    delta sets the scale below which stretch is treated ~quadratically (joints may stretch)."""
    r = _rel_stretch(xyz_posed, edges, rest)
    return (torch.sqrt(r * r + delta * delta) - delta).mean()


def vol_loss(M_cov: torch.Tensor, eps: float = 1e-8):
    """L_vol: mean (log|det M|)^2. Identically 0 for rigid (DQS) skinning."""
    ld = torch.log(torch.linalg.det(M_cov).abs().clamp_min(eps))
    return (ld * ld).mean()


def beta_smooth_loss(beta: torch.Tensor, edges: torch.Tensor):
    """L_beta: smoothness of the LBS<->DQS blend map over the canonical graph."""
    b = beta.reshape(-1)
    return ((b[edges[:, 0]] - b[edges[:, 1]]) ** 2).mean()


@torch.no_grad()
def neighbour_drift(xyz_posed: torch.Tensor, edges: torch.Tensor, rest: torch.Tensor):
    """E5 metric D_nn: mean |relative edge-length change|. Use a k different from training."""
    return _rel_stretch(xyz_posed, edges, rest).abs().mean()


@torch.no_grad()
def part_volume_ratio(opacity: torch.Tensor, scales: torch.Tensor, M_cov: torch.Tensor,
                      part: torch.Tensor, n_parts: int):
    """E5 metric: per-part opacity-weighted Gaussian volume ratio, posed / canonical.

    canonical volume_i ~ prod(scales_i); posed volume_i = |det M_i| * canonical volume_i.
    part: (N,) long body-part label (e.g. argmax skinning weight). Returns (n_parts,), NaN if empty.
    """
    a = opacity.reshape(-1)
    v0 = a * scales.prod(dim=1)
    v1 = v0 * torch.linalg.det(M_cov).abs()
    num = torch.zeros(n_parts, device=a.device, dtype=a.dtype).index_add_(0, part, v1)
    den = torch.zeros(n_parts, device=a.device, dtype=a.dtype).index_add_(0, part, v0)
    return torch.where(den > 0, num / den.clamp_min(1e-12), torch.full_like(den, float("nan")))
