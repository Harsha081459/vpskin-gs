"""Animation / geometry metrics (E4-E6). Pure torch, device-agnostic.

Chamfer & P2S in centimetres; hole rate in [0, 1]; volume factors = |det M_cov|.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from vpskin.smpl_lite import JOINT_NAMES

# joints used for "near-joint" errors (shoulders, elbows, wrists, hips, knees)
NEAR_JOINTS = [16, 17, 18, 19, 20, 21, 1, 2, 4, 5]
# twist/bend test: rotated joint -> (child joint defining the bone axis, parent bone whose weights mix)
TWIST_JOINTS = {
    "l_elbow": (18, 20, 16), "r_elbow": (19, 21, 17),
    "l_knee": (4, 7, 1), "r_knee": (5, 8, 2),
    "l_shoulder": (16, 18, 13), "r_shoulder": (17, 19, 14),
}


def _nn_dist(a: torch.Tensor, b: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """For each point in a, distance to its nearest neighbour in b."""
    out = []
    for s in range(0, a.shape[0], chunk):
        out.append(torch.cdist(a[s:s + chunk], b, compute_mode="donot_use_mm_for_euclid_dist").min(1).values)
    return torch.cat(out) if out else a.new_zeros(0)


def chamfer_cm(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    """Symmetric Chamfer (mean of the two directed mean NN distances), in cm."""
    if len(pred) == 0 or len(gt) == 0:
        return {"chamfer": float("nan"), "p2gt": float("nan"), "gt2p": float("nan")}
    d1 = _nn_dist(pred, gt).mean().item() * 100
    d2 = _nn_dist(gt, pred).mean().item() * 100
    return {"chamfer": 0.5 * (d1 + d2), "p2gt": d1, "gt2p": d2}


def near_mask(points: torch.Tensor, joints: torch.Tensor, ids=NEAR_JOINTS, radius: float = 0.10) -> torch.Tensor:
    return torch.cdist(points, joints[ids]).min(1).values < radius


def sample_surface(verts: torch.Tensor, faces: torch.Tensor, n: int, seed: int = 0) -> torch.Tensor:
    """Area-weighted uniform samples on a triangle mesh."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    f = faces.long()
    a, b, c = verts[f[:, 0]], verts[f[:, 1]], verts[f[:, 2]]
    area = torch.linalg.cross(b - a, c - a).norm(dim=1)
    idx = torch.multinomial(area.cpu().double(), n, replacement=True, generator=g).to(verts.device)
    u = torch.rand(n, 1, generator=g).to(verts)
    v = torch.rand(n, 1, generator=g).to(verts)
    flip = (u + v) > 1
    u, v = torch.where(flip, 1 - u, u), torch.where(flip, 1 - v, v)
    return a[idx] + u * (b[idx] - a[idx]) + v * (c[idx] - a[idx])


def hole_rate(alpha: torch.Tensor, silhouette: torch.Tensor, erode_px: int = 3, thresh: float = 0.5) -> float:
    """Fraction of (eroded) body-silhouette pixels where the rendered alpha is below `thresh`.
    Erosion ignores boundary pixels where skinning/clothing legitimately differ from SMPL."""
    s = silhouette.float()[None, None]
    if erode_px > 0:
        s = 1 - F.max_pool2d(1 - s, 2 * erode_px + 1, stride=1, padding=erode_px)
    s = s[0, 0] > 0.5
    n = s.sum().item()
    return float(((alpha < thresh) & s).sum().item() / n) if n else float("nan")


def axis_pose(joint: int, axis: torch.Tensor, theta: float, base: torch.Tensor | None = None) -> torch.Tensor:
    """(1, 72) pose: rotate `joint` by `theta` about `axis` (unit, expressed in the T-pose frame).
    With base = T-pose (zeros), every joint's local frame equals the world frame."""
    p = torch.zeros(1, 72, dtype=axis.dtype, device=axis.device) if base is None else base.clone()
    p[0, 3 * joint:3 * joint + 3] = theta * axis / axis.norm()
    return p


def twist_and_bend_axes(joints_tpose: torch.Tensor, joint: int, child: int):
    """Axial (twist) axis along the bone, and a bend axis orthogonal to it (and to world up
    when possible), from T-pose joint locations."""
    bone = joints_tpose[child] - joints_tpose[joint]
    twist = bone / bone.norm()
    up = torch.tensor([0.0, 1.0, 0.0], dtype=bone.dtype, device=bone.device)
    bend = torch.linalg.cross(twist, up)
    if bend.norm() < 1e-3:                                   # vertical bone (legs): bend about x
        bend = torch.tensor([1.0, 0.0, 0.0], dtype=bone.dtype, device=bone.device)
    return twist, bend / bend.norm()


def joint_region(W: torch.Tensor, a: int, b: int, lo: float = 0.1) -> torch.Tensor:
    """Gaussians whose skinning mixes bones a and b (both weights > lo): where LBS collapses."""
    return (W[:, a] > lo) & (W[:, b] > lo)


def lbs_det_theory(w: float, theta: float) -> float:
    """Proposition 1: det((1-w) I + w R(theta)) = 1 - 2 w (1-w) (1 - cos theta)."""
    return 1 - 2 * w * (1 - w) * (1 - math.cos(theta))


__all__ = ["NEAR_JOINTS", "TWIST_JOINTS", "JOINT_NAMES", "chamfer_cm", "near_mask", "sample_surface",
           "hole_rate", "axis_pose", "twist_and_bend_axes", "joint_region", "lbs_det_theory"]
