"""Skinning operators for articulated 3D Gaussians.

All functions are pure torch, batched over points, differentiable, and device-agnostic.

Convention: a skinned point moves as  x' = M x + t.
Each operator returns (M_pos, t, M_cov):
    mean       : mu'    = M_pos @ mu + t
    covariance : Sigma' = M_cov @ Sigma @ M_cov^T

    mode      M_pos                     M_cov                       reference
    lbs       sum_j w_j R_j             = M_pos                     PDF Eq. (3); GauHuman
    dqs       R~ (rigid)                = R~                        Kavan et al. 2007
    blend     (1-b) M_lbs + b R~        = M_pos                     ours (C2)
    quatavg   sum_j w_j R_j (LBS mean)  R(argmax-eig sum w q q^T)   arXiv 2509.11411

Shapes: weights W (N, J), bone transforms G (J, 4, 4) or (N, J, 4, 4) is NOT supported
(bones are shared by all points, as in SMPL). beta (N,) or (N, 1) in [0, 1].
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "matrix_to_quaternion", "quaternion_to_matrix", "quat_mul", "quat_conj",
    "rigid_to_dualquat", "dualquat_to_rigid",
    "lbs", "dqs", "blend", "quatavg", "skin", "volume_factor",
]


# --------------------------------------------------------------------------- quaternions
# Quaternions are (w, x, y, z), Hamilton product.

def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion (..., 4) -> rotation matrix (..., 3, 3). Input is normalised first."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    two = 2.0
    m = torch.stack([
        1 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w),
        two * (x * y + z * w), 1 - two * (x * x + z * z), two * (y * z - x * w),
        two * (x * z - y * w), two * (y * z + x * w), 1 - two * (x * x + y * y),
    ], dim=-1)
    return m.reshape(q.shape[:-1] + (3, 3))


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix (..., 3, 3) -> unit quaternion (..., 4) with w >= 0.

    Shepperd's method: choose the numerically largest of (w, x, y, z) as pivot.
    """
    m = R
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    tr = m00 + m11 + m22
    # 4 candidate (unnormalised) quaternions, each well-conditioned in one regime
    cand = torch.stack([
        torch.stack([1 + tr, m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]], -1),
        torch.stack([m[..., 2, 1] - m[..., 1, 2], 1 + m00 - m11 - m22, m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0]], -1),
        torch.stack([m[..., 0, 2] - m[..., 2, 0], m[..., 0, 1] + m[..., 1, 0], 1 - m00 + m11 - m22, m[..., 1, 2] + m[..., 2, 1]], -1),
        torch.stack([m[..., 1, 0] - m[..., 0, 1], m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1], 1 - m00 - m11 + m22], -1),
    ], dim=-2)  # (..., 4, 4)
    diag = torch.stack([1 + tr, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1)
    idx = diag.argmax(-1, keepdim=True)  # (..., 1)
    q = torch.gather(cand, -2, idx[..., None].expand(idx.shape[:-1] + (1, 4))).squeeze(-2)
    q = F.normalize(q, dim=-1)
    return torch.where(q[..., :1] < 0, -q, q)


# --------------------------------------------------------------------------- dual quaternions

def rigid_to_dualquat(G: torch.Tensor) -> torch.Tensor:
    """(..., 4, 4) rigid transform -> (..., 8) unit dual quaternion [q_r | q_d]."""
    qr = matrix_to_quaternion(G[..., :3, :3])
    t = G[..., :3, 3]
    tq = torch.cat([torch.zeros_like(t[..., :1]), t], -1)
    qd = 0.5 * quat_mul(tq, qr)
    return torch.cat([qr, qd], -1)


def dualquat_to_rigid(dq: torch.Tensor, eps: float = 1e-8):
    """(..., 8) (not necessarily unit) dual quaternion -> (R (...,3,3), t (...,3)).

    Normalisation (Kavan 2007): divide both parts by |q_r|; the translation is
    t = 2 * vec(q_d * conj(q_r)) of the normalised dual quaternion.
    The non-orthogonal component of q_d is dropped implicitly by this formula.
    """
    qr, qd = dq[..., :4], dq[..., 4:]
    n = qr.norm(dim=-1, keepdim=True).clamp_min(eps)
    qr, qd = qr / n, qd / n
    R = quaternion_to_matrix(qr)
    t = 2.0 * quat_mul(qd, quat_conj(qr))[..., 1:]
    return R, t


# --------------------------------------------------------------------------- skinning modes

def _check(W: torch.Tensor, G: torch.Tensor):
    assert W.dim() == 2 and G.dim() == 3 and G.shape[-2:] == (4, 4), (W.shape, G.shape)
    assert W.shape[1] == G.shape[0], "weights and bones disagree on J"


def lbs(W: torch.Tensor, G: torch.Tensor):
    """Linear blend skinning. Returns (M, t, M)."""
    _check(W, G)
    A = (W @ G.reshape(G.shape[0], 16)).reshape(-1, 4, 4)
    M, t = A[:, :3, :3], A[:, :3, 3]
    return M, t, M


def _sign_aligned_weights(W: torch.Tensor, qr: torch.Tensor) -> torch.Tensor:
    """Antipodality fix: flip each bone's sign so it lies in the hemisphere of the
    point's max-weight bone. Returns signed weights (N, J)."""
    pivot = W.argmax(dim=1)                       # (N,)
    dots = qr @ qr.T                              # (J, J)
    s = torch.where(dots[pivot] < 0, -1.0, 1.0).to(W.dtype)  # (N, J)
    return W * s


def dqs(W: torch.Tensor, G: torch.Tensor):
    """Dual quaternion skinning. Returns (R, t, R) with R in SO(3)."""
    _check(W, G)
    dq = rigid_to_dualquat(G)                     # (J, 8)
    Ws = _sign_aligned_weights(W, dq[:, :4])
    b = Ws @ dq                                   # (N, 8)
    R, t = dualquat_to_rigid(b)
    return R, t, R


def blend(W: torch.Tensor, G: torch.Tensor, beta: torch.Tensor):
    """Per-point convex blend of LBS and DQS (ours, C2). beta=0 -> LBS, beta=1 -> DQS."""
    beta = beta.reshape(-1, 1)
    M_l, t_l, _ = lbs(W, G)
    R_d, t_d, _ = dqs(W, G)
    b3 = beta[..., None]
    M = (1 - b3) * M_l + b3 * R_d
    t = (1 - beta) * t_l + beta * t_d
    return M, t, M


def quatavg(W: torch.Tensor, G: torch.Tensor, iters: int = 15):
    """Comparator B1 (arXiv 2509.11411): LBS means; covariance rotated by the weighted
    quaternion average (Markley et al.): the dominant eigenvector of sum_j w_j q_j q_j^T.

    Power iteration started from the sign-aligned weighted sum, which is already a
    close estimate, so a few iterations converge and gradients stay well defined.
    """
    M_l, t_l, _ = lbs(W, G)
    qr = matrix_to_quaternion(G[:, :3, :3])       # (J, 4)
    Ws = _sign_aligned_weights(W, qr)
    v = F.normalize(Ws @ qr, dim=-1)              # (N, 4) initial estimate
    # A_i v = sum_j w_ij q_j (q_j . v)  (never materialise A_i)
    for _ in range(iters):
        proj = v @ qr.T                           # (N, J)
        v = F.normalize((W * proj) @ qr, dim=-1)
    return M_l, t_l, quaternion_to_matrix(v)


def skin(W: torch.Tensor, G: torch.Tensor, mode: str = "lbs", beta: torch.Tensor | None = None):
    """Dispatch. Returns (M_pos, t, M_cov)."""
    if mode == "lbs":
        return lbs(W, G)
    if mode == "dqs":
        return dqs(W, G)
    if mode == "blend":
        assert beta is not None, "blend mode needs beta"
        return blend(W, G, beta)
    if mode == "quatavg":
        return quatavg(W, G)
    raise ValueError(f"unknown skinning mode {mode!r}")


def volume_factor(M_cov: torch.Tensor) -> torch.Tensor:
    """Per-Gaussian volume ratio V'/V = |det M_cov| (Gaussian volume ~ sqrt(det Sigma))."""
    return torch.linalg.det(M_cov).abs()
