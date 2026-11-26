"""Unit tests for vpskin.skinning and vpskin.regularizers (CPU, float64).

Run:  python -m pytest -q tests
"""
import math

import pytest
import torch

from vpskin.regularizers import (iso_loss, knn_graph, neighbour_drift, part_volume_ratio,
                                 vol_loss)
from vpskin.skinning import (blend, dqs, lbs, matrix_to_quaternion, quatavg,
                             quaternion_to_matrix, rigid_to_dualquat, dualquat_to_rigid,
                             volume_factor)

torch.set_default_dtype(torch.float64)
GEN = torch.Generator().manual_seed(0)


def rand_rot(n):
    q = torch.randn(n, 4, generator=GEN)
    return quaternion_to_matrix(q)


def axis_angle(axis, theta):
    axis = torch.as_tensor(axis, dtype=torch.float64)
    axis = axis / axis.norm()
    K = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return torch.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * K @ K  # Rodrigues (L8)


def rigid(R, t):
    G = torch.eye(4).repeat(R.shape[0], 1, 1)
    G[:, :3, :3], G[:, :3, 3] = R, t
    return G


def rand_bones(J=24):
    return rigid(rand_rot(J), torch.randn(J, 3, generator=GEN))


def rand_weights(N, J=24, sparsity=4):
    W = torch.zeros(N, J)
    idx = torch.stack([torch.randperm(J, generator=GEN)[:sparsity] for _ in range(N)])
    W.scatter_(1, idx, torch.rand(N, sparsity, generator=GEN))
    return W / W.sum(1, keepdim=True)


# ---------------------------------------------------------------- conversions

def test_quaternion_roundtrip():
    R = rand_rot(1000)
    assert torch.allclose(quaternion_to_matrix(matrix_to_quaternion(R)), R, atol=1e-10)
    # near-180 degree rotations (the branch that breaks naive formulas)
    R180 = torch.stack([axis_angle(a, math.pi - 1e-9) for a in torch.randn(50, 3, generator=GEN)])
    assert torch.allclose(quaternion_to_matrix(matrix_to_quaternion(R180)), R180, atol=1e-7)


def test_dualquat_roundtrip():
    G = rand_bones(100)
    R, t = dualquat_to_rigid(rigid_to_dualquat(G))
    assert torch.allclose(R, G[:, :3, :3], atol=1e-10)
    assert torch.allclose(t, G[:, :3, 3], atol=1e-10)


# ---------------------------------------------------------------- Proposition 1

@pytest.mark.parametrize("w", [0.1, 0.3, 0.5, 0.8])
@pytest.mark.parametrize("theta", [0.3, 1.0, math.pi / 2, 2.5, math.pi])
def test_prop1_lbs_det_closed_form(w, theta):
    """det((1-w) R1 + w R1 R_rel) = 1 - 2w(1-w)(1-cos theta), for ANY axis and any R1."""
    for axis in torch.randn(5, 3, generator=GEN):
        R1 = rand_rot(1)[0]
        R2 = R1 @ axis_angle(axis, theta)
        G = rigid(torch.stack([R1, R2]), torch.randn(2, 3, generator=GEN))
        W = torch.tensor([[1 - w, w]])
        M, _, _ = lbs(W, G)
        expected = 1 - 2 * w * (1 - w) * (1 - math.cos(theta))
        assert abs(torch.linalg.det(M)[0].item() - expected) < 1e-10
        # the shrink is isotropic in the plane orthogonal to the axis: singular values (1, r, r)
        r = math.sqrt(expected)
        sv = torch.linalg.svdvals(M[0])
        assert torch.allclose(sv.sort().values, torch.tensor(sorted([1.0, r, r])), atol=1e-8)


def test_candy_wrapper_collapse():
    """w = 1/2, 180 degree twist: LBS collapses (det 0), DQS keeps det 1."""
    G = rigid(torch.stack([torch.eye(3), axis_angle([1, 0, 0], math.pi)]), torch.zeros(2, 3))
    W = torch.tensor([[0.5, 0.5]])
    assert volume_factor(lbs(W, G)[2]).item() < 1e-12
    assert abs(volume_factor(dqs(W, G)[2]).item() - 1) < 1e-12


# ---------------------------------------------------------------- Proposition 2 / DQS sanity

def test_dqs_is_rigid_and_preserves_volume():
    W, G = rand_weights(500), rand_bones()
    R, _, Rc = dqs(W, G)
    I = torch.eye(3).expand_as(R)
    assert torch.allclose(R @ R.transpose(1, 2), I, atol=1e-10)
    assert torch.allclose(torch.linalg.det(R), torch.ones(500), atol=1e-10)
    S = torch.diag_embed(torch.rand(500, 3, generator=GEN) + 0.1)
    Q = rand_rot(500)
    Sigma = Q @ S @ S @ Q.transpose(1, 2)
    Sigma2 = Rc @ Sigma @ Rc.transpose(1, 2)
    assert torch.allclose(torch.linalg.det(Sigma2), torch.linalg.det(Sigma), rtol=1e-9)


def test_single_bone_all_modes_agree():
    G = rand_bones()
    W = torch.nn.functional.one_hot(torch.randint(0, 24, (200,), generator=GEN), 24).double()
    M_l, t_l, _ = lbs(W, G)
    for fn in (dqs, quatavg):
        M, t, Mc = fn(W, G)
        assert torch.allclose(M, M_l, atol=1e-10) and torch.allclose(t, t_l, atol=1e-10)
        assert torch.allclose(Mc, M_l, atol=1e-10)


def test_dqs_antipodal_invariance():
    """q and -q are the same rotation: DQS must not depend on the sign convention."""
    from vpskin import skinning as sk
    W, G = rand_weights(300), rand_bones()
    R0, t0, _ = dqs(W, G)
    orig = sk.rigid_to_dualquat
    flip = torch.where(torch.rand(24, generator=GEN) < 0.5, -1.0, 1.0)[:, None]
    sk.rigid_to_dualquat = lambda G_: orig(G_) * flip
    try:
        R1, t1, _ = sk.dqs(W, G)
    finally:
        sk.rigid_to_dualquat = orig
    assert torch.allclose(R0, R1, atol=1e-10) and torch.allclose(t0, t1, atol=1e-10)


def test_dqs_two_bone_midpoint_is_half_rotation():
    """Equal weights on I and a rotation by theta about a fixed joint -> rotation by theta/2."""
    theta = 2.0
    Rb = axis_angle([0, 0, 1], theta)
    G = rigid(torch.stack([torch.eye(3), Rb]), torch.zeros(2, 3))
    R, t, _ = dqs(torch.tensor([[0.5, 0.5]]), G)
    assert torch.allclose(R[0], axis_angle([0, 0, 1], theta / 2), atol=1e-10)
    _, _, Rq = quatavg(torch.tensor([[0.5, 0.5]]), G)
    assert torch.allclose(Rq[0], axis_angle([0, 0, 1], theta / 2), atol=1e-8)


def test_blend_endpoints():
    W, G = rand_weights(100), rand_bones()
    M_l, t_l, _ = lbs(W, G)
    M_d, t_d, _ = dqs(W, G)
    M0, t0, _ = blend(W, G, torch.zeros(100))
    M1, t1, _ = blend(W, G, torch.ones(100))
    assert torch.allclose(M0, M_l) and torch.allclose(t0, t_l)
    assert torch.allclose(M1, M_d) and torch.allclose(t1, t_d)


def test_dqs_gradients_finite():
    W = rand_weights(20, J=5).requires_grad_(True)
    G = rand_bones(5)
    beta = torch.rand(20, generator=GEN, requires_grad=True)
    M, t, _ = blend(W, G, beta)
    (M.sum() + t.sum()).backward()
    assert torch.isfinite(W.grad).all() and torch.isfinite(beta.grad).all()
    assert torch.autograd.gradcheck(lambda w: dqs(w, G)[1], (W.detach().clone().requires_grad_(True),))


# ---------------------------------------------------------------- regularisers

def test_knn_graph_matches_bruteforce():
    x = torch.rand(300, 3, generator=GEN)
    e, rest = knn_graph(x, k=4, chunk=64)
    d = torch.cdist(x, x)
    d.fill_diagonal_(float("inf"))
    nn = d.topk(4, largest=False).indices
    ref = {(min(i, j), max(i, j)) for i in range(300) for j in nn[i].tolist()}
    assert {tuple(p) for p in e.tolist()} == ref
    assert torch.allclose(rest, (x[e[:, 0]] - x[e[:, 1]]).norm(dim=1))


def test_iso_zero_under_global_rigid_motion():
    x = torch.rand(400, 3, generator=GEN)
    e, rest = knn_graph(x, k=8)
    R, t = rand_rot(1)[0], torch.randn(3, generator=GEN)
    y = x @ R.T + t
    assert iso_loss(y, e, rest).item() < 1e-12
    assert neighbour_drift(y, e, rest).item() < 1e-12
    assert iso_loss(1.2 * y, e, rest).item() > 1e-3


def test_vol_loss_zero_for_dqs_positive_for_lbs():
    W, G = rand_weights(300), rand_bones()
    assert vol_loss(dqs(W, G)[2]).item() < 1e-16
    assert vol_loss(lbs(W, G)[2]).item() > 1e-4


def test_part_volume_ratio():
    N = 100
    part = torch.randint(0, 3, (N,), generator=GEN)
    M = torch.eye(3).repeat(N, 1, 1) * 0.5 ** (1 / 3)      # det = 0.5
    r = part_volume_ratio(torch.rand(N, generator=GEN), torch.rand(N, 3, generator=GEN) + 0.1, M, part, 4)
    assert torch.allclose(r[:3], torch.full((3,), 0.5)) and torch.isnan(r[3])


def test_knn_fallback_matches_cdist():
    from vpskin.knn_fallback import KNN
    ref = torch.rand(1, 500, 3, generator=GEN)
    q = torch.rand(1, 300, 3, generator=GEN)
    _, idx = KNN(k=1, chunk=64)(ref, q)
    assert idx.shape == (1, 300, 1)
    assert torch.equal(idx[0, :, 0], torch.cdist(q[0], ref[0]).argmin(1))
