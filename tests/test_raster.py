import math

import torch

from vpskin.raster import interpolate, look_at, project, rasterize, vertex_normals

K = torch.tensor([[100.0, 0, 64], [0, 100.0, 64], [0, 0, 1]], dtype=torch.float64)


def test_triangle_area_and_barycentrics():
    # big triangle at z=2 (forces recursive splitting with S=8)
    v = torch.tensor([[-0.5, -0.5, 2.0], [0.6, -0.4, 2.0], [-0.3, 0.7, 2.0]], dtype=torch.float64)
    f = torch.tensor([[0, 1, 2]])
    fid, bary, depth = rasterize(v, f, K, 128, 128, S=8)
    uv = project(v, K)
    area = 0.5 * abs((uv[1, 0] - uv[0, 0]) * (uv[2, 1] - uv[0, 1]) - (uv[1, 1] - uv[0, 1]) * (uv[2, 0] - uv[0, 0]))
    covered = (fid == 0).sum().item()
    assert abs(covered - area.item()) / area.item() < 0.03          # pixel count ~ analytic area
    assert torch.allclose(bary[fid == 0].sum(1), torch.ones(covered, dtype=torch.float64))
    assert torch.allclose(depth[fid == 0], torch.full((covered,), 2.0, dtype=torch.float64))
    # interpolating vertex positions reproduces the back-projected pixel ray at that depth
    p = interpolate(v, f, fid, bary)
    i, j = torch.nonzero(fid == 0)[100]
    x = (j.double() - 64) / 100 * 2.0
    y = (i.double() - 64) / 100 * 2.0
    assert torch.allclose(p[i, j], torch.tensor([x, y, 2.0], dtype=torch.float64), atol=1e-9)


def test_zbuffer_occlusion():
    near = torch.tensor([[-1, -1, 2.0], [1, -1, 2.0], [0, 1, 2.0]], dtype=torch.float64)
    far = near.clone(); far[:, 2] = 4.0; far[:, :2] *= 2
    v = torch.cat([far, near]); f = torch.tensor([[0, 1, 2], [3, 4, 5]])
    fid, _, depth = rasterize(v, f, K, 128, 128)
    assert fid[64, 64] == 1 and abs(depth[64, 64].item() - 2.0) < 1e-9


def test_look_at_conventions():
    R, t = look_at([0, 1, 3], [0, 1, 0])
    assert torch.allclose(R @ R.T, torch.eye(3, dtype=torch.float64), atol=1e-12)
    c = R @ torch.tensor([0, 1.0, 0], dtype=torch.float64) + t
    assert torch.allclose(c, torch.tensor([0, 0, 3.0], dtype=torch.float64))   # target straight ahead
    up = R @ torch.tensor([0, 2.0, 0], dtype=torch.float64) + t
    assert up[1] < 0                                                           # world up -> image up (-y)


def test_vertex_normals_cube_corner():
    v = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.float64)
    n = vertex_normals(v, torch.tensor([[0, 1, 2]]))
    assert torch.allclose(n, torch.tensor([[0, 0, 1.0]] * 3, dtype=torch.float64))
