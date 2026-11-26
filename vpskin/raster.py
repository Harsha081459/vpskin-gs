"""Minimal differentiable-free triangle rasterizer in pure torch (CPU or GPU).

Used to (a) render the synthetic SMPL dataset (E4) and (b) render SMPL silhouettes for the
hole-rate metric (E5). No OpenGL / pytorch3d / nvdiffrast needed, so it runs anywhere.

Camera model (L4-5, L14-15): pinhole, OpenCV axes (x right, y down, z forward),
x_cam = R x_world + t, pixel = K x_cam / z. Pixel (row i, col j) has its centre at (u=j, v=i).

Method: per-triangle bounding boxes of at most S x S pixels (larger triangles are split
recursively at edge midpoints), barycentrics via edge functions, perspective-correct
depth, and a z-buffer via scatter_reduce(amin) on packed (depth, face) int64 keys.
"""
from __future__ import annotations

import torch

__all__ = ["look_at", "project", "rasterize", "interpolate", "vertex_normals"]


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    """World->camera (R, t) for an OpenCV camera at `eye` looking at `target`.
    `up` is the world up direction (image 'up' = -y_cam)."""
    eye, target, up = (torch.as_tensor(v, dtype=torch.float64) for v in (eye, target, up))
    z = target - eye
    z = z / z.norm()
    x = torch.linalg.cross(z, up)          # right = forward x up  (for y-down image axes)
    x = x / x.norm()
    y = torch.linalg.cross(z, x)           # down
    R = torch.stack([x, y, z], 0)          # rows = camera axes in world coords
    t = -R @ eye
    return R, t


def project(v_cam: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """(V, 3) camera-space points -> (V, 2) pixel coords (u, v)."""
    p = v_cam @ K.T
    return p[:, :2] / p[:, 2:3]


def _split_faces(corner_bary, orig_id, big):
    """Split `big` (sub-)faces into 4 at edge midpoints. A sub-face is represented by the
    barycentric coordinates of its 3 corners w.r.t. its ORIGINAL face (corner_bary (F, 3, 3))
    plus the original face index, so no new vertices are created and attributes stay exact."""
    cb, ob = corner_bary[big], orig_id[big]
    m01, m12, m20 = (cb[:, 0] + cb[:, 1]) / 2, (cb[:, 1] + cb[:, 2]) / 2, (cb[:, 2] + cb[:, 0]) / 2
    c0, c1, c2 = cb[:, 0], cb[:, 1], cb[:, 2]
    new_cb = torch.cat([
        torch.stack([c0, m01, m20], 1), torch.stack([m01, c1, m12], 1),
        torch.stack([m20, m12, c2], 1), torch.stack([m01, m12, m20], 1)], 0)
    new_ob = ob.repeat(4)
    keep = ~big
    return torch.cat([corner_bary[keep], new_cb], 0), torch.cat([orig_id[keep], new_ob], 0)


@torch.no_grad()
def rasterize(v_cam: torch.Tensor, faces: torch.Tensor, K: torch.Tensor, H: int, W: int,
              S: int = 16, near: float = 1e-3, max_split_rounds: int = 8):
    """Rasterize a mesh.

    Args: v_cam (V, 3) camera-space vertices, faces (F, 3) long, K (3, 3).
    Returns: face_id (H, W) long (-1 = background), bary (H, W, 3) perspective-correct
             barycentrics w.r.t. the original face, depth (H, W) (inf = background).
    """
    dev, dt = v_cam.device, v_cam.dtype
    faces = faces.long()
    tri = v_cam[faces]                                           # (F, 3, 3)
    ok = (tri[..., 2] > near).all(1)
    faces = faces[ok]
    orig_id = torch.nonzero(ok).squeeze(1)
    corner_bary = torch.eye(3, device=dev, dtype=dt).expand(faces.shape[0], 3, 3).clone()
    orig_id_map = torch.arange(faces.shape[0], device=dev)       # sub-face -> index into `faces`

    def corners_cam():
        return torch.einsum("fkj,fjc->fkc", corner_bary, v_cam[faces[orig_id_map]])

    # iterative splitting of (on-screen) faces whose bbox exceeds S pixels
    for _ in range(max_split_rounds):
        cc = corners_cam()                                       # (F', 3, 3) camera coords
        uv = (cc @ K.T)[..., :2] / (cc[..., 2:3])
        lo = uv.amin(1).floor()
        hi = uv.amax(1).ceil()
        big = ((hi - lo) >= S).any(1) & (hi[:, 0] >= 0) & (hi[:, 1] >= 0) & (lo[:, 0] < W) & (lo[:, 1] < H)
        if not big.any():
            break
        corner_bary, orig_id_map = _split_faces(corner_bary, orig_id_map, big)

    cc = corners_cam()
    uv = (cc @ K.T)[..., :2] / cc[..., 2:3]                       # (F', 3, 2)
    z = cc[..., 2]                                               # (F', 3)
    lo = uv.amin(1).floor().clamp(min=0)
    lo[:, 0] = lo[:, 0].clamp(max=W - 1)
    lo[:, 1] = lo[:, 1].clamp(max=H - 1)
    hi_u = uv[..., 0].amax(1).ceil().clamp(max=W - 1)
    hi_v = uv[..., 1].amax(1).ceil().clamp(max=H - 1)

    oy, ox = torch.meshgrid(torch.arange(S, device=dev), torch.arange(S, device=dev), indexing="ij")
    offs = torch.stack([ox.reshape(-1), oy.reshape(-1)], 1).to(dt)          # (S*S, 2) (du, dv)
    px = lo[:, None, :] + offs[None]                                         # (F', S*S, 2)
    inb = (px[..., 0] <= hi_u[:, None]) & (px[..., 1] <= hi_v[:, None])

    a, b, c = uv[:, 0:1], uv[:, 1:2], uv[:, 2:3]                             # (F', 1, 2)

    def edge(p, q, r):  # signed area * 2 of (p, q, r)
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])

    area = edge(a, b, c)                                                     # (F', 1)
    valid_area = area.abs() > 1e-12
    w0 = edge(b, c, px) / area
    w1 = edge(c, a, px) / area
    w2 = 1.0 - w0 - w1
    eps = -1e-6
    inside = inb & valid_area & (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
    # perspective-correct depth
    iz = w0 / z[:, 0:1] + w1 / z[:, 1:2] + w2 / z[:, 2:3]
    depth = 1.0 / iz
    inside &= depth > near

    fidx = torch.arange(uv.shape[0], device=dev)[:, None].expand_as(inside)
    pix = (px[..., 1].long() * W + px[..., 0].long())
    pix, d, f = pix[inside], depth[inside], fidx[inside]
    nF = uv.shape[0]
    key = (d * 1e6).long().clamp(max=2**40) * (nF + 1) + f
    zbuf = torch.full((H * W,), torch.iinfo(torch.long).max, device=dev, dtype=torch.long)
    zbuf.scatter_reduce_(0, pix, key, reduce="amin")
    hit = zbuf != torch.iinfo(torch.long).max
    win = torch.full((H * W,), -1, device=dev, dtype=torch.long)
    win[hit] = zbuf[hit] % (nF + 1)

    # recompute barycentrics for the winning face of every covered pixel
    face_id = torch.full((H * W,), -1, device=dev, dtype=torch.long)
    bary = torch.zeros(H * W, 3, device=dev, dtype=dt)
    depth_map = torch.full((H * W,), float("inf"), device=dev, dtype=dt)
    hp = torch.nonzero(hit).squeeze(1)
    if len(hp):
        wf = win[hp]
        p = torch.stack([(hp % W).to(dt), (hp // W).to(dt)], 1)[:, None]    # (n, 1, 2)
        A, B, C = uv[wf, 0:1], uv[wf, 1:2], uv[wf, 2:3]
        ar = edge(A, B, C)
        s0 = (edge(B, C, p) / ar).squeeze(1)
        s1 = (edge(C, A, p) / ar).squeeze(1)
        s2 = 1 - s0 - s1
        zz = z[wf]
        q = torch.stack([s0 / zz[:, 0], s1 / zz[:, 1], s2 / zz[:, 2]], 1)
        q = q / q.sum(1, keepdim=True)                                       # perspective-correct
        # map sub-face barycentrics to the original face
        bary[hp] = torch.einsum("nk,nkj->nj", q, corner_bary[wf])
        face_id[hp] = orig_id[orig_id_map[wf]]
        depth_map[hp] = 1.0 / (s0 / zz[:, 0] + s1 / zz[:, 1] + s2 / zz[:, 2])
    return face_id.view(H, W), bary.view(H, W, 3), depth_map.view(H, W)


def interpolate(attr: torch.Tensor, faces: torch.Tensor, face_id: torch.Tensor, bary: torch.Tensor):
    """Per-vertex attribute (V, C) -> per-pixel (H, W, C); background pixels get 0."""
    H, W = face_id.shape
    out = torch.zeros(H, W, attr.shape[1], device=attr.device, dtype=attr.dtype)
    m = face_id >= 0
    fv = faces.long()[face_id[m]]                                  # (n, 3)
    out[m] = torch.einsum("nk,nkc->nc", bary[m].to(attr.dtype), attr[fv])
    return out


def vertex_normals(v: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    f = faces.long()
    fn = torch.linalg.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = torch.zeros_like(v).index_add_(0, f.reshape(-1), fn.repeat_interleave(3, 0))
    return vn / vn.norm(dim=1, keepdim=True).clamp_min(1e-12)
