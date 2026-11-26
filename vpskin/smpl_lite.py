"""Small torch SMPL forward pass (LBS + shape/pose blend shapes), loadable from any .pkl path.

Matches GauHuman's conventions (smpl/smpl_numpy.py): joints = posed joint locations,
vertices = LBS(v_template + shape + pose correctives). No chumpy, no smplx package.
"""
from __future__ import annotations

import numpy as np
import torch

SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
JOINT_NAMES = ["pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2", "l_ankle", "r_ankle",
               "spine3", "l_foot", "r_foot", "neck", "l_collar", "r_collar", "head", "l_shoulder",
               "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hand", "r_hand"]


def rodrigues(aa: torch.Tensor) -> torch.Tensor:
    """(N, 3) axis-angle -> (N, 3, 3) (Rodrigues' formula, L8)."""
    th = aa.norm(dim=1, keepdim=True).clamp_min(1e-12)
    k = aa / th
    kx, ky, kz = k.unbind(1)
    z = torch.zeros_like(kx)
    Kx = torch.stack([z, -kz, ky, kz, z, -kx, -ky, kx, z], 1).view(-1, 3, 3)
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    I = torch.eye(3, dtype=aa.dtype, device=aa.device).expand_as(Kx)
    return I + s * Kx + (1 - c) * (Kx @ Kx)


class SMPLLite:
    def __init__(self, path: str, device="cpu", dtype=torch.float32):
        from tools.clean_smpl import load_clean
        d = load_clean(path)
        J = d["J_regressor"]
        J = np.asarray(J.todense()) if hasattr(J, "todense") else np.asarray(J)
        t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=dtype, device=device)  # noqa: E731
        self.v_template = t(d["v_template"])
        self.shapedirs = t(np.asarray(d["shapedirs"])[..., :10])
        self.posedirs = t(d["posedirs"])
        self.J_regressor = t(J)
        self.weights = t(d["weights"])
        self.faces = torch.as_tensor(np.asarray(d["f"], dtype=np.int64), device=device)
        kt = np.asarray(d["kintree_table"]).astype(np.int64)
        id_to_col = {int(kt[1, i]): i for i in range(kt.shape[1])}
        self.parents = [-1] + [id_to_col[int(kt[0, i])] for i in range(1, kt.shape[1])]
        self.device, self.dtype = device, dtype

    def forward(self, poses: torch.Tensor, betas: torch.Tensor):
        """poses (B, 72), betas (B, 10) -> verts (B, V, 3), joints (B, 24, 3), bone transforms (B, 24, 4, 4)."""
        B = poses.shape[0]
        poses, betas = poses.to(self.device, self.dtype), betas.to(self.device, self.dtype)
        v_shaped = self.v_template[None] + torch.einsum("vcl,bl->bvc", self.shapedirs, betas)
        J = torch.einsum("jv,bvc->bjc", self.J_regressor, v_shaped)
        R = rodrigues(poses.reshape(-1, 3)).view(B, 24, 3, 3)
        I = torch.eye(3, dtype=self.dtype, device=self.device)
        pose_feat = (R[:, 1:] - I).reshape(B, -1)
        v_posed = v_shaped + torch.einsum("vck,bk->bvc", self.posedirs, pose_feat)
        G = []
        for i in range(24):
            Ti = torch.eye(4, dtype=self.dtype, device=self.device).repeat(B, 1, 1)
            Ti[:, :3, :3] = R[:, i]
            Ti[:, :3, 3] = J[:, i] - (J[:, self.parents[i]] if i > 0 else 0)
            G.append(Ti if i == 0 else G[self.parents[i]] @ Ti)
        G = torch.stack(G, 1)                                       # (B, 24, 4, 4)
        joints = G[:, :, :3, 3].clone()
        G = G.clone()
        G[:, :, :3, 3] -= torch.einsum("bjcd,bjd->bjc", G[:, :, :3, :3], J)
        T = torch.einsum("vj,bjcd->bvcd", self.weights, G)
        v = torch.einsum("bvcd,bvd->bvc", T[:, :, :3, :3], v_posed) + T[:, :, :3, 3]
        return v, joints, G
