"""Pose sources for the image-free regularisers (C3) and for unseen-motion evaluation (E5).

SMPL pose vector convention (as GauHuman / ZJU-MoCap): (72,) axis-angle for 24 joints.
Joint 0 (root) is zeroed: global rotation is rigid for every Gaussian, so it changes
neither neighbour distances nor det(M). It is irrelevant to L_iso / L_vol.

AMASS (SMPL-H or SMPL-X npz) stores `poses` as (T, 156|165): root 0:3, 21 body joints 3:66,
then hands. SMPL's joints 22/23 (hands) have no AMASS body counterpart, so they are set to 0.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch

SMPL_DIM = 72


def amass_to_smpl72(poses: np.ndarray) -> np.ndarray:
    """(T, >=66) AMASS poses -> (T, 72) SMPL poses with zero root and zero hands."""
    poses = np.asarray(poses, dtype=np.float32)
    assert poses.ndim == 2 and poses.shape[1] >= 66, poses.shape
    out = np.zeros((poses.shape[0], SMPL_DIM), dtype=np.float32)
    out[:, 3:66] = poses[:, 3:66]
    return out


def iter_amass_files(root: str, subsets: list[str]):
    for sub in subsets:
        for f in sorted(glob.glob(os.path.join(root, sub, "**", "*.npz"), recursive=True)):
            if f.endswith("shape.npz"):
                continue
            yield sub, f


def load_amass_sequence(path: str, target_fps: float = 30.0) -> np.ndarray | None:
    d = np.load(path, allow_pickle=True)
    if "poses" not in d:
        return None
    fps = float(d.get("mocap_framerate", d.get("mocap_frame_rate", target_fps)))
    stride = max(1, int(round(fps / target_fps)))
    return amass_to_smpl72(d["poses"][::stride])


def build_pose_bank(root: str, subsets: list[str], frame_stride: int = 10, max_poses: int | None = None,
                    seed: int = 0) -> np.ndarray:
    """Pool of individual poses from the given AMASS subsets (for training-time sampling)."""
    bank = []
    for _, f in iter_amass_files(root, subsets):
        seq = load_amass_sequence(f)
        if seq is not None and len(seq):
            bank.append(seq[::frame_stride])
    bank = np.concatenate(bank, 0) if bank else np.zeros((0, SMPL_DIM), np.float32)
    if max_poses is not None and len(bank) > max_poses:
        bank = bank[np.random.default_rng(seed).choice(len(bank), max_poses, replace=False)]
    return bank


class PoseSampler:
    """Draws one SMPL pose (1, 72) per call for the regulariser.

    mode: 'train'       -> a random training-frame pose (3DGS-Avatar-style isometry at training poses)
          'train_noise' -> training pose + N(0, std) on body joints (3DGS-Avatar pose augmentation)
          'amass'       -> a random pose from an AMASS bank disjoint from the test motions (ours)
    """

    def __init__(self, mode: str, train_poses: torch.Tensor | None = None, bank_path: str = "",
                 noise_std: float = 0.1, device: str = "cuda", seed: int = 0):
        assert mode in ("train", "train_noise", "amass"), mode
        self.mode, self.noise_std, self.device = mode, noise_std, device
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        if mode == "amass":
            assert bank_path and os.path.exists(bank_path), f"pose bank not found: {bank_path!r}"
            bank = torch.from_numpy(np.load(bank_path).astype(np.float32))
            assert bank.ndim == 2 and bank.shape[1] == SMPL_DIM, bank.shape
            self.poses = bank
        else:
            assert train_poses is not None and len(train_poses), "need training poses"
            self.poses = train_poses.detach().float().cpu().reshape(-1, SMPL_DIM)
        self.poses = self.poses.clone()
        self.poses[:, :3] = 0.0

    def __len__(self):
        return len(self.poses)

    def sample(self) -> torch.Tensor:
        i = torch.randint(len(self.poses), (1,), generator=self.g).item()
        p = self.poses[i:i + 1].clone()
        if self.mode == "train_noise":
            p[:, 3:] += self.noise_std * torch.randn(p[:, 3:].shape, generator=self.g)
        return p.to(self.device)
