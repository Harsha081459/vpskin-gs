"""Convert a raw People-Snapshot subject into ZJU-MoCap(-refine) format + split.json (E2).

Raw input (graphics.tu-bs.de/people-snapshot), as read by InstantAvatar's preprocessing:
  <root>/<subject>/camera.pkl, <subject>.mp4, masks.hdf5, reconstructed_poses.hdf5
Optional --poses_npz: refined SMPL fits (e.g. Anim-NeRF's, used by 3DGS-Avatar) with keys
  {global_orient, body_pose, transl, betas} or {poses|thetas, trans|transl, betas}.

SMPL convention written: poses = full 72-d pose INCLUDING the global orientation, Rh = 0,
Th = translation, so world = SMPL(pose, betas) + trans (standard SMPL).
Split = 3DGS-Avatar's novel-pose split (configs/dataset/ps_*.yaml in mikeqzy/3dgs-avatar-release).

Usage:
  python tools/prep_peoplesnapshot.py --root /data/people_snapshot_public --subject female-3-casual \
      --smpl assets/SMPL_NEUTRAL.pkl --out data/people_snapshot/female-3-casual
"""
import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vpskin.smpl_lite import SMPLLite  # noqa: E402

# 3DGS-Avatar novel-pose splits: (train start, stop, step), (test start, stop, step)
SPLITS = {
    "male-3-casual": ((0, 456, 4), (456, 676, 4)),
    "male-4-casual": ((0, 660, 6), (660, 873, 6)),
    "female-3-casual": ((0, 446, 4), (446, 648, 4)),
    "female-4-casual": ((0, 336, 4), (335, 524, 4)),
}


def load_poses(raw_dir, poses_npz):
    if poses_npz:
        d = dict(np.load(poses_npz))
        if "global_orient" in d:
            poses = np.concatenate([d["global_orient"], d["body_pose"]], 1)
        else:
            poses = d.get("poses", d.get("thetas"))
        trans = d.get("transl", d.get("trans"))
        betas = d["betas"]
    else:
        import h5py
        with h5py.File(os.path.join(raw_dir, "reconstructed_poses.hdf5"), "r") as h:
            poses, trans, betas = np.asarray(h["pose"]), np.asarray(h["trans"]), np.asarray(h["betas"])
    betas = np.asarray(betas, np.float32).reshape(-1)[:10]
    return np.asarray(poses, np.float32).reshape(-1, 72), np.asarray(trans, np.float32).reshape(-1, 3), betas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--subject", required=True, choices=sorted(SPLITS))
    ap.add_argument("--smpl", default="assets/SMPL_NEUTRAL.pkl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--poses_npz", default="")
    ap.add_argument("--image_scaling", type=float, default=0.5, help="1080 -> 540, as in 3DGS-Avatar")
    a = ap.parse_args()
    raw = os.path.join(a.root, a.subject)
    with open(os.path.join(raw, "camera.pkl"), "rb") as f:
        cam = pickle.load(f, encoding="latin1")
    K = np.eye(3)
    K[0, 0], K[1, 1] = cam["camera_f"]
    K[:2, 2] = cam["camera_c"]
    dist = np.asarray(cam["camera_k"], np.float64)
    Rw = cv2.Rodrigues(np.asarray(cam["camera_rt"], np.float64))[0]
    tw = np.asarray(cam["camera_t"], np.float64).reshape(3, 1)

    poses, trans, betas = load_poses(raw, a.poses_npz)
    import h5py
    with h5py.File(os.path.join(raw, "masks.hdf5"), "r") as h:
        masks = np.asarray(h["masks"]).astype(np.uint8)
    smpl = SMPLLite(a.smpl)

    for d in ("images/00", "mask/00", "smpl_params", "smpl_vertices", "smpl_joints"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    capv = cv2.VideoCapture(os.path.join(raw, f"{a.subject}.mp4"))
    n = min(int(capv.get(cv2.CAP_PROP_FRAME_COUNT)), len(poses), len(masks))
    ims = []
    with torch.no_grad():
        for i in range(n):
            ok, frame = capv.read()
            if not ok:
                n = i
                break
            frame = cv2.undistort(frame, K, dist)
            m = cv2.undistort(masks[i], K, dist)
            cv2.imwrite(os.path.join(a.out, f"images/00/{i:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(a.out, f"mask/00/{i:06d}.png"), (m > 0).astype(np.uint8) * 255)
            v, J, _ = smpl.forward(torch.from_numpy(poses[i:i + 1]), torch.from_numpy(betas[None]))
            np.save(os.path.join(a.out, f"smpl_vertices/{i}.npy"), (v[0].numpy() + trans[i]).astype(np.float32))
            np.save(os.path.join(a.out, f"smpl_joints/{i}.npy"), (J[0].numpy() + trans[i]).astype(np.float32))
            np.save(os.path.join(a.out, f"smpl_params/{i}.npy"),
                    {"Rh": np.zeros((1, 3), np.float32), "Th": trans[i:i + 1], "shapes": betas[None],
                     "poses": poses[i:i + 1]})
            ims.append({"ims": [f"images/00/{i:06d}.jpg"]})
    # undistorted images -> zero distortion in the written calibration; T in millimetres
    cams = {"K": [K], "D": [np.zeros((5, 1))], "R": [Rw], "T": [tw * 1000.0]}
    np.save(os.path.join(a.out, "annots.npy"), {"cams": cams, "ims": ims})
    (tr, te) = SPLITS[a.subject]
    fr = lambda s: [f for f in range(*s) if f < n]  # noqa: E731
    json.dump({"image_scaling": a.image_scaling, "train": {"views": [0], "frames": fr(tr)},
               "test": {"views": [0], "frames": fr(te)}}, open(os.path.join(a.out, "split.json"), "w"), indent=1)
    print(f"{a.subject}: {n} frames, train {len(fr(tr))}, test (novel pose) {len(fr(te))} -> {a.out}")


if __name__ == "__main__":
    main()
