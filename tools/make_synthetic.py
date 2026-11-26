"""Synthetic 3D-ground-truth subject (E4): textured SMPL driven by AMASS, written in
ZJU-MoCap(-refine) format plus split.json, so GauHuman's loader trains on it unchanged.

Ground truth: SMPL *with* its pose-corrective blend shapes (learned from real scans), so the
target deformation is data-driven, not plain LBS. (ASSUMPTION pending TA, Q9.)

Texture: licence-free procedural "solid" texture evaluated per pixel on the interpolated
canonical (T-pose) surface point: 5 cm 3D checker x body-part colour. Twisting a limb
visibly rotates the checker, so skinning errors are observable photometrically.
Shading: albedo x (ambient + diffuse), with the light fixed in the world frame.

Layout written (ZJU-refine conventions; T in millimetres, world = R_h x + Th):
  annots.npy, images/VV/FFFFFF.jpg, mask/VV/FFFFFF.png, smpl_params/F.npy,
  smpl_vertices/F.npy, smpl_joints/F.npy, split.json, split_val.json
Frames: [0, n_train) training motion (camera 0, slow 360 deg yaw spin so one camera sees
all sides); then validation and test motions seen by orbit cameras 1..n_views.

Usage:
  python tools/make_synthetic.py --smpl assets/SMPL_NEUTRAL.pkl --motions data/amass \
         --out data/synthetic/subj0 --res 512
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vpskin.raster import interpolate, look_at, rasterize, vertex_normals  # noqa: E402
from vpskin.smpl_lite import SMPLLite, rodrigues  # noqa: E402

PART_COLOURS = np.array([[0.85, 0.33, 0.10], [0.93, 0.69, 0.13], [0.49, 0.18, 0.56], [0.47, 0.67, 0.19],
                         [0.30, 0.75, 0.93], [0.64, 0.08, 0.18], [0.20, 0.40, 0.80], [0.90, 0.50, 0.70]])


def solid_texture(p_canon: torch.Tensor, part: torch.Tensor, cell: float = 0.05) -> torch.Tensor:
    """(n, 3) canonical points, (n,) part ids -> (n, 3) albedo."""
    c = torch.floor(p_canon / cell).long().sum(1) % 2
    base = torch.as_tensor(PART_COLOURS, dtype=p_canon.dtype, device=p_canon.device)[part % len(PART_COLOURS)]
    return base * (0.55 + 0.45 * c[:, None].to(p_canon.dtype))


def render_view(smpl, verts_w, R, t, K, H, W, canon, part_v, light=(0.3, -0.8, -0.5)):
    v_cam = verts_w @ R.T + t
    fid, bary, _ = rasterize(v_cam, smpl.faces, K, H, W)
    m = fid >= 0
    img = torch.zeros(H, W, 3, dtype=verts_w.dtype)
    if m.any():
        pc = interpolate(canon, smpl.faces, fid, bary)[m]
        # part of the pixel = part of the face's first vertex (faces rarely straddle parts)
        part = part_v[smpl.faces[fid[m], 0]]
        alb = solid_texture(pc, part)
        n = interpolate(vertex_normals(verts_w, smpl.faces), smpl.faces, fid, bary)[m]
        n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-9)
        L = torch.as_tensor(light, dtype=verts_w.dtype); L = -L / L.norm()
        shade = 0.65 + 0.35 * (n @ L).clamp(min=0)
        img[m] = (alb * shade[:, None]).clamp(0, 1)
    return img.numpy(), m.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smpl", required=True)
    ap.add_argument("--motions", required=True, help="dir with train_motions.npz and test_motions.npz (tools/make_pose_bank.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--focal", type=float, default=1.15, help="focal length in units of image width")
    ap.add_argument("--n_train", type=int, default=100)
    ap.add_argument("--n_views", type=int, default=8)
    ap.add_argument("--n_test_motions", type=int, default=4)
    ap.add_argument("--test_stride", type=int, default=10, help="frame stride inside each test motion")
    ap.add_argument("--n_val", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    torch.set_grad_enabled(False)
    dt = torch.float64

    smpl = SMPLLite(a.smpl, dtype=dt)
    betas = torch.as_tensor(rng.normal(0, 0.8, (1, 10)), dtype=dt)
    canon, _, _ = smpl.forward(torch.zeros(1, 72, dtype=dt), betas)
    canon = canon[0]
    part_v = smpl.weights.argmax(1)

    trm = np.load(os.path.join(a.motions, "train_motions.npz"))
    tem = np.load(os.path.join(a.motions, "test_motions.npz"))
    train_seq = trm[sorted(trm.files)[0]]
    assert len(train_seq) >= a.n_train, f"training motion too short ({len(train_seq)} < {a.n_train})"
    train_poses = train_seq[np.linspace(0, len(train_seq) - 1, a.n_train).astype(int)]
    val_seq = trm[sorted(trm.files)[1]] if len(trm.files) > 1 else train_seq[::-1]
    val_poses = val_seq[np.linspace(0, len(val_seq) - 1, a.n_val).astype(int)]
    test_names = sorted(tem.files)[: a.n_test_motions]
    test_poses = np.concatenate([tem[k][:: a.test_stride] for k in test_names])

    # frame table: (pose72, root yaw, camera views)
    H = W = a.res
    f = a.focal * W
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=dt)
    frames = []
    for i, p in enumerate(train_poses):
        frames.append((p, 2 * math.pi * i / a.n_train, [0], "train"))
    for p in val_poses:
        frames.append((p, rng.uniform(0, 2 * math.pi), list(range(1, a.n_views + 1)), "val"))
    for p in test_poses:
        frames.append((p, rng.uniform(0, 2 * math.pi), list(range(1, a.n_views + 1)), "test"))

    # cameras: 0 = training camera (front, 3.6 m); 1..n = orbit at 3.6 m, 45 deg apart
    centre = np.array([0.0, float(canon[:, 1].mean()), 0.0])
    cams = {"K": [], "D": [], "R": [], "T": []}
    for v in range(a.n_views + 1):
        az = 0.0 if v == 0 else 2 * math.pi * (v - 1) / a.n_views + math.pi / a.n_views
        eye = centre + 3.6 * np.array([math.sin(az), 0.15, math.cos(az)])
        R, t = look_at(eye, centre)
        cams["K"].append(K.numpy()); cams["D"].append(np.zeros((5, 1)))
        cams["R"].append(R.numpy()); cams["T"].append((t.numpy() * 1000.0)[:, None])   # mm

    for d in ("smpl_params", "smpl_vertices", "smpl_joints"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    ims = []
    for fi, (p, yaw, views, kind) in enumerate(frames):
        pose = torch.as_tensor(p, dtype=dt)[None].clone()
        pose[:, :3] = 0
        Rh = np.array([[0.0, yaw, 0.0]])
        Rw = rodrigues(torch.as_tensor(Rh, dtype=dt))[0]
        v, J, _ = smpl.forward(pose, betas)
        vw = v[0] @ Rw.T            # world = R_h x + Th, Th = 0
        Jw = J[0] @ Rw.T
        np.save(os.path.join(a.out, "smpl_vertices", f"{fi}.npy"), vw.numpy().astype(np.float32))
        np.save(os.path.join(a.out, "smpl_joints", f"{fi}.npy"), Jw.numpy().astype(np.float32))
        np.save(os.path.join(a.out, "smpl_params", f"{fi}.npy"),
                {"Rh": Rh.astype(np.float32), "Th": np.zeros((1, 3), np.float32),
                 "shapes": betas.numpy().astype(np.float32), "poses": pose.numpy().astype(np.float32)})
        row = []
        for v_id in range(a.n_views + 1):
            rel = f"images/{v_id:02d}/{fi:06d}.jpg"
            row.append(rel)
            if v_id not in views:
                continue
            img, msk = render_view(smpl, vw, torch.as_tensor(cams["R"][v_id]), torch.as_tensor(cams["T"][v_id][:, 0] / 1000.0),
                                   K, H, W, canon, part_v)
            os.makedirs(os.path.join(a.out, "images", f"{v_id:02d}"), exist_ok=True)
            os.makedirs(os.path.join(a.out, "mask", f"{v_id:02d}"), exist_ok=True)
            cv2.imwrite(os.path.join(a.out, rel), (img[..., ::-1] * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(a.out, rel.replace("images", "mask").replace(".jpg", ".png")), msk.astype(np.uint8) * 255)
        ims.append({"ims": row, "kind": kind})
        if fi % 20 == 0:
            print(f"frame {fi}/{len(frames)} ({kind})")

    np.save(os.path.join(a.out, "annots.npy"), {"cams": cams, "ims": ims})
    idx = {k: [i for i, fr in enumerate(frames) if fr[3] == k] for k in ("train", "val", "test")}
    test_views = list(range(1, a.n_views + 1))
    base = {"image_scaling": 1.0, "train": {"views": [0], "frames": idx["train"]}}
    json.dump({**base, "test": {"views": test_views, "frames": idx["test"]}}, open(os.path.join(a.out, "split.json"), "w"), indent=1)
    json.dump({**base, "test": {"views": test_views, "frames": idx["val"]}}, open(os.path.join(a.out, "split_val.json"), "w"), indent=1)
    json.dump({"betas": betas.numpy().tolist(), "train_motion": sorted(trm.files)[0], "test_motions": test_names,
               "frames": {k: len(v) for k, v in idx.items()}, "res": a.res, "seed": a.seed},
              open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print("done:", {k: len(v) for k, v in idx.items()}, "->", a.out)


if __name__ == "__main__":
    main()
