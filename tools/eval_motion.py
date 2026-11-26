"""Animation evaluation of a trained model (E4 geometry, E5 unseen motions, E6 twist/bend).

Usage (after train.py; needs CUDA + the dataset the model was trained on):
  python tools/eval_motion.py -m output/zju_mocap_refine/A6_my_377 --motions data/amass/test_motions.npz
  python tools/eval_motion.py -m output/synthetic/A6_subj0 --motions data/amass/test_motions.npz --gt

Writes <model>/eval_motion.json and appends cost to results/runs.csv.
Unseen motions are deformed WITHOUT GauHuman's pose-correction MLP (it was fit to the
training frames' SMPL estimates); the identical rule applies to every variant.
"""
import json
import math
import os
import sys
import time
from argparse import ArgumentParser

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import Scene, GaussianModel  # noqa: E402
from scene.cameras import MiniCam  # noqa: E402
from utils.graphics_utils import getWorld2View2, getProjectionMatrix_refine  # noqa: E402
from utils.runlog import RunLog  # noqa: E402
from vpskin.anim_eval import (TWIST_JOINTS, axis_pose, chamfer_cm, hole_rate, joint_region,  # noqa: E402
                              lbs_det_theory, near_mask, sample_surface, twist_and_bend_axes)
from vpskin.raster import look_at, rasterize  # noqa: E402
from vpskin.regularizers import drift_stats, knn_graph, part_volume_ratio  # noqa: E402
from vpskin.smpl_lite import SMPLLite  # noqa: E402


def make_cam(R, t, K, H, W):
    w2v = torch.tensor(getWorld2View2(R.T, t)).transpose(0, 1).cuda().float()   # GauHuman stores R transposed
    proj = getProjectionMatrix_refine(torch.tensor(K).cuda().float(), H, W, 0.001, 1000).transpose(0, 1)
    full = (w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    fx = K[0, 0]
    fov = 2 * math.atan(W / (2 * fx))
    return MiniCam(W, H, fov, fov, 0.001, 1000, w2v, full)


class Deformer:
    def __init__(self, g, cam0):
        self.g, self.cam0 = g, cam0
        self.R = torch.eye(3, device="cuda")
        self.Th = torch.zeros(1, 3, device="cuda")

    @torch.no_grad()
    def __call__(self, pose, shapes=None, R=None, Th=None, correct=False):
        g, c = self.g, self.cam0
        xyz = g.get_xyz
        params = {"poses": pose.cuda().float(), "shapes": c.smpl_param["shapes"] if shapes is None else shapes,
                  "R": self.R if R is None else R, "Th": self.Th if Th is None else Th}
        lbs_w = g.lweight_offset_decoder(xyz[None]).permute(0, 2, 1) if g.motion_offset_flag else None
        cR = g.pose_decoder(params["poses"][:, 3:])["Rs"] if (correct and g.motion_offset_flag) else None
        _, world, bw, tr, tl = g.coarse_deform_c2source(xyz[None], params, c.big_pose_smpl_param,
                                                        c.big_pose_world_vertex[None], lbs_weights=lbs_w,
                                                        correct_Rs=cR, return_transl=True)
        return world[0], tr, tl, g.last_transforms_cov, bw[0]


def main():
    ap = ArgumentParser()
    model = ModelParams(ap, sentinel=True)
    pipeline = PipelineParams(ap)
    ap.add_argument("--iteration", default=-1, type=int)
    ap.add_argument("--motions", default="data/amass/test_motions.npz")
    ap.add_argument("--frame_stride", type=int, default=4)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--n_views", type=int, default=4)
    ap.add_argument("--knn", type=int, default=16, help="evaluation k (training uses 8)")
    ap.add_argument("--gt", action="store_true", help="synthetic data: Chamfer vs GT meshes on test frames")
    ap.add_argument("--no_twist", action="store_true")
    ap.add_argument("--out", default="")
    args = get_combined_args(ap)
    dataset, pipe = model.extract(args), pipeline.extract(args)
    torch.set_grad_enabled(False)

    with RunLog("eval_motion:" + dataset.model_path, config={"motions": args.motions}) as rl:
        g = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender,
                          skin_mode=getattr(dataset, "skin_mode", "lbs"), beta_init=getattr(dataset, "beta_init", 0.0))
        scene = Scene(dataset, g, load_iteration=args.iteration, shuffle=False)
        cam0 = scene.getTrainCameras()[0]
        deform = Deformer(g, cam0)
        smpl = SMPLLite("assets/SMPL_NEUTRAL.pkl", device="cuda")
        betas = cam0.smpl_param["shapes"].reshape(1, -1)[:, :10]
        bg = torch.zeros(3, device="cuda")
        out = {"model": dataset.model_path, "skin_mode": g.skin_mode, "n_gaussians": int(g.get_xyz.shape[0])}

        xyz0 = g.get_xyz
        edges, rest = knn_graph(xyz0, k=args.knn)
        opac = g.get_opacity.reshape(-1)
        scales = g.get_scaling
        _, _, _, _, W0 = deform(torch.zeros(1, 72))
        part = W0.argmax(1)

        # ---------------- E5: unseen AMASS motions ----------------
        motions = np.load(args.motions)
        H = Wd = args.res
        f = 1.15 * Wd
        K = np.array([[f, 0, Wd / 2], [0, f, H / 2], [0, 0, 1.0]])
        per_motion, fps_t, fps_n = {}, 0.0, 0
        for name in motions.files:
            seq = torch.as_tensor(motions[name][:: args.frame_stride]).float()
            acc = {"drift": [], "drift_median": [], "tear": [], "vol_dev": [], "vol_global": [], "hole": []}
            for p in seq:
                p = p[None].clone(); p[:, :3] = 0
                pts, tr, tl, tcov, _ = deform(p)
                for k_, v_ in drift_stats(pts, edges, rest).items():
                    acc[k_].append(v_)
                r = part_volume_ratio(opac, scales, tcov[0], part, 24)
                acc["vol_dev"].append(torch.nanmean((r - 1).abs()).item())
                acc["vol_global"].append((opac * scales.prod(1) * torch.linalg.det(tcov[0]).abs()).sum().item()
                                         / (opac * scales.prod(1)).sum().item())
                v, _, _ = smpl.forward(p.cuda(), betas)
                centre = v[0].mean(0).cpu().double()
                for k in range(args.n_views):
                    az = 2 * math.pi * k / args.n_views
                    eye = centre + 3.6 * torch.tensor([math.sin(az), 0.1, math.cos(az)], dtype=torch.float64)
                    Rc, tc = look_at(eye, centre)
                    cam = make_cam(Rc.numpy(), tc.numpy(), K, H, Wd)
                    torch.cuda.synchronize(); t0 = time.time()
                    ro = render(cam, g, pipe, bg, transforms=tr, translation=tl, transforms_cov=tcov)
                    torch.cuda.synchronize(); fps_t += time.time() - t0; fps_n += 1
                    vc = v[0] @ Rc.cuda().float().T + tc.cuda().float()
                    fid, _, _ = rasterize(vc, smpl.faces, torch.tensor(K).cuda().float(), H, Wd)
                    acc["hole"].append(hole_rate(ro["render_alpha"][0], fid >= 0))
            per_motion[name] = {k: float(np.nanmean(v)) for k, v in acc.items()}
            per_motion[name]["drift_max"] = float(np.max(acc["drift"]))
            print(f"[E5] {name}: " + ", ".join(f"{k}={v:.4f}" for k, v in per_motion[name].items()))
        keys = ["drift", "drift_median", "tear", "drift_max", "vol_dev", "vol_global", "hole"]
        out["E5"] = {k: float(np.mean([m[k] for m in per_motion.values()])) for k in keys}
        out["E5"]["fps_render_sync"] = fps_n / max(fps_t, 1e-9)
        out["E5_per_motion"] = per_motion

        # ---------------- E6: twist / bend sweeps from the T-pose ----------------
        if not args.no_twist:
            _, Jt, _ = smpl.forward(torch.zeros(1, 72, device="cuda"), betas)
            Jt = Jt[0]
            sweeps = {}
            pts_t, _, _, _, _ = deform(torch.zeros(1, 72))          # T-pose centres (reference)
            for jn, (j, child, parent) in TWIST_JOINTS.items():
                ax_t, ax_b = twist_and_bend_axes(Jt, j, child)
                region = joint_region(W0, parent, j)
                limb = (part == parent) | (part == j) | (part == child)

                def radius(p, a=ax_t.float(), c=Jt[j].float()):
                    # distance of centres from the twisting bone's axis (candy-wrapper shrinks it)
                    d = p - c
                    return (d - (d @ a)[:, None] * a).norm(dim=1)
                r0 = radius(pts_t)[region].mean() if region.any() else None
                for kind, ax, thetas in (("twist", ax_t, range(0, 181, 15)), ("bend", ax_b, range(0, 151, 15))):
                    rows = []
                    for th in thetas:
                        pts_th, _, _, tcov, _ = deform(axis_pose(j, ax, math.radians(th)).cpu())
                        fac = torch.linalg.det(tcov[0]).abs()
                        v0 = opac * scales.prod(1)
                        rows.append({
                            "theta": th,
                            "region_mean": fac[region].mean().item() if region.any() else float("nan"),
                            "region_min": fac[region].min().item() if region.any() else float("nan"),
                            "limb_ratio": ((v0 * fac)[limb].sum() / v0[limb].sum()).item(),
                            "theory_w05": lbs_det_theory(0.5, math.radians(th)),
                            # centres: radius about the bone axis vs T-pose (only meaningful for twist)
                            "radius_ratio": (radius(pts_th)[region].mean() / r0).item()
                                            if (kind == "twist" and r0 is not None) else float("nan"),
                        })
                    sweeps[f"{jn}_{kind}"] = rows
            out["E6"] = sweeps
            t180 = [s[-1]["region_mean"] for k, s in sweeps.items() if k.endswith("twist")]
            r180 = [s[-1]["radius_ratio"] for k, s in sweeps.items() if k.endswith("twist")]
            out["E6_summary"] = {"twist180_region_mean": float(np.nanmean(t180)),
                                 "twist180_radius_ratio": float(np.nanmean(r180))}
            print(f"[E6] 180 deg twist: covariance volume factor {out['E6_summary']['twist180_region_mean']:.4f}, "
                  f"centre radius ratio {out['E6_summary']['twist180_radius_ratio']:.4f}")
        if g.beta_field is not None:                     # C2 diagnostics: where did the blend go?
            b = g.get_beta(g.get_xyz)
            out["beta"] = {"mean": b.mean().item(), "frac_dqs": (b > 0.5).float().mean().item(),
                           "region_mean": {jn: b[joint_region(W0, p_, j_)].mean().item()
                                           for jn, (j_, _, p_) in TWIST_JOINTS.items()}}
            print(f"[C2] beta mean {out['beta']['mean']:.3f}, fraction > 0.5: {out['beta']['frac_dqs']:.3f}")

        # ---------------- E4: synthetic GT geometry on test frames ----------------
        if args.gt:
            root = dataset.source_path
            seen, rows = set(), []
            for c in scene.getTestCameras():
                if c.pose_id in seen:
                    continue
                seen.add(c.pose_id)
                fr = int(os.path.basename(c.image_name))
                gtv = torch.as_tensor(np.load(os.path.join(root, "smpl_vertices", f"{fr}.npy"))).cuda().float()
                gtj = torch.as_tensor(np.load(os.path.join(root, "smpl_joints", f"{fr}.npy"))).cuda().float()
                pts, _, _, _, _ = deform(c.smpl_param["poses"], c.smpl_param["shapes"], c.smpl_param["R"],
                                         c.smpl_param["Th"], correct=True)
                pts = pts[opac > 0.5]
                gts = sample_surface(gtv, smpl.faces, 30000, seed=fr)
                allm = chamfer_cm(pts, gts)
                nj = chamfer_cm(pts[near_mask(pts, gtj)], gts[near_mask(gts, gtj)])
                rows.append({"frame": fr, **{f"all_{k}": v for k, v in allm.items()},
                             **{f"joint_{k}": v for k, v in nj.items()}})
            out["E4"] = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0] if k != "frame"}
            out["E4"]["n_frames"] = len(rows)
            print(f"[E4] Chamfer all {out['E4']['all_chamfer']:.3f} cm, near joints {out['E4']['joint_chamfer']:.3f} cm")

        rl.log(**{f"E5_{k}": v for k, v in out["E5"].items()})
        path = args.out or os.path.join(dataset.model_path, "eval_motion.json")
        with open(path, "w") as fh:
            json.dump(out, fh, indent=1)
        print("wrote", path)


if __name__ == "__main__":
    main()
