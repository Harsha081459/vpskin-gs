"""Run the pre-registered ablation (PHASE2_plan §3) end to end.

For every (variant, subject, seed): train -> render (E1/E2/E4 image metrics on the dataset's
test split) -> render on the novel-pose split (E3, ZJU only) -> eval_motion (E5/E6; E4 geometry
for synthetic). Every step logs peak VRAM / wall-clock / GPU-hours to results/runs.csv.
Finished steps are skipped (their output files exist), so the script can be re-run after a
Kaggle session dies.

Examples:
  python tools/run_experiments.py --datasets synthetic --variants B0 A1 A4 A6 --dry_run
  python tools/run_experiments.py --datasets zju --variants all
  python tools/run_experiments.py --datasets zju --subjects my_377 --variants B0 A6 --seeds 0 1 2
  python tools/run_experiments.py --sweep            # lambda / k sensitivity on synthetic validation
"""
import argparse
import itertools
import os
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- frozen defaults (lambdas fixed on validation before any test run; see NOTES.md) ----
LAMBDA_ISO = 0.1
LAMBDA_VOL = 0.01
REG = "--lambda_iso {li:g} --lambda_vol {lv:g} --reg_knn {k}"

VARIANTS = {   # id: (skin_mode, reg_poses, use_iso, use_vol)
    "B0":  ("lbs",     "none",        False, False),   # GauHuman baseline
    "B1":  ("quatavg", "none",        False, False),   # arXiv 2509.11411-style rotation blending
    "A1":  ("dqs",     "none",        False, False),   # C1 alone
    "A2":  ("lbs",     "train",       True,  False),   # isometry at training poses (3DGS-Avatar style)
    "A3":  ("lbs",     "amass",       True,  False),   # C3: isometry at unseen AMASS poses
    "A3n": ("lbs",     "train_noise", True,  False),   # pose-noise augmentation instead of a motion prior
    "A4":  ("dqs",     "amass",       True,  False),   # C1 + C3
    "A6":  ("blend",   "amass",       True,  True),    # full method (C1 + C2 + C3)
    "A7":  ("blend",   "train",       True,  True),    # full method without unseen-pose sampling
}

DATASETS = {
    "zju": {"subjects": ["my_377", "my_386", "my_387", "my_392", "my_393", "my_394"],
            "src": "data/zju_mocap_refine/{s}", "iters": 1200,
            "novel_pose_split": "configs/splits/zju_{s}_novelpose.json", "gt": False},
    "ps": {"subjects": ["male-3-casual", "male-4-casual", "female-3-casual", "female-4-casual"],
           "src": "data/people_snapshot/{s}", "iters": 2000, "novel_pose_split": None, "gt": False},
    "synthetic": {"subjects": ["subj0"], "src": "data/synthetic/{s}", "iters": 1200,
                  "novel_pose_split": None, "gt": True},
}
COMMON = "--eval --motion_offset_flag --smpl_type smpl --actor_gender neutral"


def sh(cmd, dry):
    print("$", cmd, flush=True)
    if not dry:
        r = subprocess.run(shlex.split(cmd), cwd=ROOT)
        if r.returncode != 0:
            print(f"!! failed ({r.returncode}): {cmd}", flush=True)
            return False
    return True


def run_one(ds_name, subj, vid, seed, a, li=LAMBDA_ISO, lv=LAMBDA_VOL, k=8, tag="", extra_args=""):
    ds = DATASETS[ds_name]
    skin, poses, iso, vol = VARIANTS[vid]
    exp = f"{ds_name}/{vid}{tag}_{subj}_s{seed}"
    out = os.path.join(ROOT, "output", exp)
    src = ds["src"].format(s=subj)
    it = a.iters or ds["iters"]
    reg = ""
    if poses != "none":
        reg = f"--reg_poses {poses} " + REG.format(li=li if iso else 0.0, lv=lv if vol else 0.0, k=k)
        if poses == "amass":
            reg += f" --pose_bank {a.pose_bank}"
    extra = (f"--split_file {a.split_file} " if a.split_file else "") + extra_args
    ok = True
    if not os.path.exists(os.path.join(out, "point_cloud", f"iteration_{it}")):
        ok = sh(f"{sys.executable} train.py -s {src} --exp_name {exp} {COMMON} --skin_mode {skin} "
                f"--iterations {it} --seed {seed} {reg} {extra}", a.dry_run)
    if ok and not os.path.exists(os.path.join(out, f"metrics_test_{it}.json")):
        ok = sh(f"{sys.executable} render.py -m output/{exp} {COMMON} --iteration {it} --skip_train", a.dry_run)
    if ok and ds["novel_pose_split"] and not a.split_file:
        sp = ds["novel_pose_split"].format(s=subj)
        tagn = os.path.splitext(os.path.basename(sp))[0]
        if not os.path.exists(os.path.join(out, f"metrics_test_{tagn}_{it}.json")):
            ok = sh(f"{sys.executable} render.py -m output/{exp} {COMMON} --iteration {it} --skip_train "
                    f"--split_file {sp}", a.dry_run)
    if ok and not a.no_motion and not os.path.exists(os.path.join(out, "eval_motion.json")):
        gt = "--gt" if ds["gt"] else ""
        ok = sh(f"{sys.executable} tools/eval_motion.py -m output/{exp} --iteration {it} "
                f"--motions {a.motions} {gt}", a.dry_run)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["synthetic"], choices=list(DATASETS))
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--variants", nargs="+", default=["all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--iters", type=int, default=0, help="override iterations (e.g. smoke tests)")
    ap.add_argument("--pose_bank", default="data/amass/pose_bank_reg.npy")
    ap.add_argument("--motions", default="data/amass/test_motions.npz")
    ap.add_argument("--split_file", default="", help="override split (e.g. synthetic split_val.json for tuning)")
    ap.add_argument("--no_motion", action="store_true", help="skip eval_motion (E4-E6)")
    ap.add_argument("--sweep", action="store_true", help="sensitivity sweep of A6 on synthetic validation")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    variants = list(VARIANTS) if a.variants == ["all"] else a.variants
    fails = []

    if a.sweep:   # PHASE2_plan §3 sensitivity: validation split only, never test
        a.split_file = a.split_file or "data/synthetic/subj0/split_val.json"
        for mult, k in itertools.chain(((m, 8) for m in (0.1, 1.0, 10.0)), ((1.0, kk) for kk in (4, 16))):
            if not run_one("synthetic", "subj0", "A6", 0, a, li=LAMBDA_ISO * mult, lv=LAMBDA_VOL * mult, k=k,
                           tag=f"-sweep-x{mult}-k{k}"):
                fails.append(f"sweep x{mult} k{k}")
        # blend-field initialisation: beta=0.5 (default) vs start-at-LBS
        if not run_one("synthetic", "subj0", "A6", 0, a, tag="-sweep-binit-4", extra_args="--beta_init -4"):
            fails.append("sweep beta_init -4")
        print("failures:", fails or "none")
        sys.exit(1 if fails else 0)

    for ds in a.datasets:
        subs = a.subjects or DATASETS[ds]["subjects"]
        for subj, vid, seed in itertools.product(subs, variants, a.seeds):
            if not run_one(ds, subj, vid, seed, a):
                fails.append(f"{ds}/{vid}_{subj}_s{seed}")
    print("failures:", fails or "none")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
