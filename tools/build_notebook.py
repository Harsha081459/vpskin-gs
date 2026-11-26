"""Generate notebooks/vpskin_kaggle.ipynb (run: python tools/build_notebook.py)."""
import json
import os

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md(r"""
# VP-Skin: reproducible Kaggle notebook (AID 836, Problem Statement 2)
**Baseline:** GauHuman (CVPR'24) @ `732ec0d`. **Ours:** volume-preserving rigid skinning (DQS for means *and*
covariances), a learned LBS↔DQS blend field, and image-free isometry/volume regularisation at unseen AMASS poses.

**Accelerator:** GPU T4 ×1 (16 GB). Every train/render/eval step appends peak VRAM, wall-clock and GPU-hours to
`results/runs.csv`.

**Inputs.** Licences forbid redistribution, so each user creates these as *private* Kaggle datasets from their own
downloads (see README):

| Kaggle dataset | Contents | Needed for |
|---|---|---|
| `smpl-models` | `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` (from SMPLify) | everything |
| `amass-npz` | AMASS SMPL-H npz folders `CMU/`, `ACCAD/`, `BMLmovi/` | regulariser + unseen motions |
| `zju-mocap-refine` | `my_377 … my_394` (Instant-NVR release) | E1, E3 |
| `people-snapshot` | raw `male-3-casual/ …` folders | E2 |
| `vpskin-wheels` (optional) | CUDA wheels exported by a previous run | fast install |
""")
code(r"""
# ---------------- configuration ----------------
MODE     = "smoke"      # smoke | synthetic | zju | ps | all
VARIANTS = ["all"]      # or e.g. ["B0", "A1", "A4", "A6"]
SEEDS    = [0]          # main table; use [0, 1, 2] for the B0-vs-A6 seed study
REPO_URL = "https://github.com/Harsha081459/vpskin-gs.git"
BRANCH   = "vpskin"
INPUT, WORK = "/kaggle/input", "/kaggle/working"
USE_FAKE_AMASS = False  # smoke test without AMASS only. NEVER for reported results.
""")
code(r"""
import subprocess, sys, os, glob, torch
print(sys.version)
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print(subprocess.run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; nvcc --version | tail -1",
                     shell=True, capture_output=True, text=True).stdout)
""")
code(r"""
%cd {WORK}
!git clone -q -b {BRANCH} {REPO_URL} vpskin-gs 2>/dev/null || (cd vpskin-gs && git pull -q)
%cd {WORK}/vpskin-gs
!git log --oneline -1
""")
code(r"""
# ---------------- install: cached wheels or a one-time build (~10 min) ----------------
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"          # T4
cached = glob.glob(f"{INPUT}/vpskin-wheels/*.whl")
if cached:
    !pip install -q {" ".join(cached)}
else:
    !pip wheel -q --no-build-isolation --no-deps submodules/diff-gaussian-rasterization submodules/simple-knn -w {WORK}/wheels
    !pip install -q {WORK}/wheels/*.whl
!pip install -q plyfile lpips==0.1.4 h5py
import diff_gaussian_rasterization, simple_knn; print("CUDA extensions OK")
""")
code(r"""
# ---------------- data preparation ----------------
os.makedirs("assets", exist_ok=True); os.makedirs("data", exist_ok=True)
smpl_src = (glob.glob(f"{INPUT}/smpl-models/**/basicModel_neutral*.pkl", recursive=True) +
            glob.glob(f"{INPUT}/smpl-models/**/SMPL_NEUTRAL.pkl", recursive=True))[0]
!python tools/clean_smpl.py "{smpl_src}" assets/SMPL_NEUTRAL.pkl

if USE_FAKE_AMASS:   # smoke test only
    from tests.test_data_pipeline import fake_amass
    fake_amass(f"{WORK}/fake_amass", ["CMU", "ACCAD", "BMLmovi"]); amass_root = f"{WORK}/fake_amass"
else:
    amass_root = glob.glob(f"{INPUT}/amass-npz*")[0]
if not os.path.exists("data/amass/pose_bank_reg.npy"):
    !python tools/make_pose_bank.py --amass_root "{amass_root}" --out data/amass

if MODE in ("smoke", "synthetic", "all") and not os.path.exists("data/synthetic/subj0/split.json"):
    !python tools/make_synthetic.py --smpl assets/SMPL_NEUTRAL.pkl --motions data/amass --out data/synthetic/subj0 --res 512

if MODE in ("zju", "all"):
    !ln -sfn {INPUT}/zju-mocap-refine data/zju_mocap_refine
    !python tools/make_splits.py --zju_root data/zju_mocap_refine --out configs/splits

if MODE in ("ps", "all"):
    ps_root = glob.glob(f"{INPUT}/people-snapshot*")[0]
    for s in ["male-3-casual", "male-4-casual", "female-3-casual", "female-4-casual"]:
        if not os.path.exists(f"data/people_snapshot/{s}/split.json"):
            !python tools/prep_peoplesnapshot.py --root "{ps_root}" --subject {s} --out data/people_snapshot/{s}
""")
code(r"""
!python -m pytest -q tests
""")
md(r"""
## Experiments
`tools/run_experiments.py` runs every (variant, subject, seed): train → render (E1/E2/E4 image metrics) →
novel-pose render (E3) → `eval_motion` (E4 geometry, E5 unseen AMASS motions, E6 twist/bend). Steps that already
finished are skipped, so you can re-run after a session timeout.
""")
code(r"""
v = " ".join(VARIANTS); s = " ".join(map(str, SEEDS))
if MODE == "smoke":
    !python tools/run_experiments.py --datasets synthetic --variants B0 A6 --iters 300
else:
    ds = {"synthetic": "synthetic", "zju": "zju", "ps": "ps", "all": "synthetic zju ps"}[MODE]
    !python tools/run_experiments.py --datasets {ds} --variants {v} --seeds {s}
""")
code(r"""
!python tools/aggregate.py
from IPython.display import Image, display
for f in sorted(glob.glob("results/figures/*.png")):
    display(Image(f))
""")
code(r"""
# ---------------- export (then 'Save Version' to keep /kaggle/working) ----------------
!mkdir -p {WORK}/export && cp -r results {WORK}/export/ && (cp -r {WORK}/wheels {WORK}/export/ 2>/dev/null || true)
!cd output && find . -name "*.json" | tar -czf {WORK}/export/metrics_json.tgz -T -
!ls -la {WORK}/export
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "kaggle": {"accelerator": "gpu", "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
os.makedirs("notebooks", exist_ok=True)
json.dump(nb, open("notebooks/vpskin_kaggle.ipynb", "w"), indent=1)
print("wrote notebooks/vpskin_kaggle.ipynb with", len(cells), "cells")
