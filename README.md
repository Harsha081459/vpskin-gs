# VP-Skin: volume-preserving rigid skinning for articulated 3D Gaussians

AID 836 (3D Computer Vision, IIIT Bangalore) final project, Problem Statement 2.
This is a fork of [GauHuman](https://github.com/skhu101/GauHuman) at commit `732ec0d`, which is our baseline. GauHuman's original README is [README_GauHuman.md](README_GauHuman.md).
GauHuman is under the S-Lab License 1.0 (non-commercial), which this fork keeps. Our additions live in `vpskin/`, `tools/`, `tests/`, `notebooks/` and `utils/runlog.py`.

> Status: **Day 2 of the plan (setup).** Skinning maths and logger are unit-tested. No experimental results yet.

## Idea in one paragraph
GauHuman skins each Gaussian with linear blend skinning (LBS): M = Σ w_j R_j, μ' = Mμ + t, Σ' = MΣMᵀ.
Each Gaussian's volume therefore scales by |det M|.
For two bones with relative rotation θ about any axis and weights (1−w, w), that factor is **det M = 1 − 2w(1−w)(1 − cos θ)**. It reaches 0 at w = ½, θ = π: the candy-wrapper collapse, measured per Gaussian.
We contribute:
1. **C1:** dual-quaternion skinning of means and covariances. The motion is rigid, so volume is preserved exactly.
2. **C2:** a learned per-Gaussian blend between LBS and DQS.
3. **C3:** image-free isometry and volume penalties evaluated at **unseen AMASS poses** during training.

## Layout
| Path | What |
|---|---|
| `vpskin/skinning.py` | `lbs`, `dqs`, `blend`, `quatavg` (arXiv 2509.11411 comparator), `volume_factor` |
| `vpskin/regularizers.py` | `knn_graph`, `iso_loss`, `vol_loss`, `beta_smooth_loss`, plus metrics `neighbour_drift` and `part_volume_ratio` |
| `vpskin/knn_fallback.py` | pure-torch stand-in for `knn_cuda` |
| `utils/runlog.py` | peak VRAM, wall-clock, GPU-hours → `results/runs.csv` (hooked into `train.py` and `render.py`) |
| `tools/clean_smpl.py` | strips chumpy from the official SMPL pickle, so no chumpy install is needed |
| `tests/` | `pytest` checks of Props 1–2, DQS sanity, regularisers, logger |
| `notebooks/vpskin_kaggle.ipynb` | end-to-end Kaggle notebook (T4, 16 GB) |

## Setup
**Data.** Licences forbid redistribution, so request each dataset yourself:
- SMPL (neutral model, from the SMPLify download): https://smplify.is.tue.mpg.de. Then run `python tools/clean_smpl.py <basicModel_neutral_…pkl> assets/SMPL_NEUTRAL.pkl`.
- ZJU-MoCap-refine: follow [Instant-NVR's instructions](https://github.com/zju3dv/instant-nvr/blob/master/docs/install.md#set-up-datasets). This needs the ZJU-MoCap request form. Then link it as `data/zju_mocap_refine`.
- AMASS (for unseen motions): https://amass.is.tue.mpg.de

**Kaggle (recommended).** Upload the data above as *private* datasets. Open `notebooks/vpskin_kaggle.ipynb`, choose GPU T4, set `REPO_URL`, and run all cells.

**Local (Linux or WSL2).**
```bash
pip install torch torchvision          # any recent CUDA build
pip install --no-build-isolation submodules/diff-gaussian-rasterization submodules/simple-knn
pip install plyfile lpips==0.1.4 tqdm imageio opencv-python pytest
python -m pytest -q tests              # the skinning tests also run on CPU-only torch
```

## Baseline reproduction (B0)
```bash
bash train_zju_mocap_refine.sh && bash eval_zju_mocap_refine.sh
```
- Target (GauHuman paper, Table 1, average of 6 subjects): **PSNR 31.34, SSIM 0.965, LPIPS\* 30.51** (LPIPS-VGG × 1000).
- The metric code is GauHuman's, unchanged. We only add `torch.cuda.synchronize()` so the FPS figure measures GPU time.

## Results
To be filled in from `results/runs.csv` and `output/**/metrics_*.json`.
