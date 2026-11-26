import json
import math
import os
import subprocess
import sys

import pytest
import torch

from vpskin.anim_eval import (axis_pose, chamfer_cm, hole_rate, joint_region, lbs_det_theory, near_mask,
                              sample_surface, twist_and_bend_axes)
from vpskin.skinning import dqs, lbs
from vpskin.smpl_lite import SMPLLite

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_chamfer_units_and_symmetry():
    a = torch.rand(500, 3)
    assert chamfer_cm(a, a)["chamfer"] < 1e-6
    b = a + torch.tensor([0.01, 0, 0])            # 1 cm shift
    assert abs(chamfer_cm(a, b)["chamfer"] - 1.0) < 0.35


def test_hole_rate():
    sil = torch.zeros(32, 32, dtype=torch.bool); sil[8:24, 8:24] = True
    alpha = torch.zeros(32, 32); alpha[8:24, 8:24] = 1.0
    assert hole_rate(alpha, sil) == 0.0
    alpha[14:18, 14:18] = 0.0                      # 16-pixel hole inside the eroded region (10x10)
    assert abs(hole_rate(alpha, sil, erode_px=3) - 16 / 100) < 1e-9


def test_surface_samples_on_mesh():
    v = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0.0]])
    p = sample_surface(v, torch.tensor([[0, 1, 2]]), 1000)
    assert (p[:, 2].abs() < 1e-6).all() and (p[:, 0] + p[:, 1] <= 1 + 1e-6).all() and (p >= -1e-6).all()


def test_twist_sweep_matches_prop1_on_mock_smpl(tmp_path):
    """Elbow twist on the mock SMPL: LBS volume factor in the 50/50 region follows Prop. 1; DQS stays 1."""
    from tests.mock_smpl import make_mock_smpl
    smpl = SMPLLite(make_mock_smpl(str(tmp_path / "m.pkl")), dtype=torch.float64)
    _, J, _ = smpl.forward(torch.zeros(1, 72, dtype=torch.float64), torch.zeros(1, 10, dtype=torch.float64))
    tw, bend = twist_and_bend_axes(J[0], 18, 20)
    assert abs(tw @ bend) < 1e-9
    W = smpl.weights
    half = ((W[:, 16] - 0.5).abs() < 1e-6) & ((W[:, 18] - 0.5).abs() < 1e-6)   # 50/50 elbow ring
    assert half.any() and joint_region(W, 16, 18)[half].all()
    for th in (0.0, math.pi / 2, math.pi):
        _, _, G = smpl.forward(axis_pose(18, tw, th), torch.zeros(1, 10, dtype=torch.float64))
        M_l = lbs(W[half], G[0])[2]
        M_d = dqs(W[half], G[0])[2]
        assert torch.allclose(torch.linalg.det(M_l), torch.full((int(half.sum()),), lbs_det_theory(0.5, th),
                                                                dtype=torch.float64), atol=1e-6)
        assert torch.allclose(torch.linalg.det(M_d), torch.ones(int(half.sum()), dtype=torch.float64), atol=1e-9)


def test_near_mask():
    j = torch.zeros(24, 3); j[18] = torch.tensor([1.0, 0, 0])
    p = torch.tensor([[1.05, 0, 0], [0.5, 0.5, 0.5]])
    assert near_mask(p, j).tolist() == [True, False]


def test_aggregate_on_fake_outputs(tmp_path):
    pytest.importorskip("pandas"); pytest.importorskip("matplotlib")
    out = tmp_path / "output"
    curve = [{"theta": t, "region_mean": 1.0, "region_min": 1.0, "limb_ratio": 1.0,
              "theory_w05": lbs_det_theory(0.5, math.radians(t))} for t in range(0, 181, 15)]
    for v, ps in (("B0", 30.0), ("A6", 30.1)):
        for s in ("my_377", "my_386"):
            d = out / "zju" / f"{v}_{s}_s0"; d.mkdir(parents=True)
            json.dump({"psnr": ps, "ssim": 0.96, "lpips_x1000": 31.0, "fps_sync": 100, "n_gaussians": 1e4, "tag": ""},
                      open(d / "metrics_test_1200.json", "w"))
            json.dump({"psnr": ps - 2, "ssim": 0.95, "lpips_x1000": 40.0, "tag": "zju_novelpose"},
                      open(d / "metrics_test_zju_novelpose_1200.json", "w"))
            json.dump({"E5": {"drift": 0.1 if v == "B0" else 0.05, "vol_dev": 0.2, "hole": 0.01},
                       "E6": {"l_elbow_twist": curve}, "E6_summary": {"twist180_region_mean": 0.0 if v == "B0" else 1.0}},
                      open(d / "eval_motion.json", "w"))
    res = tmp_path / "results"
    subprocess.run([sys.executable, "tools/aggregate.py", "--output", str(out), "--results", str(res)],
                   cwd=ROOT, check=True, capture_output=True)
    md = open(res / "tables.md", encoding="utf-8").read()
    assert "| B0 GauHuman (LBS) | 30.00" in md and "| A6 Full (C1+C2+C3) | 30.10" in md
    assert (res / "figures" / "l_elbow_twist.png").exists()
