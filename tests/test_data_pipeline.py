"""CPU end-to-end check of the data tools with a MOCK SMPL and FAKE AMASS motions:
make_pose_bank -> make_synthetic -> GauHuman's ZJU loader via split.json.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tests.mock_smpl import make_mock_smpl  # noqa: E402


def fake_amass(root, subsets, n_files=3, T=400, seed=0):
    rng = np.random.default_rng(seed)
    for s in subsets:
        os.makedirs(os.path.join(root, s, "seq"), exist_ok=True)
        for k in range(n_files):
            t = np.linspace(0, 4 * np.pi, T)[:, None]
            phase, amp = rng.uniform(0, 6, (1, 156)), rng.uniform(0, 0.4, (1, 156))
            np.savez(os.path.join(root, s, "seq", f"m{k}_poses.npz"),
                     poses=(amp * np.sin(t + phase)).astype(np.float32), mocap_framerate=60.0)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipe")
    smpl = make_mock_smpl(str(tmp / "SMPL_MOCK.pkl"))
    fake_amass(str(tmp / "amass"), ["CMU", "ACCAD", "BMLmovi"])
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=ROOT, check=True, capture_output=True, text=True)  # noqa: E731
    run("tools/make_pose_bank.py", "--amass_root", str(tmp / "amass"), "--out", str(tmp / "motions"),
        "--n_test", "2", "--test_len", "60", "--train_len", "150")
    run("tools/make_synthetic.py", "--smpl", smpl, "--motions", str(tmp / "motions"), "--out", str(tmp / "syn"),
        "--res", "64", "--n_train", "6", "--n_views", "2", "--n_test_motions", "1", "--test_stride", "30", "--n_val", "2")
    return tmp, smpl


def test_pose_bank(data):
    tmp, _ = data
    bank = np.load(tmp / "motions" / "pose_bank_reg.npy")
    assert bank.shape[1] == 72 and len(bank) > 0
    assert np.allclose(bank[:, :3], 0) and np.allclose(bank[:, 66:], 0)    # root and hands zeroed
    tm = np.load(tmp / "motions" / "test_motions.npz")
    assert all(k.split("/")[0] in ("ACCAD", "BMLmovi") for k in tm.files)   # disjoint from CMU


def test_synthetic_layout(data):
    tmp, _ = data
    syn = tmp / "syn"
    sp = json.load(open(syn / "split.json"))
    ann = np.load(syn / "annots.npy", allow_pickle=True).item()
    assert len(ann["ims"]) == 6 + 2 + 2                                      # train + val + test frames
    assert sp["train"]["views"] == [0] and sp["train"]["frames"] == list(range(6))
    for f in sp["test"]["frames"]:
        for v in sp["test"]["views"]:
            rel = ann["ims"][f]["ims"][v]
            assert os.path.exists(syn / rel)
            m = __import__("cv2").imread(str(syn / rel.replace("images", "mask").replace(".jpg", ".png")), 0)
            assert m.max() == 255 and (m > 0).mean() > 0.01                  # the body is visible
    p = np.load(syn / "smpl_params" / "0.npy", allow_pickle=True).item()
    assert p["poses"].shape == (1, 72) and p["Rh"].shape == (1, 3)


def test_gauhuman_loader_reads_split(data, monkeypatch):
    """GauHuman's own ZJU reader (via our split-file path) must parse the synthetic set."""
    tmp, smpl = data
    import types
    for name in ("simple_knn", "simple_knn._C"):          # CUDA-only extension, unused by the reader
        if name not in sys.modules:
            m = types.ModuleType(name); m.distCUDA2 = None
            monkeypatch.setitem(sys.modules, name, m)
    try:
        import scene.dataset_readers as dr  # imports plyfile etc.
    except Exception as e:  # pragma: no cover - CPU box without GauHuman deps
        pytest.skip(f"GauHuman reader deps missing: {e}")
    os.makedirs(tmp / "assets", exist_ok=True)
    import shutil
    shutil.copy(smpl, tmp / "assets" / "SMPL_NEUTRAL.pkl")
    monkeypatch.chdir(tmp)
    cams = dr.readCamerasZJUMoCapRefine(str(tmp / "syn"), [1, 2], False, image_scaling=1.0, split="test",
                                        frames=json.load(open(tmp / "syn" / "split.json"))["test"]["frames"])
    assert len(cams) == 2 * 2
    c = cams[0]
    assert c.width == 64 and c.smpl_param["poses"].shape == (1, 72)
