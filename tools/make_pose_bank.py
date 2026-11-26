"""Build the AMASS pose bank (regulariser) and the held-out test motions (evaluation).

The two use DISJOINT AMASS subsets (PHASE2_plan §4):
    regulariser : CMU                (individual poses, every 10th frame at 30 fps)
    test        : ACCAD, BMLmovi     (whole sequences, cropped to --test_len frames)

Usage:
    python tools/make_pose_bank.py --amass_root /path/to/AMASS --out data/amass
Outputs:
    data/amass/pose_bank_reg.npy        (P, 72) float32
    data/amass/test_motions.npz         {sequence_name: (T, 72)}
    data/amass/train_motions.npz        {sequence_name: (T, 72)}  (from the regulariser subsets;
                                        used only to drive the synthetic training video)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vpskin.pose_prior import build_pose_bank, iter_amass_files, load_amass_sequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amass_root", required=True)
    ap.add_argument("--out", default="data/amass")
    ap.add_argument("--reg_subsets", nargs="+", default=["CMU"])
    ap.add_argument("--test_subsets", nargs="+", default=["ACCAD", "BMLmovi"])
    ap.add_argument("--reg_stride", type=int, default=10)
    ap.add_argument("--reg_max", type=int, default=50000)
    ap.add_argument("--n_test", type=int, default=20, help="number of held-out test sequences")
    ap.add_argument("--test_len", type=int, default=120, help="frames per test sequence (30 fps)")
    ap.add_argument("--n_train_motions", type=int, default=2)
    ap.add_argument("--train_len", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    assert not set(a.reg_subsets) & set(a.test_subsets), "regulariser and test subsets must be disjoint"
    os.makedirs(a.out, exist_ok=True)

    bank = build_pose_bank(a.amass_root, a.reg_subsets, a.reg_stride, a.reg_max, a.seed)
    np.save(os.path.join(a.out, "pose_bank_reg.npy"), bank)
    print(f"regulariser bank: {bank.shape} from {a.reg_subsets}")

    rng = np.random.default_rng(a.seed)
    tr = {}
    tr_files = list(iter_amass_files(a.amass_root, a.reg_subsets))
    rng.shuffle(tr_files)
    for sub, f in tr_files:
        seq = load_amass_sequence(f)
        if seq is None or len(seq) < a.train_len:
            continue
        s0 = (len(seq) - a.train_len) // 2
        tr[f"{sub}/{os.path.splitext(os.path.basename(f))[0]}"] = seq[s0:s0 + a.train_len]
        if len(tr) == a.n_train_motions:
            break
    np.savez(os.path.join(a.out, "train_motions.npz"), **tr)
    print(f"train motions (synthetic only): {list(tr)}")

    files = list(iter_amass_files(a.amass_root, a.test_subsets))
    rng.shuffle(files)
    motions = {}
    for sub, f in files:
        seq = load_amass_sequence(f)
        if seq is None or len(seq) < a.test_len:
            continue
        s0 = (len(seq) - a.test_len) // 2           # centre crop (skips T-pose calibration starts)
        name = f"{sub}/{os.path.splitext(os.path.relpath(f, os.path.join(a.amass_root, sub)))[0]}"
        motions[name.replace(os.sep, "/")] = seq[s0:s0 + a.test_len]
        if len(motions) == a.n_test:
            break
    np.savez(os.path.join(a.out, "test_motions.npz"), **motions)
    print(f"test motions: {len(motions)} x {a.test_len} frames from {a.test_subsets}")
    for k in motions:
        print("  ", k)


if __name__ == "__main__":
    main()
