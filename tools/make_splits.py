"""Write the ZJU-MoCap novel-pose split files (E3, our extension of GauHuman's protocol).

GauHuman's published split (E1) tests 22 novel views on frames 0..480 (step 30), which are a
subset of its training frames (0..495, step 5). So it never tests novel poses. E3 keeps the
same training set and tests frames AFTER the training range on the same 22 views.

Usage: python tools/make_splits.py --zju_root data/zju_mocap_refine --out configs/splits
"""
import argparse
import json
import os

import numpy as np

SUBJECTS = ["my_377", "my_386", "my_387", "my_392", "my_393", "my_394"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zju_root", required=True)
    ap.add_argument("--out", default="configs/splits")
    ap.add_argument("--first_test", type=int, default=500)
    ap.add_argument("--max_test_frames", type=int, default=17, help="same count as GauHuman's test set")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    train = {"views": [4], "frames": {"start": 0, "stop": 500, "step": 5}}
    views = [v for v in range(23) if v != 4]
    for s in SUBJECTS:
        ann = os.path.join(a.zju_root, s, "annots.npy")
        if not os.path.exists(ann):
            print("skip (missing)", ann)
            continue
        n = len(np.load(ann, allow_pickle=True).item()["ims"])
        avail = list(range(a.first_test, n))
        if not avail:
            print(f"{s}: only {n} frames, no novel-pose frames after {a.first_test}")
            continue
        step = max(1, int(np.ceil(len(avail) / a.max_test_frames)))
        frames = avail[::step][: a.max_test_frames]
        sp = {"image_scaling": 0.5, "train": train, "test": {"views": views, "frames": frames}}
        path = os.path.join(a.out, f"zju_{s}_novelpose.json")
        json.dump(sp, open(path, "w"), indent=1)
        print(f"{s}: {n} frames -> {len(frames)} novel-pose test frames {frames[0]}..{frames[-1]} -> {path}")


if __name__ == "__main__":
    main()
