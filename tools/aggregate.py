"""Collect every run under output/ into the report tables (T1-T3, seeds) and figures.

Reads metrics_test_*.json (render.py), eval_motion.json (tools/eval_motion.py) and
results/runs.csv (utils/runlog.py). Writes results/tables.md, results/tables.tex,
results/all_runs.csv and results/figures/*.png.

Usage: python tools/aggregate.py [--output output] [--results results]
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

ORDER = ["B0", "B1", "A1", "A2", "A3", "A3n", "A4", "A6", "A7"]
NAMES = {"B0": "GauHuman (LBS)", "B1": "Quat. avg. [2509.11411]", "A1": "DQS (C1)", "A2": "LBS + iso@train",
         "A3": "LBS + iso@AMASS (C3)", "A3n": "LBS + iso@noise", "A4": "DQS + iso@AMASS", "A6": "Full (C1+C2+C3)",
         "A7": "Full, reg@train"}


def parse_exp(ds, name):
    vidtag = name.split("_")[0]
    vid = vidtag.split("-")[0]
    m = re.match(r".*_s(\d+)$", name)
    seed = int(m.group(1)) if m else 0
    subj = name[len(vidtag) + 1: name.rfind("_s")]
    return {"dataset": ds, "variant": vid, "tag": vidtag[len(vid):], "subject": subj, "seed": seed, "exp": f"{ds}/{name}"}


def collect(out_dir):
    rows = []
    for d in sorted(glob.glob(os.path.join(out_dir, "*", "*"))):
        if not os.path.isdir(d):
            continue
        ds, name = d.replace("\\", "/").split("/")[-2:]
        r = parse_exp(ds, name)
        for f in glob.glob(os.path.join(d, "metrics_test*.json")):
            j = json.load(open(f))
            key = "np" if j.get("tag") else "test"          # tagged = novel-pose split (E3)
            for k in ("psnr", "ssim", "lpips_x1000", "fps_sync", "n_gaussians"):
                r[f"{key}_{k}"] = j.get(k)
        em = os.path.join(d, "eval_motion.json")
        if os.path.exists(em):
            j = json.load(open(em))
            for k, v in j.get("E5", {}).items():
                r[f"E5_{k}"] = v
            for k, v in j.get("E4", {}).items():
                r[f"E4_{k}"] = v
            r["E6_twist180"] = j.get("E6_summary", {}).get("twist180_region_mean")
            r["E6_radius180"] = j.get("E6_summary", {}).get("twist180_radius_ratio")
            r["beta_mean"] = j.get("beta", {}).get("mean")
            r["_E6"] = j.get("E6")
        rows.append(r)
    return pd.DataFrame(rows)


def fmt(m, s=None, nd=2):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "–"
    return f"{m:.{nd}f}" if s is None or np.isnan(s) or s == 0 else f"{m:.{nd}f}±{s:.{nd}f}"


def table(df, cols, title, seed0=True):
    """Variant x metric table, mean over subjects (seed 0 unless seed0=False)."""
    d = df[df.seed == 0] if seed0 else df
    lines = [f"### {title}", "", "| Variant | " + " | ".join(c[0] for c in cols) + " |",
             "|---|" + "---|" * len(cols)]
    tex = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
           "Variant & " + " & ".join(c[0] for c in cols) + r" \\", r"\midrule"]
    for v in [v for v in ORDER if v in set(d.variant)]:
        cells = []
        for _, ds, col, nd in cols:
            x = d[(d.variant == v) & (d.dataset == ds)] if ds else d[d.variant == v]
            x = x[col].dropna() if col in x else pd.Series(dtype=float)
            cells.append(fmt(x.mean(), None, nd) if len(x) else "–")
        lines.append(f"| {v} {NAMES.get(v, '')} | " + " | ".join(cells) + " |")
        tex.append(f"{v} & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n", "\n".join(tex) + "\n"


def cost_table(runs):
    if runs is None or runs.empty:
        return "### T3 Cost\n\n(no runs.csv)\n", ""
    r = runs[runs.run.str.startswith("train:")].copy()
    r["exp"] = r.run.str[len("train:"):].str.replace("\\", "/")
    r["variant"] = r.exp.str.split("/").str[-1].str.split("_").str[0].str.split("-").str[0]
    g = r.groupby("variant").agg(train_s=("wall_clock_s", "mean"), gpu_h=("gpu_hours", "sum"),
                                 peak_mb=("peak_alloc_mb", "max"), runs=("run", "count"))
    lines = ["### T3 Cost (training runs)", "", "| Variant | runs | mean train time (s) | total GPU-h | peak VRAM (MB) |",
             "|---|---|---|---|---|"]
    for v in [v for v in ORDER if v in g.index]:
        x = g.loc[v]
        lines.append(f"| {v} | {int(x.runs)} | {x.train_s:.0f} | {x.gpu_h:.3f} | {x.peak_mb:.0f} |")
    tot = runs.gpu_hours.sum()
    lines.append(f"\nAll logged steps (train + render + eval): **{tot:.2f} GPU-h**, max peak VRAM "
                 f"**{runs.peak_alloc_mb.max():.0f} MB** on {', '.join(sorted(set(runs.gpu)))}.\n")
    return "\n".join(lines) + "\n", ""


def seeds_table(df):
    d = df[df.variant.isin(["B0", "A6"])]
    if d.seed.nunique() < 2:
        return ""
    cols = [c for c in ("test_psnr", "np_psnr", "E5_drift", "E5_vol_dev", "E5_hole", "E4_joint_chamfer") if c in d]
    lines = ["### Seeds (B0 vs A6, mean ± std over seeds of the subject mean)", "",
             "| Variant | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for v in ("B0", "A6"):
        x = d[d.variant == v].groupby("seed")[cols].mean()
        lines.append(f"| {v} | " + " | ".join(fmt(x[c].mean(), x[c].std(), 3) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def plots(df, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df[(df.seed == 0) & df["_E6"].notna()] if "_E6" in df else df.iloc[:0]
    if d.empty:
        return []
    os.makedirs(fig_dir, exist_ok=True)
    made = []
    for key in ("l_elbow_twist", "l_elbow_bend", "l_knee_twist", "l_shoulder_twist"):
        fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=150)
        theory = None
        for v in [v for v in ORDER if v in set(d.variant)]:
            curves = [r[key] for r in d[d.variant == v]["_E6"] if r and key in r]
            if not curves:
                continue
            th = [p["theta"] for p in curves[0]]
            y = np.nanmean([[p["region_mean"] for p in c] for c in curves], 0)
            ax.plot(th, y, marker="o", ms=2.5, lw=1.4, label=v)
            theory = (th, [p["theory_w05"] for p in curves[0]])
        if theory and key.endswith("twist"):
            ax.plot(*theory, "k--", lw=1, label="Prop. 1 (w=½)")
        ax.set_xlabel(f"{key.replace('_', ' ')} angle (deg)")
        ax.set_ylabel("mean |det M| in joint region")
        ax.set_ylim(-0.02, 1.15)
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        p = os.path.join(fig_dir, f"{key}.png")
        fig.savefig(p)
        plt.close(fig)
        made.append(p)
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="output")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows consoles
    except Exception:
        pass
    df = collect(a.output)
    os.makedirs(a.results, exist_ok=True)
    if df.empty:
        print("no runs found under", a.output)
        return
    df.drop(columns=[c for c in df if c.startswith("_")]).to_csv(os.path.join(a.results, "all_runs.csv"), index=False)
    runs_p = os.path.join(a.results, "runs.csv")
    runs = pd.read_csv(runs_p) if os.path.exists(runs_p) else None
    main_df = df[df.tag == ""]

    t1 = [("ZJU-NV PSNR↑", "zju", "test_psnr", 2), ("SSIM↑", "zju", "test_ssim", 4), ("LPIPS*↓", "zju", "test_lpips_x1000", 2),
          ("ZJU-NP PSNR↑", "zju", "np_psnr", 2), ("LPIPS*↓", "zju", "np_lpips_x1000", 2),
          ("PS-NP PSNR↑", "ps", "test_psnr", 2), ("LPIPS*↓", "ps", "test_lpips_x1000", 2),
          ("Syn PSNR↑", "synthetic", "test_psnr", 2), ("LPIPS*↓", "synthetic", "test_lpips_x1000", 2)]
    t2 = [("Tear↓", None, "E5_tear", 4), ("Drift med↓", None, "E5_drift_median", 4),
          ("|1−V|↓", None, "E5_vol_dev", 4), ("Hole↓", None, "E5_hole", 4),
          ("Twist180 V↑", None, "E6_twist180", 3), ("Twist180 r↑", None, "E6_radius180", 3),
          ("Syn CD↓ cm", "synthetic", "E4_all_chamfer", 3),
          ("Syn CD joints↓ cm", "synthetic", "E4_joint_chamfer", 3)]
    md1, tex1 = table(main_df, t1, "T1 Image metrics (mean over subjects, seed 0)")
    md2, tex2 = table(main_df, t2, "T2 Animation & geometry (E4–E6, mean over subjects/datasets, seed 0)")
    md3, _ = cost_table(runs)
    md4 = seeds_table(main_df)
    figs = plots(main_df, os.path.join(a.results, "figures"))
    with open(os.path.join(a.results, "tables.md"), "w", encoding="utf-8") as f:
        f.write("# Results (auto-generated by tools/aggregate.py)\n\n" + md1 + "\n" + md2 + "\n" + md3 + "\n" + md4)
        f.write("\nFigures: " + ", ".join(os.path.basename(p) for p in figs) + "\n")
    with open(os.path.join(a.results, "tables.tex"), "w", encoding="utf-8") as f:
        f.write("% T1\n" + tex1 + "\n% T2\n" + tex2)
    print(open(os.path.join(a.results, "tables.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
