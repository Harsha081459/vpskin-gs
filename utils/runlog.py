"""Run logger: peak VRAM, wall-clock, GPU-hours for every run (rubric requirement).

Usage:
    from utils.runlog import RunLog
    with RunLog("B0_zju377", config=vars(args)) as rl:
        ... train ...
        rl.log(psnr=..., n_gaussians=...)
Appends one row to results/runs.csv (created with header on first use).
"""
from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone

try:
    import torch
except ImportError:  # logger must never break a run
    torch = None

FIELDS = ["timestamp_utc", "run", "git_commit", "gpu", "n_gpus", "wall_clock_s", "gpu_hours",
          "peak_alloc_mb", "peak_reserved_mb", "status", "metrics", "config"]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


class RunLog:
    def __init__(self, run: str, config: dict | None = None, csv_path: str = "results/runs.csv",
                 n_gpus: int | None = None):
        self.run, self.config, self.csv_path = run, config or {}, csv_path
        self.metrics: dict = {}
        cuda = torch is not None and torch.cuda.is_available()
        self.cuda = cuda
        self.n_gpus = n_gpus if n_gpus is not None else (1 if cuda else 0)
        self.gpu = torch.cuda.get_device_name(0) if cuda else f"cpu:{platform.processor() or 'unknown'}"

    def log(self, **kw):
        self.metrics.update(kw)

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.cuda:
            torch.cuda.synchronize()
        wall = time.perf_counter() - self.t0
        peak_a = torch.cuda.max_memory_allocated() / 2**20 if self.cuda else 0.0
        peak_r = torch.cuda.max_memory_reserved() / 2**20 if self.cuda else 0.0
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run": self.run,
            "git_commit": _git_commit(),
            "gpu": self.gpu,
            "n_gpus": self.n_gpus,
            "wall_clock_s": round(wall, 2),
            "gpu_hours": round(wall / 3600 * self.n_gpus, 4),
            "peak_alloc_mb": round(peak_a, 1),
            "peak_reserved_mb": round(peak_r, 1),
            "status": "ok" if exc_type is None else f"error:{exc_type.__name__}",
            "metrics": json.dumps(self.metrics, default=float),
            "config": json.dumps(self.config, default=str),
        }
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        new = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"[runlog] {self.run}: {row['wall_clock_s']} s, peak {row['peak_alloc_mb']} MB "
              f"(reserved {row['peak_reserved_mb']} MB), {row['gpu_hours']} GPU-h, {row['status']}")
        return False  # never swallow exceptions
