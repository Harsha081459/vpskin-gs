"""A fake SMPL model with the real SMPL interface (6890 vertices, 24 joints, same keys, same
kinematic tree), built from capsules around the bones. For pipeline tests only: it lets
everything run before the real (licensed) SMPL model is downloaded. Never used for results.
"""
import pickle

import numpy as np
import scipy.sparse as sp

from vpskin.smpl_lite import SMPL_PARENTS

# rough T-pose joint locations (metres, y up), SMPL joint order
JOINTS = np.array([
    [0.00, 0.90, 0.0], [0.09, 0.82, 0.0], [-0.09, 0.82, 0.0], [0.00, 1.02, 0.0],
    [0.10, 0.45, 0.0], [-0.10, 0.45, 0.0], [0.00, 1.15, 0.0], [0.10, 0.08, 0.0],
    [-0.10, 0.08, 0.0], [0.00, 1.28, 0.0], [0.10, 0.02, 0.10], [-0.10, 0.02, 0.10],
    [0.00, 1.45, 0.0], [0.08, 1.40, 0.0], [-0.08, 1.40, 0.0], [0.00, 1.60, 0.0],
    [0.18, 1.40, 0.0], [-0.18, 1.40, 0.0], [0.45, 1.40, 0.0], [-0.45, 1.40, 0.0],
    [0.70, 1.40, 0.0], [-0.70, 1.40, 0.0], [0.80, 1.40, 0.0], [-0.80, 1.40, 0.0]], dtype=np.float64)
N_V = 6890


def _capsule(a, b, r, n_ring=12, n_len=10):
    d = b - a
    L = np.linalg.norm(d)
    z = d / L
    x = np.cross(z, [0, 0, 1.0]) if abs(z[2]) < 0.9 else np.cross(z, [1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    ts = np.linspace(0, 1, n_len)
    angs = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    V = np.array([a + t * d + r * (np.cos(g) * x + np.sin(g) * y) for t in ts for g in angs])
    F = []
    for i in range(n_len - 1):
        for j in range(n_ring):
            p, q = i * n_ring + j, i * n_ring + (j + 1) % n_ring
            F += [[p, q, p + n_ring], [q, q + n_ring, p + n_ring]]
    return V, np.array(F), ts


def make_mock_smpl(path: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    Vs, Fs, Ws = [], [], []
    off = 0
    for j in range(1, 24):
        p = SMPL_PARENTS[j]
        if p < 0:
            continue
        V, F, ts = _capsule(JOINTS[p], JOINTS[j], 0.04 if j < 16 else 0.03)
        W = np.zeros((len(V), 24))
        t = np.repeat(ts, len(V) // len(ts))
        W[:, p] = 0.5 * np.clip(1 - t / 0.3, 0, 1)     # 50/50 at the joint, child-only after 30%
        W[:, j] = 1 - W[:, p]
        Vs.append(V); Fs.append(F + off); Ws.append(W); off += len(V)
    V = np.concatenate(Vs); F = np.concatenate(Fs); W = np.concatenate(Ws)
    assert len(V) < N_V
    pad = N_V - len(V)                                     # unused vertices parked at the pelvis
    V = np.concatenate([V, np.repeat(JOINTS[:1], pad, 0)])
    W = np.concatenate([W, np.eye(24)[np.zeros(pad, int)]])
    # J_regressor: each joint = mean of the 12 ring vertices closest to it
    Jreg = np.zeros((24, N_V))
    for j in range(24):
        idx = np.argsort(np.linalg.norm(V[: N_V - pad] - JOINTS[j], axis=1))[:12]
        Jreg[j, idx] = 1 / 12
    kintree = np.array([[4294967295] + SMPL_PARENTS[1:], list(range(24))], dtype=np.int64)
    d = {
        "v_template": V, "f": F.astype(np.uint32), "weights": W,
        "shapedirs": 0.01 * rng.standard_normal((N_V, 3, 10)),
        "posedirs": 0.001 * rng.standard_normal((N_V, 3, 207)),
        "J_regressor": sp.csc_matrix(Jreg), "kintree_table": kintree,
    }
    with open(path, "wb") as f:
        pickle.dump(d, f, protocol=2)
    return path
