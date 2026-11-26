"""Convert an official SMPL .pkl (which contains chumpy objects) into a plain-numpy .pkl.

chumpy does not install on Python >= 3.11 / numpy >= 1.24, so we unpickle with a stub
that turns every chumpy array into its underlying numpy value. No chumpy needed.

Usage:
    python tools/clean_smpl.py basicModel_neutral_lbs_10_207_0_v1.0.0.pkl assets/SMPL_NEUTRAL.pkl
"""
import argparse
import pickle

import numpy as np


class _ChStub:
    """Stands in for chumpy.Ch and its subclasses during unpickling."""

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"x": state})

    def value(self):
        for key in ("x", "a", "r"):
            if key in self.__dict__:
                v = self.__dict__[key]
                return v.value() if isinstance(v, _ChStub) else np.asarray(v)
        raise ValueError(f"cannot recover array from chumpy object with keys {list(self.__dict__)}")


class _Unpickler(pickle._Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _ChStub
        return super().find_class(module, name)


def load_clean(path: str) -> dict:
    with open(path, "rb") as f:
        u = _Unpickler(f)
        u.encoding = "latin1"
        data = u.load()
    out = {}
    for k, v in data.items():
        if isinstance(v, _ChStub):
            v = v.value()
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args()
    d = load_clean(a.src)
    with open(a.dst, "wb") as f:
        pickle.dump(d, f, protocol=2)
    for k, v in d.items():
        print(f"{k:20s} {type(v).__name__:12s} {getattr(v, 'shape', '')}")


if __name__ == "__main__":
    main()
