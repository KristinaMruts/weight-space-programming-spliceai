"""Paired bootstrap of the AUPRC / AUROC difference between two scored models on the same test variants.

  python scripts/bootstrap.py original_rs10 opus
  python scripts/bootstrap.py control_temperature_only opus

Reads results/ds_<name>.npy (from score.py) and labels from data/test/test_variants.tsv.gz.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import evaluate as ev

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--test-dir", default=str(ROOT / "data/test")); ap.add_argument("--results", default=str(ROOT / "results"))
    a = ap.parse_args()
    y = np.array([int(r["label"]) for r in ev.read_variants(Path(a.test_dir) / "test_variants.tsv.gz")])
    sa = np.load(Path(a.results) / f"ds_{a.a}.npy").max(1); sb = np.load(Path(a.results) / f"ds_{a.b}.npy").max(1)
    rng = np.random.default_rng(0)
    d = {"AUPRC": [], "AUROC": []}
    for _ in range(a.n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() in (0, len(i)):
            continue
        d["AUPRC"].append(ev.auprc(y[i], sb[i]) - ev.auprc(y[i], sa[i]))
        d["AUROC"].append(ev.auroc(y[i], sb[i]) - ev.auroc(y[i], sa[i]))
    for k, v in d.items():
        v = np.array(v)
        point = (ev.auprc if k == "AUPRC" else ev.auroc)(y, sb) - (ev.auprc if k == "AUPRC" else ev.auroc)(y, sa)
        lo, hi = np.percentile(v, [2.5, 97.5])
        print(f"{a.b} - {a.a}  d{k} = {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  share <= 0: {(v <= 0).mean():.4f}")


if __name__ == "__main__":
    main()
