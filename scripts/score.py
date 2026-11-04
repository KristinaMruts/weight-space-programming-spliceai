"""Score a state_dict (or the 5-seed ensemble) on the test set: variants (AUPRC, AUROC, ...) and guard (top-k).

  python scripts/score.py --weights submissions/opus/model_edited.pt --name opus
  python scripts/score.py --ensemble weights --name ensemble_5_seeds

Appends one JSON line to results/scores.jsonl and saves per-variant delta scores to results/ds_<name>.npy
(shape (n, 4): acceptor gain, acceptor loss, donor gain, donor loss; DS_max = max over the 4).
CPU, about 20-30 min for one model on a laptop; the ensemble takes ~5x longer.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch

import evaluate as ev
import osai

ROOT = Path(__file__).resolve().parents[1]


class Ensemble(torch.nn.Module):
    """Mean of softmax outputs, as in OpenSpliceAI / SpliceAI."""
    def __init__(self, models):
        super().__init__(); self.ms = torch.nn.ModuleList(models)

    def forward(self, x, logits=False):
        return torch.stack([m(x) for m in self.ms]).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights"); ap.add_argument("--ensemble", help="folder with model_10000nt_rs10..14.pt")
    ap.add_argument("--name", required=True); ap.add_argument("--test-dir", default=str(ROOT / "data/test"))
    ap.add_argument("--out", default=str(ROOT / "results")); ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.ensemble:
        model = Ensemble([osai.load(Path(a.ensemble) / f"model_10000nt_rs{s}.pt") for s in range(10, 15)]).eval()
    else:
        model = osai.load(a.weights)
    t = Path(a.test_dir)
    rows = ev.read_variants(t / "test_variants.tsv.gz")
    genes = [json.loads(l) for l in gzip.open(t / "test_guard_genes.jsonl.gz", "rt")]
    t0 = time.time()
    ds = ev.score_variants(model, rows)
    rec = {"name": a.name, "test_variants": ev.variant_metrics(rows, ds), "test_guard": ev.guard_topk(model, genes),
           "seconds": round(time.time() - t0, 1), "scored_at": time.strftime("%Y-%m-%d %H:%M")}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"ds_{a.name}.npy", ds)
    with open(out / "scores.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
