"""Evaluate an OpenSpliceAI state_dict on variant data and on the splice-site guard.

  python evaluate.py --weights model.pt --variants data/dev_fast_variants.tsv.gz
  python evaluate.py --weights model.pt --variants data/dev_variants.tsv.gz --guard data/dev_guard_genes.jsonl.gz

Variant metric: DS_max = max(acceptor gain, acceptor loss, donor gain, donor loss), raw delta score,
distance 50 — the standard SpliceAI score. Pathogenic/likely pathogenic = 1, benign/likely benign = 0.
Reported: AUPRC (primary), AUROC, and both per region (exon side vs intron side of the boundary).
Guard: top-k accuracy for acceptors and donors on whole genes, as in the SpliceAI paper.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time

import numpy as np
import torch

import osai


def auroc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r); r = (sums / cnt)[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(y, s):
    o = np.argsort(-s, kind="mergesort"); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def read_variants(path):
    with gzip.open(path, "rt") as f:
        cols = f.readline().rstrip("\n").split("\t")
        return [dict(zip(cols, l.rstrip("\n").split("\t"))) for l in f]


def score_variants(model, rows, batch=64):
    ds = osai.delta_scores(model, [r["window"] for r in rows], [r["alt_tx"] for r in rows], batch=batch)
    return ds


def variant_metrics(rows, ds):
    y = np.array([int(r["label"]) for r in rows]); s = ds.max(1)
    out = {"n": len(y), "n_pathogenic": int(y.sum()), "AUPRC": round(auprc(y, s), 4), "AUROC": round(auroc(y, s), 4)}
    side = np.array([("exon" if (r["site"][0] == "D" and int(r["site"][1:]) < 0) or (r["site"][0] == "A" and int(r["site"][1:]) > 0)
                      else "intron") for r in rows])
    for g in ("exon", "intron"):
        m = side == g
        if y[m].sum() > 0 and (1 - y[m]).sum() > 0:
            out[f"{g}_AUPRC"] = round(auprc(y[m], s[m]), 4); out[f"{g}_AUROC"] = round(auroc(y[m], s[m]), 4)
            out[f"{g}_n"] = int(m.sum())
    # sensitivity and specificity at the clinical 0.2 threshold
    out["sens@0.2"] = round(float((s[y == 1] >= 0.2).mean()), 4)
    out["spec@0.2"] = round(float((s[y == 0] < 0.2).mean()), 4)
    return out


@torch.no_grad()
def guard_topk(model, genes, batch=8):
    blocks, owners = [], []
    for gi, g in enumerate(genes):
        s, L = g["seq"], g["length"]
        for j in range(0, L, 5000):
            b = s[j: j + 15000]
            blocks.append(b + "N" * (15000 - len(b))); owners.append((gi, j))
    preds = [np.zeros((3, g["length"]), dtype=np.float32) for g in genes]
    for i in range(0, len(blocks), batch):
        y = osai.predict(model, blocks[i:i + batch], batch=batch)
        for (gi, j), yy in zip(owners[i:i + batch], y):
            n = min(5000, genes[gi]["length"] - j)
            preds[gi][:, j:j + n] = yy[:, :n]
    res = {}
    for ch, key in ((1, "acceptors"), (2, "donors")):
        p = np.concatenate([pr[ch] for pr in preds])
        t = np.zeros(len(p), dtype=bool); off = 0
        for g in genes:
            t[off + np.array(g[key], dtype=int)] = True; off += g["length"]
        k = int(t.sum()); top = np.argsort(-p)[:k]
        res[f"top_k_{key}"] = round(float(t[top].mean()), 4)
        res[f"auprc_{key}"] = round(auprc(t.astype(int), p), 4)
    res["top_k_mean"] = round((res["top_k_acceptors"] + res["top_k_donors"]) / 2, 4)
    res["genes"] = len(genes)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--variants"); ap.add_argument("--guard"); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    model = osai.load(a.weights)
    out = {"weights": a.weights}
    t0 = time.time()
    if a.variants:
        rows = read_variants(a.variants)
        out["variants"] = variant_metrics(rows, score_variants(model, rows, a.batch))
    if a.guard:
        genes = [json.loads(l) for l in gzip.open(a.guard, "rt")]
        out["guard"] = guard_topk(model, genes)
    out["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
