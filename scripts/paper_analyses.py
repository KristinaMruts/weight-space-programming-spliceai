"""Remaining numbers of the preprint that extra_analyses.py does not print.

  python scripts/paper_analyses.py                           # part A: test-set tables (about 2 min)
  python scripts/paper_analyses.py --feats <cache>/var_feats.npy   # + part B: dev temperature sweep, G1 margins

Part A reads results/ds_<name>.npy, results/dev/ds_original_rs10.npy and labels from data/. It prints
Table 2 of the preprint (paired bootstrap differences in test AUPRC with 95 % and Bonferroni intervals),
its sub-rows (gene-level, chr1, exon side), the AUROC gain, the exon-side Opus - G1/G2 intervals, the
share of benign variants at DS <= 0.1 and the distance-only baseline on dev.

Part B needs the dev residual-stream features written by

  python scripts/gradient_baseline.py feats --data data/dev --weights weights/model_10000nt_rs10.pt --cache <cache>

and the original and G1 weights. It prints the dev AUPRC as a function of the logit scale applied to
final_conv (Fig. 4 of the preprint) and how G1 changes the site-minus-neither margins. Interval numbers
depend on the order of random draws, which this script keeps fixed.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np

from extra_analyses import auprc, auroc, read

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"orig": "original_rs10", "ens": "ensemble_5_seeds", "E2": "control_temperature_only",
         "G1": "gradient_G1_final_conv", "G2": "gradient_G2_readout", "opus": "opus"}
PAIRS = [("opus", "orig"), ("E2", "orig"), ("opus", "E2"), ("ens", "orig"), ("opus", "ens"), ("E2", "ens"),
         ("G1", "ens"), ("G2", "ens"), ("opus", "G1"), ("opus", "G2"), ("G1", "E2"), ("G2", "E2"),
         ("opus", "LR"), ("G2", "LR")]
B = 2000
SKIPS = ["initial_skip.conv"] + [f"residual_units.{i}.conv" for i in (4, 9, 14, 19)]


def exon_side(sites):
    return np.array([(s[0] == "D" and int(s[1:]) < 0) or (s[0] == "A" and int(s[1:]) > 0) for s in sites])


def position_lr(y_dev, site_dev, ds_dev, site_test, ds_test):
    """Logistic regression on [log DS, dev log-odds of the position]; leave-one-out prior on dev."""
    cnt = collections.defaultdict(lambda: [0, 0])
    for s, l in zip(site_dev, y_dev):
        cnt[s][0] += l; cnt[s][1] += 1
    pr = y_dev.mean()

    def prior(S, Y=None):
        out = []
        for j, s in enumerate(S):
            a, n = cnt[s]
            if Y is not None:
                a -= Y[j]; n -= 1
            p = (a + 2 * pr) / (n + 2); out.append(np.log(p / (1 - p)))
        return np.array(out)

    lg = lambda d: np.log(np.clip(d, 1e-6, 1))
    X = np.c_[np.ones(len(y_dev)), lg(ds_dev), prior(site_dev, y_dev)]; w = np.zeros(3)
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w)); g = X.T @ (p - y_dev) + 1e-3 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-3 * np.eye(3); w -= np.linalg.solve(H, g)
    return np.c_[np.ones(len(site_test)), lg(ds_test), prior(site_test)] @ w


def part_a():
    test = read(ROOT / "data/test/test_variants.tsv.gz"); dev = read(ROOT / "data/dev/dev_variants.tsv.gz")
    y = np.array([int(r["label"]) for r in test]); site = np.array([r["site"] for r in test])
    gene = np.array([r["gene"] for r in test]); chrom = np.array([r["chrom"] for r in test])
    yd = np.array([int(r["label"]) for r in dev]); sd = np.array([r["site"] for r in dev])
    S = {k: np.load(ROOT / f"results/ds_{v}.npy").max(1) for k, v in NAMES.items()}
    S["LR"] = position_lr(yd, sd, np.load(ROOT / "results/dev/ds_original_rs10.npy").max(1), site, S["orig"])
    n = len(y); ex = exon_side(site)
    print(f"LR [log DS, position prior] on original: test AUPRC {auprc(y, S['LR']):.4f}")

    print("\n== Table 2: paired variant bootstrap of test AUPRC differences (2,000 resamples)")
    r = np.random.default_rng(0); I = [r.integers(0, n, n) for _ in range(B)]
    for a, b in PAIRS:
        d = np.array([auprc(y[i], S[a][i]) - auprc(y[i], S[b][i]) for i in I])
        print(f"  {a:>4s} - {b:<4s} {auprc(y, S[a]) - auprc(y, S[b]):+.4f}  95% [{np.percentile(d, 2.5):+.4f}, "
              f"{np.percentile(d, 97.5):+.4f}]  Bonferroni 99.64% [{np.percentile(d, 0.18):+.4f}, "
              f"{np.percentile(d, 99.82):+.4f}]  share <= 0: {np.mean(d <= 0):.4f}")

    print("\n== Opus - original: AUPRC and AUROC, variant and gene-level bootstrap")
    members = [np.where(gene == g)[0] for g in np.unique(gene)]
    for metric, fn in (("AUPRC", auprc), ("AUROC", auroc)):
        for lab, cl in (("variants", False), ("genes", True)):
            rng = np.random.default_rng(0); d = []
            for _ in range(B):
                i = (np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))]) if cl
                     else rng.integers(0, n, n))
                d.append(fn(y[i], S["opus"][i]) - fn(y[i], S["orig"][i]))
            d = np.array(d)
            print(f"  {metric} {lab:8s} {fn(y, S['opus']) - fn(y, S['orig']):+.4f}  95% "
                  f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]  share <= 0: {np.mean(d <= 0):.4f}")
    for metric, fn in (("AUPRC", auprc), ("AUROC", auroc)):
        share = (fn(y, S["E2"]) - fn(y, S["orig"])) / (fn(y, S["opus"]) - fn(y, S["orig"]))
        print(f"  share of the Opus {metric} gain reached by E2 alone: {share:.3f}")

    print("\n== Subsets: Opus - original and Opus - E2 (variant bootstrap within the subset)")
    rng = np.random.default_rng(1)
    for lab, m in (("exon side", ex), ("intron side", ~ex), ("chr1", chrom == "chr1"), ("chr11+16", chrom != "chr1")):
        idx0 = np.where(m)[0]; v1, v2 = [], []
        for _ in range(B):
            i = rng.choice(idx0, len(idx0))
            if y[i].sum() == 0:
                continue
            v1.append(auprc(y[i], S["opus"][i]) - auprc(y[i], S["orig"][i]))
            v2.append(auprc(y[i], S["opus"][i]) - auprc(y[i], S["E2"][i]))
        yy = y[m]
        print(f"  {lab:11s} n={m.sum()} P={yy.sum()}  opus-orig {auprc(yy, S['opus'][m]) - auprc(yy, S['orig'][m]):+.4f} "
              f"[{np.percentile(v1, 2.5):+.4f}, {np.percentile(v1, 97.5):+.4f}]  opus-E2 "
              f"{auprc(yy, S['opus'][m]) - auprc(yy, S['E2'][m]):+.4f} [{np.percentile(v2, 2.5):+.4f}, {np.percentile(v2, 97.5):+.4f}]")

    print("\n== Exon side: Opus - G1 and Opus - G2 (uncorrected 95 % intervals)")
    for b in ("G1", "G2"):
        yy, A, Bb = y[ex], S["opus"][ex], S[b][ex]; rr = np.random.default_rng(1); d = []
        for _ in range(B):
            i = rr.integers(0, len(yy), len(yy)); d.append(auprc(yy[i], A[i]) - auprc(yy[i], Bb[i]))
        print(f"  opus - {b}: {auprc(yy, A) - auprc(yy, Bb):+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")

    print("\n== Benign variants at DS <= 0.1 (ClinGen BP4 cut-off), test")
    for k in ("orig", "E2", "opus"):
        print(f"  {k:4s} {np.mean(S[k][y == 0] <= 0.1):.4f}")

    off = lambda ss: np.array([abs(int(s[1:])) for s in ss], dtype=float)
    print(f"\n== Distance to the boundary only: AUROC dev {auroc(yd, -off(sd)):.4f}, test {auroc(y, -off(site)):.4f}")


def readout_logits(sd, skip):
    """final_conv applied to the summed skip stream: (N, 32, L) -> (N, 3, L)."""
    return (np.einsum("ci,nil->ncl", sd["final_conv.weight"][:, :, 0].double().numpy(), skip)
            + sd["final_conv.bias"].double().numpy()[None, :, None])


def dev_skip(feats_path, sd):
    """Summed skip stream for ref and alt dev windows from var_feats.npy (N, 2, 5, 32, 101)."""
    F = np.load(feats_path, mmap_mode="r")
    W = np.stack([sd[k + ".weight"][:, :, 0].double().numpy() for k in SKIPS])
    b = np.stack([sd[k + ".bias"].double().numpy() for k in SKIPS]).sum(0)
    return [np.einsum("kij,nkjl->nil", W, np.asarray(F[:, r]).astype(np.float64)) + b[None, :, None] for r in (0, 1)]


def ds_from_logits(L, a=1.0):
    """DS_max at logit scale a from ref/alt logits L (2, N, 3, 101)."""
    z = L * a; z = z - z.max(2, keepdims=True); p = np.exp(z); p /= p.sum(2, keepdims=True)
    d = p[1] - p[0]
    return np.stack([d[:, 1].max(1), (-d[:, 1]).max(1), d[:, 2].max(1), (-d[:, 2]).max(1)], 1).max(1)


def dev_logits(feats_path, weights):
    import torch
    o = torch.load(weights, map_location="cpu", weights_only=True)
    return np.stack([readout_logits(o, s) for s in dev_skip(feats_path, o)])


def part_b(feats_path, weights, g1_weights):
    import torch

    dev = read(ROOT / "data/dev/dev_variants.tsv.gz"); yd = np.array([int(r["label"]) for r in dev])
    o = torch.load(weights, map_location="cpu", weights_only=True)
    g = torch.load(g1_weights, map_location="cpu", weights_only=True)
    skip = dev_skip(feats_path, o)
    L = np.stack([readout_logits(o, s) for s in skip])          # (2, N, 3, 101)
    ds_at = lambda a: ds_from_logits(L, a)

    ref = np.load(ROOT / "results/dev/ds_original_rs10.npy").max(1)
    print(f"\n== Check: DS from cached features vs results/dev, max abs diff {np.abs(ds_at(1.0) - ref).max():.2e}")
    print("\n== Fig. 4: dev AUPRC of the original model vs logit scale on final_conv (= 1/T)")
    for a in [1, 0.7, 0.5, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07, 0.05]:
        s = ds_at(a); print(f"  scale {a:<5} AUPRC {auprc(yd, s):.4f}  AUROC {auroc(yd, s):.4f}")

    print("\n== G1: read-out weights (site-minus-neither rows) against the original")
    W0, b0 = o["final_conv.weight"][:, :, 0].double(), o["final_conv.bias"].double()
    W1, b1 = g["final_conv.weight"][:, :, 0].double(), g["final_conv.bias"].double()
    for c, nm in ((1, "acceptor"), (2, "donor")):
        d0 = torch.cat([W0[c] - W0[0], (b0[c] - b0[0])[None]]); d1 = torch.cat([W1[c] - W1[0], (b1[c] - b1[0])[None]])
        print(f"  {nm:8s} cosine {float(d0 @ d1 / d0.norm() / d1.norm()):.3f}  channels 28/29: original "
              f"{[round(float(x), 3) for x in d0[[28, 29]]]}, G1 {[round(float(x), 3) for x in d1[[28, 29]]]}")
    L1 = readout_logits(g, skip[0])
    print("\n== G1: site-minus-neither margins on dev reference windows, G1 regressed on original")
    for c, nm in ((1, "acceptor"), (2, "donor")):
        m0 = (L[0][:, c] - L[0][:, 0]).ravel(); m1 = (L1[:, c] - L1[:, 0]).ravel()
        for lab, sel in (("all positions", np.ones_like(m0, bool)), ("margin > 0", m0 > 0), ("margin > 3", m0 > 3)):
            coef = np.linalg.lstsq(np.c_[m0[sel], np.ones(sel.sum())], m1[sel], rcond=None)[0]
            print(f"  {nm:8s} {lab:13s} n={sel.sum():6d} slope {coef[0]:.3f}  r {np.corrcoef(m0[sel], m1[sel])[0, 1]:.3f}  "
                  f"median margin {np.median(m0[sel]):.2f} -> {np.median(m1[sel]):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", type=Path, help="var_feats.npy from gradient_baseline.py feats (enables part B)")
    ap.add_argument("--weights", type=Path, default=ROOT / "weights/model_10000nt_rs10.pt")
    ap.add_argument("--g1", type=Path, default=ROOT / "controls/grad_G1.pt")
    ap.add_argument("--skip-a", action="store_true", help="only part B")
    a = ap.parse_args()
    if not a.skip_a:
        part_a()
    if a.feats:
        part_b(a.feats, a.weights, a.g1)


if __name__ == "__main__":
    main()
