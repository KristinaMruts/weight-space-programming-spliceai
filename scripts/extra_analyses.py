"""Additional analyses requested in review: positional baselines, held-out chr1, gene-level bootstrap,
operating points, star-filtered subset, per-chromosome gains and Bonferroni intervals.

  python scripts/extra_analyses.py

Reads results/ds_<name>.npy (test), results/dev/ds_<name>.npy (dev; original, temperature-only and Opus
only) and labels from data/. No model is run; takes about a minute.
"""
from __future__ import annotations

import collections
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["original_rs10", "ensemble_5_seeds", "control_temperature_only",
          "gradient_G1_final_conv", "gradient_G2_readout", "opus"]
DEV_MODELS = ["original_rs10", "control_temperature_only", "opus"]


def auroc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True); r = (np.bincount(inv, weights=r) / cnt)[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(y, s):
    o = np.argsort(-s, kind="mergesort"); y = y[o]; tp = np.cumsum(y)
    return float((tp / np.arange(1, len(y) + 1) * y).sum() / y.sum())


def read(path):
    import gzip
    with gzip.open(path, "rt") as f:
        cols = f.readline().rstrip("\n").split("\t")
        return [dict(zip(cols, l.rstrip("\n").split("\t"))) for l in f]


def boot(y, sa, sb, idx_sets=None, n=2000, q=(2.5, 97.5), seed=0):
    """Paired bootstrap of AUPRC(sb) - AUPRC(sa). idx_sets: list of index arrays to resample as clusters."""
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        if idx_sets is None:
            i = rng.integers(0, len(y), len(y))
        else:
            i = np.concatenate([idx_sets[k] for k in rng.integers(0, len(idx_sets), len(idx_sets))])
        if y[i].sum() in (0, len(i)):
            continue
        d.append(auprc(y[i], sb[i]) - auprc(y[i], sa[i]))
    return auprc(y, sb) - auprc(y, sa), np.percentile(d, q)


def group(site):
    k, o = site[0], int(site[1:])
    if k == "D" and o >= 7:
        return "D+7+"
    return "other"


def main():
    test = read(ROOT / "data/test/test_variants.tsv.gz"); dev = read(ROOT / "data/dev/dev_variants.tsv.gz")
    y = np.array([int(r["label"]) for r in test]); yd = np.array([int(r["label"]) for r in dev])
    chrom = np.array([r["chrom"] for r in test]); chd = np.array([r["chrom"] for r in dev])
    gene = np.array([r["gene"] for r in test]); gd = np.array([r["gene"] for r in dev])
    site = np.array([r["site"] for r in test]); sd = np.array([r["site"] for r in dev])
    stars = np.array([int(r["stars"]) for r in test])
    S = {m: np.load(ROOT / f"results/ds_{m}.npy").max(1) for m in MODELS}
    D = {m: np.load(ROOT / f"results/dev/ds_{m}.npy").max(1) for m in DEV_MODELS}

    print("== Test metrics by model")
    spec0 = (S["original_rs10"][y == 0] < 0.2).mean(); chr1 = chrom == "chr1"; st2 = stars >= 2
    print(f"{'model':28s} AUPRC  chr1   2+stars sens@spec{spec0:.3f} sens>=0.5 sens>=0.8 P_with_DS<=0.1")
    for m, s in S.items():
        thr = np.quantile(s[y == 0], spec0)
        print(f"{m:28s} {auprc(y, s):.4f} {auprc(y[chr1], s[chr1]):.4f} {auprc(y[st2], s[st2]):.4f}  "
              f"{(s[y == 1] > thr).mean():.3f}         {(s[y == 1] >= 0.5).mean():.3f}     "
              f"{(s[y == 1] >= 0.8).mean():.3f}     {int((s[y == 1] <= 0.1).sum())}")
    print(f"2+ star subset: n={st2.sum()}, pathogenic={y[st2].sum()}; 1-star share of test: {(stars == 1).mean():.3f}")

    print("\n== Bootstrap, Opus - original (AUPRC)")
    genes = [np.where(gene == g)[0] for g in np.unique(gene)]
    for lab, kw in [("variants", {}), ("genes (cluster)", {"idx_sets": genes})]:
        p, ci = boot(y, S["original_rs10"], S["opus"], **kw); print(f"  {lab:16s} {p:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    i1 = np.where(chr1)[0]
    for a, b in [("original_rs10", "opus"), ("control_temperature_only", "opus")]:
        p, ci = boot(y[i1], S[a][i1], S[b][i1], seed=1); print(f"  chr1 {b} - {a}: {p:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]")

    print("\n== Bonferroni-adjusted 99.64 % intervals (14 comparisons), variant bootstrap")
    for a, b in [("gradient_G1_final_conv", "opus"), ("gradient_G2_readout", "opus"),
                 ("ensemble_5_seeds", "opus"), ("control_temperature_only", "opus")]:
        p, ci = boot(y, S[a], S[b], q=(0.18, 99.82)); print(f"  {b} - {a}: {p:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]")

    print("\n== Positional baselines (rates estimated on dev only)")
    off = lambda ss: np.array([abs(int(x[1:])) for x in ss], float)
    print(f"  -|offset| only: test AUROC {auroc(y, -off(site)):.4f}")
    cnt = collections.defaultdict(lambda: [0, 0])
    for s_, l in zip(sd, yd):
        cnt[s_][0] += l; cnt[s_][1] += 1
    pr = yd.mean()
    prior = np.array([(cnt[s_][0] + 2 * pr) / (cnt[s_][1] + 2) for s_ in site])
    print(f"  per-position pathogenic rate: test AUROC {auroc(y, prior):.4f}, AUPRC {auprc(y, prior):.4f}")
    logit = lambda p: np.log(p / (1 - p))
    prior_loo = np.array([(cnt[s_][0] - l + 2 * pr) / (cnt[s_][1] - 1 + 2) for s_, l in zip(sd, yd)])

    def fit(X, t, it=50, l2=1e-3):
        X = np.c_[np.ones(len(X)), X]; w = np.zeros(X.shape[1])
        for _ in range(it):
            p = 1 / (1 + np.exp(-X @ w)); g = X.T @ (p - t) + l2 * w
            H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w)); w -= np.linalg.solve(H, g)
        return w
    lg = lambda d: np.log(np.clip(d, 1e-6, 1))
    for m in DEV_MODELS:
        w = fit(np.c_[lg(D[m]), logit(prior_loo)], yd)
        s = np.c_[np.ones(len(y)), lg(S[m]), logit(prior)] @ w
        print(f"  logistic regression [log DS, position prior] on {m:24s}: test AUPRC {auprc(y, s):.4f}, AUROC {auroc(y, s):.4f}")

    print("\n== Within-position AUROC (pair-weighted), test")
    for m in ["original_rs10", "control_temperature_only", "opus"]:
        num = den = 0
        for u in np.unique(site):
            k = site == u; n1 = y[k].sum(); n0 = k.sum() - n1
            if n1 and n0:
                num += auroc(y[k], S[m][k]) * n1 * n0; den += n1 * n0
        print(f"  {m:28s} {num / den:.4f}")

    print("\n== D+7 and deeper (new-site side), test AUROC")
    g = np.array([group(x) for x in site]) == "D+7+"
    print("  " + "  ".join(f"{m} {auroc(y[g], S[m][g]):.3f}" for m in ["original_rs10", "control_temperature_only", "opus"]))

    print("\n== Per-chromosome AUPRC gain, Opus - original")
    for nm, yy, cc, A, B in [("dev", yd, chd, D["original_rs10"], D["opus"]), ("test", y, chrom, S["original_rs10"], S["opus"])]:
        for c in np.unique(cc):
            k = cc == c
            print(f"  {nm:4s} {c:5s} n={k.sum():4d} P={yy[k].sum():3d} original {auprc(yy[k], A[k]):.3f} gain {auprc(yy[k], B[k]) - auprc(yy[k], A[k]):+.3f}")
    k = gd != "NF1"
    print(f"  dev without NF1: gain {auprc(yd[k], D['opus'][k]) - auprc(yd[k], D['original_rs10'][k]):+.3f}; "
          f"NF1 pathogenic {int(yd[~k].sum())} of {int(yd.sum())}")


if __name__ == "__main__":
    main()
