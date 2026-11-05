"""Build the figures of the preprint from the files of this repository.

  python scripts/make_figures.py                                   # Figs. 1-3
  python scripts/make_figures.py --feats <cache>/var_feats.npy     # + Fig. 4 (dev temperature sweep)

Inputs (read only): results/ds_<model>.npy (one (N, 4) array per model: acceptor gain, acceptor loss,
donor gain, donor loss for the hidden-test variants, in the order of the test TSV),
results/dev/ds_original_rs10.npy (dev scores of the original model, for the position + DS baseline),
data/test/test_variants.tsv.gz and data/dev/dev_variants.tsv.gz. Fig. 4 needs the dev residual-stream
features from `python scripts/gradient_baseline.py feats --data data/dev --weights
weights/model_10000nt_rs10.pt --cache <cache>` and the original weights.

Outputs: figures/fig_saturation.pdf, fig_pr.pdf, fig_auprc.pdf, fig_temperature.pdf (with --feats).

Metrics are computed exactly as in evaluate.py (step-wise AUPRC = average precision, rank AUROC with
ties averaged). The script prints every number it draws so the figures can be checked against the
tables of the paper.
"""
from __future__ import annotations

import argparse
import collections
import gzip
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELS = [  # file stem, label, colour
    ("original_rs10", "Original (seed 10)", "#7f7f7f"),
    ("ensemble_5_seeds", "Ensemble, 5 seeds", "#9467bd"),
    ("control_temperature_only", "E2 only ($T\\approx2.86$)", "#2ca02c"),
    ("gradient_G1_final_conv", "G1: gradient, final_conv (99)", "#ff7f0e"),
    ("gradient_G2_readout", "G2: gradient, read-out (5,379)", "#d62728"),
    ("opus", "Opus edit (E1+E2, 165)", "#1f77b4"),
]

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "figure.dpi": 150})


def read_tsv(path, keep=("label", "site")):
    with gzip.open(path, "rt") as f:
        cols = f.readline().rstrip("\n").split("\t")
        idx = [cols.index(k) for k in keep]
        return [[line.rstrip("\n").split("\t")[i] for i in idx] for line in f]


def auprc(y, s):
    o = np.argsort(-s, kind="mergesort"); y = y[o]
    tp = np.cumsum(y); return float((tp / np.arange(1, len(y) + 1) * y).sum() / y.sum())


def auroc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.bincount(inv, weights=r) / cnt)[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pr_curve(y, s):
    o = np.argsort(-s, kind="mergesort"); y = y[o]
    tp = np.cumsum(y); k = np.arange(1, len(y) + 1)
    return np.r_[0, tp / y.sum()], np.r_[1, tp / k]


def softmax_ds(L, a):
    """DS_max at temperature scale a from cached logits L (2, N, 3, 101)."""
    z = L * a; z = z - z.max(2, keepdims=True); p = np.exp(z); p /= p.sum(2, keepdims=True)
    d = p[1] - p[0]
    return np.stack([d[:, 1].max(1), (-d[:, 1]).max(1), d[:, 2].max(1), (-d[:, 2]).max(1)], 1).max(1)


PNG = False  # set by --png: also write PNG previews next to the PDFs


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf")
    if PNG:
        fig.savefig(out / f"{name}.png", dpi=200)


def fig_saturation(out):
    sig = lambda z: 1 / (1 + np.exp(-z))
    z = np.linspace(-4, 9, 400)
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.3), sharey=True)
    for a_, (scale, title) in zip(ax, [(1.0, "Original read-out ($T=1$)"), (0.35, "Edit E2: logits $\\times$0.35 ($T\\approx2.86$)")]):
        a_.plot(z, sig(scale * z), color="k", lw=1)
        z0, z1 = 7.0, 4.0  # a strong site that loses 3 logits
        p0, p1 = sig(scale * z0), sig(scale * z1)
        a_.plot([z0, z1], [p0, p1], "o", color="#d62728", ms=4)
        a_.annotate("", xy=(z1, p1), xytext=(z0, p0), arrowprops=dict(arrowstyle="->", color="#d62728", lw=1))
        a_.axhline(p0, color="#d62728", lw=0.5, ls=":"); a_.axhline(p1, color="#d62728", lw=0.5, ls=":")
        # label placed clear of the curve and of the dotted p-lines in each panel
        tx, ty, ha = (-3.7, 0.80, "left") if scale == 1.0 else (9.6, 0.35, "right")
        a_.text(tx, ty, f"site logit {z0:.0f} $\\to$ {z1:.0f}\n$p$: {p0:.3f} $\\to$ {p1:.3f}\n$\\Delta p$ = {p0 - p1:.3f}",
                color="#d62728", fontsize=7, va="center", ha=ha)
        a_.axvline(5.2, color="#1f77b4", lw=0.7, ls="--")
        a_.set_title(title); a_.set_xlabel("site logit $z$ (site minus 'neither')")
        print(f"saturation scale {scale}: p {p0:.4f} -> {p1:.4f}, dp = {p0 - p1:.4f}")
    ax[0].set_ylabel("$p$(site) = $\\sigma(z/T)$")
    ax[0].text(5.35, 0.08, "median dev\nsite logit 5.2", color="#1f77b4", fontsize=6.5)
    fig.tight_layout(); save(fig, out, "fig_saturation"); plt.close(fig)


def fig_pr(out, y, S):
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.7))
    for stem, label, col in MODELS:
        r, p = pr_curve(y, S[stem])
        for a_ in ax:
            a_.step(r, p, where="post", color=col, lw=1.0 if stem != "opus" else 1.5,
                    label=f"{label}: {auprc(y, S[stem]):.4f}")
    ax[0].axhline(y.mean(), color="k", lw=0.5, ls=":")
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0.3, 1.01)
    ax[1].set_xlim(0.6, 1.0); ax[1].set_ylim(0.75, 1.005); ax[1].set_title("Zoom: recall 0.6–1.0")
    for a_ in ax:
        a_.set_xlabel("Recall"); a_.set_ylabel("Precision")
    ax[0].set_title(f"Hidden test, chr1/11/16 (n = {len(y)}, {int(y.sum())} P)")
    ax[0].legend(loc="lower left", frameon=False, title="AUPRC", title_fontsize=7)
    fig.tight_layout(); save(fig, out, "fig_pr"); plt.close(fig)


def fig_auprc(out, y, S, baselines, B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(y)
    idx = [rng.integers(0, n, n) for _ in range(B)]
    rows = [(label, S[stem], col) for stem, label, col in MODELS]
    vals, lo, hi = [], [], []
    for label, s, _ in rows:
        v = auprc(y, s); bs = np.array([auprc(y[i], s[i]) for i in idx])
        vals.append(v); lo.append(v - np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5) - v)
        print(f"AUPRC {label:34s} {v:.4f}  95% CI [{np.percentile(bs, 2.5):.4f}, {np.percentile(bs, 97.5):.4f}]  AUROC {auroc(y, s):.4f}")
    fig, ax = plt.subplots(figsize=(6.3, 2.4))
    xs = np.arange(len(rows))
    ax.bar(xs, vals, color=[c for *_, c in rows], width=0.65, yerr=[lo, hi], capsize=2,
           error_kw=dict(lw=0.7))
    for x, v in zip(xs, vals):
        ax.text(x, 0.912, f"{v:.4f}", ha="center", va="bottom", fontsize=7, color="white", fontweight="bold")
    styles = {"position only (dev rates)": ":", "original DS + position (LR on dev)": "--"}
    for name, v in baselines.items():
        ax.axhline(v, color="k", lw=0.8, ls=styles.get(name, "-."))
        ax.text(-0.45, v + 0.0012, f"{name}: {v:.4f}", ha="left", va="bottom", fontsize=6.5,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85))
    ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in rows], rotation=12, ha="right")
    ax.set_ylim(0.895, 0.985); ax.set_ylabel("AUPRC, hidden test")
    fig.tight_layout(); save(fig, out, "fig_auprc"); plt.close(fig)


def position_baselines(test, dev, S_orig_test, ds_dev):
    yd = np.array([int(r[0]) for r in dev]); sd = [r[1] for r in dev]
    site = [r[1] for r in test]; y = np.array([int(r[0]) for r in test])
    cnt = collections.defaultdict(lambda: [0, 0])
    for s, l in zip(sd, yd):
        cnt[s][0] += l; cnt[s][1] += 1
    pr = yd.mean()
    rate = lambda s, a=0, m=0: (cnt[s][0] - a + 2 * pr) / (cnt[s][1] - m + 2)  # smoothed P share per site
    pos_test = np.array([rate(s) for s in site])
    out = {"position only (dev rates)": auprc(y, pos_test)}
    print(f"position-only baseline (dev P-share per site -> test): AUPRC {out['position only (dev rates)']:.4f}  AUROC {auroc(y, pos_test):.4f}")
    logit = lambda p: np.log(p / (1 - p)); lg = lambda d: np.log(np.clip(d, 1e-6, 1))
    Xd = np.c_[np.ones(len(yd)), lg(ds_dev), logit(np.array([rate(s, l, 1) for s, l in zip(sd, yd)]))]  # leave-one-out prior
    w = np.zeros(3)
    for _ in range(50):  # Newton steps, tiny ridge
        p = 1 / (1 + np.exp(-Xd @ w)); g = Xd.T @ (p - yd) + 1e-3 * w
        H = (Xd * (p * (1 - p))[:, None]).T @ Xd + 1e-3 * np.eye(3); w -= np.linalg.solve(H, g)
    Xt = np.c_[np.ones(len(y)), lg(S_orig_test), logit(pos_test)]
    out["original DS + position (LR on dev)"] = auprc(y, Xt @ w)
    print(f"original DS + position LR (fit on dev, applied to test): AUPRC {out['original DS + position (LR on dev)']:.4f}  AUROC {auroc(y, Xt @ w):.4f}")
    return out


def fig_temperature(out, dev, feats, weights):
    from paper_analyses import dev_logits
    yd = np.array([int(r[0]) for r in dev])
    L = dev_logits(feats, weights)
    alphas = [1, 0.7, 0.5, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07, 0.05]
    ap = [auprc(yd, softmax_ds(L, a)) for a in alphas]
    for a, v in zip(alphas, ap):
        print(f"dev temperature sweep: scale {a:<5} T {1 / a:5.2f}  AUPRC {v:.4f}")
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    ax.plot(alphas, ap, "o-", color="#2ca02c", ms=3, lw=1)
    ax.axvline(0.35, color="#1f77b4", lw=0.7, ls="--"); ax.text(0.36, min(ap) + 0.001, "E2 = 0.35", color="#1f77b4", fontsize=7)
    ax.set_xscale("log"); ax.set_xticks([0.05, 0.1, 0.2, 0.35, 0.5, 1]); ax.set_xticklabels(["0.05", "0.1", "0.2", "0.35", "0.5", "1"])
    ax.set_xlabel("logit scale $1/T$ (final_conv multiplier)"); ax.set_ylabel("AUPRC, dev")
    fig.tight_layout(); save(fig, out, "fig_temperature"); plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=root / "results")
    ap.add_argument("--test", type=Path, default=root / "data/test/test_variants.tsv.gz")
    ap.add_argument("--dev", type=Path, default=root / "data/dev/dev_variants.tsv.gz")
    ap.add_argument("--feats", type=Path, help="var_feats.npy from gradient_baseline.py feats (Fig. 4)")
    ap.add_argument("--weights", type=Path, default=root / "weights/model_10000nt_rs10.pt")
    ap.add_argument("--out", type=Path, default=root / "figures")
    ap.add_argument("--png", action="store_true", help="also write PNG previews")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    global PNG
    PNG = a.png
    test, dev = read_tsv(a.test), read_tsv(a.dev)
    y = np.array([int(r[0]) for r in test])
    S = {stem: np.load(a.results / f"ds_{stem}.npy").max(1) for stem, *_ in MODELS}
    assert all(len(s) == len(y) for s in S.values())
    print(f"test n = {len(y)}, pathogenic = {int(y.sum())}; dev n = {len(dev)}")
    fig_saturation(a.out)
    fig_pr(a.out, y, S)
    ds_dev = np.load(a.results / "dev/ds_original_rs10.npy").max(1)
    fig_auprc(a.out, y, S, position_baselines(test, dev, S["original_rs10"], ds_dev))
    if a.feats is not None:
        fig_temperature(a.out, dev, a.feats, a.weights)


if __name__ == "__main__":
    main()
