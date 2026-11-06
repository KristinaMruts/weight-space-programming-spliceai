"""Does the de-saturation found by the Opus edit carry over to the other OpenSpliceAI seeds and the ensemble,
and does OpenSpliceAI's own `calibrate` find the same temperature?

Step 1 caches final-layer logits once per seed (no weights are changed):
  python scripts/seeds_temperature.py logits --seed 10 --cache cache/seeds
    -> logits_dev_rs10.npy, logits_test_rs10.npy: (N, 2 [ref, alt], 3, 101) float32
    -> guard_dev_rs10.npy: (P, 3) float16 logits at every position of the dev guard genes, guard_dev_labels.npy
Step 2 is CPU-cheap and only reads the cache:
  python scripts/seeds_temperature.py analyze --cache cache/seeds --out results/seeds_temperature.json

Scores (all from the same logits z, DS = max over the 101 positions of the four SpliceAI deltas):
  T = 1            the stock delta score
  T = 1/0.35       the temperature inside the Opus edit (E2), applied to each seed
  T* (dev)         one scalar T per seed, chosen on dev AUPRC (grid), applied to test
  calibrate        class-wise T vector fitted like `openspliceai calibrate` (NLL on all positions of the dev
                   guard genes, Adam lr 0.01, clamp [0.05, 5], ReduceLROnPlateau, early stop), see
                   openspliceai/calibrate/temperature_scaling.py; note `openspliceai variant` does not apply it
  T -> inf limit   max over positions and the two site classes of |delta(z_c - mean_k z_k)| (pure logit score)
Ensemble = mean of the five softmax outputs (as in OpenSpliceAI / SpliceAI), each seed with its own T.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import evaluate as ev
import osai

SEEDS = [10, 11, 12, 13, 14]
ROOT = Path(__file__).resolve().parents[1]
T_GRID = np.round(np.concatenate([np.arange(0.5, 2.0, 0.1), np.arange(2.0, 6.01, 0.25), [8, 12, 20]]), 3)


# ---------------------------------------------------------------- step 1: logits
@torch.no_grad()
def var_logits(model, rows, batch=32):
    out = []
    for i in range(0, len(rows), batch):
        r = rows[i:i + batch]
        ref = [x["window"] for x in r]
        alt = [s[:osai.CENTER] + x["alt_tx"] + s[osai.CENTER + 1:] for s, x in zip(ref, r)]
        z = model(osai.onehot(ref + alt), logits=True).numpy()          # (2b, 3, 101)
        out.append(np.stack([z[:len(r)], z[len(r):]], 1).astype(np.float32))
    return np.concatenate(out)


@torch.no_grad()
def guard_logits(model, genes):
    zs, ys = [], []
    for g in genes:
        s, L = g["seq"], g["length"]
        blocks = [s[j:j + 15000] for j in range(0, L, 5000)]
        blocks = [b + "N" * (15000 - len(b)) for b in blocks]
        parts = [model(osai.onehot(blocks[i:i + 8]), logits=True).permute(0, 2, 1).reshape(-1, 3)
                 for i in range(0, len(blocks), 8)]
        zs.append(torch.cat(parts)[:L].half().numpy())
        y = np.zeros(L, np.int8); y[g["acceptors"]] = 1; y[g["donors"]] = 2; ys.append(y)
    return np.concatenate(zs), np.concatenate(ys)


def cmd_logits(a):
    torch.manual_seed(0)
    cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
    model = osai.load(ROOT / "weights" / f"model_10000nt_rs{a.seed}.pt")
    t = time.time()
    for split in ("dev", "test"):
        f = cache / f"logits_{split}_rs{a.seed}.npy"
        if not f.exists():
            rows = ev.read_variants(ROOT / "data" / split / f"{split}_variants.tsv.gz")
            np.save(f, var_logits(model, rows))
            print(f"rs{a.seed} {split} variants done {time.time() - t:.0f}s", flush=True)
    f = cache / f"guard_dev_rs{a.seed}.npy"
    if not f.exists():
        genes = [json.loads(l) for l in gzip.open(ROOT / "data/dev/dev_guard_genes.jsonl.gz", "rt")]
        z, y = guard_logits(model, genes)
        np.save(f, z)
        if not (cache / "guard_dev_labels.npy").exists():
            np.save(cache / "guard_dev_labels.npy", y)
        print(f"rs{a.seed} dev guard done {time.time() - t:.0f}s", flush=True)


# ---------------------------------------------------------------- step 2: analysis
def softmax_T(z, T):
    """z (..., 3, L); T scalar or (3,) class-wise."""
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    zz = z / (T[:, None] if T.size == 3 else T[0])
    zz = zz - zz.max(-2, keepdims=True)
    e = np.exp(zz)
    return e / e.sum(-2, keepdims=True)


def ds_from_probs(p):
    """p (N, 2, 3, L) -> DS_max (N,)"""
    d = p[:, 1] - p[:, 0]
    return np.stack([d[:, 1].max(1), (-d[:, 1]).max(1), d[:, 2].max(1), (-d[:, 2]).max(1)], 1).max(1)


def ds_logit_limit(z):
    c = z - z.mean(-2, keepdims=True)
    d = c[:, 1] - c[:, 0]
    return np.abs(d[:, 1:]).max((1, 2))


def calibrate_T(z, y, epochs=2000, patience=2):
    """Re-implementation of ModelWithTemperature.set_temperature (class-wise, NLL, Adam 0.01, clamp)."""
    zt = torch.from_numpy(z.astype(np.float32)); yt = torch.from_numpy(y.astype(np.int64))
    T = torch.nn.Parameter(torch.ones(3))
    opt = torch.optim.Adam([T], lr=0.01)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=2)
    best, best_T, bad = float("inf"), T.data.clone(), 0
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(zt / torch.clamp(T, 0.05, 5.0), yt)
        loss.backward(); opt.step()
        T.data = torch.clamp(T.data, 0.05, 5.0)
        cur = loss.item(); sch.step(cur)
        if best - cur > 1e-6:
            best, best_T, bad = cur, T.data.clone(), 0
        else:
            bad += 1
        if bad >= patience:
            break
    return best_T.numpy().astype(float), best


def metrics(y, s, spec_ref=None):
    out = {"AUPRC": ev.auprc(y, s), "AUROC": ev.auroc(y, s)}
    if spec_ref is not None:   # sensitivity at the specificity of the stock score (threshold from benign quantile)
        thr = np.quantile(s[y == 0], spec_ref)
        out["sens@spec_ref"] = float((s[y == 1] > thr).mean())
    return out


def paired_boot(y, s1, s2, n=2000, seed=0):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() in (0, len(i)):
            continue
        d.append(ev.auprc(y[i], s2[i]) - ev.auprc(y[i], s1[i]))
    d = np.array(d)
    return [float(np.mean(d)), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def cmd_analyze(a):
    cache = Path(a.cache)
    ydev = np.array([int(r["label"]) for r in ev.read_variants(ROOT / "data/dev/dev_variants.tsv.gz")])
    ytest = np.array([int(r["label"]) for r in ev.read_variants(ROOT / "data/test/test_variants.tsv.gz")])
    gy = np.load(cache / "guard_dev_labels.npy")
    Z = {s: {sp: np.load(cache / f"logits_{sp}_rs{s}.npy").astype(np.float64) for sp in ("dev", "test")} for s in SEEDS}
    res = {"T_grid": T_GRID.tolist(), "seeds": {}, "ensemble": {}}

    # sanity: T = 1 reproduces the published stock scores
    for name, s in (("original_rs10", 10),):
        ref = np.load(ROOT / "results" / f"ds_{name}.npy").max(1)
        mine = ds_from_probs(softmax_T(Z[s]["test"], 1.0))
        res[f"check_{name}_max_abs_diff"] = float(np.abs(ref - mine).max())

    T_choice = {}
    for s in SEEDS:
        r = {}
        dev_curve = [ev.auprc(ydev, ds_from_probs(softmax_T(Z[s]["dev"], T))) for T in T_GRID]
        T_star = float(T_GRID[int(np.argmax(dev_curve))])
        gz = np.load(cache / f"guard_dev_rs{s}.npy").astype(np.float32)
        T_cal, nll = calibrate_T(gz, gy)
        r["dev_AUPRC_curve"] = dev_curve
        r["T_star_dev"] = T_star
        r["T_calibrate"] = T_cal.tolist(); r["calibrate_NLL"] = nll
        base = ds_from_probs(softmax_T(Z[s]["test"], 1.0))
        spec_ref = float((base[ytest == 0] < 0.2).mean())
        r["test"] = {}
        for lab, T in (("T=1", 1.0), ("T=1/0.35", 1 / 0.35), ("T*_dev", T_star), ("calibrate", T_cal)):
            sc = ds_from_probs(softmax_T(Z[s]["test"], T))
            r["test"][lab] = metrics(ytest, sc, spec_ref) | {"guard_note": "scalar T keeps guard top-k" if np.size(T) == 1 else "class-wise"}
        r["test"]["logit_limit"] = metrics(ytest, ds_logit_limit(Z[s]["test"]), spec_ref)
        r["test"]["delta_T*_vs_T1"] = paired_boot(ytest, base, ds_from_probs(softmax_T(Z[s]["test"], T_star)))
        res["seeds"][f"rs{s}"] = r
        T_choice[s] = {"T=1": 1.0, "T=1/0.35": 1 / 0.35, "T*_dev": T_star, "calibrate": T_cal}
        print(f"rs{s}: T*={T_star} calibrate T={np.round(T_cal, 3)} test AUPRC "
              + " ".join(f"{k}={v['AUPRC']:.4f}" for k, v in r["test"].items() if isinstance(v, dict)), flush=True)

    # ensemble: mean of softmax, each seed with its own temperature; plus one common T chosen on dev
    def ens(split, how):
        return np.mean([softmax_T(Z[s][split], how(s)) for s in SEEDS], 0)
    ens_dev_curve = [ev.auprc(ydev, ds_from_probs(ens("dev", lambda s, T=T: T))) for T in T_GRID]
    T_ens = float(T_GRID[int(np.argmax(ens_dev_curve))])
    base = ds_from_probs(ens("test", lambda s: 1.0))
    spec_ref = float((base[ytest == 0] < 0.2).mean())
    variants = {"T=1": lambda s: 1.0, "T=1/0.35": lambda s: 1 / 0.35, "common T*_dev": lambda s: T_ens,
                "per-seed T*_dev": lambda s: T_choice[s]["T*_dev"], "per-seed calibrate": lambda s: T_choice[s]["calibrate"]}
    res["ensemble"]["dev_AUPRC_curve_common_T"] = ens_dev_curve
    res["ensemble"]["T_star_dev_common"] = T_ens
    res["ensemble"]["test"] = {}
    for lab, how in variants.items():
        sc = ds_from_probs(ens("test", how))
        res["ensemble"]["test"][lab] = metrics(ytest, sc, spec_ref)
        if lab != "T=1":
            res["ensemble"]["test"][lab]["delta_vs_T1_boot"] = paired_boot(ytest, base, sc)
    lim = np.mean([ds_logit_limit(Z[s]["test"]) for s in SEEDS], 0)
    res["ensemble"]["test"]["logit_limit (mean over seeds)"] = metrics(ytest, lim, spec_ref)
    opus = np.load(ROOT / "results/ds_opus.npy").max(1)
    res["ensemble"]["test"]["opus_single_model (reference)"] = metrics(ytest, opus, spec_ref)
    res["ensemble"]["test"]["opus_vs_ensemble_common_T_boot"] = paired_boot(ytest, ds_from_probs(ens("test", variants["common T*_dev"])), opus)
    for k, v in res["ensemble"]["test"].items():
        if isinstance(v, dict):
            print(f"ensemble {k}: AUPRC {v['AUPRC']:.4f} AUROC {v['AUROC']:.4f} sens@spec {v.get('sens@spec_ref', float('nan')):.3f}"
                  + (f" delta {np.round(v['delta_vs_T1_boot'], 4)}" if "delta_vs_T1_boot" in v else ""))
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("saved", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["logits", "analyze"])
    ap.add_argument("--seed", type=int); ap.add_argument("--cache", required=True); ap.add_argument("--out")
    a = ap.parse_args()
    {"logits": cmd_logits, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
