"""Control "what one would normally do": gradient fine-tuning on the same dev labels the participants had.

Only the read-out layers are trained: the five 1x1 skip convolutions and final_conv. They do not feed back
into the residual stream, so the input of every skip convolution (the residual stream at 5 points) is
computed once and cached; after that the model is a linear read-out + softmax. This is the same parameter
subspace the Opus edit touched (skip after group 1 and final_conv), so the comparison is direct.

  G1  final_conv              99 parameters  (contains "temperature")
  G2  5 skip convs + final  5,379 parameters

Loss: pairwise logistic (AUC surrogate) on variant DS_max + pairwise loss on guard splice sites
(true site vs hard and random non-sites, on class log p, which does not penalise temperature) + optional
L2 to the original weights. Selection: train on chr17+chr19, validate on chr21+chr22, criterion =
validation AUPRC with the dev guard dropping at most 1 pp. Then refit on all dev with the chosen settings.

  python gradient_baseline.py feats  --data <dev data dir> --weights <rs10.pt> --cache <dir>
  python gradient_baseline.py train  --data <dev data dir> --weights <rs10.pt> --cache <dir> --out <dir>
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

SKIP_KEYS = ["initial_skip.conv"] + [f"residual_units.{i}.conv" for i in (4, 9, 14, 19)]
C = osai.CL // 2


@torch.no_grad()
def stream(model, x):
    """Inputs of the five skip convolutions, cropped to the output part of the window: (N, 5, 32, L - CL)."""
    x = model.initial_conv(x); out = [x]
    for m in model.residual_units:
        if isinstance(m, osai.Skip):
            out.append(x)
        else:
            x, _ = m(x, 0)
    return torch.stack([o[:, :, C:o.shape[2] - C] for o in out], 1)


def readout(p, Fe):
    """p: W (5,32,32), b (5,32), Wf (3,32), bf (3). Fe (N,5,32,L) -> logits (N,3,L)."""
    skip = torch.einsum("kij,nkjl->nil", p["W"], Fe) + p["b"].sum(0)[None, :, None]
    return torch.einsum("ci,nil->ncl", p["Wf"], skip) + p["bf"][None, :, None]


def params_from_sd(sd):
    return {"W": torch.stack([sd[k + ".weight"][:, :, 0] for k in SKIP_KEYS]).float(),
            "b": torch.stack([sd[k + ".bias"] for k in SKIP_KEYS]).float(),
            "Wf": sd["final_conv.weight"][:, :, 0].float().clone(), "bf": sd["final_conv.bias"].float().clone()}


def sd_from_params(sd0, p):
    sd = {k: v.clone() for k, v in sd0.items()}
    for i, k in enumerate(SKIP_KEYS):
        sd[k + ".weight"] = p["W"][i].detach().clone()[:, :, None].to(sd0[k + ".weight"].dtype)
        sd[k + ".bias"] = p["b"][i].detach().clone().to(sd0[k + ".bias"].dtype)
    sd["final_conv.weight"] = p["Wf"].detach().clone()[:, :, None].to(sd0["final_conv.weight"].dtype)
    sd["final_conv.bias"] = p["bf"].detach().clone().to(sd0["final_conv.bias"].dtype)
    return sd


def cmd_feats(a):
    torch.manual_seed(0)
    model = osai.load(a.weights); cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
    rows = ev.read_variants(Path(a.data) / "dev_variants.tsv.gz")
    t = time.time(); fv = []
    for i in range(0, len(rows), 32):
        r = [x["window"] for x in rows[i:i + 32]]
        alt = [s[:osai.CENTER] + x["alt_tx"] + s[osai.CENTER + 1:] for s, x in zip(r, rows[i:i + 32])]
        f = stream(model, osai.onehot(r + alt)).numpy()            # (2b,5,32,101)
        fv.append(np.stack([f[:len(r)], f[len(r):]], 1).astype(np.float32))
        if i % 320 == 0:
            print(f"variants {i}/{len(rows)} {time.time() - t:.0f}s", flush=True)
    fv = np.concatenate(fv)
    np.save(cache / "var_feats.npy", fv)
    np.save(cache / "var_meta.npy", np.array([[int(x["label"]), int(x["chrom"][3:])] for x in rows]))
    genes = [json.loads(l) for l in gzip.open(Path(a.data) / "dev_guard_genes.jsonl.gz", "rt")]
    gf, ta, td = [], [], []
    for g in genes:
        s, L = g["seq"], g["length"]
        blocks = [s[j:j + 15000] for j in range(0, L, 5000)]
        blocks = [b + "N" * (15000 - len(b)) for b in blocks]
        parts = []
        for i in range(0, len(blocks), 8):
            parts.append(stream(model, osai.onehot(blocks[i:i + 8])).permute(0, 3, 1, 2).reshape(-1, 5, 32))
        gf.append(torch.cat(parts)[:L].half().numpy())
        a_ = np.zeros(L, bool); a_[g["acceptors"]] = True; ta.append(a_)
        d_ = np.zeros(L, bool); d_[g["donors"]] = True; td.append(d_)
    np.save(cache / "guard_feats.npy", np.concatenate(gf))
    np.save(cache / "guard_labels.npy", np.stack([np.concatenate(ta), np.concatenate(td)]))
    print(f"done {time.time() - t:.0f}s", flush=True)


def ds_max(p, fv):
    lo = readout(p, fv.reshape(-1, 5, 32, fv.shape[-1]))
    pr = F.softmax(lo, 1).reshape(fv.shape[0], 2, 3, -1)
    d = pr[:, 1] - pr[:, 0]
    return torch.stack([d[:, 1].amax(1), (-d[:, 1]).amax(1), d[:, 2].amax(1), (-d[:, 2]).amax(1)], 1).amax(1)


@torch.no_grad()
def guard_logp(p, gf, chunk=200_000):
    out = []
    for i in range(0, len(gf), chunk):
        x = gf[i:i + chunk].float().permute(1, 2, 0)[None]           # (1,5,32,n)
        out.append(F.log_softmax(readout(p, x)[0], 0)[1:].T)
    return torch.cat(out)                                              # (P, 2): log p acceptor, donor


def topk_mean(lp, gl):
    r = []
    for c in range(2):
        k = int(gl[c].sum()); top = torch.topk(lp[:, c], k).indices
        r.append(gl[c][top].float().mean().item())
    return float(np.mean(r))


def fit(p0, fv, y, gf, gl, gsamp, mode, mu, steps, lr, eval_fn=None, every=10, lam=0.0):
    p = {k: v.clone().requires_grad_(mode == "G2" or k in ("Wf", "bf")) for k, v in p0.items()}
    opt = torch.optim.Adam([v for v in p.values() if v.requires_grad], lr=lr)
    pos, neg = torch.where(y == 1)[0], torch.where(y == 0)[0]
    g = torch.Generator().manual_seed(0); hist = []
    for s in range(1, steps + 1):
        opt.zero_grad()
        ds = ds_max(p, fv)
        loss = F.softplus(-10 * (ds[pos][:, None] - ds[neg][None, :])).mean()
        if mu > 0:
            idx_pos, idx_neg = gsamp
            lg = 0
            for c in range(2):
                nb = idx_neg[c][torch.randint(len(idx_neg[c]), (4096,), generator=g)]
                lp = F.log_softmax(readout(p, gf[torch.cat([idx_pos[c], nb])].float().permute(1, 2, 0)[None])[0], 0)[1 + c]
                lpp, lpn = lp[:len(idx_pos[c])], lp[len(idx_pos[c]):]
                lg = lg + F.softplus(-(lpp[:, None] - lpn[None, :])).mean()
            loss = loss + mu * lg
        if lam > 0:
            loss = loss + lam * sum(((p[k] - p0[k]) ** 2).sum() for k in p if p[k].requires_grad)
        loss.backward(); opt.step()
        if eval_fn and s % every == 0:
            hist.append({"step": s, "loss": round(loss.item(), 4), **eval_fn(p),
                         "_p": {k: v.detach().clone() for k, v in p.items()}})
    return {k: v.detach() for k, v in p.items()}, hist


def cmd_train(a):
    torch.manual_seed(0); torch.set_num_threads(a.threads or torch.get_num_threads())
    cache, out = Path(a.cache), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sd0 = torch.load(a.weights, map_location="cpu", weights_only=True); p0 = params_from_sd(sd0)
    fv = torch.from_numpy(np.load(cache / "var_feats.npy")); meta = np.load(cache / "var_meta.npy")
    y = torch.from_numpy(meta[:, 0]); chrom = meta[:, 1]
    gf = torch.from_numpy(np.load(cache / "guard_feats.npy")); gl = torch.from_numpy(np.load(cache / "guard_labels.npy"))

    # check: read-out from the cache == full model
    rows = ev.read_variants(Path(a.data) / "dev_variants.tsv.gz")[:16]
    ref = osai.delta_scores(osai.load(a.weights), [r["window"] for r in rows], [r["alt_tx"] for r in rows]).max(1)
    with torch.no_grad():
        mine = ds_max(p0, fv[:16]).numpy()
    print("readout check, max |diff| =", float(np.abs(ref - mine).max()), flush=True)

    with torch.no_grad():
        lp0 = guard_logp(p0, gf)
    g0 = topk_mean(lp0, gl)
    rng = np.random.default_rng(0); idx_pos, idx_neg = [], []
    for c in range(2):
        site = gl[c].numpy(); idx_pos.append(torch.from_numpy(np.where(site)[0]))
        sc = lp0[:, c].numpy().copy(); sc[site] = -np.inf
        hard = np.argsort(-sc)[:20000]; rnd = rng.choice(np.where(~site)[0], 20000, replace=False)
        idx_neg.append(torch.from_numpy(np.unique(np.concatenate([hard, rnd]))))
    gsamp = (idx_pos, idx_neg)

    tr = torch.from_numpy(np.isin(chrom, [17, 19])); va = ~tr

    def ev_split(mask):
        def f(p):
            with torch.no_grad():
                s = ds_max(p, fv[mask]).numpy(); yy = y[mask].numpy()
                return {"val_AUPRC": round(ev.auprc(yy, s), 4), "val_AUROC": round(ev.auroc(yy, s), 4),
                        "guard": round(topk_mean(guard_logp(p, gf), gl), 4)}
        return f

    base = ev_split(va)(p0)
    print("original on val:", base, "guard", round(g0, 4), flush=True)
    opus = {k: v.clone() for k, v in p0.items()}; opus["W"][1, [28, 29]] *= 2; opus["b"][1, [28, 29]] *= 2
    opus["Wf"] *= 0.35; opus["bf"] *= 0.35
    print("opus edit on val:", ev_split(va)(opus), flush=True)

    report = {"original_val": base, "original_guard": round(g0, 4), "opus_val": ev_split(va)(opus), "grid": [], "final": {}}
    clean = lambda h: {k: v for k, v in h.items() if k != "_p"} if h else None
    guard_only = lambda p: {"guard": round(topk_mean(guard_logp(p, gf), gl), 4)}
    for mode, lr, lams in (("G1", 1e-3, (0.0,)), ("G2", 2e-4, (0.0, 0.1, 1.0))):
        best = None
        for mu in (0.0, 0.3, 1.0, 3.0):
            for lam in lams:
                t = time.time()
                _, hist = fit(p0, fv[tr], y[tr], gf, gl, gsamp, mode, mu, a.steps, lr, ev_split(va), lam=lam)
                ok = [h for h in hist if h["guard"] >= g0 - 0.01]
                pick = max(ok, key=lambda h: h["val_AUPRC"]) if ok else None
                report["grid"].append({"mode": mode, "mu": mu, "lam": lam, "pick": clean(pick),
                                       "hist": [clean(h) for h in hist]})
                print(mode, "mu", mu, "lam", lam, "pick", clean(pick), f"{time.time() - t:.0f}s", flush=True)
                if pick and (best is None or pick["val_AUPRC"] > best[2]["val_AUPRC"]):
                    best = (mu, lam, pick)
        mu, lam, pick = best
        # refit on all dev; if the guard drops > 1 pp at the chosen step, take the last step where it does not
        _, hist = fit(p0, fv, y, gf, gl, gsamp, mode, mu, pick["step"], lr, guard_only, lam=lam)
        ok = [h for h in hist if h["guard"] >= g0 - 0.01]
        last = ok[-1]; p = last["_p"]
        with torch.no_grad():
            s = ds_max(p, fv).numpy()
            fin = {"mu": mu, "lam": lam, "lr": lr, "steps_selected": pick["step"], "steps_used": last["step"],
                   "val_pick": clean(pick), "dev_AUPRC_train_fit": round(ev.auprc(y.numpy(), s), 4),
                   "dev_guard": round(topk_mean(guard_logp(p, gf), gl), 4),
                   "changed_params": int(sum((p[k] != p0[k]).sum() for k in p))}
        report["final"][mode] = fin
        torch.save(sd_from_params(sd0, p), out / f"grad_{mode}.pt")
        print(mode, "final", fin, flush=True)
    (out / "gradient_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["feats", "train"])
    ap.add_argument("--data", required=True); ap.add_argument("--weights", required=True)
    ap.add_argument("--cache", required=True); ap.add_argument("--out")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    {"feats": cmd_feats, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
