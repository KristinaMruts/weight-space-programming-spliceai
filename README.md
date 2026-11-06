# Weight-space programming on a clinical splicing model

Can a language model improve a **useful, published** neural network by reading its weights and editing
them by hand — no gradient descent, no automated search — and does the gain survive on data it never saw?

This repository is a small, fully reproducible test of that idea (inspired by the
[MIT CSAIL weight-space programming post](https://rikiyat.github.io/blog/weight-space-programming/)) on
**OpenSpliceAI-MANE-10000nt**, an open re-training of SpliceAI, a model family whose delta scores inform
clinical variant interpretation. The task: telling pathogenic from benign ClinVar variants near exon
boundaries.

**Short answer.** The LLM edit gave a real, transferable gain (AUPRC 0.939 → 0.975 on a hidden test), but
≈ 86 % of it is a softmax temperature, and gradient fine-tuning of the same layers on the same labels gets
practically the same result. The edit matched ordinary training rather than beating it; what it added was a
diagnosis of *why* the score was losing signal. One model, one run, one editor. The temperature effect itself
is not specific to that model: it holds on all five OpenSpliceAI seeds, a tempered 5-seed ensemble matches
the edit, and reading the stock ensemble's output as floored log-odds beats it (0.979), all without touching
any weight ([below](#temperature-on-all-five-seeds-floored-log-odds-and-calibrate)).

## Result in one table

Hidden test: ClinVar on chr1, chr11, chr16 — 1,842 variants, 614 pathogenic. Guard: top-k splice-site
accuracy on 150 whole genes from chr1, a chromosome OpenSpliceAI was not trained on.

| Model (test set) | changed params | AUPRC | AUROC | exon-side AUPRC | sens @0.2 | spec @0.2 | guard top-k |
|---|---|---|---|---|---|---|---|
| Distance to boundary only (no model) | — | 0.743 | 0.834 | — | — | — | — |
| Per-position pathogenic rate, estimated on dev (no model) | — | 0.903 | 0.929 | — | — | — | — |
| Original DS + per-position rate, logistic regression fit on dev | — | 0.9714 | 0.9793 | — | — | — | — |
| Original, seed 10 | 0 | 0.9385 | 0.9556 | 0.8215 | 0.761 | 0.976 | 0.9224 |
| Original, 5-seed ensemble | — | 0.9532 | 0.9669 | 0.8487 | 0.811 | 0.985 | **0.9366** |
| Temperature only (control, `final_conv` × 0.35) | 99 | 0.9695 | 0.9780 | 0.9024 | 0.787 | 0.994 | 0.9220 |
| **Opus 5.5 edit** (hand edit, 2 changes) | 165 | **0.9746** | **0.9815** | **0.9142** | **0.816** | 0.993 | **0.9264** |
| Gradient, `final_conv` only (G1) | 99 | 0.9697 | 0.9788 | 0.8865 | 0.873 | 0.978 | 0.9165 |
| Gradient, all read-out layers (G2) | 5,379 | 0.9710 | 0.9797 | 0.8942 | 0.876 | 0.980 | 0.9197 |
| Second LLM participant | — | pending | | | | | |

Sensitivity and specificity at a fixed 0.2 cut-off are shown for completeness only: the edited and
gradient models have a different probability scale, so a fixed threshold compares different operating
points. Fairer views (`scripts/extra_analyses.py`):

| Model (test set) | AUPRC, chr1 only | AUPRC, ≥ 2-star records | sensitivity at the original's specificity (0.976) | sensitivity at DS ≥ 0.5 | sensitivity at DS ≥ 0.8 | pathogenic with DS ≤ 0.1 |
|---|---|---|---|---|---|---|
| Original, seed 10 | 0.9185 | 0.9698 | 0.761 | 0.642 | 0.490 | 111 |
| 5-seed ensemble | 0.9333 | 0.9781 | 0.850 | 0.653 | 0.467 | 86 |
| Temperature only | 0.9563 | 0.9877 | 0.873 | 0.484 | 0.083 | 65 |
| Gradient G1 | 0.9531 | 0.9875 | 0.875 | 0.634 | 0.252 | 40 |
| Gradient G2 | 0.9542 | 0.9886 | 0.894 | 0.655 | 0.298 | 38 |
| **Opus edit** | **0.9595** | **0.9895** | **0.896** | 0.550 | 0.151 | 51 |

chr1 is the only test chromosome that OpenSpliceAI itself held out (its test set is chr1, 3, 5, 7, 9);
chr11, chr16 and all dev chromosomes were in its training data. 68.5 % of test records have one review star;
the ≥ 2-star subset has 580 variants, 295 pathogenic. In ClinGen SVI guidance
([Walker et al., 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10357475/)) SpliceAI ≥ 0.2 supports PP3 and
≤ 0.1 supports BP4; 0.5 and 0.8 are the cut-offs from the original SpliceAI paper.

Paired bootstrap on the test variants (2,000 resamples, `scripts/bootstrap.py`):

| Comparison | ΔAUPRC, 95% CI | ΔAUROC, 95% CI |
|---|---|---|
| Opus edit − original | +0.036 [+0.028, +0.044] | +0.026 [+0.020, +0.032] |
| Temperature only − original | +0.031 [+0.024, +0.038] | +0.022 [+0.016, +0.028] |
| Opus edit − temperature only | +0.005 [+0.003, +0.007] | +0.004 [+0.002, +0.005] |
| 5-seed ensemble − original | +0.015 [+0.010, +0.020] | +0.011 [+0.007, +0.016] |
| Opus edit − 5-seed ensemble | +0.021 [+0.016, +0.028] | +0.015 [+0.009, +0.020] |
| Gradient G1 − original | +0.031 [+0.023, +0.040] | +0.023 [+0.017, +0.030] |
| Gradient G2 − original | +0.032 [+0.025, +0.041] | +0.024 [+0.018, +0.031] |
| Opus edit − gradient G1 | +0.005 [+0.001, +0.009] | +0.003 [−0.001, +0.006] |
| Opus edit − gradient G2 | +0.004 [−0.000, +0.007] | +0.002 [−0.002, +0.005] |
| Gradient G1 − temperature only | +0.000 [−0.005, +0.005] | +0.001 [−0.003, +0.005] |

The bootstrap treats the models as fixed and is not corrected for multiple comparisons. With a Bonferroni
correction for 14 whole-test comparisons (99.64 % intervals), the Opus edit is **not** significantly better
than either gradient control: vs G1 +0.005 [−0.001, +0.011], vs G2 +0.004 [−0.002, +0.009]; it stays
better than the ensemble (+0.021 [+0.013, +0.032]) and than temperature only (+0.005 [+0.002, +0.008]).
Other resampling schemes (`scripts/extra_analyses.py`): Opus − original, resampling whole genes,
+0.036 [+0.028, +0.046]; on chr1 alone +0.041 [+0.028, +0.056].

## What the edit was

Opus read the network and found that the class read-out is almost one-dimensional: skip channels 28 and 29
carry the acceptor and donor evidence, and real splice sites sit about +5 logits above "neither"
(p ≈ 0.995). The variant delta score is a *difference of probabilities*, so a pathogenic variant that removes
2–4 logits from such a site barely moves it — the softmax is saturated. Two edits
([`submissions/opus/edit.py`](submissions/opus/edit.py), reasoning in
[`REPORT.md`](submissions/opus/REPORT.md)):

1. **E1** — double the local-motif evidence (skip convolution after residual group 1, ±40 nt context) on
   channels 28/29: 66 parameters.
2. **E2** — scale `final_conv` by 0.35, i.e. raise the softmax temperature: 99 parameters.

E2 is exactly a softmax temperature: softmax(0.35·z) = softmax(z / T) with T ≈ 2.86. About 86 % of the
gain comes from it. E1 adds +0.005 AUPRC [+0.003, +0.007] on the whole test, but this is mostly a
reordering of variants *between* positions: within a position, AUROC is 0.957 for temperature only and
0.958 for the Opus edit. On chr1 alone E1 is not significant (+0.003 [−0.000, +0.007]). E1 was never tested
without E2. Practically all of the effect is the temperature.

## How to read it

- **The approach worked on a useful model and the gain transferred** — 0.024 % of the weights changed,
  AUPRC +0.036 on the hidden test, and the gain holds on chr1, which OpenSpliceAI never trained on.
- **It matched ordinary training rather than beating it.** Gradient fine-tuning of the same read-out layers
  on the same dev labels lands in the same place: G1 (99 parameters) is indistinguishable from the
  temperature-only control, G2 (5,379 parameters) is within noise of the Opus edit, and after correction for
  multiple comparisons neither gap is significant. A two-feature logistic regression (original DS plus the
  per-position pathogenic rate) reaches 0.9714, also at the gradient level; on top of the Opus edit the same
  regression reaches 0.9820, so the positional prior and the edit are partly complementary.
- **The valuable part is the diagnosis, not the surgery.** The fix is a classical knob (temperature), and a
  one-dimensional sweep of the multiplier on dev peaks at 0.30–0.35, exactly where Opus stopped (computed
  from the dev logits cached during the Opus run; the cache is not shipped). What reading the weights gave is
  the reason — strong sites are saturated in the softmax, so the delta score, a difference of probabilities,
  compresses large logit losses — and where in the network that is fixed. "Softmax is saturated" is a
  classical hypothesis an experienced engineer could also test. OpenSpliceAI's own `calibrate` step does not
  find it: fitted for site probabilities, it lands near T = 1 and leaves variant ranking as it was
  ([below](#temperature-on-all-five-seeds-floored-log-odds-and-calibrate)).
- **A tempered ensemble is as good as the edit.** All three single-model fixes beat the untempered 5-seed
  ensemble, which keeps the best guard (0.937). With a temperature the ensemble reaches 0.9753–0.9765,
  level with the Opus edit (difference −0.001 [−0.004, +0.002]). Reading its output as log-odds with a floor
  reaches 0.9794, above the edit (+0.005 [+0.002, +0.008]).
- **The guard barely sees temperature.** Top-k within a class is nearly invariant to it, so the guard checks
  E1 but not E2. Guard differences (Opus +0.4 pp, temperature only −0.04 pp, G1 −0.6 pp, G2 −0.3 pp) are
  within noise on 150 genes. On variants that create new sites deeper in the intron (D+7 and beyond) the
  edited models are worse than the original: test AUROC 0.868 → 0.837 (temperature only) and 0.863 (Opus).
  Deep intronic variants (> 50 nt from a boundary) are not in the benchmark.
- **The probability scale moved** (a strong site is ≈ 0.88 instead of 0.99). At DS ≥ 0.8 the Opus edit finds
  15 % of pathogenic variants instead of 49 %. Any practical use would need re-calibrated thresholds and
  re-validation.
- **The main lever does not transfer to ordinary classifiers.** Temperature changes the ranking here only
  because the score is a difference of two saturated probabilities. For a classifier that ranks by its own
  probability, temperature is monotone and leaves AUROC / AUPRC unchanged.
- **No new skill was taught.** As in the MIT work, the edit removes an obstacle; variants the network cannot
  read (e.g. exonic regulatory elements) stay missed.
- **Why the test gain (+0.036) is larger than the dev gain (+0.018).** The gain grows with the original
  model's headroom per chromosome (dev: +0.011 on chr17 to +0.058 on chr21; test: +0.028 to +0.041). Dev is
  dominated by NF1 (142 of 578 pathogenic variants), where the original is already near-perfect; without NF1
  the dev gain is +0.023.
- **Labels and sampling.** Some ClinVar classifications used SpliceAI, so labels are partly circular. Benign
  variants are one random 2:1 subsample; other subsamples were not tested. The delta-score window is ±50 nt,
  not the ±4,999 nt used by ClinGen.

## Temperature on all five seeds, floored log-odds and `calibrate`

Is the de-saturation a quirk of seed 10, and does OpenSpliceAI already offer it? `scripts/seeds_temperature.py`
caches the final-layer logits of all five seeds once (dev and test variants, dev guard genes; no weight is
changed) and then scores every variant with the logits divided by a temperature T before the softmax.
T\* is picked per seed on dev AUPRC (grid 0.5–20) and applied to test once. Sensitivity is measured at the
specificity the untempered model of the same seed has at DS = 0.2. Intervals: paired bootstrap over variants.

| Seed (test) | T = 1 | T = 1/0.35 (the Opus E2 factor) | T\* from dev | gain at T\*, 95 % CI | sensitivity T = 1 → T\* |
|---|---|---|---|---|---|
| rs10 | 0.9385 | 0.9695 | 0.9701 (T\* = 3.0) | +0.032 [+0.024, +0.039] | 0.761 → 0.884 |
| rs11 | 0.9413 | 0.9705 | 0.9708 (3.0) | +0.030 [+0.022, +0.037] | 0.769 → 0.868 |
| rs12 | 0.9413 | 0.9698 | 0.9716 (3.25) | +0.030 [+0.023, +0.038] | 0.772 → 0.862 |
| rs13 | 0.9454 | 0.9717 | 0.9731 (3.25) | +0.028 [+0.021, +0.035] | 0.779 → 0.858 |
| rs14 | 0.9346 | 0.9650 | 0.9676 (3.5) | +0.033 [+0.025, +0.042] | 0.741 → 0.850 |

| 5-seed ensemble (mean of softmax, as `openspliceai variant` does) | AUPRC | AUROC | sensitivity\* | gain vs T = 1, 95 % CI |
|---|---|---|---|---|
| T = 1 (current default) | 0.9532 | 0.9669 | 0.813 | — |
| T = 1/0.35 for every seed | 0.9756 | 0.9829 | 0.866 | +0.022 [+0.017, +0.029] |
| one common T\* from dev (2.75) | 0.9753 | 0.9827 | 0.866 | +0.022 [+0.016, +0.028] |
| per-seed T\* from dev | 0.9765 | 0.9836 | 0.871 | +0.023 [+0.017, +0.030] |
| per-seed temperatures from `calibrate` (re-implemented) | 0.9509 | 0.9654 | 0.805 | −0.002 [−0.004, −0.001] |
| T → ∞ limit (pure logit score, max \|Δ(z_c − mean z)\|) | 0.9416 | 0.9622 | 0.702 | — |
| class log-odds, no floor | 0.9268 | 0.9566 | 0.607 | — |
| **class log-odds, floor −4 chosen on dev** | **0.9794** | **0.9845** | **0.894** | +0.026 [+0.019, +0.034] |
| reference: Opus edit, single model | 0.9746 | 0.9815 | 0.863 | — |

\* at the specificity of the untempered ensemble at DS = 0.2.

**Floored log-odds** (suggested in a reader comment). Instead of a difference of probabilities, take for the
acceptor and donor class the log-odds L = log p − log(1 − p) of the stock model (T = 1), floor it at −c
(i.e. treat p below ≈ e^−c as that value), and score max |L_alt − L_ref| over the 101 positions and both
classes. The floor c is chosen on dev from {none, 12, 10, 8, 6, 5, 4, 3}; dev has a broad plateau at 4–6.
For the ensemble the five softmax outputs are averaged first.

| Seed (test) | floor (dev) | AUPRC | sensitivity at the stock specificity | vs T\* from dev, 95 % CI | no floor |
|---|---|---|---|---|---|
| rs10 | −5 | 0.9723 | 0.896 | +0.002 [−0.001, +0.005] | 0.9258 |
| rs11 | −5 | 0.9756 | 0.879 | +0.005 [+0.001, +0.009] | 0.9313 |
| rs12 | −5 | 0.9735 | 0.888 | +0.002 [−0.002, +0.006] | 0.9459 |
| rs13 | −4 | 0.9750 | 0.881 | +0.002 [−0.002, +0.006] | 0.9407 |
| rs14 | −6 | 0.9694 | 0.860 | +0.002 [−0.002, +0.005] | 0.9428 |

On the ensemble, floored log-odds beat the common temperature by +0.004 [+0.001, +0.007] and the Opus edit by
+0.005 [+0.002, +0.008]. Without a floor, log-odds are worse than the stock score: for 99 % of benign variants
the largest delta sits at a position where the site probability is ≤ 0.01, i.e. it is noise in log-odds of
near-zero probabilities. The floor removes that noise and keeps the large changes at strong sites that the
probability difference compresses. A temperature does a softer version of the same; the floor is one number
with a direct meaning (a probability floor: 0.7 % at −5, 1.8 % at −4, 0.25 % at −6). Selection: one scalar per reading, eight floors vs
a 0.5–20 grid of T, both on dev only.

- **It is not one seed.** Every seed has its dev optimum at T ≈ 3–3.5 and gains +0.028…+0.033 on test.
- **`calibrate` answers a different question.** `openspliceai calibrate` fits a per-class temperature vector by
  NLL over all positions of held-out sequences (Adam, lr 0.01, clamp [0.05, 5];
  `openspliceai/calibrate/temperature_scaling.py`). Re-implemented on the 60 dev guard genes it gives
  T = 0.87–1.25 per class and seed, i.e. the site probabilities are already close to calibrated, and variant
  AUPRC drops slightly. Honest site probabilities and good variant ranking need different temperatures. Note
  also that `openspliceai variant` does not apply a calibration temperature at all. The re-implementation uses
  our guard genes (GENCODE v46 basic), not OpenSpliceAI's own validation h5 files.
- **The gain is in how the output is read.** Temperature, floored log-odds and the Opus edit all de-saturate
  strong sites; the two readings need no weight change and the floored log-odds is the strongest. Reading raw
  logits or unfloored log-odds is worse than the stock score because of noise where there is no site.
- **What it would take to use it.** Both readings change the score scale, so the ClinGen thresholds
  (≥ 0.2 for PP3, ≤ 0.1 for BP4) would have to be re-calibrated; it could only be an extra scoring mode. The
  evidence here is ClinVar labels only, part of which were informed by SpliceAI, and chr11/chr16 were in
  OpenSpliceAI's training data; an independent check on experimentally measured splicing (minigene, MAVE,
  RNA-seq) is missing.

## Setup

| | |
|---|---|
| Model | OpenSpliceAI-MANE-10000nt, seed 10, 698,787 parameters, 10 kb context ([GPL-3.0](https://github.com/Kuanhao-Chao/OpenSpliceAI)) |
| Variants | ClinVar GRCh38 release 2026-11-01; SNVs within 50 nt of an internal MANE Select exon boundary; intronic / synonymous / splice-region consequences (missense, nonsense and canonical ±1/±2 excluded); ≥ 1 review star; P/LP = 1 vs B/LB = 0, at most 2 benign per pathogenic |
| Score | SpliceAI delta score DS_max = max(AG, AL, DG, DL) within ±50 nt |
| Metric | AUPRC (primary), AUROC; exon vs intron side; sensitivity/specificity at 0.2 |
| Guard | top-k splice-site accuracy on whole genes (as in the SpliceAI paper); allowed loss ≤ 1 pp |
| Split | dev = chr17, 19, 21, 22 (1,734 variants, given to participants); test = chr1, 11, 16 (1,842 variants) |
| Rules for participants | anything goes (code, forward passes, activations, literature) **except** gradient steps on the weights, automated weight search, other checkpoints and external variant labels; same architecture |

The exact brief given to the LLMs is [`task/TASK.md`](task/TASK.md).

**Blindness.** The Opus run (Claude Opus 5.5 in Claude Code, 4 Nov 2026, ~1 h 15 min, 7 candidates evaluated on
dev) happened while the test set was hidden; its tool-call log was audited for access to the test files.
The audit was done on the log, not enforced by a sandbox: the test files were readable by the same OS
user, internet access was allowed and web requests were not separately reviewed. The brief hints where the
headroom is (exon side) and that a monotone recalibration cannot change the metrics. The brief, the
controls and the scoring were prepared with Claude as well.
This repository, including the test set, was published before the second participant submitted, so any later
run is no longer blind. For new runs use a fresh split, e.g. ClinVar variants first submitted after this date.

**Gradient controls** ([`scripts/gradient_baseline.py`](scripts/gradient_baseline.py)) answer "is this better
than what one would normally do?". They are trained on the same dev labels, in the same read-out subspace the
edit touched: G1 = `final_conv` only (contains temperature), G2 = all five skip convolutions + `final_conv`.
Loss = pairwise AUC surrogate on DS_max + pairwise splice-site ranking loss on the dev guard (+ optional L2 to
the original weights). Hyper-parameters were chosen on chr21+22 after training on chr17+19, with the dev guard
allowed to drop ≤ 1 pp, then refit on all dev. The test set was used once per model.

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
bash scripts/fetch_data.sh --weights-only           # OpenSpliceAI weights (5 seeds, 14 MB), checksums
python scripts/check_data.py                        # shipped data/ matches the benchmark
bash scripts/reproduce_edit.sh opus                 # edit.py -> bit-identical model_edited.pt
cd scripts
python score.py --weights ../weights/model_10000nt_rs10.pt --name original_rs10
python score.py --weights ../submissions/opus/model_edited.pt --name opus
python make_controls.py && python score.py --weights ../controls/temperature_only_035.pt --name control_temperature_only
python score.py --ensemble ../weights --name ensemble_5_seeds
python bootstrap.py original_rs10 opus
python extra_analyses.py                            # positional baselines, chr1, gene bootstrap, operating points
```

`extra_analyses.py` needs only the shipped `results/` and `data/`. It also uses dev delta scores for the
original, temperature-only and Opus models (`results/dev/`), which were computed during the Opus run from
the same model code.

**Preprint numbers and figures.** Two more scripts reproduce everything else in the preprint:

```bash
python scripts/paper_analyses.py        # Table 2 with Bonferroni intervals, exon side, AUROC gain, BP4 shares (~1 min)
python scripts/make_figures.py          # Figs. 1-3 into figures/ (needs matplotlib)
python scripts/gradient_baseline.py feats --data data/dev --weights weights/model_10000nt_rs10.pt --cache cache
python scripts/paper_analyses.py --feats cache/var_feats.npy   # + dev temperature sweep, G1 margin analysis
python scripts/make_figures.py --feats cache/var_feats.npy     # + Fig. 4
```

The `feats` step runs the original model once over the dev windows (~20 min on CPU) and writes the inputs of
the five skip convolutions; from them the read-out is linear, so the temperature sweep and the G1 analysis
take seconds. The `figures/` folder holds the PDFs used in the preprint.

**Temperature on five seeds and `calibrate`:**

```bash
for s in 10 11 12 13 14; do python scripts/seeds_temperature.py logits --seed $s --cache cache/seeds; done
python scripts/seeds_temperature.py analyze --cache cache/seeds --out results/seeds_temperature.json
```

The `logits` step runs each seed once over dev and test variants and the dev guard genes (~40 min per seed on
a laptop CPU; five in parallel took 1.5 h on 10 cores, 78 MB of cache); `analyze` takes seconds.

Scoring one model takes ~20–30 min on a laptop CPU (the ensemble ~5× that). Scoring is deterministic up to
floating-point noise; the LLM runs themselves are not — a new run may find a different edit.

To rebuild the data from raw sources (~0.5 GB download):

```bash
bash scripts/fetch_data.sh                            # GRCh38 chromosomes, GENCODE v46, ClinVar 2026-11-01
python scripts/build_data.py --raw raw --out rebuilt
python scripts/check_data.py rebuilt
```

To rerun the gradient controls: `python scripts/gradient_baseline.py feats ...` then `train ...` (see the
docstring; feature caching ~25 min, training ~30 min on CPU).

To run a new participant: give it `task/TASK.md`, `scripts/osai.py`, `scripts/evaluate.py`, `data/dev/`,
`task/BASELINE_dev.json` and `weights/model_10000nt_rs10.pt` (not `data/test/`), then score its
`model_edited.pt` with `score.py`.

## Layout

```
task/          brief given to the LLM participants, dev baseline
data/dev/      dev variants (+ 600-variant fast subset) and 60 guard genes
data/test/     test variants and 150 guard genes (hidden during the Opus run)
submissions/   per participant: edit.py, model_edited.pt, REPORT.md, dev_log.jsonl
controls/      temperature-only and gradient-trained weights
results/       scores.jsonl (all test metrics), ds_<name>.npy (per-variant delta scores); dev/ — dev delta scores;
               seeds_temperature.json — temperature and floored log-odds on five seeds, ensemble, calibrate
scripts/       model code, data build, scoring, bootstrap, gradient controls, extra analyses,
               preprint numbers (paper_analyses.py) and figures (make_figures.py)
figures/       figures of the preprint
checksums/     SHA-256 of raw downloads, weights and data contents
```

## Licences and sources

- Code: GPL-3.0 (the model code follows OpenSpliceAI, GPL-3.0). Edited weights in `submissions/` and
  `controls/` are derivatives of OpenSpliceAI weights and are distributed under GPL-3.0 as well.
- OpenSpliceAI: K.-H. Chao et al., [github.com/Kuanhao-Chao/OpenSpliceAI](https://github.com/Kuanhao-Chao/OpenSpliceAI).
  SpliceAI: K. Jaganathan et al., *Cell* 2019. The original Illumina SpliceAI weights (CC BY-NC) are not used.
- Variants: [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) (NCBI). Annotation:
  [GENCODE](https://www.gencodegenes.org/) v46. Genome: GRCh38 via the UCSC Genome Browser.
- Nothing here is clinical advice; edited models are research artefacts and are not validated for diagnostics.
