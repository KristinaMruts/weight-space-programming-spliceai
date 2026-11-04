# Weight-space edit of OpenSpliceAI-MANE-10000nt (seed 10): report

Participant: Claude Opus 5.5 (Claude Code), working only in `wsp-arena/opus/`.

## Summary

Two hand-made edits change **165 of 698,787 parameters (0.024 %)** in two weight/bias tensor pairs:

| Edit | Tensor(s) | Indices | Change | Params |
|---|---|---|---|---|
| E1: gain on the local-motif stage | `residual_units.4.conv.weight`, `.bias` | output rows 28, 29 | × 2.0 | 66 |
| E2: de-saturate the read-out | `final_conv.weight`, `final_conv.bias` | all | × 0.35 | 99 |

Dev results, scored with the stock `evaluate.py` on the full dev set and the dev guard genes:

| Metric (dev, n = 1,734, 578 P) | Original | Edited | Δ |
|---|---|---|---|
| **AUPRC (primary)** | 0.9673 | **0.9850** | +0.0177 |
| AUROC | 0.9767 | 0.9879 | +0.0112 |
| exon side AUPRC / AUROC (n = 597) | 0.8714 / 0.9481 | 0.9319 / 0.9732 | +0.0605 / +0.0251 |
| intron side AUPRC / AUROC (n = 1,137) | 0.9804 / 0.9810 | 0.9910 / 0.9900 | +0.0106 / +0.0090 |
| sensitivity / specificity at DS_max ≥ 0.2 | 0.8374 / 0.9827 | 0.8893 / 0.9965 | +0.052 / +0.014 |
| guard top-k acceptors / donors / **mean** | 0.9545 / 0.9407 / **0.9476** | 0.9526 / 0.9407 / **0.9466** | −0.10 pp (limit −1.0 pp) |
| guard per-position AUPRC acceptors / donors | 0.9869 / 0.9831 | 0.9877 / 0.9833 | +0.0008 / +0.0002 |

The improvement does not come from position: AUROC *within* each site position (pair-weighted over all
positions that have both classes) rises from 0.9791 to 0.9826, and AUPRC rises on every dev chromosome
(chr17 0.979 → 0.990, chr19 0.949 → 0.977, chr21 0.906 → 0.964, chr22 0.948 → 0.972).

What this is and what it is not. Neither edit teaches the network a new sequence feature. The
network already "knows" which variants damage a site: its **logits** separate them well. The edits
change (1) how much the read-out trusts the local splice-motif stage relative to the long-range stages
and (2) how logit changes are turned into probability changes, which is where most of the ranking
error was. The second edit is close to a calibration, but it is not a monotone transform of DS_max and
it does change which variants rank as more disruptive (details below).

## What I found in the network

All numbers are on dev reference/alternative windows, from an activation cache (the residual stream
at the 5 skip points on the 101 scored positions, plus the input to group 4), checked to reproduce the
original model's logits to 1e-5 and `BASELINE.json` exactly.

**1. The class read-out runs through two skip channels.** `final_conv` rows are nearly identical
across the three classes (a common mode that the softmax ignores). The class-discriminative parts are

- acceptor − neither: −0.303 on skip channel **28**, |w| ≤ 0.016 on every other channel;
- donor − neither: −0.295 on skip channel **29**, −0.05 on channel 28, |w| ≤ 0.012 elsewhere.

So skip channels 28 and 29 are the acceptor and donor evidence lines, and the skip sum is linear: the
site logit splits exactly into a contribution from each of the 5 skip convolutions (initial, after
groups 1–4) plus the bias.

**2. Real sites are saturated, and saturation is what hurts the delta score.** The median annotated
site sits at logit +5.2 above "neither" (p ≈ 0.994). Per-stage means of the site logit (strong sites,
p > 0.9):

| stage (receptive field) | initial (1 nt) | group 1 (±40 nt) | group 2 (±200 nt) | group 3 (±1 kb) | group 4 (±5 kb) | bias |
|---|---|---|---|---|---|---|
| acceptor sites (n = 720) | −0.33 | 0.28 | 0.44 | 2.58 | 2.91 | −0.72 |
| donor sites (n = 826) | −0.30 | 0.99 | 0.15 | 1.59 | 3.39 | −0.64 |
| logit lost, pathogenic acceptor variants (n = 184) | 0.00 | 1.03 | 2.20 | 3.01 | 2.23 | |
| logit lost, benign acceptor variants | 0.00 | 0.01 | 0.05 | 0.01 | 0.02 | |
| logit lost, pathogenic donor variants (n = 297) | 0.07 | 1.61 | 1.44 | 1.74 | 2.47 | |
| logit lost, benign donor variants | 0.00 | 0.01 | 0.03 | 0.03 | 0.04 | |

On the 1,546 dev variants whose reference site has p > 0.9 (481 pathogenic), the **logit** lost at the
site ranks pathogenic above benign with AUROC 0.993, while DS_max on the same variants gets 0.981.
Typical misses of the original model are D+5, D+3, D-1 and A-3 variants that remove 2–4 logits from a
site at p = 0.995–0.999: the probability falls by only 0.01–0.10, below most benign noise. Of the 63
pathogenic variants ranked below position 578 by the original model, 58 sit at a strong reference
site.

**3. The local-motif stage is a clean site/background discriminator.** The contribution of the
group-1 skip (rows 28/29) to the class logit is +0.25 (acceptor) / +0.92 (donor) at annotated sites,
but −3.2 / −3.7 (sd ≈ 0.9) at all other scored positions. It also carries a large share of the
pathogenic signal with almost no benign movement (table above). Signal-to-noise per stage
(mean pathogenic loss / mean |benign change|) is 13–29 for stages 1–4, highest for the donor channel
in group 1 (29) — not a dramatic outlier, so I tested E1 as one hypothesis rather than assuming it.

**4. What does not work.** Shifting the "neither" bias up by 2.6 (same de-saturation of a typical
site as E2, but at full slope) made things worse (AUPRC 0.956): it inflates small logit changes and
compresses large ones. So the gain from E2 comes from the **dynamic range** of the logit-to-probability
map, not from favouring losses over gains.

## What I changed and why

- **E2 (`final_conv` × 0.35).** Equivalent to a softmax temperature of 1/0.35 inside the network.
  The median site logit goes from 5.2 to 1.8 (site p from 0.994 to 0.88; 10–90 % range 0.70–0.94),
  so site losses of 2–10 logits map to proportionate probability drops, while background positions
  (logits around −22) still map to p ≈ 1e-4, so their noise is not amplified. Within one class the
  position ranking used by top-k is essentially unchanged (one dev acceptor flips because acceptor
  and donor logits compete in the softmax).
- **E1 (group-1 skip rows 28/29 × 2).** Doubles the local-motif evidence in both class channels.
  Effects measured against E2-like read-out alone: donor-loss DS of pathogenic variants grows (median
  ×1.11), donor-gain/-loss DS of benign variants shrinks (median ×0.69 / ×0.75) because background
  positions are pushed further from "donor". Long-range stages are untouched. Guard top-k unchanged.
  Read together, E1 + E2 amount to "read the local motif at temperature 0.7 and the long-range context
  at temperature 0.35".

Both factors are round, hand-chosen numbers. I did not tune them further: × 3 instead of × 2 for E1
was worse than × 2, and I stopped there.

## Candidate edits evaluated on dev

Seven candidate edits were scored on the full dev variants (plus the original as a reference), six
guard runs, and one final stock-`evaluate.py` run of the submitted file. Every run is in
`dev_log.jsonl`; each variant entry also carries the metrics on `dev_fast` (600) and on the remaining
1,134 variants, as a rough check that a gain is not carried by one half.

| # | candidate | AUPRC | AUROC | exon AUPRC | sens / spec @0.2 | guard mean |
|---|---|---|---|---|---|---|
| 0 | original | 0.9673 | 0.9767 | 0.8714 | 0.837 / 0.983 | 0.9476 |
| 1 | final_conv × 0.7 | 0.9746 | 0.9815 | 0.8936 | 0.858 / 0.988 | — |
| 2 | final_conv × 0.5 | 0.9797 | 0.9846 | 0.9115 | 0.870 / 0.991 | 0.9466 |
| 3 | final_conv × 0.35 | 0.9825 | 0.9863 | 0.9248 | 0.863 / 0.995 | 0.9466 |
| 4 | "neither" bias + 2.6 | 0.9562 | 0.9672 | 0.8732 | 0.830 / 0.968 | — |
| 5 | × 0.5 + group-1 rows 28/29 × 2 | 0.9826 | 0.9869 | 0.9212 | 0.884 / 0.992 | 0.9466 |
| 6 | **× 0.35 + group-1 rows 28/29 × 2 (submitted)** | **0.9850** | **0.9879** | **0.9319** | **0.889 / 0.997** | **0.9466** |
| 7 | × 0.5 + group-1 rows 28/29 × 3 | 0.9821 | 0.9865 | 0.9184 | 0.884 / 0.990 | 0.9466 |

Cached evaluation reproduces the stock evaluator exactly for edits confined to the skip convolutions
and `final_conv` (verified on the original and on the submitted model).

No gradient with respect to weights was computed at any point; no optimiser, no random/grid/Bayesian
search. Analysis used forward passes, activation decomposition and dev labels only.

Time: about 1 h 15 min wall-clock (11:30–12:45), of which ~25 min was building the activation cache (one full dev
forward pass on CPU) and ~15 min the final stock evaluation.

## Region-level picture (AUROC, original → submitted)

| region | n (P) | original | edited |
|---|---|---|---|
| A-20..A-4 (polypyrimidine tract) | 424 (141) | 0.982 | 0.989 |
| A-3 | 59 (55) | 0.973 | 0.986 |
| D exon side (D-1 … D-48) | 319 (72) | 0.956 | 0.989 |
| D+3..D+6 | 289 (277) | 0.950 | 0.971 |
| D+7 and deeper | 336 (13) | 0.987 | **0.960** |
| A exon side (A+1 … A+50) | 278 (7) | 0.832 | **0.811** |
| A-21 and deeper | 29 (13) | 0.841 | 0.846 |

## Will it generalise to the hidden test? Expectation and risks

Expectation: **yes, most of the gain should carry over**, because the mechanism is global and does
not depend on any gene or chromosome: strong annotated sites are saturated everywhere, and the
pathogenic near-splice variants that the original model under-scores are mostly weakenings of strong
sites. The gain is present on every dev chromosome and on both halves of dev. I would expect hidden-test
AUPRC to rise by roughly +0.01 to +0.02 over the original checkpoint (dev: +0.018), with the guard
within −0.1 to −0.3 pp. Guard risk is low: neither edit touches the residual stream, and
within-class position ranking is nearly invariant to E2.

Risks:

1. **Selection on dev.** I picked the read-out factor among three values and the gain among two by dev
   metrics. With one or two scalars and 1,734 variants the optimism should be small, but it is not zero.
2. **Deep intronic and deep exonic gains get relatively weaker.** Regions where pathogenic variants
   create cryptic sites (D+7 and deeper, acceptor exon side) lost AUROC on dev (13 and 7 pathogenic
   variants only, so noisy). If the hidden set has more cryptic-site-creating pathogenic variants, the
   gain shrinks.
3. **Label circularity.** Some ClinVar classifications used SpliceAI scores; part of what I "fix" may
   be agreement with the original SpliceAI's logit-level behaviour rather than biology.
4. **Calibration changes; the clinical cut-offs change meaning.** Strong-site probabilities drop from
   ~0.99 to ~0.88. Delta scores are now spread differently: on dev, at the familiar 0.2 cut-off both
   sensitivity and specificity improved, but the published SpliceAI cut-offs (0.2 / 0.5 / 0.8) were
   set for the original scale and would need re-validation before clinical use. The guard (top-k) does
   not see calibration at all.
5. **Not a new skill.** The model still misses pathogenic variants whose logit change is small
   (e.g. a D-1 G>T next to a perfect GTGAGT donor, A-3 C>T keeping a YAG). Fixing those would need the
   network to read something it does not read now (exonic regulatory elements, context-dependent
   donor rules); I found no edit for that.

## Files

- `edit.py` — deterministic; reads `model/model_10000nt_rs10.pt`, writes `submission/model_edited.pt`,
  prints the changed-parameter count.
- `model_edited.pt` — the edited `state_dict` (same keys, shapes, dtypes; CPU tensors).
- `dev_log.jsonl` — every dev evaluation (variant metrics with `fast`/`rest` splits, guard runs,
  final stock evaluation).
