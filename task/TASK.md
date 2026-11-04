# Task: improve a clinical splicing model by editing its weights directly

## Why we are doing this

A recent MIT CSAIL report ("Sparks of Weight-Space Programming", Takehi, Teoh, Zhang, Khattab,
https://rikiyat.github.io/blog/weight-space-programming/) showed that a strong language model can read a
neural network's weights and improve the network by editing them directly — no gradient descent, no
fine-tuning. In their main result, the model found the neurons that made Qwen3.5-0.8B loop in its
reasoning, edited 0.016% of the weights, and MATH-500 rose from 7.9% to 48.1%.

We are a digital pathology / molecular diagnostics company. We want to know whether this approach works
on a model that matters in practice, not on a toy. You are one of two independent attempts; the other is
run by a different AI model on exactly the same package. Both results are scored by us on a hidden test set.

## The model

**OpenSpliceAI-MANE-10000nt** (Chao et al., 2025; github.com/Kuanhao-Chao/OpenSpliceAI, GPL-3.0), a
PyTorch re-training of **SpliceAI** (Jaganathan et al., Cell 2019) on GRCh38 with MANE annotations.
SpliceAI-type scores are used in clinical genetics labs worldwide to decide whether a DNA variant disrupts
splicing (ACMG/ClinGen PP3/BP4 evidence for splicing).

- Architecture: 1-D residual dilated CNN, 32 channels, 16 residual units (kernel 11/11/21/41, dilation
  1/4/10/25), skip connections, 698,787 parameters. Input: one-hot DNA (A, C, G, T) with 5,000 nt of
  context on each side; output per position: softmax over {neither, acceptor, donor}.
- You get one checkpoint: `model/model_10000nt_rs10.pt` (a plain `state_dict`, random seed 10 of 5).
- `osai.py` — the full architecture, loader, inference and variant delta score (read it first).

## The clinical task and the metric

For a single-nucleotide variant, the model predicts splice-site probabilities for the reference and the
alternative sequence over ±50 nt; the **delta score DS_max** = max(acceptor gain, acceptor loss,
donor gain, donor loss). A high DS_max says "this variant disrupts splicing".

Benchmark: ClinVar single-nucleotide variants within 50 nt of an internal exon boundary of a MANE Select
transcript, intronic / synonymous / splice-region only (no missense, nonsense, etc.), **excluding the
canonical dinucleotides** (donor +1/+2, acceptor −1/−2), review status ≥ 1 star.
Pathogenic/likely pathogenic = 1, benign/likely benign = 0 (≤ 2 benign per pathogenic).
These non-canonical variants are where labs actually argue about variants of uncertain significance.

- **Primary metric: AUPRC of DS_max** for pathogenic vs benign. Secondary: AUROC.
- **Guard (must not break):** ordinary splice-site recognition — top-k accuracy on whole genes, as in the
  SpliceAI paper. The edited model must not lose more than **1.0 percentage point** of mean top-k
  accuracy on our hidden guard genes. An edit that breaks basic splice-site recognition does not count.

Final scoring is done by us on a **hidden test set**: the same kind of ClinVar variants on other
chromosomes, plus hidden guard genes on a chromosome the model was never trained on.

## What you get (this folder)

| File | What |
|---|---|
| `osai.py` | architecture, safe loader, `predict`, `delta_scores` |
| `evaluate.py` | dev evaluation: AUPRC/AUROC, by exon/intron side, sens/spec at 0.2, guard top-k |
| `model/model_10000nt_rs10.pt` | the weights you start from |
| `data/dev_variants.tsv.gz` | dev variants: 1,734 (578 pathogenic), chr17/19/21/22 |
| `data/dev_fast_variants.tsv.gz` | 600-variant subset (200 P + 400 B) for fast iteration |
| `data/dev_guard_genes.jsonl.gz` | 60 MANE genes for the guard (top-k), dev chromosomes |
| `BASELINE.json` | the original checkpoint's dev metrics |

Variant columns: `id chrom pos ref alt strand gene site mc stars label ref_tx alt_tx window`.
`window` is the 10,101-nt reference sequence **in transcript orientation** (N outside the transcript,
exactly as OpenSpliceAI scores variants), the variant is at index 5050, `alt_tx` is the alternative base in
transcript orientation. `site` is the position relative to the nearest boundary: `D-1` = last exon base
before a donor, `D+5` = 5th intronic base after a donor, `A-3` = 3rd intronic base before an acceptor,
`A+1` = first exon base after an acceptor.

Original checkpoint on dev (from `BASELINE.json`):

| Metric (dev) | Original |
|---|---|
| AUPRC (primary) | 0.9673 |
| AUROC | 0.9767 |
| exon side AUPRC / AUROC | 0.8714 / 0.9481 (n=597) |
| intron side AUPRC / AUROC | 0.9804 / 0.981 (n=1137) |
| sensitivity / specificity at DS_max >= 0.2 | 0.8374 / 0.9827 |
| guard top-k acceptors / donors / mean | 0.9545 / 0.9407 / 0.9476 |

Context for these numbers: the original model is already strong, so headroom is small and lies mostly on the
exon side of the boundary and in sensitivity at the clinical threshold 0.2. Note also a positional shortcut: a
score that uses *only* the distance to the boundary already reaches AUROC 0.856 / AUPRC 0.764 on dev, because
pathogenic variants cluster at D+5, D+3, D-1, A-3. An improvement should come from better sequence reading,
not from leaning harder on position.

Run: `python evaluate.py --weights model/model_10000nt_rs10.pt --variants data/dev_variants.tsv.gz --guard data/dev_guard_genes.jsonl.gz`
(Speed on an 8-core Apple M-series CPU, batch 64: about 0.3 s per sequence, i.e. ~6 min for `dev_fast`
(1,200 sequences: reference + alternative), ~18 min for the full dev set, ~1 min for the guard. A CUDA or Apple
MPS GPU is much faster if you have one. Caching activations is up to you.)

## Rules

You may use **any tools**: Python, PyTorch, as many forward passes on dev data as you want, activation
analysis, attribution (including gradients with respect to inputs or activations, for analysis), the
literature, the internet, splicing biology, motif databases, the reference genome and annotations.

What makes this *weight-space programming* rather than ordinary training:

1. **No optimiser on the weights.** Do not compute weight updates by gradient descent or any loss-gradient
   with respect to the weights, and do not run automated searches over weight values (random search,
   evolutionary, Bayesian optimisation, large grid sweeps). Every change must be a deliberate edit you can
   explain. Trying a handful of hand-chosen candidate edits on dev and keeping the best is fine — record
   how many you tried.
2. **Same architecture.** The result must be a `state_dict` with exactly the same keys and shapes, scored
   by the unchanged `osai.py`. No extra layers, no post-processing outside the network.
3. **No outside variant labels.** Do not use ClinVar, HGMD, SpliceVarDB, MFASS or any other database of
   variant pathogenicity or measured splicing outcomes — the hidden test comes from ClinVar. The only
   labels you may use are the dev labels provided here. Unlabelled public data (genome, GENCODE/MANE,
   conservation, known splicing motifs) is fine.
4. **No other checkpoints.** Do not import or average weights from other OpenSpliceAI/SpliceAI
   checkpoints or seeds.

A null result is a valid result. If after honest effort you cannot improve the model without breaking
the guard, say so and explain what you learned about why.

## What to deliver (in `submission/`)

1. `edit.py` — a deterministic script: reads `model/model_10000nt_rs10.pt`, applies your edits, writes
   `submission/model_edited.pt`. Each edit commented: which tensor, which indices, why.
2. `model_edited.pt` — the edited weights.
3. `REPORT.md`:
   - what you found in the network (which units/channels/layers do what, with evidence);
   - what you changed and why; how many weights changed (count and % of 698,787);
   - dev metrics before → after (full dev + guard), by exon/intron side;
   - how many candidate edits you evaluated on dev, and roughly how much time you spent;
   - your own expectation of whether it generalises to the hidden test, and risks.
4. `dev_log.jsonl` — one line per dev evaluation you ran: candidate name, metrics.

Notes on interpretation: some ClinVar classifications for splicing variants were themselves informed
by SpliceAI scores, so labels are partly circular; and AUPRC/AUROC are ranking metrics — a monotonic
recalibration of the output (for example, only shifting a final bias) cannot change them. The improvement
has to change *which* variants the model ranks as more disruptive.
