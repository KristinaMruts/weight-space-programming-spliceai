"""Weight-space edit of OpenSpliceAI-MANE-10000nt (seed 10).

Reads  model/model_10000nt_rs10.pt
Writes submission/model_edited.pt   (same keys, same shapes, float32, CPU tensors)

Deterministic: two multiplicative edits by hand-chosen constants, no optimiser, no search.
Total changed parameters: 99 + 66 = 165 of 698,787 (0.024 %).

Background (see REPORT.md for evidence):
  * The class read-out is almost one-dimensional per class. In final_conv, the difference between
    the acceptor row and the "neither" row is ~ -0.30 on skip channel 28 and |w| < 0.02 elsewhere;
    the donor-minus-neither row is ~ -0.30 on skip channel 29 (and -0.05 on 28). So skip channels
    28 / 29 are the acceptor / donor evidence channels, and each of the 5 skip convolutions writes
    its stage's evidence into them.
  * Real splice sites sit at logit ~ +5 (median) above "neither", i.e. p ~ 0.995. The variant delta
    score is a difference of probabilities, so a pathogenic variant that removes 2-4 logits from such
    a site moves p by only 0.01-0.05: the softmax is saturated. On dev, the logit loss at the site
    separates pathogenic from benign far better (AUROC 0.993) than DS_max does (0.981) on the same
    1,546 strong-site variants.
"""
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "model" / "model_10000nt_rs10.pt"
DST = ROOT / "submission" / "model_edited.pt"

sd = torch.load(SRC, map_location="cpu", weights_only=True)
sd = {k: v.detach().clone().cpu() for k, v in sd.items()}
orig = {k: v.clone() for k, v in sd.items()}

# ---------------------------------------------------------------------------------------------
# Edit 1 — local-motif stage gain on the acceptor / donor evidence channels.
#   Tensor : residual_units.4.conv.weight [32, 32, 1], rows 28 and 29 (all 32 input channels)
#            residual_units.4.conv.bias   [32],        entries 28 and 29
#   Change : x 2.0
#   Why    : residual_units.4 is the Skip after residual group 1 (4 units, kernel 11, dilation 1),
#            whose receptive field is +-40 nt, i.e. the local splice motif itself. Rows 28/29 are the
#            part of that skip that feeds the acceptor/donor logits. Decomposing the site logit by
#            stage shows this stage carries a clean variant signal (pathogenic donor variants lose
#            1.6 logits here, benign ones 0.01), and at non-site positions its contribution is
#            negative, so doubling it (a) enlarges true motif disruptions and (b) pushes background
#            positions further from "donor"/"acceptor", shrinking spurious gain/loss noise of benign
#            variants (median donor DS of benign variants falls ~25-30 %). The long-range stages
#            (groups 2-4) are untouched, so exon-context reading is preserved; guard top-k unchanged.
#   Count  : 2 x 32 weights + 2 biases = 66 parameters.
# ---------------------------------------------------------------------------------------------
LOCAL_GAIN = 2.0
for t in ("residual_units.4.conv.weight", "residual_units.4.conv.bias"):
    sd[t][[28, 29]] *= LOCAL_GAIN

# ---------------------------------------------------------------------------------------------
# Edit 2 — de-saturate the read-out (softmax temperature inside the network).
#   Tensor : final_conv.weight [3, 32, 1] (all), final_conv.bias [3] (all)
#   Change : x 0.35
#   Why    : maps the median real-site logit (~ +5.2) to ~ +1.8, the steep part of the softmax, so a
#            loss of 2-10 logits at a strong site becomes a proportional probability drop instead of
#            being flattened at p ~ 0.99. This is not a monotone transform of DS_max: it changes which
#            variants rank higher (strong-site weakenings move above small gains/losses at weak
#            positions). Within a class the per-position ranking used by the top-k guard is almost
#            unchanged (it can only shift where acceptor and donor logits compete).
#            0.35 rather than 0.5 / 0.7 (all three were evaluated, see dev_log.jsonl): with
#            alpha * 5.2 ~ 1.8 the whole range of pathogenic site losses (2-10 logits) stays inside the
#            non-saturated part of the sigmoid, while logits at weak positions (~ -10 to -25) still map
#            to p ~ 0, so noise there is not amplified.
#   Count  : 96 weights + 3 biases = 99 parameters.
# ---------------------------------------------------------------------------------------------
TEMPERATURE_SCALE = 0.35
sd["final_conv.weight"] *= TEMPERATURE_SCALE
sd["final_conv.bias"] *= TEMPERATURE_SCALE

# ---------------------------------------------------------------------------------------------
# Sanity: same keys/shapes/dtypes; count changed parameters.
# ---------------------------------------------------------------------------------------------
assert sd.keys() == orig.keys()
changed = 0
for k in sd:
    assert sd[k].shape == orig[k].shape and sd[k].dtype == orig[k].dtype, k
    changed += int((sd[k] != orig[k]).sum())
total_all = 698_787
print(f"changed parameters: {changed} ({100 * changed / total_all:.4f} % of {total_all:,})")
torch.save(sd, DST)
print(f"wrote {DST}")
