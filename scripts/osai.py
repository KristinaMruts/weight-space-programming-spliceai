"""OpenSpliceAI-MANE-10000nt: architecture, safe weight loading, inference, variant delta score.

Self-contained (torch + numpy only). The architecture follows OpenSpliceAI
(github.com/Kuanhao-Chao/OpenSpliceAI, openspliceai/train_base/openspliceai.py, GPL-3.0),
which re-implements SpliceAI (Jaganathan et al., Cell 2019).

Input : one-hot DNA (N, 4, L), channel order A, C, G, T; 'N' = all zeros.
        Sequence must be in transcript (sense) orientation.
Output: (N, 3, L - 10000) softmax probabilities per position: 0 = neither, 1 = acceptor, 2 = donor.
        The first/last 5000 input positions are context only.

Weights are a plain state_dict and must be loaded with torch.load(..., weights_only=True).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

W = np.array([11] * 8 + [21] * 4 + [41] * 4)
AR = np.array([1] * 4 + [4] * 4 + [10] * 4 + [25] * 4)
CL = int(2 * np.sum(AR * (W - 1)))   # 10000
DIST = 50                             # variant scoring window: +-50 nt
WID = CL + 2 * DIST + 1               # 10101 nt input per variant
CENTER = WID // 2                     # 5050: variant position in the window


class ResidualUnit(nn.Module):
    def __init__(self, l, w, ar):
        super().__init__()
        self.batchnorm1 = nn.BatchNorm1d(l); self.batchnorm2 = nn.BatchNorm1d(l)
        self.conv1 = nn.Conv1d(l, l, w, dilation=ar, padding=(w - 1) * ar // 2)
        self.conv2 = nn.Conv1d(l, l, w, dilation=ar, padding=(w - 1) * ar // 2)

    def forward(self, x, y):
        out = self.conv1(F.leaky_relu(self.batchnorm1(x), 0.1))
        out = self.conv2(F.leaky_relu(self.batchnorm2(out), 0.1))
        return x + out, y


class Skip(nn.Module):
    def __init__(self, l):
        super().__init__()
        self.conv = nn.Conv1d(l, l, 1)

    def forward(self, x, y):
        return x, self.conv(x) + y


class SpliceAI(nn.Module):
    def __init__(self, L=32):
        super().__init__()
        self.initial_conv = nn.Conv1d(4, L, 1)
        self.initial_skip = Skip(L)
        self.residual_units = nn.ModuleList()
        for i, (w, r) in enumerate(zip(W, AR)):
            self.residual_units.append(ResidualUnit(L, int(w), int(r)))
            if (i + 1) % 4 == 0:
                self.residual_units.append(Skip(L))
        self.final_conv = nn.Conv1d(L, 3, 1)

    def forward(self, x, logits=False):
        x = self.initial_conv(x)
        x, skip = self.initial_skip(x, 0)
        for m in self.residual_units:
            x, skip = m(x, skip)
        out = self.final_conv(skip[:, :, CL // 2: skip.shape[2] - CL // 2])
        return out if logits else F.softmax(out, dim=1)


def load(path) -> SpliceAI:
    m = SpliceAI()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.eval()


_LUT = np.zeros((256, 4), dtype=np.float32)
for i, b in enumerate("ACGT"):
    _LUT[ord(b)] = 0; _LUT[ord(b), i] = 1; _LUT[ord(b.lower()), i] = 1


def onehot(seqs: list[str]) -> torch.Tensor:
    arr = np.frombuffer("".join(seqs).encode(), dtype=np.uint8).reshape(len(seqs), -1)
    return torch.from_numpy(_LUT[arr]).permute(0, 2, 1).contiguous()


@torch.no_grad()
def predict(model, seqs: list[str], batch=16) -> np.ndarray:
    out = []
    for i in range(0, len(seqs), batch):
        out.append(model(onehot(seqs[i:i + batch])).numpy())
    return np.concatenate(out)


@torch.no_grad()
def delta_scores(model, ref_windows: list[str], alts: list[str], batch=64) -> np.ndarray:
    """SpliceAI delta score for SNVs, raw (no masking), distance 50.

    ref_windows: 10101-nt reference windows in transcript orientation, variant at index 5050.
    alts: alternative base (already in transcript orientation).
    Returns (n, 4): acceptor gain, acceptor loss, donor gain, donor loss. DS_max = max over the 4.
    """
    res = []
    for i in range(0, len(ref_windows), batch):
        r = ref_windows[i:i + batch]; a = alts[i:i + batch]
        alt = [s[:CENTER] + b + s[CENTER + 1:] for s, b in zip(r, a)]
        y = model(onehot(r + alt)).numpy()            # (2b, 3, 101)
        yr, ya = y[:len(r)], y[len(r):]
        d = ya - yr
        res.append(np.stack([d[:, 1].max(1), (-d[:, 1]).max(1), d[:, 2].max(1), (-d[:, 2]).max(1)], 1))
    return np.concatenate(res)
