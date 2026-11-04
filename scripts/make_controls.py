"""Control: the temperature part of the Opus edit alone (final_conv x 0.35), no other change.

  python scripts/make_controls.py      # writes controls/temperature_only_035.pt
"""
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sd = torch.load(ROOT / "weights/model_10000nt_rs10.pt", map_location="cpu", weights_only=True)
sd = {k: v.detach().clone().cpu() for k, v in sd.items()}
sd["final_conv.weight"] *= 0.35
sd["final_conv.bias"] *= 0.35
(ROOT / "controls").mkdir(exist_ok=True)
torch.save(sd, ROOT / "controls/temperature_only_035.pt")
print("wrote controls/temperature_only_035.pt")
