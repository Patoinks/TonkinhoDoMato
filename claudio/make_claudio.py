"""Write claudio/claudio.pt: Timdgei's network (no training was used - the owner
called the duel first), the piece values TonkAI measured in its own games,
the learned king-safety model, and Claudio's settings."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

import torch  # noqa: E402
from match_checkpoints import _load  # noqa: E402
from tonkai.king_safety_model import load_cached  # noqa: E402
from claudio_agent import ClaudioConfig  # noqa: E402
from claudio_load import architecture_of, save_claudio  # noqa: E402

SETTINGS = ["tactic_top=4", "root_pvs=true", "mate_attack=3", "mate_gate=0.8", "king_weight=0.047",
            "repeat_as_draw=true"]
source = ROOT / "checkpoints/obj-max-continuo/latest.pt"
name, net, knowledge, cfg, kd = _load(str(source))
cfg_claudio = ClaudioConfig.with_settings(SETTINGS)
out = ROOT / "claudio/claudio.pt"
save_claudio(out, net, architecture_of(cfg), kd, cfg_claudio, [{"from": str(source), "network": name,
             "values": "K/D measured by TonkAI in its own games (external@2850)", "trained_minutes": 0}],
             king_model=load_cached(cfg.king_safety_model))
print("wrote", out, cfg_claudio)
