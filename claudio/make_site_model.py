"""Build the site checkpoint of "Tonkclaudinho Filhodegei V2" (the trained Claudio).

The site runs the v4 TonkAI engine in the browser (no Claudio search), so this
is a tonkai-cognitive-v3 checkpoint the site's exporter understands:
- the trained Claudio network (claudio-sf2850-1723-recovered-pawn1.pt);
- its piece values (K/D + mate cuts), passed to the exporter explicitly;
- Timdgei's knowledge (the retrieval memory the trained network grew up with);
- the configuration of the model now on the site, so the browser engine and
  the Python reference agree (the newer rules are not in the browser engine).

    python claudio/make_site_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402  (frozen base code first)

import json  # noqa: E402

import torch  # noqa: E402

TRAINED = ROOT / "claudio/claudio-sf2850-1723-recovered-pawn1.pt"
TIMDGEI = ROOT / "checkpoints/obj-max-continuo/latest.pt"
PUBLISHED = ROOT / "checkpoints/cognitive-mix-20260923/latest.pt"     # the model on the site now
STOPPED = ROOT / "claudio/results/stopped-values-sf2850-1723.json"
OUT = ROOT / "claudio/site/tonkclaudinho-filhodegei-v2.pt"
NAME = "Tonkclaudinho Filhodegei V2"


def main():
    trained = torch.load(TRAINED, map_location="cpu", weights_only=False)
    state = torch.load(TIMDGEI, map_location="cpu", weights_only=False)
    site_config = torch.load(PUBLISHED, map_location="cpu", weights_only=False)["config"]
    if set(trained["net"]) != set(state["model"]):
        raise ValueError("the trained network does not match TonkAI's architecture")
    config = dict(state["config"])
    for key in list(config):                    # rules the browser engine has not got: their defaults (off)
        if site_config.get(key) is None:
            config.pop(key)
    config["symmetric_piece_values"] = True     # how this network learned piece values; the site now reads it
    stopped = json.loads(STOPPED.read_text(encoding="utf-8"))            # the interrupted 1723-game run
    trained_games = sum(h.get("trained_games", 0) for h in trained.get("history", [])) + stopped["games"]
    state.update(model=trained["net"], config=config, model_name=NAME, status="stopped",
                 games=state["games"] + trained_games,
                 kd_values_override={int(k): float(v) for k, v in trained["values"].items()},
                 claudio_history=trained.get("history", []))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, OUT)
    print(f"wrote {OUT}: {NAME}, games {state['games']} (+{trained_games} Claudio), "
          f"values {state['kd_values_override']}, config keys {len(config)} (site {len(site_config)})")


if __name__ == "__main__":
    main()
