"""Claudio's checkpoint: network weights, its own piece values, its settings.

    {"format": "claudio-v1", "net": state_dict, "architecture": {...},
     "values": {1..5: pawn units}, "config": ClaudioConfig fields, "history": [...]}
"""
from dataclasses import asdict
from types import SimpleNamespace

import torch

ARCHITECTURE = ("hidden_size", "latent_size", "head_hidden", "piece_hidden", "type_embed", "square_embed",
                "side_embed", "memory_size", "ensemble_size", "future_horizons", "capture_target_weight")


def architecture_of(cfg):
    return {name: getattr(cfg, name) for name in ARCHITECTURE if hasattr(cfg, name)}


def build_net(architecture):
    from claudio_net import build_net as build
    return build(architecture)


def save_claudio(path, net, architecture, values, cfg, history, king_model=None):
    torch.save({"format": "claudio-v1", "net": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "architecture": dict(architecture), "values": {int(k): float(v) for k, v in values.items()},
                "config": asdict(cfg), "history": list(history),
                "king": None if king_model is None else {"hidden": king_model.net.hidden,
                                                         "state_dict": king_model.net.state_dict()}}, path)


def load_claudio(path, with_config=False, with_king=False):
    """(net, values) - plus (config dict, history) with with_config, plus the KS model with with_king."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("format") != "claudio-v1":
        raise ValueError(f"{path} is not a Claudio checkpoint")
    net = build_net(data["architecture"])
    net.load_state_dict(data["net"])
    net.eval().requires_grad_(False)
    if data.get("organic"):
        if data["organic"].get("format") != "martinus-organic-v1":
            raise ValueError("unsupported organic checkpoint")
        net._organic_payload = data["organic"]
    values = {int(k): float(v) for k, v in data["values"].items()}
    out = (net, values)
    if with_config:
        out += (data["config"], data.get("history", []))
    if with_king:
        king = None
        if data.get("king"):
            from tonkai.king_safety_model import KingSafetyModel, KingSafetyNet
            king_net = KingSafetyNet(data["king"]["hidden"])
            king_net.load_state_dict(data["king"]["state_dict"])
            king = KingSafetyModel(king_net)
        out += (king,)
    return out


def load_players(spec, settings=()):
    """'x.pt' -> one Claudio; 'white.pt+black.pt' -> a white and a black specialist.
    Returns [(net, values, ClaudioConfig, king model)], one or two entries."""
    from claudio_agent import ClaudioConfig
    out = []
    for path in str(spec).split("+"):
        net, values, config, _, king = load_claudio(path.strip(), with_config=True, with_king=True)
        out.append((net, values, ClaudioConfig.with_settings(list(settings), base=config), king))
    return out


def make_player(loaded, rng=None):
    """A Claudio, or a ColourPair when two checkpoints were loaded."""
    import numpy as np
    from claudio_agent import Claudio, ColourPair
    rng = rng if rng is not None else np.random.default_rng()
    made = []
    for net, values, cfg, king in loaded:
        if hasattr(net, "_organic_payload"):
            from martinus_organic.agent import OrganicClaudio
            made.append(OrganicClaudio(net, values, cfg, net._organic_payload, rng=rng, king_model=king))
        else:
            made.append(Claudio(net, values, cfg, rng=rng, king_model=king))
    return made[0] if len(made) == 1 else ColourPair(made[0], made[1])
