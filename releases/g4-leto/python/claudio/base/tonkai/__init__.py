"""TonkAI - a chess network that learns from survival, not from centipawns."""

from __future__ import annotations

import os
import sys

__version__ = "0.1.0"

_HINT = (
    "TinyOpponent is not built. From the project root run:\n"
    "    python setup.py build_ext --inplace"
)


def make_opponent(cfg, seed: int | None = None):
    """Create the ONE TinyOpponent instance a worker reuses for every move."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import tiny_opponent
    except ImportError as exc:  # pragma: no cover - environment problem
        raise ImportError(_HINT) from exc
    return tiny_opponent.TinyOpponent(
        depth=cfg.opponent_depth,
        max_nodes=cfg.opponent_nodes,
        randomness=cfg.opponent_randomness,
        seed=cfg.seed if seed is None else seed,
        quiescence=cfg.opponent_quiescence,
    )


def build(cfg, device: str | None = None):
    """(net, agent, opponent, worker, trainer, metrics) wired together."""
    import numpy as np
    import torch

    from .agent import GameWorker, TonkAIAgent
    from .metrics import Metrics
    from .model import TonkAINet
    from .trainer import Trainer

    device = device or cfg.resolved_device()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    net = TonkAINet(cfg).to(device)
    metrics = Metrics()
    import copy
    inference_device = device if cfg.inference_device == "learner" else cfg.inference_device
    # Parallel workers own their CPU models; no parent replica is necessary.
    inference_net = (copy.deepcopy(net).to(inference_device)
                     if cfg.num_workers == 1 and inference_device != device else net)
    agent = TonkAIAgent(inference_net, cfg,
                       inference_device if cfg.num_workers == 1 else device)
    opponent = make_opponent(cfg)
    worker = GameWorker(agent, opponent, cfg, metrics, rng)
    trainer = Trainer(net, cfg, device, metrics, rng)
    trainer.inference_net = inference_net
    return net, agent, opponent, worker, trainer, metrics
