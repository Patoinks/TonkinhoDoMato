"""Every knob in one place. Checkpoints store a copy of this."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- model -----------------------------------------------------------
    latent_size: int = 128
    hidden_size: int = 128
    piece_hidden: int = 32
    head_hidden: int = 32
    type_embed: int = 16
    square_embed: int = 16
    side_embed: int = 8

    # --- optimisation ----------------------------------------------------
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 512
    replay_capacity: int = 200_000
    predict_capacity: int = 20_000
    min_replay: int = 2_000           # no gradient steps before this many samples
    updates_per_game: int = 1
    samples_per_game: int = 2048    # 0 uses legacy batch_size per update
    grad_clip: float = 1.0

    # --- losses ----------------------------------------------------------
    value_weight: float = 1.0
    predict_weight: float = 1.0
    piece_value_weight: float = 0.1
    pieces_per_position: int = 1      # counterfactual probes per sampled position

    # --- behaviour -------------------------------------------------------
    temperature: float = 1.0
    draw_weight: float = 0.1
    max_plies: int = 300              # hard cap, scored as a draw

    # --- opponent --------------------------------------------------------
    opponent_depth: int = 2
    opponent_nodes: int = 1000
    opponent_randomness: float = 0.35  # "moderate"
    opponent_quiescence: bool = True

    # --- predict / theory ------------------------------------------------
    max_predict_candidates: int = 48
    surprise_threshold: float = 3.0
    theory_capacity: int = 10_000

    # --- logging ---------------------------------------------------------
    log_every: int = 100              # games
    checkpoint_every: int = 1000      # games
    checkpoint_dir: str = "checkpoints"
    log_sacrifices: bool = True

    # --- runtime ---------------------------------------------------------
    device: str = "auto"
    seed: int = 42
    num_workers: int = 1
    tonkai_plays_white_prob: float = 0.5

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f.name for f in dataclasses.fields(cls)}
        values = {k: v for k, v in d.items() if k in known}
        if "samples_per_game" not in d:
            values["samples_per_game"] = 0  # preserve old checkpoint schedules
        return cls(**values)

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"


def add_config_args(parser) -> None:
    """Expose every scalar field as --flag so experiments need no code edits."""
    for f in dataclasses.fields(Config):
        if f.type in ("bool", bool):
            parser.add_argument(
                f"--{f.name}",
                type=lambda s: s.lower() in ("1", "true", "yes", "on"),
                default=None,
            )
        elif f.type in ("int", int):
            parser.add_argument(f"--{f.name}", type=int, default=None)
        elif f.type in ("float", float):
            parser.add_argument(f"--{f.name}", type=float, default=None)
        elif f.type in ("str", str):
            parser.add_argument(f"--{f.name}", type=str, default=None)


def config_from_args(args, base: Config | None = None) -> Config:
    cfg = base or Config()
    for f in dataclasses.fields(Config):
        v = getattr(args, f.name, None)
        if v is not None:
            setattr(cfg, f.name, v)
    return cfg
