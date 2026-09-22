"""Opt-in V3 settings. Importing this module never changes a running V2 actor."""
from dataclasses import dataclass

from .survival_config import SurvivalConfig


@dataclass
class CognitiveConfig(SurvivalConfig):
    checkpoint_dir: str = "checkpoints/cognitive"
    external_probability: float = 1.0
    historical_probability: float = 0.0
    archive_capacity: int = 512
    protected_episodes: int = 24
    memory_neighbors: int = 6
    memory_similarity: float = 0.8
    memory_weight: float = 0.12
    information_bonus: float = 0.02
    calibration_bins: int = 12
    calibration_min_samples: int = 8
    planning_candidates: int = 3
    planning_replies: int = 3
    planning_nodes: int = 96
    planning_uncertainty: float = 0.10
    planning_gap: float = 0.05
    robust_weight: float = 0.5
    skill_capacity: int = 96
    skill_horizon: int = 5
    skill_min_support: int = 2
    skill_bonus: float = 0.02
    experiments_every: int = 20
    experiment_plies: int = 8
    generalization_every: int = 20
    protected_replay_every: int = 10
    piece_safety_weight: float = 0.08
    piece_exchange_tolerance: float = 0.5
    piece_exchange_depth: int = 6
    king_safety_weight: float = 0.06
    capture_weight: float = 0.08
    capture_scale: float = 3.0

    def validate(self):
        super().validate()
        for name in ("archive_capacity", "protected_episodes", "memory_neighbors", "calibration_bins",
                     "calibration_min_samples", "planning_candidates", "planning_replies", "planning_nodes",
                     "skill_capacity", "skill_horizon", "skill_min_support", "experiments_every",
                     "experiment_plies", "generalization_every", "protected_replay_every", "piece_exchange_depth"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("memory_similarity", "memory_weight", "information_bonus", "planning_uncertainty",
                     "planning_gap", "robust_weight", "skill_bonus", "piece_safety_weight", "piece_exchange_tolerance",
                     "king_safety_weight", "capture_weight"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if self.external_probability != 1 or self.historical_probability != 0:
            raise ValueError("V3 next-opponent training preserves external-only play")
        if not 0 < self.capture_scale < float("inf"):
            raise ValueError("capture_scale must be finite and positive")
