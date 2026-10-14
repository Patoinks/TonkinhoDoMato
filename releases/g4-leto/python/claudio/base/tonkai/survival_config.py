"""Explicit V2 settings; legacy checkpoints retain their original training path."""
from dataclasses import dataclass

from .config import Config


@dataclass
class SurvivalConfig(Config):
    objective: str = "survival"
    memory_size: int = 64
    ensemble_size: int = 5
    future_horizons: tuple[int, ...] = (1, 2, 4, 8)
    sequence_length: int = 48
    burn_in: int = 16
    sequence_batch_size: int = 8
    survival_gamma: float = 0.985
    target_tau: float = 0.01
    future_weight: float = 0.2
    novelty_weight: float = 0.05
    curiosity_bonus: float = 0.025
    uncertainty_bonus: float = 0.04
    novelty_scale: float = 0.1
    bootstrap_probability: float = 0.7
    min_replay: int = 256
    updates_per_game: int = 2
    temperature: float = 0.08
    external_probability: float = 0.5
    historical_probability: float = 0.3
    population_size: int = 8
    snapshot_every: int = 100
    opponent_elo: int = 1600
    opponent_move_time: float = 0.02
    opponent_path: str = ""
    checkpoint_dir: str = "checkpoints/survival"
    checkpoint_every: int = 100
    log_every: int = 10
    num_workers: int = 4
    device: str = "cpu"

    def validate(self):
        if self.objective not in ("survival", "victory", "shaped"):
            raise ValueError("objective must be survival, victory or shaped")
        if not 0 < self.survival_gamma < 1:
            raise ValueError("survival_gamma must be between zero and one")
        if not 0 < self.target_tau <= 1:
            raise ValueError("target_tau must be between zero and one")
        if not 0 < self.bootstrap_probability <= 1:
            raise ValueError("bootstrap_probability must be between zero and one")
        if min(self.memory_size, self.ensemble_size, self.sequence_length,
               self.sequence_batch_size, self.population_size, self.snapshot_every,
               self.checkpoint_every, self.log_every, self.max_plies) < 1:
            raise ValueError("model sizes, intervals and max_plies must be positive")
        if self.burn_in < 0 or not self.future_horizons or min(self.future_horizons) < 1:
            raise ValueError("burn_in must be nonnegative and future horizons positive")
        if not 1 <= self.num_workers <= 16:
            # Actors mostly wait on the external engine, so more than one per core pays off.
            raise ValueError("one to sixteen independent sequence actors are supported")
        if min(self.external_probability, self.historical_probability) < 0 or (
            self.external_probability + self.historical_probability > 1
        ):
            raise ValueError("opponent mixture probabilities must sum to at most one")
        if not 0 <= self.curiosity_bonus <= 0.1 or not 0 <= self.uncertainty_bonus <= 0.1:
            raise ValueError("exploration bonuses must remain small (0 to 0.1)")
        if self.novelty_scale <= 0:
            raise ValueError("novelty_scale must be positive")
