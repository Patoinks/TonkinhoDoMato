"""Opt-in V3 settings. Importing this module never changes a running V2 actor."""
from dataclasses import dataclass

from .survival_config import SurvivalConfig


@dataclass
class CognitiveConfig(SurvivalConfig):
    compact_sequences: bool = False
    incremental_encoder: bool = False
    batch_search: bool = False
    inference_broker: bool = False
    cuda_graphs: bool = False
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
    # Hand-written attack/protection + mobility signal (v5 experiment). Off by
    # default: a hand-set preference, not something TonkAI discovered. With 0,
    # every checkpoint plays and learns as it was trained (organic).
    piece_activity_weight: float = 0.0
    piece_exchange_tolerance: float = 0.5
    piece_exchange_depth: int = 6
    king_safety_weight: float = 0.06
    # Auxiliary target, opt-in: the piece head also predicts how many captures
    # that piece goes on to make in the rest of the game. A prediction, never a
    # reward; it does not touch move selection. Zero keeps the model unchanged.
    capture_target_weight: float = 0.0
    capture_target_clip: float = 8.0
    # Measurement only: a "hot" enemy piece had made this many captures in the
    # game before TonkAI took it; the report shows how such games end.
    hot_piece_kills: int = 2
    # objective="shaped": share of the owner's victory measure in the value
    # target (the rest is survival). Unused by the other objectives.
    victory_weight: float = 0.5
    # Share of the pooled per-type K/D (both sides, pawn = 1) in the piece
    # values used for exchange/protection decisions; the rest is the network's
    # contextual value. Zero keeps the network values alone.
    kd_weight: float = 0.0
    # Opponent mixture (tonkai/opponents.py): share of games against TonkAI
    # itself, TinyOpponent and Fairy-Stockfish. The main UCI opponent
    # (opponent_path / opponent_elo) plays the rest. All zero = main only.
    mix_self: float = 0.0
    mix_tiny: float = 0.0
    mix_fairy: float = 0.0
    fairy_path: str = ""
    fairy_elo: int = 1000
    # Self-play rival: every this many games a frozen copy of TonkAI becomes the
    # self-play opponent (snapshots/ in the checkpoint directory). It never
    # learns; only the learning side's games are recorded. 0 = play the current
    # weights, as before.
    self_snapshot_every: int = 0
    # With snapshots: this many games right after each new snapshot are played
    # against it, as one series; the rest follow the mix_* shares. 0 = none.
    self_games_per_snapshot: int = 0
    # Piece values: take the OPPONENT side only from these opponent kinds, in
    # these shares, e.g. "fairy=0.35,external=0.65". Empty = every opponent.
    value_opponents: str = ""
    # Owner's rule: when TonkAI is ahead by at least this many pawns (its own
    # piece values), moves that draw by repetition are nearly worthless (0.05),
    # so it only repeats when everything else is terrible. 0 = off.
    avoid_repetition_margin: float = 0.0
    # v4 Emergent King Safety: a learned KS(position, king) file
    # (train_king_safety.py). Set, it replaces the hand-written king safety in
    # every decision with KS_self - KS_enemy. Empty = v3 behaviour.
    king_safety_model: str = ""
    king_safety_log_every: int = 10     # games between logged KS trajectories
    # Mate calculation (exact, rules-only search in tonkai_fast):
    # attack - play a shortest forced mate found within this many own moves;
    # defence - a move after which the opponent forces mate within this many
    # moves is worth almost nothing. 0 = off.
    mate_search_moves: int = 0
    mate_defense_moves: int = 0
    mate_node_limit: int = 20_000
    # Organic attack learning (off by default):
    # threat_explore - share of exploring decisions that TRY a move creating a
    #   mate threat (were the opponent to pass, mate within threat_moves), found
    #   by the rules alone; the game's result teaches whether it was good.
    # mate_labels - the shaped victory measure of a real position is the proven
    #   truth when the exact search finds a forced mate within this many moves
    #   (1 if TonkAI forces it, 0 if the opponent does).
    threat_explore: float = 0.0
    threat_moves: int = 1
    mate_labels: int = 0
    # Organic piece safety / trading (off by default), with TonkAI's measured
    # piece values: exchange_depth - the piece-safety term of each move is the
    # EXACT material result of the forced capture sequence (tonkai_fast
    # exchange search, this many plies) instead of the per-square estimate;
    # exchange_labels - the victory measure's piece part is that exact result;
    # trade_explore - share of exploring decisions that try a capture the
    # exact search does not lose, so the results teach which trades pay.
    exchange_depth: int = 0
    # The owner's value of a drawn ending in decisions (None: the objective's own,
    # e.g. 0.75 for shaped with victory_weight 0.5). Lower = draws are avoided.
    draw_value: float | None = None
    # Owner's rule: a game is drawn by repetition only once the position HAS
    # occurred three times - repeating is a choice TonkAI values at the draw
    # value (so, with a low draw value, only when its position is worse). Off:
    # the old rule, where merely being able to claim ended the game.
    strict_repetition: bool = False
    exchange_node_limit: int = 200_000
    exchange_labels: bool = False
    trade_explore: float = 0.0

    def validate(self):
        super().validate()
        for name in ("archive_capacity", "protected_episodes", "memory_neighbors", "calibration_bins",
                     "calibration_min_samples", "planning_candidates", "planning_replies", "planning_nodes",
                     "skill_capacity", "skill_horizon", "skill_min_support", "experiments_every",
                     "experiment_plies", "generalization_every", "protected_replay_every", "piece_exchange_depth",
                     "hot_piece_kills"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("memory_similarity", "memory_weight", "information_bonus", "planning_uncertainty",
                     "planning_gap", "robust_weight", "skill_bonus", "piece_safety_weight", "piece_exchange_tolerance",
                     "piece_activity_weight", "king_safety_weight",
                     "capture_target_weight", "victory_weight", "kd_weight",
                     "mix_self", "mix_tiny", "mix_fairy"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if self.external_probability != 1 or self.historical_probability != 0:
            raise ValueError("V3 mixes opponents with mix_*, not the V2 population probabilities")
        if self.mix_self + self.mix_tiny + self.mix_fairy > 1 + 1e-9:
            raise ValueError("mix_self + mix_tiny + mix_fairy must be at most one")
        if self.mix_fairy > 0 and (not self.fairy_path or self.fairy_elo < 1):
            raise ValueError("mix_fairy needs fairy_path and a positive fairy_elo")
        if not 0 < self.capture_target_clip < float("inf"):
            raise ValueError("capture_target_clip must be finite and positive")
        if self.draw_value is not None and not 0 <= self.draw_value <= 1:
            raise ValueError("draw_value must be between 0 and 1")
        if self.exchange_depth < 0 or self.exchange_node_limit < 1 or not 0 <= self.trade_explore <= 1:
            raise ValueError("exchange_depth >= 0, exchange_node_limit positive, trade_explore in [0, 1]")
        if not 0 <= self.threat_explore <= 1 or self.threat_moves < 1 or self.mate_labels < 0:
            raise ValueError("threat_explore must be in [0, 1], threat_moves positive, mate_labels >= 0")
        if self.mate_search_moves < 0 or self.mate_defense_moves < 0 or self.mate_node_limit < 1:
            raise ValueError("mate_search_moves/mate_defense_moves must be >= 0 and mate_node_limit positive")
