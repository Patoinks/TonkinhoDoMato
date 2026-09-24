"""Opt-in V3 settings. Importing this module never changes a running V2 actor."""
from dataclasses import dataclass

from .survival_config import SurvivalConfig



MEASURE_PARTS = ("owner", "enemy_king", "enemy_box", "material", "simplify")


def measure_parts(text):
    """victory_measure -> {part: weight}, weights summing to 1."""
    parts = {}
    for item in str(text).split(","):
        name, _, weight = item.strip().partition(":")
        if name not in MEASURE_PARTS:
            raise ValueError(f"victory_measure part must be one of {MEASURE_PARTS}, not {name!r}")
        parts[name] = parts.get(name, 0.0) + (float(weight) if weight else 1.0)
    total = sum(parts.values())
    if total <= 0 or any(w < 0 for w in parts.values()):
        raise ValueError("victory_measure weights must be >= 0 with a positive sum")
    return {name: weight / total for name, weight in parts.items()}


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
    # The measure the shaped objective blends in, one part or a weighted mix
    # "part:weight,part:weight" (weights are normalized). Parts:
    #   owner      - the owner's victory measure (kings + pieces)
    #   enemy_king - 1 - KS of the enemy king (learned King Safety model)
    #   enemy_box  - 1 - (squares next to the enemy king we do not control) / 8
    #   material   - 0.5 + 0.5 tanh((own - enemy pieces) / 5), K/D values (else 1/3/3/5/9)
    #   simplify   - (32 - pieces on the board) / 27: 1 at 5 pieces, where the
    #                tablebase decides the game
    # Finished games, tablebase results and proven mates still override it.
    victory_measure: str = "owner"
    # Owner's rule: the network values an enemy piece exactly as it would the
    # same piece of its own on the mirrored square (seen from its owner's side),
    # so both sides are measured with one scale. False = separate side embedding.
    symmetric_piece_values: bool = False
    # Shaped objective: learn the measure as it will be one horizon later (the
    # longest future horizon, or the finished game's result when it ends
    # sooner), half of it bootstrapped from the target network there - the
    # measure then looks ahead instead of describing the current position.
    measure_lookahead: bool = False
    # One-move tactics (exact, rules only): for the best `tactic_candidates`
    # moves, the exchange result also counts what the opponent wins by force
    # with ONE move of its own (forks, double attacks) after our best reply.
    # 0 = captures only.
    tactic_candidates: int = 0
    # > 0: those candidates are searched instead over every move for this many
    # half-moves after ours (material and mate only, captures then settled
    # exactly): 4 = a 5-half-move calculation. Node budget per candidate.
    tactic_depth: int = 0
    tactic_node_limit: int = 2_000_000
    # Endgame technique. With at most `endgame_pieces` pieces on the board
    # (0 = off) the mate search looks `endgame_mate_moves` own moves deep with
    # its own node budget, whatever the king safety model says.
    endgame_pieces: int = 0
    endgame_mate_moves: int = 5
    endgame_node_limit: int = 1_000_000
    # Ahead by at least this much material (pawns, same values for both sides),
    # a move that ends the game in a draw (stalemate, repetition, 50 moves,
    # insufficient material, drawn tablebase) ranks under every other move
    # that does not allow mate. 0 = off.
    avoid_draws_ahead: float = 0.0
    # Owner's "fast win" objective (shaped only): in a won game every position
    # also earns win_speed_gamma ** (half-moves until the win), blended into the
    # value target with this weight. Losses, draws: 0; unfinished: unchanged.
    win_speed_weight: float = 0.0
    win_speed_gamma: float = 0.99
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
    # Mate searches only where the learned King Safety model (KS, 0 unsafe ..
    # 1 safe) sees a king in danger; 0 = always search. Attack: the enemy KS
    # after some candidate move is below mate_attack_ks. Defence: mate in 1 is
    # always checked; the deeper search runs when the own KS after some
    # candidate move is below mate_defense_ks.
    mate_attack_ks: float = 0.0
    mate_defense_ks: float = 0.0
    # Owner's rule: never go back to a position already seen since the last
    # capture/pawn move while another move exists (repetitions are broken).
    break_repetitions: bool = False
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
    # Owner's anti-stalling rule: this many half-moves in a row without a
    # capture end the game as a LOSS FOR BOTH sides (0 = off; 30 = 15 moves each).
    stall_limit: int = 0
    # Endgame tablebases (Syzygy 3-4-5, rules-only truth). With a path, a move
    # entering a tablebase ending is valued at its exact result; with
    # tablebase_adjudicate, training/match games end there with that result.
    syzygy_path: str = ""
    tablebase_adjudicate: bool = False
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
        if self.stall_limit < 0:
            raise ValueError("stall_limit must be >= 0")
        if self.draw_value is not None and not 0 <= self.draw_value <= 1:
            raise ValueError("draw_value must be between 0 and 1")
        if self.exchange_depth < 0 or self.exchange_node_limit < 1 or not 0 <= self.trade_explore <= 1:
            raise ValueError("exchange_depth >= 0, exchange_node_limit positive, trade_explore in [0, 1]")
        if not 0 <= self.threat_explore <= 1 or self.threat_moves < 1 or self.mate_labels < 0:
            raise ValueError("threat_explore must be in [0, 1], threat_moves positive, mate_labels >= 0")
        if (self.tactic_candidates < 0 or self.tactic_depth < 0 or self.tactic_node_limit < 1
                or self.endgame_pieces < 0 or self.endgame_mate_moves < 1
                or self.endgame_node_limit < 1 or self.avoid_draws_ahead < 0):
            raise ValueError("tactic/endgame settings must be non-negative (mate moves and node limit >= 1)")
        if not (0.0 <= self.win_speed_weight <= 1.0 and 0.0 < self.win_speed_gamma <= 1.0):
            raise ValueError("win_speed_weight must be in [0, 1] and win_speed_gamma in (0, 1]")
        parts = measure_parts(self.victory_measure)          # raises on an unknown part
        if "enemy_king" in parts and not self.king_safety_model:
            raise ValueError("victory_measure enemy_king needs a king_safety_model")
        if self.mate_search_moves < 0 or self.mate_defense_moves < 0 or self.mate_node_limit < 1:
            raise ValueError("mate_search_moves/mate_defense_moves must be >= 0 and mate_node_limit positive")
        if not (0.0 <= self.mate_attack_ks <= 1.0 and 0.0 <= self.mate_defense_ks <= 1.0):
            raise ValueError("mate_attack_ks/mate_defense_ks must be in [0, 1]")
