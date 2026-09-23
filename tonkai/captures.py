"""Who captured whom, piece by piece, in a finished game.

Two uses, neither of which ever rewards a capture or touches a move choice:

* **Auxiliary training target** for the V3 piece head: how many captures
  THIS piece, standing on THIS square, goes on to make in the rest of the game.
  It is what actually happened - contextual and learned, never a per-type table.
* **Measurement**: kills/deaths per piece type, and the owner's hypothesis put to
  the test - when TonkAI captures an enemy piece that had already been killing
  in this game (a "hot" piece), how does that game end?

Pieces keep their identity when they move, castle or promote, so a promoted
pawn's later captures still belong to the pawn that walked there.
"""

from __future__ import annotations

from collections import Counter

import chess
import numpy as np

from .encoding import OFF_STM, PIECE_NAMES, unpack_many
from . import fast_rules

MAX_TARGET = 255
OUTCOMES = ("win", "draw", "loss", "resigned", "unfinished")
# Owner's rule for a piece's measured value: equal parts of the three measures,
# each pooled over both sides and taken relative to a pawn.
VALUE_MEASURES = ("kd", "mobility", "threats")


def _captured_square(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        return move.to_square + (-8 if board.turn == chess.WHITE else 8)
    return move.to_square


def track(moves, initial_fen: str | None = None):
    """Replay a game keeping each piece's identity.

    Returns ``(snapshots, capturer, events, board)``:
      * ``snapshots[k]``: {square: piece id} for the position BEFORE move k
        (k = 0..len(moves)), i.e. aligned with an episode's packed states
      * ``capturer[k]``: id of the piece that captured on move k, or None
      * ``events``: one dict per capture, including how many captures the
        victim itself had made earlier in the game
    """
    board = chess.Board(initial_fen or chess.STARTING_FEN)
    ident = {sq: i for i, sq in enumerate(sorted(board.piece_map()))}
    snapshots = [dict(ident)]
    capturer: list[int | None] = []
    events: list[dict] = []
    kills: Counter = Counter()
    for move in moves:
        if isinstance(move, str):
            move = chess.Move.from_uci(move)
        piece = board.piece_at(move.from_square)
        if piece is None or move not in board.legal_moves:
            raise ValueError(f"illegal move {move.uci()} in {board.fen()}")
        mover = ident.pop(move.from_square)
        by = None
        if board.is_capture(move):
            square = _captured_square(board, move)
            victim = ident.pop(square)
            victim_piece = board.piece_at(square)
            events.append({"ply": len(board.move_stack), "capturer": mover,
                           "capturer_type": piece.piece_type, "capturer_color": piece.color,
                           "victim": victim, "victim_type": victim_piece.piece_type,
                           "victim_color": victim_piece.color, "victim_kills": kills[victim]})
            kills[mover] += 1
            by = mover
        if board.is_castling(move):
            rank = 0 if piece.color == chess.WHITE else 7
            if board.is_kingside_castling(move):
                king_to, rook_from, rook_to = 6, 7, 5
            else:
                king_to, rook_from, rook_to = 2, 0, 3
            rook = ident.pop(chess.square(rook_from, rank))
            ident[chess.square(rook_to, rank)] = rook
            ident[chess.square(king_to, rank)] = mover
        else:
            ident[move.to_square] = mover
        board.push(move)
        capturer.append(by)
        snapshots.append(dict(ident))
    return snapshots, capturer, events, board


def activity(moves, snapshots, initial_fen: str | None = None):
    """Per-type activity sums for both colours, from the rules only.

    Sampled in every position for the side to move, kings excluded:
      * mobility: legal destination squares of a piece, counted only once that
        piece has made its first move (an unmoved piece is not sampled)
      * threats: enemy pieces (not the king) it attacks that are defended

    Returns ``{colour: {"mobility": Counter, "mobility_n": Counter,
    "threats": Counter, "threats_n": Counter}}`` keyed by piece type.
    """
    board = chess.Board(initial_fen or chess.STARTING_FEN)
    out = {colour: {key: Counter() for key in ("mobility", "mobility_n", "threats", "threats_n")}
           for colour in (chess.WHITE, chess.BLACK)}
    moved: set[int] = set()
    for k, move in enumerate(moves):
        if isinstance(move, str):
            move = chess.Move.from_uci(move)
        turn, rows = board.turn, out[board.turn]
        destinations = Counter(m.from_square for m in board.legal_moves)
        enemy = board.occupied_co[not turn]
        own = board.occupied_co[turn] & ~board.kings
        for square in chess.SquareSet(own):
            kind = board.piece_type_at(square)
            if snapshots[k][square] in moved:
                rows["mobility"][kind] += destinations[square]
                rows["mobility_n"][kind] += 1
            threats = 0
            for target in chess.SquareSet(board.attacks_mask(square) & enemy & ~board.kings):
                if board.is_attacked_by(not turn, target):
                    threats += 1
            rows["threats"][kind] += threats
            rows["threats_n"][kind] += 1
        moved.add(snapshots[k][move.from_square])
        board.push(move)
    return out


def capture_targets(snapshots, capturer, owner_color: bool) -> np.ndarray:
    """(T, 64) uint8: future captures of the piece on each square, per state.

    Squares are in TonkAI's orientation (flipped when it plays black), the
    same convention as the encoding and the piece head.
    """
    total = len(snapshots)
    out = np.zeros((total, 64), dtype=np.uint8)
    future: Counter = Counter()
    for k in range(total - 1, -1, -1):
        # `future` now holds captures made by moves k, k+1, ... (after state k)
        for square, pid in snapshots[k].items():
            count = future.get(pid, 0)
            if count:
                out[k, square if owner_color else square ^ 56] = min(count, MAX_TARGET)
        if k > 0 and capturer[k - 1] is not None:
            future[capturer[k - 1]] += 1
    return out


def episode_owner_color(episode) -> bool:
    board = chess.Board(episode.initial_fen or chess.STARTING_FEN)
    own_turn = unpack_many(episode.packed[:1])[0, OFF_STM] > .5
    return board.turn if own_turn else not board.turn


def episode_capture_targets(episode) -> np.ndarray | None:
    """Targets aligned with ``episode.packed``; None when actions are missing."""
    if not episode.actions or len(episode.actions) != len(episode.packed) - 1:
        return None
    snapshots, capturer, _, _ = track(episode.actions, episode.initial_fen)
    return capture_targets(snapshots, capturer, episode_owner_color(episode))


def parse_weights(text):
    """'fairy=0.35,external=0.65' -> {'fairy': .35, 'external': .65}, normalised; '' -> None."""
    if not text:
        return None
    weights = {}
    for part in str(text).split(","):
        kind, _, value = part.partition("=")
        weights[kind.strip()] = float(value)
    total = sum(weights.values())
    if not weights or total <= 0 or min(weights.values()) < 0:
        raise ValueError("value opponent weights must be non-negative and sum above zero")
    return {kind: value / total for kind, value in weights.items()}


MEASURES = ("mobility", "mobility_n", "threats", "threats_n")


class CaptureStats:
    """Kills/deaths per piece type, and how games end after a hot capture.

    With ``opponent_weights`` (owner's rule, e.g. {'fairy': .35, 'external': .65})
    the piece values take the opponent side ONLY from those opponent kinds, in
    those shares, per game; TonkAI's own pieces still count from every game.
    """

    def __init__(self, hot_kills: int = 2, opponent_weights=None):
        if hot_kills < 1:
            raise ValueError("hot_kills must be positive")
        self.hot_kills = hot_kills
        self.opponent_weights = opponent_weights
        # opponent kind -> the opponent side's games, kills, deaths and activity
        self.by_kind = {}
        self.games = 0
        self.kills = {"self": Counter(), "opponent": Counter()}
        self.deaths = {"self": Counter(), "opponent": Counter()}
        self.hot_captures = Counter()                  # victim type -> count
        self.after_hot = Counter()                      # outcome -> games
        self.without_hot = Counter()
        # side -> measure -> piece name -> sum / sample count (see ``activity``)
        self.activity = {side: {key: Counter() for key in ("mobility", "mobility_n", "threats", "threats_n")}
                         for side in ("self", "opponent")}

    def _kind(self, kind):
        return self.by_kind.setdefault(kind, {"games": 0, "kills": Counter(), "deaths": Counter(),
                                              "activity": {key: Counter() for key in MEASURES}})

    def add(self, moves, self_color: bool, outcome: str, initial_fen: str | None = None, opponent_kind=None):
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}")
        snapshots, _, events, _ = track(moves, initial_fen)
        if fast_rules.ENABLED:
            root = chess.Board(initial_fen or chess.STARTING_FEN)
            codes = [fast_rules.code(chess.Move.from_uci(move) if isinstance(move, str) else move) for move in moves]
            measurements = fast_rules.board(root).activity(codes)
        else:
            measurements = activity(moves, snapshots, initial_fen)
        self.games += 1
        mine = self._kind(opponent_kind) if opponent_kind else None
        if mine is not None:
            mine["games"] += 1
        for colour, rows in measurements.items():
            side = "self" if colour == self_color else "opponent"
            for key, counter in rows.items():
                for kind, value in counter.items():
                    self.activity[side][key][PIECE_NAMES[kind - 1]] += value
                    if side == "opponent" and mine is not None:
                        mine["activity"][key][PIECE_NAMES[kind - 1]] += value
        hot = False
        for event in events:
            killer = "self" if event["capturer_color"] == self_color else "opponent"
            victim = "self" if event["victim_color"] == self_color else "opponent"
            self.kills[killer][PIECE_NAMES[event["capturer_type"] - 1]] += 1
            self.deaths[victim][PIECE_NAMES[event["victim_type"] - 1]] += 1
            if mine is not None and killer == "opponent":
                mine["kills"][PIECE_NAMES[event["capturer_type"] - 1]] += 1
            if mine is not None and victim == "opponent":
                mine["deaths"][PIECE_NAMES[event["victim_type"] - 1]] += 1
            if killer == "self" and event["victim_kills"] >= self.hot_kills:
                hot = True
                self.hot_captures[PIECE_NAMES[event["victim_type"] - 1]] += 1
        (self.after_hot if hot else self.without_hot)[outcome] += 1

    def pooled_kd(self, min_deaths: int = 50) -> dict[int, float] | None:
        """Owner's rule: one K/D per piece type, pooling BOTH sides' pieces.

        Returned relative to a pawn (pawn = 1), keyed by chess piece type
        (1..5; kings never die, so they are excluded). None until every type
        has at least ``min_deaths`` recorded deaths.
        """
        pooled = {}
        for index, name in enumerate(PIECE_NAMES[:5], start=1):
            kills = self.kills["self"][name] + self.kills["opponent"][name]
            deaths = self.deaths["self"][name] + self.deaths["opponent"][name]
            if deaths < min_deaths:
                return None
            pooled[index] = kills / deaths
        pawn = pooled[chess.PAWN]
        if pawn <= 0:
            return None
        return {kind: kd / pawn for kind, kd in pooled.items()}

    def pooled_activity(self, measure: str, min_samples: int = 500) -> dict[int, float] | None:
        """Mean ``measure`` ("mobility" or "threats") per sample, both sides
        pooled, relative to a pawn. None until every type has enough samples
        or while the pawn's mean is zero."""
        means = {}
        for index, name in enumerate(PIECE_NAMES[:5], start=1):
            total = sum(self.activity[side][measure][name] for side in ("self", "opponent"))
            samples = sum(self.activity[side][measure + "_n"][name] for side in ("self", "opponent"))
            if samples < min_samples:
                return None
            means[index] = total / samples
        pawn = means[chess.PAWN]
        if pawn <= 0:
            return None
        return {kind: mean / pawn for kind, mean in means.items()}

    def _weighted_rate(self, pick, name):
        """(raw count, per-game rate) of one quantity: TonkAI's side over every
        game plus the weighted opponents' side, each per game of that kind."""
        raw = pick(self.kills["self"], self.deaths["self"], self.activity["self"])[name]
        rate = raw / max(1, self.games)
        for kind, weight in self.opponent_weights.items():
            row = self.by_kind.get(kind)
            if not row or not row["games"]:
                return None
            value = pick(row["kills"], row["deaths"], row["activity"])[name]
            raw += value
            rate += weight * value / row["games"]
        return raw, rate

    def weighted_kd(self, min_deaths: int = 50, min_games: int = 30):
        if not self.opponent_weights or any(self.by_kind.get(k, {}).get("games", 0) < min_games
                                            for k in self.opponent_weights):
            return None
        values = {}
        for index, name in enumerate(PIECE_NAMES[:5], start=1):
            kills = self._weighted_rate(lambda k, d, a: k, name)
            deaths = self._weighted_rate(lambda k, d, a: d, name)
            if kills is None or deaths is None or deaths[0] < min_deaths or deaths[1] <= 0:
                return None
            values[index] = kills[1] / deaths[1]
        pawn = values[chess.PAWN]
        return None if pawn <= 0 else {kind: kd / pawn for kind, kd in values.items()}

    def weighted_activity(self, measure: str, min_samples: int = 500, min_games: int = 30):
        if not self.opponent_weights or any(self.by_kind.get(k, {}).get("games", 0) < min_games
                                            for k in self.opponent_weights):
            return None
        means = {}
        for index, name in enumerate(PIECE_NAMES[:5], start=1):
            total = self._weighted_rate(lambda k, d, a: a[measure], name)
            samples = self._weighted_rate(lambda k, d, a: a[measure + "_n"], name)
            if total is None or samples is None or samples[0] < min_samples or samples[1] <= 0:
                return None
            means[index] = total[1] / samples[1]
        pawn = means[chess.PAWN]
        return None if pawn <= 0 else {kind: mean / pawn for kind, mean in means.items()}

    def measures(self) -> dict:
        # The weighted opponents' rule once there is enough of their data;
        # until then the plain two-side pooling keeps decisions supplied.
        weighted = {"kd": self.weighted_kd(), "mobility": self.weighted_activity("mobility"),
                    "threats": self.weighted_activity("threats")}
        if all(value is not None for value in weighted.values()):
            return weighted
        return {"kd": self.pooled_kd(), "mobility": self.pooled_activity("mobility"),
                "threats": self.pooled_activity("threats")}

    def value_source(self) -> str:
        weighted = [self.weighted_kd(), self.weighted_activity("mobility"), self.weighted_activity("threats")]
        return "weighted" if all(value is not None for value in weighted) else "pooled"

    def piece_values(self) -> dict[int, float] | None:
        """Owner's rule: 1/3 K/D + 1/3 mobility after the first move + 1/3
        threats on defended pieces, each relative to a pawn. None until all
        three measures exist."""
        measures = self.measures()
        if any(measures[name] is None for name in VALUE_MEASURES):
            return None
        return {kind: sum(measures[name][kind] for name in VALUE_MEASURES) / len(VALUE_MEASURES)
                for kind in range(1, 6)}

    def summary(self) -> dict:
        def table(side):
            rows = {}
            for name in PIECE_NAMES:
                k, d = self.kills[side][name], self.deaths[side][name]
                rows[name] = {"kills": k, "deaths": d, "kd": None if d == 0 else k / d,
                              "kills_per_game": k / max(1, self.games)}
            return rows

        def outcomes(counter):
            games = sum(counter.values())
            return {"games": games, **{o: counter[o] for o in OUTCOMES},
                    "score": None if not games else
                    (counter["win"] + .5 * counter["draw"]) / games}

        pooled = self.pooled_kd()

        def named(values):
            return None if values is None else {PIECE_NAMES[kind - 1]: value for kind, value in values.items()}

        measures = self.measures()
        return {"mobility_vs_pawn": named(measures["mobility"]), "threats_vs_pawn": named(measures["threats"]),
                "piece_values_vs_pawn": named(self.piece_values()), "games": self.games, "hot_kills": self.hot_kills,
                "kd_self": table("self"), "kd_opponent": table("opponent"),
                "kd_pooled_vs_pawn": None if pooled is None else
                {PIECE_NAMES[kind - 1]: value for kind, value in pooled.items()},
                "hot_captures_by_victim": dict(self.hot_captures),
                "games_with_hot_capture": outcomes(self.after_hot),
                "games_without_hot_capture": outcomes(self.without_hot),
                "value_source": self.value_source(), "value_opponents": self.opponent_weights,
                "value_opponent_games": {kind: row["games"] for kind, row in self.by_kind.items()}}

    def state_dict(self) -> dict:
        return {"hot_kills": self.hot_kills, "games": self.games,
                "kills": {k: dict(v) for k, v in self.kills.items()},
                "deaths": {k: dict(v) for k, v in self.deaths.items()},
                "hot_captures": dict(self.hot_captures), "after_hot": dict(self.after_hot),
                "without_hot": dict(self.without_hot),
                "activity": {side: {key: dict(value) for key, value in rows.items()}
                             for side, rows in self.activity.items()},
                "by_kind": {kind: {"games": row["games"], "kills": dict(row["kills"]), "deaths": dict(row["deaths"]),
                                   "activity": {key: dict(value) for key, value in row["activity"].items()}}
                            for kind, row in self.by_kind.items()}}

    def load_state_dict(self, state: dict) -> None:
        self.games = state["games"]
        for side in ("self", "opponent"):
            self.kills[side].update(state["kills"][side])
            self.deaths[side].update(state["deaths"][side])
        self.hot_captures.update(state["hot_captures"])
        self.after_hot.update(state["after_hot"])
        self.without_hot.update(state["without_hot"])
        for side, rows in state.get("activity", {}).items():   # older states lack it
            for key, value in rows.items():
                self.activity[side][key].update(value)
        for kind, row in state.get("by_kind", {}).items():     # older states lack it
            mine = self._kind(kind)
            mine["games"] += row["games"]
            mine["kills"].update(row["kills"])
            mine["deaths"].update(row["deaths"])
            for key, value in row["activity"].items():
                mine["activity"][key].update(value)
