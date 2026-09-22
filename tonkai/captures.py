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

MAX_TARGET = 255
OUTCOMES = ("win", "draw", "loss", "resigned", "unfinished")


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


class CaptureStats:
    """Kills/deaths per piece type, and how games end after a hot capture."""

    def __init__(self, hot_kills: int = 2):
        if hot_kills < 1:
            raise ValueError("hot_kills must be positive")
        self.hot_kills = hot_kills
        self.games = 0
        self.kills = {"self": Counter(), "opponent": Counter()}
        self.deaths = {"self": Counter(), "opponent": Counter()}
        self.hot_captures = Counter()                  # victim type -> count
        self.after_hot = Counter()                      # outcome -> games
        self.without_hot = Counter()

    def add(self, moves, self_color: bool, outcome: str, initial_fen: str | None = None):
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}")
        _, _, events, _ = track(moves, initial_fen)
        self.games += 1
        hot = False
        for event in events:
            killer = "self" if event["capturer_color"] == self_color else "opponent"
            victim = "self" if event["victim_color"] == self_color else "opponent"
            self.kills[killer][PIECE_NAMES[event["capturer_type"] - 1]] += 1
            self.deaths[victim][PIECE_NAMES[event["victim_type"] - 1]] += 1
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
        return {"games": self.games, "hot_kills": self.hot_kills,
                "kd_self": table("self"), "kd_opponent": table("opponent"),
                "kd_pooled_vs_pawn": None if pooled is None else
                {PIECE_NAMES[kind - 1]: value for kind, value in pooled.items()},
                "hot_captures_by_victim": dict(self.hot_captures),
                "games_with_hot_capture": outcomes(self.after_hot),
                "games_without_hot_capture": outcomes(self.without_hot)}

    def state_dict(self) -> dict:
        return {"hot_kills": self.hot_kills, "games": self.games,
                "kills": {k: dict(v) for k, v in self.kills.items()},
                "deaths": {k: dict(v) for k, v in self.deaths.items()},
                "hot_captures": dict(self.hot_captures), "after_hot": dict(self.after_hot),
                "without_hot": dict(self.without_hot)}

    def load_state_dict(self, state: dict) -> None:
        self.games = state["games"]
        for side in ("self", "opponent"):
            self.kills[side].update(state["kills"][side])
            self.deaths[side].update(state["deaths"][side])
        self.hot_captures.update(state["hot_captures"])
        self.after_hot.update(state["after_hot"])
        self.without_hot.update(state["without_hot"])
