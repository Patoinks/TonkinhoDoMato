"""V3 opponent mixture: TonkAI itself, TinyOpponent and external UCI engines.

Every opponent only returns moves; none of them ever labels a position. The
learner draws each game's kind with its own RNG, so the mixture is part of the
campaign's reproducible state.

  self      TonkAI with the same weights, its own memory and exploration style
  tiny      the project's C++ TinyOpponent, in process (no engine to wait on)
  fairy     Fairy-Stockfish at a nominal Elo (strength-limited UCI)
  external  the main UCI opponent (--opponent-path / --opponent-elo)
"""
from __future__ import annotations

import chess
import numpy as np

KINDS = ("self", "tiny", "fairy", "external")


def mixture(cfg) -> dict[str, float]:
    """Kind -> probability; the main external opponent takes the remainder."""
    mix = {"self": cfg.mix_self, "tiny": cfg.mix_tiny, "fairy": cfg.mix_fairy}
    mix["external"] = max(0.0, 1.0 - sum(mix.values()))
    return mix


def uses_mixture(cfg) -> bool:
    return mixture(cfg)["external"] < 1.0


def choose_kind(cfg, rng) -> str:
    mix = mixture(cfg)
    kinds = [kind for kind in KINDS if mix[kind] > 0]
    weights = np.array([mix[kind] for kind in kinds])
    return kinds[int(rng.choice(len(kinds), p=weights / weights.sum()))]


class TinyOpponentAdapter:
    """TinyOpponent defaults (depth 2, 1000 nodes, randomness .35), seeded per game."""

    def __init__(self, seed):
        import tiny_opponent
        self.engine = tiny_opponent.TinyOpponent(seed=int(seed) % (2 ** 63))
        self.description = "TinyOpponent depth 2"

    def reset(self):
        pass

    def choose(self, board, rng=None):
        return chess.Move.from_uci(self.engine.choose_move(board.fen()))


class CognitiveOpponent:
    """TonkAI playing the other side: same weights, separate recurrent state.

    It observes every real ply in order (never skipping), exactly as the
    learning side does, and explores with its own ensemble head and temperature.
    """

    description = "TonkAI (self-play)"

    def __init__(self, agent, cfg, rng):
        self.agent, self.cfg, self.rng = agent, cfg, rng
        self.reset()

    def reset(self):
        self.agent.reset()
        self.board = None
        self.head = int(self.rng.integers(self.cfg.ensemble_size))
        self.temperature = self.cfg.temperature * float(self.rng.uniform(0.7, 1.3))

    def choose(self, board, rng=None):
        colour = board.turn
        if (self.board is None or len(self.board.move_stack) > len(board.move_stack)
                or self.board.move_stack != board.move_stack[:len(self.board.move_stack)]):
            self.agent.reset()
            self.board = board.root()
            self.agent.observe(self.board, colour)
        for move in board.move_stack[len(self.board.move_stack):]:
            self.board.push(move)
            self.agent.observe(self.board, colour)
        choice = self.agent.choose(self.board, colour, self.rng, temperature=self.temperature,
                                   explore=True, head=self.head)
        return choice.move
