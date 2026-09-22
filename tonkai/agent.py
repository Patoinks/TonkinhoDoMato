"""TonkAI's decision making, and the self-play worker that produces games.

Two rules that this file exists to enforce:

  * TonkAI only ever chooses from `board.legal_moves`.
  * All candidate moves of a turn go through the encoder in ONE forward pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import chess
import numpy as np
import torch

from . import chess_env as env
from .config import Config
from .encoding import (
    INPUT_SIZE,
    PACKED_SIZE,
    PIECE_NAMES,
    encode_after_moves,
    encode_into,
    oriented_square,
    pack_into,
)
from .model import TonkAINet, probabilities, utility
from .theory_buffer import SacrificeEvent, TheoryEvent


@dataclass
class MoveChoice:
    moves: list              # every legal move, in evaluation order
    index: int               # the one that was picked
    probs: np.ndarray        # (N, 3) win/draw/loss for every legal move
    utilities: np.ndarray    # (N,)
    encoded: np.ndarray      # (N, 782) resulting positions

    @property
    def move(self) -> chess.Move:
        return self.moves[self.index]


@dataclass
class PredictInfo:
    moves: list
    probs: np.ndarray
    true_index: int
    true_prob: float
    surprise: float
    entropy: float
    top1: bool
    top3: bool
    encoded: np.ndarray


@dataclass
class GameRecord:
    packed: np.ndarray                     # (P, 99) positions to learn W/D/L from
    result: int
    reason: str
    plies: int
    predict_samples: list = field(default_factory=list)
    theory_events: list = field(default_factory=list)
    sacrifices: list = field(default_factory=list)
    tonkai_positions: int = 0


class TonkAIAgent:
    def __init__(self, net: TonkAINet, cfg: Config, device: str):
        self.net = net
        self.cfg = cfg
        self.device = device
        self._buf = np.empty((256, INPUT_SIZE), dtype=np.float32)
        self._one = np.empty(INPUT_SIZE, dtype=np.float32)

    # ------------------------------------------------------------ inference

    def _forward_positions(self, x: np.ndarray):
        t = torch.from_numpy(x).to(self.device, non_blocking=True)
        with torch.inference_mode():
            latent = self.net.encode(t)
            vlogits = self.net.value_logits(latent)
        return latent, vlogits

    def evaluate_moves(self, board: chess.Board, self_color: bool,
                       moves=None) -> MoveChoice | None:
        """Score every legal move in a single batch. Does not push anything."""
        if moves is None:
            moves = list(board.legal_moves)
        if not moves:
            return None
        x = encode_after_moves(board, moves, self_color, self._buf)
        _, vlogits = self._forward_positions(x)
        probs = probabilities(vlogits).float().cpu().numpy()
        u = probs[:, 0] + self.cfg.draw_weight * probs[:, 1] - probs[:, 2]
        return MoveChoice(moves, 0, probs, u, x)

    def choose(self, board: chess.Board, self_color: bool,
               temperature: float, rng: np.random.Generator) -> MoveChoice:
        mc = self.evaluate_moves(board, self_color)
        if mc is None:
            raise RuntimeError("choose() called with no legal moves")
        u = mc.utilities
        if temperature <= 0.0:
            idx = int(np.argmax(u))
        else:
            z = (u - u.max()) / temperature
            w = np.exp(z)
            s = w.sum()
            idx = int(rng.choice(len(u), p=w / s)) if s > 0 else int(np.argmax(u))
        mc.index = idx
        return mc

    def predict_enemy(self, board: chess.Board, self_color: bool,
                      moves=None) -> tuple[list, np.ndarray, np.ndarray]:
        """P(each legal opponent reply), in one forward pass."""
        if moves is None:
            moves = list(board.legal_moves)
        x = encode_after_moves(board, moves, self_color, None)
        t = torch.from_numpy(x).to(self.device, non_blocking=True)
        with torch.inference_mode():
            logits = self.net.predict_logits(self.net.encode(t))
            probs = torch.softmax(logits, dim=0).float().cpu().numpy()
        return moves, probs, x

    def piece_values(self, board: chess.Board, self_color: bool):
        """TonkAI's own contextual value for every piece on the board.

        Returns a list of (square, piece, is_self, value).
        """
        encode_into(board, self_color, self._one)
        t = torch.from_numpy(self._one[None, :]).to(self.device)
        with torch.inference_mode():
            latent = self.net.encode(t)
            entries = []
            types, squares, sides = [], [], []
            for sq, piece in board.piece_map().items():
                is_self = piece.color == self_color
                entries.append((sq, piece, is_self))
                types.append(piece.piece_type - 1)
                squares.append(oriented_square(sq, self_color))
                sides.append(0 if is_self else 1)
            if not entries:
                return []
            m = len(entries)
            vals = self.net.piece_values(
                latent.expand(m, -1),
                torch.tensor(types, device=self.device),
                torch.tensor(squares, device=self.device),
                torch.tensor(sides, device=self.device),
            )
            vals = vals.float().cpu().numpy()
        return [(sq, p, s, float(v)) for (sq, p, s), v in zip(entries, vals)]

    def value_of(self, board: chess.Board, self_color: bool):
        """(probs[3], utility) for a single position."""
        encode_into(board, self_color, self._one)
        _, vlogits = self._forward_positions(self._one[None, :])
        p = probabilities(vlogits)[0].float().cpu().numpy()
        u = float(p[0] + self.cfg.draw_weight * p[1] - p[2])
        return p, u


class GameWorker:
    """Plays one game at a time and hands back everything worth keeping.

    `num_workers > 1` will eventually mean several of these feeding one
    replay buffer; the class is already self-contained for that.
    """

    def __init__(self, agent: TonkAIAgent, opponent, cfg: Config,
                 metrics, rng: np.random.Generator):
        self.agent = agent
        self.opponent = opponent
        self.cfg = cfg
        self.metrics = metrics
        self.rng = rng

    def play_game(self, self_color: bool | None = None,
                  temperature: float | None = None,
                  collect: bool = True) -> GameRecord:
        cfg = self.cfg
        m = self.metrics
        if self_color is None:
            self_color = (self.rng.random() < cfg.tonkai_plays_white_prob)
        temp = cfg.temperature if temperature is None else temperature

        board = chess.Board()
        packed_positions: list[np.ndarray] = []
        predict_samples: list[tuple[np.ndarray, int]] = []
        theory_events: list[TheoryEvent] = []
        sacrifices: list[SacrificeEvent] = []
        n_positions = 0

        while True:
            with m.timer("env"):
                result, reason = env.terminal(board, self_color, cfg.max_plies)
            if result is not None:
                break

            if board.turn == self_color:
                n_positions += self._tonkai_turn(
                    board, self_color, temp, packed_positions, sacrifices, collect
                )
            else:
                n_positions += self._opponent_turn(
                    board, self_color, packed_positions, predict_samples,
                    theory_events, collect
                )

        m.add_game(result, len(board.move_stack), reason)
        m.tonkai_positions += n_positions

        packed = (np.stack(packed_positions) if packed_positions
                  else np.zeros((0, PACKED_SIZE), dtype=np.uint8))
        return GameRecord(
            packed=packed,
            result=result,
            reason=reason,
            plies=len(board.move_stack),
            predict_samples=predict_samples,
            theory_events=theory_events,
            sacrifices=sacrifices,
            tonkai_positions=n_positions,
        )

    # ------------------------------------------------------------- one turn

    def _tonkai_turn(self, board, self_color, temp, packed_positions,
                     sacrifices, collect) -> int:
        cfg = self.cfg
        m = self.metrics

        with m.timer("env"):
            moves = list(board.legal_moves)
            x = encode_after_moves(board, moves, self_color, self.agent._buf)

        with m.timer("inference"):
            t = torch.from_numpy(x).to(self.agent.device, non_blocking=True)
            with torch.inference_mode():
                vlogits = self.agent.net.value_logits(self.agent.net.encode(t))
                probs = torch.softmax(vlogits, dim=-1).float().cpu().numpy()

        u = probs[:, 0] + cfg.draw_weight * probs[:, 1] - probs[:, 2]
        if temp <= 0.0:
            idx = int(np.argmax(u))
        else:
            z = np.exp((u - u.max()) / temp)
            s = z.sum()
            idx = int(self.rng.choice(len(u), p=z / s)) if s > 0 else int(np.argmax(u))

        move = moves[idx]
        m.legal_moves.add(len(moves))

        if cfg.log_sacrifices and board.is_capture(move):
            ev = self._sacrifice_check(board, move, self_color, u[idx], probs[idx])
            if ev is not None:
                sacrifices.append(ev)
                m.sacrifices += 1

        if collect:
            row = np.empty(PACKED_SIZE, dtype=np.uint8)
            pack_into(x[idx], row)
            packed_positions.append(row)

        with m.timer("env"):
            board.push(move)
        return len(moves)

    def _opponent_turn(self, board, self_color, packed_positions,
                       predict_samples, theory_events, collect) -> int:
        cfg = self.cfg
        m = self.metrics

        with m.timer("env"):
            enemy_moves = list(board.legal_moves)
            ex = encode_after_moves(board, enemy_moves, self_color, None)
            fen = board.fen()

        with m.timer("inference"):
            t = torch.from_numpy(ex).to(self.agent.device, non_blocking=True)
            with torch.inference_mode():
                logits = self.agent.net.predict_logits(self.agent.net.encode(t))
                pprobs = torch.softmax(logits, dim=0).float().cpu().numpy()

        with m.timer("opponent"):
            uci = self.opponent.choose_move(fen)
        m.opponent_moves += 1

        with m.timer("env"):
            move = env.apply_opponent_move(board, uci)

        true_index = enemy_moves.index(move)
        true_prob = float(max(pprobs[true_index], 1e-12))
        surprise = -math.log(true_prob)
        entropy = float(-np.sum(pprobs * np.log(np.maximum(pprobs, 1e-12))))
        order = np.argsort(-pprobs)
        rank = int(np.where(order == true_index)[0][0])

        m.enemy_moves.add(len(enemy_moves))
        m.true_move_prob.add(true_prob)
        m.surprise.add(surprise)
        m.entropy.add(entropy)
        m.top1.add(1.0 if rank == 0 else 0.0)
        m.top3.add(1.0 if rank < 3 else 0.0)

        if collect:
            packed = np.empty((len(enemy_moves), PACKED_SIZE), dtype=np.uint8)
            for i in range(len(enemy_moves)):
                pack_into(ex[i], packed[i])
            predict_samples.append((packed, true_index))
            # the position the opponent just created is already encoded: keep
            # it for the value head, free of charge
            packed_positions.append(packed[true_index].copy())

            if surprise > cfg.surprise_threshold:
                top = [(enemy_moves[i].uci(), float(pprobs[i])) for i in order[:5]]
                theory_events.append(
                    TheoryEvent(
                        fen=fen,
                        ply=len(board.move_stack) - 1,
                        tonkai_move=(board.move_stack[-2].uci()
                                     if len(board.move_stack) >= 2 else ""),
                        actual_move=move.uci(),
                        actual_prob=true_prob,
                        surprise=surprise,
                        entropy=entropy,
                        top_predictions=top,
                    )
                )
                m.theory_events += 1

        return len(enemy_moves)

    # -------------------------------------------------------- sacrifice log

    def _sacrifice_check(self, board, move, self_color, util_after, probs_after):
        """Did TonkAI trade away a piece it valued more than the one it took?"""
        mover = board.piece_at(move.from_square)
        if board.is_en_passant(move):
            cap_square = move.to_square + (-8 if board.turn == chess.WHITE else 8)
        else:
            cap_square = move.to_square
        victim = board.piece_at(cap_square)
        if mover is None or victim is None:
            return None

        agent = self.agent
        encode_into(board, self_color, agent._one)
        t = torch.from_numpy(agent._one[None, :]).to(agent.device)
        with torch.inference_mode():
            latent = agent.net.encode(t)
            vlogits = agent.net.value_logits(latent)
            p_before = torch.softmax(vlogits, dim=-1)[0].float().cpu().numpy()
            vals = agent.net.piece_values(
                latent.expand(2, -1),
                torch.tensor([mover.piece_type - 1, victim.piece_type - 1],
                             device=agent.device),
                torch.tensor(
                    [oriented_square(move.from_square, self_color),
                     oriented_square(cap_square, self_color)],
                    device=agent.device),
                torch.tensor([0, 1], device=agent.device),
            ).float().cpu().numpy()

        self_value, enemy_value = float(vals[0]), float(vals[1])
        if self_value <= enemy_value:
            return None

        u_before = float(p_before[0] + self.cfg.draw_weight * p_before[1] - p_before[2])
        return SacrificeEvent(
            fen=board.fen(),
            ply=len(board.move_stack),
            move=move.uci(),
            self_piece=PIECE_NAMES[mover.piece_type - 1],
            enemy_piece=PIECE_NAMES[victim.piece_type - 1],
            self_piece_value=self_value,
            enemy_piece_value=enemy_value,
            utility_before=u_before,
            utility_after=float(util_after),
            win_before=float(p_before[0]),
            win_after=float(probs_after[0]),
        )
