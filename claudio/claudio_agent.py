"""Claudio: a lean, organic TonkAI player.

What it knows comes from its own experience or from calculation with the
rules of chess, never from outside knowledge:

- intuition: the survival network, trained by playing (inherited from Timdgei);
- piece values: kills/deaths it measured in its own games (K/D), no 1/3/3/5/9;
- king safety (optional): the learned KS model, grounded only in checkmates;
- calculation: the exact capture sequence after each move, and a lazy check of
  the move it intends to play (4 half-moves of material, the opponent's forced
  mates), the way a player checks the move they are about to make;
- its own forced mates, found by the rules;
- a position it has already seen in this game is a draw if it goes back to it
  (the opponent can simply repeat), which is the rules, not a preference.

Nothing else: no opening book, no endgame tablebases, no hand-made bonuses.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
import time

import chess
import numpy as np
import torch

import claudio_fast
from tonkai import fast_rules
from tonkai.encoding import encode

# Known results sit outside the range of any judgement (about -0.3..1.3), so
# they always decide, and judgements are never clipped into ties.
WIN, LOSS, ALLOWS_MATE = 10.0, -10.0, -20.0


@dataclass
class ClaudioConfig:
    safety_weight: float = 0.2      # weight of the exact capture result in a move's score
    exchange_depth: int = 5         # capture sequence length (plies) after each move
    exchange_nodes: int = 200_000
    tactic_depth: int = 3           # half-moves after our move when checking the intended move
    tactic_top: int = 1             # best candidates checked together first
    root_pvs: bool = True           # ...each only as far as needed to know if it is the best
    tactic_rounds: int = 4          # how many times a new intended move may be checked (lazy)
    tactic_nodes: int = 2_000_000
    mate_defence: int = 2           # opponent's forced mate within n, checked on the intended move
    mate_attack: int = 3            # own forced mates within n
    mate_gate: float = 0.0          # >0: beyond 2 moves only when the learned KS of the enemy king is below this
    mate_nodes: int = 20_000
    pawn_reach: int = 1             # mate searches skip the attacker's pawns this far from the king
    king_weight: float = 0.0        # learned king safety: weight of KS(own) - KS(enemy)
    simplify_weight: float = 0.0    # ahead in its own material values: trades that lose nothing score higher
    simplify_margin: float = 1.0    # lead (own units) from which it simplifies; the full bonus 3 units later
    repeat_as_draw: bool = False    # going back to a position of this game scores as a draw
    memory: bool = True             # the network's memory of the game (GRU)
    value_scaling: bool = False     # keep the intuition's weight stable when the network's value scale drifts
    value_spread_ref: float = 0.0215    # spread of move values it is scaled to (the network before training)
    value_spread_decay: float = 0.98    # running estimate of its own spread, over decisions
    head: str = "mean"              # "mean" of the 5 value heads, or "random": one head per game
    temperature: float = 0.0        # 0 = always the best move
    draw_value: float = 0.4         # the owner's scoring: a draw is worth 0.4

    @classmethod
    def with_settings(cls, items, base=None):
        cfg, kinds = (cls(**base) if base else cls()), {f.name: f.type for f in fields(cls)}
        for item in items or []:
            key, _, raw = item.partition("=")
            if key not in kinds:
                raise ValueError(f"unknown Claudio setting {key}")
            kind = kinds[key]
            value = (raw.lower() in ("1", "true", "yes") if kind in ("bool", bool) else
                     int(raw) if kind in ("int", int) else float(raw) if kind in ("float", float) else raw)
            setattr(cfg, key, value)
        return cfg


def move_of(code: int) -> chess.Move:
    return chess.Move(code & 63, (code >> 6) & 63, (code >> 12) or None)


def native_board(position: chess.Board):
    """Claudio's own C++ board (claudio_fast) with the game's root and full history."""
    root = position.root()
    fast = claudio_fast.Board(root.pawns, root.knights, root.bishops, root.rooks, root.queens, root.kings,
                              root.occupied_co[chess.WHITE], root.occupied_co[chess.BLACK], root.turn,
                              root.castling_rights, -1 if root.ep_square is None else root.ep_square,
                              root.halfmove_clock, root.fullmove_number)
    fast.push_many([fast_rules.code(move) for move in position.move_stack])
    return fast


def _key(fast):
    return fast.fen().rsplit(" ", 2)[0]      # placement, turn, castling, en passant


class Claudio:
    """choose(board) -> chess.Move. Keeps its own boards and memory in step with
    the real game, so each decision only processes the new plies."""

    def __init__(self, net, values, cfg: ClaudioConfig | None = None, rng=None, king_model=None):
        self.net = net.eval().requires_grad_(False)
        self.cfg = cfg or ClaudioConfig()
        # [0, pawn, knight, bishop, rook, queen] in its own measured units
        self.values = [0.0] + [float(values[kind]) for kind in range(1, 6)]
        self.rng = rng or np.random.default_rng()
        self.king_model = king_model
        self.clock = None               # {section: seconds} when profiling
        self.spread = self.cfg.value_spread_ref   # its own running spread of move values (kept across games)
        self.reset()

    def reset(self):
        self.fast = None                # C++ board of the real game
        self.moves = []                 # the real game's moves already seen
        self.keys = []                  # position key after every real ply (index = ply)
        self.since = 0                  # first ply after the last capture or pawn move
        self.memory = self.net.initial_memory(1)
        self.colour = None
        self.root = None
        self.head = int(self.rng.integers(self.net.ensemble_size)) if self.cfg.head == "random" else None

    # ---- keeping in step with the game -------------------------------------
    @torch.inference_mode()
    def _catch_up(self, board):
        stack = board.move_stack
        if self.colour is not None:
            seen = len(self.moves)
            if (self.colour != board.turn or len(stack) < seen or (seen and stack[seen - 1] != self.moves[-1])
                    or (not seen and board.root().fen() != self.root)):
                self.reset()            # another game, or the other colour: start afresh
        rows = []
        if self.colour is None:
            root = board.root()
            self.colour, self.root, self.moves = board.turn, root.fen(), []
            self.fast = native_board(root)
            self.keys = [_key(self.fast)]
            if self.cfg.memory:
                rows.append(np.asarray(self.fast.encode_here(self.colour), dtype=np.float32))
        for move in stack[len(self.moves):]:
            self.moves.append(move)
            self.fast.push(fast_rules.code(move))
            key, halfmove, _ = self.fast.fen().rsplit(" ", 2)
            self.keys.append(key)
            if halfmove == "0":
                self.since = len(self.keys) - 1
            if self.cfg.memory:
                rows.append(np.asarray(self.fast.encode_here(self.colour), dtype=np.float32))
        if rows:
            # The memory sees every real position of the game, as TonkAI always did.
            latent = self.net.encode(torch.from_numpy(np.stack(rows)))
            memory = self.memory
            for row in range(len(rows)):
                memory = self.net.memory_cell(latent[row:row + 1], memory)
            self.memory = memory

    # ---- the decision -------------------------------------------------------
    def _lap(self, label):
        if self.clock is not None:
            now = time.perf_counter()
            self.clock[label] = self.clock.get(label, 0.0) + now - self._t
            self._t = now

    def _gain(self, board, move):
        gain = 0.0
        if board.is_capture(move):
            gain = self.values[chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)]
        if move.promotion:
            gain += self.values[move.promotion] - self.values[chess.PAWN]
        return gain

    def _deep(self, codes):
        """What the opponent wins by force after each move (4 half-moves), NaN if aborted."""
        cfg = self.cfg
        try:
            return [loss for loss, _ in self.fast.deep_after_moves(codes, self.values, cfg.tactic_depth,
                                                                   cfg.exchange_depth, cfg.tactic_nodes, True)]
        except RuntimeError:
            return [float("nan")] * len(codes)

    def _recount(self, index, loss, gains, nets, scores):
        if loss == loss:
            new = gains[index] - loss
            scores[index] += self.cfg.safety_weight * (math.tanh(new / 3.0) - math.tanh(nets[index] / 3.0))
            nets[index] = new

    def _root(self, top, codes, gains, nets, scores, known):
        """Same choice as calculating every candidate exactly, for less: each one
        only as far as needed to know whether it beats the best so far."""
        cfg, w = self.cfg, self.cfg.safety_weight
        rest = np.ones(len(codes), dtype=bool)
        rest[top] = False
        floor = float(scores[rest].max()) if rest.any() else -math.inf
        base = [float(scores[i]) - w * math.tanh(nets[i] / 3.0) for i in top]
        try:
            _, best_score, losses, status = self.fast.root_search(
                [codes[i] for i in top], base, [float(gains[i]) for i in top], w, floor, self.values,
                cfg.tactic_depth, cfg.exchange_depth, cfg.tactic_nodes)
        except RuntimeError:
            return
        beaten = []
        for j, index in enumerate(top):
            if status[j] == 1:
                nets[index] = gains[index] - losses[j]
                scores[index] = base[j] + w * math.tanh(nets[index] / 3.0)
                known[index] = True
            elif status[j] in (0, 2):
                beaten.append(index)          # proven not better than the best
        for index in beaten:
            scores[index] = min(scores[index], best_score - 1e-9)

    @torch.inference_mode()
    def choose(self, board: chess.Board) -> chess.Move:
        cfg, me = self.cfg, board.turn
        self._t = time.perf_counter()
        self._catch_up(board)
        self._lap("memory")
        fast = self.fast
        codes = list(fast.legal_moves())
        if not codes:
            raise ValueError("no legal moves")
        if len(codes) == 1:
            return move_of(codes[0])
        moves = [move_of(code) for code in codes]

        # Own forced mate, by the rules: play the first move of the shortest one.
        # Long searches only when its own sense of king safety says the enemy king is exposed.
        attack = cfg.mate_attack
        if attack > 2 and cfg.mate_gate > 0 and self.king_model is not None:
            if float(self.king_model.ks_rows(np.asarray(fast.encode_here(not me)))[0]) >= cfg.mate_gate:
                attack = 2
        if attack > 0:
            found = fast.forced_mate(me, attack, cfg.mate_nodes, cfg.pawn_reach)[0]
            if found:
                mated = fast_rules.WHITE_WINS if me else fast_rules.BLACK_WINS
                for code, move in zip(codes, moves):
                    fast.push(code)
                    ending = fast.terminal_strict()
                    wins = ending == mated or (ending == fast_rules.NONE and found > 1 and
                                               fast.forced_mate(me, found - 1, cfg.mate_nodes, cfg.pawn_reach)[0])
                    fast.pop()
                    if wins:
                        return move
            self._lap("mate_attack")

        # Intuition: the network's value of every position we can reach.
        encoded = fast_rules.encode_after_moves(fast, codes, me)
        memory = self.memory if cfg.memory else self.net.initial_memory(1)
        contexts, _ = self.net.candidates(torch.from_numpy(encoded), memory)
        heads, _ = self.net.survival(contexts)
        values = (heads.mean(-1) if self.head is None else heads[:, self.head]).numpy().astype(np.float64)
        if cfg.value_scaling and len(values) > 1:
            # Self-calibration: the intuition's differences are brought back to the reference
            # spread, so a pessimistic or stretched network does not unbalance the calculation.
            self.spread = cfg.value_spread_decay * self.spread + (1 - cfg.value_spread_decay) * float(values.std())
            centre = float(values.mean())
            values = centre + (values - centre) * (cfg.value_spread_ref / max(self.spread, 1e-6))
        scores = values.copy()
        self._lap("network")
        if cfg.king_weight and self.king_model is not None:
            enemy = fast_rules.encode_after_moves(fast, codes, not me)
            both = self.king_model.ks_rows(np.concatenate([encoded, enemy]))     # one pass for both kings
            scores += cfg.king_weight * (both[:len(codes)] - both[len(codes):])
            self._lap("king_safety")

        # Material: the exact capture sequence after each move, in its own values.
        gains = np.array([self._gain(board, move) for move in moves])
        nets = None
        if cfg.safety_weight:
            try:
                nets = gains - np.asarray(fast.exchange_after_moves(codes, self.values, cfg.exchange_depth,
                                                                   cfg.exchange_nodes))
                scores += cfg.safety_weight * np.tanh(nets / 3.0)
            except RuntimeError:
                nets = None
        if cfg.simplify_weight and nets is not None:
            # Ahead by its own measure of material: trading down what it does not lose brings the end closer.
            lead = sum(self.values[p.piece_type] * (1 if p.color == me else -1)
                       for p in board.piece_map().values() if p.piece_type != chess.KING)
            share = min(1.0, max(0.0, (lead - cfg.simplify_margin) / 3.0))
            if share > 0:
                trades = (gains > 0) & (nets >= -1e-9)
                scores[trades] += cfg.simplify_weight * share
        self._lap("exchange")

        # Moves that end the game have their exact value; going back is a draw.
        known = np.zeros(len(codes), dtype=bool)
        earlier = (set(self.keys[self.since:]) if cfg.repeat_as_draw and len(self.keys) - self.since >= 4
                   else None)
        for index, code in enumerate(codes):
            fast.push(code)
            ending = fast.terminal_strict()
            if ending == fast_rules.DRAW:
                scores[index], known[index] = cfg.draw_value, True
            elif ending != fast_rules.NONE:
                scores[index], known[index] = (WIN if (ending == fast_rules.WHITE_WINS) == me else LOSS), True
            elif earlier is not None and not board.is_zeroing(moves[index]) and _key(fast) in earlier:
                scores[index], known[index] = cfg.draw_value, True
            fast.pop()
        self._lap("endings")

        # Calculation: the best candidates first, together.
        if nets is not None and cfg.tactic_depth > 0 and cfg.tactic_top > 1:
            top = [int(i) for i in np.argsort(scores)[::-1] if not known[i]][:cfg.tactic_top]
            if cfg.root_pvs and top:
                self._root(top, codes, gains, nets, scores, known)
            else:
                for index, loss in zip(top, self._deep([codes[i] for i in top])):
                    self._recount(index, loss, gains, nets, scores)
                    known[index] = True     # calculated: not again
            self._lap("tactic")

        # Check the move it intends to play, then the next one if that fails.
        checked, rounds = set(), 0
        while True:
            if cfg.temperature > 0 and not checked:
                weights = np.exp((scores - scores.max()) / cfg.temperature)
                best = int(self.rng.choice(len(scores), p=weights / weights.sum()))
            else:
                best = int(np.argmax(scores))
            if best in checked or scores[best] >= WIN:
                return moves[best]
            checked.add(best)
            code, before = codes[best], scores[best]
            fast.push(code)
            ending = fast.terminal_strict()
            mate = (ending == fast_rules.NONE and cfg.mate_defence > 0 and
                    fast.forced_mate(not me, cfg.mate_defence, cfg.mate_nodes, cfg.pawn_reach)[0])
            fast.pop()
            self._lap("mate_defence")
            if mate:
                scores[best] = ALLOWS_MATE + 1e-3 * mate  # a forced loss: last, later mates first
                if len(checked) >= len(codes):
                    return moves[int(np.argmax(scores))]
                continue
            if not known[best] and nets is not None and cfg.tactic_depth > 0 and rounds < cfg.tactic_rounds:
                rounds += 1
                self._recount(best, self._deep([code])[0], gains, nets, scores)
                self._lap("tactic")
            if scores[best] >= before - 1e-12 or int(np.argmax(scores)) == best:
                return moves[best]


class ColourPair:
    """Two Claudios joined for evaluations: the white specialist plays White, the black one Black."""

    def __init__(self, white, black):
        self.white, self.black = white, black

    @property
    def players(self):
        return (self.white, self.black)

    def reset(self):
        self.white.reset()
        self.black.reset()

    def choose(self, board):
        return (self.white if board.turn == chess.WHITE else self.black).choose(board)
