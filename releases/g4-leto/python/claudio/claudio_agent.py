"""Claudio: a lean, organic TonkAI player.

What it knows comes from its own experience or from calculation with the
rules of chess, never from outside knowledge:

- intuition: the survival network, trained by playing (inherited from Timdgei);
- piece values: kills/deaths it measured in its own games (K/D), no 1/3/3/5/9;
- king safety (optional): the learned KS model, grounded only in checkmates;
- calculation: the exact capture sequence after each move, and a lazy check of
  the move it intends to play (4 half-moves of material, the opponent's forced
  mates), the way a player checks the move they are about to make;
- one reply further with its own intuition (optional): the network's value of the
  positions after each answer of the opponent, for its best candidates;
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

import abertura
import claudio_fast
import pawnomics
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
    exchange_dynamic: bool = False  # Martinus: each move's exchange goes as many half-moves as there are pieces
                                    # attacking its destination square (both sides, x-rays included); no fixed limit
    tactic_depth: int = 3           # half-moves after our move when checking the intended move
    tactic_top: int = 1             # best candidates checked together first
    root_pvs: bool = True           # ...each only as far as needed to know if it is the best
    tactic_rounds: int = 4          # how many times a new intended move may be checked (lazy)
    tactic_nodes: int = 2_000_000
    shared_table: bool = False      # every calculation of one decision shares one transposition table
    quiesce_promotions: bool = False    # the calculation's capture sequences also see a pawn promoting
    quiesce_checks: bool = False        # ...and, at their first half-move, moves that give check
    check_extension: int = 0            # a move that gives check is calculated one half-move deeper (n per line)
    leaf_weight: float = 0.0        # >0: the network's value where the calculated main line ends replaces, by this
                                    # share, its value right after the move (the line cut to the opponent's turn)
    reply_top: int = 0              # >0: the best n candidates are also judged after each opponent reply (own network)
    reply_weight: float = 1.0       # share of that 2-ply value in those candidates' value (their level is kept)
    reply_mode: str = "min"         # "min": the reply worst for Claudio; "expect": weighted by its own prediction
    reply_depth: int = 2            # 3: also its own best answer to the replies it expects most
    reply_branch: int = 3           # depth 3: replies per candidate it answers (most expected, or the worst)
    mate_defence: int = 2           # opponent's forced mate within n, checked on the intended move
    mate_attack: int = 3            # own forced mates within n
    mate_gate: float = 0.0          # >0: beyond 2 moves only when the learned KS of the enemy king is below this
    mate_nodes: int = 20_000
    pawn_reach: int = 1             # mate searches skip the attacker's pawns this far from the king
    king_weight: float = 0.0        # learned king safety: weight of KS(own) - KS(enemy)
    scan_depth: int = 0             # >0: every legal move, not only the best ones, gets the material calculation at
                                    # this depth: discovered attacks, forks, pieces it wins by force
    ahead_draws: bool = False       # while its own evaluation (the network's value of its best move) is at least
    ahead_value: float = 0.62       # ahead_value, a draw is worth draw_ahead (the owner: -1, below every real
    draw_ahead: float = -1.0        # move: it does not repeat or trade away a position it thinks it is winning)
    ahead_material: float = 0.0     # >0 (the owner: 0.5): "thinks it is winning" mixes, by this share, its material
                                    # lead (own values, 0.5 + 0.5 tanh(lead / 3)) with the network's value
    critical_depth: int = 0         # >0: in a critical moment the calculation goes this deep instead of tactic_depth:
    critical_gap: float = 0.0       # the two best moves this close in score, or its own network unsure (the spread
    critical_spread: float = 0.0    # of its value heads on the best move above critical_spread)
    endgame_pieces: int = 0         # >0: with this many pieces or fewer on the board (kings and pawns included):
    endgame_mate: int = 0           # its own forced mates this long (no king-safety gate),
    endgame_depth: int = 0          # the calculation this deep (forced promotions included),
    endgame_nodes: int = 400_000    # and this node budget for the mate searches
    fifty_horizon: int = 0          # >0: thinking it is winning, a move that does not reset the fifty-move count
                                    # this close to 100 half-moves is a draw (a pawn move or a capture is not)
    phase_values: str = ""          # its own piece values by phase, "P,N,B,R,Q;P,N,B,R,Q;P,N,B,R,Q"
    phase_opening: int = 20         # (opening up to this many half-moves of the game, ending at
    phase_ending: int = 12          # this many pieces or fewer, middlegame otherwise; claudio/valores_fase.py)
    book: str = ""                  # its own opening experience (claudio/experiencia.py): good rate of each move
    book_weight: float = 0.0        # in its own games, weighted this much, for the side to move
    book_min: int = 8               # a move needs this many games to count
    book_plies: int = 16            # only this deep in the game
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
    pawn_weight: float = 0.0        # Pawnomics (claudio/pawnomics.py): how far each pawn gets without being lost or
                                    # stopped, own minus enemy, after each move (pawns; weight of tanh(x / 4))
    pawn_convert: float = 0.0       # ...and that balance counted, by this share, in the lead that converts
                                    # (simplifying, "thinks it is winning")
    pawn_v2: bool = False           # Pawnomics v2: pieces that get in front within two moves; protected, connected
                                    # passed pawns and rooks behind them
    opening_book: bool = False      # the repertoire of opening theory (claudio/abertura.py) while it lasts
    opening_book_patch: bool = False  # sparse fallback only where V2 has no move; frozen book choices win
    opening_weight: float = 0.0     # the classic opening principles, in its first opening_plies half-moves
    opening_plies: int = 24
    opening_book_version: int = 1   # 1 = frozen repertoire; 2 = deeper theoretical repertoire
    opening_principles_version: int = 1  # 1/2/3 = frozen; 4 = readiness plus structural plans
    opening_v3_scale: float = 1.0   # additional V3 terms only; 0 is exactly V2
    opening_v4_scale: float = 1.0   # gated by version=4; 0 is exactly V3, including no transition
    opening_v4_plan_scale: float = 1.0  # structural terms only; 0 keeps V4 readiness/placement
    opening_transition_plies: int = 0  # opt-in end ply, > opening_plies: V4-only terms fade linearly
    opening_fade: str = "legacy"    # frozen 2/3 plateau; linear = gradual throughout; hold = hard cutoff
    syzygy: str = ""                # Martinus: folder of the 3-4-5 Syzygy tables; with 5 pieces or fewer (or a
                                    # capture into them) the tables' exact result decides

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


_BOOKS = {}


def _load_book(path):
    """{position key: {uci: (games, wins of the side to move)}} from claudio/experiencia.py (loaded once)."""
    if path not in _BOOKS:
        import json
        from pathlib import Path
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        _BOOKS[path] = {key: {uci: (int(row[0]), int(row[1])) for uci, row in moves.items()}
                        for key, moves in data["positions"].items()}
    return _BOOKS[path]


class Claudio:
    """choose(board) -> chess.Move. Keeps its own boards and memory in step with
    the real game, so each decision only processes the new plies."""

    def __init__(self, net, values, cfg: ClaudioConfig | None = None, rng=None, king_model=None):
        self.net = net.eval().requires_grad_(False)
        self.cfg = cfg or ClaudioConfig()
        # [0, pawn, knight, bishop, rook, queen] in its own measured units
        self.values = [0.0] + [float(values[kind]) for kind in range(1, 6)]
        self.base_values = list(self.values)
        self.phase_table = None
        if self.cfg.phase_values:
            rows = [[float(x) for x in part.split(",")] for part in self.cfg.phase_values.split(";")]
            if len(rows) != 3 or any(len(row) != 5 for row in rows):
                raise ValueError("phase_values needs three phases of five values")
            self.phase_table = {name: [0.0] + row for name, row in zip(("opening", "middlegame", "ending"), rows)}
        self.book_moves = _load_book(self.cfg.book) if self.cfg.book and self.cfg.book_weight else None
        self.rng = rng or np.random.default_rng()
        self.king_model = king_model
        self.clock = None               # {section: seconds} when profiling
        self.trace = None               # a list: what its own calculation found, to learn from (train_grande)
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

    def _deep_lines(self, codes):
        """What the opponent wins by force after each move (tactic_depth half-moves, NaN if aborted),
        and the main line after each move (empty when aborted)."""
        cfg = self.cfg
        try:
            found = self.fast.deep_after_moves(codes, self.values, self._depth, cfg.exchange_depth,
                                               cfg.tactic_nodes, True)
            return [loss for loss, _ in found], [line for _, line in found]
        except RuntimeError:
            return [float("nan")] * len(codes), [[] for _ in codes]

    def _leaf(self, indices, codes, lines, scores):
        """The network's value where each calculated main line ends - cut to an even length, so the
        opponent is to move as right after the move - replaces, by leaf_weight, the value right after it.
        The calculation counts material; this lets it also see the position its line leads to."""
        after, values, centre, stretch, me = self._leafing
        fast, rows, which = self.fast, [], []
        for index, line in zip(indices, lines):
            line = list(line)[:len(line) // 2 * 2]
            if not line:
                continue
            fast.push(codes[index])
            fast.push_many(line)
            rows.append(np.asarray(fast.encode_here(me), dtype=np.float32))
            for _ in range(len(line) + 1):
                fast.pop()
            which.append(index)
        if not rows:
            return
        memory = after[which] if self.cfg.memory else self.net.initial_memory(1)
        contexts, _ = self.net.candidates(torch.from_numpy(np.stack(rows)), memory)
        for index, value in zip(which, self._value(contexts, centre, stretch)):
            scores[index] += self.cfg.leaf_weight * (value - values[index])

    def _scan(self, fast, codes, gains, nets, scores):
        """Every move, calculated a little further (scan_depth): a quiet move that wins material by force
        - a discovered attack, a fork, an undefended piece that cannot get away - is seen, not only the
        capture sequence. nets and scores are updated in place."""
        cfg = self.cfg
        try:
            deep = gains - np.array([loss for loss, _ in fast.deep_after_moves(
                codes, self.values, cfg.scan_depth, cfg.exchange_depth, cfg.tactic_nodes, True)])
        except RuntimeError:
            return
        found = ~np.isnan(deep)
        scores[found] += cfg.safety_weight * (np.tanh(deep[found] / 3.0) - np.tanh(nets[found] / 3.0))
        nets[found] = deep[found]

    @staticmethod
    def exchange_plies(board, move):
        """Pieces that take part in the exchange on the move's destination: every piece of both sides that
        attacks the square after the move, uncovering the ones behind (x-rays) as each attacker is used."""
        after = board.copy(stack=False)
        after.push(move)
        square, count = move.to_square, 0
        while True:
            found = after.attackers_mask(chess.WHITE, square) | after.attackers_mask(chess.BLACK, square)
            found &= ~chess.BB_SQUARES[square]
            if not found:
                return count
            count += chess.popcount(found)
            for attacker in chess.scan_forward(found):
                after.remove_piece_at(attacker)

    def _dynamic_exchange(self, board, fast, codes, moves):
        """What the opponent wins by captures after each move, each calculated exactly as deep as its exchange."""
        depths = [self.exchange_plies(board, move) for move in moves]
        out = np.zeros(len(codes))
        for depth in set(depths):
            if depth == 0:
                continue
            group = [i for i, d in enumerate(depths) if d == depth]
            found = fast.exchange_after_moves([codes[i] for i in group], self.values, depth, self.cfg.exchange_nodes)
            out[group] = np.asarray(found)
        return out

    def _book(self, fast, codes, moves, scores, known):
        """Moves it has played here in its own games: their good rate for the side to move (good = level or
        better 6 moves later by its own judgement and won, claudio/experiencia.py; draws count 0,
        the owner's rule), pulled towards the position's mean while there are few games, against that mean."""
        cfg = self.cfg
        seen = self.book_moves.get(_key(fast))
        if not seen:
            return
        games = sum(g for g, _ in seen.values())
        mean = sum(w for _, w in seen.values()) / max(1, games)
        for index, move in enumerate(moves):
            row = seen.get(move.uci())
            if row is None or row[0] < cfg.book_min or known[index]:
                continue
            rate = (row[1] + cfg.book_min * mean) / (row[0] + cfg.book_min)
            scores[index] += cfg.book_weight * (rate - mean)

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
                self._depth, cfg.exchange_depth, cfg.tactic_nodes)
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

    def _endings(self, fast, moves, me, earlier):
        """{index: exact value} of the moves (from the position on `fast`) that end the game or go back
        to a position of this game (a draw: the opponent can simply repeat)."""
        cfg, ends = self.cfg, {}
        # The native batch performs the identical push/terminal_strict/pop
        # sequence, avoiding two Python/C++ crossings for every reply.
        batched = (fast.terminal_after_moves(moves)
                   if earlier is None and hasattr(fast, "terminal_after_moves") else None)
        for j, move in enumerate(moves):
            if batched is None:
                fast.push(move)
                ending = fast.terminal_strict()
            else:
                ending = batched[j]
            if ending == fast_rules.DRAW:
                ends[j] = self._draw
            elif ending != fast_rules.NONE:
                ends[j] = WIN if (ending == fast_rules.WHITE_WINS) == me else LOSS
            elif earlier is not None and _key(fast) in earlier:
                ends[j] = self._draw
            if batched is None:
                fast.pop()
        return ends

    def _value(self, contexts, centre, stretch):
        heads, _ = self.net.survival(contexts)
        out = (heads.mean(-1) if self.head is None else heads[:, self.head]).numpy().astype(np.float64)
        return centre + (out - centre) * stretch

    def _material(self, fast, moves):
        """Each move's capture gain minus what the opponent then wins back by force (its own values)."""
        cfg = self.cfg
        kinds = [0] * 64
        for square, kind, _ in fast.pieces():
            kinds[square] = kind
        gains = np.zeros(len(moves))
        for j, code in enumerate(moves):
            frm, to, promotion = code & 63, (code >> 6) & 63, code >> 12
            if kinds[to]:
                gains[j] = self.values[kinds[to]]
            elif kinds[frm] == chess.PAWN and (frm - to) % 8:
                gains[j] = self.values[chess.PAWN]          # en passant
            if promotion:
                gains[j] += self.values[promotion] - self.values[chess.PAWN]
        try:
            return gains - np.asarray(fast.exchange_after_moves(moves, self.values, cfg.exchange_depth,
                                                                cfg.exchange_nodes))
        except RuntimeError:
            return gains

    def _replies(self, fast, codes, me, after, values, scores, known, earlier, centre, stretch, encoded=None):
        """The best candidates judged one reply further: the network's value of the position after
        each answer of the opponent - its worst one ("min"), or weighted by Claudio's own prediction
        of the opponent's move ("expect"). Game-ending replies have their exact value. Only the order
        of these candidates among themselves changes: their mean level is kept.

        reply_depth=3: after the replies it expects most ("min": the worst ones), also its own best
        answer, judged as at the root (network value + the exact capture sequence)."""
        cfg = self.cfg
        top = [int(i) for i in np.argsort(scores)[::-1] if not known[i]][:cfg.reply_top]
        if len(top) < 2:
            return
        rows, sizes, fixed, replies_of = [], [], [], []
        for index in top:
            fast.push(codes[index])
            replies = list(fast.legal_moves())
            fixed.append(self._endings(fast, replies, me, earlier))
            rows.append(fast_rules.encode_after_moves(fast, replies, me))
            fast.pop()
            sizes.append(len(replies))
            replies_of.append(replies)
        if cfg.memory:
            memory = torch.cat([after[i:i + 1].expand(n, -1) for i, n in zip(top, sizes)])
        else:
            memory = self.net.initial_memory(1)
        contexts, later = self.net.candidates(torch.from_numpy(np.concatenate(rows)), memory)
        answers = self._value(contexts, centre, stretch)
        chances = None
        if cfg.reply_mode == "expect":
            chances = [part.softmax(0).numpy().astype(np.float64)
                       for part in self.net.predict_logits(contexts).split(sizes)]
        firsts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int)
        parts = []
        for k, ends in enumerate(fixed):
            part = answers[firsts[k]:firsts[k] + sizes[k]].copy()
            for j, value in ends.items():
                part[j] = value
            parts.append(part)
        if self.trace is not None and encoded is not None:
            self._remember(top, parts, chances, values, encoded, centre, stretch)
        if cfg.reply_depth >= 3:
            self._own_answers(fast, codes, top, replies_of, parts, fixed, chances, firsts, later, me, earlier,
                              centre, stretch)
        # A reply that mates is a known result: it decides, however unlikely Claudio thinks the move is.
        deeper = np.array([part.min() if chances is None or part.min() <= LOSS / 2 else chances[k] @ part
                           for k, part in enumerate(parts)])
        top = np.asarray(top)
        plain = deeper > LOSS / 2          # a reply that mates is not part of the level
        shift = float(np.mean(values[top][plain] - deeper[plain])) if plain.any() else 0.0
        scores[top] += cfg.reply_weight * (deeper + shift - values[top])

    def _remember(self, top, parts, chances, values, encoded, centre, stretch):
        """What learning reads (train_grande --learn search): for the best candidates, the value one reply
        further (its own network, the replies weighted as in the search), at the candidates' mean level,
        as a target for its intuition. A candidate the opponent can answer with mate: 0."""
        deeper = np.array([part.min() if chances is None or part.min() <= LOSS / 2 else chances[k] @ part
                           for k, part in enumerate(parts)])
        top = np.asarray(top)
        plain = deeper > LOSS / 2
        if not plain.any():
            return
        level = deeper + float(np.mean(values[top][plain] - deeper[plain]))
        level = centre + (level - centre) / max(stretch, 1e-9)       # back to the network's own scale
        targets = np.where(plain, np.clip(level, 0.0, 1.0), 0.0).astype(np.float32)
        memory = (self.memory if self.cfg.memory else self.net.initial_memory(1)).numpy()
        self.trace.append((encoded[top], np.repeat(memory, len(top), 0).astype(np.float16), targets))

    def _own_answers(self, fast, codes, top, replies_of, parts, fixed, chances, firsts, later, me, earlier,
                     centre, stretch):
        """Third half-move: Claudio's best answer to the replies it expects most (or the worst ones).
        The replies not looked at keep their value, moved by the candidate's mean change."""
        cfg = self.cfg
        picks, rows, sizes, ends_of, nets_of = [], [], [], [], []
        for k, index in enumerate(top):
            order = np.argsort(-chances[k]) if chances is not None else np.argsort(parts[k])
            chosen = [int(j) for j in order if int(j) not in fixed[k]][:cfg.reply_branch]
            for j in chosen:
                fast.push(codes[index])
                fast.push(replies_of[k][j])
                mine = list(fast.legal_moves())
                if mine:
                    picks.append((k, j))
                    ends_of.append(self._endings(fast, mine, me, earlier))
                    nets_of.append(self._material(fast, mine) if cfg.safety_weight else np.zeros(len(mine)))
                    rows.append(fast_rules.encode_after_moves(fast, mine, me))
                    sizes.append(len(mine))
                fast.pop()
                fast.pop()
        if not picks:
            return
        if cfg.memory:
            memory = torch.cat([later[firsts[k] + j:firsts[k] + j + 1].expand(n, -1)
                                for (k, j), n in zip(picks, sizes)])
        else:
            memory = self.net.initial_memory(1)
        contexts, _ = self.net.candidates(torch.from_numpy(np.concatenate(rows)), memory)
        values = self._value(contexts, centre, stretch)
        best, start = {}, 0
        for (k, j), n, ends, nets in zip(picks, sizes, ends_of, nets_of):
            own = values[start:start + n] + cfg.safety_weight * np.tanh(nets / 3.0)
            start += n
            for a, value in ends.items():
                own[a] = value
            best.setdefault(k, {})[j] = float(own.max())
        for k, found in best.items():
            moves = [(j, value - parts[k][j]) for j, value in found.items() if abs(value) < WIN / 2]
            change = float(np.mean([d for _, d in moves])) if moves else 0.0
            rest = np.ones(len(parts[k]), dtype=bool)
            rest[list(fixed[k])] = False
            rest[list(found)] = False
            parts[k][rest] += change
            for j, value in found.items():
                parts[k][j] = value

    def _tablebase(self, board):
        """(move, None) when the tables settle the decision; else (None, moves to leave out, or None).

        Every legal move whose position is in the tables gets its exact result for us (2 win, 1 win the
        fifty-move rule spoils, 0 draw, ...). A win is played at once (a mate first, then a capture or pawn
        move, then the shortest way); with every move known and only losses, the longest one. Otherwise the
        known moves worse than the best known are left out and its own judgement picks among the rest."""
        from tonkai import tablebase
        if chess.popcount(board.occupied) > tablebase.MAX_PIECES + 1:
            return None, None
        tables = tablebase.tablebase(self.cfg.syzygy)
        known = {}
        for move in board.legal_moves:
            if not tablebase.enters_tables(board, move):
                continue
            board.push(move)
            try:
                if board.is_checkmate():
                    known[move] = (3, 1, 0)
                elif board.castling_rights:
                    pass
                else:
                    wdl = -tables.probe_wdl(board)
                    dtz = abs(tables.probe_dtz(board)) if wdl else 0
                    known[move] = (wdl, int(board.halfmove_clock == 0), dtz)
            except (KeyError, chess.syzygy.MissingTableError):
                pass
            finally:
                board.pop()
        if not known:
            return None, None
        best = max(r[0] for r in known.values())
        if best >= 1:
            wins = [m for m, r in known.items() if r[0] == best]
            return max(wins, key=lambda m: (known[m][1], -known[m][2])), None
        everything = len(known) == board.legal_moves.count()
        if everything and best < 0:
            return max(known, key=lambda m: (known[m][0], known[m][2])), None
        worse = {m.uci() for m, r in known.items() if r[0] < (best if everything else 0)}
        return None, worse

    def choose(self, board: chess.Board) -> chess.Move:
        if self.cfg.syzygy:
            move, worse = self._tablebase(board)
            if move is not None:
                with torch.inference_mode():
                    self._catch_up(board)              # its memory stays in step with the game
                return move
            if worse:
                before = getattr(self, "avoid", None)
                self.avoid = set(before or ()) | worse
                try:
                    return self._choose_dynamic(board)
                finally:
                    self.avoid = before
        return self._choose_dynamic(board)

    def _choose_dynamic(self, board: chess.Board) -> chess.Move:
        # Martinus: the calculated lines also end their exchanges as deep as the pieces taking part
        # (claudio_fast Board.dynamic_exchange); on only during its own decision.
        dynamic = self.cfg.exchange_dynamic and hasattr(claudio_fast.Board, "dynamic_exchange")
        if not dynamic:
            return self._choose_with_options(board)
        claudio_fast.Board.dynamic_exchange(True)
        try:
            return self._choose_with_options(board)
        finally:
            claudio_fast.Board.dynamic_exchange(False)

    def _choose_with_options(self, board: chess.Board) -> chess.Move:
        cfg = self.cfg
        options = (cfg.quiesce_promotions, cfg.quiesce_checks, cfg.check_extension)
        special = any(options) and hasattr(claudio_fast.Board, "search_options")
        session = cfg.shared_table and hasattr(claudio_fast.Board, "table_session")
        if not special and not session:
            return self._choose(board)
        # The calculation's options for this decision only; and one decision, one table: a position met
        # again (another candidate, a deeper check) is not calculated from scratch (the piece values,
        # the options and the search mode stay the same within it).
        if special:
            claudio_fast.Board.search_options(*options)
        if session:
            claudio_fast.Board.table_session(True)
        try:
            return self._choose(board)
        finally:
            if session:
                claudio_fast.Board.table_session(False)
            if special:
                claudio_fast.Board.search_options(False, False, 0)

    @torch.inference_mode()
    def _choose(self, board: chess.Board) -> chess.Move:
        cfg, me = self.cfg, board.turn
        self._t = time.perf_counter()
        self._catch_up(board)
        self._lap("memory")
        fast = self.fast
        codes = list(fast.legal_moves())
        if not codes:
            raise ValueError("no legal moves")
        avoid = getattr(self, "avoid", None)
        if avoid:
            # Moves it may not play now (the explorer's cooldown): its next best instead, if any is left.
            kept = [code for code in codes if move_of(code).uci() not in avoid]
            codes = kept or codes
        if len(codes) == 1:
            return move_of(codes[0])
        moves = [move_of(code) for code in codes]
        if cfg.opening_book and board.ply() < 40:
            theory = abertura.repertoire_move(board, cfg.opening_book_version)
            if theory is None and cfg.opening_book_patch:
                theory = abertura.repertoire_patch_move(board)
            if theory is not None and theory in moves:
                return theory

        # The phase of the game: its own piece values for it; the endgame, where calculating is cheap.
        pieces = chess.popcount(board.occupied)
        phase = ("opening" if board.ply() < cfg.phase_opening else
                 "ending" if pieces <= cfg.phase_ending else "middlegame")
        self.values = self.phase_table[phase] if self.phase_table else self.base_values
        endgame = cfg.endgame_pieces > 0 and pieces <= cfg.endgame_pieces
        self._depth = max(cfg.tactic_depth, cfg.endgame_depth) if endgame else cfg.tactic_depth

        # Own forced mate, by the rules: play the first move of the shortest one.
        # Long searches only when its own sense of king safety says the enemy king is exposed
        # (in the endgame, with few pieces, always: the search is cheap and finds the long mates).
        attack, mate_nodes = cfg.mate_attack, cfg.mate_nodes
        if endgame and cfg.endgame_mate > attack:
            attack, mate_nodes = cfg.endgame_mate, max(cfg.mate_nodes, cfg.endgame_nodes)
        elif attack > 2 and cfg.mate_gate > 0 and self.king_model is not None:
            if float(self.king_model.ks_rows(np.asarray(fast.encode_here(not me)))[0]) >= cfg.mate_gate:
                attack = 2
        if attack > 0:
            found = fast.forced_mate(me, attack, mate_nodes, cfg.pawn_reach)[0]
            if found:
                mated = fast_rules.WHITE_WINS if me else fast_rules.BLACK_WINS
                for code, move in zip(codes, moves):
                    fast.push(code)
                    ending = fast.terminal_strict()
                    wins = ending == mated or (ending == fast_rules.NONE and found > 1 and
                                               fast.forced_mate(me, found - 1, mate_nodes, cfg.pawn_reach)[0])
                    fast.pop()
                    if wins:
                        return move
            self._lap("mate_attack")

        # Intuition: the network's value of every position we can reach.
        encoded = fast_rules.encode_after_moves(fast, codes, me)
        memory = self.memory if cfg.memory else self.net.initial_memory(1)
        contexts, after = self.net.candidates(torch.from_numpy(encoded), memory)
        heads, _ = self.net.survival(contexts)
        values = (heads.mean(-1) if self.head is None else heads[:, self.head]).numpy().astype(np.float64)
        centre, stretch = 0.0, 1.0
        raw_best = float(values.max())          # the network's own value, before any rescaling
        if cfg.value_scaling and len(values) > 1:
            # Self-calibration: the intuition's differences are brought back to the reference
            # spread, so a pessimistic or stretched network does not unbalance the calculation.
            self.spread = cfg.value_spread_decay * self.spread + (1 - cfg.value_spread_decay) * float(values.std())
            centre, stretch = float(values.mean()), cfg.value_spread_ref / max(self.spread, 1e-6)
            values = centre + (values - centre) * stretch
        scores = values.copy()
        self._leafing = (after, values, centre, stretch, me) if cfg.leaf_weight else None
        self._lap("network")
        if cfg.king_weight and self.king_model is not None:
            enemy = fast_rules.encode_after_moves(fast, codes, not me)
            both = self.king_model.ks_rows(np.concatenate([encoded, enemy]))     # one pass for both kings
            scores += cfg.king_weight * (both[:len(codes)] - both[len(codes):])
            self._lap("king_safety")

        if cfg.opening_weight and board.ply() < cfg.opening_plies:
            share = cfg.opening_weight * abertura.weight_at(board.ply(), cfg.opening_plies, cfg.opening_fade)
            if share:
                if cfg.opening_principles_version == 4:
                    opening_scores = abertura.principle_scores(board, moves, board.ply(), 4,
                                                              cfg.opening_v3_scale, cfg.opening_v4_scale,
                                                              cfg.opening_v4_plan_scale)
                else:
                    opening_scores = abertura.principle_scores(board, moves, board.ply(),
                                                              cfg.opening_principles_version, cfg.opening_v3_scale)
                scores += share * np.array(opening_scores)
        elif (cfg.opening_weight and cfg.opening_principles_version == 4 and cfg.opening_v4_scale and
              cfg.opening_plies <= board.ply() < cfg.opening_transition_plies):
            import abertura_v4
            share = cfg.opening_weight * abertura_v4.transition_weight(board.ply(), cfg.opening_plies,
                                                                      cfg.opening_transition_plies)
            scores += share * np.array(abertura_v4.scores(board, moves, board.ply(), cfg.opening_v4_scale,
                                                        cfg.opening_v4_plan_scale))
        if cfg.pawn_weight:
            after_board = board.copy(stack=False)
            balances = np.zeros(len(moves))
            for index, move in enumerate(moves):
                after_board.push(move)
                if pawnomics.has_runners(after_board):
                    balances[index] = pawnomics.balance(after_board, me, self.values, cfg.pawn_v2)
                after_board.pop()
            scores += cfg.pawn_weight * np.tanh(balances / 4.0)
            self._lap("pawnomics")
        root_pawns = (pawnomics.balance(board, me, self.values, cfg.pawn_v2)
                      if cfg.pawn_convert and pawnomics.has_runners(board) else 0.0)

        # Material: the exact capture sequence after each move, in its own values.
        gains = np.array([self._gain(board, move) for move in moves])
        nets = None
        if cfg.safety_weight:
            try:
                if cfg.exchange_dynamic:
                    nets = gains - self._dynamic_exchange(board, fast, codes, moves)
                else:
                    nets = gains - np.asarray(fast.exchange_after_moves(codes, self.values, cfg.exchange_depth,
                                                                       cfg.exchange_nodes))
                scores += cfg.safety_weight * np.tanh(nets / 3.0)
            except RuntimeError:
                nets = None
        if cfg.scan_depth > 0 and nets is not None:
            self._scan(fast, codes, gains, nets, scores)
        # A draw is worth less while it thinks it is winning (its own evaluation of the position).
        ahead = float(values.max())
        if cfg.ahead_material > 0:
            lead = sum(self.values[p.piece_type] * (1 if p.color == me else -1)
                       for p in board.piece_map().values() if p.piece_type != chess.KING)
            lead += cfg.pawn_convert * root_pawns
            ahead = ((1 - cfg.ahead_material) * raw_best
                     + cfg.ahead_material * (0.5 + 0.5 * math.tanh(lead / 3.0)))
        self._draw = (cfg.draw_ahead if cfg.ahead_draws and ahead >= cfg.ahead_value
                      else cfg.draw_value)
        if cfg.simplify_weight and nets is not None:
            # Ahead by its own measure of material: trading down what it does not lose brings the end closer.
            lead = sum(self.values[p.piece_type] * (1 if p.color == me else -1)
                       for p in board.piece_map().values() if p.piece_type != chess.KING)
            lead += cfg.pawn_convert * root_pawns
            share = min(1.0, max(0.0, (lead - cfg.simplify_margin) / 3.0))
            if share > 0:
                trades = (gains > 0) & (nets >= -1e-9)
                scores[trades] += cfg.simplify_weight * share
        self._lap("exchange")

        # Moves that end the game have their exact value; going back is a draw.
        known = np.zeros(len(codes), dtype=bool)
        earlier = (set(self.keys[self.since:]) if cfg.repeat_as_draw and len(self.keys) - self.since >= 4
                   else None)
        batched = (fast.terminal_after_moves(codes)
                   if earlier is None and hasattr(fast, "terminal_after_moves") else None)
        for index, code in enumerate(codes):
            if batched is None:
                fast.push(code)
                ending = fast.terminal_strict()
            else:
                ending = batched[index]
            if ending == fast_rules.DRAW:
                scores[index], known[index] = self._draw, True
            elif ending != fast_rules.NONE:
                scores[index], known[index] = (WIN if (ending == fast_rules.WHITE_WINS) == me else LOSS), True
            elif earlier is not None and not board.is_zeroing(moves[index]) and _key(fast) in earlier:
                scores[index], known[index] = self._draw, True
            if batched is None:
                fast.pop()
        self._lap("endings")

        # Thinking it is winning, the fifty-move rule is a draw coming: near it, a move that does not reset
        # the count is one (when a pawn move or a capture is there to play).
        if cfg.fifty_horizon > 0 and self._draw < cfg.draw_value:
            resets = np.array([board.is_zeroing(move) for move in moves])
            if resets.any() and board.halfmove_clock + 1 >= 100 - cfg.fifty_horizon:
                for index in np.flatnonzero(~resets & ~known):
                    scores[index], known[index] = self._draw, True

        # Its own opening experience: the win rate of each move in its own games, for the side to move.
        if self.book_moves is not None and board.ply() < cfg.book_plies:
            self._book(fast, codes, moves, scores, known)

        # One reply further with its own intuition, for the best candidates.
        if cfg.reply_top > 0:
            self._replies(fast, codes, me, after, values, scores, known, earlier, centre, stretch, encoded)
            self._lap("replies")

        # A critical moment - the two best moves almost level, or its own network unsure of the best one:
        # the calculation goes deeper for this move.
        if cfg.critical_depth > self._depth and (cfg.critical_gap > 0 or cfg.critical_spread > 0):
            open_ = np.flatnonzero(~known)
            if len(open_) >= 2:
                ranked = open_[np.argsort(scores[open_])[::-1]]
                gap = float(scores[ranked[0]] - scores[ranked[1]])
                unsure = float(heads[ranked[0]].std()) if self.head is None else 0.0
                self.last_moment = (gap, unsure)
                if gap < cfg.critical_gap or (cfg.critical_spread > 0 and unsure > cfg.critical_spread):
                    self._depth = cfg.critical_depth
                    self.critical_moves = getattr(self, "critical_moves", 0) + 1

        # Calculation: the best candidates first, together.
        if nets is not None and self._depth > 0 and cfg.tactic_top > 1:
            top = [int(i) for i in np.argsort(scores)[::-1] if not known[i]][:cfg.tactic_top]
            if cfg.root_pvs and top and not cfg.leaf_weight:
                self._root(top, codes, gains, nets, scores, known)
            else:
                losses, lines = self._deep_lines([codes[i] for i in top])
                for index, loss in zip(top, losses):
                    self._recount(index, loss, gains, nets, scores)
                    known[index] = True     # calculated: not again
                if cfg.leaf_weight:
                    self._leaf(top, codes, lines, scores)
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
            if not known[best] and nets is not None and self._depth > 0 and rounds < cfg.tactic_rounds:
                rounds += 1
                losses, lines = self._deep_lines([code])
                self._recount(best, losses[0], gains, nets, scores)
                if cfg.leaf_weight:
                    self._leaf([best], codes, lines, scores)
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
