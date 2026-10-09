"""Organic experimental G16 v2: own policy/book, exact rules, proof-based conversion."""
from collections import OrderedDict
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
import time

import chess
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'native'))
import claudio_fast_v2 as native
from claudio_agent import Claudio, ClaudioConfig, WIN, LOSS, _key, move_of
from claudio_load import load_claudio
from tonkai import fast_rules
from .inference import FrozenInference
from .rules import adjudicate, draw_config


@dataclass
class V2Options:
    batch_repetition: bool = True
    packed_heads: bool = True
    trajectory_cache: bool = True
    learned_policy: bool = True
    capture_policy_inputs: bool = False
    opening_book: bool = True
    opening_fast: bool = True
    finish_planner: bool = True
    finish_pieces: int = 5
    finish_moves: int = 8
    finish_nodes: int = 400_000
    shared_table: bool = False
    quiet_promotions: bool = False


class OwnPolicy(nn.Module):
    def __init__(self, latent=128, hidden=32):
        super().__init__()
        self.rank = nn.Sequential(nn.Linear(latent, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.features = nn.Sequential(nn.Linear(3, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.rank[-1].weight)
        nn.init.zeros_(self.rank[-1].bias)
        nn.init.zeros_(self.features[-1].weight)
        nn.init.zeros_(self.features[-1].bias)
        self.register_buffer('temperature', torch.tensor(0.02))
        self.register_buffer('blend', torch.tensor(1.0))

    def forward(self, contexts, features=None, base=None):
        if features is None:
            features = torch.zeros(len(contexts), 3, device=contexts.device)
        residual = self.blend * 2 * torch.tanh(self.rank(contexts).squeeze(-1) + self.features(features).squeeze(-1))
        return residual if base is None else base / self.temperature + residual


class NativeBoard:
    """The same native state, with bounded pure-rule memoization (no neural state cached)."""
    def __init__(self, board):
        root = board.root()
        self.native = native.Board(root.pawns, root.knights, root.bishops, root.rooks, root.queens, root.kings,
                                   root.occupied_co[chess.WHITE], root.occupied_co[chess.BLACK], root.turn,
                                   root.castling_rights, -1 if root.ep_square is None else root.ep_square,
                                   root.halfmove_clock, root.fullmove_number)
        self.native.push_many([fast_rules.code(m) for m in board.move_stack])
        self.mates = OrderedDict()

    def __getattr__(self, name):
        return getattr(self.native, name)

    def forced_mate(self, attacker, maximum, nodes, reach=0):
        key = (self.native.fen(), attacker, maximum, nodes, reach)
        if key not in self.mates:
            self.mates[key] = self.native.forced_mate(attacker, maximum, nodes, reach)
            if len(self.mates) > 256:
                self.mates.popitem(last=False)
        return self.mates[key]


class V2Claudio(Claudio):
    def __init__(self, net, values, cfg, rng=None, king_model=None, options=None, policy=None):
        self.options = options or V2Options()
        self.policy = policy.eval().requires_grad_(False) if policy is not None else None
        self.policy_threshold = 1.01  # activated only after held-out own-policy calibration
        chosen_net = FrozenInference(net) if self.options.packed_heads else net
        cfg = replace(cfg, draw_value=0.5, draw_ahead=0.5)
        if not self.options.opening_book:
            cfg=replace(cfg,book='',book_weight=0)
        super().__init__(chosen_net, values, cfg, rng=rng, king_model=king_model)

    def reset(self):
        super().reset()
        self.real_rows, self.real_memory, self.real_latent = [], [], []
        self.stats = dict(opening_fast=0, opening_rejected=0, planner_calls=0, planner_hits=0,
                          planner_nodes=0, policy_reorders=0, decisions=0, book_hits=0)
        self.policy_contexts = None
        self.last_scores = None
        self.order_snapshot = None

    @torch.inference_mode()
    def _advance(self, board, allow_other_turn=False):
        stack = board.move_stack
        if self.colour is not None:
            seen = len(self.moves)
            if ((not allow_other_turn and self.colour != board.turn) or len(stack) < seen
                    or (seen and stack[seen - 1] != self.moves[-1])
                    or (not seen and board.root().fen() != self.root)):
                self.reset()
        rows = []
        if self.colour is None:
            root = board.root()
            self.colour, self.root, self.moves = board.turn, root.fen(), []
            self.fast = NativeBoard(root)
            self.keys = [_key(self.fast)]
            if self.cfg.memory:
                rows.append(np.asarray(self.fast.encode_here(self.colour), dtype=np.float32))
        for move in stack[len(self.moves):]:
            self.moves.append(move)
            self.fast.push(fast_rules.code(move))
            key, halfmove, _ = self.fast.fen().rsplit(' ', 2)
            self.keys.append(key)
            if halfmove == '0':
                self.since = len(self.keys) - 1
            if self.cfg.memory:
                rows.append(np.asarray(self.fast.encode_here(self.colour), dtype=np.float32))
        if rows:
            latent = self.net.encode(torch.from_numpy(np.stack(rows)))
            memory = self.memory
            for i, row in enumerate(rows):
                memory = self.net.memory_cell(latent[i:i + 1], memory)
                if self.options.trajectory_cache:
                    self.real_rows.append(row.copy())
                    self.real_memory.append(memory.clone())
                    self.real_latent.append(latent[i:i + 1].clone())
            self.memory = memory

    def _catch_up(self, board):
        self._advance(board)

    def _endings(self, fast, moves, me, earlier):
        if not self.options.batch_repetition:
            return super()._endings(fast, moves, me, earlier)
        ends, repeats = fast.terminal_repetition_after_moves(moves, list(earlier or []),
                                                           self.dead_candidate_possible)
        result = {}
        for i, ending in enumerate(ends):
            if ending == fast_rules.DRAW:
                result[i] = self._draw
            elif ending != fast_rules.NONE:
                result[i] = WIN if (ending == fast_rules.WHITE_WINS) == me else LOSS
            elif repeats[i]:
                result[i] = self._draw
        return result

    def _root(self, top, codes, gains, nets, scores, known):
        if self.options.capture_policy_inputs:
            self.order_snapshot = {'top':list(top), 'scores':scores.copy(), 'gains':gains.copy(), 'nets':nets.copy()}
        if self.options.learned_policy and self.policy is not None and self.policy_contexts is not None:
            if self.policy_threshold <= 1.0:
                base = np.asarray(scores[top] - scores[top].max(), dtype=np.float32)
                temperature = float(self.policy.temperature)
                features = np.column_stack([np.tanh(gains[top]/3), np.tanh(nets[top]/3),
                                            np.clip(base/temperature,-10,0)]).astype(np.float32)
                ranks = self.policy(self.policy_contexts[top], torch.from_numpy(features), torch.from_numpy(base))
                if float(ranks.softmax(0).max()) >= self.policy_threshold:
                    new = [top[i] for i in np.argsort(-ranks.detach().numpy(), kind='stable')]
                    self.stats['policy_reorders'] += int(new != top)
                    top = new
        return super()._root(top, codes, gains, nets, scores, known)

    def _opening(self, board):
        if (not self.options.opening_book or not self.options.opening_fast or not self.book_moves
                or board.ply() >= self.cfg.book_plies or board.is_check()):
            return None
        choices = self.book_moves.get(board.fen().rsplit(' ', 2)[0], {})
        supported = [(uci, n, good) for uci, (n, good) in choices.items() if n >= max(16, self.cfg.book_min)]
        if not supported:
            return None
        def interval(n, good):
            z, p = 1.96, good / n
            centre = (p + z*z/(2*n)) / (1 + z*z/n)
            width = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
            return centre-width, centre+width
        best = max(supported, key=lambda row: interval(row[1], row[2])[0])
        low, _ = interval(best[1], best[2])
        # Require actual evidence over alternatives, then check the chosen move by rules.
        others = [interval(n, good)[1] for uci, n, good in supported if uci != best[0]]
        if low < 0.5 or (others and low <= max(others)):
            return None
        move = chess.Move.from_uci(best[0])
        if move not in board.legal_moves:
            return None
        code = fast_rules.code(move)
        me = board.turn
        if self.fast.forced_mate(me, self.cfg.mate_attack, self.cfg.mate_nodes, self.cfg.pawn_reach)[0]:
            return None
        loss = self.fast.deep_after_moves([code], self.values, self.cfg.tactic_depth,
                                          self.cfg.exchange_depth, self.cfg.tactic_nodes, True)[0][0]
        self.fast.push(code)
        try:
            ending = self.fast.terminal_strict()
            unsafe = ending != fast_rules.NONE or _key(self.fast) in self.keys[self.since:]
            unsafe |= bool(self.fast.forced_mate(not me, self.cfg.mate_defence,
                                                self.cfg.mate_nodes, self.cfg.pawn_reach)[0])
        finally:
            self.fast.pop()
        if unsafe or not np.isfinite(loss) or self._gain(board, move) - loss < -1e-9:
            self.stats['opening_rejected'] += 1
            return None
        self.stats['opening_fast'] += 1
        return move

    def _book(self, fast, codes, moves, scores, known):
        # The experience file uses python-chess's legal EP field. Native FEN exposes
        # raw EP: after e2e4 these differ, even though no en-passant capture is legal.
        # Canonicalize only the book lookup; preserve repetition semantics elsewhere.
        canonical = self.book_board.fen()
        if canonical.rsplit(' ', 2)[0] not in self.book_moves:
            return
        class BookView:
            def fen(self):
                return canonical
        self.stats['book_hits'] += 1
        return super()._book(BookView(), codes, moves, scores, known)

    def _finish(self, board):
        if not self.options.finish_planner or chess.popcount(board.occupied) > self.options.finish_pieces:
            return None
        me = board.turn
        enemy = board.occupied_co[not me] & ~board.kings
        lead = sum(self.values[p.piece_type] * (1 if p.color == me else -1)
                   for p in board.piece_map().values() if p.piece_type != chess.KING)
        if enemy or lead <= 0:
            return None
        self.stats['planner_calls'] += 1
        before = time.perf_counter()
        fen, history = board.fen(en_passant='fen'), self.keys[self.since:-1]
        result = native.mating_plan(fen, history, min(4,self.options.finish_moves), 40_000)
        used = result['nodes']
        if not result['moves']:
            if self.real_latent:
                latent = self.real_latent[-1]
            else:
                latent = self.net.encode(torch.from_numpy(np.asarray(self.fast.encode_here(me),dtype=np.float32))[None])
            context = self.net.context_from_latent(latent, self.memory)
            confidence = float(self.net.survival(context)[0].mean())
            # Material is only a budget hint. A win is returned only after a full rules proof.
            if confidence >= self.cfg.ahead_value or board.occupied_co[me] & (board.rooks | board.queens):
                result = native.mating_plan(fen, history, self.options.finish_moves, self.options.finish_nodes)
                used += result['nodes']
        self.stats['planner_nodes'] += used
        self.stats['planner_seconds'] = self.stats.get('planner_seconds', 0.0) + time.perf_counter() - before
        if result['moves'] and result['move']:
            move = move_of(result['move'])
            if move not in board.legal_moves:
                raise AssertionError('Mate planner returned an illegal move')
            self.stats['planner_hits'] += 1
            return move
        return None

    @torch.inference_mode()
    def choose(self, board, terminal_checked=False):
        ending = None if terminal_checked else adjudicate(board)
        if ending is not None:
            raise ValueError(f"Game already ended: {ending['end']}")
        self.book_board = board
        self._catch_up(board)
        self.dead_candidate_possible = chess.popcount(board.knights | board.bishops | board.rooks | board.queens) <= 2
        self.order_snapshot = self.policy_contexts = None
        self.stats['decisions'] += 1
        phase = 'opening' if board.ply() < self.cfg.phase_opening else 'ending' if chess.popcount(board.occupied) <= self.cfg.phase_ending else 'middlegame'
        self.values = self.phase_table[phase] if self.phase_table else self.base_values
        finish = self._finish(board)
        if finish is not None:
            return finish
        opening = self._opening(board)
        if opening is not None:
            return opening
        native.Board.search_options(self.options.quiet_promotions, False, 0)
        native.Board.table_session(self.options.shared_table)
        try:
            from .decision import choose
            return choose(self, board)
        finally:
            native.Board.table_session(False)
            native.Board.search_options(False, False, 0)


def load_v2(checkpoint, assets=None, variant='full', rng=None):
    net, values, config, _, king = load_claudio(checkpoint, with_config=True, with_king=True)
    options = V2Options()
    policy = None
    if variant == 'exact':
        options.learned_policy = options.opening_book = options.opening_fast = options.finish_planner = False
    else:
        if assets:
            folder = Path(assets)
            if variant in ('full','all') and (folder / 'compact.pt').exists():
                net, _, _, _ = load_claudio(folder / 'compact.pt', with_config=True)
            if (folder / 'policy.pt').exists():
                policy = OwnPolicy(net.latent_size)
                policy.load_state_dict(torch.load(folder / 'policy.pt', map_location='cpu', weights_only=True))
            if (folder / 'opening-book.json').exists():
                config = dict(config, book=str(folder / 'opening-book.json'), book_weight=0.03, book_min=4)
        if variant == 'finish':
            options.learned_policy = options.opening_book = options.opening_fast = False
    config = draw_config(dict(config, endgame_mate=6))
    result = V2Claudio(net, values, ClaudioConfig.with_settings([], base=config), rng=rng, king_model=king,
                       options=options, policy=policy)
    if assets and (Path(assets)/'learning.json').exists():
        result.policy_threshold=json.loads((Path(assets)/'learning.json').read_text())['policy_confidence_threshold']
    if variant=='all' and result.policy is not None:
        result.policy_threshold=0.0
        result.policy.blend.fill_(1.0)
    return result
