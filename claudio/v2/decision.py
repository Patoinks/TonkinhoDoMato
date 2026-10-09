"""Pinned 065d528 decision, with native repetition batching and own-policy hooks."""
import math
import time
import chess
import numpy as np
import torch
from claudio_agent import WIN, LOSS, ALLOWS_MATE, _key, move_of
from tonkai import fast_rules

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
    self.policy_contexts = contexts
    heads, _ = self.net.survival(contexts)
    values = (heads.mean(-1) if self.head is None else heads[:, self.head]).numpy().astype(np.float64)
    centre, stretch = 0.0, 1.0
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
    if cfg.scan_depth > 0 and nets is not None:
        self._scan(fast, codes, gains, nets, scores)
    # A draw is worth less while it thinks it is winning (its own evaluation of the position).
    self._draw = (cfg.draw_ahead if cfg.ahead_draws and float(values.max()) >= cfg.ahead_value
                  else cfg.draw_value)
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
    if self.options.batch_repetition:
        endings, repeated = fast.terminal_repetition_after_moves(codes, list(earlier or []), self.dead_candidate_possible)
    else:
        endings, repeated = [], []
        for code in codes:
            fast.push(code)
            endings.append(fast.terminal_strict())
            repeated.append(earlier is not None and _key(fast) in earlier)
            fast.pop()
    for index, ending in enumerate(endings):
        if ending == fast_rules.DRAW:
            scores[index], known[index] = self._draw, True
        elif ending != fast_rules.NONE:
            scores[index], known[index] = (WIN if (ending == fast_rules.WHITE_WINS) == me else LOSS), True
        elif earlier is not None and not board.is_zeroing(moves[index]) and repeated[index]:
            scores[index], known[index] = self._draw, True
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
    self.last_scores = scores.copy()
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
