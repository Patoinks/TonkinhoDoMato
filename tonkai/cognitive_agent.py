"""V3 policy: retrieval, evidence-seeking exploration and bounded robust search."""
from collections import Counter

import chess
import numpy as np
import torch

from . import fast_rules
from .encoding import encode, encode_after_moves
from .survival_agent import SurvivalAgent, SurvivalChoice
from .cognitive_memory import structural_keys
from .piece_safety import (favorable_attack, legal_mobility, piece_senses, pawn_unit,
                           position_pressure, position_risk, protected_mobility_gain)
from .king_safety import king_senses, king_balance
from .objective import outcome_value, continuation_value, victory_weight
from .king_safety_model import load_cached as load_king_safety


def terminal_draw_value(objective="survival", weight=0.5, draw=None):
    if draw is not None:                     # the owner's explicit value of a draw
        return float(draw)
    return .5 if objective == "victory" else 1.0 - weight / 2 if objective == "shaped" else 1.0


def terminal_value(board, colour, objective="survival", weight=0.5, draw=None):
    draw = terminal_draw_value(objective, weight, draw)
    result = board.outcome(claim_draw=False)
    if result is None and board.can_claim_fifty_moves():
        return draw
    if result is None and min(board.halfmove_clock, len(board.move_stack)) >= 7:
        # A claim on the next move requires some historical position to have
        # occurred twice already. Most positions have no repeats: avoid probing
        # every legal next move there. The authoritative check still decides all
        # possible repetition cases, including claims before the third occurrence.
        history = board.copy(stack=True)
        seen = {history._transposition_key()}
        duplicate = False
        while history.move_stack:
            history.pop()
            key = history._transposition_key()
            if key in seen:
                duplicate = True
                break
            seen.add(key)
        if duplicate and board.can_claim_threefold_repetition():
            return draw
    if result is None:
        return None
    return outcome_value(result, colour, objective, weight)


class CognitiveAgent(SurvivalAgent):
    # The C++ rules return exactly what the Python ones do (tests/test_fast_rules.py,
    # tests/test_fast_agent.py); False keeps the Python reference path.
    use_fast_rules = fast_rules.ENABLED

    def __init__(self, net, cfg, knowledge, device="cpu", kd_values=None):
        self.knowledge = knowledge
        # {piece_type: pooled K/D relative to a pawn}, from the learner's
        # measurements; blended into piece values with cfg.kd_weight.
        self.kd_values = kd_values
        # v4: the learned King Safety model, when configured (see king_safety_model.py).
        path = getattr(cfg, "king_safety_model", "")
        self.king_model = load_king_safety(path) if path else None
        super().__init__(net, cfg, device)

    def reset(self):
        super().reset()
        self.trace = []
        self.diagnostics = Counter()
        self.plan = ()
        self.plan_prefix = ()
        self._piece_embedding_cache = None

    def _piece_features(self, items, self_color):
        # Embeddings do not depend on the candidate context. Reuse their exact
        # concatenated rows, but invalidate on optimizer/load/device changes.
        layers = (self.net.type_embed, self.net.square_embed, self.net.side_embed)
        stamp = tuple((id(layer.weight), layer.weight._version, layer.weight.device,
                       layer.weight.dtype) for layer in layers)
        if self._piece_embedding_cache is None or self._piece_embedding_cache[0] != stamp:
            type_rows = layers[0].weight[None, :, None, :].expand(2, 6, 64, -1)
            square_rows = layers[1].weight[None, None, :, :].expand(2, 6, 64, -1)
            side_rows = layers[2].weight[:, None, None, :].expand(2, 6, 64, -1)
            table = torch.cat((type_rows, square_rows, side_rows), dim=-1).reshape(768, -1)
            self._piece_embedding_cache = (stamp, table)
        indices = torch.tensor([int(p.color != self_color) * 384 + (p.piece_type - 1) * 64
                                + (sq if self_color else sq ^ 56) for sq, p in items], device=self.device)
        return self._piece_embedding_cache[1][indices]

    def game_over(self, board):
        if self.use_fast_rules:
            return fast_rules.board(board).terminal() != fast_rules.NONE
        return board.is_game_over(claim_draw=True)

    def king_safety_values(self, board):
        """(KS of the white king, KS of the black king) at a real position; None without a model."""
        return None if self.king_model is None else self.king_model.ks_both(board)

    @torch.inference_mode()
    def keys(self, x):
        latent = self.knowledge.encoder(x.to("cpu")).cpu().numpy()
        return structural_keys(x.detach().cpu().numpy(), latent)

    @torch.inference_mode()
    def predict_reply(self, board, self_color):
        if not self.use_fast_rules:
            return super().predict_reply(board, self_color)
        self.observe(board, self_color)
        # Preserve python-chess move order, which defines the reply labels.
        moves = list(board.legal_moves)
        encoded = fast_rules.encode_after_moves(fast_rules.board(board),
                                               [fast_rules.code(m) for m in moves], self_color)
        context, _ = self.net.candidates(torch.from_numpy(encoded).to(self.device), self.memory)
        return moves, self.net.predict_logits(context).softmax(0).cpu().numpy()

    @torch.inference_mode()
    def piece_valuer(self, board, self_color, context=None, items=None):
        """Frozen contextual values shared by protection and capture selection."""
        if context is None:
            self.observe(board, self_color)
            context = self.context
        if items is None:
            items = sorted(board.piece_map().items())
        return self._valuer(self._piece_raw([context], [items], self_color)[0], items, self_color, context)

    @torch.inference_mode()
    def piece_balance(self, board, self_color):
        """Own minus enemy pieces in the agent's contextual pawn units; kings excluded."""
        value = self.piece_valuer(board, self_color)
        return sum(value(piece, square) for square, piece in board.piece_map().items()
                   if piece.piece_type != chess.KING)

    def _piece_raw(self, contexts, item_lists, self_color):
        """Raw piece-head outputs for several positions in ONE forward pass.

        One call per position gave identical values up to the last float bit;
        the batch is several times cheaper and the difference is below any
        decision's resolution.
        """
        sizes = [len(items) for items in item_lists]
        features = torch.cat([torch.cat((context.expand(size, -1), self._piece_features(items, self_color)), dim=-1)
                              for context, items, size in zip(contexts, item_lists, sizes)])
        raw = self.net.piece_head(features).squeeze(-1).cpu().numpy()
        return np.split(raw, np.cumsum(sizes)[:-1])

    def _valuer(self, raw, items, self_color, context):
        pieces = [piece for _, piece in items]
        unit = pawn_unit(raw, pieces)
        signed = raw * np.array([1 if p.color == self_color else -1 for p in pieces])
        normalized = np.clip(signed / unit, -12, 12)
        kd_weight = float(getattr(self.cfg, "kd_weight", 0.0))
        kd = self.kd_values if kd_weight > 0 and self.kd_values else None

        def blend(piece, learned):
            # Owner's rule: kd_weight of the pooled per-type K/D, the rest the
            # network's contextual value. Same sign convention (enemy negative).
            if kd is None or piece.piece_type not in kd:
                return learned
            sign = 1 if piece.color == self_color else -1
            return kd_weight * sign * kd[piece.piece_type] + (1 - kd_weight) * learned

        cache = {(p.color, p.piece_type, sq): float(blend(p, v)) for (sq, p), v in zip(items, normalized)}

        def value(piece, square):
            if piece.piece_type == chess.KING:
                return 10000.0  # King recaptures last; it is never a material victim.
            key = (piece.color, piece.piece_type, square)
            if key not in cache:  # Promotion value at its actual destination.
                predicted = float(self.net.piece_values(context,
                    torch.tensor([piece.piece_type - 1], device=self.device),
                    torch.tensor([square if self_color else square ^ 56], device=self.device),
                    torch.tensor([int(piece.color != self_color)], device=self.device))[0])
                learned = max(-12., min(12., predicted * (1 if piece.color == self_color else -1) / unit))
                cache[key] = float(blend(piece, learned))
            return cache[key]

        return value

    @torch.inference_mode()
    def sense_pieces(self, board, self_color, context=None, value=None):
        value = value or self.piece_valuer(board, self_color, context)
        return piece_senses(board, value, tolerance=self.cfg.piece_exchange_tolerance,
                            depth=self.cfg.piece_exchange_depth)

    @torch.inference_mode()
    def position_proxy(self, board, self_color):
        """The owner's victory measure for a REAL, already observed position.

        0.5 + 0.25 * (enemy king risk - own king risk)
            + 0.25 * (threat to enemy pieces - threat to own pieces),
        with piece values as used in decisions (K/D blend when enabled).
        A finished game overrides it: win 1, draw 0.5, loss 0.
        """
        fast = fast_rules.board(board) if self.use_fast_rules else None
        if fast is not None:
            ending = fast_rules.terminal_value(fast, self_color, "victory")
            if ending is not None:
                return ending
        else:
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                return 0.5 if outcome.winner is None else float(outcome.winner == self_color)
        labels = int(getattr(self.cfg, "mate_labels", 0) or 0)
        if labels and fast_rules.AVAILABLE:
            proven = self._proven_ending(board if fast is None else fast, self_color, labels)
            if proven is not None:
                return proven
        if self.context is None:
            self.observe(board, self_color)
        value = self.piece_valuer(board, self_color, self.context)
        if self.use_fast_rules:
            king = (self.king_model.advantage(board, self_color) if self.king_model is not None
                    else fast_rules.king_balance(fast, self_color))
            own, enemy = self._fast_piece_risks(fast, self_color, self.context, value=value)
            pieces = enemy - own
        else:
            king = (self.king_model.advantage(board, self_color) if self.king_model is not None
                    else king_balance(king_senses(board), self_color))
            senses = self.sense_pieces(board, self_color, self.context, value=value)
            pieces = position_risk(senses, not self_color) - position_risk(senses, self_color)
        exchange_values = self._exchange_values() if getattr(self.cfg, "exchange_labels", False) else None
        if exchange_values is not None:
            exact = fast if fast is not None else fast_rules.board(board)
            gain, complete, _ = exact.exchange_gain(exchange_values, int(self.cfg.exchange_depth),
                                                    int(getattr(self.cfg, "exchange_node_limit", 200_000)))
            if complete:
                pieces = float(np.tanh(gain / 3.0)) * (1 if board.turn == self_color else -1)
                self.diagnostics["exchange_labels"] += 1
        activity = 0.0
        if self.cfg.piece_activity_weight:
            own_pressure, enemy_pressure = position_pressure(board, value, self_color)
            own_moves = legal_mobility(board, self_color)
            enemy_moves = legal_mobility(board, not self_color)
            mobility = (own_moves - enemy_moves) / max(1, own_moves + enemy_moves)
            activity = 0.5 * (enemy_pressure - own_pressure) + 0.5 * mobility * (1 - own_pressure)
        return float(np.clip(0.5 + 0.25 * king + 0.25 * pieces +
                             self.cfg.piece_activity_weight * activity, 0.0, 1.0))

    def _fast_piece_risks(self, fast, self_color, context, pieces=None, value=None):
        """(own, enemy) position_risk of piece_senses, for the C++ board's position."""
        pieces = fast_rules.items(fast) if pieces is None else pieces
        value = value or self.piece_valuer(None, self_color, context, items=pieces)
        white, black = fast_rules.piece_risks(fast, value, pieces, tolerance=self.cfg.piece_exchange_tolerance,
                                              depth=self.cfg.piece_exchange_depth)
        return (white, black) if self_color else (black, white)

    @torch.inference_mode()
    def _robust(self, board, colour, moves, candidate_memory, values, probabilities, root_indices, fast=None):
        """Search legal enemy replies, retaining likely and dangerous branches.

        Full response previews count against the budget. No UCI engine or engine
        evaluation participates in inference. Values at unfinished leaves remain
        network estimates; this is a bounded depth-two search, not a mate proof.
        """
        cfg = self.cfg
        pending = []
        budget, used, changes = cfg.planning_nodes, 0, {}
        root_count = len(root_indices)
        for ordinal, index in enumerate(root_indices):
            if fast is not None:
                fast.push(fast_rules.code(moves[index]))
                ending = fast_rules.terminal_value(fast, colour, cfg.objective, victory_weight(cfg), draw=cfg.draw_value)
                if ending is not None:
                    fast.pop()
                    changes[index] = ending
                    continue
            # Replies and their order depend only on the position, never on history.
            candidate = board.copy(stack=fast is None)
            candidate.push(moves[index])
            if fast is None:
                ending = terminal_value(candidate, colour, cfg.objective, victory_weight(cfg), draw=cfg.draw_value)
                if ending is not None:
                    changes[index] = ending
                    continue
            replies = list(candidate.legal_moves)
            available = (budget - used) // max(1, root_count - ordinal)
            if available < 1:
                if fast is not None:
                    fast.pop()
                break
            # Force/check/capture coverage plus deterministic diversity within the
            # strict node cap. Diagnostics expose any replies left unexplored.
            if len(replies) > available:
                if fast is not None:
                    checks = dict(zip(replies, fast.checks_after([fast_rules.code(move) for move in replies])))
                    replies.sort(key=lambda move: (not checks[move], not candidate.is_capture(move), move.uci()))
                else:
                    replies.sort(key=lambda move: (not candidate.gives_check(move), not candidate.is_capture(move), move.uci()))
                self.diagnostics["unsearched_replies"] += len(replies) - available
                replies = replies[:available]
            if fast is not None:
                encoded = fast_rules.encode_after_moves(fast, [fast_rules.code(r) for r in replies], colour)
            else:
                encoded = encode_after_moves(candidate, replies, colour)
            x = torch.from_numpy(encoded).to(self.device)
            learned = None
            if self.king_model is not None and cfg.king_safety_weight:
                enemy_rows = (fast_rules.encode_after_moves(fast, [fast_rules.code(r) for r in replies], not colour)
                              if fast is not None else encode_after_moves(candidate, replies, not colour))
                learned = self.king_model.ks_rows(encoded) - self.king_model.ks_rows(enemy_rows)
            if getattr(cfg, "batch_search", False):
                reply_values = np.zeros(len(replies), dtype=np.float32)
                overrides, adjustments = {}, {}
            else:
                contexts, _ = self.net.candidates(x, candidate_memory[index:index + 1])
                reply_values = self.net.survival(contexts)[0].mean(-1).cpu().numpy()
                reply_probabilities = self.net.predict_logits(contexts).softmax(0).cpu().numpy()
            for j, reply in enumerate(replies):
                if fast is not None:
                    fast.push(fast_rules.code(reply))
                    ending = fast_rules.terminal_value(fast, colour, cfg.objective, victory_weight(cfg), draw=cfg.draw_value)
                    if ending is not None:
                        reply_values[j] = ending
                        if getattr(cfg, "batch_search", False): overrides[j] = ending
                    elif cfg.king_safety_weight:
                        reply_values[j] += cfg.king_safety_weight * (float(learned[j]) if learned is not None
                                                                     else fast_rules.king_balance(fast, colour))
                        if getattr(cfg, "batch_search", False): adjustments[j] = float(reply_values[j])
                    fast.pop()
                    continue
                candidate.push(reply)
                ending = terminal_value(candidate, colour, cfg.objective, victory_weight(cfg), draw=cfg.draw_value)
                if ending is not None:
                    reply_values[j] = ending
                    if getattr(cfg, "batch_search", False): overrides[j] = ending
                elif cfg.king_safety_weight:
                    reply_values[j] += cfg.king_safety_weight * (float(learned[j]) if learned is not None
                                                                 else king_balance(king_senses(candidate), colour))
                    if getattr(cfg, "batch_search", False): adjustments[j] = float(reply_values[j])
                candidate.pop()
            if fast is not None:
                fast.pop()
            used += len(replies)
            if getattr(cfg, "batch_search", False):
                pending.append((index, x, candidate_memory[index:index + 1].expand(len(replies), -1),
                                overrides, adjustments))
                continue
            # Always include the worst measured reply, even when predicted unlikely.
            selected = list(np.argsort(reply_probabilities)[::-1][:max(1, cfg.planning_replies - 1)])
            worst = int(reply_values.argmin())
            if worst not in selected:
                selected.append(worst)
            selected = np.asarray(selected, dtype=int)
            expected = float(np.average(reply_values[selected], weights=reply_probabilities[selected] + 1e-8))
            continuation = (1 - cfg.robust_weight) * expected + cfg.robust_weight * float(reply_values[selected].min())
            changes[index] = continuation_value(continuation, 1, cfg)
        if pending:
            from .execution_fastpaths import batched_replies
            results = batched_replies(self.net, torch.cat([p[1] for p in pending]),
                                      torch.cat([p[2] for p in pending]), [len(p[1]) for p in pending])
            for (index, _, _, overrides, adjustments), (reply_values, reply_probabilities) in zip(pending, results):
                for j, value in overrides.items(): reply_values[j] = value
                for j, value in adjustments.items(): reply_values[j] += value
                selected = list(np.argsort(reply_probabilities)[::-1][:max(1, cfg.planning_replies - 1)])
                worst = int(reply_values.argmin())
                if worst not in selected: selected.append(worst)
                selected = np.asarray(selected, dtype=int)
                expected = float(np.average(reply_values[selected], weights=reply_probabilities[selected] + 1e-8))
                continuation = (1 - cfg.robust_weight) * expected + cfg.robust_weight * float(reply_values[selected].min())
                changes[index] = continuation_value(continuation, 1, cfg)
        self.diagnostics["planning_nodes"] += used
        self.diagnostics["planned_decisions"] += bool(changes)
        return changes

    @torch.inference_mode()
    def evaluate(self, board, self_color, *, explore=True, head=None):
        self.observe(board, self_color)
        moves = list(board.legal_moves)
        if not moves:
            raise ValueError("Cannot select a move in a terminal position")
        fast = fast_rules.board(board) if self.use_fast_rules else None
        codes = [fast_rules.code(move) for move in moves] if fast is not None else None
        encoded = (fast_rules.encode_after_moves(fast, codes, self_color) if fast is not None
                   else encode_after_moves(board, moves, self_color))
        x = torch.from_numpy(encoded).to(self.device)
        # candidates() and keys() both encode x with the same module; do it once.
        if getattr(self.cfg, "incremental_encoder", False):
            from .execution_fastpaths import incremental_encode
            root = torch.from_numpy(encode(board, self_color)).to(self.device)
            latent = incremental_encode(self.net.encoder, root, x)
        else:
            latent = self.net.encode(x)
        memories = self.net.memory_cell(latent, self.net._memory_for_batch(self.memory, len(x)))
        contexts = self.net.context_from_latent(latent, memories)
        heads, logvars = self.net.survival(contexts)
        mean = heads.mean(-1).cpu().numpy()
        epistemic = heads.std(-1, unbiased=False).cpu().numpy()
        aleatoric = logvars.exp().mean(-1).sqrt().cpu().numpy()
        raw = heads[:, head].cpu().numpy() if explore and head is not None else mean.copy()
        if self.knowledge.encoder is self.net.encoder:
            keys = structural_keys(x.detach().cpu().numpy(), latent.cpu().numpy())
        else:
            keys = self.keys(x)
        retrieved = self.knowledge.memory.query(keys, own_turn=False)
        strength = (self.cfg.memory_weight * retrieved["trust"] * retrieved["similarity"]
                    * retrieved["support"] / (retrieved["support"] + 2))
        scores = (1 - strength) * raw + strength * retrieved["value"]
        uncertainty = self.knowledge.calibration.uncertainty(mean, epistemic)
        evidence = self.knowledge.calibration.count[self.knowledge.calibration._bin(mean)]
        uncertainty = np.where(evidence < self.cfg.calibration_min_samples,
                               np.maximum(uncertainty, self.cfg.planning_uncertainty), uncertainty)
        # Seek reducible uncertainty: evidence counts/conflicting outcomes suppress
        # repeated surprise, and empirical residual alone never gives a bonus.
        information = retrieved["information"] * np.minimum(1, epistemic / .1)
        if explore:
            scores += self.cfg.information_bonus * information
        safety, kings = [], []
        king_bonuses = np.zeros(len(moves))
        activity_bonuses = np.zeros(len(moves))
        exchange_values = self._exchange_values()
        nets = (self._exchange_nets(board, moves, codes, fast, exchange_values)
                if exchange_values is not None and self.cfg.piece_safety_weight else None)
        if self.cfg.piece_safety_weight or self.cfg.piece_activity_weight:
            branches, item_lists = [], []
            for index, move in enumerate(moves):
                if fast is not None:
                    fast.push(codes[index])
                    item_lists.append(fast_rules.items(fast))
                    fast.pop()
                else:
                    branch = board.copy(stack=False)
                    branch.push(move)
                    branches.append(branch)
                    item_lists.append(sorted(branch.piece_map().items()))
            row_contexts = [contexts[index:index + 1] for index in range(len(moves))]
            raws = self._piece_raw(row_contexts, item_lists, self_color)
            previous_mobility = legal_mobility(board, self_color) if self.cfg.piece_activity_weight else None
            for index, move in enumerate(moves):
                value = self._valuer(raws[index], item_lists[index], self_color, row_contexts[index])
                branch = branches[index] if fast is None else None
                if self.cfg.piece_safety_weight and nets is not None:
                    # Exact: the forced capture sequence's material result for this move.
                    own_risk = enemy_risk = 0.0
                    scores[index] += self.cfg.piece_safety_weight * float(np.tanh(nets[index] / 3.0))
                elif self.cfg.piece_safety_weight:
                    if fast is not None:
                        fast.push(codes[index])
                        own_risk, enemy_risk = self._fast_piece_risks(fast, self_color, row_contexts[index],
                                                                      item_lists[index], value)
                        fast.pop()
                    else:
                        senses = self.sense_pieces(branch, self_color, value=value)
                        own_risk = position_risk(senses, self_color)
                        enemy_risk = position_risk(senses, not self_color)
                    scores[index] += self.cfg.piece_safety_weight * (enemy_risk - own_risk)
                else:
                    own_risk = enemy_risk = 0.0
                row = {"own_risk": own_risk, "enemy_risk": enemy_risk}
                if nets is not None:
                    row["exchange_net"] = nets[index]
                if self.cfg.piece_activity_weight:
                    if branch is None:
                        branch = board.copy(stack=False)
                        branch.push(move)
                    own_pressure, enemy_pressure = position_pressure(branch, value, self_color)
                    threat = favorable_attack(branch, move.to_square, self_color, value)
                    mobility = protected_mobility_gain(board, branch, move.to_square, self_color, value,
                                                       previous_mobility)
                    protection = float(np.clip(enemy_pressure - own_pressure + threat, -1, 1))
                    activity_bonuses[index] = self.cfg.piece_activity_weight * (0.5 * protection + 0.5 * mobility)
                    scores[index] += activity_bonuses[index]
                    row.update(own_pressure=own_pressure, enemy_pressure=enemy_pressure,
                               favorable_attack=threat, safe_mobility=mobility)
                safety.append(row)
            self.diagnostics["piece_safety_positions"] += len(moves)
        if self.cfg.king_safety_weight and self.king_model is not None:
            enemy_rows = (fast_rules.encode_after_moves(fast, codes, not self_color) if fast is not None
                          else encode_after_moves(board, moves, not self_color))
            ks_self, ks_enemy = self.king_model.ks_rows(encoded), self.king_model.ks_rows(enemy_rows)
            king_bonuses = self.cfg.king_safety_weight * (ks_self - ks_enemy)
            for own, enemy in zip(ks_self, ks_enemy):
                white, black = (own, enemy) if self_color else (enemy, own)
                kings.append({"w": {"risk": 1.0 - float(white), "ks": float(white)},
                              "b": {"risk": 1.0 - float(black), "ks": float(black)}})
            scores += king_bonuses
            self.diagnostics["king_safety_positions"] += len(moves)
        elif self.cfg.king_safety_weight:
            for index, move in enumerate(moves):
                if fast is not None:
                    # Risks only; the chosen move's full senses are rebuilt for its trace.
                    fast.push(codes[index])
                    white, black = fast.king_risks()
                    fast.pop()
                    senses = {"w": {"risk": white}, "b": {"risk": black}}
                    king_bonuses[index] = self.cfg.king_safety_weight * king_balance(senses, self_color)
                    kings.append(senses)
                    continue
                branch = board.copy(stack=False)
                branch.push(move)
                senses = king_senses(branch)
                king_bonuses[index] = self.cfg.king_safety_weight * king_balance(senses, self_color)
                kings.append(senses)
            scores += king_bonuses
            self.diagnostics["king_safety_positions"] += len(moves)
        allows_mate = {}
        defence = int(getattr(self.cfg, "mate_defense_moves", 0) or 0)
        if defence > 0 and fast_rules.AVAILABLE:
            allows_mate = self._moves_allowing_mate(board, self_color, moves, codes, fast, defence)
            self._rank_last(scores, allows_mate)
        order = np.argsort(scores)[::-1]
        gap = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else 1.0
        danger = mean[order[0]] < .35 or board.is_check()
        uncertain = uncertainty[order[0]] >= self.cfg.planning_uncertainty
        king_pressure = bool(kings) and any(row['risk'] > .1 for row in kings[order[0]].values())
        if danger or uncertain or gap <= self.cfg.planning_gap or king_pressure:
            root_indices = order[:self.cfg.planning_candidates]
            revised = self._robust(board, self_color, moves, memories, mean, None, root_indices, fast)
            for index, value in revised.items():
                # Replace immediate king pressure by pressure AFTER the reply;
                # do not keep rewarding an attack the opponent has neutralized.
                scores[index] += value - mean[index] - king_bonuses[index] - activity_bonuses[index]
            if revised:
                # Compare searched choices on the same basis. Otherwise pessimistic
                # search would merely make an untouched optimistic candidate win.
                unsearched = np.ones(len(moves), dtype=bool)
                unsearched[list(revised)] = False
                scores[unsearched] = -1e9
        else:
            self.diagnostics["fast_decisions"] += 1
        # Known endings override noisy critics and intrinsic incentives.
        if safety or kings or self.cfg.objective in ("victory", "shaped"):
            scores = np.minimum(scores, .999999)
        drawn = []   # moves whose ending is a draw (for the repetition rule)
        draw_value = terminal_draw_value(self.cfg.objective, victory_weight(self.cfg), draw=self.cfg.draw_value)
        for index, move in enumerate(moves):
            if fast is not None:
                fast.push(codes[index])
                ending = fast_rules.terminal_value(fast, self_color, self.cfg.objective, victory_weight(self.cfg), draw=self.cfg.draw_value)
                fast.pop()
            else:
                branch = board.copy(stack=True)
                branch.push(move)
                ending = terminal_value(branch, self_color, self.cfg.objective, victory_weight(self.cfg), draw=self.cfg.draw_value)
            if ending is not None:
                scores[index] = ending
                if ending == draw_value:
                    drawn.append(index)
        margin = float(getattr(self.cfg, "avoid_repetition_margin", 0.0) or 0.0)
        if margin > 0 and drawn:
            self._avoid_repetition_when_ahead(board, self_color, moves, scores, drawn, margin)
        if allows_mate:
            self._rank_last(scores, allows_mate)
        self._evaluation = {"keys": keys, "retrieved": retrieved, "uncertainty": uncertainty,
                            "information": information, "gap": gap, "piece_safety": safety, "king_safety": kings,
                            "fast": fast, "codes": codes, "allows_mate": allows_mate, "exchange_nets": nets}
        return SurvivalChoice(moves, scores, mean, epistemic, aleatoric, information)

    def _moves_allowing_mate(self, board, self_color, moves, codes, fast, depth):
        """Mate calculation, rules only: {index: n} for the moves after which the
        opponent forces checkmate in n <= depth moves. Moves that end the game
        keep their known value."""
        fast = fast_rules.board(board) if fast is None else fast
        codes = [fast_rules.code(move) for move in moves] if codes is None else codes
        limit = int(getattr(self.cfg, "mate_node_limit", 20_000))
        allows = {}
        for index, code in enumerate(codes):
            fast.push(code)
            if fast.terminal() == fast_rules.NONE:
                found = fast.forced_mate(not self_color, depth, limit)[0]
                if found:
                    allows[index] = found
            fast.pop()
        self.diagnostics["mate_defence_searches"] += len(codes)
        if len(allows) < len(codes):                      # a safe move exists: these are avoided
            self.diagnostics["mate_moves_avoided"] += len(allows)
        return allows

    @staticmethod
    def _rank_last(scores, allows_mate):
        """A forced loss ranks under every move the planner compared (-1e9 marks
        the ones it left out); among forced losses a later mate ranks higher, so
        such a move is played only when every move allows mate."""
        others = np.ones(len(scores), dtype=bool)
        others[list(allows_mate)] = False
        others &= scores > -1e8
        floor = float(scores[others].min()) - 1.0 if others.any() else 0.0
        for index, found in allows_mate.items():
            if scores[index] > -1e8:
                scores[index] = floor + 1e-3 * found

    def _exchange_values(self):
        """[0, pawn, knight, bishop, rook, queen] in TonkAI's measured values, or
        None when the exact exchange search is off or no values are measured."""
        if int(getattr(self.cfg, "exchange_depth", 0) or 0) <= 0 or not self.kd_values or not fast_rules.AVAILABLE:
            return None
        return [0.0] + [float(self.kd_values[kind]) for kind in range(1, 6)]

    def _exchange_nets(self, board, moves, codes, fast, values):
        """Per move: material it takes minus what the opponent then wins by force
        (exact capture search), in TonkAI's pawn units; None if a search aborts."""
        fast = fast_rules.board(board) if fast is None else fast
        codes = [fast_rules.code(move) for move in moves] if codes is None else codes
        try:
            after = fast.exchange_after_moves(codes, values, int(self.cfg.exchange_depth),
                                              int(getattr(self.cfg, "exchange_node_limit", 200_000)))
        except RuntimeError:
            self.diagnostics["exchange_aborted"] += 1
            return None
        nets = []
        for move, loss in zip(moves, after):
            gain = 0.0
            if board.is_capture(move):
                gain = values[chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)]
            if move.promotion:
                gain += values[move.promotion] - values[chess.PAWN]
            nets.append(gain - loss)
        self.diagnostics["exchange_searches"] += len(moves)
        return nets

    def _proven_ending(self, position, self_color, depth):
        """Rules-only truth for the victory measure: 1 if TonkAI forces mate
        within `depth` moves, 0 if the opponent does, else None."""
        fast = fast_rules.board(position) if isinstance(position, chess.Board) else position
        limit = int(getattr(self.cfg, "mate_node_limit", 20_000))
        if fast.forced_mate(self_color, depth, limit)[0]:
            self.diagnostics["mate_labels_won"] += 1
            return 1.0
        if fast.forced_mate(not self_color, depth, limit)[0]:
            self.diagnostics["mate_labels_lost"] += 1
            return 0.0
        return None

    def _trade_move(self, board, choice, rng):
        """Exploration: a capture the exact exchange search does not lose
        (net >= 0), sampled by TonkAI's own scores; None if there is none."""
        nets = self._evaluation.get("exchange_nets")
        if not nets:
            return None
        vetoed = self._evaluation.get("allows_mate") or {}
        candidates = [index for index, move in enumerate(choice.moves)
                      if board.is_capture(move) and nets[index] >= 0 and index not in vetoed]
        if not candidates:
            return None
        scores = choice.scores[candidates]
        weights = np.exp((scores - scores.max()) / max(1e-6, float(self.cfg.temperature)))
        self.diagnostics["trades_explored"] += 1
        return candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))]

    def _threat_move(self, board, self_color, choice, rng):
        """Exploration: index of a move that creates a mate threat (were the
        opponent to pass, TonkAI would force mate within threat_moves), sampled by
        TonkAI's own scores among those moves; None when there is none. Moves that
        allow the opponent a forced mate are never tried."""
        fast = self._evaluation.get("fast")
        fast = fast_rules.board(board) if fast is None else fast
        codes = self._evaluation.get("codes")
        codes = [fast_rules.code(move) for move in choice.moves] if codes is None else codes
        position = {code: index for index, code in enumerate(codes)}
        found, depths, _ = fast.threats(int(getattr(self.cfg, "threat_moves", 1)),
                                        int(getattr(self.cfg, "mate_node_limit", 20_000)))
        vetoed = self._evaluation.get("allows_mate") or {}
        candidates = [position[code] for code, depth in zip(found, depths)
                      if depth and code in position and position[code] not in vetoed]
        if not candidates:
            return None
        scores = choice.scores[candidates]
        weights = np.exp((scores - scores.max()) / max(1e-6, float(self.cfg.temperature)))
        self.diagnostics["threat_moves_explored"] += 1
        return candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))]

    def _mating_line(self, board, self_color, moves, depth):
        """Mate calculation, rules only: (indices of the moves on a shortest
        forced mate within `depth` own moves, its length), ([], 0) if none."""
        fast = self._evaluation.get("fast")
        fast = fast_rules.board(board) if fast is None else fast
        codes = self._evaluation.get("codes")
        codes = [fast_rules.code(move) for move in moves] if codes is None else codes
        limit = int(getattr(self.cfg, "mate_node_limit", 20_000))
        found = fast.forced_mate(self_color, depth, limit)[0]
        self.diagnostics["mate_attack_searches"] += 1
        line = []
        if found:
            mated = fast_rules.WHITE_WINS if self_color else fast_rules.BLACK_WINS
            for index, code in enumerate(codes):
                fast.push(code)
                ending = fast.terminal()                  # a draw (e.g. a repetition) ends the line
                if ending == mated or (ending == fast_rules.NONE and found > 1
                                       and fast.forced_mate(self_color, found - 1, limit)[0]):
                    line.append(index)
                fast.pop()
        if not line:
            return [], 0
        self.diagnostics["forced_mate_moves"] += 1
        return line, found

    def _avoid_repetition_when_ahead(self, board, self_color, moves, scores, drawn, margin):
        """Owner's rule: ahead by `margin` pawns (own piece values) -> a repetition draw is nearly worthless."""
        repeated = []
        for index in drawn:
            branch = board.copy(stack=True)
            branch.push(moves[index])
            if branch.can_claim_threefold_repetition() or branch.is_repetition(5):
                repeated.append(index)
        if not repeated:
            return
        balance = self.piece_balance(board, self_color)
        if balance >= margin:
            for index in repeated:
                scores[index] = min(scores[index], 0.05)
            self.diagnostics["repetitions_avoided"] += len(repeated)

    @torch.inference_mode()
    def choose(self, board, self_color, rng, *, temperature=None, explore=True, head=None):
        choice = self.evaluate(board, self_color, explore=explore, head=head)
        position_x = torch.from_numpy(encode(board, self_color)[None])
        position_key = self.keys(position_x)[0]
        history = tuple(move.uci() for move in board.move_stack)
        expected = self.plan_prefix + self.plan
        consumed = len(history) - len(self.plan_prefix)
        if not self.plan or consumed < 0 or history != expected[:len(history)] or consumed >= len(self.plan):
            self.plan = self.knowledge.skills.retrieve(position_key, board, self_color,
                                                      fast=self._evaluation.get("fast"))
            self.plan_prefix = history
            consumed = 0
        if self.plan:
            suggested = chess.Move.from_uci(self.plan[consumed])
            if suggested in choice.moves:
                index = choice.moves.index(suggested)
                # Learned plans only break near-ties; they cannot force a bad move.
                if choice.scores[index] >= choice.scores.max() - self.cfg.skill_bonus:
                    choice.scores[index] += self.cfg.skill_bonus
                    self.diagnostics["skill_uses"] += 1
                else:
                    self.plan = ()
            else:
                self.plan = ()
        temperature = self.cfg.temperature if temperature is None else temperature
        winning = []
        if self.cfg.objective in ("victory", "shaped"):
            # A real checkmate is the maximum under both; always take it.
            fast = self._evaluation["fast"]
            for index, move in enumerate(choice.moves):
                if fast is not None:
                    fast.push(self._evaluation["codes"][index])
                    wins = fast_rules.terminal_value(fast, self_color, "victory") == 1
                    fast.pop()
                else:
                    branch = board.copy(stack=True)
                    branch.push(move)
                    wins = terminal_value(branch, self_color, "victory") == 1
                if wins:
                    winning.append(index)
        attack = int(getattr(self.cfg, "mate_search_moves", 0) or 0)
        if not winning and attack > 0 and fast_rules.AVAILABLE:
            winning, choice.forced_mate = self._mating_line(board, self_color, choice.moves, attack)
        threat = None
        share = float(getattr(self.cfg, "threat_explore", 0.0) or 0.0)
        if not winning and explore and share > 0 and fast_rules.AVAILABLE and rng.random() < share:
            threat = self._threat_move(board, self_color, choice, rng)
        trade = None
        trade_share = float(getattr(self.cfg, "trade_explore", 0.0) or 0.0)
        if not winning and threat is None and explore and trade_share > 0 and rng.random() < trade_share:
            trade = self._trade_move(board, choice, rng)
        if winning:
            choice.index = winning[0]
        elif threat is not None:
            choice.index = threat
        elif trade is not None:
            choice.index = trade
        elif explore and temperature > 0:
            weights = np.exp((choice.scores - choice.scores.max()) / temperature)
            choice.index = int(rng.choice(len(weights), p=weights / weights.sum()))
        else:
            choice.index = int(choice.scores.argmax())
        current_value = float(self.net.survival(self.context)[0].mean())
        self.trace.append({"ply": len(board.move_stack), "fen": board.fen(), "key": position_key,
                           "prediction": current_value, "move": choice.move.uci(),
                           "next_key": self._evaluation["keys"][choice.index],
                           "next_prediction": float(choice.mean[choice.index]),
                           "uncertainty": float(self._evaluation["uncertainty"][choice.index]),
                           "gap": self._evaluation["gap"]})
        if choice.forced_mate:
            self.trace[-1]["forced_mate"] = choice.forced_mate
        if self._evaluation.get("piece_safety"):
            self.trace[-1]["piece_safety"] = self._evaluation["piece_safety"][choice.index]
        if self._evaluation.get("king_safety"):
            senses = self._evaluation["king_safety"][choice.index]
            if self._evaluation["fast"] is not None and self.king_model is None:
                branch = board.copy(stack=False)
                branch.push(choice.move)
                senses = king_senses(branch)
            self.trace[-1]["king_safety"] = senses
        self.diagnostics["retrieval_hits"] += int(self._evaluation["retrieved"]["support"][choice.index] > 0)
        return choice
