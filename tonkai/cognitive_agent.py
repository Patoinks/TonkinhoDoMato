"""V3 policy: retrieval, evidence-seeking exploration and bounded robust search."""
from collections import Counter

import chess
import numpy as np
import torch

from .encoding import encode, encode_after_moves
from .survival_agent import SurvivalAgent, SurvivalChoice
from .cognitive_memory import structural_keys
from .piece_safety import piece_senses, pawn_unit, position_risk
from .king_safety import king_senses, king_balance
from .objective import outcome_value, continuation_value, victory_weight


def terminal_value(board, colour, objective="survival", weight=0.5):
    draw = .5 if objective == "victory" else 1.0 - weight / 2 if objective == "shaped" else 1.0
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
    def __init__(self, net, cfg, knowledge, device="cpu", kd_values=None):
        self.knowledge = knowledge
        # {piece_type: pooled K/D relative to a pawn}, from the learner's
        # measurements; blended into piece values with cfg.kd_weight.
        self.kd_values = kd_values
        super().__init__(net, cfg, device)

    def reset(self):
        super().reset()
        self.trace = []
        self.diagnostics = Counter()
        self.plan = ()
        self.plan_prefix = ()

    @torch.inference_mode()
    def keys(self, x):
        latent = self.knowledge.encoder(x.to("cpu")).cpu().numpy()
        return structural_keys(x.detach().cpu().numpy(), latent)

    @torch.inference_mode()
    def piece_valuer(self, board, self_color, context=None):
        """Frozen contextual values shared by protection and capture selection."""
        if context is None:
            self.observe(board, self_color)
            context = self.context
        items = sorted(board.piece_map().items())
        pieces = [piece for _, piece in items]
        types = torch.tensor([p.piece_type - 1 for p in pieces], device=self.device)
        squares = torch.tensor([sq if self_color else sq ^ 56 for sq, _ in items], device=self.device)
        sides = torch.tensor([int(p.color != self_color) for p in pieces], device=self.device)
        raw = self.net.piece_values(context.expand(len(items), -1), types, squares, sides).cpu().numpy()
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
    def sense_pieces(self, board, self_color, context=None):
        return piece_senses(board, self.piece_valuer(board, self_color, context), tolerance=self.cfg.piece_exchange_tolerance,
                            depth=self.cfg.piece_exchange_depth)

    @torch.inference_mode()
    def position_proxy(self, board, self_color):
        """The owner's victory measure for a REAL, already observed position.

        0.5 + 0.25 * (enemy king risk - own king risk)
            + 0.25 * (threat to enemy pieces - threat to own pieces),
        with piece values as used in decisions (K/D blend when enabled).
        A finished game overrides it: win 1, draw 0.5, loss 0.
        """
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return 0.5 if outcome.winner is None else float(outcome.winner == self_color)
        if self.context is None:
            self.observe(board, self_color)
        king = king_balance(king_senses(board), self_color)
        senses = self.sense_pieces(board, self_color, self.context)
        pieces = position_risk(senses, not self_color) - position_risk(senses, self_color)
        return float(np.clip(0.5 + 0.25 * king + 0.25 * pieces, 0.0, 1.0))

    @torch.inference_mode()
    def _robust(self, board, colour, moves, candidate_memory, values, probabilities, root_indices):
        """Search legal enemy replies, retaining likely and dangerous branches.

        Full response previews count against the budget. No UCI engine or engine
        evaluation participates in inference. Values at unfinished leaves remain
        network estimates; this is a bounded depth-two search, not a mate proof.
        """
        cfg = self.cfg
        budget, used, changes = cfg.planning_nodes, 0, {}
        root_count = len(root_indices)
        for ordinal, index in enumerate(root_indices):
            candidate = board.copy(stack=True)
            candidate.push(moves[index])
            ending = terminal_value(candidate, colour, cfg.objective, victory_weight(cfg))
            if ending is not None:
                changes[index] = ending
                continue
            replies = list(candidate.legal_moves)
            available = (budget - used) // max(1, root_count - ordinal)
            if available < 1:
                break
            # Force/check/capture coverage plus deterministic diversity within the
            # strict node cap. Diagnostics expose any replies left unexplored.
            if len(replies) > available:
                replies.sort(key=lambda move: (not candidate.gives_check(move), not candidate.is_capture(move), move.uci()))
                self.diagnostics["unsearched_replies"] += len(replies) - available
                replies = replies[:available]
            x = torch.from_numpy(encode_after_moves(candidate, replies, colour)).to(self.device)
            contexts, _ = self.net.candidates(x, candidate_memory[index:index + 1])
            reply_values = self.net.survival(contexts)[0].mean(-1).cpu().numpy()
            reply_probabilities = self.net.predict_logits(contexts).softmax(0).cpu().numpy()
            for j, reply in enumerate(replies):
                candidate.push(reply)
                ending = terminal_value(candidate, colour, cfg.objective, victory_weight(cfg))
                if ending is not None:
                    reply_values[j] = ending
                elif cfg.king_safety_weight:
                    reply_values[j] += cfg.king_safety_weight * king_balance(king_senses(candidate), colour)
                candidate.pop()
            used += len(replies)
            # Always include the worst measured reply, even when predicted unlikely.
            selected = list(np.argsort(reply_probabilities)[::-1][:max(1, cfg.planning_replies - 1)])
            worst = int(reply_values.argmin())
            if worst not in selected:
                selected.append(worst)
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
        x = torch.from_numpy(encode_after_moves(board, moves, self_color)).to(self.device)
        contexts, memories = self.net.candidates(x, self.memory)
        heads, logvars = self.net.survival(contexts)
        mean = heads.mean(-1).cpu().numpy()
        epistemic = heads.std(-1, unbiased=False).cpu().numpy()
        aleatoric = logvars.exp().mean(-1).sqrt().cpu().numpy()
        raw = heads[:, head].cpu().numpy() if explore and head is not None else mean.copy()
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
        if self.cfg.piece_safety_weight:
            for index, move in enumerate(moves):
                branch = board.copy(stack=False)
                branch.push(move)
                senses = self.sense_pieces(branch, self_color, contexts[index:index + 1])
                own_risk = position_risk(senses, self_color)
                enemy_risk = position_risk(senses, not self_color)
                scores[index] += self.cfg.piece_safety_weight * (enemy_risk - own_risk)
                safety.append({"own_risk": own_risk, "enemy_risk": enemy_risk})
            self.diagnostics["piece_safety_positions"] += len(moves)
        if self.cfg.king_safety_weight:
            for index, move in enumerate(moves):
                branch = board.copy(stack=False)
                branch.push(move)
                senses = king_senses(branch)
                king_bonuses[index] = self.cfg.king_safety_weight * king_balance(senses, self_color)
                kings.append(senses)
            scores += king_bonuses
            self.diagnostics["king_safety_positions"] += len(moves)
        order = np.argsort(scores)[::-1]
        gap = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else 1.0
        danger = mean[order[0]] < .35 or board.is_check()
        uncertain = uncertainty[order[0]] >= self.cfg.planning_uncertainty
        king_pressure = bool(kings) and any(row['risk'] > .1 for row in kings[order[0]].values())
        if danger or uncertain or gap <= self.cfg.planning_gap or king_pressure:
            root_indices = order[:self.cfg.planning_candidates]
            revised = self._robust(board, self_color, moves, memories, mean, None, root_indices)
            for index, value in revised.items():
                # Replace immediate king pressure by pressure AFTER the reply;
                # do not keep rewarding an attack the opponent has neutralized.
                scores[index] += value - mean[index] - king_bonuses[index]
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
        for index, move in enumerate(moves):
            branch = board.copy(stack=True)
            branch.push(move)
            ending = terminal_value(branch, self_color, self.cfg.objective, victory_weight(self.cfg))
            if ending is not None:
                scores[index] = ending
        self._evaluation = {"keys": keys, "retrieved": retrieved, "uncertainty": uncertainty,
                            "information": information, "gap": gap, "piece_safety": safety, "king_safety": kings}
        return SurvivalChoice(moves, scores, mean, epistemic, aleatoric, information)

    @torch.inference_mode()
    def choose(self, board, self_color, rng, *, temperature=None, explore=True, head=None):
        choice = self.evaluate(board, self_color, explore=explore, head=head)
        position_x = torch.from_numpy(encode(board, self_color)[None])
        position_key = self.keys(position_x)[0]
        history = tuple(move.uci() for move in board.move_stack)
        expected = self.plan_prefix + self.plan
        consumed = len(history) - len(self.plan_prefix)
        if not self.plan or consumed < 0 or history != expected[:len(history)] or consumed >= len(self.plan):
            self.plan = self.knowledge.skills.retrieve(position_key, board, self_color)
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
            for index, move in enumerate(choice.moves):
                branch = board.copy(stack=True)
                branch.push(move)
                if terminal_value(branch, self_color, "victory") == 1:
                    winning.append(index)
        if winning:
            choice.index = winning[0]
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
        if self._evaluation.get("piece_safety"):
            self.trace[-1]["piece_safety"] = self._evaluation["piece_safety"][choice.index]
        if self._evaluation.get("king_safety"):
            self.trace[-1]["king_safety"] = self._evaluation["king_safety"][choice.index]
        self.diagnostics["retrieval_hits"] += int(self._evaluation["retrieved"]["support"][choice.index] > 0)
        return choice
