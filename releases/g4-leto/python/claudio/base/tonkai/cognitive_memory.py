"""Bounded episodic memory, empirical error calibration and learned short skills.

All evidence comes from completed, real trajectories before learning on them.
Search rollouts never masquerade as historical outcomes or confidence evidence.
"""
from copy import deepcopy

import chess
import numpy as np


def normalized(vector):
    vector = np.asarray(vector, dtype=np.float32)
    return vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), 1e-8)


def structural_keys(encoded, latent):
    """Pooled occupancy + exact side/castling/clock features + fixed latent.

    The encoder supplying `latent` is frozen for the entire V3 lifetime, so old
    memories and new queries remain in the same coordinate system.
    """
    x = np.asarray(encoded)
    planes = x[:, :768].reshape(-1, 12, 4, 2, 4, 2).mean(axis=(3, 5)).reshape(len(x), -1)
    return normalized(np.concatenate((normalized(planes), x[:, 768:] * 0.5,
                                     normalized(latent) * 0.5), axis=-1))


def perspective_action(uci, colour):
    move = chess.Move.from_uci(uci)
    if colour == chess.BLACK:
        move = chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), move.promotion)
    return move.uci()


class ErrorCalibration:
    """Held-out sequential residuals of a discounted value (not a win probability)."""
    def __init__(self, bins=12):
        self.count = np.zeros(bins, dtype=np.int64)
        self.square_error = np.zeros(bins, dtype=np.float64)
        self.absolute_error = np.zeros(bins, dtype=np.float64)

    def _bin(self, values):
        return np.minimum((np.clip(values, 0, 1) * len(self.count)).astype(int), len(self.count) - 1)

    def observe(self, predictions, outcomes):
        p, y = np.asarray(predictions), np.asarray(outcomes)
        if p.shape != y.shape or not np.isfinite(p).all() or not np.isfinite(y).all():
            raise ValueError("Calibration needs paired finite predictions and outcomes")
        bins = self._bin(p)
        np.add.at(self.count, bins, 1)
        np.add.at(self.square_error, bins, (p - y) ** 2)
        np.add.at(self.absolute_error, bins, np.abs(p - y))

    def uncertainty(self, values, disagreement):
        indices = self._bin(np.asarray(values))
        # Conservative prior decays with evidence; agreement alone is insufficient.
        residual = np.sqrt((self.square_error[indices] + 0.25) / (self.count[indices] + 1))
        return np.maximum(np.asarray(disagreement), residual).astype(np.float32)

    def state_dict(self):
        return {name: getattr(self, name).copy() for name in ("count", "square_error", "absolute_error")}

    def load_state_dict(self, state):
        for name, value in state.items():
            setattr(self, name, np.array(value, copy=True))


class ExperienceMemory:
    def __init__(self, cfg):
        self.cfg = cfg
        self.entries = []
        self.episodes = []
        self.next_id = 0
        self._keys = None
        self._retention = None

    def _matrix(self):
        if self._keys is None and self.entries:
            self._keys = np.stack([entry["key"] for entry in self.entries])
            # Rebuilt together with the keys: entries are replaced, never edited in
            # place, except trust (audits), which is read live below.
            self._turns = np.array([entry.get("own_turn") is not None for entry in self.entries]), \
                np.array([bool(entry.get("own_turn")) for entry in self.entries])
        return self._keys

    def query(self, keys, own_turn=None):
        keys = np.atleast_2d(keys)
        result = {name: np.zeros(len(keys), dtype=np.float32) for name in
                  ("value", "support", "similarity", "variance", "trust", "information")}
        result["ids"] = [[] for _ in keys]
        if not self.entries:
            result["information"].fill(1.0)
            return result
        similarities = keys @ self._matrix().T
        if own_turn is not None:
            known, turns = self._turns
            similarities[:, known & (turns != bool(own_turn))] = -1
        # One call for every row; row-wise it is the same sort as np.argsort(values).
        orders = np.argsort(similarities, axis=1)[:, -self.cfg.memory_neighbors:][:, ::-1]
        for row, values in enumerate(similarities):
            best = orders[row]
            best = best[values[best] >= self.cfg.memory_similarity]
            if not len(best):
                result["information"][row] = 1
                continue
            entries = [self.entries[i] for i in best]
            weights = np.array([max(0.01, values[i] - self.cfg.memory_similarity + .01)
                                * entry["trust"] for i, entry in zip(best, entries)])
            targets = np.array([entry["target"] for entry in entries])
            # np.average performs these same reductions for each statistic.
            # Reuse the denominator without changing dtype or summation order.
            dtype = np.result_type(targets.dtype, weights.dtype)
            total = weights.sum(dtype=dtype)
            if total == 0:
                raise ZeroDivisionError("Weights sum to zero, can't be normalized")
            mean = float(np.multiply(targets, weights, dtype=dtype).sum() / total)
            variance = float(np.multiply((targets - mean) ** 2, weights, dtype=dtype).sum() / total)
            # Repeated examples from one game do not pretend to be independent evidence.
            support = len({entry["game"] for entry in entries})
            result["value"][row], result["variance"][row] = mean, variance
            result["support"][row], result["similarity"][row] = support, float(values[best[0]])
            trust = np.array([entry["trust"] for entry in entries])
            trust_dtype = np.result_type(trust.dtype, weights.dtype)
            trust_total = total if trust_dtype == dtype else weights.sum(dtype=trust_dtype)
            result["trust"][row] = float(np.multiply(trust, weights, dtype=trust_dtype).sum() / trust_total)
            # Local expected evidence gain proxy: shrinks with independent experience
            # and conflicting outcomes, rather than rewarding raw prediction noise.
            result["information"][row] = 1 / ((support + 1) * (1 + 8 * variance))
            result["ids"][row] = [entry["id"] for entry in entries]
        return result

    def add(self, key, target, prediction, game, fen, actions=(), colour=True, own_turn=None):
        nearest = self.query(np.asarray(key)[None], own_turn=own_turn)
        rarity = 1 - float(nearest["similarity"][0])
        priority = min(2.0, abs(float(target) - float(prediction)) + rarity + .05)
        entry = {"id": self.next_id, "key": np.array(key, copy=True), "target": float(target),
                 "prediction": float(prediction), "game": int(game), "fen": fen, "actions": tuple(actions),
                 "colour": bool(colour), "own_turn": own_turn, "priority": priority, "trust": 1.0, "audits": 0}
        self.next_id += 1
        if len(self.entries) >= self.cfg.archive_capacity:
            # Protect high-error/rare memories; discount redundant neighbours.
            # Rejected candidates do not change the archive. Its pairwise
            # redundancy and priorities remain exactly the same until insertion.
            if self._retention is None:
                matrix = self._matrix()
                duplicate = matrix @ matrix.T
                np.fill_diagonal(duplicate, -1)
                self._retention = np.array([e["priority"] for e in self.entries]) + .25 * (1 - duplicate.max(axis=1))
            retention = self._retention
            index = int(retention.argmin())
            if priority < retention[index]:
                return None
            self.entries.pop(index)
        self.entries.append(entry)
        self._keys = None
        self._retention = None
        return entry["id"]

    def protect_episode(self, episode, priority, game):
        self.episodes.append({"episode": deepcopy(episode), "priority": float(priority), "game": int(game)})
        self.episodes.sort(key=lambda entry: entry["priority"], reverse=True)
        del self.episodes[self.cfg.protected_episodes:]

    def audit(self, ids, agreement, *, observed=False):
        # Bootstrap-based perturbation tests are weaker evidence than actual endings.
        weight = .2 if observed else .05
        wanted = set(ids)
        for entry in self.entries:
            if entry["id"] in wanted:
                entry["trust"] = float(np.clip((1 - weight) * entry["trust"] + weight * agreement, .1, 1))
                entry["audits"] += 1

    def state_dict(self, *, include_episodes=True, copy=True):
        state = {"entries": self.entries, "episodes": self.episodes if include_episodes else [],
                 "next_id": self.next_id}
        return deepcopy(state) if copy else state

    def load_state_dict(self, state):
        self.entries, self.episodes = deepcopy(state["entries"]), deepcopy(state["episodes"])
        self.next_id, self._keys = state["next_id"], None
        self._retention = None


class SkillLibrary:
    """Mine short legal continuations; support and real returns decide reuse.

    These are episodic macro-actions, not named chess concepts or DIAYN skills.
    Any unexpected reply, illegal continuation or inferior value cancels a plan.
    """
    def __init__(self, cfg):
        self.cfg, self.skills = cfg, []

    def learn(self, key, actions, colour, target, baseline, game):
        actions = tuple(perspective_action(action, colour) for action in actions[:self.cfg.skill_horizon])
        if len(actions) < 3 or target <= baseline:
            return
        for skill in self.skills:
            if skill["actions"] == actions and float(skill["key"] @ key) >= self.cfg.memory_similarity:
                if game in skill["games"]:
                    return
                skill["games"].add(game)
                n = len(skill["games"])
                skill["value"] += (target - skill["value"]) / n
                return
        self.skills.append({"key": np.array(key, copy=True), "actions": actions,
                            "value": float(target), "games": {int(game)}, "trust": 1.0})
        if len(self.skills) > self.cfg.skill_capacity:
            # Rare one-off motifs give way to repeatable, useful sequences.
            worst = min(range(len(self.skills)), key=lambda i: (len(self.skills[i]["games"]), self.skills[i]["value"]))
            self.skills.pop(worst)

    def retrieve(self, key, board, colour, fast=None):
        candidates = []
        for skill in self.skills:
            similarity = float(skill["key"] @ key)
            if (len(skill["games"]) < self.cfg.skill_min_support or similarity < self.cfg.memory_similarity
                    or skill["trust"] < .5):
                continue
            actions = tuple(perspective_action(action, colour) for action in skill["actions"])
            if fast is not None:
                from .fast_rules import code, NONE
                replay = fast.copy()
                for action in actions:
                    move = code(chess.Move.from_uci(action))
                    if move not in replay.legal_moves() or replay.terminal() != NONE:
                        break
                    replay.push(move)
                else:
                    candidates.append((similarity * skill["value"] * skill["trust"], actions))
            else:
                replay = board.copy(stack=True)
                for action in actions:
                    move = chess.Move.from_uci(action)
                    if move not in replay.legal_moves or replay.is_game_over(claim_draw=True):
                        break
                    replay.push(move)
                else:
                    candidates.append((similarity * skill["value"] * skill["trust"], actions))
        return max(candidates, key=lambda item: item[0])[1] if candidates else ()

    def audit(self, key, agreement, *, observed=False):
        weight = .2 if observed else .05
        for skill in self.skills:
            if float(skill["key"] @ key) >= self.cfg.memory_similarity:
                skill["trust"] = float(np.clip((1 - weight) * skill["trust"] + weight * agreement, .1, 1))

    def state_dict(self, *, copy=True):
        return deepcopy(self.skills) if copy else self.skills

    def load_state_dict(self, state):
        self.skills = deepcopy(state)
