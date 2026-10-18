"""Cognitive state and experiments layered on the V2 survival learner.

Experiments use legal continuations and the external opponent's moves, never its
evaluation. Censored experiment returns are explicitly identified as estimates.
"""
from copy import deepcopy

import chess
import numpy as np
import torch

from .cognitive_memory import ErrorCalibration, ExperienceMemory, SkillLibrary
from .encoding import encode, pack_many
from .sequence_replay import SequenceEpisode
from .survival_agent import SurvivalAgent
from .survival_trainer import SurvivalTrainer
from .objective import observed_target, outcome_value, continuation_value, episode_result


class CognitiveKnowledge:
    def __init__(self, cfg, encoder):
        self.encoder = deepcopy(encoder).cpu().eval().requires_grad_(False)
        self.memory = ExperienceMemory(cfg)
        self.calibration = ErrorCalibration(cfg.calibration_bins)
        self.skills = SkillLibrary(cfg)

    def state_dict(self, *, include_episodes=True, copy=True):
        return {"encoder": self.encoder.state_dict(),
                "memory": self.memory.state_dict(include_episodes=include_episodes, copy=copy),
                "calibration": self.calibration.state_dict(), "skills": self.skills.state_dict(copy=copy)}

    def load_state_dict(self, state):
        self.encoder.load_state_dict(state["encoder"])
        self.memory.load_state_dict(state["memory"])
        self.calibration.load_state_dict(state["calibration"])
        self.skills.load_state_dict(state["skills"])


def at_ply(episode, ply):
    board = chess.Board(episode.initial_fen or chess.STARTING_FEN)
    for action in episode.actions[:ply]:
        board.push_uci(action)
    return board


class CognitiveTrainer(SurvivalTrainer):
    def __init__(self, cfg, net=None):
        super().__init__(cfg, net)
        self.knowledge = CognitiveKnowledge(cfg, self.future_encoder)
        self.experiments = 0
        self.generalizations = 0
        self.experiment_positions = 0

    def ingest(self, episode, trace, colour):
        """Evaluate predictions collected before updates, only with observed endings."""
        self.replay.add(episode)
        self.games += 1
        if not episode.terminated or not trace:
            return {"calibration_samples": 0, "memory_added": 0}
        # Spread samples through the trajectory; avoid filling an archive from one
        # long game or counting many adjacent states as independent neighbours.
        sample = [trace[i] for i in np.unique(np.linspace(0, len(trace) - 1, min(8, len(trace))).astype(int))]
        targets = [observed_target(episode, row["ply"], self.cfg) for row in sample]
        predictions = [row["prediction"] for row in sample]
        self.knowledge.calibration.observe(predictions, targets)
        added = 0
        # Keep one chronological replay for the sampled successor positions.
        # The samples are normally ordered; restart only for an out-of-order trace.
        following = None
        following_ply = 0
        for row, target in zip(sample, targets):
            actions = episode.actions[row["ply"]:row["ply"] + self.cfg.skill_horizon]
            identifier = self.knowledge.memory.add(row["key"], target, row["prediction"], self.games,
                                                   row["fen"], actions, colour, own_turn=True)
            added += identifier is not None
            if "next_key" in row:
                next_target = observed_target(episode, row["ply"] + 1, self.cfg)
                wanted_ply = row["ply"] + 1
                if following is None or wanted_ply < following_ply:
                    following = chess.Board(episode.initial_fen or chess.STARTING_FEN)
                    following_ply = 0
                for action in episode.actions[following_ply:wanted_ply]:
                    following.push_uci(action)
                following_ply = wanted_ply
                next_id = self.knowledge.memory.add(row["next_key"], next_target, row["next_prediction"],
                                                    self.games, following.fen(), colour=colour, own_turn=False)
                added += next_id is not None
                self.knowledge.calibration.observe([row["next_prediction"]], [next_target])
            self.knowledge.skills.learn(row["key"], actions, colour, target, row["prediction"], self.games)
        priority = float(np.mean(np.abs(np.array(targets) - predictions)))
        priority += float(np.mean([row["uncertainty"] for row in sample]))
        self.knowledge.memory.protect_episode(episode, priority, self.games)
        if self.games % self.cfg.protected_replay_every == 0 and self.knowledge.memory.episodes:
            # Independent archive survives FIFO eviction. Rehearsal carries the
            # original outcome and bounded frequency; no synthetic success labels.
            item = self.knowledge.memory.episodes[int(self.rng.integers(len(self.knowledge.memory.episodes)))]
            self.replay.add(item["episode"])
        return {"calibration_samples": len(sample), "memory_added": added,
                "value_absolute_error": priority}

    @torch.inference_mode()
    def _experiment_policy(self):
        # CUDA benefits the batched learner. Tiny sequential experiment calls
        # stay on CPU with the very same current parameters, not an old actor.
        if torch.device(self.device).type != 'cuda':
            return self.net, self.device
        stamp = tuple((id(p), p._version) for p in self.net.parameters())
        if getattr(self, '_experiment_stamp', None) != stamp:
            if not hasattr(self, '_experiment_net'):
                self._experiment_net = deepcopy(self.net).cpu().eval().requires_grad_(False)
            else:
                self._experiment_net.load_state_dict(self.net.state_dict())
            self._experiment_stamp = stamp
        return self._experiment_net, 'cpu'

    @torch.inference_mode()
    def _rollout(self, board, first_move, colour, opponent):
        """Finish a short real branch from an intact history, then bootstrap if cut."""
        branch = board.copy(stack=True)
        if first_move not in branch.legal_moves or branch.is_game_over(claim_draw=True):
            raise ValueError("Experiment root action must be legal in an ongoing game")
        policy, device = self._experiment_policy()
        agent = SurvivalAgent(policy, self.cfg, device)
        agent.observe(branch, colour)
        first_ply = len(branch.move_stack)
        branch.push(first_move)
        agent.observe(branch, colour)
        opponent.reset()
        while not branch.is_game_over(claim_draw=True) and len(branch.move_stack) - first_ply < self.cfg.experiment_plies:
            move = (agent.choose(branch, colour, self.rng, explore=False).move if branch.turn == colour
                    else opponent.choose(branch, self.rng))
            if move not in branch.legal_moves:
                raise ValueError("Illegal experiment response")
            branch.push(move)
            agent.observe(branch, colour)
        outcome = branch.outcome(claim_draw=True)
        lost = bool(outcome and outcome.winner is not None and outcome.winner != colour)
        steps = len(branch.move_stack) - first_ply
        bootstrap = float(policy.survival(agent.context)[0].mean())
        if self.cfg.objective == "victory":
            value = outcome_value(outcome, colour, "victory") if outcome else bootstrap
        else:
            value = (1 - self.cfg.survival_gamma ** steps if lost else 1.0) if outcome else continuation_value(bootstrap, steps, self.cfg)
        replay = branch.root()
        positions = [encode(replay, colour)]
        for move in branch.move_stack:
            replay.push(move)
            positions.append(encode(replay, colour))
        episode = SequenceEpisode(pack_many(np.stack(positions)), lost, outcome is not None,
            opponent="counterfactual:" + getattr(opponent, "description", "external"),
            actions=tuple(move.uci() for move in branch.move_stack), initial_fen=branch.root().fen(),
            result_value=outcome_value(outcome, colour, "victory"))
        self.replay.add(episode)
        self.experiment_positions += steps
        return {"value": value, "observed": outcome is not None, "plies": steps,
                "result": outcome.result() if outcome else "*", "actions": episode.actions[first_ply:],
                "initial_fen": board.fen(), "first_move": first_move.uci()}

    def investigate(self, episode, trace, colour, opponent, *, force=False):
        events = []
        if not trace:
            return events
        counterfactual = force or self.games % self.cfg.experiments_every == 0
        generalize = force or self.games % self.cfg.generalization_every == 0
        if not counterfactual and not generalize:
            return events
        # Prefer decisions where the policy lacked evidence or alternatives were close.
        choices = sorted(trace, key=lambda row: row["uncertainty"] - row["gap"], reverse=True)
        row = next((row for row in choices if len(list(at_ply(episode, row["ply"]).legal_moves)) > 1), None)
        if row is None:
            return events
        board = at_ply(episode, row["ply"])
        original = chess.Move.from_uci(row["move"])
        baseline = self._rollout(board, original, colour, opponent)
        if counterfactual:
            policy, device = self._experiment_policy()
            evaluator = SurvivalAgent(policy, self.cfg, device)
            candidates = evaluator.evaluate(board, colour, explore=False)
            alternative = next(candidates.moves[i] for i in np.argsort(candidates.scores)[::-1]
                               if candidates.moves[i] != original)
            changed = self._rollout(board, alternative, colour, opponent)
            events.append({"kind": "counterfactual", "root_ply": row["ply"], "baseline": baseline,
                           "alternative": changed, "difference": changed["value"] - baseline["value"],
                           "both_observed": baseline["observed"] and changed["observed"]})
            self.experiments += 1
        if generalize and row["ply"] > 0:
            # Change the opponent's immediately preceding move through legal play.
            # Preserve every earlier move, castling/repetition state, and side-to-move.
            previous = at_ply(episode, row["ply"] - 1)
            actual_reply = chess.Move.from_uci(episode.actions[row["ply"] - 1])
            variants = []
            for move in previous.legal_moves:
                if move == actual_reply:
                    continue
                variant = previous.copy(stack=True)
                variant.push(move)
                if not variant.is_game_over(claim_draw=True) and original in variant.legal_moves:
                    variants.append(variant)
            if variants:
                variant = variants[int(self.rng.integers(len(variants)))]
                tested = self._rollout(variant, original, colour, opponent)
                drop = max(0.0, baseline["value"] - tested["value"])
                agreement = float(np.clip(1 - drop / .25, 0, 1))
                ids = self.knowledge.memory.query(row["key"][None], own_turn=True)["ids"][0]
                observed = baseline["observed"] and tested["observed"]
                self.knowledge.memory.audit(ids, agreement, observed=observed)
                self.knowledge.skills.audit(row["key"], agreement, observed=observed)
                events.append({"kind": "generalization", "baseline": baseline, "variant": tested,
                               "agreement": agreement, "both_observed": observed,
                               "changed_opponent_move": variant.peek().uci(), "memory_ids": ids})
                self.generalizations += 1
        return events

    def state_dict(self):
        state = super().state_dict()
        state.update(format="tonkai-cognitive-v3", knowledge=self.knowledge.state_dict(),
                     experiments=self.experiments, generalizations=self.generalizations,
                     experiment_positions=self.experiment_positions)
        return state

    def load_state_dict(self, state):
        if state.get("format") not in ("tonkai-survival-v2", "tonkai-cognitive-v3"):
            raise ValueError("Cognitive training requires a survival V2 or cognitive V3 checkpoint")
        super().load_state_dict({**state, "format": "tonkai-survival-v2"})
        if state["format"] == "tonkai-cognitive-v3":
            self.knowledge.load_state_dict(state["knowledge"])
            self.experiments, self.generalizations = state["experiments"], state["generalizations"]
            self.experiment_positions = state["experiment_positions"]
        else:
            self.knowledge = CognitiveKnowledge(self.cfg, self.future_encoder)
        if state["config"].get("objective", "survival") != self.cfg.objective:
            # Value-labelled neighbours, calibration and plans encode the OLD
            # reward. Preserve neural representations and real trajectories,
            # but never treat survival memories as evidence of winning.
            from dataclasses import replace
            replay = self.replay.state_dict()
            for row, episode in zip(replay["episodes"], self.replay._episodes.values()):
                row["result_value"] = episode_result(episode)
                row["priorities"][:] = 1.0
            self.replay.load_state_dict(replay)
            self.knowledge = CognitiveKnowledge(self.cfg, self.future_encoder)
            self.optimizer.state.clear()
