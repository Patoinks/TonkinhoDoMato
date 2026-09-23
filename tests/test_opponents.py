"""V3 opponent mixture: self-play, TinyOpponent, Fairy-Stockfish and the main engine."""
import json
from collections import Counter

import chess
import numpy as np
import pytest
import torch

import train_cognitive as cli
from tonkai import opponents
from tonkai.cognitive_agent import CognitiveAgent
from tonkai.cognitive_config import CognitiveConfig
from tonkai.cognitive_trainer import CognitiveKnowledge
from tonkai.survival_agent import play_episode
from tonkai.temporal_model import SurvivalNet

from test_cognitive_cli import Engine, environment  # noqa: F401  (fixture)


def small(**overrides):
    values = dict(device="cpu", inference_device="cpu", hidden_size=24, latent_size=16, memory_size=12,
                  head_hidden=8, piece_hidden=8, ensemble_size=3, max_plies=40, planning_nodes=6)
    values.update(overrides)
    return CognitiveConfig(**values)


def agent_for(cfg):
    torch.manual_seed(3)
    net = SurvivalNet(cfg).eval()
    return CognitiveAgent(net, cfg, CognitiveKnowledge(cfg, net.encoder))


def test_owner_mixture_proportions():
    cfg = small(mix_self=.75, mix_tiny=.20, mix_fairy=.04, fairy_path="fairy.exe")
    assert opponents.mixture(cfg) == pytest.approx({"self": .75, "tiny": .20, "fairy": .04, "external": .01})
    rng = np.random.default_rng(0)
    drawn = Counter(opponents.choose_kind(cfg, rng) for _ in range(20000))
    for kind, share in opponents.mixture(cfg).items():
        assert drawn[kind] / 20000 == pytest.approx(share, abs=.01)
    assert not opponents.uses_mixture(small())


def test_mixture_validation():
    with pytest.raises(ValueError):
        small(mix_self=.8, mix_tiny=.3).validate()
    with pytest.raises(ValueError):
        small(mix_fairy=.1).validate()                      # no Fairy path
    small(mix_self=.75, mix_tiny=.25).validate()


def test_games_against_tiny_and_itself_are_complete_and_legal():
    cfg = small()
    for opponent in (opponents.TinyOpponentAdapter(7),
                     opponents.CognitiveOpponent(agent_for(cfg), cfg, np.random.default_rng(1))):
        episode, board, colour, stats = play_episode(agent_for(cfg), opponent, cfg, np.random.default_rng(2),
                                                     opponent_name=opponent.description)
        replay = chess.Board()
        for action in episode.actions:
            move = chess.Move.from_uci(action)
            assert move in replay.legal_moves
            replay.push(move)
        assert stats["opponent"] == opponent.description
        assert len(episode.packed) == len(episode.actions) + 1


def test_self_opponent_observes_every_ply_once():
    cfg = small()
    twin = agent_for(cfg)
    seen = []
    original = twin.observe
    twin.observe = lambda board, colour: seen.append(len(board.move_stack)) or original(board, colour)
    opponent = opponents.CognitiveOpponent(twin, cfg, np.random.default_rng(4))
    play_episode(agent_for(cfg), opponent, cfg, np.random.default_rng(5), self_color=chess.WHITE)
    # Its own observations are consecutive plies (evaluate re-reads the current one).
    distinct = sorted(set(seen))
    assert distinct == list(range(distinct[0], distinct[-1] + 1))


class MixPool:
    """The real pool's contract: kind is forwarded and reported back."""

    def __init__(self, cfg, engine_path):
        self.cfg, self.capacity, self.pending = cfg, cfg.num_workers, []

    def submit(self, net, knowledge, seed, game_number=0, resign_plan=None, kd_values=None, kind="external"):
        self.pending.append((net, knowledge, seed, game_number, kind))

    def receive(self):
        net, knowledge, seed, game_number, kind = self.pending.pop(0)
        agent = CognitiveAgent(net, self.cfg, knowledge)
        opponent = (opponents.CognitiveOpponent(CognitiveAgent(net, self.cfg, knowledge), self.cfg,
                                                np.random.default_rng(seed + 1)) if kind == "self"
                    else opponents.TinyOpponentAdapter(seed) if kind == "tiny" else Engine())
        episode, board, colour, stats = play_episode(agent, opponent, self.cfg, np.random.default_rng(seed),
                                                     opponent_name=opponent.description)
        stats.update(cognitive=dict(agent.diagnostics), game_number=game_number, opponent_kind=kind)
        return episode, board, colour, stats, agent.trace

    def close(self):
        pass


def test_training_runs_the_mixture_and_reports_each_opponent(tmp_path, environment, monkeypatch):  # noqa: F811
    monkeypatch.setattr(cli, "CognitivePool", MixPool)
    _, args = environment
    assert cli.main(args + ["--game-cap", "12", "--games", "12", "--mix-self", ".5", "--mix-tiny", ".4"]) == 0
    state = torch.load(tmp_path / "out/latest.pt", weights_only=False)
    assert state["config"]["mix_self"] == .5 and state["config"]["mix_tiny"] == .4
    rows = [json.loads(line) for path in (tmp_path / "history").glob("*.jsonl")
            for line in path.read_text().splitlines()]
    final = [row for row in rows if row["kind"] == "metrics" and row["final"]][0]
    assert sum(sum(row.values()) for row in final["opponents"].values()) == 12
    assert set(final["opponents"]) <= {"self", "tiny", "external"}
