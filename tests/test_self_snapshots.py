"""Self-play against a frozen TonkAI saved every N games (the rival never learns)."""
import json

import numpy as np
import torch

import train_cognitive as cli
from tonkai import cognitive_pool
from tonkai.actor_snapshot import pack_snapshot
from tonkai.cognitive_agent import CognitiveAgent
from tonkai.cognitive_trainer import CognitiveKnowledge
from tonkai.survival_agent import play_episode
from tonkai.temporal_model import SurvivalNet

from test_cognitive_cli import Engine, environment  # noqa: F401  (fixture)


class RivalPool:
    """Records the rival each self-play game was given."""
    rivals = []

    def __init__(self, cfg, engine_path, **_):
        self.cfg, self.capacity, self.pending = cfg, cfg.num_workers, []

    def submit(self, net, knowledge, seed, game_number=0, resign_plan=None, kd_values=None, kind="external",
               rival=None):
        self.pending.append((net, knowledge, seed, game_number, kind, rival))

    def receive(self):
        net, knowledge, seed, game_number, kind, rival = self.pending.pop(0)
        if kind == "self":
            RivalPool.rivals.append(None if rival is None else rival[0])
        agent = CognitiveAgent(net, self.cfg, knowledge)
        episode, board, colour, stats = play_episode(agent, Engine(), self.cfg, np.random.default_rng(seed))
        stats.update(cognitive=dict(agent.diagnostics), game_number=game_number, opponent_kind=kind)
        if rival is not None:
            stats["opponent_version"] = rival[0]
        return episode, board, colour, stats, agent.trace

    def close(self):
        pass


def test_rival_is_refrozen_every_n_games_and_restored_on_resume(tmp_path, environment, monkeypatch):  # noqa: F811
    monkeypatch.setattr(cli, "CognitivePool", RivalPool)
    RivalPool.rivals = []
    _, args = environment
    run = args + ["--game-cap", "20", "--games", "10", "--mix-self", "0.9", "--mix-tiny", "0.1",
                  "--self-snapshot-every", "4", "--checkpoint-every", "5"]
    assert cli.main(run) == 0
    snapshots = sorted((tmp_path / "out/snapshots").glob("tonkinho-jogo-*.pt"))
    versions = sorted(int(p.stem.rsplit("-", 1)[1]) for p in snapshots)
    assert versions == [0, 4, 8]                            # at start, then every 4 games
    assert RivalPool.rivals and set(RivalPool.rivals) <= {0, 4, 8}
    state = torch.load(tmp_path / "out/latest.pt", weights_only=False)
    assert state["self_play_rival"]["version"] == 8
    assert state["config"]["checkpoint_every"] == 5
    rows = [json.loads(line) for path in (tmp_path / "history").glob("*.jsonl")
            for line in path.read_text().splitlines()]
    final = [row for row in rows if row["kind"] == "metrics" and row["final"]][0]
    assert final["self_play_rival"]["version"] == 8
    assert final["self_play_rival"]["name"] == "Tonkinho congelado (jogo 8)"
    assert sum(sum(row.values()) for row in final["self_versions"].values()) == len(RivalPool.rivals)

    # Resume: the saved rival comes back instead of a new snapshot at the start.
    RivalPool.rivals = []
    resumed = list(run)
    resumed[0:2] = ["--resume", str(tmp_path / "out/latest.pt")]
    assert cli.main(resumed) == 0
    assert min(RivalPool.rivals) >= 8


def test_actor_rival_uses_exactly_the_frozen_weights():
    from tonkai.cognitive_config import CognitiveConfig
    cfg = CognitiveConfig(device="cpu", inference_device="cpu", hidden_size=24, latent_size=16, memory_size=12,
                          head_hidden=8, piece_hidden=8, ensemble_size=3)
    torch.manual_seed(1)
    frozen = SurvivalNet(cfg)
    payload = pack_snapshot(frozen, CognitiveKnowledge(cfg, frozen.encoder))
    torch.manual_seed(2)
    live = SurvivalNet(cfg)
    state = {"cfg": cfg, "net": live, "knowledge": CognitiveKnowledge(cfg, live.encoder)}
    opponent = cognitive_pool._opponent("self", state, 7, None, rival=(123, payload))
    assert opponent.description == "Tonkinho congelado (jogo 123)"
    twin = opponent.agent.net
    assert twin is not live
    for key, value in frozen.state_dict().items():
        assert torch.equal(twin.state_dict()[key], value)
    # Cached per version: the same objects are reused for the next game.
    again = cognitive_pool._opponent("self", state, 8, None, rival=(123, payload))
    assert again.agent.net is twin


def test_each_snapshot_gets_one_series_of_games_then_the_normal_mix(tmp_path, environment, monkeypatch):  # noqa: F811
    monkeypatch.setattr(cli, "CognitivePool", RivalPool)
    RivalPool.rivals = []
    _, args = environment
    assert cli.main(args + ["--game-cap", "14", "--games", "14", "--mix-self", "0", "--mix-tiny", "1",
                            "--self-snapshot-every", "6", "--self-games-per-snapshot", "2"]) == 0
    # One actor: snapshots at 0, 6 and 12, each followed by exactly 2 games against it.
    assert RivalPool.rivals == [0, 0, 6, 6, 12, 12]
    state = torch.load(tmp_path / "out/latest.pt", weights_only=False)
    assert state["self_play_rival"]["version"] == 12 and state["self_play_rival"]["remaining"] == 0
