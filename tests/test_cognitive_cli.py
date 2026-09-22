import json

import numpy as np
import pytest
import torch

import train_cognitive as cli
from tonkai.cognitive_agent import CognitiveAgent
from tonkai.cognitive_trainer import CognitiveKnowledge
from tonkai.survival_agent import play_episode
from tonkai.survival_config import SurvivalConfig
from tonkai.survival_trainer import SurvivalTrainer
from tonkai.temporal_model import SurvivalNet


class Engine:
    description = "test-external"

    def __init__(self, path=None, elo=1600, move_time=.02):
        self.info = {"engine_name": "test-external", "elo": elo, "move_time": move_time}

    def reset(self):
        pass

    def choose(self, board, rng):
        moves = list(board.legal_moves)
        return moves[int(rng.integers(len(moves)))]

    def close(self):
        pass


class Pool:
    def __init__(self, cfg, engine_path):
        self.cfg, self.capacity, self.pending = cfg, cfg.num_workers, []

    def submit(self, net, knowledge, seed, game_number=0):
        self.pending.append((net, knowledge, seed, game_number))

    def receive(self):
        net, knowledge, seed, game_number = self.pending.pop(0)
        agent = CognitiveAgent(net, self.cfg, knowledge)
        episode, board, colour, stats = play_episode(agent, Engine(), self.cfg, np.random.default_rng(seed))
        stats["cognitive"] = dict(agent.diagnostics)
        stats["game_number"] = game_number
        return episode, board, colour, stats, agent.trace

    def close(self):
        pass


@pytest.fixture
def environment(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(cli, "UCIOpponent", Engine)
    monkeypatch.setattr(cli, "CognitivePool", Pool)
    cfg = SurvivalConfig(device="cpu", hidden_size=24, latent_size=16, memory_size=12,
                         head_hidden=8, piece_hidden=8, ensemble_size=3, max_plies=6,
                         min_replay=1, replay_capacity=1000, sequence_length=4, burn_in=2,
                         sequence_batch_size=2, updates_per_game=1, win_pgn_path="")
    source = tmp_path / "v2.pt"
    SurvivalTrainer(cfg).save(source)
    engine = tmp_path / "engine.exe"
    engine.touch()
    args = ["--from-survival", str(source), "--opponent-elo", "1700", "--opponent-path", str(engine),
            "--games", "2", "--workers", "1", "--checkpoint-dir", str(tmp_path / "out"),
            "--history-dir", str(tmp_path / "history"), "--planning-nodes", "6", "--log-every", "1"]
    return source, args


def test_next_opponent_entry_saves_v3_and_leaves_source_unchanged(tmp_path, environment):
    source, args = environment
    original = source.read_bytes()
    assert cli.main(args) == 0
    result = torch.load(tmp_path / "out/latest.pt", weights_only=False)
    assert result["format"] == "tonkai-cognitive-v3"
    assert result["games"] == 2
    assert result["config"]["opponent_elo"] == 1700
    assert result["config"]["external_probability"] == 1
    assert result["status"] == "completed"
    assert source.read_bytes() == original
    rows = [json.loads(line) for path in (tmp_path / "history").glob("*.jsonl") for line in path.read_text().splitlines()]
    assert rows[-1]["games"] == 2
    assert rows[-1]["cognitive"]["planning_nodes"] > 0
    with pytest.raises(SystemExit):
        cli.main(args)  # migration never overwrites an existing V3 destination


def test_stop_before_next_run_preserves_weights_and_source(tmp_path, environment):
    source, args = environment
    stop_file = tmp_path / "STOP"
    stop_file.touch()
    assert cli.main(args + ["--stop-file", str(stop_file)]) == 0
    stopped = torch.load(tmp_path / "out/latest.pt", weights_only=False)
    original = torch.load(source, weights_only=False)
    assert stopped["games"] == original["games"] == 0
    assert stopped["status"] == "stopped"
    for key, value in original["model"].items():
        torch.testing.assert_close(stopped["model"][key], value, rtol=0, atol=0)


def test_live_survival_output_directory_is_reserved(environment):
    source, args = environment
    index = args.index("--checkpoint-dir") + 1
    args[index] = str(cli.ROOT / "checkpoints/survival")
    with pytest.raises(SystemExit):
        cli.main(args)


def test_running_production_source_cannot_start_next_version(tmp_path, environment, monkeypatch):
    source, args = environment
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    active = tmp_path / "checkpoints/survival/latest.pt"
    active.parent.mkdir(parents=True)
    state = torch.load(source, weights_only=False)
    state["status"] = "running"
    torch.save(state, active)
    args[args.index("--from-survival") + 1] = str(active)
    with pytest.raises(SystemExit):
        cli.main(args)


def test_cap_survives_resume_and_never_oversubmits(tmp_path, environment):
    source, args = environment
    assert cli.main(args + ["--game-cap", "3", "--workers", "4"]) == 0
    output = tmp_path / "out/latest.pt"
    resumed = list(args)
    resumed[0:2] = ["--resume", str(output)]
    resumed[resumed.index("--games") + 1] = "500"
    assert cli.main(resumed + ["--workers", "4"]) == 0
    state = torch.load(output, weights_only=False)
    assert state["games"] == 3
    assert state["campaign_cap"] == 3
    assert sum(state['campaign_results'].values()) == 3
    assert len(state["opening_history"]) == 3
    before = output.read_bytes()
    assert cli.main(resumed) == 0
    assert output.read_bytes() == before


def test_explicit_new_campaign_retains_learning_and_resets_only_cap_and_openings(tmp_path, environment):
    _, args = environment
    assert cli.main(args + ['--game-cap', '2']) == 0
    previous = tmp_path / 'out/latest.pt'
    previous_bytes = previous.read_bytes()
    resumed = list(args)
    resumed[0:2] = ['--resume', str(previous)]
    resumed += ['--new-campaign', '--game-cap', '100000']
    with pytest.raises(SystemExit):
        cli.main(resumed)  # Cannot replace the original campaign.
    resumed[resumed.index('--checkpoint-dir') + 1] = str(tmp_path / 'new')
    assert cli.main(resumed) == 0
    state = torch.load(tmp_path / 'new/latest.pt', weights_only=False)
    assert state['games'] == 4 and state['campaign_start_games'] == 2
    assert state['campaign_cap'] == 100000 and len(state['opening_history']) == 2
    assert sum(state['campaign_results'].values()) == 2
    assert previous.read_bytes() == previous_bytes


class ResignPool(Pool):
    """The real pool forwards a resignation plan to the actor; so does this."""

    def submit(self, net, knowledge, seed, game_number=0, resign_plan=None, kd_values=None):
        self.pending.append((net, knowledge, seed, game_number, resign_plan, kd_values))

    def receive(self):
        from tonkai.resign import ResignTracker
        net, knowledge, seed, game_number, plan, kd_values = self.pending.pop(0)
        agent = CognitiveAgent(net, self.cfg, knowledge, kd_values=kd_values)
        episode, board, colour, stats = play_episode(agent, Engine(), self.cfg, np.random.default_rng(seed),
                                                     resign=ResignTracker.from_plan(plan))
        stats["cognitive"] = dict(agent.diagnostics)
        stats["game_number"] = game_number
        return episode, board, colour, stats, agent.trace


def test_resign_and_capture_target_run_end_to_end_and_resume(tmp_path, environment, monkeypatch):
    monkeypatch.setattr(cli, "CognitivePool", ResignPool)
    _, args = environment
    # Without the new options: no capture head, no resign state.
    assert cli.main(args + ["--game-cap", "4"]) == 0
    output = tmp_path / "out/latest.pt"
    plain = torch.load(output, weights_only=False)
    assert not any(key.startswith("capture_out.") for key in plain["model"])
    assert plain["resign"] is None
    assert plain["capture_stats"]["games"] == 2
    # Resume the SAME checkpoint with both options on: the head is added on load.
    resumed = list(args)
    resumed[0:2] = ["--resume", str(output)]
    resumed += ["--resign", "--resign-threshold", "0.99", "--resign-min-ply", "0",
                "--resign-patience", "1", "--capture-target-weight", "0.1"]
    assert cli.main(resumed) == 0
    state = torch.load(output, weights_only=False)
    assert state["games"] == 4
    assert state["config"]["capture_target_weight"] == 0.1
    assert any(key.startswith("capture_out.") for key in state["model"])
    assert state["resign"]["state"] == "shadow"          # never live in two games
    assert state["resign"]["counts"]["triggered"] == 2
    assert state["capture_stats"]["games"] == 4
    rows = [json.loads(line) for path in (tmp_path / "history").glob("*.jsonl")
            for line in path.read_text().splitlines()]
    # Two runs; glob order between their history files is not guaranteed.
    finals = [row for row in rows if row["kind"] == "metrics" and row["final"]]
    assert len(finals) == 2
    with_resign = [row for row in finals if "resign" in row]
    assert len(with_resign) == 1                 # the first run, options off, has none
    metrics = with_resign[0]
    assert metrics["resign"]["state"] == "shadow" and "captures" in metrics
    assert "capture_loss" in metrics["loss"]


def test_owner_configuration_switches_to_shaped_and_records_victory_measures(tmp_path, environment, monkeypatch):
    monkeypatch.setattr(cli, "CognitivePool", ResignPool)
    _, args = environment
    assert cli.main(args + ["--game-cap", "4"]) == 0                     # survival first
    output = tmp_path / "out/latest.pt"
    resumed = list(args)
    resumed[0:2] = ["--resume", str(output)]
    resumed += ["--objective", "shaped", "--kd-weight", "0.75", "--capture-target-weight", "0.05",
                "--resign", "--resign-min-non-loss", "0"]
    assert cli.main(resumed) == 0
    state = torch.load(output, weights_only=False)
    assert state["config"]["objective"] == "shaped"
    assert state["config"]["kd_weight"] == 0.75
    assert state["resign"]["state"] in ("shadow", "active")
    with_measures = [row for row in state["replay"]["episodes"] if row.get("proxies") is not None]
    assert len(with_measures) == 2                                      # the two shaped games
    for row in with_measures:
        assert len(row["proxies"]) == len(row["packed"])
    rows = [json.loads(line) for path in (tmp_path / "history").glob("*.jsonl")
            for line in path.read_text().splitlines()]
    assert any(row.get("kind") == "metrics" and row.get("objective") == "shaped" for row in rows)
