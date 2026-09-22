"""CLI persistence and stopping contracts; no external engine is launched."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

import train_survival
from tonkai.encoding import encode, pack_many
from tonkai.population import Population
from tonkai.sequence_replay import SequenceEpisode
from tonkai.survival_config import SurvivalConfig
from tonkai.survival_trainer import SurvivalTrainer


@pytest.fixture(autouse=True)
def isolated_cpu_runtime(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train_survival.signal, "signal", lambda *args: None)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def source_checkpoint(tmp_path):
    cfg = SurvivalConfig(
        device="cpu", inference_device="cpu", num_workers=1,
        hidden_size=24, latent_size=16, memory_size=8, head_hidden=8,
        piece_hidden=8, ensemble_size=2, future_horizons=(1, 2),
        burn_in=2, sequence_length=8, sequence_batch_size=2,
        replay_capacity=512, min_replay=256,
        checkpoint_dir=str(tmp_path / "output"),
        opponent_path=str(tmp_path / "saved-engine.exe"),
        opponent_elo=1730, opponent_move_time=.037,
        external_probability=1.0, historical_probability=0.0,
        win_pgn_path=str(tmp_path / "longest.pgn"),
    )
    trainer = SurvivalTrainer(cfg)
    trainer.games = 17
    population = Population(cfg, trainer.net, seed=91)
    actor_rng = np.random.default_rng(92)
    source = tmp_path / "source.pt"
    trainer.save(source, population=population.state_dict(),
                 actor_rng=deepcopy(actor_rng.bit_generator.state), pretrained_games=20000)
    return source, cfg


@pytest.fixture
def fake_engine(monkeypatch):
    instances = []

    class Engine:
        def __init__(self, path, elo, move_time):
            self.path, self.elo, self.move_time = Path(path), elo, move_time
            self.closed = False
            self.description = "isolated test external engine"
            self.info = {"path": str(path), "sha256": "test-sha256",
                         "elo": elo, "move_time": move_time}
            instances.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(train_survival, "UCIOpponent", Engine)
    return instances


def history_rows(tmp_path):
    return [json.loads(line) for path in (tmp_path / "history").glob("*.jsonl")
            for line in path.read_text(encoding="utf-8").splitlines()]


def cli_args(source, tmp_path, games="3"):
    return ["--resume", str(source), "--games", games,
            "--history-dir", str(tmp_path / "history"), "--torch-threads", "1"]


def scripted_game(sans=("e4", "e5"), owner=chess.WHITE):
    board = chess.Board()
    encoded = [encode(board, owner)]
    for san in sans:
        board.push_san(san)
        encoded.append(encode(board, owner))
    outcome = board.outcome(claim_draw=True)
    episode = SequenceEpisode(
        packed=pack_many(np.stack(encoded)),
        lost=bool(outcome and outcome.winner is not None and outcome.winner != owner),
        terminated=outcome is not None, opponent="isolated test external engine",
        actions=tuple(move.uci() for move in board.move_stack))
    stats = {"plies": len(board.move_stack), "lost": episode.lost,
             "terminated": episode.terminated,
             "result": outcome.result() if outcome else "*", "opponent": episode.opponent}
    return episode, board, owner, stats


@pytest.mark.parametrize("override_engine", [False, True])
def test_resume_preserves_engine_settings_and_stop_file_saves_without_playing(
        source_checkpoint, fake_engine, monkeypatch, tmp_path, override_engine):
    source, cfg = source_checkpoint
    initial = torch.load(source, map_location="cpu", weights_only=False)
    stop = tmp_path / "STOP"
    stop.touch()
    args = cli_args(source, tmp_path) + ["--stop-file", str(stop)]
    expected_engine = Path(cfg.opponent_path).resolve()
    if override_engine:
        expected_engine = tmp_path / "replacement-engine.exe"
        args += ["--opponent-path", str(expected_engine)]
    monkeypatch.setattr(train_survival, "play_episode",
                        lambda *args, **kwargs: pytest.fail("A preexisting stop file must prevent games"))
    assert train_survival.main(args) == 0
    saved = torch.load(Path(cfg.checkpoint_dir) / "latest.pt", map_location="cpu", weights_only=False)
    assert saved["status"] == "stopped" and saved["games"] == 17
    assert saved["config"]["opponent_path"] == str(expected_engine)
    assert saved["config"]["opponent_elo"] == 1730
    assert saved["config"]["opponent_move_time"] == .037
    assert saved["opponent"] == fake_engine[0].info
    assert saved["opponent"]["sha256"] == "test-sha256"
    assert saved["actor_rng"] == initial["actor_rng"]
    assert saved["pretrained_games"] == 20000
    assert fake_engine[0].path == expected_engine and fake_engine[0].closed
    assert all(torch.equal(saved["model"][key], value) for key, value in initial["model"].items())
    assert history_rows(tmp_path)[-1]["state"] == "stopped"
    assert history_rows(tmp_path)[-1]["games"] == 0


def test_stop_file_created_during_game_keeps_completed_episode_and_saves(
        source_checkpoint, fake_engine, monkeypatch, tmp_path):
    source, cfg = source_checkpoint
    stop = tmp_path / "STOP"
    calls = []

    def play(*args, **kwargs):
        calls.append(True)
        stop.touch()
        return scripted_game()

    monkeypatch.setattr(train_survival, "play_episode", play)
    assert train_survival.main(cli_args(source, tmp_path) + ["--stop-file", str(stop)]) == 0
    saved = torch.load(Path(cfg.checkpoint_dir) / "latest.pt", map_location="cpu", weights_only=False)
    assert len(calls) == 1 and saved["games"] == 18 and saved["status"] == "stopped"
    restored = SurvivalTrainer(cfg)
    restored.load_state_dict(saved)
    assert len(restored.replay) == 3 and restored.replay.episode_count == 1
    assert fake_engine[0].closed
    assert history_rows(tmp_path)[-1]["truncated"] == 1


def test_parallel_stop_drains_submitted_games_without_submitting_more(
        source_checkpoint, fake_engine, monkeypatch, tmp_path):
    import tonkai.survival_pool

    source, cfg = source_checkpoint
    stop = tmp_path / "STOP"
    pools = []

    class Pool:
        def __init__(self, config, engine_path, workers):
            assert workers == 2
            self.pending = []
            self.submitted = []
            self.closed = False
            pools.append(self)

        def submit(self, worker, *args):
            self.pending.append(worker)
            self.submitted.append(worker)

        def receive(self, timeout):
            stop.touch()
            return self.pending.pop(0), *scripted_game()

        def close(self):
            self.closed = True

    monkeypatch.setattr(tonkai.survival_pool, "SurvivalActorPool", Pool)
    assert train_survival.main(cli_args(source, tmp_path, games="5")
                               + ["--workers", "2", "--stop-file", str(stop)]) == 0
    saved = torch.load(Path(cfg.checkpoint_dir) / "latest.pt", map_location="cpu", weights_only=False)
    assert saved["games"] == 19 and saved["status"] == "stopped"
    assert len(pools[0].submitted) == 2 and not pools[0].pending and pools[0].closed
    assert fake_engine[0].closed
    assert history_rows(tmp_path)[-1]["games"] == 2


def test_init_from_cannot_overwrite_existing_checkpoint(tmp_path, monkeypatch, capsys):
    source = tmp_path / "legacy.pt"
    torch.save({"config": {}, "model": {}}, source)
    target = tmp_path / "existing" / "latest.pt"
    target.parent.mkdir()
    target.write_bytes(b"preserve an existing checkpoint exactly")
    monkeypatch.setattr(train_survival, "UCIOpponent",
                        lambda *args: pytest.fail("Refusal must precede engine launch"))
    with pytest.raises(SystemExit) as caught:
        train_survival.main(["--init-from", str(source), "--games", "1", "--device", "cpu",
                             "--checkpoint-dir", str(target.parent),
                             "--history-dir", str(tmp_path / "history")])
    assert caught.value.code == 2
    assert "refusing to overwrite" in capsys.readouterr().err
    assert target.read_bytes() == b"preserve an existing checkpoint exactly"
    assert not (tmp_path / "history").exists()


@pytest.mark.parametrize("error_type", [OSError, ValueError])
def test_win_history_errors_do_not_abort_completed_training(
        source_checkpoint, fake_engine, monkeypatch, tmp_path, error_type):
    source, cfg = source_checkpoint
    monkeypatch.setattr(train_survival, "play_episode",
                        lambda *args, **kwargs: scripted_game(("f3", "e5", "g4", "Qh4#"), chess.BLACK))

    class BrokenRecorder:
        def __init__(self, path):
            assert path == cfg.win_pgn_path

        def consider(self, *args):
            raise error_type("isolated invalid or unavailable history")

    monkeypatch.setattr(train_survival, "LongestWinRecorder", BrokenRecorder)
    assert train_survival.main(cli_args(source, tmp_path, games="1")) == 0
    saved = torch.load(Path(cfg.checkpoint_dir) / "latest.pt", map_location="cpu", weights_only=False)
    assert saved["status"] == "completed" and saved["games"] == 18
    rows = history_rows(tmp_path)
    assert any(row["kind"] == "warning" and row["component"] == "win_history" for row in rows)
    assert rows[-1]["wins"] == 1 and rows[-1]["state"] == "completed"
    assert fake_engine[0].closed
