"""The game review, the per-colour training and the white+black pair player.
Nothing is saved; one learner step runs on a throwaway trainer.

    python -m pytest claudio/test_review.py -q
"""
import sys
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

START = ROOT / "claudio/claudio-grande-g0.pt"
pytestmark = pytest.mark.skipif(not START.exists(), reason="needs the grown Claudio checkpoint")


def test_review_notes_material_and_the_decisive_move():
    from review import review, MATE_PAWNS
    # 1.e4 e5 2.Qh5 Nc6 3.Bc4 Nf6?? 4.Qxf7#  - black's 3...Nf6 allows mate, 4.Qxf7# mates.
    rv = review("e2e4 e7e5 d1h5 b8c6 f1c4 g8f6 h5f7".split())
    notes = {n["san"]: n for n in rv["notes"]}
    assert rv["winner"] == "white"
    assert notes["Nf6"]["label"] == "bad" and notes["Nf6"]["delta"] <= -MATE_PAWNS / 2
    assert notes["Qxf7#"]["label"] == "good"
    assert rv["decisive"]["black"]["san"] == "Nf6"
    # a quiet opening move changes nothing by itself; with the consequences it shares in the mate
    assert notes["e4"]["label"] == "neutral"
    later = {n["san"]: n for n in review("e2e4 e7e5 d1h5 b8c6 f1c4 g8f6 h5f7".split(), horizon=20)["notes"]}
    assert later["e4"]["immediate_label"] == "neutral" and later["e4"]["delayed"] > 0


def test_pair_player_uses_each_colours_specialist():
    from claudio_load import load_players, make_player
    player = make_player(load_players(f"{START}+{START}"), np.random.default_rng(0))
    board = chess.Board()
    board.push(player.choose(board))              # white to move: the white specialist
    board.push(player.choose(board))              # black to move: the black specialist
    assert player.white.colour == chess.WHITE and player.black.colour == chess.BLACK


def test_per_colour_game_and_review_targets_then_one_learning_step():
    import train_grande as tg
    from claudio_load import load_claudio
    from match_checkpoints import _historical_config
    from tonkai.sequence_replay import SequenceEpisode
    net, values, config, _, _ = load_claudio(START, with_config=True, with_king=True)
    arch = torch.load(START, map_location="cpu", weights_only=False)["architecture"]
    out = ROOT / "claudio/results/test-review"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / "actor-0.pt")
    tg._init_worker(str(out), arch, values, config, str(START), 0.0, "shaped", 0.002)
    games = [tg.play(g, 0, 5, tg.parse_stage("tiny"), 0, 120, 8, 0.03, "black", True) for g in range(3)]
    for r in games:
        assert r["colour"] == chess.BLACK                       # always Black
        assert len(r["proxies"]) == len(r["actions"]) + 1
        assert np.all((r["proxies"] >= 0) & (r["proxies"] <= 1))
        assert all(ply % 2 == 1 for ply, *_ in r["review_labels"])   # only Claudio's (black) moves get notes
    net.requires_grad_(True).train()
    net.rnd_target.requires_grad_(False)
    cfg = _historical_config(torch.load(tg.TIMDGEI, map_location="cpu", weights_only=False))
    cfg.device, cfg.sequence_batch_size, cfg.min_replay = "cpu", 8, 32
    cfg.objective, cfg.victory_weight, cfg.measure_lookahead = "shaped", 1.0, False
    Trainer = tg.make_trainer_class()
    Trainer.review_weight = 0.5
    trainer = Trainer(cfg, net=net)
    for r in games:
        episode = SequenceEpisode(packed=r["packed"], lost=r["lost"], terminated=r["terminated"],
                                  opponent="TinyOpponent (claudio)", surprises=r["surprises"], actions=r["actions"],
                                  result_value=r["result_value"], proxies=r["proxies"])
        if r["review_labels"]:
            trainer.__dict__.setdefault("review_labels", {})[hash(tuple(r["actions"]))] = r["review_labels"]
        trainer.ingest(episode, [], r["colour"])
    result = trainer.update()
    assert result is not None and np.isfinite(result["value_loss"])
