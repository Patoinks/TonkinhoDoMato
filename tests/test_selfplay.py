"""End to end: a full game, legality, learning, and checkpoints."""

from __future__ import annotations

import os

import chess
import numpy as np
import pytest
import torch

import tonkai
from tonkai import chess_env as env
from tonkai.config import Config
from tonkai.encoding import PACKED_SIZE


@pytest.fixture(scope="module")
def rig():
    cfg = Config(device="cpu", seed=3, max_plies=120, opponent_depth=1,
                 opponent_nodes=200, min_replay=32, batch_size=32,
                 predict_capacity=500, replay_capacity=5000)
    return cfg, tonkai.build(cfg, "cpu")


def test_a_full_game_completes(rig):
    cfg, (net, agent, opponent, worker, trainer, metrics) = rig
    rec = worker.play_game(self_color=chess.WHITE)
    assert rec.result in (0, 1, 2)
    assert rec.reason in (
        env.TERM_CHECKMATE, env.TERM_STALEMATE, env.TERM_INSUFFICIENT,
        env.TERM_FIFTY, env.TERM_REPETITION, env.TERM_PLY_LIMIT,
    )
    assert rec.plies > 0
    assert rec.packed.shape[1] == PACKED_SIZE
    assert rec.packed.shape[0] > 0


def test_tonkai_only_ever_plays_legal_moves(rig):
    """Replay a game move by move and check every single one."""
    cfg, (net, agent, opponent, worker, trainer, metrics) = rig
    rng = np.random.default_rng(11)
    for _ in range(3):
        board = chess.Board()
        self_color = chess.WHITE
        while True:
            res, _why = env.terminal(board, self_color, 60)
            if res is not None:
                break
            if board.turn == self_color:
                mc = agent.evaluate_moves(board, self_color)
                for mv in mc.moves:
                    assert mv in board.legal_moves
                choice = agent.choose(board, self_color, 1.0, rng)
                assert choice.move in board.legal_moves
                board.push(choice.move)
            else:
                env.apply_opponent_move(board, opponent.choose_move(board.fen()))


def test_all_legal_moves_are_scored_in_one_batch(rig):
    cfg, (net, agent, opponent, worker, trainer, metrics) = rig
    board = chess.Board()
    mc = agent.evaluate_moves(board, chess.WHITE)
    assert len(mc.moves) == 20
    assert mc.encoded.shape == (20, 782)
    assert mc.probs.shape == (20, 3)
    assert mc.utilities.shape == (20,)
    np.testing.assert_allclose(mc.probs.sum(axis=1), 1.0, atol=1e-5)


def test_predict_distribution_covers_exactly_the_legal_replies(rig):
    cfg, (net, agent, opponent, worker, trainer, metrics) = rig
    board = chess.Board()
    board.push_san("e4")
    moves, probs, x = agent.predict_enemy(board, chess.WHITE)
    assert sorted(m.uci() for m in moves) == sorted(
        m.uci() for m in board.legal_moves
    )
    assert probs.shape == (len(moves),)
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)
    assert x.shape[0] == len(moves)


def test_training_runs_and_changes_the_weights(rig):
    cfg, (net, agent, opponent, worker, trainer, metrics) = rig
    for _ in range(4):
        trainer.add_game(worker.play_game())
    assert trainer.ready()
    before = net.value_head[-1].weight.detach().clone()
    piece_before = net.piece_head[-1].weight.detach().clone()
    losses = [trainer.train_step() for _ in range(5)]
    assert all(np.isfinite(list(l.values())).all() for l in losses)
    assert not torch.equal(before, net.value_head[-1].weight)
    assert not torch.equal(piece_before, net.piece_head[-1].weight), (
        "PieceHead got no gradient"
    )


def test_value_head_learns_a_forced_label(rig):
    """Feed one repeated label and check the loss actually goes down."""
    cfg, (net, agent, opponent, worker, trainer, metrics) = rig
    first = trainer.train_step()["value"]
    for _ in range(60):
        trainer.train_step()
    last = trainer.train_step()["value"]
    assert last < first + 0.05, (first, last)


def test_checkpoint_roundtrip(rig, tmp_path):
    cfg, (net, agent, opponent, worker, trainer, metrics) = rig
    path = str(tmp_path / "ckpt.pt")
    trainer.save(path)
    assert os.path.exists(path)

    cfg2 = Config.from_dict(
        torch.load(path, map_location="cpu", weights_only=False)["config"]
    )
    net2, agent2, opp2, worker2, trainer2, metrics2 = tonkai.build(cfg2, "cpu")
    trainer2.load(path)
    assert trainer2.games == trainer.games
    assert trainer2.steps == trainer.steps
    for (k1, v1), (k2, v2) in zip(net.state_dict().items(),
                                  net2.state_dict().items()):
        assert k1 == k2
        torch.testing.assert_close(v1, v2)

    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    a = agent.evaluate_moves(board, chess.WHITE)
    b = agent2.evaluate_moves(board, chess.WHITE)
    np.testing.assert_allclose(a.utilities, b.utilities, atol=1e-6)


def test_piece_values_cover_every_piece(rig):
    cfg, (net, agent, opponent, worker, trainer, metrics) = rig
    board = chess.Board()
    vals = agent.piece_values(board, chess.WHITE)
    assert len(vals) == 32
    assert sum(1 for _sq, _p, is_self, _v in vals if is_self) == 16
    assert all(np.isfinite(v) for *_rest, v in vals)


def test_many_games_without_crashing(rig):
    cfg, (net, agent, opponent, worker, trainer, metrics) = rig
    for _ in range(12):
        rec = worker.play_game()
        trainer.add_game(rec)
        trainer.train(1)
    assert metrics.games >= 12
