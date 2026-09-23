"""Organic piece safety / trading in play: the exact exchange search with
TonkAI's measured values, exploring non-losing captures, exact labels."""
import chess
import numpy as np
import pytest
import torch

from tonkai import fast_rules
from tonkai.cognitive_agent import CognitiveAgent
from tonkai.cognitive_config import CognitiveConfig
from tonkai.cognitive_trainer import CognitiveKnowledge
from tonkai.exchange import exchange_after_moves
from tonkai.temporal_model import SurvivalNet

pytestmark = pytest.mark.skipif(not fast_rules.AVAILABLE, reason="tonkai_fast is not built")

VALUES = {1: 1.0, 2: 1.39, 3: 1.64, 4: 1.84, 5: 3.65}
QUEENS = "4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1"          # Qxd5 wins a queen; Qd2/Qd4... lines matter


def agent(seed=0, **settings):
    cfg = CognitiveConfig(device="cpu", inference_device="cpu", hidden_size=32, latent_size=24, memory_size=16,
                          head_hidden=12, piece_hidden=12, ensemble_size=3, future_horizons=(1, 2, 4), burn_in=4,
                          sequence_length=12, sequence_batch_size=4, replay_capacity=512, min_replay=1,
                          max_plies=60, seed=27, planning_nodes=24, objective="shaped", **settings)
    torch.manual_seed(seed)
    net = SurvivalNet(cfg).eval()
    tonkai = CognitiveAgent(net, cfg, CognitiveKnowledge(cfg, net.encoder), kd_values=VALUES)
    tonkai.reset()
    return tonkai


def test_exchange_nets_are_the_exact_material_result():
    board = chess.Board(QUEENS)
    tonkai = agent(exchange_depth=6)
    tonkai.choose(board, chess.WHITE, np.random.default_rng(0), explore=False)
    nets = tonkai._evaluation["exchange_nets"]
    moves = list(board.legal_moves)
    reference = exchange_after_moves(board, moves, [0.0] + [VALUES[k] for k in range(1, 6)], 6)
    for move, net, loss in zip(moves, nets, reference):
        taken = VALUES[board.piece_type_at(move.to_square)] if board.is_capture(move) else 0.0
        assert net == pytest.approx(taken - loss)
    assert nets[moves.index(chess.Move.from_uci("d1d5"))] == pytest.approx(3.65)   # a free queen
    assert tonkai.diagnostics["exchange_searches"] == len(moves)


def test_off_by_default_and_without_measured_values():
    board = chess.Board(QUEENS)
    plain = agent()
    plain.choose(board, chess.WHITE, np.random.default_rng(0), explore=False)
    assert plain._evaluation["exchange_nets"] is None and "exchange_searches" not in plain.diagnostics
    unmeasured = agent(exchange_depth=6)
    unmeasured.kd_values = None
    unmeasured.choose(board, chess.WHITE, np.random.default_rng(0), explore=False)
    assert unmeasured._evaluation["exchange_nets"] is None


def test_trade_exploration_only_tries_captures_that_do_not_lose():
    board = chess.Board("4k3/8/8/3q4/2P5/8/8/3QK3 w - - 0 1")    # cxd5 and Qxd5 both win the queen
    for seed in range(3):
        tonkai = agent(seed, exchange_depth=6, trade_explore=1.0)
        choice = tonkai.choose(board, chess.WHITE, np.random.default_rng(seed), explore=True)
        assert board.is_capture(choice.move)
        assert tonkai._evaluation["exchange_nets"][choice.index] >= 0
        assert tonkai.diagnostics["trades_explored"] == 1


def test_labels_use_the_exact_result_for_the_side_to_move():
    board = chess.Board(QUEENS)                                    # White to move wins a queen
    white = agent(exchange_depth=6, exchange_labels=True)
    black = agent(exchange_depth=6, exchange_labels=True)
    w, b = white.position_proxy(board, chess.WHITE), black.position_proxy(board, chess.BLACK)
    assert white.diagnostics["exchange_labels"] == 1 and black.diagnostics["exchange_labels"] == 1
    assert w > 0.5 > b
