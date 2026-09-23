"""Organic attack learning: trying mate threats found by the rules, and the
victory measure grounded in proven forced mates. Checked with python-chess."""
import chess
import numpy as np
import pytest
import torch

from tonkai import fast_rules
from tonkai.cognitive_agent import CognitiveAgent
from tonkai.cognitive_config import CognitiveConfig
from tonkai.cognitive_trainer import CognitiveKnowledge
from tonkai.temporal_model import SurvivalNet

pytestmark = pytest.mark.skipif(not fast_rules.AVAILABLE, reason="tonkai_fast is not built")

THREATS = "r5k1/5ppp/8/8/8/2Q5/5PPP/6K1 w - - 0 1"         # no mate now; e.g. Qc6 threatens Qe8#
MATE_IN_TWO = "3r2k1/5ppp/8/8/8/8/4RPPP/4R1K1 w - - 0 1"


def agent(seed=0, **settings):
    cfg = CognitiveConfig(device="cpu", inference_device="cpu", hidden_size=32, latent_size=24, memory_size=16,
                          head_hidden=12, piece_hidden=12, ensemble_size=3, future_horizons=(1, 2, 4), burn_in=4,
                          sequence_length=12, sequence_batch_size=4, replay_capacity=512, min_replay=1,
                          max_plies=60, seed=27, planning_nodes=24, objective="shaped", **settings)
    torch.manual_seed(seed)
    net = SurvivalNet(cfg).eval()
    tonkai = CognitiveAgent(net, cfg, CognitiveKnowledge(cfg, net.encoder))
    tonkai.reset()
    return tonkai


def creates_threat(board, move):
    """After `move` and a pass by the opponent, the mover mates at once."""
    after = board.copy()
    after.push(move)
    if after.is_check() or after.is_game_over():
        return False
    after.push(chess.Move.null())
    for reply in list(after.legal_moves):
        after.push(reply)
        mate = after.is_checkmate()
        after.pop()
        if mate:
            return True
    return False


def test_exploration_tries_moves_that_create_a_mate_threat():
    board = chess.Board(THREATS)
    threats = {move for move in board.legal_moves if creates_threat(board, move)}
    assert threats
    for seed in range(4):
        tonkai = agent(seed, threat_explore=1.0)
        choice = tonkai.choose(board, chess.WHITE, np.random.default_rng(seed), explore=True)
        assert choice.move in threats
        assert tonkai.diagnostics["threat_moves_explored"] == 1


def test_no_threat_exploration_when_off_or_not_exploring():
    board = chess.Board(THREATS)
    plain = agent(0)
    plain.choose(board, chess.WHITE, np.random.default_rng(0), explore=True)
    assert "threat_moves_explored" not in plain.diagnostics
    greedy = agent(0, threat_explore=1.0)
    greedy.choose(board, chess.WHITE, np.random.default_rng(0), explore=False)
    assert "threat_moves_explored" not in greedy.diagnostics


def test_victory_measure_is_the_proven_result():
    board = chess.Board(MATE_IN_TWO)
    white, black = agent(0, mate_labels=2), agent(0, mate_labels=2)
    assert white.position_proxy(board, chess.WHITE) == 1.0          # White forces mate in two
    assert black.position_proxy(board, chess.BLACK) == 0.0          # ...so Black is proven lost
    unlabeled = agent(0).position_proxy(board, chess.WHITE)
    assert 0.0 < unlabeled < 1.0                                      # off: the usual estimate
    quiet = chess.Board()
    assert agent(0, mate_labels=2).position_proxy(quiet, chess.WHITE) == agent(0).position_proxy(quiet, chess.WHITE)


def test_settings_are_validated():
    with pytest.raises(ValueError):
        CognitiveConfig(threat_explore=1.5).validate()
    with pytest.raises(ValueError):
        CognitiveConfig(mate_labels=-1).validate()
