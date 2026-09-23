"""Owner's anti-stalling rule: N half-moves without a capture - both lose."""
import chess
import numpy as np
import pytest
import torch

from tonkai import fast_rules
from tonkai.cognitive_agent import CognitiveAgent, quiet_plies
from tonkai.cognitive_config import CognitiveConfig
from tonkai.cognitive_trainer import CognitiveKnowledge
from tonkai.survival_agent import play_episode
from tonkai.temporal_model import SurvivalNet


def agent(**settings):
    cfg = CognitiveConfig(device="cpu", inference_device="cpu", hidden_size=32, latent_size=24, memory_size=16,
                          head_hidden=12, piece_hidden=12, ensemble_size=3, future_horizons=(1, 2, 4), burn_in=4,
                          sequence_length=12, sequence_batch_size=4, replay_capacity=512, min_replay=1,
                          max_plies=60, seed=27, planning_nodes=24, objective="shaped", **settings)
    torch.manual_seed(0)
    net = SurvivalNet(cfg).eval()
    tonkai = CognitiveAgent(net, cfg, CognitiveKnowledge(cfg, net.encoder))
    tonkai.reset()
    return tonkai


def test_quiet_plies_counts_since_the_last_capture():
    board = chess.Board()
    for san in ("e4", "d5", "exd5", "Nf6", "Nc3", "Nxd5", "Nf3"):
        board.push_san(san)
    assert quiet_plies(board, 30) == 1                       # Nf3 after the capture Nxd5
    assert quiet_plies(chess.Board(), 30) == 0


def test_stalling_ends_the_game_and_is_a_loss():
    board = chess.Board()
    for san in ("Nf3", "Nf6", "Ng1", "Ng8"):
        board.push_san(san)
    tonkai = agent(stall_limit=4)
    assert tonkai.stalled(board) and tonkai.game_over(board)
    assert tonkai.position_proxy(board, chess.WHITE) == 0.0
    assert not agent().stalled(board)                         # off by default


def test_a_quiet_move_reaching_the_limit_is_scored_as_a_loss():
    board = chess.Board()
    for san in ("Nf3", "Nf6", "Ng1"):
        board.push_san(san)
    tonkai = agent(stall_limit=4)
    choice = tonkai.choose(board, chess.BLACK, np.random.default_rng(0), explore=False)
    assert all(score == 0.0 for move, score in zip(choice.moves, choice.scores) if not board.is_capture(move))


class Shuffler:
    description = "shuffler"

    def reset(self):
        pass

    def choose(self, board, rng):
        for move in board.legal_moves:
            if board.piece_type_at(move.from_square) == chess.KNIGHT and not board.is_capture(move):
                return move
        return next(iter(board.legal_moves))


def test_a_stalled_game_is_recorded_as_lost():
    tonkai = agent(stall_limit=6)
    episode, board, colour, stats = play_episode(tonkai, Shuffler(), tonkai.cfg, np.random.default_rng(1))
    if stats["stalled"]:
        assert episode.lost and episode.terminated and stats["result"] == "0-0" and episode.result_value == 0.0
    assert quiet_plies(board, 6) < 6 or stats["stalled"]
