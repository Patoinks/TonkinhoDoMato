"""Endgame tablebases: exact results for few-piece positions, adjudication of
training/match games and moves that enter a tablebase ending."""
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from tonkai import tablebase as tb
from tonkai.cognitive_agent import CognitiveAgent
from tonkai.cognitive_config import CognitiveConfig
from tonkai.cognitive_trainer import CognitiveKnowledge
from tonkai.survival_agent import play_episode
from tonkai.temporal_model import SurvivalNet

PATH = Path(__file__).resolve().parents[1] / "engines" / "syzygy"
NEEDED = ["KQvK.rtbw", "KRvK.rtbw", "KQvKR.rtbw", "KBNvK.rtbw"]
pytestmark = pytest.mark.skipif(not all((PATH / f).is_file() for f in NEEDED), reason="Syzygy tables not downloaded")


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


def test_verdicts():
    assert tb.verdict(chess.Board("8/8/8/4k3/8/8/8/4K2Q w - - 0 1"), str(PATH)) is chess.WHITE    # KQ v K
    assert tb.verdict(chess.Board("8/8/8/4k3/8/8/8/4K2Q b - - 0 1"), str(PATH)) is chess.WHITE
    assert tb.verdict(chess.Board("8/8/8/4k3/8/8/8/2B1K1N1 w - - 0 1"), str(PATH)) is chess.WHITE  # KBN v K
    assert tb.verdict(chess.Board(), str(PATH)) == ""                                             # 32 pieces
    assert tb.verdict(chess.Board("8/8/8/4k3/8/8/8/4K2Q w - - 0 1"), "") == ""                    # off


def test_training_games_end_at_the_tables_with_their_result():
    board = chess.Board("8/8/8/4k3/8/8/8/4K2Q w - - 0 1")
    on = agent(syzygy_path=str(PATH), tablebase_adjudicate=True)
    assert on.adjudicated(board) is chess.WHITE and on.game_over(board)
    assert agent().game_over(board) is False                          # default: play on

    class Mover:
        description = "mover"

        def reset(self):
            pass

        def choose(self, b, rng):
            return next(iter(b.legal_moves))
    start = chess.Board("8/8/8/4k3/8/8/1q6/Q3K3 w - - 0 1")           # Qxb2 enters KQ v K
    tonkai = agent(syzygy_path=str(PATH), tablebase_adjudicate=True)
    rng = np.random.default_rng(0)
    choice = tonkai.choose(start, chess.WHITE, rng, explore=False)
    capture = choice.moves.index(chess.Move.from_uci("a1b2"))
    assert choice.scores[capture] == 1.0                               # entering a won ending is a known win
    assert choice.move == chess.Move.from_uci("a1b2")


def test_victory_measure_is_the_tablebase_result():
    board = chess.Board("8/8/8/4k3/8/8/8/4K2Q b - - 0 1")
    tonkai = agent(syzygy_path=str(PATH))
    assert tonkai.position_proxy(board, chess.WHITE) == 1.0 and tonkai.position_proxy(board, chess.BLACK) == 0.0
