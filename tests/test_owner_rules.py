"""Owner's rules: forced resignation, and no repetition draw when ahead."""
import chess
import numpy as np
import torch

from tonkai.cognitive_agent import CognitiveAgent
from tonkai.cognitive_config import CognitiveConfig
from tonkai.cognitive_trainer import CognitiveKnowledge
from tonkai.resign import ACTIVE, SHADOW, ResignConfig, ResignController
from tonkai.temporal_model import SurvivalNet


def test_forced_resignation_is_always_active_and_never_verified_or_tightened():
    ctl = ResignController(ResignConfig(enabled=True, force=True, min_verified=5), 0.2)
    assert ctl.state == ACTIVE
    for _ in range(50):
        assert ctl.plan()[0] == ACTIVE                       # no played-out verification games
    for lost in (False,) * 30:                               # would be "wrong" resignations
        ctl.update({"mode": ACTIVE, "threshold": .2, "triggered": True, "trigger_ply": 50,
                    "resigned": True, "plies": 50, "lowest_hope": 0.0}, lost=lost)
    assert ctl.state == ACTIVE and ctl.threshold == .2
    shadow = ResignController(ResignConfig(enabled=True, min_verified=5), 0.2)
    assert shadow.state == SHADOW
    resumed = ResignController(ResignConfig(enabled=True, force=True), 0.2)
    resumed.load_state_dict(shadow.state_dict())            # a saved shadow state
    assert resumed.state == ACTIVE


def agent(margin, own_queen=True):
    cfg = CognitiveConfig(device="cpu", inference_device="cpu", hidden_size=24, latent_size=16, memory_size=12,
                          head_hidden=8, piece_hidden=8, ensemble_size=3, planning_nodes=6,
                          objective="shaped", kd_weight=1.0, avoid_repetition_margin=margin)
    torch.manual_seed(4)
    net = SurvivalNet(cfg).eval()
    return CognitiveAgent(net, cfg, CognitiveKnowledge(cfg, net.encoder),
                          kd_values={1: 1.0, 2: 3.0, 3: 3.0, 4: 5.0, 5: 9.0})


def repeating_board(fen):
    board = chess.Board(fen)
    for uci in ("b1b2", "a8a7", "b2b1", "a7a8", "b1b2", "a8a7", "b2b1", "a7a8"):
        board.push_uci(uci)
    return board


def repetition_score(margin, fen="k7/8/8/8/8/8/8/KQ6 w - - 0 1"):
    board = repeating_board(fen)
    choice = agent(margin).evaluate(board, chess.WHITE, explore=False)
    index = choice.moves.index(chess.Move.from_uci("b1b2"))          # the third repetition
    return float(choice.scores[index])


def test_ahead_by_a_queen_the_repetition_is_nearly_worthless():
    assert repetition_score(0.0) == 1.0 - 0.5 / 2                    # shaped draw value
    assert repetition_score(2.0) <= 0.05 + 1e-6


def test_level_material_keeps_the_draw_value():
    # Both sides have a queen: balance 0 < margin, so the draw is not penalised.
    level = "kq6/8/8/8/8/8/8/KQ6 w - - 0 1"
    board = chess.Board(level)
    for uci in ("b1b2", "b8b7", "b2b1", "b7b8", "b1b2", "b8b7", "b2b1", "b7b8"):
        board.push_uci(uci)
    choice = agent(2.0).evaluate(board, chess.WHITE, explore=False)
    index = choice.moves.index(chess.Move.from_uci("b1b2"))
    assert float(choice.scores[index]) == 1.0 - 0.5 / 2
