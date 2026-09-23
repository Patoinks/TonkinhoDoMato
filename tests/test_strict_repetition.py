"""Owner's rule: a draw by repetition only once the position HAS occurred three
times; repeating is TonkAI's own choice, valued at the draw value."""
import random

import chess
import numpy as np
import pytest
import torch

from tonkai import fast_rules
from tonkai.cognitive_agent import CognitiveAgent, strict_ending, terminal_value
from tonkai.cognitive_config import CognitiveConfig
from tonkai.cognitive_trainer import CognitiveKnowledge
from tonkai.temporal_model import SurvivalNet

pytestmark = pytest.mark.skipif(not fast_rules.AVAILABLE, reason="tonkai_fast is not built")
SHUFFLE = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]      # ...f6g8 would be the 3rd time


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


def test_could_claim_is_not_the_end_but_a_real_threefold_is():
    board = chess.Board()
    for move in SHUFFLE:
        board.push_uci(move)
    assert board.can_claim_threefold_repetition()                  # the old rule would end it here
    assert strict_ending(board) == ""                              # now Black may play on
    assert fast_rules.board(board).terminal_strict() == fast_rules.NONE
    assert agent(strict_repetition=True).game_over(board) is False
    assert agent().game_over(board) is True                         # default keeps the old rule
    board.push_uci("f6g8")                                         # Black chose to repeat: 3rd time
    assert strict_ending(board) is None and fast_rules.board(board).terminal_strict() == fast_rules.DRAW
    assert terminal_value(board, chess.WHITE, "shaped", draw=0.4, strict=True) == 0.4


def test_cpp_and_python_strict_rules_agree_on_random_games():
    rng = random.Random(11)
    checked = 0
    for _ in range(60):
        board = chess.Board()
        while strict_ending(board) == "" and len(board.move_stack) < 200:
            moves = list(board.legal_moves)
            # bias towards shuffling so repetitions really happen
            knights = [m for m in moves if board.piece_type_at(m.from_square) in (chess.KNIGHT, chess.KING)]
            board.push(rng.choice(knights if knights and rng.random() < .7 else moves))
            ending = strict_ending(board)
            code = fast_rules.board(board).terminal_strict()
            expected = (fast_rules.NONE if ending == "" else fast_rules.DRAW if ending is None
                        else fast_rules.WHITE_WINS if ending else fast_rules.BLACK_WINS)
            assert code == expected, board.fen()
            checked += 1
    assert checked > 1000


def test_repeating_is_a_choice_valued_at_the_draw_value():
    board = chess.Board()
    for move in SHUFFLE:
        board.push_uci(move)
    tonkai = agent(strict_repetition=True, draw_value=0.4)
    choice = tonkai.choose(board, chess.BLACK, np.random.default_rng(0), explore=False)
    repeat = choice.moves.index(chess.Move.from_uci("f6g8"))
    assert choice.scores[repeat] == pytest.approx(0.4)            # a known ending: the owner's draw value
