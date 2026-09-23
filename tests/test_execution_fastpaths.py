import copy
import chess
import numpy as np
import pytest
import torch

from tonkai.cognitive_config import CognitiveConfig
from tonkai.cognitive_agent import CognitiveAgent
from tonkai.cognitive_trainer import CognitiveKnowledge, CognitiveTrainer
from tonkai.encoding import encode, encode_after_moves
from tonkai.execution_fastpaths import compact_encode, incremental_encode
from tonkai.temporal_model import SurvivalNet


def test_compact_preserves_padding_biases_and_valid_gradients():
    torch.manual_seed(9)
    a = SurvivalNet(CognitiveConfig()).encoder
    b = copy.deepcopy(a)
    x = torch.randn(4, 12, 782)
    valid = torch.rand(4, 12) > .4
    x[~valid] = 0
    expected = a(x.flatten(0, 1)).reshape(4, 12, -1)
    actual = compact_encode(b, x, valid)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    expected[valid].square().sum().backward()
    actual[valid].square().sum().backward()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize('fen', [chess.STARTING_FEN,
    'r3k2r/8/8/3pP3/8/8/P6p/R3K2R w KQkq d6 0 20',
    '4k3/P7/8/8/8/8/7p/4K3 w - - 98 40'])
@pytest.mark.parametrize('colour', [True, False])
def test_incremental_special_moves_and_weight_changes(fen, colour):
    torch.manual_seed(18)
    net = SurvivalNet(CognitiveConfig()).eval()
    board = chess.Board(fen)
    moves = list(board.legal_moves)
    root = torch.from_numpy(encode(board, colour))
    candidates = torch.from_numpy(encode_after_moves(board, moves, colour))
    for _ in range(2):
        torch.testing.assert_close(incremental_encode(net.encoder, root, candidates),
                                   net.encode(candidates), atol=1e-6, rtol=1e-5)
        with torch.no_grad(): net.encoder[0].weight.add_(.001)


@pytest.mark.parametrize('native', [True, False])
def test_search_batches_preserve_branches_probabilities_and_board(native):
    torch.set_num_threads(1); torch.manual_seed(3)
    cfg = CognitiveConfig(planning_uncertainty=0, planning_nodes=37)
    net = SurvivalNet(cfg).eval()
    reference = CognitiveAgent(net, cfg, CognitiveKnowledge(cfg, net.encoder))
    quick_cfg = copy.deepcopy(cfg); quick_cfg.batch_search = True
    quick = CognitiveAgent(net, quick_cfg, CognitiveKnowledge(quick_cfg, net.encoder))
    reference.use_fast_rules = quick.use_fast_rules = native
    board = chess.Board()
    rng = np.random.default_rng(16)
    for _ in range(18):
        if board.is_game_over(): break
        before = board.fen(), tuple(board.move_stack)
        a = reference.evaluate(board, True, explore=False)
        b = quick.evaluate(board, True, explore=False)
        # evaluate returns legal moves, final utilities and diagnostic payload.
        assert before == (board.fen(), tuple(board.move_stack))
        assert a.moves == b.moves
        np.testing.assert_allclose(a.scores, b.scores, rtol=1e-5, atol=1e-6)
        assert reference.diagnostics == quick.diagnostics
        moves = list(board.legal_moves)
        board.push(moves[int(rng.integers(len(moves)))])
