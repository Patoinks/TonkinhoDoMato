"""Whole games with the C++ rules must be identical to games with the Python ones."""
import chess
import numpy as np
import pytest
import torch

from tonkai import fast_rules
from tonkai.cognitive_agent import CognitiveAgent
from tonkai.cognitive_config import CognitiveConfig
from tonkai.cognitive_trainer import CognitiveKnowledge
from tonkai.survival_agent import play_episode
from tonkai.temporal_model import SurvivalNet

pytestmark = pytest.mark.skipif(not fast_rules.AVAILABLE, reason="tonkai_fast is not built")


class CaptureFirst:
    """A deterministic opponent that trades a lot, so exchanges get probed."""
    description = "capture-first"

    def reset(self):
        pass

    def choose(self, board, rng):
        moves = sorted(board.legal_moves, key=lambda m: (not board.is_capture(m), m.uci()))
        return moves[0] if rng.random() < .6 else moves[int(rng.integers(len(moves)))]


def same(a, b):
    """Deep, exact equality through dicts, lists and numpy arrays."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    return type(a) is type(b) and a == b


def play(fast, seed, objective, kd):
    cfg = CognitiveConfig(device="cpu", inference_device="cpu", hidden_size=32, latent_size=24, memory_size=16,
                          head_hidden=12, piece_hidden=12, ensemble_size=3, future_horizons=(1, 2, 4), burn_in=4,
                          sequence_length=12, sequence_batch_size=4, replay_capacity=512, min_replay=1,
                          max_plies=90, seed=27, planning_nodes=24, objective=objective,
                          kd_weight=.75 if kd else 0.0)
    torch.manual_seed(seed)
    net = SurvivalNet(cfg).eval()
    agent = CognitiveAgent(net, cfg, CognitiveKnowledge(cfg, net.encoder),
                           kd_values={1: 1.0, 2: 1.9, 3: 1.8, 4: 2.2, 5: 3.4} if kd else None)
    agent.use_fast_rules = fast
    scores, original = [], agent.choose

    def choose(*args, **kwargs):
        choice = original(*args, **kwargs)
        scores.append((choice.scores.copy(), choice.index))
        return choice
    agent.choose = choose
    episode, board, colour, stats = play_episode(agent, CaptureFirst(), cfg, np.random.default_rng(seed))
    return episode, board, stats, agent.trace, scores, dict(agent.diagnostics)


@pytest.mark.parametrize("seed,objective,kd", [(1, "survival", False), (2, "shaped", True),
                                               (3, "victory", False), (4, "shaped", False)])
def test_games_are_identical_with_and_without_cpp(seed, objective, kd):
    slow = play(False, seed, objective, kd)
    quick = play(True, seed, objective, kd)
    assert quick[0].actions == slow[0].actions
    assert np.array_equal(quick[0].packed, slow[0].packed)
    if slow[0].proxies is not None:
        assert np.array_equal(quick[0].proxies, slow[0].proxies)
    assert quick[1].fen() == slow[1].fen()
    assert quick[2] == slow[2]
    assert same(quick[3], slow[3])                          # every trace field, king senses included
    assert len(quick[4]) == len(slow[4])
    for (a, i), (b, j) in zip(quick[4], slow[4]):
        assert i == j and np.array_equal(a, b)              # bit-identical move scores
    assert quick[5] == slow[5]


def test_reply_predictions_keep_order_probabilities_and_real_memory():
    from tonkai.survival_agent import SurvivalAgent
    torch.set_num_threads(1)
    cfg = CognitiveConfig()
    net = SurvivalNet(cfg).eval()
    agent = CognitiveAgent(net, cfg, CognitiveKnowledge(cfg, net.encoder))
    board = chess.Board()
    for action in ("e2e4", "e7e5", "g1f3", "b8c6"):
        board.push_uci(action)
    for colour in (True, False):
        moves, probabilities = SurvivalAgent.predict_reply(agent, board, colour)
        memory, context = agent.memory.clone(), agent.context.clone()
        fen, history = board.fen(), list(board.move_stack)
        agent.use_fast_rules = True
        fast_moves, fast_probabilities = agent.predict_reply(board, colour)
        assert fast_moves == moves
        np.testing.assert_array_equal(fast_probabilities, probabilities)
        assert torch.equal(agent.memory, memory) and torch.equal(agent.context, context)
        assert board.fen() == fen and board.move_stack == history
