"""Mate calculation in play: TonkAI's exact, rules-only mate search (tonkai_fast).

Attack: a forced mate within mate_search_moves is played along a shortest line.
Defence: a move after which the opponent forces mate within mate_defense_moves
ranks under every other move. Checked here against python-chess, not against
the C++ search itself."""
from types import SimpleNamespace

import chess
import numpy as np
import pytest
import torch

from tonkai import fast_rules
from tonkai.cognitive_agent import CognitiveAgent
from tonkai.cognitive_config import CognitiveConfig
from tonkai.cognitive_trainer import CognitiveKnowledge
from tonkai.resign import ACTIVE, ResignTracker
from tonkai.survival_agent import SurvivalChoice, play_episode
from tonkai.temporal_model import SurvivalNet

pytestmark = pytest.mark.skipif(not fast_rules.AVAILABLE, reason="tonkai_fast is not built")

MATE_IN_TWO = "3r2k1/5ppp/8/8/8/8/4RPPP/4R1K1 w - - 0 1"     # 1.Re8+ Rxe8 2.Rxe8#
BACK_RANK = "4r1k1/R4ppp/8/8/8/8/5PPP/6K1 w - - 0 1"        # ...Re1# unless White sees it


def agent(seed=0, **mates):
    cfg = CognitiveConfig(device="cpu", inference_device="cpu", hidden_size=32, latent_size=24, memory_size=16,
                          head_hidden=12, piece_hidden=12, ensemble_size=3, future_horizons=(1, 2, 4), burn_in=4,
                          sequence_length=12, sequence_batch_size=4, replay_capacity=512, min_replay=1,
                          max_plies=60, seed=27, planning_nodes=24, objective="shaped", **mates)
    torch.manual_seed(seed)
    net = SurvivalNet(cfg).eval()
    tonkai = CognitiveAgent(net, cfg, CognitiveKnowledge(cfg, net.encoder))
    tonkai.reset()
    return tonkai


def mates_now(board):
    """The side to move has a checkmate in one."""
    for move in list(board.legal_moves):
        board.push(move)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            return True
    return False


def allows_mate_in_one(board, move):
    board.push(move)
    try:
        return mates_now(board)
    finally:
        board.pop()


def test_plays_a_forced_mate_along_the_shortest_line():
    for seed in range(3):
        board = chess.Board(MATE_IN_TWO)
        assert not mates_now(board)
        tonkai = agent(seed, mate_search_moves=3)
        choice = tonkai.choose(board, chess.WHITE, np.random.default_rng(seed), explore=True)
        assert choice.forced_mate == 2 and tonkai.trace[-1]["forced_mate"] == 2
        board.push(choice.move)
        tonkai.observe(board, chess.WHITE)                          # as play_episode does, every ply
        assert all(allows_mate_in_one(board, reply) for reply in list(board.legal_moves))
        board.push(next(iter(board.legal_moves)))                  # 1...Rxe8, the only reply
        tonkai.observe(board, chess.WHITE)
        choice = tonkai.choose(board, chess.WHITE, np.random.default_rng(seed), explore=True)
        board.push(choice.move)
        assert board.is_checkmate()
        assert tonkai.diagnostics["forced_mate_moves"] == 1         # the mate itself is the known ending


def test_without_the_settings_nothing_is_searched():
    tonkai = agent()
    choice = tonkai.choose(chess.Board(MATE_IN_TWO), chess.WHITE, np.random.default_rng(0), explore=False)
    assert choice.forced_mate == 0
    assert not any(key.startswith("mate_") for key in tonkai.diagnostics)
    assert tonkai._evaluation["allows_mate"] == {}


def test_never_allows_a_mate_that_can_be_avoided():
    board = chess.Board(BACK_RANK)
    losing = {move for move in board.legal_moves if allows_mate_in_one(board, move)}
    assert 0 < len(losing) < board.legal_moves.count()
    for seed in range(4):
        tonkai = agent(seed, mate_defense_moves=1)
        choice = tonkai.choose(board, chess.WHITE, np.random.default_rng(seed), explore=False)
        assert choice.move not in losing
        flagged = tonkai._evaluation["allows_mate"]
        assert {choice.moves[index] for index in flagged} == losing and set(flagged.values()) == {1}
        compared = [score for move, score in zip(choice.moves, choice.scores) if move not in losing and score > -1e8]
        assert max(choice.scores[index] for index in flagged) < min(compared)
        assert tonkai.diagnostics["mate_moves_avoided"] == len(losing)
        assert board.fen() == BACK_RANK                              # the search leaves the board alone


def test_forced_losses_rank_under_every_compared_move():
    scores = np.array([.2, -1e9, .9, -.5, .4])                       # -1e9: left out by the planner
    CognitiveAgent._rank_last(scores, {2: 1, 4: 2})
    assert list(np.argsort(-scores)) == [0, 3, 4, 2, 1]
    every = np.array([.3, .8, -1e9])
    CognitiveAgent._rank_last(every, {0: 2, 1: 1, 2: 1})
    assert every[0] > every[1] and every[2] == -1e9                  # all lose: the later mate first


class Mater:
    """Mates whenever it can, otherwise captures first."""
    description = "mater"

    def reset(self):
        pass

    def choose(self, board, rng):
        moves = list(board.legal_moves)
        for move in moves:
            board.push(move)
            mate = board.is_checkmate()
            board.pop()
            if mate:
                return move
        moves.sort(key=lambda m: (not board.is_capture(m), m.uci()))
        return moves[0] if rng.random() < .6 else moves[int(rng.integers(len(moves)))]


def test_a_whole_game_never_walks_into_an_avoidable_mate():
    tonkai = agent(3, mate_search_moves=2, mate_defense_moves=1)
    original, checked = tonkai.choose, []

    def choose(board, colour, rng, **kwargs):
        choice = original(board, colour, rng, **kwargs)
        losing = {move for move in choice.moves if allows_mate_in_one(board, move)}
        assert choice.move not in losing or len(losing) == len(choice.moves)
        checked.append(len(losing))
        return choice
    tonkai.choose = choose
    episode, board, colour, stats = play_episode(tonkai, Mater(), tonkai.cfg, np.random.default_rng(5))
    assert checked and 0 < tonkai.diagnostics["mate_attack_searches"] <= len(checked)   # not after mate-in-one
    assert tonkai.diagnostics["mate_defence_searches"] > 0


class HopelessStub:
    """The network says hopeless on every move; the rules may know better."""

    def __init__(self, forced_mate):
        self.forced_mate, self.trace = forced_mate, []

    def reset(self):
        self.trace = []

    def observe(self, board, colour):
        return None

    def choose(self, board, colour, rng, **_):
        moves = list(board.legal_moves)
        n = len(moves)
        self.trace.append({"ply": len(board.move_stack)})
        return SurvivalChoice(moves, np.full(n, .99), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n),
                              forced_mate=self.forced_mate)

    def predict_reply(self, board, colour):
        moves = list(board.legal_moves)
        return moves, np.full(len(moves), 1 / len(moves))


class FirstMove:
    def reset(self):
        pass

    def choose(self, board, rng):
        return sorted(board.legal_moves, key=lambda m: m.uci())[0]


@pytest.mark.parametrize("forced_mate, resigns", [(0, True), (3, False)])
def test_never_resigns_with_a_forced_mate(forced_mate, resigns):
    cfg = SimpleNamespace(ensemble_size=3, temperature=0.1, max_plies=40)
    tracker = ResignTracker(ACTIVE, 0.05, patience=1, min_ply=0)
    _, _, _, stats = play_episode(HopelessStub(forced_mate), FirstMove(), cfg, np.random.default_rng(1),
                                  self_color=chess.WHITE, resign=tracker)
    assert stats["resign"]["resigned"] is resigns


def test_settings_are_validated():
    with pytest.raises(ValueError):
        CognitiveConfig(mate_search_moves=-1).validate()
    with pytest.raises(ValueError):
        CognitiveConfig(mate_node_limit=0).validate()
    old = CognitiveConfig.from_dict({key: value for key, value in CognitiveConfig().__dict__.items()
                                     if not key.startswith("mate_")})
    assert (old.mate_search_moves, old.mate_defense_moves) == (0, 0)  # older checkpoints: off
