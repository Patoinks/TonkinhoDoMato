"""Native learning inputs must preserve piece identity and inference exactly."""
import random

import chess
import numpy as np
import pytest
import torch

from tonkai import fast_rules
from tonkai.captures import activity, capture_targets, track
from tonkai.cognitive_agent import CognitiveAgent
from tonkai.cognitive_config import CognitiveConfig
from tonkai.cognitive_trainer import CognitiveKnowledge
from tonkai.temporal_model import SurvivalNet

pytestmark = pytest.mark.skipif(not fast_rules.AVAILABLE, reason="tonkai_fast is not built")


def check_targets(fen, actions):
    root = chess.Board(fen)
    native = fast_rules.board(root)
    before = native.legal_moves()
    snapshots, capturers, _, _ = track(actions, fen)
    codes = [fast_rules.code(chess.Move.from_uci(a)) for a in actions]
    assert native.activity(codes) == activity(actions, snapshots, fen)
    for colour in (True, False):
        actual = native.capture_targets([fast_rules.code(chess.Move.from_uci(a)) for a in actions], colour)
        expected = capture_targets(snapshots, capturers, colour)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
    assert native.legal_moves() == before


@pytest.mark.parametrize("fen,actions", [
    (chess.STARTING_FEN, "e2e4 a7a6 e4e5 d7d5 e5d6"),  # en passant
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1 e8c8 f1f8 d8f8"),
    ("2r5/1P5k/8/8/8/8/6K1/8 w - - 0 1", "b7b8q h7g6 b8c8"),
    ("2r5/1P5k/8/8/8/8/6K1/8 w - - 0 1", "b7b8n c8b8"),
    (chess.STARTING_FEN, ""),
])
def test_native_capture_targets_special_moves(fen, actions):
    check_targets(fen, actions.split())


def test_native_capture_targets_random_games():
    rng = random.Random(915)
    for _ in range(12):
        board = chess.Board()
        for _ in range(200):
            if board.is_game_over(claim_draw=True):
                break
            board.push(rng.choice(list(board.legal_moves)))
        check_targets(chess.STARTING_FEN, [m.uci() for m in board.move_stack])


def test_native_capture_targets_reject_illegal_history():
    with pytest.raises(ValueError):
        fast_rules.board(chess.Board()).capture_targets([fast_rules.code(chess.Move.from_uci("e2e5"))], True)


def test_embedding_cache_tracks_weight_updates_and_loads():
    cfg = CognitiveConfig()
    net = SurvivalNet(cfg).eval()
    agent = CognitiveAgent(net, cfg, CognitiveKnowledge(cfg, net.encoder))
    board = chess.Board()
    for action in ("e2e4", "d7d5", "e4d5", "d8d5"):
        board.push_uci(action)
    for stage in range(3):
        for colour in (True, False):
            agent.use_fast_rules = False
            slow = agent.piece_valuer(board, colour)
            agent.use_fast_rules = True
            quick = agent.piece_valuer(board, colour)
            assert [slow(p, s) for s, p in board.piece_map().items()] == [quick(p, s) for s, p in board.piece_map().items()]
        if stage == 0:
            with torch.no_grad():
                net.type_embed.weight.add_(.125)
                net.side_embed.weight.mul_(.9)
        elif stage == 1:
            net.load_state_dict(SurvivalNet(cfg).state_dict())


def test_actor_snapshot_omits_only_protected_replay():
    from tonkai.cognitive_memory import ExperienceMemory
    cfg = CognitiveConfig()
    memory = ExperienceMemory(cfg)
    memory.entries = [{"key": np.arange(5), "target": .75}]
    memory.episodes = [{"episode": np.ones((20, 4))}]
    snapshot = memory.state_dict(include_episodes=False)
    assert snapshot["episodes"] == []
    assert snapshot["next_id"] == memory.next_id
    np.testing.assert_array_equal(snapshot["entries"][0]["key"], memory.entries[0]["key"])
    snapshot["entries"][0]["key"][0] = 99
    assert memory.entries[0]["key"][0] == 0
    assert len(memory.episodes) == 1


def test_actor_payload_is_frozen_before_next_learner_update():
    from tonkai.actor_snapshot import unpack_snapshot
    from concurrent.futures import Future
    from tonkai.cognitive_pool import CognitivePool
    cfg = CognitiveConfig()
    net = SurvivalNet(cfg)
    knowledge = CognitiveKnowledge(cfg, net.encoder)
    knowledge.memory.entries = [{"key": np.arange(5), "target": .75}]
    knowledge.memory.episodes = [{"episode": np.ones((20, 4))}]
    sent = []
    class Executor:
        def submit(self, fn, payload, *args):
            sent.append(payload)
            return Future()
    pool = CognitivePool.__new__(CognitivePool)
    pool.pending, pool.capacity, pool.executor = set(), 1, Executor()
    expected_weights = {key: value.clone() for key, value in net.state_dict().items()}
    pool.submit(net, knowledge, 7)
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.add_(1)
    knowledge.memory.entries[0]['key'][0] = 99
    weights, state = unpack_snapshot(sent[0])
    for key in weights:
        assert torch.equal(weights[key], expected_weights[key])
    assert state['memory']['entries'][0]['key'][0] == 0
    assert state['memory']['episodes'] == []
