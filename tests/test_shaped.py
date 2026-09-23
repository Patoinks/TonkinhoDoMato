"""The owner's victory definition ("shaped"), pooled K/D piece values, and
resignation counted as a real loss."""
from __future__ import annotations

from types import SimpleNamespace

import chess
import numpy as np
import pytest
import torch

from tonkai.captures import CaptureStats
from tonkai.cognitive_agent import CognitiveAgent, terminal_value
from tonkai.cognitive_config import CognitiveConfig
from tonkai.cognitive_trainer import CognitiveKnowledge
from tonkai.encoding import encode, pack_many
from tonkai.objective import continuation_value, observed_target, outcome_value
from tonkai.resign import ResignConfig, hope_threshold
from tonkai.sequence_replay import SequenceEpisode, SequenceReplay
from tonkai.survival_agent import play_episode
from tonkai.survival_trainer import SurvivalTrainer
from tonkai.temporal_model import SurvivalNet


def cognitive(**overrides):
    values = dict(device="cpu", inference_device="cpu", hidden_size=32, latent_size=24,
                  memory_size=16, head_hidden=12, piece_hidden=12, ensemble_size=3,
                  future_horizons=(1, 2, 4), burn_in=4, sequence_length=12,
                  sequence_batch_size=4, replay_capacity=512, min_replay=1,
                  max_plies=12, seed=27, planning_nodes=6)
    values.update(overrides)
    return CognitiveConfig(**values)


class RandomEngine:
    description = "random"

    def reset(self):
        pass

    def choose(self, board, rng):
        moves = list(board.legal_moves)
        return moves[int(rng.integers(len(moves)))]


def agent_for(cfg, kd_values=None):
    torch.manual_seed(5)
    net = SurvivalNet(cfg).eval()
    return CognitiveAgent(net, cfg, CognitiveKnowledge(cfg, net.encoder), kd_values=kd_values)


# ---------------------------------------------------------------- objective

def test_shaped_orders_win_above_draw_above_any_loss():
    draw = SimpleNamespace(winner=None)
    assert outcome_value(draw, chess.WHITE, "shaped", 0.5) == pytest.approx(0.75)
    assert outcome_value(draw, chess.WHITE, "survival") == 1.0
    assert outcome_value(draw, chess.WHITE, "victory") == 0.5
    mate = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    mate.push_san("Ra8#")
    assert terminal_value(mate, chess.WHITE, "shaped", 0.5) == 1.0
    assert terminal_value(mate, chess.BLACK, "shaped", 0.5) == 0.0


def test_observed_target_blends_survival_with_the_victory_measure():
    cfg = cognitive(objective="shaped", victory_weight=0.5)
    board = chess.Board()
    positions = [encode(board, chess.WHITE)]
    for san in ("e4", "e5"):
        board.push_san(san)
        positions.append(encode(board, chess.WHITE))
    episode = SequenceEpisode(pack_many(np.stack(positions)), False, False,
                              actions=tuple(m.uci() for m in board.move_stack),
                              proxies=np.array([0.2, 0.4, 0.8], dtype=np.float32))
    assert observed_target(episode, 1, cfg) == pytest.approx(0.5 * 1.0 + 0.5 * 0.4)
    survival = cognitive(objective="survival")
    assert observed_target(episode, 1, survival) == 1.0


def test_shaped_continuation_only_credits_survival_for_living():
    cfg = cognitive(objective="shaped", victory_weight=0.5)
    survival = cognitive(objective="survival")
    assert continuation_value(0.4, 1, cfg) == pytest.approx(
        0.5 * continuation_value(0.4, 1, survival) + 0.5 * 0.4)


def test_shaped_resign_threshold():
    rc = ResignConfig()
    expected = 0.5 * (1 - 0.985 ** 10) + 0.5 * 0.25
    assert hope_threshold(rc, "shaped", 0.985, 0.5) == pytest.approx(expected)


# ------------------------------------------------------------- pooled K/D

def test_pooled_kd_pools_both_sides_relative_to_a_pawn():
    stats = CaptureStats()
    for side in ("self", "opponent"):
        for name in ("Pawn", "Knight", "Bishop", "Rook", "Queen"):
            stats.deaths[side][name] = 100
    stats.kills["self"]["Pawn"], stats.kills["opponent"]["Pawn"] = 50, 50      # 100/200
    stats.kills["self"]["Queen"], stats.kills["opponent"]["Queen"] = 100, 300  # 400/200
    pooled = stats.pooled_kd()
    assert pooled[chess.PAWN] == pytest.approx(1.0)
    assert pooled[chess.QUEEN] == pytest.approx(4.0)
    assert stats.summary()["kd_pooled_vs_pawn"]["Queen"] == pytest.approx(4.0)


def test_pooled_kd_waits_for_enough_deaths():
    assert CaptureStats().pooled_kd() is None


# ------------------------------------------------------ K/D in piece values

def test_piece_values_blend_kd_with_the_network():
    board = chess.Board()
    plain = agent_for(cognitive())
    learned = plain.piece_valuer(board, chess.WHITE)
    kd = {1: 1.0, 2: 1.9, 3: 1.9, 4: 0.7, 5: 3.1}
    blended = agent_for(cognitive(kd_weight=0.75), kd_values=kd).piece_valuer(board, chess.WHITE)
    own_queen, enemy_queen = chess.Piece(chess.QUEEN, chess.WHITE), chess.Piece(chess.QUEEN, chess.BLACK)
    assert blended(own_queen, chess.D1) == pytest.approx(0.75 * 3.1 + 0.25 * learned(own_queen, chess.D1))
    assert blended(enemy_queen, chess.D8) == pytest.approx(-0.75 * 3.1 + 0.25 * learned(enemy_queen, chess.D8))
    assert blended(chess.Piece(chess.KING, chess.WHITE), chess.E1) == 10000.0


def test_kd_weight_zero_keeps_network_values():
    board = chess.Board()
    a = agent_for(cognitive()).piece_valuer(board, chess.WHITE)
    b = agent_for(cognitive(kd_weight=0.0), kd_values={1: 9.0}).piece_valuer(board, chess.WHITE)
    piece = chess.Piece(chess.PAWN, chess.WHITE)
    assert a(piece, chess.E2) == b(piece, chess.E2)


# ------------------------------------------------------ victory measure

def test_position_proxy_is_bounded_and_ends_override_it():
    agent = agent_for(cognitive(objective="shaped"))
    board = chess.Board()
    agent.observe(board, chess.WHITE)
    assert 0.0 <= agent.position_proxy(board, chess.WHITE) <= 1.0
    mate = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    mate.push_san("Ra8#")
    assert agent.position_proxy(mate, chess.WHITE) == 1.0
    assert agent.position_proxy(mate, chess.BLACK) == 0.0


def test_shaped_games_record_one_measure_per_state():
    cfg = cognitive(objective="shaped")
    agent = agent_for(cfg)
    episode, _, _, _ = play_episode(agent, RandomEngine(), cfg, np.random.default_rng(4))
    assert episode.proxies is not None and len(episode.proxies) == len(episode.packed)
    assert ((episode.proxies >= 0) & (episode.proxies <= 1)).all()


def test_other_objectives_record_no_measure():
    cfg = cognitive()
    episode, _, _, _ = play_episode(agent_for(cfg), RandomEngine(), cfg, np.random.default_rng(4))
    assert episode.proxies is None


# ------------------------------------------------------ replay and learning

def test_replay_keeps_and_validates_the_measures():
    cfg = cognitive(objective="shaped")
    episode, _, _, _ = play_episode(agent_for(cfg), RandomEngine(), cfg, np.random.default_rng(4))
    replay = SequenceReplay(512, seed=1)
    replay.add(episode)
    other = SequenceReplay(512, seed=2)
    other.load_state_dict(replay.state_dict())
    stored = next(iter(other._episodes.values()))
    assert np.array_equal(stored.proxies, episode.proxies)
    bad = SequenceEpisode(episode.packed, False, False, proxies=np.full(len(episode.packed), 2.0))
    with pytest.raises(ValueError):
        replay.add(bad)


def test_shaped_training_uses_the_measures():
    cfg = cognitive(objective="shaped", victory_weight=0.5)
    trainer = SurvivalTrainer(cfg)
    agent = agent_for(cfg)
    for seed in range(4):
        episode, _, _, _ = play_episode(agent, RandomEngine(), cfg, np.random.default_rng(seed))
        trainer.replay.add(episode)
    batch = trainer.replay.sample(4, cfg.burn_in, cfg.sequence_length, 4)
    values, present = trainer._proxy_targets(batch, cfg.burn_in, cfg.sequence_length)
    assert present.any()
    assert values[present].min() >= 0 and values[present].max() <= 1
    assert trainer.update() is not None


# ------------------------------------------- mobility and threats (1/3 each)

def test_threats_count_only_defended_pieces():
    from tonkai.captures import activity, track
    fen = "4k3/8/8/4p3/3N4/8/8/3R2K1 b - - 0 1"          # e5 pawn hits Nd4, defended by Rd1
    snapshots, _, _, _ = track(["e8e7"], fen)
    rows = activity(["e8e7"], snapshots, fen)[chess.BLACK]
    assert rows["threats"][chess.PAWN] == 1 and rows["threats_n"][chess.PAWN] == 1
    undefended = "4k3/8/8/4p3/3N4/8/8/6K1 b - - 0 1"
    snapshots, _, _, _ = track(["e8e7"], undefended)
    assert activity(["e8e7"], snapshots, undefended)[chess.BLACK]["threats"][chess.PAWN] == 0


def test_mobility_counts_only_after_the_first_move():
    from tonkai.captures import activity, track
    fen = "4k3/8/8/4p3/3N4/8/8/3R2K1 b - - 0 1"
    moves = ["e5e4", "g1g2", "e4e3"]
    snapshots, _, _, _ = track(moves, fen)
    black = activity(moves, snapshots, fen)[chess.BLACK]
    assert black["mobility_n"][chess.PAWN] == 1        # sampled once, after e5-e4
    assert black["mobility"][chess.PAWN] == 1          # e4-e3 is its only square
    white = activity(moves, snapshots, fen)[chess.WHITE]
    assert sum(white["mobility_n"].values()) == 0      # no white piece other than the king moved


def test_piece_values_are_a_third_of_each_measure():
    stats = CaptureStats()
    names = ("Pawn", "Knight", "Bishop", "Rook", "Queen")
    assert stats.piece_values() is None
    for side in ("self", "opponent"):
        for name in names:
            stats.activity[side]["mobility_n"][name] = 500
            stats.activity[side]["threats_active_n"][name] = 500
            stats.activity[side]["mobility"][name] = 500
            stats.activity[side]["threats_active"][name] = 100
            stats.pairs[side][f"{name}>{name}"] = 100               # every type trades 1:1 with itself
    stats.pair_games = 10
    stats.activity["self"]["mobility"]["Rook"] = 2500          # rook: (500+2500)/1000 = 3 pawns
    stats.activity["opponent"]["threats_active"]["Rook"] = 500   # rook: 600/1000 vs pawn 200/1000 = 3 pawns
    values = stats.piece_values()
    # Level-weighted K/D of the rook = its own value v; v = (v + 3 + 3) / 3 -> v = 3.
    assert values[chess.PAWN] == pytest.approx(1.0)
    assert values[chess.ROOK] == pytest.approx(3.0)
    restored = CaptureStats()
    restored.load_state_dict(stats.state_dict())
    assert restored.piece_values() == values
    assert restored.summary()["piece_values_vs_pawn"]["Rook"] == pytest.approx(3.0)
