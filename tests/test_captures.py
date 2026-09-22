"""Capture tracking: the auxiliary target and the K/D measurement."""
from __future__ import annotations

import dataclasses
from copy import deepcopy

import chess
import numpy as np
import pytest
import torch

from tonkai.captures import (CaptureStats, capture_targets, episode_capture_targets,
                             episode_owner_color, track)
from tonkai.cognitive_config import CognitiveConfig
from tonkai.encoding import encode, pack_many
from tonkai.sequence_replay import SequenceEpisode
from tonkai.survival_config import SurvivalConfig
from tonkai.survival_trainer import SurvivalTrainer
from tonkai.temporal_model import SurvivalNet

OPERA_GAME = (
    "e4 e5 Nf3 d6 d4 Bg4 dxe5 Bxf3 Qxf3 dxe5 Bc4 Nf6 Qb3 Qe7 "
    "Nc3 c6 Bg5 b5 Nxb5 cxb5 Bxb5+ Nbd7 O-O-O Rd8 Rxd7 Rxd7 "
    "Rd1 Qe6 Bxd7+ Nxd7 Qb8+ Nxb8 Rd8#"
).split()


def ucis(sans, fen=chess.STARTING_FEN):
    board = chess.Board(fen)
    out = []
    for san in sans:
        move = board.parse_san(san)
        out.append(move.uci())
        board.push(move)
    return out


def episode(sans, colour):
    board = chess.Board()
    positions = [encode(board, colour)]
    for san in sans:
        board.push_san(san)
        positions.append(encode(board, colour))
    outcome = board.outcome(claim_draw=True)
    lost = bool(outcome and outcome.winner is not None and outcome.winner != colour)
    return SequenceEpisode(pack_many(np.stack(positions)), lost, outcome is not None,
                           opponent="test", surprises=np.zeros(len(positions), dtype=np.float32),
                           actions=tuple(m.uci() for m in board.move_stack))


# ------------------------------------------------------------------ tracking

def test_identity_survives_the_whole_opera_game():
    snapshots, capturer, events, board = track(ucis(OPERA_GAME))
    assert len(snapshots) == len(OPERA_GAME) + 1 == len(capturer) + 1
    assert set(snapshots[-1]) == set(board.piece_map())
    assert len(events) == sum(1 for c in capturer if c is not None)
    assert board.is_checkmate()


def test_castling_moves_the_rook_identity():
    snapshots, _, _, _ = track(ucis("e4 e5 Nf3 Nc6 Bc4 Bc5 O-O".split()))
    before, after = snapshots[-2], snapshots[-1]
    assert after[chess.F1] == before[chess.H1]           # the same rook
    assert after[chess.G1] == before[chess.E1]           # the same king


def test_en_passant_removes_the_right_pawn():
    moves = ucis("e4 a6 e5 d5 exd6".split())
    _, _, events, board = track(moves)
    assert events[-1]["victim_type"] == chess.PAWN
    assert board.piece_at(chess.D5) is None


def test_promotion_keeps_the_pawns_identity():
    fen = "8/1P5k/8/8/8/8/6K1/8 w - - 0 1"
    snapshots, _, _, _ = track(["b7b8q"], fen)
    assert snapshots[1][chess.B8] == snapshots[0][chess.B7]


def test_rejects_illegal_moves():
    with pytest.raises(ValueError):
        track(["e2e5"])


# ------------------------------------------------------------------- targets

def test_targets_count_future_captures_per_state():
    moves = ucis("e4 d5 exd5 Qxd5".split())
    snapshots, capturer, _, _ = track(moves)
    white = capture_targets(snapshots, capturer, chess.WHITE)
    assert white.shape == (5, 64)
    assert white[0, chess.E2] == 1          # this pawn will capture once
    assert white[0, chess.D8] == 1          # so will the black queen
    assert white[2, chess.E4] == 1
    assert white[3, chess.D5] == 0          # the pawn is about to be taken
    assert white[3, chess.D8] == 1
    assert white[4].sum() == 0              # nothing captures after the last move
    black = capture_targets(snapshots, capturer, chess.BLACK)
    assert black[0, chess.E2 ^ 56] == 1 and black[0, chess.D8 ^ 56] == 1


def test_episode_targets_align_and_orient_to_the_owner():
    for colour in (chess.WHITE, chess.BLACK):
        ep = episode(OPERA_GAME, colour)
        assert episode_owner_color(ep) == colour
        table = episode_capture_targets(ep)
        assert table.shape == (len(ep.packed), 64)
        snapshots, capturer, _, _ = track(ep.actions)
        assert np.array_equal(table, capture_targets(snapshots, capturer, colour))


def test_episode_without_actions_has_no_targets():
    ep = dataclasses.replace(episode(["e4"], chess.WHITE), actions=())
    assert episode_capture_targets(ep) is None


# ------------------------------------------------------------- measurement

def test_hot_capture_is_measured_with_the_game_outcome():
    # Black's queen captures twice (d5, then g2); then white's bishop takes it.
    sans = "e4 d5 exd5 Qxd5 Nc3 Qxg2 Bxg2".split()
    stats = CaptureStats(hot_kills=2)
    stats.add(ucis(sans), chess.WHITE, "win")
    stats.add(ucis(OPERA_GAME[:4]), chess.WHITE, "loss")      # no captures at all
    summary = stats.summary()
    assert summary["hot_captures_by_victim"] == {"Queen": 1}
    assert summary["games_with_hot_capture"]["win"] == 1
    assert summary["games_without_hot_capture"]["loss"] == 1
    assert summary["kd_opponent"]["Queen"]["kills"] == 2
    assert summary["kd_opponent"]["Queen"]["deaths"] == 1
    assert summary["kd_self"]["Pawn"]["deaths"] == 2
    assert summary["kd_self"]["Bishop"]["kills"] == 1
    other = CaptureStats(hot_kills=2)
    other.load_state_dict(stats.state_dict())
    assert other.summary() == summary


def test_unknown_outcome_is_rejected():
    with pytest.raises(ValueError):
        CaptureStats().add([], chess.WHITE, "victory")


# ------------------------------------------------------ auxiliary target

def cognitive(**overrides):
    values = dict(device="cpu", inference_device="cpu", hidden_size=32, latent_size=24,
                  memory_size=16, head_hidden=12, piece_hidden=12, ensemble_size=3,
                  future_horizons=(1, 2, 4), burn_in=4, sequence_length=12,
                  sequence_batch_size=4, replay_capacity=512, min_replay=1,
                  max_plies=12, seed=27)
    values.update(overrides)
    return CognitiveConfig(**values)


def trainer_with(cfg):
    torch.manual_seed(71)
    trainer = SurvivalTrainer(cfg)
    for colour in (chess.WHITE, chess.BLACK):
        trainer.replay.add(episode(OPERA_GAME, colour))
        trainer.games += 1
    return trainer


def test_capture_head_exists_only_when_requested_and_is_registered_last():
    assert not SurvivalNet(cognitive()).has_capture_head
    assert not SurvivalNet(SurvivalConfig(device="cpu")).has_capture_head
    net = SurvivalNet(cognitive(capture_target_weight=0.1))
    assert net.has_capture_head
    names = [name for name, p in net.named_parameters() if p.requires_grad]
    assert names[-2:] == ["capture_out.weight", "capture_out.bias"]


def test_value_path_is_unchanged_by_the_capture_head():
    torch.manual_seed(3)
    net = SurvivalNet(cognitive(capture_target_weight=0.1))
    context = torch.randn(5, net.latent_size)
    kind, square, side = torch.tensor([0, 1, 2, 3, 4]), torch.arange(5), torch.tensor([0, 1, 0, 1, 0])
    value, captures = net.piece_values_and_captures(context, kind, square, side)
    assert torch.equal(value, net.piece_values(context, kind, square, side))
    assert captures.shape == (5,)


def test_update_trains_the_capture_head():
    trainer = trainer_with(cognitive(capture_target_weight=0.1))
    before = trainer.net.capture_out.weight.detach().clone()
    losses = [trainer.update() for _ in range(3)]
    assert all("capture_loss" in l and np.isfinite(l["capture_loss"]) for l in losses)
    assert not torch.equal(before, trainer.net.capture_out.weight)


def test_capture_off_changes_nothing():
    trainer = trainer_with(cognitive())
    assert "capture_loss" not in trainer.update()


@pytest.mark.parametrize("source_weight,target_weight", [(0.0, 0.1), (0.1, 0.0)])
def test_checkpoints_migrate_when_the_capture_head_is_turned_on_or_off(source_weight, target_weight):
    source = trainer_with(cognitive(capture_target_weight=source_weight))
    source.update()
    state = deepcopy(source.state_dict())
    target = SurvivalTrainer(cognitive(capture_target_weight=target_weight))
    target.load_state_dict(state)
    shared = {k: v for k, v in source.net.state_dict().items() if not k.startswith("capture_out.")}
    for key, value in shared.items():
        assert torch.equal(value, target.net.state_dict()[key]), key
    assert target.update() is not None                 # optimizer state matches the model


def test_targets_line_up_with_the_encoded_positions_in_a_replay_batch():
    """Every square with a future-capture target must hold a piece in the very
    position the learner sees - catches misaligned states and wrong orientation."""
    from tonkai.encoding import unpack_many
    trainer = trainer_with(cognitive(capture_target_weight=0.1))
    checked = nonzero = 0
    for _ in range(10):
        batch = trainer.replay.sample(4, 4, 12, 4)
        for i, ep in enumerate(batch["episodes"]):
            table = episode_capture_targets(ep)
            for j, state in enumerate(batch["state_indices"][i]):
                if state < 0:
                    continue
                x = unpack_many(batch["packed"][i, j][None])[0]
                occupied = x[:768].reshape(12, 64).sum(0) > 0
                wanted = table[state] > 0
                assert occupied[wanted].all(), (i, j, int(state))
                checked += 1
                nonzero += int(wanted.sum())
    assert checked > 100 and nonzero > 0
