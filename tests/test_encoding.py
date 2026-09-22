from __future__ import annotations

import chess
import numpy as np
import pytest

from tonkai import encoding as E


def test_sizes():
    assert E.INPUT_SIZE == 782
    assert E.PLANE_SIZE == 768
    assert E.PACKED_SIZE == 99


def test_startpos_planes_from_white():
    b = chess.Board()
    v = E.encode(b, chess.WHITE)
    planes = v[: E.PLANE_SIZE].reshape(12, 64)
    # 8 pawns and 1 king for each side
    assert planes[0].sum() == 8 and planes[5].sum() == 1     # self pawns/king
    assert planes[6].sum() == 8 and planes[11].sum() == 1    # opponent
    # white king really is on e1 (square 4) in white's orientation
    assert planes[5][chess.E1] == 1.0
    assert planes[11][chess.E8] == 1.0
    assert v[E.OFF_STM] == 1.0  # white to move, and self is white


def test_startpos_is_identical_from_both_sides():
    """The start position is symmetric, so both orientations must agree.

    This is the test that catches a broken vertical flip.
    """
    b = chess.Board()
    w = E.encode(b, chess.WHITE)
    x = E.encode(b, chess.BLACK)
    assert np.array_equal(w[: E.PLANE_SIZE], x[: E.PLANE_SIZE])


def test_black_orientation_puts_own_king_on_e1_square():
    b = chess.Board()
    v = E.encode(b, chess.BLACK)
    planes = v[: E.PLANE_SIZE].reshape(12, 64)
    # black's king is on e8; flipped, it lands on index 4 (e1)
    assert planes[5][4] == 1.0
    assert planes[11][60] == 1.0
    assert v[E.OFF_STM] == 0.0  # white to move but self is black


def test_after_1_e4_black_sees_a_self_pawn_flipped():
    b = chess.Board()
    b.push_san("e4")
    v = E.encode(b, chess.BLACK)
    planes = v[: E.PLANE_SIZE].reshape(12, 64)
    # the white pawn on e4 is an OPPONENT pawn, flipped e4 -> e5
    assert planes[6][chess.E4 ^ 56] == 1.0
    assert planes[6][chess.E5] == 1.0


def test_castling_and_ep_and_halfmove():
    b = chess.Board("r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 7 20")
    v = E.encode(b, chess.WHITE)
    assert v[E.OFF_CASTLE + 0] == 1.0  # self K
    assert v[E.OFF_CASTLE + 1] == 1.0  # self Q
    assert v[E.OFF_CASTLE + 2] == 1.0  # opp K
    assert v[E.OFF_CASTLE + 3] == 1.0  # opp Q
    assert v[E.OFF_EP + chess.square_file(chess.D6)] == 1.0
    assert v[E.OFF_HALFMOVE] == pytest.approx(0.07)

    v2 = E.encode(chess.Board(), chess.WHITE)
    assert v2[E.OFF_EP : E.OFF_EP + 8].sum() == 0.0


def test_castling_flags_swap_with_perspective():
    b = chess.Board("r3k2r/8/8/8/8/8/8/R3K3 w Qkq - 0 1")
    w = E.encode(b, chess.WHITE)
    assert (w[E.OFF_CASTLE + 0], w[E.OFF_CASTLE + 1]) == (0.0, 1.0)
    assert (w[E.OFF_CASTLE + 2], w[E.OFF_CASTLE + 3]) == (1.0, 1.0)
    x = E.encode(b, chess.BLACK)
    assert (x[E.OFF_CASTLE + 0], x[E.OFF_CASTLE + 1]) == (1.0, 1.0)
    assert (x[E.OFF_CASTLE + 2], x[E.OFF_CASTLE + 3]) == (0.0, 1.0)


def test_pack_unpack_roundtrip_is_exact():
    rng = np.random.default_rng(0)
    b = chess.Board()
    vecs = []
    for _ in range(60):
        moves = list(b.legal_moves)
        if not moves:
            break
        b.push(moves[rng.integers(len(moves))])
        vecs.append(E.encode(b, chess.WHITE))
        vecs.append(E.encode(b, chess.BLACK))
    packed = np.stack([E.pack(v) for v in vecs])
    back = E.unpack_many(packed)
    assert back.shape == (len(vecs), E.INPUT_SIZE)
    np.testing.assert_allclose(back, np.stack(vecs), atol=1e-6)


def test_encode_after_moves_matches_one_by_one():
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    moves = list(b.legal_moves)
    batch = E.encode_after_moves(b, moves, chess.WHITE)
    assert batch.shape == (len(moves), E.INPUT_SIZE)
    for i, mv in enumerate(moves):
        b.push(mv)
        np.testing.assert_allclose(batch[i], E.encode(b, chess.WHITE))
        b.pop()
    # and the board is untouched afterwards
    assert b.fen() == (
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    )


def test_oriented_square_roundtrip():
    for sq in range(64):
        for color in (chess.WHITE, chess.BLACK):
            assert E.real_square(E.oriented_square(sq, color), color) == sq
