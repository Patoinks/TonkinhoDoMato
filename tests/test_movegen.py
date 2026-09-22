"""Move generation is a law of physics here: TonkAI must never be able to
exploit a bug in it. These tests are the ones that matter most."""

from __future__ import annotations

import chess
import pytest

import tiny_opponent

# The standard perft suite. Any movegen bug shows up here.
PERFT = [
    ("startpos", chess.STARTING_FEN, [20, 400, 8902, 197281]),
    ("kiwipete",
     "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
     [48, 2039, 97862]),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
     [14, 191, 2812, 43238]),
    ("promotions",
     "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
     [6, 264, 9467]),
    ("position5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
     [44, 1486, 62379]),
    ("position6",
     "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
     [46, 2079, 89890]),
]


@pytest.mark.parametrize("name,fen,counts", PERFT, ids=[p[0] for p in PERFT])
def test_perft_matches_known_values(name, fen, counts):
    for depth, expected in enumerate(counts, start=1):
        assert tiny_opponent.perft(fen, depth) == expected, (
            f"{name} perft({depth})"
        )


@pytest.mark.parametrize("name,fen,counts", PERFT, ids=[p[0] for p in PERFT])
def test_legal_moves_match_python_chess(name, fen, counts):
    mine = sorted(tiny_opponent.TinyOpponent().legal_moves(fen))
    theirs = sorted(m.uci() for m in chess.Board(fen).legal_moves)
    assert mine == theirs


def test_fen_roundtrip():
    for _, fen, _ in PERFT:
        assert tiny_opponent.roundtrip_fen(fen) == chess.Board(fen).fen()


# --------------------------------------------------------------- rule cases

RULE_CASES = [
    # (name, fen, a move that must be legal)
    ("white_kingside_castle", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
    ("white_queenside_castle", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1"),
    ("black_kingside_castle", "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8"),
    ("black_queenside_castle", "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8"),
    ("promotion_queen", "8/P6k/8/8/8/8/8/K7 w - - 0 1", "a7a8q"),
    ("promotion_knight", "8/P6k/8/8/8/8/8/K7 w - - 0 1", "a7a8n"),
    ("en_passant_white", "8/8/8/3pP3/8/8/8/K6k w - d6 0 2", "e5d6"),
    ("en_passant_black", "8/8/8/8/3Pp3/8/8/K6k b - d3 0 2", "e4d3"),
]


@pytest.mark.parametrize("name,fen,move", RULE_CASES, ids=[c[0] for c in RULE_CASES])
def test_special_moves_are_generated(name, fen, move):
    eng = tiny_opponent.TinyOpponent()
    assert move in eng.legal_moves(fen)
    assert move in [m.uci() for m in chess.Board(fen).legal_moves]


def test_cannot_castle_out_of_check():
    # the rook on e8 checks the king on e1, so neither castle is legal
    fen = "4r3/8/8/8/8/8/8/R3K2R w KQ - 0 1"
    board = chess.Board(fen)
    assert board.is_check()
    eng = tiny_opponent.TinyOpponent()
    assert sorted(eng.legal_moves(fen)) == sorted(m.uci() for m in board.legal_moves)
    assert "e1g1" not in eng.legal_moves(fen)
    assert "e1c1" not in eng.legal_moves(fen)


def test_cannot_castle_through_an_attacked_square():
    # the rook on f8 covers f1, so kingside castling is illegal but
    # queenside is fine
    fen = "5r2/8/8/8/8/8/8/R3K2R w KQ - 0 1"
    board = chess.Board(fen)
    eng = tiny_opponent.TinyOpponent()
    assert sorted(eng.legal_moves(fen)) == sorted(m.uci() for m in board.legal_moves)
    assert "e1g1" not in eng.legal_moves(fen)
    assert "e1c1" in eng.legal_moves(fen)


def test_cannot_castle_without_rights():
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1"
    eng = tiny_opponent.TinyOpponent()
    assert "e1g1" not in eng.legal_moves(fen)
    assert "e1c1" not in eng.legal_moves(fen)


def test_pinned_piece_cannot_move():
    # the knight on e2 is pinned to the king on e1 by the rook on e8
    fen = "4r3/8/8/8/8/8/4N3/4K3 w - - 0 1"
    eng = tiny_opponent.TinyOpponent()
    assert sorted(eng.legal_moves(fen)) == sorted(
        m.uci() for m in chess.Board(fen).legal_moves
    )
    assert not any(m.startswith("e2") for m in eng.legal_moves(fen))


def test_en_passant_discovered_check_is_illegal():
    # taking en passant would expose the king on the fifth rank
    fen = "8/8/8/K2pP2r/8/8/8/7k w - d6 0 2"
    eng = tiny_opponent.TinyOpponent()
    assert sorted(eng.legal_moves(fen)) == sorted(
        m.uci() for m in chess.Board(fen).legal_moves
    )
