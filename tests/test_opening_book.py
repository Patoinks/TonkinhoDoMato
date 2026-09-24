"""Opening books (owner's choice, not organic): the book move in a book
position of the book's side; everything else is TonkAI's own."""
import json

import chess
import numpy as np

from test_mate_calculation import agent


def book(tmp_path, side, moves):
    positions, board = {}, chess.Board()
    for uci in moves:
        if board.turn == (side == "white"):
            positions[" ".join(board.fen().split(" ")[:4])] = [{"move": uci, "eval": None}]
        board.push_uci(uci)
    path = tmp_path / f"{side}.json"
    path.write_text(json.dumps({"name": side, "side": side, "positions": positions}), encoding="utf-8")
    return str(path)


def test_book_moves_are_played_for_the_book_side_only(tmp_path):
    white = book(tmp_path, "white", ["e2e4", "e7e5", "f1b5"])
    black = book(tmp_path, "black", ["e2e4", "d7d5", "e4d5", "g8f6"])
    books = f"{white},{black}"
    rng = np.random.default_rng(0)
    board = chess.Board()
    as_white = agent(opening_book=books)
    assert as_white.choose(board, chess.WHITE, rng, explore=True).move.uci() == "e2e4"
    board.push_uci("e2e4")
    as_black = agent(opening_book=books)
    assert as_black.choose(board, chess.BLACK, rng, explore=True).move.uci() == "d7d5"
    board.push_uci("e7e5")                                          # the white book's line instead
    assert as_white.choose(board, chess.WHITE, rng, explore=True).move.uci() == "f1b5"
    fresh = agent(opening_book=books)
    fresh.choose(chess.Board(), chess.WHITE, rng, explore=True)
    assert fresh.diagnostics["book_moves"] == 1
    board.push_uci("f1b5")
    board.push_uci("g8f6")                                          # out of the white book
    fresh = agent(opening_book=books)
    fresh.choose(board, chess.WHITE, rng, explore=True)
    assert fresh.diagnostics["book_moves"] == 0


def test_no_book_by_default():
    tonkai = agent()
    tonkai.choose(chess.Board(), chess.WHITE, np.random.default_rng(0), explore=False)
    assert tonkai.diagnostics["book_moves"] == 0
