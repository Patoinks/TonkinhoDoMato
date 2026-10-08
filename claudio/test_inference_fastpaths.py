"""Native search requests must remain independent when table storage is reused."""
import math
from pathlib import Path
import sys

import chess
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: F401,E402
from claudio_agent import native_board  # noqa: E402


def test_search_storage_does_not_reuse_another_requests_piece_values():
    board = native_board(chess.Board('4k3/8/8/8/8/8/4q3/4K3 w - - 0 1'))
    for queen in (9.0, 20.0, 2.0, 9.0):
        values = [0.0, 1.0, 2.0, 3.0, 4.0, queen]
        assert board.forced_gain(values, 2, 3, 100_000) == pytest.approx(queen)


def test_search_storage_cannot_bypass_a_new_node_budget():
    board = native_board(chess.Board())
    values = [0.0, 1.0, 2.0, 3.0, 4.0, 9.0]
    full = board.forced_gain(values, 3, 3, 100_000)
    assert math.isfinite(full)
    assert math.isnan(board.forced_gain(values, 3, 3, 0))
    assert math.isnan(board.forced_gain(values, 3, 3, 1))
    assert board.forced_gain(values, 3, 3, 100_000) == full
