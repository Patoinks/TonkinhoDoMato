import io
import json

import chess
import chess.pgn

from tonkai.opening_history import OpeningHistory


def board(line):
    result = chess.Board()
    for san in line.split():
        result.push_san(san)
    return result


def test_openings_are_ordered_bounded_and_separated_by_colour(tmp_path):
    audit = OpeningHistory(tmp_path)
    for number, line, colour in [(4, "d4 d5", False), (2, "e4 e5", True),
                                 (1, "e4 c5", True), (5, "e4 e5", False), (3, "d4 Nf6", True)]:
        audit.add(number, board(line), colour, "external")
    audit.add(6, board("h4"), True, "external")
    report = json.loads((tmp_path / "opening-report.json").read_text())
    assert report["complete"]
    assert [g["game"] for g in report["games"]] == [1, 2, 3, 4, 5]
    assert report["by_colour"]["white"]["first_moves"] == {"e4": 2, "d4": 1}
    assert report["by_colour"]["white"]["same_first_move"] is False
    stream = io.StringIO((tmp_path / "first-five.pgn").read_text())
    games = [chess.pgn.read_game(stream) for _ in range(5)]
    assert all(g is not None and not g.errors for g in games)
    assert chess.pgn.read_game(stream) is None
    restored = OpeningHistory(tmp_path, audit.state())
    assert restored.report() == report


def test_same_first_move_does_not_mean_same_opening(tmp_path):
    audit = OpeningHistory(tmp_path)
    audit.add(1, board("e4 e5"), True, "external")
    assert audit.report()["by_colour"]["white"]["same_first_move"] is None
    audit.add(2, board("e4 c5"), True, "external")
    white = audit.report()["by_colour"]["white"]
    assert white["same_first_move"] is True
    assert white["same_first_ten_plies"] is False
