"""How pieces die: an equal immediate exchange, an unequal one, or for free."""
import chess

from tonkai.captures import CaptureStats, death_kinds, track


def kinds(sans, fen=chess.STARTING_FEN):
    board = chess.Board(fen)
    for san in sans:
        board.push_san(san)
    return [(e["victim_type"], kind) for e, kind in death_kinds(track(board.move_stack, fen)[2])]


def test_equal_trade_unequal_trade_and_free_loss():
    # exd5 Qxd5: pawn for pawn, both deaths are an equal trade.
    assert kinds(["e4", "d5", "exd5", "Qxd5"]) == [(chess.PAWN, "trade"), (chess.PAWN, "trade")]
    # Knight takes a defended pawn and is taken back by a pawn: unequal.
    fen = "4k3/3p4/4p3/8/8/2N5/8/4K3 w - - 0 1"
    assert kinds(["Nb5", "Kd8", "Nd6", "Kc7", "Nf5", "exf5"], fen)[0][1] == "free"
    fen = "4k3/3p4/4p3/8/8/5N2/8/4K3 w - - 0 1"
    assert kinds(["Ng5", "Ke7", "Nxe6", "dxe6"], fen) == [(chess.PAWN, "unequal"), (chess.KNIGHT, "unequal")]


def test_stats_count_death_kinds_per_side_and_roundtrip():
    board = chess.Board()
    for san in ("e4", "d5", "exd5", "Qxd5"):
        board.push_san(san)
    stats = CaptureStats()
    stats.add(board.move_stack, chess.WHITE, "unfinished")
    assert stats.death_kinds["self"]["Pawn:trade"] == 1 and stats.death_kinds["opponent"]["Pawn:trade"] == 1
    assert stats.free_share("self") == {"Pawn": 0.0}
    restored = CaptureStats()
    restored.load_state_dict(stats.state_dict())
    assert restored.death_kinds == stats.death_kinds
    assert "deaths_free_share" in restored.summary()
