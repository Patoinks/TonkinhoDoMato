"""Piece values v2 (owner's rule): 1/3 K/D where a kill is worth the victim's
measured value, 1/3 mobility after the first move, 1/3 threats on defended
pieces after the first move."""
import chess
import pytest

from tonkai import fast_rules
from tonkai.captures import CaptureStats, activity, replay_kind, track

NAMES = ("Pawn", "Knight", "Bishop", "Rook", "Queen")


def flat(stats, sides=("self", "opponent")):
    for side in sides:
        for name in NAMES:
            for key in ("mobility", "threats_active"):
                stats.activity[side][key][name] = 500
                stats.activity[side][key + "_n"][name] = 500


def test_eating_high_level_pieces_counts_more():
    stats = CaptureStats()
    flat(stats)
    stats.pair_games = 10
    for side in ("self", "opponent"):
        for name in NAMES:
            stats.pairs[side][f"Pawn>{name}"] = 30            # pawns die to everything, kill pawns
            stats.pairs[side][f"{name}>Pawn"] = 30
        stats.pairs[side]["Knight>Queen"] = 30                # the knight also eats queens
        stats.pairs[side]["Bishop>Pawn"] += 30                # the bishop eats the same number of pawns
    for side in ("self", "opponent"):
        stats.activity[side]["mobility"]["Queen"] = 5000     # the queen is a high-level piece
    kd = stats.measures()["kd"]
    # Same number of kills and deaths for knight and bishop; the knight's are worth more.
    assert kd[chess.KNIGHT] > kd[chess.BISHOP]


def test_threats_count_only_after_the_first_move():
    fen = "4k3/8/8/4p3/3N4/8/8/3R2K1 b - - 0 1"          # e5 pawn hits the defended Nd4
    moves = ["e8e7", "g1g2", "e5e4"]
    snapshots, _, _, _ = track(moves, fen)
    black = activity(moves, snapshots, fen)[chess.BLACK]
    assert black["threats"][chess.PAWN] == 2               # old rule: sampled before each black move
    assert black["threats_active"][chess.PAWN] == 0        # new rule: the pawn had not moved yet


@pytest.mark.skipif(not fast_rules.AVAILABLE, reason="tonkai_fast is not built")
def test_native_activity_matches_python_with_active_threats():
    board = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Bxc6", "dxc6", "O-O", "Bg4", "h3", "h5", "hxg4", "hxg4"):
        board.push_san(san)
    moves = list(board.move_stack)
    snapshots, _, _, _ = track(moves)
    native = fast_rules.board(chess.Board()).activity([fast_rules.code(m) for m in moves])
    assert native == activity(moves, snapshots)


def test_records_who_killed_whom_and_roundtrips():
    stats = CaptureStats(opponent_weights={"fairy": .35, "external": .65})
    board = chess.Board()
    for san in ("e4", "d5", "exd5", "Qxd5"):
        board.push_san(san)
    stats.add(board.move_stack, chess.WHITE, "unfinished", opponent_kind="fairy")
    assert stats.pairs["self"]["Pawn>Pawn"] == 1 and stats.pairs["opponent"]["Queen>Pawn"] == 1
    assert stats.by_kind["fairy"]["pairs"]["opponent"]["Queen>Pawn"] == 1 and stats.pair_games == 1
    restored = CaptureStats(opponent_weights={"fairy": .35, "external": .65})
    restored.load_state_dict(stats.state_dict())
    assert restored.pairs == stats.pairs and restored.by_kind["fairy"]["pair_games"] == 1


def test_older_statistics_are_seeded_once_from_stored_games():
    stats = CaptureStats()
    assert stats.needs_seed()
    board = chess.Board()
    for san in ("e4", "d5", "exd5", "Qxd5"):
        board.push_san(san)
    stats.kills["self"]["Pawn"] = 7                           # history already counted stays as it is
    stats.seed_new_measures([(board.move_stack, chess.WHITE, "unfinished", None, "fairy")])
    assert not stats.needs_seed() and stats.kills["self"]["Pawn"] == 7 and stats.games == 0
    assert stats.pairs["opponent"]["Queen>Pawn"] == 1


def test_replay_kinds():
    assert replay_kind("Fairy-Stockfish 14; nominal Elo 1000") == "fairy"
    assert replay_kind("Stockfish 11 64 BMI2; nominal Elo 1800") == "external"
    assert replay_kind("TinyOpponent depth 2") == "tiny"
    assert replay_kind("Tonkinho congelado (jogo 65348)") == "self"
    assert replay_kind("counterfactual:abc") is None


def test_weighted_level_kd_ignores_tiny_and_self_play():
    def build(tiny_kills):
        stats = CaptureStats(opponent_weights={"fairy": .35, "external": .65})
        flat(stats)
        stats.pair_games = stats.games = 100
        for kind in ("fairy", "external", "tiny"):
            row = stats._kind(kind)
            row["pair_games"] = row["games"] = 40
            for name in NAMES:
                for key in ("mobility", "threats_active"):
                    row["activity"][key][name] = row["activity"][key + "_n"][name] = 400
        for name in NAMES:
            stats.pairs["self"][f"{name}>Pawn"] = 60
            stats.pairs["opponent"][f"Pawn>{name}"] = 60
            for kind in ("fairy", "external"):
                stats.by_kind[kind]["pairs"]["opponent"][f"{name}>Pawn"] = 40
                stats.by_kind[kind]["pairs"]["self"][f"Pawn>{name}"] = 40
            stats.by_kind["tiny"]["pairs"]["opponent"][f"{name}>Queen"] = tiny_kills
        return stats
    assert build(0).value_source() == "weighted"
    assert build(0).measures()["kd"] == build(10 ** 6).measures()["kd"]
