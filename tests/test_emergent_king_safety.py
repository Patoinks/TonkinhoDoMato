"""v4 Emergent King Safety: exact mate grounding, targets, model and integration.

Nothing here tests *what* the model learns; only that its training signal is
exactly the checkmate consequence it claims to be.
"""
import glob

import chess
import chess.pgn
import numpy as np
import pytest
import torch

from tonkai import fast_rules
from tonkai.encoding import encode
from tonkai.king_safety_model import KingSafetyModel, KingSafetyNet, load_cached
from tonkai.king_safety_training import KSConfig, KingSafetyTrainer, attack_episode, search_targets

pytestmark = pytest.mark.skipif(not fast_rules.AVAILABLE, reason="tonkai_fast is not built")

MATE_IN_1 = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"            # Ra8#
STALEMATE = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"                 # black to move, no moves, no check


def reference_attack(board, n):
    """Python reference: the side to move forces mate within n of its moves."""
    for move in list(board.legal_moves):
        board.push(move)
        try:
            if board.is_checkmate() or (n > 1 and reference_defend(board, n - 1)):
                return True
        finally:
            board.pop()
    return False


def reference_defend(board, n):
    moves = list(board.legal_moves)
    if not moves:
        return False
    for move in moves:
        board.push(move)
        try:
            if not reference_attack(board, n):
                return False
        finally:
            board.pop()
    return True


def near_mates(limit=14):
    """Positions 1-3 plies before a real checkmate, with the winner as attacker."""
    out = []
    for path in sorted(glob.glob("benchmarks/*.pgn")):
        with open(path, encoding="utf-8") as stream:
            while (game := chess.pgn.read_game(stream)) is not None and len(out) < limit:
                if game.headers.get("Termination") != "checkmate":
                    continue
                moves = list(game.mainline_moves())
                winner = not game.end().board().turn            # the side that gave mate
                board = game.board()
                for move in moves[:-3]:
                    board.push(move)
                for move in moves[-3:]:
                    out.append((board.fen(), winner))
                    board.push(move)
    return out


def test_forced_mate_matches_the_python_reference():
    checked = mates = 0
    for fen, attacker in near_mates():
        board = chess.Board(fen)
        expected = 0
        for n in (1, 2):
            found = reference_attack(board, n) if board.turn == attacker else reference_defend(board, n)
            if found:
                expected = n
                break
        found, proven, _ = fast_rules.board(board).forced_mate(attacker, 2, 10 ** 8)
        assert found == expected, fen
        assert proven == (2 if expected == 0 else expected - 1)
        checked += 1
        mates += expected > 0
    assert checked >= 10 and mates >= 3                    # real mates were exercised


def test_encode_here_is_the_standard_encoding():
    board = chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    fast = fast_rules.board(board)
    for colour in (chess.WHITE, chess.BLACK):
        assert np.array_equal(fast.encode_here(colour), encode(board, colour))


def _move(code):
    return chess.Move(code & 63, (code >> 6) & 63, ((code >> 12) & 7) or None)


def test_expand2_lists_every_continuation_with_its_leaf():
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1")
    fast = fast_rules.board(board)
    first, parent, leaf_status, leaf_row, rows, count = fast.expand2(False)
    first, parent, leaf_row = np.asarray(first), np.asarray(parent), np.asarray(leaf_row)
    moves = [_move(code) for code in fast.legal_moves()]                 # expand2's own order
    assert sorted(moves, key=str) == sorted(board.legal_moves, key=str)
    assert first[moves.index(chess.Move.from_uci("f3f7"))] == 1          # Qxf7# mates the defender
    total = 0
    for i, move in enumerate(moves):
        board.push(move)
        if not first[i]:
            child = fast_rules.board(board)
            total += len(child.legal_moves())
            j = int(np.flatnonzero(parent == i)[0])                       # the first reply's leaf
            board.push(_move(child.legal_moves()[0]))
            if leaf_row[j] >= 0:
                assert np.array_equal(rows[leaf_row[j]], encode(board, chess.BLACK))
            board.pop()
        board.pop()
    assert len(parent) == total and count == int((leaf_row >= 0).sum())


def small_model():
    torch.manual_seed(0)
    return KingSafetyModel(KingSafetyNet(16))


def test_targets_follow_the_checkmate_definition():
    cfg = KSConfig(gamma=0.8, mate_moves=2, node_limit=10 ** 7, hidden=16)
    model = small_model()
    rows, targets, founds, provens = search_targets(
        [(MATE_IN_1, False),            # black's king, white to move: mate in 1  -> U = 1
         (MATE_IN_1, True),             # white's king: black has no attack       -> U <= gamma ** 2
         (STALEMATE, False),            # stalemate: the king survives            -> U = 0
         ], model, cfg)
    assert targets[0] == pytest.approx(1.0) and founds[0] == 1
    assert targets[1] <= 0.8 ** 2 + 1e-6 and provens[1] == 2
    assert targets[2] == 0.0
    assert rows.shape == (3, 782)
    mated = chess.Board(MATE_IN_1)
    mated.push_san("Ra8#")
    _, targets, founds, _ = search_targets([(mated.fen(), False)], model, cfg)
    assert targets[0] == 1.0 and founds[0] == -1           # already mated


def test_mate_in_two_is_gamma():
    cfg = KSConfig(gamma=0.8, mate_moves=2, node_limit=10 ** 7, hidden=16)
    model = small_model()
    for fen, attacker in near_mates(40):
        board = chess.Board(fen)
        if board.turn != attacker:
            continue
        found, _, _ = fast_rules.board(board).forced_mate(attacker, 2, 10 ** 8)
        if found == 2:
            _, targets, founds, _ = search_targets([(fen, not attacker)], model, cfg)
            assert founds[0] == 2 and targets[0] == pytest.approx(0.8)
            return
    pytest.skip("no mate in two among the stored games")


def test_model_save_load_and_both_kings(tmp_path):
    model = small_model()
    board = chess.Board()
    white, black = model.ks_both(board)
    assert 0 < white < 1 and 0 < black < 1
    assert model.ks(board, chess.WHITE) == pytest.approx(white)
    assert model.advantage(board, chess.WHITE) == pytest.approx(white - black)
    model.save(tmp_path / "ks.pt", note="test")
    loaded = KingSafetyModel.load(tmp_path / "ks.pt")
    assert loaded.ks_both(board) == pytest.approx((white, black))
    assert loaded.meta["note"] == "test"
    assert load_cached(tmp_path / "ks.pt") is load_cached(tmp_path / "ks.pt")


def test_one_training_iteration_runs():
    cfg = KSConfig(gamma=0.8, mate_moves=1, node_limit=10 ** 6, hidden=16, roots_per_iteration=4,
                   steps_per_iteration=3, episodes_per_iteration=2, episode_plies=4, batch=8, chunk=2)
    trainer = KingSafetyTrainer(cfg, [MATE_IN_1, chess.STARTING_FEN], workers=1)
    metrics = trainer.iterate()
    assert metrics["samples"] == 8 and metrics["iteration"] == 1
    assert np.isfinite(metrics["loss"]) and metrics["explored_pool"] > 0
    trainer.close()


def test_attack_episode_takes_the_mate_the_rules_reveal():
    cfg = KSConfig(explore=0.0, episode_plies=2, hidden=16)
    visited, mated = attack_episode(MATE_IN_1, False, small_model(), cfg, np.random.default_rng(0))
    assert len(visited) == 2 and mated


def test_agent_uses_the_learned_king_safety_and_logs_it(tmp_path, monkeypatch):
    from tonkai import cognitive_agent
    from tonkai.cognitive_config import CognitiveConfig
    from tonkai.cognitive_trainer import CognitiveKnowledge
    from tonkai.survival_agent import play_episode
    from tonkai.temporal_model import SurvivalNet
    small_model().save(tmp_path / "ks.pt")
    cfg = CognitiveConfig(device="cpu", inference_device="cpu", hidden_size=24, latent_size=16, memory_size=12,
                          head_hidden=8, piece_hidden=8, ensemble_size=3, max_plies=12, planning_nodes=6,
                          objective="shaped", king_safety_model=str(tmp_path / "ks.pt"))
    forbidden = lambda *a, **k: pytest.fail("the hand-written king safety must not be used")  # noqa: E731
    monkeypatch.setattr(cognitive_agent, "king_senses", forbidden)
    monkeypatch.setattr(fast_rules, "king_balance", forbidden)
    torch.manual_seed(1)
    net = SurvivalNet(cfg).eval()
    agent = cognitive_agent.CognitiveAgent(net, cfg, CognitiveKnowledge(cfg, net.encoder))

    class Random:
        description = "random"

        def reset(self):
            pass

        def choose(self, board, rng):
            moves = list(board.legal_moves)
            return moves[int(rng.integers(len(moves)))]

    episode, board, colour, stats = play_episode(agent, Random(), cfg, np.random.default_rng(3))
    assert len(stats["king_safety"]) == stats["plies"] + 1
    assert all(0 < w < 1 and 0 < b < 1 for w, b in stats["king_safety"])
    assert "ks" in agent.trace[-1]["king_safety"]["w"]


def test_threats_match_a_null_move_reference():
    from tonkai.king_safety_training import fens_from_pgn
    fens = fens_from_pgn(sorted(glob.glob("benchmarks/*.pgn")))
    rng = np.random.default_rng(5)
    checked = threats_seen = 0
    for fen in [fens[int(i)] for i in rng.integers(len(fens), size=25)]:
        board = chess.Board(fen)
        codes, threat, check = fast_rules.board(board).threats(1, 10 ** 7)
        for code, depth, checking in zip(codes, threat, check):
            move = next(m for m in board.legal_moves if fast_rules.code(m) == code)
            assert checking == board.gives_check(move)
            if checking:
                continue
            board.push(move)
            board.push(chess.Move.null())                          # the defender passes
            expected = reference_attack(board, 1)
            board.pop()
            board.pop()
            assert (depth == 1) == expected, (fen, move.uci())
            checked += 1
            threats_seen += expected
    assert checked > 200


def test_attacker_explores_a_move_that_creates_a_mate_threat():
    from tonkai.king_safety_training import mating_moves, threat_moves
    fen = "r6k/6pp/8/8/8/8/1B6/3Q2K1 w - - 0 1"      # no mate now (Ra8 guards rank 8); Qd4 threatens Qxg7#
    cfg = KSConfig(explore=0.0, threat_explore=1.0, threat_moves=1, mate_moves=1, episode_plies=1, hidden=16)
    board = chess.Board(fen)
    assert mating_moves(fast_rules.board(board), True, cfg) == []
    allowed = threat_moves(fast_rules.board(board), cfg)
    assert fast_rules.code(chess.Move.from_uci("d1d4")) in allowed
    counts = {}
    visited, _ = attack_episode(fen, False, small_model(), cfg, np.random.default_rng(2), counts)
    played = next(m for m in board.legal_moves if (board.push(m), board.fen(), board.pop())[1] == visited[1])
    assert fast_rules.code(played) in allowed and counts.get("threat") == 1


def test_series_rebuilt_afterwards_equals_the_per_ply_log():
    import chess
    from tonkai.king_safety_model import KingSafetyModel
    model = KingSafetyModel()
    board = chess.Board()
    logged = [[round(v, 4) for v in model.ks_both(board)]]
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4"):
        board.push_san(san)
        logged.append([round(v, 4) for v in model.ks_both(board)])
    rebuilt = model.series(board.move_stack)
    assert len(rebuilt) == len(logged)
    assert all(abs(a - b) <= 1e-4 for row, other in zip(rebuilt, logged) for a, b in zip(row, other))
