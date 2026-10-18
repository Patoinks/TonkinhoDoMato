"""Legalidade, determinismo, linhas utilizáveis e preferências posicionais de abertura."""
import importlib.util
from pathlib import Path
import sys

import chess
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: F401
import abertura


def board_of(line):
    board = chess.Board()
    for san in line.split():
        board.push_san(san)
    return board


def test_every_line_legal_and_book_deterministic():
    for version, lines in ((1, abertura.LINES_V1), (2, abertura.LINES)):
        expected = {}
        for text in lines.strip().splitlines():
            board = chess.Board()
            for san in text.split():
                move = board.parse_san(san)
                assert move in board.legal_moves
                expected.setdefault(abertura._key(board, version), move)
                board.push(move)
        assert abertura._book(version) == expected
        assert all(isinstance(move, chess.Move) for move in expected.values())


def test_every_extension_can_be_followed_by_at_least_one_colour():
    # Alternatives must be the opponent's choices, not a second move for G4.
    for text in abertura.LINES_EXTENSIONS.strip().splitlines():
        follows = {chess.WHITE: True, chess.BLACK: True}
        board = chess.Board()
        for san in text.split():
            move = board.parse_san(san)
            if abertura.repertoire_move(board, 2) != move:
                follows[board.turn] = False
            board.push(move)
        assert any(follows.values()), text


def test_frozen_v1_exactly_matches_read_only_original():
    source = Path(__file__).resolve().parent / "abertura_v1_original.py"
    spec = importlib.util.spec_from_file_location("abertura_original", source)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    assert abertura.LINES_V1 == old.LINES
    assert abertura._book(1) == old._book()
    positions = [chess.Board()]
    for text in old.LINES.strip().splitlines()[::7]:
        board = chess.Board()
        for san in text.split():
            positions.append(board.copy())
            board.push_san(san)
    for board in positions:
        assert abertura.repertoire_move(board, 1) == old.repertoire_move(board)
        for move in board.legal_moves:
            assert abertura.principles(board, move, board.ply(), 1) == old.principles(board, move, board.ply())


def test_v2_keeps_existing_choices_and_extends_all_principal_families():
    for text in abertura.LINES_V1.strip().splitlines():
        board = chess.Board()
        for san in text.split():
            assert abertura.repertoire_move(board, 2) == abertura.repertoire_move(board, 1)
            board.push_san(san)
    assert len(abertura._book(2)) > len(abertura._book(1))
    families = ("e4 e5 Nf3 Nc6 Bb5", "e4 e5 Nf3 Nc6 Bc4", "e4 c5", "e4 e6", "e4 c6",
                "d4 Nf6 c4 e6 Nf3 d5", "d4 Nf6 c4 e6 Nc3 Bb4", "c4 e5")
    for prefix in families:
        assert any(text.startswith(prefix) and len(text.split()) >= 24
                   for text in abertura.LINES_EXTENSIONS.strip().splitlines())


@pytest.mark.parametrize("line", ("e4 a5", "e4 Nf6 e5 Ng8"))
def test_bad_defences_met_by_central_space(line):
    board = board_of(line)
    assert board.san(abertura.repertoire_move(board, 2)) == "d4"
    after = board_of(line + " d4")
    assert abertura.repertoire_move(after, 2) is not None


def test_v2_key_distinguishes_real_en_passant_rights():
    ep = board_of("e4 a6 e5 d5")
    without = chess.Board(ep.fen())
    without.ep_square = None
    assert ep.has_legal_en_passant()
    assert abertura._key(ep, 2) != abertura._key(without, 2)
    assert abertura._key(ep, 1) == abertura._key(without, 1)


def test_central_knight_development_beats_rim():
    board = chess.Board()
    assert abertura.principles(board, board.parse_san("Nf3"), 0) > abertura.principles(board, board.parse_san("Nh3"), 0)


@pytest.mark.parametrize("mirror", (False, True))
def test_castled_pawn_shield_and_luft(mirror):
    board = chess.Board("rnbqk2r/ppppbppp/4pn2/8/3P4/2N1P3/PPP2PPP/R1BQ1RK1 w kq - 0 7")
    if mirror:
        board = board.mirror()
    for san in (("f4", "g4") if not mirror else ("f5", "g5")):
        move = board.parse_san(san)
        assert abertura.principles(board, move, 12, 2) < abertura.principles(board, move, 12, 1) - 0.5
    short, long = ("h3", "h4") if not mirror else ("h6", "h5")
    assert abertura.principles(board, board.parse_san(short), 12) > abertura.principles(board, board.parse_san(long), 12)


def test_unnecessary_doubled_and_isolated_capture_is_softly_penalised():
    board = chess.Board("4k3/8/8/8/3p4/4P3/2PP1PPP/4K3 w - - 0 8")
    move = board.parse_san("exd4")
    assert abertura.principles(board, move, 14, 1) == 0
    assert abertura.principles(board, move, 14, 2) < 0


@pytest.mark.parametrize("mirror", (False, True))
def test_dont_trade_protective_fianchetto_bishop(mirror):
    board = chess.Board("6k1/8/5n2/3b4/8/6P1/5PBP/6K1 w - - 0 8")
    if mirror:
        board = board.mirror()
    move = board.parse_san("Bxd5" if not mirror else "Bxd4")
    assert abertura.principles(board, move, 14, 2) < abertura.principles(board, move, 14, 1) - 0.5


def test_knight_does_not_block_own_central_pawn():
    board = chess.Board("4k3/8/8/8/8/3P1N2/2P1PP2/2B1K3 w - - 0 6")
    move = board.parse_san("Nd4")
    assert abertura.principles(board, move, 10, 2) < abertura.principles(board, move, 10, 1) - 0.6


def test_queen_development_connects_rooks():
    board = chess.Board("6k1/8/8/8/8/4P3/PPP2PPP/R2Q1RK1 w - - 0 10")
    move = board.parse_san("Qd2")
    assert abertura.principles(board, move, 18, 2) >= abertura.principles(board, move, 18, 1) + 0.5


@pytest.mark.parametrize("line", ("", "e4 a6 e5 d5", "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O"))
def test_batch_preserves_board_and_matches_single_move_and_colour_mirror(line):
    board = board_of(line)
    fen, stack = board.fen(), list(board.move_stack)
    moves = list(board.legal_moves)
    batch = abertura.principle_scores(board, moves, board.ply(), 2)
    assert batch == [abertura.principles(board, move, board.ply(), 2) for move in moves]
    assert board.fen() == fen and board.move_stack == stack
    mirrored = board.mirror()
    for move, score in zip(moves, batch):
        inverse = chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), move.promotion)
        assert score == pytest.approx(abertura.principles(mirrored, inverse, board.ply(), 2))


def test_weight_fades_and_new_options_load_with_frozen_defaults():
    from claudio_agent import ClaudioConfig
    cfg = ClaudioConfig()
    assert cfg.opening_book_version == cfg.opening_principles_version == 1
    changed = ClaudioConfig.with_settings(["opening_book_version=2", "opening_principles_version=2",
                                          "opening_weight=0.05", "opening_plies=30"])
    assert changed.opening_book_version == changed.opening_principles_version == 2
    assert changed.opening_weight == 0.05 and changed.opening_plies == 30
    for plies in (18, 24, 30, 36):
        weights = [abertura.weight_at(ply, plies) for ply in range(plies + 2)]
        assert weights[0] == 1 and weights[-1] == 0
        assert all(a >= b >= 0 for a, b in zip(weights, weights[1:]))


@pytest.mark.parametrize("ply", (22, 23))
def test_g4t_really_plays_the_deeper_book_with_both_colours(ply):
    # SF's short-time games usually leave theory early. Exercise the actual
    # player beyond V1, rather than claiming that legality alone proves integration.
    from claudio_load import load_players, make_player
    checkpoint = Path(__file__).resolve().parents[1] / "releases/g4-leto/models/g4t2-leto-v3.pt"
    player = make_player(load_players(checkpoint))
    player.cfg.opening_book_version = 2
    player.cfg.opening_principles_version = 2
    player.cfg.opening_plies = 18
    line = abertura.LINES_EXTENSIONS.strip().splitlines()[0].split()
    board = board_of(" ".join(line[:ply]))
    theory = abertura.repertoire_move(board, 2)
    assert abertura.repertoire_move(board, 1) is None
    assert theory is not None and theory in board.legal_moves
    assert player.choose(board) == theory


def load_snapshot(name, source):
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_v2_scores_books_and_fade_exactly_match_snapshot():
    source = Path(__file__).resolve().parent / "testes-baseline/abertura.py"
    old = load_snapshot("abertura_v2_frozen", source)
    assert abertura.LINES == old.LINES
    assert abertura._book(2) == old._book(2)
    for text in old.LINES.splitlines()[::13]:
        board = chess.Board()
        for san in text.split():
            moves = list(board.legal_moves)
            assert abertura.principle_scores(board, moves, board.ply(), 2) == old.principle_scores(board, moves, board.ply(), 2)
            board.push_san(san)
    for plies in (18, 24, 30, 36):
        for ply in range(plies + 2):
            assert abertura.weight_at(ply, plies) == old.weight_at(ply, plies)


@pytest.mark.parametrize("line", ("", "e4 a6 e5 d5", "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O",
                                      "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 Qc7"))
def test_v3_batch_legal_mirrored_immutable_and_exact_ablation(line):
    board = board_of(line)
    before = (board.fen(), list(board.move_stack))
    moves = list(board.legal_moves)
    batch = abertura.principle_scores(board, moves, board.ply(), 3)
    assert (board.fen(), list(board.move_stack)) == before
    assert batch == [abertura.principles(board, m, board.ply(), 3) for m in moves]
    assert abertura.principle_scores(board, moves, board.ply(), 3, 0) == abertura.principle_scores(board, moves, board.ply(), 2)
    inverse = board.mirror()
    for move, score in zip(moves, batch):
        mirrored = chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), move.promotion)
        assert score == pytest.approx(abertura.principles(inverse, mirrored, board.ply(), 3))


@pytest.mark.parametrize("mirror", (False, True))
def test_v3_castling_readiness_and_late_urgency(mirror):
    board = board_of("e4 e5 Nf3 Nc6 Bc4 Nf6 d3 d6 Nc3 Be7 Be3 a6")
    if mirror:
        board = board.mirror()
    castle = board.parse_san("O-O")
    assert abertura.principles(board, castle, 12, 3) >= abertura.principles(board, castle, 12, 2) + 0.99
    assert abertura.principles(board, castle, 12, 3) > abertura.principles(board, castle, 10, 3)


def test_v3_clearing_castling_route_and_developing_by_capture():
    board = board_of("e4 a6 Nf3 a5")
    move = board.parse_san("Bc4")
    assert abertura.principles(board, move, 4, 3) > abertura.principles(board, move, 4, 2) + 0.6
    capture_board = chess.Board("4k3/8/8/8/8/7r/PPPPPPPP/RNBQKBNR w KQ - 0 5")
    capture = capture_board.parse_san("Nxh3")
    assert abertura.principles(capture_board, capture, 8, 3) >= abertura.principles(capture_board, capture, 8, 2) + 0.5


def test_v3_does_not_treat_retired_bishop_on_b1_as_untouched():
    board = chess.Board("6k1/8/8/8/8/2N1PN2/PPP2PPP/1B1Q1RK1 w - - 0 12")
    assert not abertura._undeveloped_minors(board, chess.WHITE)
    move = board.parse_san("Ne5")
    assert abertura.principles(board, move, 22, 3) > abertura.principles(board, move, 22, 2)


def test_v3_discourages_rook_walk_and_exposed_f_pawn_but_allows_fianchetto():
    rook = board_of("a3 Nf6")
    move = rook.parse_san("Ra2")
    assert abertura.principles(rook, move, 2, 3) < abertura.principles(rook, move, 2, 2) - 0.4
    board = chess.Board()
    for san in ("f4", "g4"):
        move = board.parse_san(san)
        assert abertura.principles(board, move, 0, 3) < abertura.principles(board, move, 0, 2) - 0.3
    fianchetto = board.parse_san("g3")
    assert abertura.principles(board, fianchetto, 0, 3) >= abertura.principles(board, fianchetto, 0, 2)


def test_v3_penalties_do_not_apply_to_necessary_retreat_or_check():
    attacked = board_of("e4 d6 d4 Nd7 Nc3 Ngf6 Bf4 e5 dxe5 dxe5")
    move = attacked.parse_san("Bg3")
    assert attacked.is_attacked_by(chess.BLACK, move.from_square)
    assert abertura.principles(attacked, move, 10, 3) >= abertura.principles(attacked, move, 10, 2) - 0.32
    check = board_of("e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5 Ndb5 Qa5 Bc4 a6")
    move = check.parse_san("Nd6+")
    assert check.gives_check(move)
    assert abertura.principles(check, move, 14, 3) >= abertura.principles(check, move, 14, 2) - 0.32


@pytest.mark.parametrize("fade", ("legacy", "linear", "hold"))
def test_v3_fade_schedules_and_settings(fade):
    from claudio_agent import ClaudioConfig
    cfg = ClaudioConfig.with_settings(["opening_principles_version=3", "opening_v3_scale=0.5",
                                       f"opening_fade={fade}", "opening_plies=30"])
    assert cfg.opening_principles_version == 3 and cfg.opening_v3_scale == 0.5
    assert cfg.opening_fade == fade and cfg.opening_plies == 30
    weights = [abertura.weight_at(ply, 30, fade) for ply in range(32)]
    assert weights[0] == 1 and weights[30] == weights[31] == 0
    assert all(a >= b >= 0 for a, b in zip(weights, weights[1:]))
    assert abertura.weight_at(24, 30, "linear") == 0.2
    assert abertura.weight_at(24, 30, "hold") == 1
    assert abertura.weight_at(24, 30, "legacy") == 0.6


@pytest.fixture(scope="module")
def leto_loaded():
    import torch
    from claudio_load import load_players
    torch.set_num_threads(1)
    return load_players(Path(__file__).resolve().parents[1] / "releases/g4-leto/models/g4t2-leto-v3.pt")


def test_v3_actual_choose_uses_version_scale_and_duration_after_old_cutoff(leto_loaded, monkeypatch):
    from dataclasses import replace
    from claudio_load import make_player
    player = make_player(leto_loaded)
    player.cfg = replace(player.cfg, opening_principles_version=3, opening_plies=30,
                         opening_v3_scale=0.5, opening_fade="linear")
    board = chess.Board("r2qr1k1/ppp2ppp/5n2/b1n1B3/4P3/P1NB3P/1PP2P1P/R2QK2R w KQ - 1 12")
    assert abertura.repertoire_move(board, 2) is None and board.ply() == 22
    calls = []
    real = abertura.principle_scores
    def record(board, moves, ply, version=2, scale=1.0):
        calls.append((ply, version, scale))
        return real(board, moves, ply, version, scale)
    monkeypatch.setattr(abertura, "principle_scores", record)
    before = board.fen()
    assert player.choose(board) in board.legal_moves
    assert calls == [(22, 3, 0.5)] and board.fen() == before


def test_frozen_v2_actual_decisions_match_snapshot(leto_loaded):
    from dataclasses import replace
    import numpy as np
    from claudio_load import make_player
    frozen = load_snapshot("claudio_agent_v2_frozen", Path(__file__).resolve().parent / "testes-baseline/claudio_agent.py")
    net, values, cfg, king = leto_loaded[0]
    old_cfg = frozen.ClaudioConfig(**{k: v for k, v in vars(cfg).items()
                                     if k in frozen.ClaudioConfig.__dataclass_fields__})
    lines = ("e4 d6 d4 Nd7 Nc3 Ngf6 Bf4 e5 dxe5 dxe5",
             "d4 Nf6 Nd2 d5 e3 Nc6 Ngf3 Bf5 Bb5 a6 Bxc6+ bxc6 Ne5",
             "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5 Ndb5 Qa5 Bc4 a6",
             "a3 Nf6")
    for line in lines:
        board = board_of(line)
        original = frozen.Claudio(net, values, old_cfg, rng=np.random.default_rng(0), king_model=king)
        current = make_player(leto_loaded, np.random.default_rng(0))
        current.cfg = replace(current.cfg)
        before = (board.fen(), list(board.move_stack))
        assert current.choose(board) == original.choose(board)
        assert (board.fen(), list(board.move_stack)) == before
