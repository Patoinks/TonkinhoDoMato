"""Meaningful equivalence and rule-proof regressions for G16 v2."""
import os
from pathlib import Path
import sys

import chess
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths
from claudio_agent import Claudio, ClaudioConfig, native_board, _key
from claudio_load import load_claudio
from tonkai import fast_rules
from v2.agent import V2Claudio, V2Options, NativeBoard, native
from v2.inference import FrozenInference, cached_surprises
from v2.rules import adjudicate, dead_pawn_lock, draw_config


@pytest.fixture(scope='module')
def source():
    torch.set_num_threads(1)
    target = os.environ.get('G16_V2_SOURCE', '')
    if not target:
        target = str(Path(__file__).resolve().parent / 'models/source-600.pt')
    if not Path(target).exists():
        pytest.skip('Set G16_V2_SOURCE to the real checkpoint 600')
    return load_claudio(target, with_config=True, with_king=True)


def player(source, **options):
    net, values, config, _, king = source
    return V2Claudio(net, values, ClaudioConfig.with_settings([], base=config),
                     rng=np.random.default_rng(2027), king_model=king, options=V2Options(**options))


def test_packed_heads_are_bitwise_the_same_on_real_network(source):
    net = source[0]
    frozen = FrozenInference(net)
    with torch.inference_mode():
        contexts = torch.randn(43, net.latent_size)
        actual, expected = frozen.survival(contexts), net.survival(contexts)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


def test_repetition_batch_keeps_complete_history_and_restores_board():
    board = chess.Board()
    for uci in ('g1f3','g8f6','f3g1','f6g8','g1f3','g8f6','f3g1'):
        board.push_uci(uci)
    fast = NativeBoard(board)
    codes = list(fast.legal_moves())
    earlier = []
    reference = native_board(board.root())
    earlier.append(_key(reference))
    for move in board.move_stack:
        reference.push(fast_rules.code(move))
        earlier.append(_key(reference))
    endings, repeated = fast.terminal_repetition_after_moves(codes, earlier)
    expected = []
    for code in codes:
        reference.push(code)
        expected.append((reference.terminal_strict(), _key(reference) in earlier))
        reference.pop()
    assert list(zip(endings, repeated)) == expected
    assert fast.fen() == board.fen(en_passant='fen')


@pytest.mark.parametrize('fen', [
    'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
    '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1',
    '4k3/P7/8/8/8/8/8/4K3 w - - 0 1',
    '6k1/8/8/8/8/8/5Q2/6K1 w - - 99 80'])
def test_batch_special_positions(fen):
    board = chess.Board(fen)
    a, b = native_board(board), NativeBoard(board)
    codes = list(a.legal_moves())
    ends, reps = b.terminal_repetition_after_moves(codes, [])
    assert list(ends) == list(a.terminal_after_moves(codes))
    assert not any(reps)
    assert a.fen() == b.fen() == board.fen(en_passant='fen')


def test_pawn_mate_is_not_missed_by_distance_filter():
    fen = '7k/1R6/7B/P7/8/8/4K3/8 w - - 53 143'
    result = native.mating_plan(fen, [], 8, 2_000_000)
    assert result['moves'] == 3
    move = chess.Move(result['move'] & 63, (result['move'] >> 6) & 63, (result['move'] >> 12) or None)
    board = chess.Board(fen)
    assert move in board.legal_moves
    assert move.uci() == 'a5a6'
    # Independently verify all opponent replies using python-chess, not the native proof table.
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def proves(fen, left):
        position = chess.Board(fen)
        if position.is_checkmate():
            return position.turn == chess.BLACK
        if position.is_game_over(claim_draw=True) or left == 0:
            return False
        outcomes = []
        for reply in position.legal_moves:
            after = position.copy(stack=False)
            after.push(reply)
            outcomes.append(proves(after.fen(), left-1 if position.turn == chess.WHITE else left))
        return any(outcomes) if position.turn == chess.WHITE else all(outcomes)
    board.push(move)
    assert proves(board.fen(), 2)


def test_mate_planner_honours_fifty_move_draw_and_node_budget():
    # No pawn/capture can reset this counter. A non-immediate mate cannot beat the draw.
    fen = 'k7/8/8/8/8/2K5/7Q/8 w - - 99 80'
    result = native.mating_plan(fen, [], 8, 1_000_000)
    assert result['moves'] == 0
    aborted = native.mating_plan(chess.STARTING_FEN, [], 8, 0)
    assert aborted['moves'] == 0 and aborted['aborted']


def test_canonical_book_lookup_after_pawn_double_move(source):
    me = player(source, finish_planner=False)
    board = chess.Board()
    board.push_uci('e2e4')
    me._catch_up(board)
    me.book_board = board
    assert _key(me.fast) != board.fen().rsplit(' ',2)[0]
    me.book_moves = {board.fen().rsplit(' ',2)[0]: {'e7e5':(20,18), 'a7a6':(20,2)}}
    codes = list(me.fast.legal_moves())
    moves = [chess.Move(c&63,(c>>6)&63,(c>>12)or None) for c in codes]
    scores = np.zeros(len(codes))
    me.cfg.book_weight, me.cfg.book_min = 0.1, 4
    me._book(me.fast,codes,moves,scores,np.zeros(len(codes),dtype=bool))
    assert scores[moves.index(chess.Move.from_uci('e7e5'))] > scores[moves.index(chess.Move.from_uci('a7a6'))]


def test_cached_trajectory_matches_same_perspective_labels(source):
    me = player(source, finish_planner=False, learned_policy=False, opening_book=False)
    board = chess.Board()
    sequence = 'e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7'.split()
    for uci in sequence:
        if board.turn == chess.WHITE:
            me._catch_up(board)
        board.push_uci(uci)
    actual_rows, actual = cached_surprises(me,board)
    fast = fast_rules.board(board.root())
    rows = [np.asarray(fast.encode_here(True),dtype=np.float32)]
    candidates, groups = [], []
    for ply, move in enumerate(board.move_stack):
        if fast.fen().split(' ',2)[1]=='b':
            codes=list(fast.legal_moves())
            candidates.append(fast_rules.encode_after_moves(fast,codes,True))
            groups.append((ply,len(codes),codes.index(fast_rules.code(move))))
        fast.push(fast_rules.code(move))
        rows.append(np.asarray(fast.encode_here(True),dtype=np.float32))
    np.testing.assert_array_equal(actual_rows,np.stack(rows))
    net = source[0]
    with torch.inference_mode():
        latent=net.encode(torch.from_numpy(np.stack(rows)))
        memory, memories=net.initial_memory(1),[]
        for i in range(len(latent)):
            memory=net.memory_cell(latent[i:i+1],memory)
            memories.append(memory)
        contexts,_=net.candidates(torch.from_numpy(np.concatenate(candidates)),torch.cat([memories[p].expand(n,-1) for p,n,_ in groups]))
        logits=net.predict_logits(contexts).split([n for _,n,_ in groups])
        expected=np.zeros(len(rows),dtype=np.float32)
        for (ply,_,played),part in zip(groups,logits):
            expected[ply+1]=-float(np.log(max(float(part.softmax(0)[played]),1e-12)))
    np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=1e-5)
    assert me.colour == chess.WHITE


def test_exact_variant_keeps_real_checkpoint_decisions(source):
    net,values,config,_,king=source
    base=Claudio(net,values,ClaudioConfig.with_settings([],base=draw_config(config)),rng=np.random.default_rng(2027),king_model=king)
    v2=player(source,finish_planner=False,learned_policy=False,opening_book=False,opening_fast=False)
    board=chess.Board()
    for uci in 'e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 d2d3 b7b5 a4b3 d7d6'.split():
        if board.turn==chess.WHITE:
            assert base.choose(board)==v2.choose(board)
        board.push_uci(uci)


@pytest.mark.parametrize('fen,expected', [
    ('8/8/8/8/8/8/4k3/K7 w - - 0 1', 'INSUFFICIENT_MATERIAL'),
    ('8/8/8/8/8/8/4k3/KN6 w - - 0 1', 'INSUFFICIENT_MATERIAL'),
    ('8/8/8/8/8/8/4k3/KB6 w - - 0 1', 'INSUFFICIENT_MATERIAL'),
    ('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1', 'STALEMATE'),
    ('7k/6Q1/6K1/8/8/8/8/8 b - - 150 1', 'CHECKMATE'),
    ('k7/8/8/8/8/2K5/7Q/8 w - - 100 80', 'FIFTY_MOVES'),
    ('k7/8/8/8/8/2K5/7Q/8 w - - 150 80', 'SEVENTYFIVE_MOVES')])
def test_real_end_rules_and_mate_priority(fen,expected):
    result=adjudicate(chess.Board(fen))
    assert result['end']==expected
    if expected!='CHECKMATE':
        assert result['winner'] is None and result['draw_points']==0.5


def test_claimed_and_automatic_repetition():
    board=chess.Board()
    cycle=('g1f3','g8f6','f3g1','f6g8')
    for uci in cycle*2: board.push_uci(uci)
    assert adjudicate(board)['end']=='THREEFOLD_REPETITION'
    for uci in cycle*2: board.push_uci(uci)
    assert adjudicate(board,claim_draw=False)['end']=='FIVEFOLD_REPETITION'


def test_dead_locked_pawns_certificate_and_negative_controls():
    board=chess.Board('2k5/8/8/p1p1p1p1/P1P1P1P1/8/8/2K5 w - - 0 1')
    assert board.is_valid() and not board.is_insufficient_material() and board.outcome() is None
    assert dead_pawn_lock(board) and native.dead_pawn_lock(board.fen())
    assert adjudicate(board)['end']=='DEAD_PAWN_LOCK'
    opened=board.copy()
    opened.remove_piece_at(chess.C5)
    assert not dead_pawn_lock(opened) and not native.dead_pawn_lock(opened.fen())
    with_rook=board.copy()
    with_rook.set_piece_at(chess.H1,chess.Piece(chess.ROOK,chess.WHITE))
    assert not dead_pawn_lock(with_rook) and not native.dead_pawn_lock(with_rook.fen())


def test_no_artificial_draw_on_two_knights_or_time_cutoff():
    board=chess.Board('7k/8/8/8/8/8/8/KNN5 w - - 0 1')
    assert adjudicate(board) is None
    assert adjudicate(chess.Board()) is None


def test_v2_does_not_play_after_an_actual_draw(source):
    me=player(source)
    assert me.cfg.draw_value==me.cfg.draw_ahead==0.5
    with pytest.raises(ValueError,match='Game already ended'):
        me.choose(chess.Board('8/8/8/8/8/8/4k3/K7 w - - 0 1'))
