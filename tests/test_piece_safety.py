import chess
import pytest

from tonkai.piece_safety import (exchange_gain, favorable_attack, pawn_unit, piece_senses,
                                 position_pressure, position_risk, protected_mobility_gain, square_support)


def values(piece, square):
    # Controlled values test the exchange mechanics; the actor uses learned values.
    return {1: 1., 2: 3., 3: 3., 4: 5., 5: 9., 6: 10000.}[piece.piece_type]


def sense(fen, square, value=values):
    board = chess.Board(fen)
    before = board.fen()
    rows = piece_senses(board, value)
    assert board.fen() == before
    assert len(rows) == len(board.piece_map())
    return next(row for row in rows if row['square'] == square)


def test_losing_exchange_has_priority_over_nominal_protection():
    row = sense('4k3/8/8/8/1p6/2N5/8/4K3 b - - 0 1', 'c3')
    assert row['exchange_balance'] == -3
    assert row['priority'] == 1 and row['risk'] > .5


def test_equal_exchange_with_legal_defender_is_supported():
    row = sense('4k3/8/5n2/8/4N3/3P4/8/4K3 b - - 0 1', 'e4')
    assert row['exchange_balance'] == 0
    assert row['defender_squares'] == ['d3']
    assert row['attackers'] == row['defenders'] == 1
    assert row['priority'] == 3 and row['risk'] == 0


def test_pinned_defender_cannot_recapture():
    row = sense('k3r3/8/8/8/5n2/3N4/4B3/4K3 b - - 0 1', 'd3')
    assert row['defenders'] == 0
    assert row['exchange_balance'] == -3


def test_multiple_attackers_require_multiple_legal_defenders_only_for_close_exchange():
    def positional(piece, square):
        return .25 if piece.piece_type == chess.PAWN else values(piece, square)
    one = sense('4k3/8/3n1n2/8/4N3/3P4/8/4K3 b - - 0 1', 'e4', positional)
    two = sense('4k3/8/3n1n2/8/4N3/3P1P2/8/4K3 b - - 0 1', 'e4', positional)
    assert one['priority'] == 3 and one['deficit'] == 1 and one['attackers'] == 2
    assert two['defenders'] == 2 and two['deficit'] == 0
    assert one['risk'] > two['risk']
    normal = sense('4k3/8/3n1n2/8/4N3/3P4/8/4K3 b - - 0 1', 'e4')
    assert normal['priority'] == 1  # exchange loses a full pawn, outside 0.5


def test_profitable_exchange_is_not_penalized_for_defender_count():
    row = sense('k3r3/8/8/8/4P3/3P4/8/7K b - - 0 1', 'e4')
    assert row['exchange_balance'] == 4
    assert row['priority'] == 1 and row['risk'] == 0


def test_en_passant_is_attached_to_captured_pawn_not_destination():
    row = sense('k7/8/8/8/3Pp3/8/8/7K b - d3 0 1', 'd4')
    assert row['attacked']
    assert row['branches'][0]['move'] == 'e4d3'


def test_exchange_discovers_xray_and_accounts_for_promotion():
    board = chess.Board('k3r3/8/8/8/3rP3/8/8/4R2K b - - 0 1')
    assert exchange_gain(board, chess.Move.from_uci('d4e4'), values) == 1
    promotion = chess.Board('7k/8/8/8/8/8/1p6/R6K b - - 0 1')
    assert exchange_gain(promotion, chess.Move.from_uci('b2a1q'), values) == 13


def test_pawn_scale_uses_median_and_has_finite_fallback():
    pawns = [chess.Piece(chess.PAWN, c) for c in (True, False, True)]
    assert pawn_unit([.002, -.004, .08], pawns) == .004
    assert pawn_unit([], []) == .001
    assert position_risk([], True) == 0


def test_support_sums_learned_values_and_favorable_low_value_threat():
    fen = '7k/8/5n2/3r4/8/8/4P3/4R2K w - - 0 1'
    before = chess.Board(fen)
    after = before.copy(stack=False)
    after.push_uci('e2e4')
    def learned(piece, square):
        return {chess.PAWN: 1.2, chess.KNIGHT: 2.4, chess.ROOK: 4.8,
                chess.KING: 10000.}[piece.piece_type]
    assert square_support(after, chess.E4, chess.WHITE, learned) == pytest.approx((2.4, 4.8))
    assert favorable_attack(after, chess.E4, chess.WHITE, learned) == pytest.approx(.75)
    assert protected_mobility_gain(before, after, chess.E4, chess.WHITE, learned) > 0
    assert before.fen() == fen


def test_unsupported_attack_and_mobility_get_no_bonus():
    before = chess.Board('7k/8/5n2/3r4/8/8/4P3/7K w - - 0 1')
    after = before.copy(stack=False)
    after.push_uci('e2e4')
    assert square_support(after, chess.E4, chess.WHITE, values) == (3, 0)
    assert favorable_attack(after, chess.E4, chess.WHITE, values) == 0
    assert protected_mobility_gain(before, after, chess.E4, chess.WHITE, values) == 0
    assert position_pressure(after, values, chess.WHITE)[0] > 0


def test_activity_bonus_splits_protection_and_mobility_evenly(monkeypatch):
    import numpy as np
    import torch
    import tonkai.cognitive_agent as actor
    from tonkai.cognitive_config import CognitiveConfig
    from tonkai.cognitive_trainer import CognitiveTrainer
    from tonkai.cognitive_agent import CognitiveAgent
    torch.set_num_threads(1)
    cfg = CognitiveConfig(piece_safety_weight=0, king_safety_weight=0, piece_activity_weight=0)
    trainer = CognitiveTrainer(cfg)
    agent = CognitiveAgent(trainer.net, cfg, trainer.knowledge)
    monkeypatch.setattr(agent, '_robust', lambda *a, **kw: {})
    board = chess.Board()
    baseline = agent.evaluate(board, chess.WHITE, explore=False)
    target = baseline.moves.index(chess.Move.from_uci('e2e4'))
    signals = {'protection': 0, 'mobility': 0}
    monkeypatch.setattr(actor, 'position_pressure', lambda branch, *a: (
        0, signals['protection'] if branch.peek().to_square == chess.E4 else 0))
    monkeypatch.setattr(actor, 'favorable_attack', lambda *a: 0)
    monkeypatch.setattr(actor, 'protected_mobility_gain', lambda *a: (
        signals['mobility'] if a[2] == chess.E4 else 0))
    cfg.piece_activity_weight = .08
    signals['protection'] = 1
    protection = agent.evaluate(board, chess.WHITE, explore=False)
    signals['protection'] = 0
    signals['mobility'] = 1
    mobility = agent.evaluate(board, chess.WHITE, explore=False)
    assert protection.scores[target] - baseline.scores[target] == pytest.approx(.04)
    assert mobility.scores[target] - baseline.scores[target] == pytest.approx(.04)
    np.testing.assert_allclose(np.delete(protection.scores - baseline.scores, target), 0, atol=1e-12)
    np.testing.assert_allclose(np.delete(mobility.scores - baseline.scores, target), 0, atol=1e-12)


@pytest.mark.parametrize('difference', [.49, .5, .51])
def test_exact_tolerance_boundary(difference):
    row = sense('4k3/8/8/8/1p6/2N5/8/4K3 b - - 0 1', 'c3',
                lambda piece, sq: difference if piece.piece_type == chess.KNIGHT else values(piece, sq))
    assert row['priority'] == (2 if difference <= .5 else 1)


def test_safety_changes_policy_without_forbidding_sacrifices(monkeypatch):
    import numpy as np
    import torch
    from tonkai.cognitive_config import CognitiveConfig
    from tonkai.cognitive_trainer import CognitiveTrainer
    from tonkai.cognitive_agent import CognitiveAgent
    torch.set_num_threads(1)
    cfg = CognitiveConfig()
    trainer = CognitiveTrainer(cfg)
    agent = CognitiveAgent(trainer.net, cfg, trainer.knowledge)
    agent.use_fast_rules = False  # this test replaces the Python sense_pieces hook
    monkeypatch.setattr(agent.net, 'survival', lambda contexts: (
        torch.full((len(contexts), cfg.ensemble_size), .5), torch.zeros((len(contexts), cfg.ensemble_size))))
    monkeypatch.setattr(agent, '_robust', lambda *a, **kw: {})
    board = chess.Board()
    initial = list(board.legal_moves)[0]
    monkeypatch.setattr(agent, 'sense_pieces', lambda branch, *a, **kw: [
        {'color': 'w', 'risk': .9 if branch.peek() == initial else 0}])
    selected = agent.choose(board, True, np.random.default_rng(1), explore=False)
    assert selected.move != initial
    assert initial in selected.moves  # soft influence, no blanket ban on sacrifice
    assert agent.trace[-1]['piece_safety']['own_risk'] == 0
