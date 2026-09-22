import chess
import numpy as np
import pytest
import torch

from tonkai.king_safety import king_senses, king_balance
from tonkai.cognitive_agent import CognitiveAgent
from tonkai.cognitive_config import CognitiveConfig
from tonkai.cognitive_trainer import CognitiveTrainer


SHELTER = '5rk1/8/8/2b4q/8/8/5PPP/6K1 w - - 0 1'


def test_removing_pawn_shelter_under_attack_increases_danger():
    board = chess.Board(SHELTER)
    protected = king_senses(board)['w']
    for square in (chess.F2, chess.G2, chess.H2):
        board.remove_piece_at(square)
    exposed = king_senses(board)['w']
    assert exposed['shelter'] == 0 and protected['shelter'] == 1
    assert exposed['risk'] > protected['risk']
    assert exposed['in_check']


def test_exposure_without_attack_is_not_punished_even_with_distant_rook():
    for fen in ('k7/8/8/8/8/8/8/6K1 w - - 0 1', 'r6k/8/8/8/8/8/8/6K1 w - - 0 1'):
        assert king_senses(chess.Board(fen))['w']['risk'] == 0


def test_pinned_bishop_is_not_a_useful_defender():
    board = chess.Board('k3r3/8/8/8/7q/8/4B3/4K3 w - - 0 1')
    assert 'e2' not in king_senses(board)['w']['defenders']
    board.remove_piece_at(chess.E8)
    assert 'e2' in king_senses(board)['w']['defenders']


def test_coordinated_attack_and_escape_restriction_matter():
    board = chess.Board(SHELTER)
    combined = king_senses(board)['w']
    board.remove_piece_at(chess.C5)
    lone = king_senses(board)['w']
    assert len(combined['attackers']) > len(lone['attackers'])
    assert combined['risk'] > lone['risk']


def test_escape_squares_match_legal_king_moves_including_xrays_and_captures():
    # e1 cannot retreat to f1 through the rook ray exposed by vacating e1.
    for fen in ('7k/8/8/8/8/8/8/r3K3 w - - 0 1', SHELTER,
                '7k/8/8/8/8/5n2/4r3/4K3 w - - 0 1'):
        board = chess.Board(fen)
        expected = sorted(chess.square_name(m.to_square) for m in board.legal_moves
                          if board.piece_type_at(m.from_square) == chess.KING and not board.is_castling(m))
        assert king_senses(board)['w']['escapes'] == expected


def test_colour_symmetry_history_preservation_and_finite_bounds():
    rng = np.random.default_rng(12)
    board = chess.Board()
    for _ in range(60):
        if board.is_game_over(): break
        board.push(list(board.legal_moves)[int(rng.integers(board.legal_moves.count()))])
        before, history = board.fen(), list(board.move_stack)
        senses, mirrored = king_senses(board), king_senses(board.mirror())
        assert board.fen() == before and board.move_stack == history
        for colour, flipped in (('w', 'b'), ('b', 'w')):
            assert 0 <= senses[colour]['risk'] <= 1
            assert senses[colour]['risk'] == pytest.approx(mirrored[flipped]['risk'])
        assert king_balance(senses, True) == -king_balance(senses, False)


def make_agent(monkeypatch, **options):
    torch.set_num_threads(1)
    cfg = CognitiveConfig(piece_safety_weight=0, memory_weight=0, skill_bonus=0, **options)
    trainer = CognitiveTrainer(cfg)
    agent = CognitiveAgent(trainer.net, cfg, trainer.knowledge)
    monkeypatch.setattr(agent.net, 'survival', lambda contexts: (
        torch.full((len(contexts), cfg.ensemble_size), .5), torch.zeros((len(contexts), cfg.ensemble_size))))
    return agent


def test_king_signal_changes_policy_and_is_in_trace(monkeypatch):
    agent = make_agent(monkeypatch)
    monkeypatch.setattr(agent, '_robust', lambda *a, **kw: {})
    board = chess.Board(SHELTER)
    choice = agent.choose(board, True, np.random.default_rng(1), explore=False)
    risks = []
    for move in choice.moves:
        board.push(move); risks.append(king_balance(king_senses(board), True)); board.pop()
    assert max(risks) > min(risks)
    assert choice.scores.argmax() == np.asarray(risks).argmax()
    assert 'king_safety' in agent.trace[-1]


def test_pressure_is_recomputed_after_enemy_defence(monkeypatch):
    import tonkai.cognitive_agent as module
    agent = make_agent(monkeypatch, planning_nodes=96, planning_candidates=3)
    board = chess.Board()
    # Every immediate attack looks excellent but disappears after ANY reply.
    def temporary_threat(branch):
        return {'w': {'risk': 0}, 'b': {'risk': .8 if len(branch.move_stack) == 1 else 0}}
    monkeypatch.setattr(module, 'king_senses', temporary_threat)
    scored = agent.evaluate(board, True, explore=False)
    searched = scored.scores[scored.scores > -1e8]
    np.testing.assert_allclose(searched, 1 - agent.cfg.survival_gamma + agent.cfg.survival_gamma * .5, atol=1e-7)
    assert agent.diagnostics['planning_nodes'] > 0


def test_weight_zero_disables_feature_and_invalid_weight_is_rejected(monkeypatch):
    import tonkai.cognitive_agent as module
    agent = make_agent(monkeypatch, king_safety_weight=0)
    monkeypatch.setattr(module, 'king_senses', lambda _: pytest.fail('Disabled king module was evaluated'))
    agent.evaluate(chess.Board(), True, explore=False)
    assert not agent._evaluation['king_safety']
    with pytest.raises(ValueError):
        CognitiveConfig(king_safety_weight=-.1).validate()
