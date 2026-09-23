"""Resignation: decided only by TonkAI's own estimate, and gated so it cannot
eat the training data when results are constant."""
from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace

import chess
import numpy as np
import pytest

from tonkai.resign import (ACTIVE, OFF, SHADOW, VERIFY, ResignConfig, ResignController,
                           ResignTracker, best_hope, choose_mode, hope_threshold)
from tonkai.survival_agent import SurvivalChoice, _has_safe_ending, play_episode

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ helpers

def report(triggered=True, resigned=False, threshold=0.05, trigger_ply=50, plies=80):
    return {"mode": ACTIVE if resigned else SHADOW, "threshold": threshold,
            "triggered": triggered, "trigger_ply": trigger_ply if triggered else None,
            "resigned": resigned, "plies": plies, "lowest_hope": 0.01}


def controller(**overrides):
    values = dict(enabled=True, min_verified=20, window=50)
    values.update(overrides)
    return ResignController(ResignConfig(**values), 0.05)


class StubAgent:
    """Scores say 'great move', the network says 'hopeless'."""

    def __init__(self, mean, scores=0.99, balance=None):
        self.mean, self.scores, self.balance, self.trace = mean, scores, balance, []

    def reset(self):
        self.trace = []

    def observe(self, board, colour):
        return None

    def choose(self, board, colour, rng, **_):
        moves = list(board.legal_moves)
        n = len(moves)
        self.trace.append({"ply": len(board.move_stack)})
        return SurvivalChoice(moves, np.full(n, self.scores), np.full(n, self.mean),
                              np.zeros(n), np.zeros(n), np.zeros(n))

    def predict_reply(self, board, colour):
        moves = list(board.legal_moves)
        return moves, np.full(len(moves), 1 / len(moves))

    def piece_balance(self, board, colour):
        return self.balance


class FirstMove:
    def reset(self):
        pass

    def choose(self, board, rng):
        return sorted(board.legal_moves, key=lambda m: m.uci())[0]


def cfg():
    return SimpleNamespace(ensemble_size=3, temperature=0.1, max_plies=120)


# ------------------------------------------------------------- thresholds

def test_default_thresholds_follow_the_objective():
    rc = ResignConfig()
    assert hope_threshold(rc, "survival", 0.985) == pytest.approx(1 - 0.985 ** 10)
    assert hope_threshold(rc, "victory", 0.985) == pytest.approx(0.05)
    assert hope_threshold(ResignConfig(threshold=0.2), "survival", 0.985) == 0.2
    with pytest.raises(ValueError):
        hope_threshold(rc, "material", 0.985)


def test_best_hope_is_the_most_hopeful_move():
    assert best_hope([0.01, 0.4, 0.02]) == pytest.approx(0.4)


def test_piece_balance_threshold_is_signed_and_exclusive():
    rc = ResignConfig(enabled=True, piece_balance_threshold=-5)
    assert hope_threshold(rc, "shaped", .985) == -5
    assert ResignController(ResignConfig(enabled=True, piece_balance_threshold=-5, force=True), -5).plan()[-1] == "piece_balance"
    with pytest.raises(ValueError):
        ResignConfig(piece_balance_threshold=5).validate()
    with pytest.raises(ValueError):
        ResignConfig(threshold=.1, piece_balance_threshold=-5).validate()


# ---------------------------------------------------------------- tracker

def test_tracker_needs_patience_and_min_ply():
    tracker = ResignTracker(ACTIVE, 0.05, patience=3, min_ply=40)
    for ply in (30, 32, 34, 36, 38):              # hopeless but too early
        assert tracker.observe(0.0, ply) is False
    assert tracker.observe(0.0, 40) is False
    assert tracker.observe(0.5, 42) is False      # a hopeful decision resets it
    assert tracker.observe(0.0, 44) is False
    assert tracker.observe(0.0, 46) is False
    assert tracker.observe(0.0, 48) is True
    assert tracker.report(48)["trigger_ply"] == 48


def test_piece_balance_waits_for_three_further_worsening_turns():
    tracker = ResignTracker(ACTIVE, -5, patience=3, min_ply=40, metric="piece_balance")
    assert not tracker.observe(-6.0, 38)  # The threshold is ignored before move 40.
    assert not tracker.observe(-5.0, 40)  # Start observing, do not resign yet.
    assert not tracker.observe(-5.2, 42)
    assert not tracker.observe(-5.4, 44)
    assert tracker.observe(-5.6, 46)
    assert tracker.report(46)["trigger_ply"] == 46


def test_piece_balance_recovery_or_flat_reading_cancels_the_watch():
    tracker = ResignTracker(ACTIVE, -5, patience=3, min_ply=40, metric="piece_balance")
    for ply, balance in ((40, -5.5), (42, -5.8), (44, -1.0),  # queen recaptured
                         (46, -5.1), (48, -5.2), (50, -5.2),
                         (52, -5.3), (54, -5.4)):
        assert not tracker.observe(balance, ply)
    assert tracker.observe(-5.5, 56)


def test_constant_low_piece_balance_does_not_resign():
    tracker = ResignTracker(ACTIVE, -5, patience=3, min_ply=40, metric="piece_balance")
    assert not any(tracker.observe(-5.5, ply) for ply in range(40, 80, 2))


@pytest.mark.parametrize("mode", [SHADOW, VERIFY])
def test_shadow_and_verify_record_but_never_stop(mode):
    tracker = ResignTracker(mode, 0.05, patience=2, min_ply=0)
    assert not any(tracker.observe(0.0, ply) for ply in range(0, 20, 2))
    result = tracker.report(20)
    assert result["triggered"] and not result["resigned"] and result["trigger_ply"] == 2


def test_choose_mode():
    rng = np.random.default_rng(0)
    assert choose_mode(OFF, 0.1, rng) is None
    assert choose_mode(SHADOW, 0.1, rng) == SHADOW
    modes = [choose_mode(ACTIVE, 0.25, rng) for _ in range(4000)]
    assert set(modes) == {ACTIVE, VERIFY}
    assert modes.count(VERIFY) / len(modes) == pytest.approx(0.25, abs=0.03)


# ------------------------------------------------------------- controller

def test_disabled_controller_never_plans():
    ctl = ResignController(ResignConfig(enabled=False), 0.05)
    assert ctl.state == OFF and ctl.plan() is None
    assert ctl.update(None, lost=True) == []


def test_constant_losses_never_enable_resignation():
    """The trap: a saturated value head is 'right' every time TonkAI loses
    every game, so the playout check alone would approve resigning at move one.
    """
    ctl = controller()
    for _ in range(300):
        ctl.update(report(triggered=True), lost=True)
    assert ctl.false_rate == 0.0              # every resignation would have been right...
    assert ctl.state == SHADOW                # ...and it still must not turn on
    assert "constant" in ctl.reason


def test_enables_with_verified_triggers_low_false_rate_and_real_variance():
    ctl = controller()
    messages = []
    for i in range(60):
        if i % 4 == 0:
            messages += ctl.update(report(triggered=False), lost=False)   # a draw or win
        else:
            messages += ctl.update(report(triggered=True), lost=True)
    assert ctl.state == ACTIVE
    assert any("ACTIVE" in m for m in messages)


def test_too_many_wrong_resignations_tighten_the_threshold():
    ctl = controller()
    for i in range(20):
        ctl.update(report(triggered=True), lost=(i % 5 != 0))   # 20% would be wrong
    assert ctl.threshold == pytest.approx(0.05 * 0.8)
    assert len(ctl.verified) == 0
    assert ctl.state == SHADOW


def test_wrong_piece_balance_resignations_move_the_limit_further_negative():
    ctl = controller(piece_balance_threshold=-5)
    ctl.threshold = -5
    for i in range(20):
        ctl.update(report(triggered=True, threshold=-5), lost=(i % 5 != 0))
    assert ctl.threshold == pytest.approx(-6.25)


def test_only_current_threshold_verifications_count():
    ctl = controller()
    ctl.update(report(triggered=True, threshold=0.2), lost=False)
    assert len(ctl.verified) == 0


def test_active_falls_back_to_shadow_when_results_become_constant():
    ctl = controller()
    for i in range(60):
        ctl.update(report(triggered=i % 4 != 0), lost=i % 4 != 0)
    assert ctl.state == ACTIVE
    for _ in range(60):
        ctl.update(report(triggered=True, resigned=True), lost=False)
    assert ctl.state == SHADOW
    assert ctl.counts["resigned"] > 0


def test_plan_and_state_roundtrip():
    ctl = controller()
    for i in range(40):
        ctl.update(report(triggered=i % 3 != 0), lost=i % 3 != 0)
    other = controller()
    other.load_state_dict(ctl.state_dict())
    assert other.summary() == ctl.summary()
    assert other.plan() == ctl.plan()


# --------------------------------------------------------- in a real game

def test_resigns_on_the_network_estimate_not_on_the_heuristic_scores():
    """Scores claim every move is excellent; the raw network says hopeless.
    Resignation must follow the network."""
    agent = StubAgent(mean=0.0, scores=0.99)
    tracker = ResignTracker(ACTIVE, 0.05, patience=3, min_ply=10)
    episode, board, colour, stats = play_episode(agent, FirstMove(), cfg(),
                                                 np.random.default_rng(1),
                                                 self_color=chess.WHITE, resign=tracker)
    assert stats["resign"]["resigned"]
    assert episode.terminated and episode.lost               # resigning is losing
    assert episode.result_value == 0.0
    assert stats["result"] == "0-1"                          # TonkAI had white
    assert len(episode.actions) == len(episode.packed) - 1 == stats["plies"]
    assert len(agent.trace) == (stats["plies"] + 1) // 2    # the unplayed decision was dropped


def test_high_network_estimate_never_resigns():
    tracker = ResignTracker(ACTIVE, 0.05, patience=1, min_ply=0)
    _, _, _, stats = play_episode(StubAgent(mean=0.9, scores=0.0), FirstMove(), cfg(),
                                  np.random.default_rng(1), self_color=chess.WHITE, resign=tracker)
    assert not stats["resign"]["triggered"]


def test_piece_balance_resigns_below_minus_five_even_if_survival_head_is_high():
    class WorseningAgent(StubAgent):
        def piece_balance(self, board, colour):
            return -5.0 - 0.1 * len(board.move_stack)

    agent = WorseningAgent(mean=.9)
    tracker = ResignTracker(ACTIVE, -5, patience=3, min_ply=10, metric="piece_balance")
    episode, _, _, stats = play_episode(agent, FirstMove(), cfg(), np.random.default_rng(1),
                                        self_color=chess.WHITE, resign=tracker)
    assert episode.lost and stats["resign"]["resigned"]
    assert stats["resign"]["lowest_piece_balance"] < -5.0
    assert stats["resign"]["lowest_hope"] is None


def test_piece_balance_above_minus_five_does_not_use_low_survival_head():
    tracker = ResignTracker(ACTIVE, -5, patience=1, min_ply=0, metric="piece_balance")
    _, _, _, stats = play_episode(StubAgent(mean=0, balance=-4.9), FirstMove(), cfg(),
                                  np.random.default_rng(1), self_color=chess.WHITE, resign=tracker)
    assert not stats["resign"]["triggered"]


def test_without_resign_the_game_and_stats_are_unchanged():
    first = play_episode(StubAgent(mean=0.0), FirstMove(), cfg(), np.random.default_rng(3),
                         self_color=chess.BLACK)
    second = play_episode(StubAgent(mean=0.0), FirstMove(), cfg(), np.random.default_rng(3),
                          self_color=chess.BLACK)
    assert "resign" not in first[3]
    assert first[0].actions == second[0].actions
    assert np.array_equal(first[0].packed, second[0].packed)


def test_never_resigns_with_a_win_or_draw_on_the_board():
    mate_in_one = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")     # Ra8#
    assert _has_safe_ending(mate_in_one, chess.WHITE)
    assert mate_in_one.fen() == "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"    # board untouched
    assert not _has_safe_ending(chess.Board(), chess.WHITE)


# ----------------------------------------------------- stays in its lane

FORBIDDEN = {"tiny_opponent", "uci_opponent", "king_safety", "piece_safety",
             "cognitive_agent", "chess.engine", "tonkai.uci_opponent",
             "tonkai.king_safety", "tonkai.piece_safety", "tonkai.cognitive_agent"}


def test_resignation_code_never_touches_engines_material_or_heuristics():
    tree = ast.parse((ROOT / "tonkai" / "resign.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported |= {alias.name for alias in node.names}
    assert not (imported & FORBIDDEN), imported & FORBIDDEN
    source = (ROOT / "tonkai" / "resign.py").read_text(encoding="utf-8")
    for word in ("last_score", "choose_move", "piece_values", "pieces_mask", "scores"):
        assert word not in source.replace("move scores", ""), word
