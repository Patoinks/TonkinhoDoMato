"""Owner's fast-win objective: in a won game every position also earns
gamma ** (half-moves until the win); losses and draws earn 0, unfinished
games keep their target."""
from types import SimpleNamespace

import pytest
import torch

from tonkai.cognitive_config import CognitiveConfig
from tonkai.objective import observed_target, win_speed
from tonkai.survival_trainer import blend_win_speed


def test_a_nearer_win_is_worth_more():
    assert win_speed(True, 0, 0.99) == 1.0
    assert win_speed(True, 40, 0.99) == pytest.approx(0.99 ** 40)
    assert win_speed(True, 40, 0.99) > win_speed(True, 120, 0.99)
    assert win_speed(False, 10, 0.99) == 0.0


def test_training_targets_blend_the_speed_of_finished_games_only():
    targets = torch.full((3, 2), 0.6)
    remaining = torch.tensor([[2, 1], [2, 1], [2, 1]])
    won = torch.tensor([[True], [False], [False]])
    terminated = torch.tensor([[True], [True], [False]])      # won, lost/drawn, unfinished
    out = blend_win_speed(targets, won, terminated, remaining, 0.3, 0.9)
    assert out[0].tolist() == pytest.approx([0.7 * 0.6 + 0.3 * 0.81, 0.7 * 0.6 + 0.3 * 0.9])
    assert out[1].tolist() == pytest.approx([0.42, 0.42])
    assert out[2].tolist() == pytest.approx([0.6, 0.6])


def test_observed_targets_use_the_same_rule():
    cfg = SimpleNamespace(objective="shaped", survival_gamma=0.99, victory_weight=0.5,
                          win_speed_weight=0.3, win_speed_gamma=0.9)
    won = SimpleNamespace(actions=("a",) * 4, lost=False, terminated=True, result_value=1.0, proxies=[0.5] * 5)
    drawn = SimpleNamespace(actions=("a",) * 4, lost=False, terminated=True, result_value=0.5, proxies=[0.5] * 5)
    base = 0.5 * 1.0 + 0.5 * 0.5
    assert observed_target(won, 2, cfg) == pytest.approx(0.7 * base + 0.3 * 0.81)
    assert observed_target(drawn, 2, cfg) == pytest.approx(0.7 * base)
    cfg.win_speed_weight = 0.0
    assert observed_target(won, 2, cfg) == pytest.approx(base)                  # off: unchanged


def test_settings_are_validated():
    with pytest.raises(ValueError):
        CognitiveConfig(win_speed_weight=1.5).validate()
    with pytest.raises(ValueError):
        CognitiveConfig(win_speed_gamma=0.0).validate()
