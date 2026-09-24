"""measure_lookahead: the shaped objective learns the measure one horizon
ahead (half bootstrapped there), or the finished game's result."""
import numpy as np
import pytest
import torch

from tonkai.survival_agent import play_episode
from tonkai.survival_trainer import SurvivalTrainer, lookahead_measure
from test_shaped import RandomEngine, agent_for, cognitive


def test_lookahead_blends_the_future_measure_with_the_value_there():
    ahead = torch.tensor([[0.8, 0.2]])
    bootstrap = torch.tensor([[0.4, 0.6]])
    ended = torch.tensor([[False, True]])
    assert lookahead_measure(ahead, bootstrap, ended)[0].tolist() == pytest.approx([0.6, 0.2])


def test_training_with_the_measure_alone_and_lookahead_runs():
    cfg = cognitive(objective="shaped", victory_weight=1.0, measure_lookahead=True,
                    victory_measure="simplify:1,material:1")
    trainer = SurvivalTrainer(cfg)
    agent = agent_for(cfg)
    for seed in range(4):
        episode, _, _, _ = play_episode(agent, RandomEngine(), cfg, np.random.default_rng(seed))
        trainer.replay.add(episode)
    result = trainer.update()
    assert result is not None and np.isfinite(result["value_loss"])


def test_future_states_are_read_where_asked():
    cfg = cognitive(objective="shaped")
    trainer = SurvivalTrainer(cfg)
    episode, _, _, _ = play_episode(agent_for(cfg), RandomEngine(), cfg, np.random.default_rng(1))
    trainer.replay.add(episode)
    batch = trainer.replay.sample(1, cfg.burn_in, cfg.sequence_length, 4)
    indices = np.array([[0, len(episode.proxies) - 1, len(episode.proxies) + 5]])
    batch["episodes"] = [episode]
    values, present = trainer._proxy_values(batch, indices)
    assert present.tolist() == [[True, True, False]]
    assert values[0, :2].tolist() == pytest.approx([episode.proxies[0], episode.proxies[-1]])
