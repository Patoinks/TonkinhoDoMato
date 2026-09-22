from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tonkai.population import Population
from tonkai.temporal_model import SurvivalNet


@pytest.fixture
def cfg():
    return SimpleNamespace(hidden_size=24, latent_size=16, head_hidden=8,
                           memory_size=8, ensemble_size=3, future_horizons=(1, 3),
                           population_size=4, external_probability=0.5,
                           historical_probability=0.3)


@pytest.fixture
def net(cfg):
    torch.manual_seed(51)
    return SurvivalNet(cfg)


def assert_frozen(net):
    assert not net.training
    assert not net.rnd_target.training
    assert all(parameter.device.type == "cpu" and not parameter.requires_grad
               for parameter in net.parameters())


def test_snapshots_and_current_copy_are_frozen_and_independent(cfg, net):
    population = Population(cfg, net)
    population.snapshot(net, 10)
    current = population.current(net)
    nets = [net, population.snapshots[0].net, population.snapshots[1].net, current]
    storages = [set(parameter.data_ptr() for parameter in model.parameters()) for model in nets]
    for index, pointers in enumerate(storages):
        assert all(not pointers.intersection(other) for other in storages[index + 1:])
    for frozen in nets[1:]:
        assert_frozen(frozen)
    original = population.snapshots[0].net.encoder[0].weight.clone()
    with torch.no_grad():
        net.encoder[0].weight.add_(10)
    assert torch.equal(population.snapshots[0].net.encoder[0].weight, original)
    assert torch.equal(current.encoder[0].weight, original)
    assert net.training and net.encoder[0].weight.requires_grad


def test_retention_preserves_anchor_latest_and_diverse_training_ages(cfg, net):
    population = Population(cfg, net)
    anchor = population.snapshots[0]
    for games in range(1, 101):
        population.snapshot(net, games)
        assert len(population) <= cfg.population_size
        assert population.snapshots[0] is anchor
        assert population.snapshots[-1].games == games
    ages = [entry.games for entry in population.snapshots]
    assert ages == sorted(ages)
    assert any(0 < age < 50 for age in ages)
    assert any(50 <= age < 100 for age in ages)
    assert len({entry.id for entry in population.snapshots}) == len(population)


def test_crowded_interior_age_is_removed_before_distinct_eras(cfg, net):
    population = Population(cfg, net)
    for games in (10, 20, 21, 22):
        population.snapshot(net, games)
    assert [entry.games for entry in population.snapshots] == [0, 10, 20, 22]


@pytest.mark.parametrize("capacity", [1, 2])
def test_small_capacity_keeps_anchor_and_latest_when_possible(cfg, net, capacity):
    cfg.population_size = capacity
    population = Population(cfg, net)
    for games in (10, 20, 30):
        population.snapshot(net, games)
    assert [entry.games for entry in population.snapshots] == ([0] if capacity == 1 else [0, 30])


def test_mixture_is_fixed_reproducible_and_has_current_residual(cfg, net):
    first = Population(cfg, net, seed=75)
    second = Population(cfg, net, seed=75)
    kinds = [first.choose_kind() for _ in range(12000)]
    assert kinds == [second.choose_kind() for _ in range(12000)]
    counts = Counter(kinds)
    for name, probability in (("external", 0.5), ("historical", 0.3), ("current", 0.2)):
        assert abs(counts[name] / len(kinds) - probability) < 0.02
    external = np.random.default_rng(123)
    control = np.random.default_rng(123)
    private_before = deepcopy(first.rng.bit_generator.state)
    assert [first.choose_kind(external) for _ in range(50)] == [second.choose_kind(control) for _ in range(50)]
    assert first.rng.bit_generator.state == private_before


def test_historical_sampling_is_uniform_across_retained_ages(cfg, net):
    population = Population(cfg, net, seed=26)
    for age in (5, 500, 5000):
        population.snapshot(net, age)
    counts = Counter(population.historical()[1] for _ in range(12000))
    assert set(counts) == {entry.name for entry in population.snapshots}
    assert all(abs(count / 12000 - 0.25) < 0.02 for count in counts.values())
    historical, name = population.historical(np.random.default_rng(3))
    assert historical is next(entry.net for entry in population.snapshots if entry.name == name)
    assert_frozen(historical)


def test_checkpoint_roundtrip_restores_full_models_ids_ages_and_rng(cfg, net, tmp_path):
    population = Population(cfg, net, seed=615)
    for age in (5, 30, 100, 250):
        with torch.no_grad():
            net.encoder[0].bias.add_(0.1)
        population.snapshot(net, age)
    for _ in range(9):
        population.choose_kind()
        population.historical()
    path = tmp_path / "population.pt"
    torch.save(population.state_dict(), path)
    restored = Population(cfg, SurvivalNet(cfg), seed=1)
    restored.load_state_dict(torch.load(path, weights_only=False))
    assert [(s.id, s.games) for s in restored.snapshots] == [(s.id, s.games) for s in population.snapshots]
    for actual, expected in zip(restored.snapshots, population.snapshots):
        assert_frozen(actual.net)
        for key, value in expected.net.state_dict().items():
            assert torch.equal(actual.net.state_dict()[key], value)
    expected = [(population.choose_kind(), population.historical()[1]) for _ in range(50)]
    assert expected == [(restored.choose_kind(), restored.historical()[1]) for _ in range(50)]
    population.snapshot(net, 300)
    restored.snapshot(net, 300)
    assert restored.snapshots[-1].id == population.snapshots[-1].id


def test_checkpoint_tensors_never_alias_live_snapshots(cfg, net):
    population = Population(cfg, net)
    state = population.state_dict()
    restored = Population(cfg, net)
    restored.load_state_dict(state)
    before = population.snapshots[0].net.encoder[0].weight.clone()
    state["snapshots"][0]["model"]["encoder.0.weight"].zero_()
    assert torch.equal(population.snapshots[0].net.encoder[0].weight, before)
    assert torch.equal(restored.snapshots[0].net.encoder[0].weight, before)


def test_invalid_restore_does_not_replace_existing_population(cfg, net):
    population = Population(cfg, net)
    anchor = population.snapshots[0]
    state = population.state_dict()
    state["snapshots"][0]["id"] = 3
    with pytest.raises(ValueError, match="anchor"):
        population.load_state_dict(state)
    assert population.snapshots == [anchor]


@pytest.mark.parametrize("external,historical", [(0.7, 0.4), (-0.1, 0.5), (float("nan"), 0.2)])
def test_invalid_mixture_is_rejected(cfg, net, external, historical):
    cfg.external_probability = external
    cfg.historical_probability = historical
    with pytest.raises(ValueError, match="mixture"):
        Population(cfg, net)
