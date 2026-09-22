from types import SimpleNamespace

import pytest
import torch

from tonkai.config import Config
from tonkai.model import TonkAINet
from tonkai.temporal_model import SurvivalNet


@pytest.fixture
def net():
    torch.manual_seed(19)
    return SurvivalNet()


def test_shapes_and_configurable_independent_heads():
    cfg = SimpleNamespace(latent_size=40, hidden_size=72, memory_size=24,
                          ensemble_size=3, future_horizons=(1, 3), head_hidden=16)
    net = SurvivalNet(cfg)
    context, memory = net.observe(torch.randn(4, 782), net.initial_memory(4))
    assert context.shape == (4, 40)
    assert memory.shape == (4, 24)
    means, log_variances = net.survival(context)
    assert means.shape == log_variances.shape == (4, 3)
    assert torch.all((means >= 0) & (means <= 1))
    assert torch.all((log_variances >= -8) & (log_variances <= 2))
    assert net.future(context).shape == (4, 2, 40)
    assert net.novelty_error(context).shape == (4,)
    assert net.predict_logits(context).shape == (4,)
    assert net.piece_values(context, torch.tensor([0, 1, 4, 5]),
                            torch.tensor([0, 8, 60, 63]), torch.tensor([0, 1, 0, 1])).shape == (4,)
    assert net.n_parameters() == sum(p.numel() for p in net.parameters())
    assert not hasattr(net, "value_head")


def test_same_position_retains_different_histories(net):
    current = torch.randn(1, 782)
    _, first_history = net.observe(torch.randn(1, 782), net.initial_memory(1))
    _, second_history = net.observe(torch.randn(1, 782), net.initial_memory(1))
    first, _ = net.observe(current, first_history)
    second, _ = net.observe(current, second_history)
    assert not torch.allclose(first, second)


def test_candidates_share_the_prefix_without_mutating_or_leaking_state(net):
    memory = torch.randn(1, net.memory_size)
    original = memory.clone()
    candidates = torch.randn(5, 782)
    context, candidate_memory = net.candidates(candidates, memory)
    assert torch.equal(memory, original)
    for index in range(len(candidates)):
        one_context, one_memory = net.observe(candidates[index:index + 1], memory)
        torch.testing.assert_close(context[index:index + 1], one_context)
        torch.testing.assert_close(candidate_memory[index:index + 1], one_memory)
    repeated, repeated_memory = net.candidates(candidates, memory)
    assert torch.equal(context, repeated)
    assert torch.equal(candidate_memory, repeated_memory)
    first_reset = net.initial_memory(1)
    first_reset.add_(1)
    assert torch.count_nonzero(net.initial_memory(1)) == 0
    assert all("memory_state" not in key for key in net.state_dict())


def test_observe_encodes_once_and_context_reuses_the_existing_latent(net):
    calls = []
    handle = net.encoder.register_forward_hook(lambda *_: calls.append(1))
    x = torch.randn(3, 782)
    context, memory = net.observe(x, net.initial_memory(3))
    assert len(calls) == 1
    latent = net.encode(x)
    calls.clear()
    torch.testing.assert_close(net.context_from_latent(latent, memory), context)
    assert not calls
    handle.remove()


def test_gradients_cross_twenty_observations_without_detaching(net):
    positions = torch.randn(20, 2, 782, requires_grad=True)
    initial = net.initial_memory(2).requires_grad_()
    memory = initial
    for position in positions:
        context, memory = net.observe(position, memory)
    means, _ = net.survival(context)
    means.sum().backward()
    assert torch.isfinite(positions.grad).all()
    assert torch.all(positions.grad.abs().sum(dim=(1, 2)) > 0)
    assert initial.grad is not None and initial.grad.abs().sum() > 0
    assert net.memory_cell.weight_hh.grad.abs().sum() > 0
    assert net.encoder[0].weight.grad.abs().sum() > 0


def test_survival_bootstrap_heads_do_not_share_parameters(net):
    heads = [set(p.data_ptr() for p in head.parameters()) for head in net.survival_heads]
    assert all(not a.intersection(b) for index, a in enumerate(heads) for b in heads[index + 1:])
    untouched = {name: p.detach().clone() for name, p in net.survival_heads[1].named_parameters()}
    context = torch.randn(3, net.latent_size)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    means, _ = net.survival(context)
    means[:, 0].sum().backward()
    optimizer.step()
    for name, parameter in net.survival_heads[1].named_parameters():
        assert torch.equal(parameter, untouched[name])


def test_rnd_target_stays_frozen_and_in_eval_while_predictor_learns(net):
    before = {name: value.clone() for name, value in net.rnd_target.state_dict().items()}
    predictor_before = net.rnd_predictor[0].weight.detach().clone()
    net.train()
    assert not net.rnd_target.training
    assert all(not parameter.requires_grad for parameter in net.rnd_target.parameters())
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss = net.novelty_error(torch.randn(4, net.latent_size)).mean()
    loss.backward()
    optimizer.step()
    for name, value in net.rnd_target.state_dict().items():
        assert torch.equal(value, before[name])
    assert all(parameter.grad is None for parameter in net.rnd_target.parameters())
    assert not torch.equal(predictor_before, net.rnd_predictor[0].weight)


def test_survival_log_variances_are_clamped_per_head(net):
    with torch.no_grad():
        for index, head in enumerate(net.survival_heads):
            head[-1].weight.zero_()
            head[-1].bias[1] = -100 if index % 2 else 100
    _, variance = net.survival(torch.zeros(2, net.latent_size))
    assert variance[0].tolist() == [2, -8, 2, -8, 2]


@pytest.mark.parametrize("checkpoint", [False, True])
def test_legacy_migration_preserves_encoder_and_shared_heads_exactly(checkpoint):
    torch.manual_seed(71)
    legacy = TonkAINet(Config())
    net = SurvivalNet()
    recurrent_before = {name: value.clone() for name, value in net.memory_cell.state_dict().items()}
    new_before = {name: value.clone() for name, value in net.rnd_target.state_dict().items()}
    source = legacy.state_dict()
    report = net.load_legacy({"model": source, "games": 15} if checkpoint else source)
    assert report["loaded"]
    assert all(key.startswith("value_head.") for key in report["skipped"])
    for key in report["loaded"]:
        assert torch.equal(net.state_dict()[key], source[key])
    x = torch.randn(3, 782)
    assert torch.equal(net.encode(x), legacy.encode(x))
    for key, value in net.memory_cell.state_dict().items():
        assert torch.equal(value, recurrent_before[key])
    for key, value in net.rnd_target.state_dict().items():
        assert torch.equal(value, new_before[key])


def test_migration_skips_mismatched_and_nonlegacy_weights(net):
    original = net.encoder[0].weight.detach().clone()
    memory = net.memory_cell.weight_ih.detach().clone()
    report = net.load_legacy({
        "encoder.0.weight": torch.zeros(1, 1),
        "memory_cell.weight_ih": torch.zeros_like(memory),
        "predict_head.2.bias": torch.ones_like(net.predict_head[2].bias),
    })
    assert report["loaded"] == ["predict_head.2.bias"]
    assert report["skipped"] == ["encoder.0.weight", "memory_cell.weight_ih"]
    assert torch.equal(net.encoder[0].weight, original)
    assert torch.equal(net.memory_cell.weight_ih, memory)


def test_state_dict_roundtrip_retains_recurrent_future_and_fixed_random_targets(net):
    restored = SurvivalNet()
    restored.load_state_dict(net.state_dict())
    x, memory = torch.randn(3, 782), torch.randn(3, net.memory_size)
    first, next_first = net.observe(x, memory)
    second, next_second = restored.observe(x, memory)
    assert torch.equal(first, second) and torch.equal(next_first, next_second)
    assert torch.equal(net.future(first), restored.future(second))
    assert torch.equal(net.novelty_error(first), restored.novelty_error(second))


def test_initial_memory_tracks_dtype_and_rejects_invalid_batch_shapes(net):
    net.double()
    assert net.initial_memory(2).dtype == torch.float64
    assert net.initial_memory(0).shape == (0, net.memory_size)
    with pytest.raises(ValueError, match="batch"):
        net.initial_memory(-1)
    with pytest.raises(ValueError, match="memory batch"):
        net.observe(torch.randn(3, 782, dtype=torch.float64), net.initial_memory(2))
