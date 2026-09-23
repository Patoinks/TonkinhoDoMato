from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from tonkai.encoding import unpack_many
from tonkai.network_fastpaths import UnpackBuffer, bootstrap_values, cached_future, gru_step, reply_losses
from tonkai.temporal_model import SurvivalNet
from tonkai.cognitive_config import CognitiveConfig


def test_unpack_buffer_preserves_all_bytes_and_clock_values_and_reuses_storage():
    rng = np.random.default_rng(52)
    buffer = UnpackBuffer()
    for count in (256, 256, 13, 13):
        packed = rng.integers(0, 256, (count, 99), dtype=np.uint8)
        packed[:, -1] = np.arange(count, dtype=np.uint8)
        previous = buffer.output
        result = buffer(packed)
        np.testing.assert_array_equal(result, unpack_many(packed))
        if previous is not None and previous.shape == result.shape:
            assert result is previous


def test_batched_reply_loss_preserves_per_sequence_weights_and_gradients():
    torch.manual_seed(17)
    sizes, labels = [1, 7, 3, 25], [0, 2, 1, 12]
    reference = torch.randn(sum(sizes), dtype=torch.float64, requires_grad=True)
    actual = reference.detach().clone().requires_grad_(True)
    weights = torch.tensor([.2, 1., .7, .5], dtype=torch.float64)
    expected = torch.stack([F.cross_entropy(logit[None], torch.tensor([label]))
                            for logit, label in zip(reference.split(sizes), labels)])
    result = reply_losses(actual, sizes, labels)
    torch.testing.assert_close(result, expected, atol=1e-14, rtol=1e-14)
    (expected*weights).mean().backward()
    (result*weights).mean().backward()
    torch.testing.assert_close(actual.grad, reference.grad, atol=1e-14, rtol=1e-14)


def test_projected_gru_preserves_gradient_boundaries_and_padding_in_float64():
    torch.manual_seed(18)
    cell = torch.nn.GRUCell(5, 7).double()
    other = deepcopy(cell)
    x = torch.randn(3, 9, 5, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    valid = torch.ones(3, 9, dtype=torch.bool)
    valid[0, :2] = False
    valid[1, 7:] = False
    def unroll(module, values, projected):
        h = values.new_zeros(3, 7)
        inputs = F.linear(values, module.weight_ih, module.bias_ih) if projected else None
        states = []
        for t in range(9):
            if t == 3:
                h = h.detach()
            with torch.set_grad_enabled(t >= 3):
                candidate = gru_step(module, inputs[:, t], h) if projected else module(values[:, t], h)
                h = torch.where(valid[:, t, None], candidate, h)
                states.append(h)
        return torch.stack(states, 1)
    a, b = unroll(cell, x, False), unroll(other, y, True)
    torch.testing.assert_close(a, b, atol=1e-14, rtol=1e-14)
    a[:, 3:].square().sum().backward()
    b[:, 3:].square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-13, rtol=1e-13)
    assert torch.count_nonzero(y.grad[:, :3]) == 0
    for pa, pb in zip(cell.parameters(), other.parameters()):
        torch.testing.assert_close(pa.grad, pb.grad, atol=1e-13, rtol=1e-13)


def test_bootstrap_values_select_the_same_endpoints():
    net = SurvivalNet(CognitiveConfig()).eval()
    contexts = torch.randn(2, 15, net.latent_size)
    endpoints = torch.tensor([[4, 7, 9, 9], [6, 8, 11, 14]])
    with torch.no_grad():
        expected = net.survival(contexts.flatten(0, 1))[0].mean(-1).reshape(2, 15).gather(1, endpoints)
        actual = bootstrap_values(net, contexts, endpoints)
    torch.testing.assert_close(actual, expected)


def test_frozen_cache_keys_include_all_packed_position_bytes_and_eviction():
    encoder = torch.nn.Sequential(torch.nn.Linear(782, 12), torch.nn.SiLU()).eval().requires_grad_(False)
    trainer = SimpleNamespace(future_encoder=encoder)
    rng = np.random.default_rng(4)
    packed = rng.integers(0, 100, (1, 4, 99), dtype=np.uint8)
    def check():
        x = torch.from_numpy(unpack_many(packed.reshape(-1,99))).reshape(1,4,-1)
        with torch.no_grad():
            expected = F.normalize(encoder(x.flatten(0,1)),dim=-1).reshape(1,4,-1)
            actual = cached_future(trainer, {'packed':packed}, x, capacity=3)
        torch.testing.assert_close(actual, expected)
        assert len(trainer._future_cache) <= 3
    check()
    check()
    assert trainer._future_cache_stats[0] == 3
    packed[0,1,-1] += 1
    check()
