"""A bigger Claudio network that starts out playing exactly like the current one.

Growing, not replacing (function-preserving, Net2Net style):
- every hidden layer is widened by copying units; the copies share the old
  unit's outgoing weights in random proportions that add up to 1, so the output
  is unchanged but the copies learn differently from the first update on;
- a residual block (latent -> wide -> latent) is added at the end of the encoder
  with its last layer at zero, so it starts as the identity and adds depth.
Nothing is learned from outside: the grown network knows exactly what Claudio knew.
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return x + self.fc2(F.silu(self.fc1(x)))


def add_residuals(net, count: int, hidden: int):
    for _ in range(count):
        net.encoder.append(Residual(net.latent_size, hidden))
    return net


@torch.no_grad()
def _widen(first: nn.Linear, second_layers: list[nn.Linear], new: int, generator) -> tuple[nn.Linear, list[nn.Linear]]:
    """first: in -> h, each second: h -> out. Returns the widened copies (in -> new, new -> out)."""
    old = first.out_features
    if new < old:
        raise ValueError("can only widen")
    source = torch.cat([torch.arange(old), torch.randint(0, old, (new - old,), generator=generator)])
    wide_first = nn.Linear(first.in_features, new)
    wide_first.weight.copy_(first.weight[source])
    wide_first.bias.copy_(first.bias[source])
    # Random shares per copy, summing to 1 for every original unit.
    shares = torch.rand(new, generator=generator) + 0.5
    totals = torch.zeros(old).index_add_(0, source, shares)
    shares = shares / totals[source]
    wide_second = []
    for second in second_layers:
        layer = nn.Linear(new, second.out_features)
        layer.weight.copy_(second.weight[:, source] * shares[None, :])
        layer.bias.copy_(second.bias)
        wide_second.append(layer)
    return wide_first, wide_second


def _widen_head(head: nn.Sequential, new: int, generator):
    first, second = _widen(head[0], [head[2]], new, generator)
    head[0], head[2] = first, second[0]


@torch.no_grad()
def grow(net, hidden: int, head_hidden: int, piece_hidden: int, residuals: int, residual_hidden: int, seed=0):
    """A widened, deeper copy of `net` computing the same function."""
    generator = torch.Generator().manual_seed(seed)
    big = copy.deepcopy(net)
    first, (second,) = _widen(big.encoder[0], [big.encoder[2]], hidden, generator)
    big.encoder[0], big.encoder[2] = first, second
    for head in [*big.survival_heads, *big.future_heads, big.rnd_predictor, big.rnd_target, big.predict_head]:
        _widen_head(head, head_hidden, generator)
    outputs = [big.piece_head[2]] + ([big.capture_out] if hasattr(big, "capture_out") else [])
    first, wide = _widen(big.piece_head[0], outputs, piece_hidden, generator)
    big.piece_head[0], big.piece_head[2] = first, wide[0]
    if hasattr(big, "capture_out"):
        big.capture_out = wide[1]
    big.rnd_target.requires_grad_(False)
    add_residuals(big, residuals, residual_hidden)
    return big


def build_net(architecture: dict):
    """SurvivalNet with the stored sizes, plus the residual blocks when present."""
    from types import SimpleNamespace
    from tonkai.temporal_model import SurvivalNet
    shape = dict(architecture)
    residuals = int(shape.pop("residuals", 0) or 0)
    residual_hidden = int(shape.pop("residual_hidden", 0) or 0)
    if "future_horizons" in shape:
        shape["future_horizons"] = tuple(shape["future_horizons"])
    net = SurvivalNet(SimpleNamespace(**shape))
    return add_residuals(net, residuals, residual_hidden)
