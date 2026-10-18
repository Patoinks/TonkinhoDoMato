"""The Leto network in numpy, for the small batches of one decision.

torch's fixed cost per operation dominates at ~35 positions; numpy float32 with one BLAS thread does
the same layers with far less overhead. The first layer (782 -> 512) only multiplies the input
columns that are non-zero in some row of the batch (piece planes are mostly zero): adding exact
zeros is skipped, the rest is the same product. Results match torch to float32 rounding, not bit for
bit (a different summation order), so a game can differ in rare near-ties; strength is checked in the
mutation tournament.

    from rede_rapida import acelerar
    acelerar(net)          # replaces encode/memory_cell/candidates/survival/predict_logits in place
"""
import numpy as np
import torch
from torch import nn


def _np(t):
    return t.detach().cpu().numpy().astype(np.float32, copy=True)


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class _Encoder:
    """Sequential of Linear / SiLU / Residual, as built by claudio_net."""

    def __init__(self, sequential):
        self.steps = []
        for module in sequential:
            if isinstance(module, nn.Linear):
                self.steps.append(("linear", _np(module.weight).T.copy(), _np(module.bias)))
            elif isinstance(module, nn.SiLU):
                self.steps.append(("silu",))
            elif type(module).__name__ == "Residual":
                self.steps.append(("residual", _np(module.fc1.weight).T.copy(), _np(module.fc1.bias),
                                   _np(module.fc2.weight).T.copy(), _np(module.fc2.bias)))
            else:
                raise TypeError(f"unsupported encoder layer {type(module).__name__}")

    def __call__(self, x):
        first = True
        for step in self.steps:
            if step[0] == "linear":
                weight, bias = step[1], step[2]
                if first:
                    cols = np.flatnonzero(np.any(x != 0, axis=0))
                    x = x[:, cols] @ weight[cols] + bias
                else:
                    x = x @ weight + bias
                first = False
            elif step[0] == "silu":
                x = _silu(x)
            else:
                _, w1, b1, w2, b2 = step
                x = x + _silu(x @ w1 + b1) @ w2 + b2
        return x


def acelerar(net):
    if getattr(net, "_rede_rapida", False):
        return net
    encoder = _Encoder(net.encoder)
    cell = net.memory_cell
    w_ih, w_hh = _np(cell.weight_ih).T.copy(), _np(cell.weight_hh).T.copy()
    b_ih, b_hh = _np(cell.bias_ih), _np(cell.bias_hh)
    hidden = cell.hidden_size
    fusion_w, fusion_b = _np(net.fusion[0].weight).T.copy(), _np(net.fusion[0].bias)
    heads = net.survival_heads
    s1_w = np.concatenate([_np(h[0].weight) for h in heads]).T.copy()          # (D, E*H)
    s1_b = np.concatenate([_np(h[0].bias) for h in heads])
    s2_w = np.stack([_np(h[2].weight) for h in heads])                          # (E, 2, H)
    s2_b = np.stack([_np(h[2].bias) for h in heads])                            # (E, 2)
    p1_w, p1_b = _np(net.predict_head[0].weight).T.copy(), _np(net.predict_head[0].bias)
    p2_w, p2_b = _np(net.predict_head[2].weight).T.copy(), _np(net.predict_head[2].bias)

    def gru(x, h):
        gi = x @ w_ih + b_ih
        gh = h @ w_hh + b_hh
        r = _sigmoid(gi[:, :hidden] + gh[:, :hidden])
        z = _sigmoid(gi[:, hidden:2 * hidden] + gh[:, hidden:2 * hidden])
        n = np.tanh(gi[:, 2 * hidden:] + r * gh[:, 2 * hidden:])
        return (1.0 - z) * n + z * h

    def as_np(t):
        return t.detach().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)

    def encode(x):
        return torch.from_numpy(encoder(as_np(x).astype(np.float32, copy=False)))

    def memory_cell(latent, memory):
        lat, mem = as_np(latent), as_np(memory)
        if mem.shape[0] == 1 and lat.shape[0] > 1:
            mem = np.broadcast_to(mem, (lat.shape[0], mem.shape[1]))
        return torch.from_numpy(gru(lat, mem).astype(np.float32, copy=False))

    def candidates(x, memory):
        lat = encoder(as_np(x).astype(np.float32, copy=False))
        mem = as_np(memory)
        if mem.shape[0] == 1 and lat.shape[0] > 1:
            mem = np.broadcast_to(mem, (lat.shape[0], mem.shape[1]))
        after = gru(lat, mem)
        context = _silu(np.concatenate([lat, after], axis=1) @ fusion_w + fusion_b)
        return (torch.from_numpy(context.astype(np.float32, copy=False)),
                torch.from_numpy(after.astype(np.float32, copy=False)))

    def survival(context):
        c = as_np(context)
        h = _silu(c @ s1_w + s1_b).reshape(len(c), len(heads), -1)            # (B, E, H)
        raw = np.einsum("beh,eoh->beo", h, s2_w) + s2_b
        return (torch.from_numpy(_sigmoid(raw[..., 0]).astype(np.float32, copy=False)),
                torch.from_numpy(np.clip(raw[..., 1], -8.0, 2.0).astype(np.float32, copy=False)))

    def predict_logits(context):
        c = as_np(context)
        return torch.from_numpy((_silu(c @ p1_w + p1_b) @ p2_w + p2_b)[:, 0].astype(np.float32, copy=False))

    # object.__setattr__: memory_cell is a registered submodule; the instance attribute shadows it.
    for name, function in (("encode", encode), ("memory_cell", memory_cell), ("candidates", candidates),
                           ("survival", survival), ("predict_logits", predict_logits), ("_rede_rapida", True)):
        object.__setattr__(net, name, function)
    return net


def acelerar_king(model):
    """The king-safety model (782 -> h -> h -> 1, sigmoid) in numpy, same sparse first layer."""
    if model is None or getattr(model, "_rede_rapida", False):
        return model
    encoder = _Encoder(model.net.body[:-1])
    last = model.net.body[-1]
    w, b = _np(last.weight).T.copy(), _np(last.bias)

    def ks_rows(rows):
        x = np.asarray(rows, dtype=np.float32)
        if x.ndim == 1:
            x = x[None]
        if not len(x):
            return np.zeros(0, dtype=np.float32)
        return _sigmoid(encoder(x) @ w + b)[:, 0].astype(np.float32, copy=False)

    model.ks_rows = ks_rows
    model._rede_rapida = True
    return model
