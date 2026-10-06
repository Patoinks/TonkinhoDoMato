"""Experimental data/math helpers; NOT enabled in the production trainer.

See docs/network_fastpaths.md. Trials did not establish a worthwhile, exactly
equivalent speedup. In particular, GRU/fused arithmetic changes FP32 trajectories.
"""
from collections import OrderedDict

import numpy as np
import torch
from torch.nn import functional as F


def gru_step(cell, input_gates, memory):
    """GRU recurrence after the input projection has been batched over time."""
    ir, iz, inn = input_gates.chunk(3, -1)
    hr, hz, hn = F.linear(memory, cell.weight_hh, cell.bias_hh).chunk(3, -1)
    reset, update = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
    candidate = torch.tanh(inn + reset * hn)
    return candidate + (memory - candidate) * update


def bootstrap_values(net, contexts, endpoints):
    selected = contexts.gather(1, endpoints[..., None].expand(-1, -1, contexts.shape[-1]))
    return net.survival(selected.flatten(0, 1))[0].mean(-1).reshape(endpoints.shape)


def reply_losses(logits, sizes, labels):
    """One equally weighted distribution per original sequence, ignoring padding."""
    padded = torch.nn.utils.rnn.pad_sequence(logits.split(sizes), batch_first=True,
                                            padding_value=-float('inf'))
    return F.cross_entropy(padded, torch.tensor(labels, device=logits.device), reduction='none')


class UnpackBuffer:
    """One CPU update at a time; the caller finishes backward before reusing it."""
    def __init__(self):
        self.output = None

    def __call__(self, packed):
        from .encoding import INPUT_SIZE, N_BINARY, OFF_HALFMOVE, PACK_BITS_BYTES
        shape = (len(packed), INPUT_SIZE)
        if self.output is None or self.output.shape != shape:
            self.output = np.empty(shape, dtype=np.float32)
        bits = np.unpackbits(packed[:, :PACK_BITS_BYTES], axis=1, bitorder='little')
        self.output[:, :N_BINARY] = bits[:, :N_BINARY]
        np.divide(packed[:, PACK_BITS_BYTES], np.float32(100), out=self.output[:, OFF_HALFMOVE],
                  dtype=np.float32)
        return self.output


def unpack_reusing(trainer, packed):
    if not hasattr(trainer, '_unpack_buffer'):
        trainer._unpack_buffer = UnpackBuffer()
    return trainer._unpack_buffer(packed)


def cached_future(trainer, batch, x, capacity=16384):
    """Experimental cache for a frozen encoder, CPU only, local to this trainer.

    Not serialized. Discard it when loading/replacing encoder weights. Packed
    position bytes include perspective and clocks; no latent from a mutable
    encoder is reused. Batch-size changes can affect FP32 rounding.
    """
    if not hasattr(trainer, '_future_cache'):
        trainer._future_cache = OrderedDict()
        trainer._future_cache_stats = [0, 0]
    cache = trainer._future_cache
    packed = batch['packed'].reshape(-1, batch['packed'].shape[-1])
    keys = [row.tobytes() for row in packed]
    missing = {}
    found = {}
    for index, key in enumerate(keys):
        if key in cache:
            found[key] = cache[key]
            cache.move_to_end(key)
            trainer._future_cache_stats[0] += 1
        else:
            missing.setdefault(key, index)
            trainer._future_cache_stats[1] += 1
    if missing:
        indices = torch.tensor(list(missing.values()), device=x.device)
        values = F.normalize(trainer.future_encoder(x.flatten(0, 1).index_select(0, indices)), dim=-1)
        for key, value in zip(missing, values):
            # Clone rows; do not keep the whole miss batch alive per cache entry.
            found[key] = cache[key] = value.detach().clone()
        while len(cache) > capacity:
            cache.popitem(last=False)
    return torch.stack([found[key] for key in keys]).reshape(*x.shape[:2], -1)
