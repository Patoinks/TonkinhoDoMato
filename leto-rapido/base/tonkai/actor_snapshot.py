"""Immutable actor snapshots with efficient, lossless tensor serialization."""
import pickle

import numpy as np
import torch


def _array(tensor):
    cpu = tensor.detach().cpu()
    try:
        return cpu.numpy()
    except TypeError:  # Preserve unusual torch dtypes unsupported by numpy.
        return cpu


def pack_snapshot(net, knowledge):
    # pickle.dumps completes before the learner can mutate these views. Numpy's
    # protocol-5 representation avoids a separate torch archive per tensor.
    weights = {key: _array(value) for key, value in net.state_dict().items()}
    state = knowledge.state_dict(include_episodes=False, copy=False)
    state['encoder'] = {key: _array(value) for key, value in state['encoder'].items()}
    return pickle.dumps((weights, state), protocol=pickle.HIGHEST_PROTOCOL)


def unpack_snapshot(payload):
    weights, state = pickle.loads(payload)
    def tensors(values):
        return {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                for key, value in values.items()}
    state['encoder'] = tensors(state['encoder'])
    return tensors(weights), state
