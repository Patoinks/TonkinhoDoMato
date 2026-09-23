"""Opt-in execution paths. No parameter, target, replay or search-budget changes."""
import torch
import torch.nn.functional as F


def compact_encode(encoder, x, valid):
    """Keep the original B,T layout; encode only real states, with autograd."""
    flat = x.flatten(0, 1)
    indices = valid.flatten().nonzero(as_tuple=False).flatten()
    latent = encoder(flat.index_select(0, indices))
    # Padding is an all-zero input, not an all-zero latent: encoder biases
    # matter even for some masked diagnostics/replay priority calculations.
    padding = encoder(flat.new_zeros((1, flat.shape[-1])))
    output = padding.expand(len(flat), -1).clone()
    return output.index_copy(0, indices, latent).reshape(*x.shape[:2], -1)


@torch.inference_mode()
def incremental_encode(encoder, root, candidates):
    """Rebuild root every call, then apply exact feature deltas before SiLU.

    Uses FP32, including clock/castling/en-passant deltas. No accumulator lives
    across moves or weight changes. Arithmetic order can change rounding.
    """
    if candidates.device.type != "cpu" or candidates.dtype != torch.float32:
        raise ValueError("incremental encoder currently requires CPU FP32")
    first = encoder[0]
    base = F.linear(root.reshape(1, -1), first.weight, first.bias)
    delta = candidates - root.reshape(1, -1)
    rows, columns = delta.nonzero(as_tuple=True)
    contributions = first.weight.T.index_select(0, columns) * delta[rows, columns, None]
    hidden = base.expand(len(candidates), -1).clone()
    hidden.index_add_(0, rows, contributions)
    return encoder[3](encoder[2](encoder[1](hidden)))


@torch.inference_mode()
def batched_replies(net, encoded, memories, sizes):
    """One network call, independent softmax per root; never mix probabilities."""
    contexts, _ = net.candidates(encoded, memories)
    values = net.survival(contexts)[0].mean(-1).cpu().numpy()
    logits = net.predict_logits(contexts).split(sizes)
    probabilities = [part.softmax(0).cpu().numpy() for part in logits]
    start = 0
    outputs = []
    for size, probability in zip(sizes, probabilities):
        outputs.append((values[start:start + size].copy(), probability))
        start += size
    return outputs
