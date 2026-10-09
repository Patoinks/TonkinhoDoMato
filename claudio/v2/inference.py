"""Frozen inference: pack existing heads once; reuse real trajectory encodings."""
import torch
from torch.nn import functional as F


class FrozenInference:
    def __init__(self, net):
        self.net = net.eval().requires_grad_(False)
        self.refresh()

    def refresh(self):
        heads = self.net.survival_heads
        self.first = torch.cat([h[0].weight.detach() for h in heads])
        self.first_bias = torch.cat([h[0].bias.detach() for h in heads])
        self.second = torch.stack([h[2].weight.detach() for h in heads])
        self.second_bias = torch.stack([h[2].bias.detach() for h in heads])

    def __getattr__(self, name):
        return getattr(self.net, name)

    def eval(self):
        self.net.eval()
        return self

    def requires_grad_(self, value):
        if value:
            raise ValueError('FrozenInference is only for actors; train the underlying model')
        self.net.requires_grad_(False)
        return self

    def load_state_dict(self, *args, **kwargs):
        result = self.net.load_state_dict(*args, **kwargs)
        self.refresh()
        return result

    def survival(self, context):
        hidden = F.silu(F.linear(context, self.first, self.first_bias))
        hidden = hidden.view(len(context), len(self.net.survival_heads), -1)
        raw = torch.einsum('beh,eoh->beo', hidden, self.second) + self.second_bias
        return torch.sigmoid(raw[..., 0]), raw[..., 1].clamp(-8.0, 2.0)


@torch.inference_mode()
def cached_surprises(player, board):
    """Learning labels with the real per-turn GRU history already observed by the actor.

    Finishing flushes the final actual moves, including positions skipped by mate fast paths.
    This removes a second encoder/GRU pass over the whole real trajectory. Opponent-candidate
    evaluation still uses the same frozen network, current-game perspective and memory prefix.
    """
    import numpy as np
    from tonkai import fast_rules
    player._advance(board, allow_other_turn=True)
    if len(player.real_rows) != len(board.move_stack) + 1:
        raise AssertionError('Incomplete cached game trajectory')
    me = player.colour
    fast = fast_rules.board(board.root())
    candidates, groups = [], []
    for ply, move in enumerate(board.move_stack):
        if fast.fen().split(' ', 2)[1] != ('w' if me else 'b'):
            codes = list(fast.legal_moves())
            candidates.append(fast_rules.encode_after_moves(fast, codes, me))
            groups.append((ply, len(codes), codes.index(fast_rules.code(move))))
        fast.push(fast_rules.code(move))
    surprise = np.zeros(len(board.move_stack) + 1, dtype=np.float32)
    if groups:
        memory = torch.cat([player.real_memory[ply].expand(n, -1) for ply, n, _ in groups])
        contexts, _ = player.net.candidates(torch.from_numpy(np.concatenate(candidates)), memory)
        logits = player.net.predict_logits(contexts).split([n for _, n, _ in groups])
        for (ply, _, actual), part in zip(groups, logits):
            surprise[ply + 1] = -float(np.log(max(float(part.softmax(0)[actual]), 1e-12)))
    return np.stack(player.real_rows), surprise
