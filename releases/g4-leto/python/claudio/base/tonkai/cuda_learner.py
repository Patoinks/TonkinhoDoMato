"""Capture the recurrent learner/target regions; leave dynamic losses eager.

This is deliberately opt-in. Capture preserves the complete sequence and
burn-in detach boundaries. Optimizer steps, targets and replay remain normal.
"""
import torch
from torch import nn


class _Unroll(nn.Module):
    def __init__(self, net, function, burn, length, gradients, window):
        super().__init__()
        self.net, self.function = net, function
        self.burn, self.length, self.gradients, self.window = burn, length, gradients, window

    def forward(self, x, valid):
        return self.function(self.net, x, valid, self.burn, self.length,
                             self.gradients, learner_window_only=self.window)


class CudaUnroll:
    def __init__(self, trainer):
        if torch.device(trainer.device).type != 'cuda':
            raise ValueError("CUDA graphs require a CUDA learner")
        if getattr(trainer.cfg, 'compact_sequences', False):
            raise ValueError("compact sequences use dynamic shapes; test separately from CUDA graphs")
        self.function = trainer._unroll
        self.entries = {}

    def __call__(self, net, x, valid, burn_in, learn_length, gradients=False, *, learner_window_only=False):
        key = (id(net), tuple(x.shape), x.dtype, burn_in, learn_length, gradients, learner_window_only)
        if key not in self.entries:
            module = _Unroll(net, self.function, burn_in, learn_length, gradients, learner_window_only)
            # No RNG in this region. Capture warms up forward/backward without
            # optimizer steps; preserve any pre-existing parameter gradients.
            saved = [p.grad for p in net.parameters()]
            sx, sv = x.clone(), valid.clone()
            if gradients:
                with torch.enable_grad():
                    captured = torch.cuda.make_graphed_callables(module, (sx, sv),
                        num_warmup_iters=3, allow_unused_input=True)
                entry = ('train', captured)
            else:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream), torch.no_grad():
                    for _ in range(3): module(sx, sv)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph), torch.no_grad():
                    outputs = module(sx, sv)
                entry = ('target', graph, sx, sv, outputs)
            for p, grad in zip(net.parameters(), saved): p.grad = grad
            self.entries[key] = entry
        entry = self.entries[key]
        if entry[0] == 'train':
            return entry[1](x, valid)
        _, graph, sx, sv, outputs = entry
        sx.copy_(x); sv.copy_(valid)
        graph.replay()
        # Callers may retain results until backward; give each call ownership.
        return tuple(value.clone() for value in outputs)
