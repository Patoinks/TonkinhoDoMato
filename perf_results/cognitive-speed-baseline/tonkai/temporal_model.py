"""Explicit recurrent state and learned prediction heads for survival training.

Memory belongs to the caller: observing an actual position advances it, while
scoring candidates returns independent hypothetical states. No game history is
stored on the module. The legacy encoder and auxiliary head names are retained
so a checkpoint can seed them without importing its W/D/L value head.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .config import Config
from .model import TonkAINet


class SurvivalNet(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        architecture = Config()
        for field in (
            "hidden_size", "latent_size", "head_hidden", "piece_hidden",
            "type_embed", "square_embed", "side_embed",
        ):
            setattr(architecture, field, getattr(cfg, field, getattr(architecture, field)))

        legacy = TonkAINet(architecture)
        for name in (
            "encoder", "predict_head", "piece_head",
            "type_embed", "square_embed", "side_embed",
        ):
            setattr(self, name, getattr(legacy, name))

        self.latent_size = architecture.latent_size
        self.memory_size = getattr(cfg, "memory_size", 64)
        self.ensemble_size = getattr(cfg, "ensemble_size", 5)
        self.future_horizons = tuple(getattr(cfg, "future_horizons", (1, 2, 4, 8)))
        if self.memory_size < 1 or self.ensemble_size < 1:
            raise ValueError("memory_size and ensemble_size must be positive")
        if (not self.future_horizons
                or any(not isinstance(h, int) or h < 1 for h in self.future_horizons)
                or len(set(self.future_horizons)) != len(self.future_horizons)):
            raise ValueError("future_horizons must contain distinct positive integers")

        d = self.latent_size
        h = architecture.head_hidden
        self.memory_cell = nn.GRUCell(d, self.memory_size)
        self.fusion = nn.Sequential(nn.Linear(d + self.memory_size, d), nn.SiLU())
        self.survival_heads = nn.ModuleList([
            self._head(d, h, 2) for _ in range(self.ensemble_size)
        ])
        self.future_heads = nn.ModuleList([
            self._head(d, h, d) for _ in self.future_horizons
        ])
        self.rnd_target = self._head(d, h, 32)
        self.rnd_predictor = self._head(d, h, 32)
        self.rnd_target.requires_grad_(False)
        self.rnd_target.eval()

    @staticmethod
    def _head(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the fixed random target immutable even when the parent is trained.
        self.rnd_target.eval()
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def initial_memory(self, batch: int, device=None) -> torch.Tensor:
        """A fresh zero state on the model's dtype/device unless overridden."""
        if batch < 0:
            raise ValueError("batch must be nonnegative")
        reference = next(self.encoder.parameters())
        return torch.zeros(batch, self.memory_size,
                           device=reference.device if device is None else device,
                           dtype=reference.dtype)

    def _memory_for_batch(self, memory: torch.Tensor, batch: int) -> torch.Tensor:
        if memory.ndim != 2 or memory.shape[1] != self.memory_size:
            raise ValueError(f"memory must have shape (batch, {self.memory_size})")
        if memory.shape[0] == batch:
            return memory
        if memory.shape[0] == 1:
            return memory.expand(batch, -1)
        raise ValueError("memory batch must match the positions or contain one shared state")

    def context_from_latent(self, latent: torch.Tensor,
                            memory_next: torch.Tensor) -> torch.Tensor:
        """Fuse an existing encoding with an already advanced memory state."""
        memory = self._memory_for_batch(memory_next, latent.shape[0])
        return self.fusion(torch.cat((latent, memory), dim=-1))

    def observe(self, x: torch.Tensor, memory: torch.Tensor):
        """Encode once, advance explicit history, and return (context, next state)."""
        latent = self.encode(x)
        memory_next = self.memory_cell(latent, self._memory_for_batch(memory, x.shape[0]))
        return self.context_from_latent(latent, memory_next), memory_next

    def candidates(self, x: torch.Tensor, memory_before_candidate: torch.Tensor):
        """Score hypothetical positions in parallel without committing any state.

        A (1, memory_size) state is shared by all candidates; a matching batch
        supplies one distinct prefix history per row. Neither input is mutated.
        """
        return self.observe(x, memory_before_candidate)

    def survival(self, context: torch.Tensor):
        """Independent means and aleatoric log variances, each shaped (B, H)."""
        raw = torch.stack([head(context) for head in self.survival_heads], dim=1)
        return torch.sigmoid(raw[..., 0]), raw[..., 1].clamp(-8.0, 2.0)

    def future(self, context: torch.Tensor) -> torch.Tensor:
        """One latent predictor per horizon, ordered as future_horizons: (B, K, D)."""
        return torch.stack([head(context) for head in self.future_heads], dim=1)

    def novelty_error(self, context: torch.Tensor) -> torch.Tensor:
        """Unnormalized RND error per position. Normalization belongs to the trainer."""
        with torch.no_grad():
            target = self.rnd_target(context)
        return F.mse_loss(self.rnd_predictor(context), target, reduction="none").mean(dim=-1)

    def piece_values(self, context: torch.Tensor, piece_type: torch.Tensor,
                     square: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        features = torch.cat((context, self.type_embed(piece_type),
                              self.square_embed(square), self.side_embed(side)), dim=-1)
        return self.piece_head(features).squeeze(-1)

    def predict_logits(self, context: torch.Tensor) -> torch.Tensor:
        return self.predict_head(context).squeeze(-1)

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def load_legacy(self, payload: Mapping) -> dict[str, list[str]]:
        """Copy compatible shared weights only; report all supplied source keys.

        Accept a raw state_dict or a checkpoint containing ``model`` (or
        ``state_dict``). Recurrent, survival, future and RND weights stay new.
        """
        if not isinstance(payload, Mapping):
            raise TypeError("legacy weights must be a state_dict or checkpoint mapping")
        source = payload.get("model", payload.get("state_dict", payload))
        if not isinstance(source, Mapping):
            raise TypeError("legacy checkpoint does not contain a state_dict")
        prefixes = ("encoder.", "predict_head.", "piece_head.",
                    "type_embed.", "square_embed.", "side_embed.")
        current = self.state_dict()
        compatible = {}
        skipped = []
        for key, value in source.items():
            if (isinstance(key, str) and key.startswith(prefixes)
                    and key in current and isinstance(value, torch.Tensor)
                    and value.shape == current[key].shape):
                compatible[key] = value
            else:
                skipped.append(str(key))
        self.load_state_dict(compatible, strict=False)
        return {"loaded": sorted(compatible), "skipped": sorted(skipped)}
