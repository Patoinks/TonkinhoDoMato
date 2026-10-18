"""TonkAI.

One shared board encoder feeds three heads:

    board (782) -> Linear -> SiLU -> Linear -> SiLU -> latent (128)
                                                 |
                       +-------------------------+-------------------------+
                       |                         |                         |
                   ValueHead                PredictHead                PieceHead
                 P(win/draw/loss)     logit per candidate move     one scalar per piece

There is no centipawn anywhere in this file, and no hard-coded piece value.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .encoding import INPUT_SIZE


class TonkAINet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        h = cfg.hidden_size
        d = cfg.latent_size

        self.encoder = nn.Sequential(
            nn.Linear(INPUT_SIZE, h),
            nn.SiLU(),
            nn.Linear(h, d),
            nn.SiLU(),
        )

        self.value_head = nn.Sequential(
            nn.Linear(d, cfg.head_hidden),
            nn.SiLU(),
            nn.Linear(cfg.head_hidden, 3),
        )

        self.predict_head = nn.Sequential(
            nn.Linear(d, cfg.head_hidden),
            nn.SiLU(),
            nn.Linear(cfg.head_hidden, 1),
        )

        self.type_embed = nn.Embedding(6, cfg.type_embed)
        self.square_embed = nn.Embedding(64, cfg.square_embed)
        self.side_embed = nn.Embedding(2, cfg.side_embed)
        piece_in = d + cfg.type_embed + cfg.square_embed + cfg.side_embed
        self.piece_head = nn.Sequential(
            nn.Linear(piece_in, cfg.piece_hidden),
            nn.SiLU(),
            nn.Linear(cfg.piece_hidden, 1),
        )

        self.latent_size = d

    # ------------------------------------------------------------------ core

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    # ----------------------------------------------------------------- heads

    def value_logits(self, latent: torch.Tensor) -> torch.Tensor:
        """(N, 3) logits over win / draw / loss, from TonkAI's point of view."""
        return self.value_head(latent)

    def predict_logits(self, latent: torch.Tensor) -> torch.Tensor:
        """(N,) one logit per candidate opponent reply."""
        return self.predict_head(latent).squeeze(-1)

    def piece_values(
        self,
        latent: torch.Tensor,        # (M, d) - already gathered per piece
        piece_type: torch.Tensor,    # (M,) 0..5
        square: torch.Tensor,        # (M,) 0..63, in TonkAI's orientation
        side: torch.Tensor,          # (M,) 0 = self, 1 = opponent
    ) -> torch.Tensor:
        feats = torch.cat(
            [
                latent,
                self.type_embed(piece_type),
                self.square_embed(square),
                self.side_embed(side),
            ],
            dim=-1,
        )
        return self.piece_head(feats).squeeze(-1)

    # ------------------------------------------------------------- utilities

    def value_and_latent(self, x: torch.Tensor):
        latent = self.encode(x)
        return self.value_logits(latent), latent

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def parameter_report(self) -> str:
        rows = []
        for name, module in [
            ("encoder", self.encoder),
            ("value_head", self.value_head),
            ("predict_head", self.predict_head),
            ("piece_head", self.piece_head),
            ("embeddings", nn.ModuleList([self.type_embed, self.square_embed,
                                          self.side_embed])),
        ]:
            n = sum(p.numel() for p in module.parameters())
            rows.append(f"  {name:14s} {n:>9,}")
        rows.append(f"  {'TOTAL':14s} {self.n_parameters():>9,}")
        return "\n".join(rows)


def utility(value_logits: torch.Tensor, draw_weight: float) -> torch.Tensor:
    """U = P(win) + draw_weight * P(draw) - P(loss).

    The full distribution is kept everywhere else; this scalar exists only to
    rank moves.
    """
    p = F.softmax(value_logits, dim=-1)
    return p[..., 0] + draw_weight * p[..., 1] - p[..., 2]


def probabilities(value_logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(value_logits, dim=-1)
