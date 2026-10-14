"""Emergent King Safety (v4): a learned KS(position, king) in [0, 1].

Only the CONSEQUENCE is defined - vulnerability to checkmate - never what a
safe king looks like. With n = the number of moves the attacker needs to FORCE
checkmate against the best defence:

    U(position, king) = gamma ** (n - 1)      (1 = mated or unavoidably mated,
                                               0 = no mating attack at all)
    KS(position, king) = 1 - U

The input is the same raw encoding TonkAI uses everywhere (piece planes, side
to move, castling, en passant), seen from the side of the king being judged.
No pawn shields, attackers, defenders, open files, patterns or material: what
makes a king resist the strongest attack is for the network to discover
(tonkai/king_safety_training.py grounds it in exact mates and adversarial
attack/defence search).
"""
from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import torch
from torch import nn

from .encoding import INPUT_SIZE, encode

FORMAT = "tonkai-king-safety-v1"


class KingSafetyNet(nn.Module):
    """encoding (782, from the king owner's side) -> logit of KS."""

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.hidden = hidden
        self.body = nn.Sequential(nn.Linear(INPUT_SIZE, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x).squeeze(-1)                     # logit; sigmoid = KS


class KingSafetyModel:
    """KS for any position and king; inspectable, saved on its own."""

    def __init__(self, net: KingSafetyNet | None = None, meta: dict | None = None):
        self.net = (net or KingSafetyNet()).eval()
        self.meta = dict(meta or {})

    # ---------------------------------------------------------------- values
    @torch.inference_mode()
    def ks_rows(self, rows) -> np.ndarray:
        """KS for encoded rows (each already from its king owner's side)."""
        x = torch.as_tensor(np.asarray(rows, dtype=np.float32))
        if x.ndim == 1:
            x = x[None]
        if not len(x):
            return np.zeros(0, dtype=np.float32)
        return torch.sigmoid(self.net(x)).numpy()

    def ks(self, board: chess.Board, colour: bool) -> float:
        """KS(position, king of `colour`)."""
        return float(self.ks_rows(encode(board, colour))[0])

    def ks_both(self, board: chess.Board) -> tuple[float, float]:
        """(KS of the white king, KS of the black king)."""
        values = self.ks_rows(np.stack([encode(board, chess.WHITE), encode(board, chess.BLACK)]))
        return float(values[0]), float(values[1])

    def series(self, moves, initial_fen=None) -> list:
        """[KS white, KS black] after every ply of a game (as play_episode logs
        it), computed afterwards in one batch."""
        board = chess.Board(initial_fen or chess.STARTING_FEN)
        rows = [encode(board, chess.WHITE), encode(board, chess.BLACK)]
        for move in moves:
            board.push(move if isinstance(move, chess.Move) else chess.Move.from_uci(move))
            rows += [encode(board, chess.WHITE), encode(board, chess.BLACK)]
        values = self.ks_rows(np.stack(rows))
        return [[round(float(values[i]), 4), round(float(values[i + 1]), 4)] for i in range(0, len(values), 2)]

    def advantage(self, board: chess.Board, colour: bool) -> float:
        """KS_self - KS_enemy for `colour`, in [-1, 1]."""
        white, black = self.ks_both(board)
        return white - black if colour else black - white

    # ------------------------------------------------------------------ files
    def save(self, path, **extra) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({"format": FORMAT, "hidden": self.net.hidden, "state_dict": self.net.state_dict(),
                    "meta": {**self.meta, **extra}}, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path) -> "KingSafetyModel":
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} file")
        net = KingSafetyNet(data["hidden"])
        net.load_state_dict(data["state_dict"])
        return cls(net, data.get("meta"))


_CACHE: dict[str, KingSafetyModel] = {}


def load_cached(path) -> KingSafetyModel:
    """One copy per process (actors load it once)."""
    key = str(Path(path).resolve())
    if key not in _CACHE:
        _CACHE[key] = KingSafetyModel.load(key)
    return _CACHE[key]
