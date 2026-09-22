"""Export TonkAI for a browser, with golden vectors to verify the port.

    python export_web.py --checkpoint checkpoints/latest.pt --out web

Writes into `out/`:

    tonkai.bin      raw float32 weights, little-endian, concatenated
    tonkai.json     manifest: layer shapes and byte offsets into tonkai.bin
    golden.json     positions + expected outputs, to test the JS port against
    ENCODING.md     the exact 782-feature layout

The whole network is 131k parameters, so it runs client-side with no server.
The forward pass is five matmuls - reimplementing it in JS is less work than
pulling in a runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import chess
import numpy as np
import torch

import tonkai
from tonkai import encoding as E
from tonkai.config import Config

# positions chosen to exercise every part of the encoding
GOLDEN_FENS = [
    ("startpos_white", chess.STARTING_FEN, chess.WHITE),
    ("startpos_black", chess.STARTING_FEN, chess.BLACK),
    ("after_e4_black", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
     chess.BLACK),
    ("italian_white",
     "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
     chess.WHITE),
    ("castling_rights",
     "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 7 20", chess.WHITE),
    ("castling_partial",
     "r3k2r/8/8/8/8/8/8/R3K3 w Qkq - 0 1", chess.BLACK),
    ("endgame_white", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", chess.WHITE),
    ("promotion_black", "8/P6k/8/8/8/8/8/K7 b - - 12 40", chess.BLACK),
    ("halfmove_high", "8/8/4k3/8/8/4K3/8/6R1 w - - 99 80", chess.WHITE),
]


def export_weights(net, out_dir):
    tensors, manifest, offset = [], [], 0
    for name, p in net.state_dict().items():
        arr = p.detach().cpu().numpy().astype("<f4")
        tensors.append(arr.tobytes(order="C"))
        manifest.append({
            "name": name,
            "shape": list(arr.shape),
            "offset": offset,
            "count": int(arr.size),
        })
        offset += arr.nbytes
    blob = b"".join(tensors)
    with open(os.path.join(out_dir, "tonkai.bin"), "wb") as fh:
        fh.write(blob)
    return manifest, blob, offset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/latest.pt")
    ap.add_argument("--out", default="web")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = Config()
    games = 0
    if os.path.exists(args.checkpoint):
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = Config.from_dict(payload.get("config", {}))
        games = payload.get("games", 0)
    cfg.device = "cpu"
    net, agent, opponent, worker, trainer, metrics = tonkai.build(cfg, "cpu")
    if os.path.exists(args.checkpoint):
        trainer.load(args.checkpoint, load_optimizer=False)
    net.eval()

    manifest, blob, nbytes = export_weights(net, args.out)
    digest = hashlib.sha256(blob).hexdigest()

    with open(os.path.join(args.out, "tonkai.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "format": "float32-le",
            "binary": "tonkai.bin",
            "bytes": nbytes,
            "sha256": digest,
            "parameters": int(sum(m["count"] for m in manifest)),
            "input_size": E.INPUT_SIZE,
            "latent_size": cfg.latent_size,
            "hidden_size": cfg.hidden_size,
            "draw_weight": cfg.draw_weight,
            "activation": "silu",
            "games_trained": games,
            "tensors": manifest,
        }, fh, indent=1)

    # ------------------------------------------------------------- golden
    golden = []
    for label, fen, color in GOLDEN_FENS:
        board = chess.Board(fen)
        vec = E.encode(board, color)
        x = torch.from_numpy(vec[None, :])
        with torch.no_grad():
            latent = net.encode(x)
            vlogits = net.value_logits(latent)[0].numpy()
            plogit = float(net.predict_logits(latent)[0])
        p = torch.softmax(torch.from_numpy(vlogits), dim=-1).numpy()
        entry = {
            "label": label,
            "fen": fen,
            "self_is_white": bool(color == chess.WHITE),
            "active_features": [int(i) for i in np.flatnonzero(vec)],
            "feature_values": [round(float(vec[i]), 6)
                               for i in np.flatnonzero(vec)],
            "feature_sum": round(float(vec.sum()), 6),
            "latent_sum": round(float(latent.sum()), 6),
            "value_logits": [round(float(v), 6) for v in vlogits],
            "win_draw_loss": [round(float(v), 6) for v in p],
            "utility": round(float(p[0] + cfg.draw_weight * p[1] - p[2]), 6),
            "predict_logit": round(plogit, 6),
        }
        golden.append(entry)

    # one full move ranking, the thing the site actually renders
    board = chess.Board(GOLDEN_FENS[3][1])
    mc = agent.evaluate_moves(board, chess.WHITE)
    ranking = [{
        "uci": mc.moves[i].uci(),
        "win_draw_loss": [round(float(v), 6) for v in mc.probs[i]],
        "utility": round(float(mc.utilities[i]), 6),
    } for i in (-mc.utilities).argsort()]

    # One full predict distribution. NOTE: the predict head is only ever
    # trained on positions where the OPPONENT is to move, so self must be the
    # side that just moved - here white, having played 1.e4.
    b2 = chess.Board(GOLDEN_FENS[2][1])
    pmoves, pprobs, _ = agent.predict_enemy(b2, chess.WHITE)
    predict = [{"uci": m.uci(), "p": round(float(q), 6)}
               for m, q in sorted(zip(pmoves, pprobs), key=lambda t: -t[1])]

    with open(os.path.join(args.out, "golden.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "note": "Expected outputs from the reference Python implementation. "
                    "A correct JS port must match value_logits to ~1e-4.",
            "sha256": digest,
            "tolerance": 1e-4,
            "positions": golden,
            "move_ranking": {"fen": GOLDEN_FENS[3][1], "self_is_white": True,
                             "moves": ranking},
            "predict_distribution": {"fen": GOLDEN_FENS[2][1],
                                     "self_is_white": True, "moves": predict},
        }, fh, indent=1)

    with open(os.path.join(args.out, "ENCODING.md"), "w", encoding="utf-8") as fh:
        fh.write(ENCODING_DOC)

    print(f"checkpoint     : {args.checkpoint} ({games} games)")
    print(f"parameters     : {sum(m['count'] for m in manifest):,}")
    print(f"tonkai.bin     : {nbytes:,} bytes  sha256 {digest[:16]}...")
    print(f"tensors        : {len(manifest)}")
    print(f"golden.json    : {len(golden)} positions + "
          f"{len(ranking)} ranked moves + {len(predict)} predictions")
    print(f"written to     : {os.path.abspath(args.out)}")
    return 0


ENCODING_DOC = """# TonkAI input encoding (782 floats)

A JS port MUST reproduce this exactly. The network will run happily on a wrong
encoding and return confident nonsense, so verify against `golden.json` before
trusting anything.

## Orientation

Everything is written from **TonkAI's own side**, never from white's. TonkAI
does not know which colour it is playing.

* If TonkAI is WHITE: squares are used as-is.
* If TonkAI is BLACK: every square index is flipped vertically, `sq ^ 56`.
  Files are NOT mirrored.

Squares are `0 = a1 .. 63 = h8`, i.e. `square = rank * 8 + file` (the
python-chess convention, same as chess.js `SQUARES` order reversed - check it).

## Layout

| index | meaning |
|---|---|
| 0-63 | SELF pawns |
| 64-127 | SELF knights |
| 128-191 | SELF bishops |
| 192-255 | SELF rooks |
| 256-319 | SELF queens |
| 320-383 | SELF king |
| 384-447 | OPPONENT pawns |
| 448-511 | OPPONENT knights |
| 512-575 | OPPONENT bishops |
| 576-639 | OPPONENT rooks |
| 640-703 | OPPONENT queens |
| 704-767 | OPPONENT king |
| 768 | 1.0 if it is SELF to move, else 0.0 |
| 769 | SELF may castle kingside |
| 770 | SELF may castle queenside |
| 771 | OPPONENT may castle kingside |
| 772 | OPPONENT may castle queenside |
| 773-780 | en-passant target FILE, one-hot a..h (all zero if none) |
| 781 | `min(halfmove_clock, 100) / 100` |

Indices 0-780 are strictly 0.0 or 1.0. Index 781 is the only real-valued one.
The en-passant file is NOT flipped when playing black (files never are).

## Forward pass

```
h1     = silu(x  @ W_enc0.T + b_enc0)      // 782 -> 128
latent = silu(h1 @ W_enc2.T + b_enc2)      // 128 -> 128

v1     = silu(latent @ W_val0.T + b_val0)  // 128 -> 32
logits = v1 @ W_val2.T + b_val2            // 32  -> 3   [win, draw, loss]

p1     = silu(latent @ W_prd0.T + b_prd0)  // 128 -> 32
plogit = p1 @ W_prd2.T + b_prd2            // 32  -> 1
```

`silu(x) = x / (1 + exp(-x))`. Weights are stored row-major as PyTorch keeps
them: a `Linear(in, out)` weight has shape `[out, in]`, so it is applied as
`y[o] = sum_i W[o][i] * x[i] + b[o]`.

## Choosing a move

For every legal move: apply it, encode the RESULTING position (still from
TonkAI's own side), and run the forward pass. Then

```
[pw, pd, pl] = softmax(value_logits)
U = pw + draw_weight * pd - pl        // draw_weight is in tonkai.json
```

and play the highest `U`. All legal moves go through as one batch.

## Predicting the human's reply

Same thing, but the positions are the ones after each of the OPPONENT's legal
replies, still encoded from TonkAI's side, and the score is `predict_logit`.
Softmax over exactly the legal replies gives the distribution.
`surprise = -log(p_of_the_move_actually_played)`.
"""


if __name__ == "__main__":
    raise SystemExit(main())
