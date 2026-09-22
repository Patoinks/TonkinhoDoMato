# TonkAI input encoding (782 floats)

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
