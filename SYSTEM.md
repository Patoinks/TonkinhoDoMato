# TonkAI — system description

A self-contained technical description of what TonkAI is, how it works, and
what has actually been measured. Written to be handed to someone (or something)
with no prior context.

---

## 1. What it is, and what it refuses to be

TonkAI is a 131,701-parameter neural network that plays chess. It is **not**
trying to be strong, and it is not trying to imitate an engine. It is an
experiment in whether strategic concepts can *emerge* from four things only:

1. **Survival** — the real result of the game (win / draw / loss)
2. **Prediction** — what the opponent actually played next
3. **Risk vs return** — the full W/D/L distribution, never collapsed too early
4. **Contextual piece values** — one learned scalar per piece on the board

The following are **deliberately absent**, and their absence is the point:

- no piece values (no pawn=1, knight=3, …) anywhere in the model or its losses
- no centipawn evaluation
- no Stockfish or any engine as a teacher
- no MCTS, no minimax, no search of any kind inside TonkAI
- no opening book, no tablebases
- no reward for captures, checks, centre control, development, material, or novelty

The behaviour this is built to allow:

> *"This rook of mine looks more valuable than that knight, but trading it
> produces a position I consider better."*

That requires that the value of a piece be **contextual**, and that utility is
**never** a sum of piece values. Both are enforced by construction — see §6.

A test (`tests/test_model.py`) parses the whole package with `ast` and fails the
build if any list/tuple/dict of numbers looks like a classical piece-value table.

---

## 2. Shape of the system

```
          TinyOpponent (C++)
                 ↕
         chess environment (python-chess = source of truth for rules)
                 ↕
            TonkAI (PyTorch)
                 ↓
            Replay buffers
                 ↓
              Training
```

Two processes of thought, kept strictly separate:

- **TinyOpponent** is *only* an opponent. It has material values and
  piece-square tables inside it. **None of that is ever visible to TonkAI** —
  TonkAI sees board positions and game results, nothing else.
- **TonkAI** never receives a heuristic, an evaluation, or a hint.

---

## 3. Board encoding — 782 floats

This is the single most important detail to get right. A wrong encoding does
not throw an error; the network runs happily and returns confident nonsense.

| index | meaning |
|---|---|
| 0–63 | SELF pawns |
| 64–127 | SELF knights |
| 128–191 | SELF bishops |
| 192–255 | SELF rooks |
| 256–319 | SELF queens |
| 320–383 | SELF king |
| 384–447 | OPPONENT pawns |
| 448–511 | OPPONENT knights |
| 512–575 | OPPONENT bishops |
| 576–639 | OPPONENT rooks |
| 640–703 | OPPONENT queens |
| 704–767 | OPPONENT king |
| 768 | 1.0 if it is SELF to move |
| 769–772 | castling rights: self K, self Q, opp K, opp Q |
| 773–780 | en-passant target **file**, one-hot a–h (all zero if none) |
| 781 | `min(halfmove_clock, 100) / 100` |

**Orientation.** Everything is written from TonkAI's own side — `SELF` and
`OPPONENT`, never `WHITE` and `BLACK`. When TonkAI plays black, every square
index is flipped vertically (`sq ^ 56`). Files are *not* mirrored, and the
en-passant file is not flipped. TonkAI therefore learns a single perspective
and does not know which colour it is playing.

Squares are `0 = a1 … 63 = h8` (`square = rank * 8 + file`, the python-chess
convention).

Indices 0–780 are strictly 0.0 or 1.0; index 781 is the only real-valued one.
That is why a position packs into **99 bytes** (98 bytes of bits + 1 byte of
halfmove clock) for storage in the replay buffers.

---

## 4. Network

One shared encoder, three heads:

```
board (782)
   │
   ├─ Linear(782,128) → SiLU → Linear(128,128) → SiLU  ──►  latent (128)
   │
   ├──────────────┬──────────────────┬─────────────────┐
   ▼              ▼                  ▼                 │
ValueHead     PredictHead        PieceHead             │
128→32→3      128→32→1           (see §6)              │
W/D/L logits  logit per move     scalar per piece
```

| module | parameters |
|---|---:|
| encoder | 116,736 |
| value_head | 4,227 |
| predict_head | 4,161 |
| piece_head | 5,441 |
| embeddings | 1,136 |
| **total** | **131,701** |

No transformer, no CNN, no RNN. Policy and Predict deliberately **share the
same encoder** — the project wants one body of knowledge, not two networks.

---

## 5. How a turn works

**TonkAI's turn.** There is no fixed 4096-move output layer. Instead:

1. generate the legal moves (from python-chess)
2. apply each one and encode the **resulting position**
3. stack them into a single `(N, 782)` array
4. **one forward pass** for all N
5. the value head returns `P(win), P(draw), P(loss)` per move

The move is chosen by

```
U = P(win) + draw_weight · P(draw) − P(loss)        draw_weight = 0.1
```

sampled with a temperature (`argmax` at `temperature = 0`).

A single forward pass for the whole move list is a hard requirement, not an
optimisation — 22 legal moves means one batch of 22, never 22 forward passes.

**The opponent's turn.** Identical trick: every legal reply is applied and
encoded, the predict head puts one logit on each, and a softmax **over exactly
the legal replies** gives a distribution:

```
Nc6 25%   Nf6 18%   e5 17%   d5 11%   …
```

TinyOpponent then moves, and the predict head is trained to raise the
probability of the move that actually happened. `surprise = −log P(actual)`.

Tracked: top-1, top-3, `P(true move)`, entropy, surprise.

---

## 6. Contextual piece values — the subtle part

The piece head produces one scalar per piece on the board. Its input is:

```
latent (128) ⊕ piece-type embedding (16) ⊕ square embedding (16) ⊕ self/opp embedding (8)
      → Linear(168, 32) → SiLU → Linear(32, 1)
```

**It is never taught classical values.** Its training signal is *counterfactual*
and generated by TonkAI against itself:

```
target = U(position) − U(position with that piece erased from the input)
```

i.e. "how much worse do I think my survival gets if this piece vanishes". One
or two pieces per position per batch (kings excluded), loss weight 0.1, and the
target is detached.

Crucially, **utility is never computed as a sum of piece values**. The piece
values are a read-out of the network's own opinion, not the objective. That is
exactly what leaves room for a sacrifice: TonkAI can rate its own rook above an
enemy knight and still take the trade, because the resulting position scores
higher. Those events are logged as `CONTEXTUAL_SACRIFICE` with both valuations,
the utility before and after, and the eventual result.

---

## 7. TinyOpponent (C++)

A deliberately small opponent, exposed to Python through pybind11 and created
**once** per worker — never a process per move.

- own legal move generation (file/rank arithmetic, no bitboard tricks)
- tiny alpha-beta with **root iterative deepening**
- material + piece-square static evaluation
- short capture-only quiescence
- controlled stochastic selection among near-best root moves, seeded

**Correctness.** Move generation is verified by perft against the six standard
positions to depth 5–6 — **479,436,419 nodes, all exact** — and cross-checked
against python-chess. TonkAI can never exploit a movegen bug, and every move
TinyOpponent returns is re-validated against `board.legal_moves` before being
played.

**Strength dials** (measured over 80-game matches):

| knob | effect |
|---|---|
| `opponent_depth` | depth 2 is +470 Elo over depth 1; depth 3 is +512 |
| `opponent_nodes` | 100 / 300 / 1000 / 20000 spans roughly 700 Elo |
| `opponent_randomness` | 0.15 / 0.35 / 0.7 costs 44 / 132 / 215 Elo |
| quiescence off | −1600 Elo (loses 100%) |

These are **relative** numbers. There is no absolute Elo anchor yet; that needs
an external rated reference. All settings beat a uniform random mover 100%.

**Why root iterative deepening matters.** Without it, a tight node cap made the
root mix deep scores with shallow ones, and the engine got *worse* as depth
increased (depth 3 scored +70 Elo over depth 1, while depth 2 scored +352). A
depth's scores are now only accepted once *every* root move has been searched
at that depth, so running out of nodes degrades to a shallower but coherent
ranking.

---

## 8. Training

**Replay.** Positions are stored packed at 99 bytes each.

- value replay: 200k positions, each labelled with the game's final result
- predict replay: 20k samples, each = every legal opponent reply + which was played
  (capped at 48 candidates; wider positions are subsampled keeping the real move)

**Losses.**

```
total = 1.0 · CE(value_logits, real game result)
      + 1.0 · CE(predict_logits over legal replies, real reply)
      + 0.1 · MSE(piece_value, counterfactual contribution)
```

AdamW, lr 3e-4. Every position in a game receives the final result as its
label (V0 simplification; noisy by construction).

**Theory buffer.** When `surprise > threshold`, the FEN, latent, TonkAI's move,
the predicted distribution, the actual reply and the result are recorded.
**In V0 this influences no decision.** It is infrastructure for later work.

---

## 9. The opponent curriculum — and why it exists

This is the most important empirical finding in the project so far.

Trained for 5,000 games against the fixed default opponent, TonkAI lost
**99.8%** of its games. The consequence:

| | result |
|---|---|
| value loss | collapsed to **0.008** |
| every contextual piece value | **0.000** |
| Predict | unaffected, learned well (top-1 3% → 34%) |

Nothing was broken. "You lose" was simply the correct answer in every position,
so there was no gradient left for Value or Piece. **A constant result is not a
training signal.**

Weakening the opponent fixed it. 2,000 games, weakened settings:

| | default opponent | weakened opponent |
|---|---|---|
| W/D/L | 0.0 / 0.0 / 100.0 | 0.1 / 9.4 / 90.5 |
| value loss | 0.0000 | **0.2545** |
| piece loss | 0.00001 | **0.00459** |
| queen value mean/std | +0.00 / 0.00 | **−0.13 / 0.13** |
| pawn value mean/std | +0.00 / 0.00 | −0.04 / 0.07 |
| game length | 51 plies | 102 plies |

With a winnable opponent the piece head separates the piece types **on its
own** — queen largest, pawn smallest — with no piece-value table anywhere, and
with a spread that says the value is position-dependent. That is the behaviour
the project exists to produce.

So a **curriculum** now moves TinyOpponent up and down a 10-rung ladder to hold
TonkAI's score in a band it can learn from. It teaches TonkAI nothing about
chess; it only turns the three strength knobs, and TonkAI never sees which rung
it is on.

Two mistakes were made building it, both worth knowing:

1. **Weak ≠ incoherent.** Two extra bottom rungs using `randomness` 1.4 and 2.0
   made the opponent *random* rather than *weak*. A random opponent cannot
   convert a won position and cannot be predicted: games ran to 207 plies, 41%
   ended in stalemate, Predict degraded (14.1% → 13.0% top-1), and the piece
   values separated **less** than against a weak-but-coherent opponent (queen
   std 0.06 vs 0.13). Removed. Weaken with depth, nodes and quiescence instead.
2. **A draw-heavy regime stalls the ladder.** Draws count half, so the score
   parks near 0.4 and a `promote_at` of 0.55 is unreachable. The rung sat at 0
   for 3,000 games at exactly 0.41. A stagnation rule now promotes a healthy
   but static score anyway.

---

## 10. Performance — measured, not guessed

Current single-process throughput: **7.3 games/s, 18,300 positions/s**, GPU at
~15% on an RTX 5060 Ti with 6 physical cores.

There is no single bottleneck. Two ceilings, roughly equal, that **add** because
training and self-play are serial in one process:

| side | ceiling | why |
|---|---|---|
| learner | ~20 games/s | `train_step` = 12.5 ms, at 4 updates/game |
| self-play | ~14–18 games/s | measured with training disabled |
| combined | **7.3** | consistent: 1/7.3 ≈ 1/20 + 1/14 |

Per-step / per-ply costs actually measured:

```
train_step (batch 512)                12.51 ms
  predict sampling + unpacking         2.26 ms   (expands 304 KB → 9.61 MB)
  extra valid-candidate copy           1.17 ms   (a further ~7.67 MB)
  piece-probe rng.random((512,768))    1.29 ms
inference incl. transfers/readback      376 µs/ply
encoding every legal candidate          302 µs/ply
candidate packing: python loop vs vectorised   4.5–5.2× slower
```

**About 60% of a training step is fixed cost, independent of batch size:**

```
batch  128 →  9.09 ms        batch 1024 → 28.34 ms
batch  256 → 10.13 ms        batch 2048 → 48.79 ms
batch  512 → 15.93 ms        batch 4096 → 73.77 ms
```

So for **identical training data** (2048 samples):

```
batch  512 × 4 updates → 53.09 ms/game → ceiling 18.8 games/s
batch 2048 × 1 update  → 35.84 ms/game → ceiling 27.9 games/s   (+48%)
```

Roughly half of a training step is **CPU data marshalling, not GPU compute** —
which is why the GPU sits at 15%. A forward/backward pass is also not "five
matmuls": the profiler counted **14 forward and 18 backward matrix operations
per update**, so kernel-launch overhead matters.

Two further measured results worth recording:

- **CPU inference beats CUDA for self-play by 22–26%** with a matched frozen
  model. Batches of ~30 rows are dominated by launch and transfer overhead.
- `torch.set_num_threads(1)` was **tested and rejected**: train_step 36.93 vs
  39.05 ms (better) but self-play 14.6 vs 16.0 games/s (worse).

Work in progress: two CPU actors feeding one GPU learner (two actors measured
38–39 games/s without learning), consolidated updates, CPU inference in the
actors, and removing the CPU waste above. `--workers 1` must stay
deterministic under a seed; N > 1 will not be, because async ordering changes
the training sequence.

---

## 11. Layout

```
tonkai/
    config.py         every knob; each is also a CLI flag
    encoding.py       board → 782 features, always from TonkAI's side
    chess_env.py      rules; python-chess is the source of truth
    model.py          shared encoder + Value / Predict / Piece heads
    agent.py          move selection, prediction, and the self-play GameWorker
    replay.py         compact ring buffers (99 bytes per position)
    trainer.py        the three losses and checkpointing
    curriculum.py     the opponent strength ladder
    theory_buffer.py  surprises and contextual sacrifices (collected only)
    metrics.py        counters, timers, the compact log line
    history.py        per-run JSONL history
tiny_opponent/
    engine.hpp/.cpp   the C++ opponent
    bindings.cpp      pybind11 bindings
train.py  evaluate.py  analyze.py  play.py  calibrate_opponent.py
export_web.py         exports weights + golden vectors for the browser
tests/                91 tests
web/                  tonkai.bin, tonkai.json, golden.json, ENCODING.md
```

About 4,500 lines total.

---

## 12. Current state

- **91 Python tests pass.** Perft exact to depth 5–6 on all six standard
  positions.
- A browser build is **live** at `https://anw-games.web.app/tonkai` — the
  network runs fully client-side (527 KB of float32 weights, no server). A JS
  port is verified against the Python reference to `1e-4` on 9 positions, a
  full 42-move ranking and a full 20-move predict distribution.
- The deployed checkpoint (5,020 games) has a **saturated value head**:
  `P(loss) ≈ 1.0` everywhere, and the spread between its best and worst move is
  about 0.02. **Its move choice is close to noise** and any human beats it.
  Its **predict head is genuinely good** (34% top-1).
- Training with the curriculum is paused while the throughput work lands.

## 13. Known limits (deliberate, V0)

- every position in a game gets the final result as its label — noisy
- predict candidates capped at 48 per stored sample; wider positions subsampled
- `temperature = 1.0` is near-uniform, because utility lives in `[−1, 1]`;
  `0.1` is roughly where move choice starts to bite
- `surprise_threshold` is an absolute bar, so an untrained predictor (near-
  uniform over ~35 replies, surprise ≈ 3.5) floods the theory buffer; a
  threshold relative to current entropy would be the better fix
- no absolute Elo anchor for TinyOpponent
- Theory and Innovation are **infrastructure only** — they influence nothing.
  Innovation, when built, must mean *measured improvement in Predict and
  Survival caused by a new Theory*, never "did something different".
