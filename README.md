# TonkAI

**Próximo adversário — V3 opt-in:** [memória associativa, contrafactuais e análise adaptativa](docs/cognitive_next_opponent.md).
`train_cognitive.py` prepara os nove módulos selecionados, com checkpoints próprios;
não altera nem substitui o treino V2 já em execução.

**Novo treino V2:** [sobrevivência, memória temporal e população de adversários](docs/survival_training.md).
`train_survival.py` usa sequências, previsão futura, curiosidade e incerteza, sem
bónus de vitória. Importa pesos compatíveis do legado e guarda checkpoints
separados. [Stockfish 11 nominal 1600](docs/opponent_1600.md) é o adversário externo.
O modelo e o caminho de treino descritos abaixo continuam disponíveis como legado.

A small chess network that is **not** trying to imitate an engine.

It has no piece values, no centipawns, no opening book, no tablebases, no MCTS
and no minimax. It learns from four things only:

1. **Survival** - the real result of the game (win / draw / loss)
2. **Prediction** - what the opponent actually played next
3. **Risk vs return** - the full W/D/L distribution is kept, never collapsed early
4. **Contextual piece values** - one learned scalar per piece on the board,
   trained from counterfactual contribution, not from a table

TonkAI is allowed to conclude *"this rook of mine looks more valuable than that
knight, but trading it gives me a position I like better"*. That is the point.

## Layout

```
tonkai/
    config.py         every knob, and --flags for all of them
    encoding.py       board -> 782 binary features, always from TonkAI's side
    chess_env.py      the rules (python-chess is the source of truth)
    model.py          shared encoder + Value / Predict / Piece heads
    agent.py          move selection, prediction, and the self-play GameWorker
    replay.py         compact ring buffers (positions stored as 99 bytes)
    trainer.py        the three losses and the checkpoints
    theory_buffer.py  surprises and contextual sacrifices, collected only
    metrics.py        counters, timers, the one compact log line
tiny_opponent/
    engine.hpp/.cpp   the C++ opponent
    bindings.cpp      pybind11 bindings
    CMakeLists.txt    (setup.py does the same job and is what Windows uses)
train.py  evaluate.py  analyze.py  play.py  calibrate_opponent.py
tests/
```

## Setup

```bash
pip install -r requirements.txt
python setup.py build_ext --inplace
python -m pytest
```

The extension lands in the project root as `tiny_opponent*.pyd` / `.so`, so
`import tiny_opponent` works from there with no install step. A C++17 compiler
is required (MSVC, gcc or clang).

## Running

```bash
python train.py --games 5000
python train.py --resume checkpoints/latest.pt --games 20000
python evaluate.py --games 200
python analyze.py --fen "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
python calibrate_opponent.py --games 80
python play.py                       # play it yourself
```

`play.py` is the interesting one to run by hand: before each of your moves it
shows what the predict head expects you to play, and afterwards how surprised
it was. Commands inside the game: `eval`, `pieces`, `predict`, `undo`, `fen`,
`help`, `quit`. `hint` shows the most expected human reply, not strength advice.
`moves` prints the move list; `pgn` prints PGN and `pgn filename.pgn` exports it.
Use `--pgn filename.pgn` to save on exit (including resignation or an unfinished
game). Quiet mode still scores predictions, and undo removes the corresponding
prediction observation from the final hit count and mean surprise.

Training and evaluation against TinyOpponent, and local human games in `play.py`,
automatically keep **only TonkAI's longest completed win** in
`history/longest_external_win.pgn`. The full legal move sequence, colours,
opponent, result and termination are stored. Length counts individual moves by
either side (`PlyCount`); shorter wins and ties leave the existing file intact,
including across restarts. Draws, losses and unfinished games are not saved in
this record. A human resignation counts as a completed TonkAI win. This is
recording only and has no effect on training or move selection.

Use `--win_pgn_path path/to/record.pgn` to choose its location, or
`--win_pgn_path=` to disable it. The optional `play.py --pgn` export remains
independent and can still save any requested game. Old runs that recorded only
aggregate metrics cannot supply a complete historical game afterwards.

## Run histories and plots

Each training invocation now writes a separate `history/<UTC-time>-<id>.jsonl`.
The first row records config, resume path and initial checkpoint game count;
subsequent rows snapshot metrics at `log_every` and once at orderly exit.
Counters and timers are cumulative as reported by Metrics; prediction/loss
means and standard deviations use the existing rolling windows and include
their sample counts. `elapsed_seconds` is local to this process. There is no
extra per-position logging or plotting dependency.

```bash
python train.py --history-dir history/experiment-a --games 2000
python plot_history.py "history/*.jsonl" --metric top1.mean --out top1.svg
python plot_history.py "history/*.jsonl" --metric surprise.mean --out surprise.svg
```

Pass `--no-history` to disable logging. SVGs are standalone and compare all
matched runs. Other metric paths include `value_loss.mean`, `wins`, `entropy.mean`
and `times.inference`. The final snapshot replaces an earlier sample at the
same game count in a plot. Forced termination can leave no final row; earlier
completed rows remain on disk.

## Optional rating anchor

```bash
python calibrate_opponent.py --anchor /path/to/rated-uci-reference --anchor-elo 1300 --anchor-time 0.05 --games 100 --settings 1
```

This mode only measures TinyOpponent against a supplied UCI reference, alternating
colours. It does not run the relative ladder. A missing executable is skipped;
no reference is downloaded or required. `--anchor-options` accepts a JSON object
of UCI options if the rated configuration needs them. The supplied rating must
be supported for that exact configuration and time budget: the output is a
conditional estimate with an approximate sampling interval, not an established
absolute rating. No rated reference was available during implementation, so no
new measured Elo is claimed. Calibration never passes reference moves, scores,
or weights to TonkAI training.

Every config field is also a flag, so experiments need no code edits:

```bash
python train.py --games 2000 --temperature 0.1 --piece_value_weight 0.0
python train.py --games 2000 --device cpu --updates_per_game 1
```

## How a turn works

**TonkAI's turn.** Generate the legal moves, apply each one, encode all the
resulting positions into a single `(N, 782)` array, and push that through the
encoder in **one** forward pass. The value head returns `P(win), P(draw),
P(loss)` per move; the move is picked by

```
U = P(win) + draw_weight * P(draw) - P(loss)
```

sampled with `temperature` (argmax at `temperature = 0`).

**The opponent's turn.** Same batching trick: every legal reply is applied and
encoded, and the predict head puts one logit on each. Softmax over exactly the
legal replies gives a distribution. TinyOpponent then plays, and the predict
head is trained to raise the probability of the move that really happened.
`surprise = -log P(actual move)`.

Both heads share one encoder, and the piece head reads the same latent.

## Contextual piece values

The piece head gets `latent + piece type + square + self/opponent` and returns
one scalar per piece. Its target is counterfactual, computed by TonkAI against
itself:

```
target = U(position) - U(position with that piece erased from the input)
```

One or two pieces per position per batch, kings excluded, weight 0.1. It is
an *estimate of contribution*, not a chess value, and utility is never a sum of
piece values - which is exactly what leaves room for a sacrifice.

When TonkAI captures with a piece it values **above** the piece it takes, the
event is recorded as a `CONTEXTUAL_SACRIFICE` (`--dump-sacrifices out.json`).

## TinyOpponent

A deliberately small C++ opponent. Legal move generation, a tiny alpha-beta
with root iterative deepening, material + piece-square evaluation, a short
capture-only quiescence, and controlled randomness at the root under a fixed
seed. **None of this is ever visible to TonkAI** - it only ever sees board
positions and results.

Strength is dialled with three knobs, measured over 80-game matches:

| knob | effect |
|---|---|
| `opponent_depth` | depth 2 is +470 Elo over depth 1, depth 3 is +512 |
| `opponent_nodes` | 100 / 300 / 1000 / 20000 spans roughly 700 Elo |
| `opponent_randomness` | 0.15 / 0.35 / 0.7 costs 44 / 132 / 215 Elo |

Root iterative deepening matters here: a depth's scores are only used once
*every* root move has been searched at that depth. Without it a tight node cap
mixes deep and shallow scores at the root and the engine gets *worse* as depth
goes up.

Move generation is verified by perft against the six standard positions to
depth 5-6 (479M nodes, exact), and cross-checked against python-chess.

## Reading the log

```
Games: 500   W/D/L: 0.4 / 3.2 / 96.4
Games/s: 9.1   Positions/s: 12050   Batches: 2000
Loss  value 0.5623  predict 3.1004  piece 0.00042
Predict  top1 12.4%  top3 28.1%  p(true) 0.089  surprise 3.01  entropy 3.24
Avg len 61 plies   legal 21.4   enemy 33.8
Piece values  Pa +0.01/0.03  Kn +0.02/0.04  ...
Sacrifices: 41   Theory events: 900
```

`train.py` also prints a timing breakdown and names the bottleneck at the end.
Measure before optimising anything.

## Known V0 limits

These are deliberate. They are the first things to revisit.

* **At the default opponent, Value and Piece get no gradient.** Measured over
  5000 games: TonkAI loses 99.8%, so "you lose" is simply the correct answer
  everywhere, value loss collapses to 0.008 and every contextual piece value
  goes to 0.000. The mechanism is fine - the *signal* is constant. Predict is
  unaffected and learns well (top1 3% -> 34%).

  Weakening the opponent fixes it. 2000 games at
  `--temperature 0.1 --opponent_quiescence false --opponent_depth 1
  --opponent_nodes 100 --opponent_randomness 0.9`:

  | | default opponent | weakened opponent |
  |---|---|---|
  | W/D/L | 0.0 / 0.0 / 100.0 | 0.1 / 9.4 / 90.5 |
  | value loss | 0.0000 | 0.2545 |
  | piece loss | 0.00001 | 0.00459 |
  | Queen value mean/std | +0.00 / 0.00 | **-0.13 / 0.13** |
  | Pawn value mean/std | +0.00 / 0.00 | -0.04 / 0.07 |
  | game length | 51 plies | 102 plies |

  With a winnable opponent the piece head separates the piece types on its own
  - queen largest, pawn smallest - with no piece-value table anywhere, and a
  spread that says the value is position-dependent. That is the behaviour the
  project is after, and it needs a result signal that is not constant.

* **`temperature = 1.0` is close to uniform.** Utility lives in `[-1, 1]`, so
  `softmax(U / 1.0)` barely separates the moves and TonkAI plays almost
  randomly during self-play. `--temperature 0.1` is roughly where move choice
  starts to bite. The default is left at 1.0 because that is what the spec
  asks for. On its own it is not enough - see the table above, experiment A.
* **`surprise_threshold = 3.0` is an absolute bar.** An untrained predictor is
  near-uniform over ~35 replies, so `surprise ≈ log(35) ≈ 3.5` and nearly every
  move lands in the theory buffer. Once Predict trains this settles down; a
  threshold relative to the current entropy would be the better long-term fix.
* **Every position in a game gets the final result as its label.** The spec's
  V0 choice, and it is noisy by construction.
* **Predict candidates are capped at 48** per stored sample; wider positions
  are subsampled with the real move always kept. `truncated` in the log counts
  how often that happened.
* **One worker.** `GameWorker` is self-contained so several can later feed one
  replay buffer, but V0 runs a single one.
* **The opponent has no measured absolute Elo yet.** Default calibration gives
  relative numbers. Optional `--anchor` can measure against a supplied rated
  reference; an absolute ~1300 still needs evidence under matched conditions.

## What is deliberately absent

No MCTS. No minimax inside TonkAI. No opening book. No tablebases. No reward
for captures, checks, centre control, development, material, or novelty. No
Stockfish teacher. No centipawns. `tests/test_model.py` fails the build if a
classical piece-value table appears anywhere in `tonkai/`.

Training throughput: `--samples_per_game 2048 --updates_per_game 1` is the
new default. Samples are split across the requested updates (including the
remainder); `--samples_per_game 0 --batch_size 512 --updates_per_game 4`
restores the old schedule. Old checkpoints retain that legacy schedule unless
overridden. Larger batches change Adam's update trajectory, not the labels or
loss definitions. A step is capped at the current replay size as before.

`--workers N` (alias `--num_workers N`) defaults to 1: the original in-process
worker and shared seeded RNG are retained. N > 1 uses CPU actors and one learner
on `--device`; asynchronous arrival order makes this mode **not deterministic**.
Each actor scores all legal candidates in one network batch. A bounded queue
limits outstanding records. `--weight_refresh_games 20` publishes a shared
snapshot every 20 ingested games; actors load it between games. In-flight games
can therefore use older weights. Curriculum changes also apply between games.
Ctrl+C finishes the current learner iteration, stops actors, and saves latest.pt;
a second Ctrl+C forces exit. Worker failures also run cleanup and save a checkpoint.
Parent timing percentages measure waiting, learning and publication; actor CPU
work overlaps and is recorded separately in `metrics.actor_times`.

Single-process self-play now defaults to `--inference_device cpu`, while
`--device auto` still selects the GPU for training. Its CPU replica is refreshed
immediately after each update and checkpoint load, without consuming RNG draws.
`--inference_device learner` restores the original device path for exact legacy
comparisons; old checkpoints retain it unless explicitly overridden. Both paths
remain in-process and repeatable under a seed. CPU/GPU floating-point results
can differ slightly, so device changes are not a promise of identical training
trajectories. `inference_refresh_seconds` measures copy cost already included in
training time. Parallel actors were already CPU-only in step 2.

Performance verification scripts and raw measurements are in `perf_*.py` and
`perf_results/`. See `PERFORMANCE.md` for the measurement conditions, comparisons,
wall-time breakdowns, and regressions. Run tests with a fresh writable temp path:
`python -m pytest --basetemp=perf_results/pytest-local -p no:cacheprovider`.
The vectorized packing preserves all 99 bytes, including float64 clock rounding;
valid-only predict sampling keeps candidate order and RNG draws. Piece scratch
reuse retains float64 random generation followed by float32 rounding.
