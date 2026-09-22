# TonkAI performance implementation

Four steps implemented in order, separate local commits, no push.

## Conditions and limits

RTX 5060 Ti; Python 3.13.14; PyTorch 2.11.0+cu128; NumPy 2.5.2.
12 logical CPUs, existing six PyTorch threads unchanged. No thread-count tuning.
Benchmarks load weights only from `checkpoints/deployed_5020games.pt`, never
write checkpoints, and export into `perf_results/export/`, never `web/`.

`perf_verify.py`: seed 42, temperature 0.1, default opponent, populated replay,
three warm-up learner iterations, 30 timed learner iterations, 60 games per
worker count. Every timed game trains. Worker counts run sequentially on a
continuing learner; their learned weights and game lengths differ. Positions/s
is reported alongside games/s. Parallel windows include Windows process startup,
imports and shutdown. Short runs on a shared machine vary; both step-4 runs are
retained. The initial baseline with insufficient replay warm-up was discarded
and rerun with a ready replay before implementation.

## Step 1: consolidated updates (6333346)

Default: 2,048 value samples per game in one update, with proportional predict
batches and the existing minimum. Multiple updates divide the budget including
its remainder. `--samples_per_game 0 --batch_size 512 --updates_per_game 4`
restores the legacy schedule; old checkpoints retain it. Direct train_step()
still honors batch_size. Larger Adam batches deliberately change the optimization
trajectory, while preserving loss definitions, labels and the sample budget.

Learner: **63.26 -> 38.54 ms/game**, ceiling **15.81 -> 25.95 games/s**, below
~28 predicted. Combined: **7.93 -> 11.01 games/s**, **11,952 -> 15,457 positions/s**.
92 tests passed; encoding, golden export and frozen seeded game matched.

## Step 2: CPU actors (fe28e33)

`--workers N`, default 1; spawn-based CPU actors, bounded record queue, locked
shared weights, configurable refresh, curriculum propagation, error propagation,
clean shutdown and checkpointing. One worker retains in-process execution and
its shared seeded RNG. **N > 1 is nondeterministic due to asynchronous ordering.**
Actors score all legal candidates in one batch. Their overlapping CPU work is
reported separately from learner wall time.

Step 1 one-worker **11.01** -> step 2 **10.83 / 12.92 / 10.65 games/s** for
workers 1/2/4. Two actors helped, four did not. Publication cost **3.78 ms total**
in the 60-game two-worker run. The prior actor-only 38-39 games/s prediction
excluded training and is not directly comparable. Workers 2/4 did not exist
before this step, so no invented before numbers are reported for them.

95 tests passed, including real spawned records, refresh, actor death, shutdown
and SIGINT through the CLI saving a checkpoint. The signal test caught Windows
InterruptedError during queue polling, now handled. Encoding/golden/game matched.

## Step 3: CPU self-play, GPU learner (fc37dfb)

Single-worker GPU learning now uses a CPU inference replica refreshed immediately
after updates and checkpoint loads without RNG draws. `--inference_device learner`
restores the original shared-device path; old checkpoints retain it unless
overridden. Both paths are seeded and in-process. CPU/GPU arithmetic can differ,
so changing devices does not promise identical optimization trajectories.
Actors already ran on CPU in step 2: their inference gain is not counted twice.

Matched frozen models, two runs of 40 games/device, identical 2,084 plies:
**CUDA 16.02-16.29 -> CPU 18.32-19.20 games/s**;
**24,259-24,668 -> 27,756-29,090 positions/s**. Average gain **16.2%**, below the
predicted 22-26%. Combined single-worker throughput did not improve:
**10.83 -> 10.38 games/s**. CPU replica copying and runtime variation matter.
The final step-4 repeat measured **58.74 ms/60 updates = 0.98 ms/update** for
replica refresh, already included in training time, not counted twice.

97 tests passed, including repeatable games with learning and replica sync after
update/load. Frozen seeded game and encoding/golden remained identical.

## Step 4: CPU waste

Predict sampling gathers valid packed candidates before decoding, retaining
candidate order, truth indices and RNG draws. Padding is never decoded, and the
additional dense float selection copy is eliminated (about 7.67 MB in the
original investigation; actual size depends on candidate count). Dense sample()
remains compatible. Piece probes reuse plane, float64 random and float32 score
arrays; float64 RNG generation then float32 rounding is unchanged. Candidate
packing is vectorized with float64 clock rounding to preserve the packed bytes.

| Component | Before | After | Assessment |
|---|---:|---:|---|
| train_step, 2,048 samples | 37.77 ms | 28.49 ms | 24.6% lower |
| predict sampling, 256 rows | 9.46 ms | 8.20 ms | 1.26 ms saved; original 1.39 ms estimate used 64 rows |
| Packing 20 candidates | 39.44 us | 10.83 us | 3.64x, below predicted 4.54x |
| Probe RNG/cast/multiply/argmax, 512 | 2.03 ms | 0.86 ms | 1.17 ms saved, above predicted 0.75 ms |
| Probe RNG/cast/multiply/argmax, 2,048 | 8.01 ms | 3.69 ms | 4.32 ms saved |

Microbenchmarks use medians of three blocks; raw blocks are retained. Probe
microbenchmarks use an all-ones mask, exclude network work, and preserve random
selection semantics. Component savings should not simply be added together.

Combined step 3 -> step 4: **10.38 -> 13.00 games/s** (1 worker),
**12.54 -> 15.13** (2), **10.13 -> 13.09** (4). Final repeat:
**12.22 / 13.99 / 13.93 games/s**; retained rather than selecting the best run.
Learner-only: **27.56 ms/game**, then **25.34** on the repeat.

100 tests passed. New checks cover valid replay bytes/RNG, packing with every
clock value 0-100 in both orientations, and probe RNG across changing batch
sizes. perf_equivalence.py loads the actual original Trainer from 1804bfc:
five games and 20 legacy updates produce **bit-identical packed games AND final
parameters on CPU and CUDA**. This separates performance equivalence from the
intentional batch-schedule change.

## Longer runs and refresh cost

Separate 180-game windows after 20 warm-up games and process startup, same
initial weights and 70-game replay corpus. These exclude startup/shutdown and
must not be compared directly with the short baseline as a speedup ratio.
Asynchronous games and learned weights differ. Refresh interval alone does not
explain all throughput differences.

| Workers | Refresh interval | Games/s | Positions/s | Publish time / 180 games |
|---:|---:|---:|---:|---:|
| 1 | immediate replica | 11.94 | 17,559 | included in training |
| 2 | 1 | 25.52 | 38,955 | 177.91 ms |
| 2 | 20 | 30.13 | 46,413 | 8.68 ms |
| 2 | 100 | 25.98 | 39,777 | 2.04 ms |
| 4 | 20 | 24.18 | 35,598 | 9.41 ms |

Two workers at interval 20 were best in this run. Four workers had higher
training cost. Publishing every game cost about 1 ms/publication, 2.5% of wall
time; interval 20 cost 0.15%. Interval 100 saved copying but did not improve
throughput. Keep interval 20 as the default.

## Verification and reproduction

Tests per step: **92 / 95 / 97 / 100 passed**. The initial plain pytest invocation
hit permissions in its pre-existing Windows Temp directory; checks then used
fresh writable workspace temp paths, without disabling tests.

Encoding corpus: golden positions in both orientations and all legal children.
Before/after every step SHA256:
`2b5d2e061e5d446ffa6882821e91c49b44acdef433f9ebae3f27d73683447796`.
Frozen seeded packed-game SHA256 before/after every step:
`9b54d942fbb7150c7a74793f826160c914e5dd024fc648fa1e16c85e09e0a6a3`.

Complete parsed exported golden JSON exactly matched web/golden.json, including
model outputs. The deployed checkpoint was explicitly selected because latest.pt
need not be the website model. perf_results/export is the writable Windows
equivalent of the requested temporary checkexp path. No web files or tensor
names/shapes changed. No piece values, search, centipawns or engine heuristics
were added to TonkAI. One forward pass over all legal candidates is retained.

The deployed checkpoint was only read. Final file SHA256:
`c647c9b6f2ede30eca333245bb8687b9b93d79f3ab3c92f4495b821293de4911`.
Existing investigation files and unrelated SYSTEM.md were preserved. No push.

```powershell
python -m pytest --basetemp=perf_results/pytest-new -p no:cacheprovider
python perf_verify.py another_run
python perf_equivalence.py
python perf_micro.py another_micro
python perf_frozen.py
python perf_details.py
python perf_steady.py
python export_web.py --checkpoint checkpoints/deployed_5020games.pt --out perf_results/export
```

Consolidated training, also suitable as explicit overrides when resuming:

```powershell
python train.py --workers 2 --samples_per_game 2048 --updates_per_game 1 --weight_refresh_games 20
```

## Throughput and wall split after every step

Each cell is games/s / positions/s. A dash means parallel execution did not yet exist.

| Stage | 1 worker | 2 workers | 4 workers | Learner ms/game |
|---|---:|---:|---:|---:|
| Baseline | 7.93 / 11,952 | - | - | 63.26 |
| 1: consolidate | 11.01 / 15,457 | - | - | 38.54 |
| 2: CPU actors | 10.83 / 15,205 | 12.92 / 19,170 | 10.65 / 19,041 | 43.51 |
| 3: CPU inference | 10.38 / 14,578 | 12.54 / 18,762 | 10.13 / 15,315 | 40.96 |
| 4: CPU waste | 13.00 / 18,254 | 15.13 / 21,664 | 13.09 / 20,875 | 27.56 |
| 4: final repeat | 12.22 / 17,159 | 13.99 / 21,216 | 13.93 / 22,461 | 25.34 |

Actual metrics.timing_breakdown() per timed window follows. Percentages are of measured scopes; ingestion and bookkeeping account for unmeasured wall time. Actor work overlaps and is stored separately in JSON. Early cumulative formatting in baseline/step1 was corrected from stored time deltas; elapsed measurements were not changed. Worker-4 actor times were similarly converted to per-window deltas.

### Baseline, workers 1, wall 7.566s

```text
  training              3.5s   49.1%
  env                   1.4s   19.9%
  inference             1.4s   19.1%
  opponent              0.8s   11.8%
```

### 1: consolidate, workers 1, wall 5.451s

```text
  training              2.1s   41.8%
  env                   1.2s   22.9%
  inference             1.1s   21.4%
  opponent              0.7s   13.9%
```

### 2: CPU actors, workers 1, wall 5.542s

```text
  training              2.1s   41.2%
  env                   1.2s   23.4%
  inference             1.1s   21.3%
  opponent              0.7s   14.1%
```

### 2: CPU actors, workers 2, wall 4.645s

```text
  training              2.7s   57.6%
  actor_wait            1.5s   32.9%
  actor_stop            0.4s    9.2%
  actor_start           0.0s    0.2%
  weight_publish        0.0s    0.1%
  env                   0.0s    0.0%
  inference             0.0s    0.0%
  opponent              0.0s    0.0%
```

### 2: CPU actors, workers 4, wall 5.631s

```text
  training              3.5s   61.9%
  actor_wait            1.5s   26.8%
  actor_stop            0.6s   11.0%
  actor_start           0.0s    0.3%
  weight_publish        0.0s    0.1%
  env                   0.0s    0.0%
  inference             0.0s    0.0%
  opponent              0.0s    0.0%
```

### 3: CPU inference, workers 1, wall 5.780s

```text
  training              2.5s   47.0%
  env                   1.5s   27.4%
  opponent              0.9s   16.2%
  inference             0.5s    9.4%
```

### 3: CPU inference, workers 2, wall 4.783s

```text
  training              2.7s   57.4%
  actor_wait            1.6s   32.9%
  actor_stop            0.4s    9.4%
  actor_start           0.0s    0.3%
  weight_publish        0.0s    0.1%
  env                   0.0s    0.0%
  inference             0.0s    0.0%
  opponent              0.0s    0.0%
```

### 3: CPU inference, workers 4, wall 5.921s

```text
  training              3.6s   61.0%
  actor_wait            1.7s   29.1%
  actor_stop            0.6s    9.4%
  actor_start           0.0s    0.5%
  weight_publish        0.0s    0.1%
  env                   0.0s    0.0%
  inference             0.0s    0.0%
  opponent              0.0s    0.0%
```

### 4: CPU waste, workers 1, wall 4.616s

```text
  training              1.7s   39.0%
  env                   1.4s   31.4%
  opponent              0.9s   19.5%
  inference             0.4s   10.2%
```

### 4: CPU waste, workers 2, wall 3.966s

```text
  training              2.1s   52.4%
  actor_wait            1.5s   37.2%
  actor_stop            0.4s   10.0%
  actor_start           0.0s    0.3%
  weight_publish        0.0s    0.1%
  env                   0.0s    0.0%
  inference             0.0s    0.0%
  opponent              0.0s    0.0%
```

### 4: CPU waste, workers 4, wall 4.585s

```text
  training              2.3s   50.2%
  actor_wait            1.7s   37.7%
  actor_stop            0.5s   11.6%
  actor_start           0.0s    0.5%
  weight_publish        0.0s    0.1%
  env                   0.0s    0.0%
  inference             0.0s    0.0%
  opponent              0.0s    0.0%
```

### 4: final repeat, workers 1, wall 4.911s

```text
  training              1.9s   40.0%
  env                   1.4s   30.8%
  opponent              0.9s   18.7%
  inference             0.5s   10.4%
```

### 4: final repeat, workers 2, wall 4.288s

```text
  training              2.0s   46.3%
  actor_wait            1.8s   43.0%
  actor_stop            0.4s   10.2%
  actor_start           0.0s    0.3%
  weight_publish        0.0s    0.1%
  env                   0.0s    0.0%
  inference             0.0s    0.0%
  opponent              0.0s    0.0%
```

### 4: final repeat, workers 4, wall 4.307s

```text
  training              2.1s   49.1%
  actor_wait            1.6s   38.5%
  actor_stop            0.5s   11.8%
  actor_start           0.0s    0.5%
  weight_publish        0.0s    0.1%
  env                   0.0s    0.0%
  inference             0.0s    0.0%
  opponent              0.0s    0.0%
```

### Steady workers 1, wall 15.077s

```text
  training              5.5s   38.4%
  env                   4.6s   32.0%
  opponent              2.7s   19.1%
  inference             1.5s   10.5%
```

### Steady workers 2, interval 20, wall 5.975s

```text
  training              5.6s   94.3%
  actor_wait            0.3s    5.5%
  weight_publish        0.0s    0.1%
  actor_start           0.0s    0.0%
```

### Steady workers 4, wall 7.445s

```text
  training              7.3s   99.0%
  actor_wait            0.1s    0.8%
  weight_publish        0.0s    0.1%
  actor_start           0.0s    0.0%
```

