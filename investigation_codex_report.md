Investigation of TonkAI self-play and training performance — 22 September 2026

The principal bottleneck is a serial pipeline containing substantial CPU preparation and small, synchronizing GPU calls. The 131k parameter count does not describe the work of a training step. Predict training processes thousands of candidate boards, piece training performs another counterfactual encoder pass, and the learner repeatedly expands and copies packed data. The GPU cannot overlap this work with game generation in the current main loop.

No library source, train.py, opponent source, or checkpoint was modified. No commit was made. HEAD remained `1804bfcf793cc693830ecc338096bbf817305481`. All new files are `investigation_codex_*` benchmark/report artifacts. The existing untracked `profile_hotspots.py` was left untouched.

Measurements used Python 3.13.14, PyTorch 2.11.0+cu128, NumPy 2.5.2, RTX 5060 Ti 16 GB, and an AMD Ryzen 5 7500F: **6 physical cores / 12 logical processors**, not 12 physical cores. Torch initially used 6 intra-op and 6 inter-op threads. OMP_NUM_THREADS, MKL_NUM_THREADS and OPENBLAS_NUM_THREADS were unset. The benchmark used the checkpoint's 2,500-game model weights, default depth-2/1,000-node opponent, temperature 0.1, and real collected replay; optimizer state was fresh. Short in-memory training measurements never saved models. Microbenchmarks generally report the median of five blocks; process scaling excludes startup and warm-up. Another investigation may have been running, so small differences are not significant.

**1. The reported 7.3 games/s is plausible, but the quoted timing split is not universal.**

The existing `history/20260922T142140Z-4cfa262e.jsonl` supplies a much longer observation than a microbenchmark. Its 4,000-game snapshot is 7.387 games/s and 18,314 candidate positions/s, closely matching the question. It reports 413,876 plies, or 103.47 plies/game.

| Component | Historical seconds | % of actual elapsed | % of timed subtotal |
|---|---:|---:|---:|
| Training | 195.134 | 36.0% | 39.9% |
| Inference | 145.648 | 26.9% | 29.8% |
| Environment | 139.864 | 25.8% | 28.6% |
| Opponent | 8.386 | 1.5% | 1.7% |
| Outside all timers | 52.459 | 9.7% | excluded |
| Actual elapsed | 541.491 | 100% | — |

`Metrics.timing_breakdown()` divides by the sum of recorded timers, not elapsed wall time (`tonkai/metrics.py:132`). Move selection, prediction statistics, per-row replay packing, sacrifice checks, ingestion, and other overhead lie outside those scopes. Environment includes candidate encoding, not just move generation. The history's training average is **12.269 ms/update**, across 15,904 updates.

A fresh short combined run measured **8.219 games/s**, with **80 actual updates / 20 games**, 1,228 plies, and 2.433 seconds elapsed. Training took 1.111 s (45.6% of wall), environment 0.451 s (18.5%), inference 0.439 s (18.1%), opponent 0.289 s (11.9%), and the remaining 5.9% was unscoped. Normalizing only timed work gives approximately **48.5/19.7/19.2/12.6**, close to the supplied 50/20/17.5/12 split. Different game lengths and opponent settings materially change the split.

The historical opponent averages **40.3 microseconds/opponent move**; this investigation's default opponent averages about **489 microseconds** in game. A fixed-position check measured **506–513 microseconds** at default curriculum rung 6, versus **46.1 microseconds** at rung 0. This supports a different effective opponent configuration as an explanation. It does not establish which rung the historical run used: curriculum mutates the opponent without updating the saved Config, and that history does not record the rung. Do not attribute the entire discrepancy to measurement noise.

The provided `profile_hotspots.py` also has a readiness trap. Its first 12 games produced only 582 value samples; `Trainer.train()` silently returns below `min_replay=2000`. Its printed “self-play + 4 updates” result was 18.1 games/s versus 14.1 for self-play, but it never checks how many updates ran. Those labels alone are not evidence of training throughput. The dedicated combined measurement above explicitly verified 80 updates.

**2. Per-ply breakdown: candidate encoding and inference dominate game generation.**

An instrumented 24-game run produced 1,386 plies and 40,161 candidates in 1.570 s: **15.29 games/s, 883 plies/s, 25,588 candidate boards/s**. Each played move generated an average of **29.0 candidate encodings**. Subtimers are nested, so the table below separates their exclusive contributions.

| Work | Microseconds per played ply | % of self-play wall |
|---|---:|---:|
| GPU inference, transfer and CPU result | 375.6 | 33.2% |
| Push/encode/pop every legal candidate | 301.8 | 26.6% |
| TinyOpponent, amortized across both sides | 245.5 | 21.7% |
| Other environment work | 67.3 | 5.9% |
| Replay row packing | 47.4 | 4.2% |
| Sacrifice checks | 16.3 | 1.4% |
| Terminal checks | 14.2 | 1.3% |
| Ingestion | 3.9 | 0.3% |
| Remaining unscoped work | 60.5 | 5.3% |
| Total | **1,132.4** | **100%** |

TinyOpponent's entry corresponds to 0.340 s over the whole experiment, approximately 0.49 ms per opponent move. Its standalone moves/s cannot be equated to games/s: a game contains dozens of opponent moves, both sides evaluate all candidates, and training adds four updates/game. C++ perft measures the opponent's generator; the agent path uses python-chess push/pop, rules and encoding (`tonkai/agent.py:232`, `:273`; `tonkai/encoding.py:88`).

On a fixed opening position with 38 legal moves, the original benchmark measured legal move listing 29.7 us, FEN generation 24.6 us, one encoding 5.3 us, all-candidate encoding 332.4 us, and opponent search 495.0 us. Optimizing only C++ move generation cannot remove the Python candidate loop.

**3. Per-update breakdown: most of the time is not a small forward pass.**

Uninstrumented complete `train_step()` measured **12.506 ms median**, with block means 11.566–13.226 ms. A separate experiment synchronized phase boundaries and subtracted nested timers; its total was 11.626 ms. The difference is normal run-to-run variation and instrumentation changes, not negative overhead. Use this table for attribution, not an exact decomposition of the 12.506 ms observation.

| Synchronized exclusive phase | ms/update | Share of phase total |
|---|---:|---:|
| Piece loss, including RNG, transfers, masked forward and metrics | 3.161 | 27.2% |
| Predict work after sampling, including valid-row copy, transfers and forward | 2.788 | 24.0% |
| Predict replay sampling/unpacking | 1.953 | 16.8% |
| Backward | 1.744 | 15.0% |
| Value forward/loss/transfers after sampling | 0.614 | 5.3% |
| AdamW | 0.380 | 3.3% |
| Value replay sampling/unpacking | 0.359 | 3.1% |
| Gradient clipping | 0.327 | 2.8% |
| Remaining step work | 0.300 | 2.6% |

Independent microbenchmarks clarify the expensive preparations:

| Operation | Median ms |
|---|---:|
| PredictBuffer.sample(64) | 2.265 |
| Copy valid rows from expanded predict batch | 1.172 |
| Generate float64 piece scores and cast to float32, before multiplication/argmax | 1.286 |
| Transfer 2,451 valid predict rows to CUDA | 0.652 |
| ReplayBuffer.sample(512) | 0.362 |
| Transfer value inputs to CUDA | 0.171 |

The independent sampling plus masking figures alone are roughly **27.5% of a 12.506 ms step**, not 40% for predict sampling alone. Predict's whole forward/loss path occupies approximately 40.8% of the synchronized phase total.

Predict sampling gathers **304,128 packed bytes**, creates **2,408,448 bytes** of unpacked bits, and writes **9,609,216 bytes** of float32 input. A measured mask kept 2,451/3,072 rows; the next advanced-index operation copied another **7,666,728 bytes**, then transferred that payload to the GPU. Thus 9.6 MB is only one output array, not total memory traffic. Counting main-pass source reads and destination writes, including the final transfer's host read, gives roughly **38 MB of host-side accesses/update**, plus 7.67 MB written on the device. This is array-level accounting, not measured DRAM traffic; caches and transfer staging change actual traffic. Value inputs add 1.602 MB of expanded floats per step. Avoid confusing host traffic with PCIe traffic.

The piece probe generates **393,216 random float64 values = 3.146 MB**, casts to a 1.573 MB float32 array, multiplies into another array, and scans for argmax, once per configured probe (`tonkai/trainer.py:108`). These are overwhelmingly zero-valued candidate locations after masking, but changing how random pieces are sampled changes the seeded stream unless handled carefully.

CPU profiler evidence found **14 forward addmm calls and 18 backward mm calls per update**, rather than five matmuls for the entire update. It also recorded 18 copy operations/update. Copies occupied 19.8% of the profiler's captured CPU operator time; this is not 19.8% of total step time because NumPy/Python preparation is outside those operator records.

Every game ply reads probabilities back with `.cpu().numpy()`. Sacrifice checks have two additional readbacks per check. Piece training reads its metric values back before backward; step completion converts three CUDA loss tensors to Python floats. All are synchronization opportunities. `non_blocking=True` on inference input does not remove the dependency on the immediately needed CPU output. Standalone H2D timings include allocation/API and synchronization overhead; they are not measurements of raw PCIe bandwidth.

Torch thread counts 1, 2 and 6 yielded approximately **12.10, 12.13 and 12.19 ms/update**, followed by a noisy repeated 1-thread result of 12.79 ms. No significant CUDA-training improvement from reducing intra-op threads was established. CPU inference on a 48-row fixed input improved from 0.15–0.16 ms at one thread to 0.103 ms at six, but six threads per actor would oversubscribe this CPU.

**4. Ranked hotspots and proposals.**

By measured opportunity, the main hotspots are (1) predict preparation and the remaining CPU-heavy piece loss, (2) synchronous per-turn GPU inference, (3) candidate encoding, (4) opponent work at default strength, and (5) packing/logging. Backward is meaningful at 1.74 ms/update but is not the leading opportunity. Removing an entire component is only an upper bound: historical removal of all training caps improvement at 1.56x, all inference at 1.37x, or all environment at 1.35x, considered individually.

The following ranking considers both likely benefit and implementation effort. Expected end-to-end changes use the historical approximately 135 ms/game and four updates/game unless stated otherwise. They are estimates, not implemented training results; gains overlap and must not be multiplied blindly.

| Rank | Proposal | Evidence and expected benefit | Estimated effort / determinism |
|---|---|---|---|
| 1 | Gather valid packed predict rows before expanding them; supply the same valid logits to the existing masked loss | Isolated output-equal experiment: **3.141 -> 1.756 ms**, saving **1.386 ms/update** for 2,528 valid rows. About **11% faster training**, **4% faster complete loop** | Small, around 0.5–1 day including equivalence tests. Preserve sampled indices, candidate order, labels and RNG calls. Keep public tensor contracts; use an internal compact path if needed |
| 2 | Reuse float64/float32 piece-score scratch buffers while preserving draw order and cast/multiply order | **1.544 -> 0.793 ms** for generation, cast, mask and argmax: **0.751 ms/update saved**, around **6% training / 2% overall** | Small, around 0.5 day. First-draw selected indices matched; full seeded regression still required. Do not switch to float32 RNG or sparse sampling and claim the same seed trajectory |
| 3 | CPU inference copy of the model, GPU learner; refresh at the existing game/update boundary | Matched 30-game tests: **CPU 21.13–21.57 vs CUDA 16.76–17.66 games/s**, a **22–26% self-play gain**. Same 1,557 plies and 45,893 candidates each trial. Saves about 0.19–0.24 ms/ply, suggesting **roughly 15–20% overall** on 100-ply games before model-copy cost | Medium, around 1–2 days. One thread per parallel actor. Full legal-move batch stays intact. CPU/GPU arithmetic need not be bit-identical; matching counts here do not prove all seeded trajectories remain identical |
| 4 | Vectorize packing all opponent candidates in one NumPy call | 38-row example **74.5 -> 16.4 us**, **4.54x faster packing**, identical packed bytes. If the measured packing cost transfers to real games, save around **37 us/ply**, or **2–3% overall** | Small, around 0.5 day. Test all halfmove clocks and special positions; preserve current float64-before-rounding behavior for the clock |
| 5 | Two CPU actor processes feeding one GPU learner | Measured actor scaling below; analytical end-to-end ceiling near **20 games/s** with the present learner, compared with 7.4 in history. A cautious initial target is **15–20 games/s (2.0–2.7x)**, not a measured integrated result | Larger, around 3–5 days including queues, weight updates and ordering. Async actors introduce policy lag and alter replay/RNG ordering. Exact original online trajectory is not preserved; deterministic rounds and per-worker seeds can provide a new reproducible schedule |
| 6 | Batch/vectorize candidate encoding while retaining python-chess push/pop and the frozen 782-feature layout | Current **302 us/ply**. A hypothetical 2x encoder improvement saves **151 us/ply**, approximately **12–13% overall** at historical game length | Medium/large, around 2–4 days. Speedup unmeasured. Golden encodings, both orientations, castling, en passant, promotions and clock rounding must remain exact |
| 7 | Transfer packed batches and unpack on GPU; persistent buffers, fewer transfers/readbacks, potentially fused training execution | Could attack the **3–4+ ms** preparation/copy region plus dispatch overhead. Provisional **10–25% training / 4–10% overall** opportunity, overlapping ranks 1–2 | Larger, several days. No integrated estimate validated. Piece sampling still needs occupancy information; compiling only matrix layers leaves NumPy and synchronization intact. Preserve precision, losses and seeded draws |

No proposal changes the loss objectives, update ratio, opponent strength, piece values, legal move source, or number of candidates evaluated per turn. Reducing updates, dropping the piece loss, weakening the opponent or subsampling inference moves would change the experiment and are excluded as performance fixes. Dropping logging would also lose requested observability; batch it instead.

**5. Worker count and actual limits.**

These tests used real GameWorker CPU inference, one Torch intra/inter-op thread per process, independent seeds, 30 measured games per worker after warm-up, and a start barrier. No learner, queue transfer or weight refresh was included. Different worker seeds produce different game lengths, so plies/s is the more reliable scaling comparison.

| CPU actor processes | Aggregate games/s | Aggregate plies/s | Ply throughput relative to one |
|---|---:|---:|---:|
| 1 | 18.57 | 1,079 | 1.00x |
| 2 | 38.35; repeat 39.47 | 2,177; repeat 2,240 | 2.02–2.08x |
| 4 | 70.73 | 3,788 | 3.51x |
| 6 | 95.05 | 5,149 | 4.77x |

**Start with two actors and one learner**, not twelve actors. The learner's 12.5 ms/update at four updates/game caps it near **20 games/s** before concurrent contention. Two actors already exceed that for the measured roughly 50–60-ply games. At the historical 103.5 plies/game, two actors' 2,177 plies/s projects to around 21 games/s, only just sufficient. Three actors are a sensible next experiment for longer games or after learner optimization; benchmark four if preprocessing and learner throughput improve. Reserve CPU capacity for the learner, OS and transfers. Six saturated actors leave little headroom on six physical cores.

Use processes rather than Python threads: the encoding loop is Python-heavy, the pybind `choose_move` binding does not release the GIL (`tiny_opponent/bindings.cpp`), and encoding has module-global scratch state. Actor processes need only the small model and local game records, not a complete independent Trainer/replay allocation. Packed records should cross the queue in contiguous blocks, with bounded queue depth; returning Python lists of small arrays adds serialization overhead. The benchmark's games/s must not be advertised as integrated training throughput.

**6. What remains unproved.**

* GPU utilization of 15% was supplied in the question, not independently reproduced as a sustained trace. CUPTI failed with `CUPTI_ERROR_INVALID_DEVICE`; exported profiler records contain CPU events only. CUDA-event forward timing (0.174 ms for a resident 38-row forward) includes host submission gaps and cannot establish kernel busy time. Exact launch-versus-kernel attribution needs a working GPU profiler.
* Integrated CPU actors plus GPU learning, model refresh, queue costs, contention and policy staleness were not built. Neither 20 games/s nor the optimum actor count is guaranteed by isolated actor scaling.
* Prototype gains were measured separately; their combined full-step improvement, learning curves, long-run memory behavior and exact seeded equivalence remain to be tested. The RNG prototype checked first-draw selected indices, and packing equivalence covered the measured example rather than the full golden suite.
* CPU/GPU inference matched game/candidate counts in the controlled trials, but bitwise output equality and arbitrary seeded trajectory equivalence were not established.
* The historical effective opponent rung and exact conditions of the supplied 50/20/17.5/12 breakdown are unavailable. The measured fast rung demonstrates a plausible explanation, not proof of historical configuration.
* The first benchmark file's initial `full_selfplay_cpu/cuda` results were collected across differing model snapshots and should not be compared. The separate follow-up's `matched_cpu/cuda` measurements correct that by freezing identical weights across trials. The benchmark source was also corrected for future reruns.

Reproducible artifacts: `investigation_codex_bench.py`, `investigation_codex_results.json`, `investigation_codex_console.txt`, `investigation_codex_followup.py`, `investigation_codex_followup.json`, `investigation_codex_followup_console.txt`, `investigation_codex_opponent.json`, and CPU-only `investigation_codex_profiler.txt` / `investigation_codex_trace.json`. Run scripts from the repository root with `python -B -u <script>`. They write only investigation artifacts and execute bounded in-memory measurements.
