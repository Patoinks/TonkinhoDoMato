# Handoff: TonkAI on the ANW Games website

Brief for a second agent. Assume no prior context.

## What TonkAI is

A 131k-parameter chess network that deliberately does **not** play like an
engine. It has no piece values, no centipawns, no search, no opening book. It
picks moves purely from `P(win) / P(draw) / P(loss)` and it separately tries to
**predict the human's next move** before they play it.

Reference implementation: `C:\Users\Martim\Desktop\Github\TonkAI` (Python).
Read `README.md` there first.

## Hard constraints - do not negotiate these

The whole point of the project is what is absent. When the model plays badly,
the fix is never to help it.

* **Never** add piece values, material counting, centipawns, or a static
  evaluation anywhere in the web code.
* **Never** add search (minimax, alpha-beta, MCTS), an opening book, or
  tablebases to the site.
* **Never** substitute another engine's move when TonkAI's move looks bad.
* The move shown must be exactly `argmax(utility)` over the legal moves, from
  the network as exported.

If the model plays a weak move, that is data, not a bug. Show it.

## What to build

A `/tonkai` page, built in the git worktree
`C:\Users\Martim\Desktop\Github\ANW-TONKAI-PAGE`, branch `tonkai-page`.

**Work in that worktree, not in `ANW-GAMES-WEBSITE`.** The main checkout has
the owner's uncommitted work in it and must not be touched. The worktree is a
clean checkout of `main` at `f3daf8f`; `node_modules` is not there yet, so run
`npm install` first. Commit your work on `tonkai-page`; do not merge to `main`
and do not push.

Stack already there: React 19, Vite 7, TypeScript, react-router-dom 7,
Firebase (Firestore). Pages live in `src/pages`, follow how
`LolLeaderboardPage.tsx` is wired into the router.

**Everything runs client-side.** The weights are 527 KB and the forward pass is
five matmuls. Do not add a backend, a Cloud Function, or an inference service -
the site has no Functions by design.

### 1. Play against it

* A chessboard the visitor can play on. `chess.js` for rules and legal moves;
  `react-chessboard` or `chessground` for the UI - whichever is less work.
* Visitor picks colour. Legal moves only.
* TonkAI replies using the exported network (see below).

### 2. Show the prediction - this is the interesting part

Before the visitor moves, show TonkAI's guess at what they are about to play:

```
TonkAI expects:  Nc6 25%   Nf6 18%   e5 17%   d5 11%
```

After they move, show whether it called it, and the surprise
`-log(p)`. Keep a running mean surprise for the game, and show it at the end
("TonkAI saw 4 of your 12 moves coming"). That number is the actual demo.

### 3. Show the thinking

* Per-move `win / draw / loss` and utility for TonkAI's chosen move.
* An expandable list ranking every legal move by utility.
* Optional: the contextual piece values. Be careful - at the current
  checkpoint these are all ~0.003 and say nothing. Only surface this once a
  checkpoint with real values lands (see "Current model quality").

### 4. Metrics (Firestore, follow the `/lol` pattern)

Anonymous, no login required, one document per finished game:

`result`, `plies`, `tonkai_colour`, `mean_surprise`, `top1_hits`,
`moves_played`, `client_timestamp`, `weights_sha256`.

Then a small stats strip on the page: games played, TonkAI's W/D/L, mean
surprise, top1 prediction rate. Mirror the existing Firestore rules style -
do not loosen rules to make writes easier; if the rules block an anonymous
write, say so rather than opening the database.

## The exported model

In `C:\Users\Martim\Desktop\Github\TonkAI\web\`:

| file | what |
|---|---|
| `tonkai.bin` | 526,804 bytes, raw float32 little-endian, all 19 tensors concatenated |
| `tonkai.json` | manifest: tensor names, shapes, byte offsets, `draw_weight`, sha256 |
| `ENCODING.md` | **the exact 782-feature input layout and forward pass** |
| `golden.json` | expected outputs from the Python reference |

Copy these into the site's `public/` and fetch them at runtime.

### Read ENCODING.md properly before writing any inference code

The encoding is the one thing that will silently ruin this. 782 features,
always written from **TonkAI's own side**, and when TonkAI plays black every
square index is flipped `sq ^ 56` (ranks flipped, files NOT mirrored). A wrong
encoding does not throw - the network returns confident nonsense.

Note also: `chess.js` square indexing is not the same as the
`0 = a1 .. 63 = h8` convention used here. Convert explicitly, do not assume.

### Verification gate - do this before building any UI

`golden.json` has 9 positions with their expected `value_logits`, plus one full
move ranking and one full predict distribution. Write a test that runs the JS
implementation on those FENs and asserts agreement to `1e-4`.

**Do not build the board until that test passes.** If it does not pass, the
encoding is wrong, and every move the page shows afterwards is meaningless.

The predict head is only ever valid on positions where the **opponent** is to
move. Calling it when TonkAI is to move is off-distribution and returns
garbage - `golden.json`'s `predict_distribution` shows the correct usage.

## Current model quality - set expectations accordingly

The checkpoint in `web/` is trained but **weak on purpose-to-be-fixed grounds**:

* It lost 99.8% of its 5000 training games, so the value head is saturated:
  `P(loss) ≈ 1.0` for every move, and the spread between its best and worst
  move is about `0.02`. **Its move choice is close to noise.** It plays things
  like `Na6` on move 2. Any visitor will beat it.
* The **predict head is genuinely good** (top1 34%, and it puts Nc6/Nf6/e5/d5
  at the top after 1.e4). That is what the page should be built around.

So: frame the page as *"watch it guess your moves"*, not *"beat the chess AI"*.
A better checkpoint is coming and is a drop-in replacement - the page must load
the weights at runtime and key the metrics on `weights_sha256`, so a swap does
not mix statistics from two different models.

## Second job: local improvements in the TonkAI repo

`C:\Users\Martim\Desktop\Github\TonkAI` is a git repo at commit `4455d88`,
clean. You own these there:

1. **Better metrics.** `tonkai/metrics.py` already tracks W/D/L, losses,
   predict top1/top3, surprise, entropy, piece-value mean/std and timers.
   Missing and wanted: per-run history written to CSV or JSONL so runs can be
   compared and plotted afterwards, plus a small script that plots it. Keep
   logging cheap - it runs inside the game loop.
2. **`play.py` polish.** It works (human vs TonkAI in the terminal, shows the
   predict distribution and the surprise). Worth adding: move list / PGN
   export, a `hint` command, a clearer end-of-game summary.
3. **Elo anchoring.** `calibrate_opponent.py` produces only *relative* numbers
   today. If you can anchor TinyOpponent against a reference of known strength,
   add it as an **optional** mode (`--anchor ...`) that is skipped when the
   reference is absent. Never make it a required dependency, and never let any
   reference engine touch TonkAI's training - it may only ever measure
   TinyOpponent.

### Hard limits in the TonkAI repo

* **Do not change `tonkai/encoding.py`, `tonkai/model.py`, or any tensor name
  or shape.** The exported weights, `golden.json` and the whole web port depend
  on them. Changing the encoding silently invalidates everything.
* **Do not retrain or overwrite `checkpoints/latest.pt`.** Another agent is
  working on the training signal in parallel.
* `python -m pytest` must stay green - 84 tests pass right now. Run it before
  you finish.
* Commit your work with clear messages. Never force-push or rewrite history.

If you conclude the encoding or the model must change, stop and say so in your
final summary instead of doing it.

## Done means

1. `golden.json` verification test passes at 1e-4.
2. A human can play a full game in the browser, legal moves only, including
   castling, promotion and en passant.
3. Predictions and surprise shown live; mean surprise at game end.
4. Finished games written to Firestore; stats strip reads them back.
5. `npm run build` and `npm run lint` clean.
6. No piece values, no search, no engine fallback anywhere in the diff.
