# Benchmarks

Frozen Tonkinhos and reference results, kept out of `checkpoints/` so training never touches them.

| File | What |
|---|---|
| `tonkinho-checkpoint1-jogo52895.pt` | **Checkpoint 1**: the first frozen Tonkinho (game 52 895). The fixed benchmark rival. |
| `tonkinho-v3.1-jogo65963.pt` | Tonkinho do Mato v3.1 at game 65 963, with its piece values (`kd_values`), frozen. |
| `benchmark-v3.1-vs-checkpoint1-500.json` / `.pgn` | The reference result below and its 500 games. |

## Reference: v3.1 (game 65 963) vs Checkpoint 1, 500 games (2026-09-23)

| | Wins | Draws | Losses | Score |
|---|---|---|---|---|
| **Total** | **188** | **161** | **150** | **53.8%** (95%: 50.2%–57.4%) |
| As white (250) | 96 | 82 | 72 | 54.8% |
| As black (250) | 92 | 79 | 78 | 52.8% |

About **+27 Elo** over Checkpoint 1 in ~13 000 training games (one game was cut at the move limit).

## Running it again for a future model

```bash
python scripts/match_checkpoints.py --current checkpoints/cognitive-mix-20260923/latest.pt \
    --rival benchmarks/tonkinho-checkpoint1-jogo52895.pt --name "Tonkinho Checkpoint 1 (jogo 52895)" \
    --games 500 --workers 8 --out benchmarks --label benchmark-<version>-vs-checkpoint1-500
```

Same rival, seed 2026, 500 games, alternating colours, no resignation, both sides exploring
as in training. A score clearly above 53.8% means the new model beats v3.1's result.
