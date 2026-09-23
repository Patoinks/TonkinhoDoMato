# Mate calculation in play

TonkAI calculates forced mates while it plays, with the exact rules-only search of `tonkai_fast`
(`Board.forced_mate`, the same solver that grounds v4 King Safety). No network, no heuristics:
a mate is either proven by the rules or not claimed. Off by default, so older checkpoints and the
frozen rivals play exactly as before.

| Setting | What it does |
|---|---|
| `mate_search_moves` N | **Attack.** Before every move, look for a forced mate within N own moves. If there is one, play a move on a shortest line (like an immediate mate, this overrides exploration). The choice carries `forced_mate = n`, the decision's trace too, and TonkAI never resigns with a forced mate on the board. |
| `mate_defense_moves` D | **Defence.** Every candidate move after which the opponent forces mate within D moves ranks under every move the reply search compared, before that search (so it looks at safe moves) and after it. Among such moves a later mate ranks higher; one is played only when every move allows mate. Moves that end the game keep their known value. |
| `mate_node_limit` | Nodes per search. Past it, nothing is claimed (the search never invents a mate). |

```bash
python train_cognitive.py ... --mate-search-moves 3 --mate-defense-moves 2
python scripts/match_checkpoints.py ... --mate-search-moves 3 --mate-defense-moves 2   # current side only
```

Counters in the metrics (`cognitive`): `mate_attack_searches`, `forced_mate_moves`,
`mate_defence_searches`, `mate_moves_avoided`.

## Measured (2026-09-23, v3.1 at game 65 963, N = 3, D = 2)

On 309 real positions from the benchmark games:

| | Without | With |
|---|---|---|
| Time per decision | 6.9 ms (p95 9.5) | 15.0 ms (p95 25.4) |
| Forced mates (≤ 3) found and played | 4 of 8 | 8 of 8 |
| Moves allowing a forced mate (≤ 2) | 2 | 0 |

Match against Checkpoint 1 (500 games, seed 2026, as the benchmark; only v3.1 calculates):

| | Wins | Draws | Losses | Score |
|---|---|---|---|---|
| v3.1 without (benchmark) | 188 | 161 | 150 | 53.8% (95%: 50.2–57.4%) |
| **v3.1 with mate calculation** | **252** | **160** | **88** | **66.4%** (95%: 63.1–69.7%) |

About +90 Elo from calculation alone (+118 over Checkpoint 1). Result: `history/matches/v3.1-mates-vs-checkpoint1-500.json`.

Inside a line the search ignores repetition and the 50-move rule; the move actually played is
always checked against them.
