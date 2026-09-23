# v4 — Emergent King Safety

**We define the consequence — vulnerability to checkmate — the network discovers what king safety is.**

## What the value means

For a position and one king (the defender), the opponent is the attacker. With `n` = the number of
moves the attacker needs to **force** checkmate against the best defence:

```
U(position, king)  = gamma ** (n - 1)     1 = mated / unavoidably mated, 0 = no mating attack
KS(position, king) = 1 - U                gamma = 0.8
```

Nothing else is scored: not material, checks, sacrifices, pawn shields, attackers/defenders, open
files, patterns or win probability. A sacrifice that forces mate lowers KS exactly as much as any other
forced mate; an "ugly" king that cannot be mated keeps a high KS.

## How it is learned (`tonkai/king_safety_training.py`)

| Piece | What it does |
|---|---|
| Input | the raw 782-value encoding TonkAI already uses, seen from the king's side |
| Exact grounding (C++, `forced_mate`) | rules-only forced-mate search up to 3 attacker moves: a mate gives U exactly; a full search without mate proves U ≤ γ³ |
| Beyond the horizon (`expand2`) | every legal attacker/defender continuation two plies deep, leaves valued by a lagged copy of the model (value iteration), capped by the exact bound |
| Attack/defence episodes | the attacker plays what the model thinks hurts the king most, the defender what protects it most (10% random); their positions join the training pool |
| Positions | real TonkAI games (campaign replay) + benchmark games + those episodes, both kings each |

## Using it

```bash
python train_king_safety.py --minutes 60 --workers 8          # checkpoints/king_safety/latest.pt
python train_cognitive.py ... --king-safety-model checkpoints/king_safety/latest.pt   # v4
```

With `--king-safety-model`, every decision uses `KS_self - KS_enemy` instead of the hand-written
king safety (candidate moves, the reply search and the shaped objective's measure). Without it,
v3.1 is unchanged.

## Inspecting it

- `KingSafetyModel.ks(board, colour)`, `.ks_both(board)` → KS of the white and the black king.
- Every game's `stats["king_safety"]`: `[KS white, KS black]` after every ply.
- Training logs `king_safety` records (every 10th game and every win) with that series.
- Evaluation matches write `[%ks w=… b=…]` comments on every move of their PGN.
