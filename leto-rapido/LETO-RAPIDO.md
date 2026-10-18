# Leto rápido

Medido em 2026-10-18 no Ryzen 5 7500F, um worker, contra o G4 Leto original (`torneio.py`, 48 jogos).

| Configuração | `root_search` por lance | Decisão por lance | Score contra o original |
| --- | ---: | ---: | ---: |
| Original (pesquisa exata a 6 meios-lances) | 27,6 ms | 37 ms | 0,50 |
| `root_prunings=True` (null move + reduções) | 2,5 ms | 9,7 ms | 0,375 |
| **`root_prunings=True tactic_depth=7`** | **6,3 ms** | **13 ms** | **0,479** |
| `tactic_nodes=20000 tactic_top=6` | 3,9 ms | 11 ms | 0,25 |
| `tactic_depth=4` | 0,9 ms | 7 ms | 0,27 |

Recomendado: `--set root_prunings=True --set tactic_depth=7` (força igual dentro da margem de 48 jogos, ~3x mais rápido).
O módulo C++ por omissão continua exato: sem estas definições o Leto joga exatamente como o original.
Compilação: `fast/build_pgo_*.cmd` (PGO + AVX2, x1,12 com lances idênticos).

## Pawnomics v4 (`pawn_v4=True`, `fast/pawnomics_v4.inc`)

Every pawn in C++, in Leto's own piece values: passed pawns by how likely and how soon they queen (each square of the
path: free, contested, lost by the exact exchange, or blocked; the king race with the tempo; own king, rook behind,
protected and connected passers; the enemy material left), and the structure of all pawns (doubled, isolated,
backward, phalanx/protected, candidate passers). v1 only looked at pawns with no enemy pawn in front on their file.

Speed: all candidates of a decision in 0,005 ms (v1: 2,5 ms; ~490x).

Strength against v1 (fast configuration both sides, 192 games each, ±0,07):

| pawn_weight with v4 | 0,15 | 0,3 | **0,5** | 0,8 | 1,2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| score | 0,464 | 0,471 | **0,536** | 0,497 | 0,445 |

Best at `pawn_weight=0.5`, level with v1 within the margin, at a fraction of the cost.

## Leto rápido completo (`--rapido`)

`root_prunings=True tactic_depth=7 scan_prunings=True fast_net=True pawn_v4=True pawn_weight=0.5`, with the aspiration
window in the root search, the numpy networks (`rede_rapida.py`), native capture gains and PGO + AVX2.

| | Original | Leto rápido |
| --- | ---: | ---: |
| Decision (ms) | 37 | ~6 |
| root_search per decision (ms) | 27,6 | 2,6 |
| Game, Leto vs Leto, one worker (s) | ~10 | **~1,5** |
| Strength vs the original (192 games) | 0,50 | **0,487 ± 0,071** |

Where the ~6 ms go: root_search 2,6; networks (candidates + replies) ~1,1; scan 0,95; king safety, memory,
values, endings ~0,6; Python glue ~0,7. The search runs ~2,7 M deep nodes/s (~370 ns per node with its quiescence).
Tried and rejected: keeping the table between moves, larger tables (2^18, 2^20: no change), deeper late-move
reductions and adaptive null move (slower: more re-searches), 6 or 8 root candidates (no change).

    python leto-rapido/torneio.py --rapido --games 48 --workers 6 --set reply_top=5
    python leto-rapido/vs_stockfish.py --rapido --deus zeus --games 24 --movetime 0.01
