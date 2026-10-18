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

## Pesquisa: PEXT, PGO e o fim de jogo (2026-10-18)

Todas exatas: o Leto original continua com lances idênticos a `base-4.json` (`compara.py`, 4 jogos) e o resultado de
`root_search` em 5 posições tem o mesmo digest.

| Alteração | Ganho | Nota |
| --- | ---: | --- |
| Tabelas de ataque PEXT/BMI2 (`/DCLAUDIO_PEXT`, com AVX2) | ~10% na pesquisa | 15,87 -> 14,29 ms por root search |
| PGO refeito com PEXT (`build_pgo_gen/use.cmd`, 6 jogos `--rapido`) | ~8% na pesquisa | original: 10,05 -> 7,19 s/jogo (x1,40) |
| `fim_rapido.Judge` em `torneio.py`/`vs_stockfish.py` | 0,17 -> 0,013 ms por meio-lance (~3% do jogo) | 408 804 meios-lances comparados com `board.outcome(claim_draw=True)`: 0 diferenças |
| Mate sem lista na quiescência (`any_legal` a profundidade 0) | ruído | mantido: exato |
| `LAZY_PICKS` 2, 3, 5 | ruído | fica 1 |

Make/unmake não foi feito: `Pos` tem ~104 bytes (copiar custa menos que desfazer o lance), ganho estimado < 2%.
Perfil nativo depois do PEXT: DeepSearch 42% (ordenação dos lances e `place` ~23%), ExchangeSearch 32%
(`pseudo_captures` 8,5%), `Pos::make` 9%. Rede numpy por lote de ~54 linhas: 0,38 ms (residual 128->512->128 0,19 ms),
limitada pelo BLAS. `fast/build_var.cmd NOME "DEFINES"` compila variantes para A/B; `perfil_rede.py` mede cada camada.

## Pesquisa e scan, segunda volta (2026-10-18)

| Tentativa | Tempo (mesmas posições) | Lances iguais | Força vs Stockfish 2250, 500 jogos | Decisão |
| --- | ---: | ---: | ---: | --- |
| m039 (referência) | 13,5 ms | - | 0,611 | - |
| `bad_capture_pruning` + `delta_pruning` + `shared_table` | 11,3 ms (-16%) | 93% | **0,548** | rejeitada, fica desligada |
| `shared_table=True` | -6% | 98% | não medida | - |
| `tactic_top=6` | -6% | 96% | não medida | - |
| `tactic_rounds=2` | 0% | 100% | - | - |
| Scan sem linhas principais (`lines=False`) | -3% do scan | exato | = | entra |
| Scan: aprofundamento a partir de 2 ou 3 | 0% | muda valores | - | não |
| Capturas da quiescência por tipo de peça | 0% | exato | - | não |

Perfil nativo do scan (`deep_after_moves`): ExchangeSearch 38%, DeepSearch 32%, `make` 9%, lances legais 7%.
O Leto original continua com lances idênticos a `base-4.json`; o Leto rápido com lances idênticos a antes.
