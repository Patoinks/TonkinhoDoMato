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
