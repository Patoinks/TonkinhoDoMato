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

Scan incremental (só recalcular o que as últimas peças mexidas tocam), medido com `scan_reuso.py` em 833 decisões: só 52% dos lances mantêm o valor dois meios-lances depois, 54% mesmo longe (>2 casas) das peças mexidas; só 254/833 posições ficam todas iguais. Reaproveitar erraria metade dos valores: não.

## Sem respostas do adversário + tabela partilhada (`--rapido` desde 2026-10-18)

`ab_stockfish.py`: as configurações jogam intercaladas no mesmo pool (o Stockfish a 2 ms muda de força com a carga
do PC: duas corridas separadas da mesma configuração deram 0.642 e 0.561). m039 de base, 500 jogos cada:

| Configuração | Score vs SF 2250 | ±95% | s/jogo | ms/lance (mesmas posições, 1 processo) |
| --- | ---: | ---: | ---: | ---: |
| m039 | 0.598 | 0.043 | 1.20 | 12.8 |
| + `shared_table=True` | 0.640 | 0.042 | 1.16 | 11.9 |
| + `reply_top=0` (sem respostas) | 0.627 | 0.042 | 1.04 | 11.4 |
| **+ ambos** | **0.644** | 0.042 | **0.98** | **10.7** (-16%) |

Nenhuma perde força; `--rapido` passa a incluir `shared_table=True reply_top=0`.

Correção: `tempo_mesmas_posicoes.py` usava um jogador para as duas cores, e a memória do jogo recomeçava a cada
lance (~2,6 ms a mais em todas as configurações). Com um jogador por cor: m039 antigo 10,36 ms -> **8,00 ms** (-23%).

Distribuição dos 8,0 ms (`perfil_posicoes.py`, 1095 decisões): root_search 4,95 (62%), scan 1,93 (24%),
rede candidatos 0,30, memória 0,18, segurança do rei 0,17, princípios de abertura 0,12, resto ~0,4.

## Núcleo rápido (`fast_core`) e tática a 6 (`--rapido` desde 2026-10-18)

`fast/zobrist.inc`, `fast/nucleo_q.inc`, `fast/nucleo_deep.inc`: as mesmas regras de pesquisa com chaves Zobrist
incrementais, listas de lances em arrays na pilha, ordenação por chave compactada e countermove. Por omissão
desligado: o Leto original continua com lances idênticos a `base-4.json`.

| | ms/decisão (mesmas posições) | Lances iguais | Score vs SF 2250 (A/B intercalado, 500 jogos) | s/jogo vs SF |
| --- | ---: | ---: | ---: | ---: |
| m039, tática 7 | 8.09 | - | 0.609 ± 0.043 | 1.09 |
| + `fast_core` | 7.39 (-9%) | 99% | 0.595 ± 0.043 | 1.00 |
| **+ `fast_core` + `tactic_depth=6`** | **4.93 (-39%)** | 90% | 0.573 ± 0.043 | **0.63** |

O núcleo rápido só dá ~10% (3.1 -> 3.4 M nós/s): o custo por nó não estava nas listas nem na chave. A tática a 6
corta 39% do tempo e talvez custe um pouco de força (-0.036, dentro de ~1.2 desvios). Leto rápido vs Leto rápido:
0.57 s por jogo.

## Ganhos pequenos somados (2026-10-18, tarde)

| Alteração | Efeito medido |
| --- | --- |
| Verificação do lance escolhido com as podas da raiz (era exata a 6) | 4,41 -> 3,95 ms por decisão |
| Tabela de 16 bytes (2^16 cabe no L2 de 1 MB do Zen 4) + prefetch da entrada do filho | +5% nós/s (3,45 -> 3,61 M) |
| Scan do núcleo rápido começa o aprofundamento em 2 | ~-3% do scan |
| `delta_pruning=True` + `scan_keep=20` (scan exato só para quem pode estar entre os 20 melhores) | ~-2%, 99% lances iguais |
| `tactic_top=6` | -6% mas 0,478 vs 0,528 contra o Stockfish: fora |
| Kernels OpenBLAS (`OPENBLAS_CORETYPE`) | o automático já é o melhor (AVX-512) |
| Workers fixos a núcleos físicos (`ab_stockfish.py --pin`) | igual (6,1 vs 6,4 ms por lance) |

Força, código de antes (9756d87) e de agora lado a lado (dois pools ao mesmo tempo, 500 jogos cada vs Stockfish 2250):
antes 0,549, agora 0,529, agora + pequenos 0,531 (±0,044): sem perda medível.
O Stockfish a 2 ms fica mais forte com o PC menos carregado: 6 workers 0,39-0,45, 11-12 workers 0,52 (mesma
configuração). Só comparações intercaladas ou simultâneas valem.

`--rapido` agora: 3,92 ms por decisão (PGO) nas mesmas 1095 posições. Leto original: lances idênticos a `base-4.json`.

## Rede nativa AVX-512 (`fast/rede_fast.cpp`, por omissão desde 2026-10-18)

Encoder, memória GRU, fusão, cabeças de sobrevivência e previsão, e segurança do rei em C++ compilado com
`/arch:AVX512` (Zen 4): kernel de 4 linhas x 64 colunas com FMA, só as colunas não nulas do lote, exp vetorizado.
Diferenças de arredondamento float32 (~2e-5). `LETO_REDE_NATIVA=0` volta à versão numpy.

| | numpy | nativa |
| --- | ---: | ---: |
| candidatos | 0,273 ms | 0,147 ms |
| segurança do rei | 0,154 ms | 0,114 ms |
| sobrevivência + encode + memória | 0,176 ms | 0,086 ms |
| decisão inteira | 3,85 ms | **3,57 ms** |
| vs Stockfish 2250 (pools simultâneos, 500 jogos) | 0,516 | 0,551 |

clang `-march=znver4`: não instalado (o instalador do LLVM pede UAC e não foi possível confirmar): saltado.

## Campeão no `--rapido`: t0467 (2026-10-18)

`--rapido` passa a incluir o campeão do torneio com prova de tempo (m039 + cinco mutações): `safety_weight=3.416
value_scaling=True reply_weight=0.2404 fifty_horizon=28 scan_keep=12`, com `opening_weight=0.08524 draw_value=0.5594
simplify_margin=0.411 ahead_draws=False` do m039. Final: 0,596 vs 0,570 da base (500 jogos), ~5% mais rápido.
Já não é preciso passar o m039 com `--set`.
