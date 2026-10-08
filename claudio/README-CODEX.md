# Claudio: guia para o Codex (treino na GPU do Martim)

Escrito pelo Claude a 2026-10-08, no fim de uma sessão na nuvem (Linux, 4 núcleos, sem GPU).
O Martim vai continuar o trabalho no PC dele, com GPU. Isto é o que precisas de saber para
continuar sem repetir o que já foi medido. O histórico completo está em [ESTADO.md](ESTADO.md).

## 1. Regras do Martim (valem sempre)

- **Orgânico.** Nada de valores de peças à mão (1-3-3-5-9), nada de bónus feitos à mão. O
  Stockfish e o Fairy-Stockfish são **só** adversários ou juízes, nunca professores.
- **Winrate = vitórias / jogos.** Um empate não conta como meia vitória.
- No máximo **6 workers**, prioridade baixa. Não parar nem relançar treinos sem o Martim pedir.
- Antes de editar, reservar os ficheiros no [WORKING.md](../WORKING.md); commit só dos teus ficheiros.
- Não ler o código do Gepeto (a IA do ChatGPT no duelo; commits `group_stage`, pasta `Gepeto`).
- Português de Portugal, direto, com tabelas quando há números.

## 2. O que é o Claudio

Um jogador de xadrez que decide com: **intuição** (a rede do TonkAI, 810k parâmetros, com memória
GRU do jogo) + **cálculo exato** em C++ (`claudio/fast/claudio_fast.cpp`) com os **valores das
peças medidos por ele** (P 1, C 2,13, B 2,73, T 3,98, D 9,06). O código está em
`claudio/claudio_agent.py`; corre sobre uma cópia congelada do TonkAI em `claudio/base/`
(`claudio/paths.py` trata do `sys.path`). Todas as opções estão em `ClaudioConfig`.

Como decide um lance (por ordem):
1. Mate forçado próprio (`mate_attack` lances; os longos só se o modelo de segurança do rei
   disser que o rei inimigo está exposto: `mate_gate`).
2. Rede: valor de cada posição alcançável.
3. Material: sequência exata de trocas depois de cada lance (`safety_weight` × tanh(ganho/3)).
4. **Varrimento** (`scan_depth`, novo): todos os lances calculados mais fundo.
5. Fins de jogo e repetições com o valor exato; **empate = −1 quando ele acha que está a
   ganhar** (`ahead_draws`, novo).
6. **Respostas** (`reply_top`, novo): os melhores candidatos julgados depois de cada resposta do
   adversário, pesada pela própria previsão do lance do adversário (`reply_mode=expect`).
7. Cálculo de material fundo (`tactic_depth`) nos `tactic_top` melhores (PVS em C++).
8. Verifica o lance que vai jogar (defesa contra mate, `tactic_rounds` verificações).

## 3. Os modelos (todos com a mesma rede g0; o que muda são as definições)

| Ficheiro | Definições | Força medida |
|---|---|---|
| `claudio-grande-g0.pt` | a rede g0, definições antigas | Fairy 1450: 53,5% · SF1800: 5,5% |
| `claudio-grande-g0-respostas.pt` | + `reply_top=6 reply_mode=expect` | Fairy 1450: 63,2% · Fairy 1700: 24,5% |
| `claudio-grande-g0-forte.pt` | + `tactic_depth=5` | Fairy 1700: 32,7% · **SF1800: 15,2% (26% dos pontos, ~1620)** |
| `claudio-campeao.pt` | campeão G10 do torneio + módulos (ver 5) | ganha aos outros Claudios; ainda não medido contra motores |
| **`claudio-g16.pt`** | G16 artificial + **todos os módulos** (ver 5b) | BO30 vs G15 treinado: 13 vitórias, 2 empates, 8 derrotas, 7 inacabados |

Vitórias contra motores: 600 jogos (Fairy) ou 400 (Stockfish), aberturas emparelhadas.

**O melhor contra motores medido é o `forte`.** O `campeao` e o G15 (abaixo) são os melhores
entre Claudios, mas **ainda falta confirmá-los contra o Fairy 1700 e o Stockfish 1800** - é o
primeiro passo a fazer.

G15 (o melhor entre Claudios até agora): o `claudio-campeao.pt` com
`safety_weight=1.5747,tactic_top=10,tactic_rounds=7,mate_gate=0.25634,king_weight=0.02288`.

## 4. O que já foi medido (para não repetir)

Ajudou (contra motores):

| Mudança | Efeito |
|---|---|
| `reply_top=6 reply_mode=expect` | Fairy 1450 +9,7 pp; Fairy 1700 +6,0 pp |
| `tactic_depth=5` (por omissão 3) | Fairy 1700 +8,2 pp; SF1800 6,75% → 15,2% |

Não ajudou ou piorou:

| Mudança | Efeito |
|---|---|
| `tactic_depth=6` / `7` | 4×/16× mais lento, SF1800 13,0% (sem ganho) |
| `reply_depth=3` (3.º meio-lance) | −14,8 pp (conta o material duas vezes, otimista) |
| `leaf_weight` (rede no fim da linha de material) | peso 1: −13,8 pp |
| `tactic_top=6`, `reply_top=10`, `exchange_depth=7`, `mate_defence=3`, `tactic_rounds=8` | nada (rounds8: +5,5 no Fairy, −4,0 no SF1800: sorte) |
| treino `--learn search` (TD-Leaf, 3100 jogos) | +0,3 ± 4,8 pp: nada |

Tempo por lance (`tactic_depth`, resto = forte, 1 núcleo): 1-3: ~10 ms; 4: 22; **5: 69**;
6: 272; 7: 1142; 8: 2601 (o limite de nós corta). O ponto ótimo medido é o 5.

A rede julga bem até 2 meios-lances à frente (as respostas ajudam); mais fundo engana-se (o
3.º meio-lance e o `leaf_weight` pioram). **O que falta para 1800 é uma rede melhor → treino.**

## 5. Torneio de mutações (resultados em `results/torneio-p3.json(l)`, `results/grupos-*.json`)

`claudio/torneio_mutacoes.py`: 10 gerações (200 mutantes na 1.ª, depois 127 + o vencedor),
cada parâmetro numérico muda ±20-50% com probabilidade 0,3, eliminatórias à melhor de 5,
profundidade fixa 3 (`--fixed`), fase de grupos a profundidade 5 (`--group-set`).

Tendência dos campeões G1→G10: **peso do material 0,2 → 0,79**, **candidatos a fundo 4 → 8**,
mates longos só com o rei inimigo muito exposto (`mate_gate` 0,8 → 0,375), segurança do rei
0,047 → 0,029. Os limites de nós desceram, mas isso é artefacto (desempate por tempo).

| Fase de grupos (prof. 5, empate −1 ao ganhar) | Resultado |
|---|---|
| original + G1..G10, 10 jogos/par | G8 47%, G9 46%, G10 44% … G1 16%, original 13% |
| G8/G9/G10, 200 jogos/par | **G10 28,7%**, G9 26,2%, G8 23,2% |
| G8/G9/G10 com módulos, 25 jogos/par | G10 24%, G9 20%, G8 20% |
| G10 vs tendência ×2 ("G20 artificial"), 20 jogos | G10 6/11/3 (o G20 passa do ponto) |
| G13..G17 artificiais, 20 jogos/par | **G15 32,5%**, G16 26,2%, G14 22,5%, G17 22,5%, G13 21,2% |
| G15 vs G10 com 200 jogos de treino, 50 jogos (mesmos módulos e profundidade) | 16/19/15: igual |

Gerações artificiais = a tendência G0→G10 prolongada: material `0,2·(0,7915/0,2)^(n/10)`,
candidatos `4+0,4n`, verificações `4+0,2n`, `mate_gate` `0,8·(0,3746/0,8)^(n/10)`,
segurança do rei `0,047·(0,0291/0,047)^(n/10)`. Ótimo entre G14 e G16 (material 1,4-1,8).

Módulos novos (pedidos pelo Martim, desligados por omissão; ligados no `campeao`):
`scan_depth=2` (ataques descobertos/garfos que ganham material à força, +3 ms),
`ahead_draws=true` (empate −1 quando o valor do melhor lance ≥ `ahead_value`=0,62),
`simplify_weight=0.05` (trocas que não perdem nada quando está à frente). **Ainda não medidos
um a um contra motores.**

## 5b. O G16 com todos os módulos (`claudio-g16.pt`, sem treino)

O pedido do Martim: o G16 artificial com todos os módulos orgânicos, sem treino. Tudo em
`claudio_agent.py` (`ClaudioConfig`), desligado por omissão, ligado neste checkpoint:

| Módulo | Definições no G16 | O que faz |
|---|---|---|
| tendência G16 | `safety_weight=1.8069 tactic_top=10 tactic_rounds=7 mate_gate=0.23761 king_weight=0.02181` | a tendência do torneio prolongada |
| limites do forte | `exchange_nodes=200000 tactic_nodes=2000000 mate_nodes=20000` | a descida no torneio era artefacto do desempate por tempo |
| varrimento | `scan_depth=2` | todos os lances calculados a 2 meios-lances (ataques descobertos, garfos) |
| empate −1 ao ganhar | `ahead_draws=true ahead_value=0.62 draw_ahead=-1` | pela avaliação dele (valor do melhor lance), não pelo material |
| simplificar | `simplify_weight=0.05` | trocas que não perdem nada quando está à frente |
| **momentos críticos** | `critical_depth=6 critical_gap=0.0032 critical_spread=0.0163` | profundidade 6 quando os 2 melhores estão quase iguais ou as 5 cabeças da rede discordam (~27% dos lances) |
| **fim de jogo** | `endgame_pieces=8 endgame_mate=6 endgame_depth=7 endgame_nodes=400000` | com ≤ 8 peças: mates até 6 lances (sem o limiar do rei) e cálculo a 7 (promoções forçadas incluídas) |
| **horizonte dos 50 lances** | `fifty_horizon=10` | a ganhar, nos últimos 10 meios-lances antes do limite, um lance que não reinicia a contagem é empate (−1) |
| **valores por fase** | `phase_values=...` (abertura; meio-jogo; final) | medidos com `claudio/valores_fase.py` em 11 324 jogos dele (regra K/D do Martim, por fase): ver `results/valores-fase-claudio.json` |
| **livro de experiência** | `book=claudio/experiencia.json book_weight=0.3 book_min=8 book_plies=16` | **por gerar** (ver abaixo) |

No perfil isolado anterior: ~124 ms por lance (mediana 57) com tudo, contra 65 ms só com a tendência. No BO30 real, o G16 gastou **149,3 ms/decisão**, contra **30,7 ms** do G15 treinado. A partida completa demorou 4 min 26 s com 2 workers.

Valores por fase medidos (P = 1): abertura C 1,52 B 1,65 T 5,48 D 7,52 (poucas mortes de torres e
damas: pouco fiável); meio-jogo C 1,64 B 2,18 T 3,65 D 4,97; **final C 5,59 B 6,96 T 11,0 D 22,0**
(nos finais morrem muitos peões, por isso o peão vale pouco face às peças - pode levá-lo a dar
peões no final: medir com e sem).

O G16 completo já foi medido em BO15 e BO30 contra o G15 treinado. Ainda falta medir cada módulo
sozinho, comparar o G16 com os módulos desligados e testar contra Fairy 1700 e Stockfish 1800
(`vs_engine.py`, secção 7), **com os resultados separados por cor** (o campo `white` de cada jogo).
Exemplo do G16 só com a tendência:
`claudio/claudio-g16.pt@critical_depth=0,endgame_pieces=0,fifty_horizon=0,phase_values=`.

### Livro de experiência (a regra do Martim)

Um lance de abertura é **Bom** num jogo quando **6 lances depois** (12 meios-lances) a posição está
**igual ou melhor** para quem o jogou - a rede dele avalia-a pelo menos tão bem do lado de quem
jogou como do lado do adversário - **E** essa cor ganhou o jogo. O livro guarda, por posição e por
lance, (jogos, jogos Bons), brancas e pretas separadas (cada posição é de quem está a jogar).

1. Gerar (jogos do G16 contra ele próprio a partir da posição inicial, com um pouco de variedade
   nos primeiros 12 meios-lances):

       python claudio/experiencia.py --checkpoint claudio/claudio-g16.pt --games 3000 --temperature 0.005 `
           --set tactic_depth=3 --set critical_depth=0 --workers 6

   Escreve `claudio/experiencia.json` (o livro) e `claudio/results/experiencia-relatorio.json`
   (os jogos e as tabelas), e mostra: de brancas, cada primeiro lance (Bom %, WR brancas, WR pretas);
   de pretas, cada resposta (Bom %, WR pretas). **Isto é trabalho de CPU** (jogar = cálculo C++ e
   avaliações pequenas da rede): mais workers/núcleos aceleram, a GPU quase não. Na nuvem (4 núcleos)
   não chegou a 100 jogos em 12 min: conta com várias horas para 3000 com 6 workers.
   `--temperature`: 0,03 era demais (jogava 1.e3, 1.Na3...); 0,005 varia só entre lances quase iguais.
2. Refazer o livro com outra regra sem voltar a jogar: `python claudio/experiencia.py --rebuild`
   (relê os jogos de `experiencia-relatorio.json`; `--later 12` = 6 lances).
3. Usar: `--set book=claudio/experiencia.json --set book_weight=0.3` (ou pô-lo no checkpoint).
   Cada lance com ≥ `book_min` jogos ganha `book_weight × (taxa de Bom − média da posição)`,
   a taxa puxada para a média enquanto há poucos jogos. Só nos primeiros `book_plies` meios-lances.
4. Medir a partir da **posição inicial** (as medições normais começam com 4 lances aleatórios e o
   livro quase nunca seria usado): `vs_engine.py ... --random-plies 0`, com e sem `book_weight`.

### Resultado e tempos do BO30

O BO30 terminou com 13 vitórias, 2 empates, 8 derrotas e 7 partidas inacabadas para o G16; o G15
treinado fez 8 vitórias, 2 empates, 13 derrotas e 7 inacabadas. Os tempos acumulados foram 14,27 s
por jogo para o G16 e 2,92 s para o G15. O G16 levou **4,9× mais por decisão**. Nas 12 partidas
com 250 ou mais meios-lances, subiu para 170,9 ms/decisão, coerente com as pesquisas mais fundas
que o G16 ativa em posições críticas e no final. O torneio guarda os tempos por partida em
[`g16-vs-g15-100-bo30.jsonl`](results/matches/g16-vs-g15-100-bo30.jsonl) e a tabela em
[`g16-vs-g15-100-bo30.md`](results/matches/g16-vs-g15-100-bo30.md). O tempo não foi separado por
módulo, por isso não há uma percentagem exata de custo para cada um.

### Pipeline para as próximas batalhas

Os comandos, os 15 participantes registados, a retoma e os resultados estão em
[`battle-pipeline/README.md`](battle-pipeline/README.md). O comando mais usado é:

    .\claudio\battle-pipeline\battle.ps1 -A G16 -B G15Treinado -Games 30

O runner usa 2 workers de prioridade baixa por omissão; os resultados ficam na pasta ignorada
`claudio/battle-pipeline/runs/`. A otimização C++ validada e as medições ABBA estão em
[`docs/claudio-performance.md`](../docs/claudio-performance.md).

## 6. Treino na GPU (o que o Martim quer fazer agora)

Na nuvem o treino com `--learn outcomes --fresh` era limitado pela aprendizagem na CPU
(~5 s por passo, 99,7% do tempo): só fez 200 jogos. Na GPU deve ir muito mais depressa
(`train_grande.py` usa `cuda` automaticamente para o learner; os atores jogam na CPU).

Comando (PowerShell, a partir da raiz do repositório):

    python claudio/train_grande.py --start claudio/claudio-campeao.pt --anchor-to claudio/claudio-campeao.pt `
        --out-dir claudio/train-campeao --learn outcomes --fresh --anchor 1e-3 --lr 3e-5 --updates 2 `
        --opponent self --set tactic_depth=3 --eval-every 1000 --eval-games 300 --eval-opponent fairy1700 `
        --save-every 1000 --hours 6 --workers 6

- `--learn outcomes --fresh`: aprende com o resultado dos jogos **dele** (não carrega a replay
  antiga do Timdgei, que foi jogada por outro jogador).
- `--anchor 1e-3 --anchor-to …`: depois de cada passo puxa os pesos de volta para o campeão (os
  treinos antigos estragavam o g0 por esquecerem e mudarem a escala do valor).
- `--opponent self`: contra uma cópia congelada do ponto de partida (orgânico).
- `--set tactic_depth=3`: jogos de treino rápidos. **Atenção:** os checkpoints gravados levam
  `tactic_depth=3` nas definições → avaliar sempre com `--set tactic_depth=5` (vs_engine) ou
  `@tactic_depth=5` (grupos_finais).
- Para continuar noutro bloco: `--start claudio/train-campeao/claudio-grande-final.pt`
  mantendo `--anchor-to claudio/claudio-campeao.pt`. Parar a meio: criar `claudio/train-campeao/STOP`.
- A avaliação interna (300 jogos, sem aberturas aleatórias) é ruidosa (±5 pp): usar só para
  escolher o melhor checkpoint; a medição a sério é a da secção 7.

## 7. Como medir (sempre emparelhado, mesmas aberturas)

    # contra motores (comparar com os ficheiros results/engine-nuvem-*.json do mesmo seed)
    python claudio/vs_engine.py --player claudio --checkpoint X.pt --set tactic_depth=5 --engine fairy --elo 1700 --games 600 --workers 6 --seed 1700 --label nome
    python claudio/vs_engine.py --player claudio --checkpoint X.pt --set tactic_depth=5 --engine sf --elo 1800 --games 400 --workers 6 --seed 1800 --label nome

    # entre Claudios (redes ou definições diferentes)
    python claudio/grupos_finais.py --label nome --games 50 --player A=claudio/claudio-campeao.pt --player B=X.pt@tactic_depth=5
    python claudio/grupos_finais.py --label nome --champions claudio/results/torneio-p3.json --player treinado=X.pt@tactic_depth=5

    # juiz (Stockfish só juiz): perda média em cp em 300 posições, com erro emparelhado
    python claudio/judge_positions.py --label nome --models g0=claudio/claudio-grande-g0.pt b=claudio/claudio-campeao.pt@scan_depth=0

A diferença emparelhada jogo a jogo (mesmo seed) e o seu erro: ver o script em ESTADO.md /
copiar de `results/engine-nuvem-*.json` (campo `list`, chave `game`). Com 600 jogos o erro é
~±5 pp; à melhor de 5 ou de 10 a sorte pesa muito (um campeão de bo10 perdeu nos 200 jogos/par).

## 8. Próximos passos sugeridos

0. Medir cada módulo do G16 sozinho e comparar com a configuração completa; gerar e medir o livro.
1. Medir `claudio-campeao.pt` e o G15 contra Fairy 1700 (600) e Stockfish 1800 (400), contra os
   números do `forte` (32,7% / 15,2%). Ganhar aos outros Claudios não garante ganhar a motores.
2. Medir cada módulo novo sozinho (scan, empate −1, simplificar) da mesma forma.
3. Treinar na GPU (secção 6) e comparar o treinado com o não treinado (secção 7).
4. Juntar o melhor dos dois: a rede treinada com as definições do G15
   (`X.pt@safety_weight=1.5747,tactic_top=10,tactic_rounds=7,mate_gate=0.25634,king_weight=0.02288,tactic_depth=5`).

## 9. Notas práticas

- Testes do Claudio: `python -m pytest claudio/test_claudio.py claudio/test_search_learning.py claudio/test_winners.py claudio/test_review.py -q`.
- Depois de mexer no C++: recompila (`claudio/fast`, `python setup.py build_ext --inplace --force`) e corre
  `python -m pytest claudio/test_fast_exactness.py -q` (tem de dar exatamente o que o G16 original dava). Velocidade:
  `python claudio/bench_velocidade.py --a git:d5e6fda --b claudio/fast` (3,8× no G16, ver `docs/claudio-performance.md`).
- `python -m pytest -q` (repositório) tem uma falha antiga, não do Claudio:
  `tests/test_model.py` reclama dos 1/3/3/5/9 em `tonkai/cognitive_agent.py:945`.
- Na nuvem: `claudio/setup_cloud.sh` prepara tudo em Linux; o modelo KS e o `evolution.json`
  ficaram no PC (o treino usa a cópia do KS dentro do checkpoint; o `arena.py` passa sem o
  `evolution.json`).
