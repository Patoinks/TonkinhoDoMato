# Claudio — estado e regras (para o Claude que continuar isto)

O Claudio é a IA de xadrez do Claude dentro do TonkAI (projeto do Martim). Nasceu de um
duelo com o "Gepeto" (a IA do ChatGPT). Isto é o resumo do que a sessão anterior sabia;
a memória dela ficou no PC do Martim, por isso tudo o que importa está aqui.

## Regras do Martim (valem sempre)

- **Orgânico.** Nada de valores de peças à mão (1-3-3-5-9 = "0/10 organicidade"). O
  Stockfish e o Fairy-Stockfish **só** são adversários ou juízes, nunca professores.
- **Winrate = vitórias / jogos.** Os empates não contam como meia vitória.
- **Duelo com o Gepeto:** 60% tempo por jogo + 40% winrate. Não leias o código do Gepeto
  (há commits dele no histórico: `group_stage`, a pasta `Gepeto` local). Não espreitar.
- **Nunca bloquear a conversa.** Tudo o que demora mais de ~1 min vai para segundo plano;
  responde logo e volta quando acabar.
- No máximo 6 workers, prioridade baixa. Não relançar nem parar treinos sem ele pedir.
- **Não inventes passos extra** e não mudes a filosofia do projeto sem perguntar.
- Responde ao que ele perguntou, à letra. Ex.: "os primeiros 500 de cada" = os primeiros
  500 jogos contra cada adversário, não checkpoints. Ele irrita-se, e com razão, quando
  a resposta é a outra coisa.
- Português de Portugal, direto, com tabelas quando há números.
- Push/publicação só quando ele pedir. Nunca `git add -A` / `git add .` (há outros agentes:
  ver `WORKING.md`).

## Arrancar numa máquina nova (nuvem = Linux, sem GPU)

    bash claudio/setup_cloud.sh

Instala os pacotes, compila os 3 módulos C++ e compila os motores nas mesmas versões
do PC (Stockfish 11, Fairy-Stockfish 14) para `engines/linux/`. `claudio/paths.py` escolhe
os motores por sistema (ou `CLAUDIO_STOCKFISH` / `CLAUDIO_FAIRY`). O treino usa a GPU se
houver; senão CPU. **Corre em Linux desde 2026-10-06** (máquina da nuvem: 4 núcleos, sem GPU,
`download.pytorch.org` bloqueado, por isso o torch vem do PyPI; ~15 min). O modelo KS e o
`evolution.json` ficaram no PC: o treino usa a cópia do KS dentro do checkpoint do Claudio e
o `arena.py` passa sem o `evolution.json` (só o Timdgei precisa dele, com os genes de torneio).

O Claudio corre numa cópia congelada do código do TonkAI (`claudio/base`, com o seu
`tonkai_fast`), nunca no `tonkai/` partilhado (`paths.py` trata da ordem do sys.path).

## Modelos no repositório

| Ficheiro | O quê |
|---|---|
| `claudio/claudio-grande-g0-forte.pt` | **o melhor até hoje** (2026-10-07): os pesos do g0 com respostas + `tactic_depth=5` (~60 ms/lance) |
| `claudio/claudio-grande-g0-respostas.pt` | os pesos do g0 com `reply_top=6 reply_mode=expect` (rápido, ~14 ms/lance) |
| `claudio/claudio-grande-g0.pt` | a melhor rede. Rede crescida (Net2Net, 254k→810k parâmetros) |
| `claudio/claudio.pt` | o Claudio do duelo com o Gepeto |
| `claudio/train-1550/claudio-grande-last.pt` | "Este (treino 1550)", usado na fase de grupos |
| `checkpoints/obj-max-continuo/latest.pt` | o Timdgei (o TonkAI de referência; o treino lê a config e o replay dele) |

Os ~180 checkpoints de treino ficaram no PC (os números estão em `claudio/results/`).

## Como o Claudio joga (`claudio_agent.py`)

Intuição da rede (valor de cada posição alcançável, com memória do jogo) + troca de
peças exata + cálculo na raiz (PVS sobre os 4 melhores, 4 meios-lances, em C++) +
defesa contra mate + ataque de mate a 3 lances quando o seu próprio modelo de
segurança do rei diz que o rei inimigo está exposto + simplificar quando está à frente.
Valores das peças medidos por ele: P 1, C 2,13, B 2,73, T 3,98, D 9,06.

Opcional (desligado por omissão, 2026-10-06): **respostas** (`reply_top`, `reply_mode`,
`reply_weight`). Os n melhores candidatos são julgados também depois de cada resposta do
adversário, com a própria rede: a pior resposta (`min`) ou a média pesada pela sua previsão
do lance do adversário (`expect`, a cabeça `predict_head`). Só muda a ordem entre esses
candidatos (o nível médio deles fica). Ligar: `--set reply_top=6 --set reply_mode=expect`.

## Ferramentas principais

| Script | Para quê |
|---|---|
| `train_grande.py` | treino (adversários, escadinha, avaliação fixa a cada N jogos com melhor checkpoint, `--colour`, `--learn outcomes/winners/review/notes`) |
| `eval_colour.py` | um checkpoint, uma cor, contra um adversário, sem exploração |
| `judge_positions.py` | o juiz: 300 posições reais, que lance escolhe cada versão e quanto perde |
| `review.py` | revisão de um jogo pelo próprio Claudio (bom/mau por material forçado) |
| `review_lab.py` | compara variantes da revisão contra o juiz (juiz em cache) |
| `vs_engine.py`, `arena.py`, `groups.py`, `duel.py` | jogos contra motores, contra o Timdgei, fases de grupos, duelos |
| `uci.py` | o Claudio como motor UCI |

Testes: `python -m pytest claudio/test_claudio.py claudio/test_winners.py claudio/test_review.py -q`.

## O que se aprendeu

1. **Todo o treino a partir do g0 até agora piorou-o** (escadinha, adaptativo, 1550,
   especialistas por cor, só-notas, revisão). Muda ~metade dos lances sem melhorar a
   qualidade média. Causas medidas: o valor da rede muda de escala/significado contra
   pesos de decisão fixos; o crédito por jogo é ruído; esquece.
2. **O juiz tinha um erro (corrigido):** comparava a posição antes e depois do lance, e
   o Stockfish dá ~60 cp a quem joga. Agora compara o lance jogado com o melhor *da mesma
   posição* (`root_moves`). Números antigos de "perda média" estão inflacionados; as
   comparações entre versões continuam válidas.
3. **A revisão orgânica** (material forçado) a 5 meios-lances apanha 18% das asneiras
   reais (3 meios-lances: 5%; 7: igual a 5 e 20x mais lento). Quando diz "mau" numa
   posição ainda por decidir, acerta quase sempre. Metade das asneiras são posicionais
   (nem o juiz vê material a mudar em 10 meios-lances): só uma rede melhor as vê. As
   consequências (culpa partilhada para trás) ficaram desligadas: davam crédito a lances
   maus pela asneira seguinte do adversário.
4. **Referência do g0 contra Fairy 1450**, 200 jogos limpos: brancas 55,0%, pretas 54,5%.

## Último experimento (2026-09-26, `results/melhora.log`)

Do g0, 1000 jogos por cor contra Fairy 1450, avaliação de 200 jogos a cada 250:

| Versão | Brancas 250/500/750/1000 | Pretas 250/500/750/1000 | Juiz (perda média) |
|---|---|---|---|
| g0 (início) | 55,0% | 54,5% | 82 cp |
| só notas da revisão | 38 / 32,5 / 39,5 / 32,5% | 53,5 / 51 / 41,5 / 46% | 109 cp |
| revisão a 5 meios-lances | 45 / 47 / 39 / 45% | 59,5 / 41,5 / 36 / 39,5% | 86 cp |
| especialistas antigos | — | — | 83 cp |

A hipótese (o valor a copiar o material apagava o sentido posicional) estava errada:
só notas foi o pior. **O g0 continua a ser a melhor rede.** (Contado ao Martim em 2026-10-06.)

## Sessão na nuvem (2026-10-06): respostas com a própria rede

Sem treino nenhum: os 6 melhores candidatos veem cada resposta do adversário pela rede.
Fairy 1450, 600 jogos cada, mesmas aberturas (`--seed 1450`, 4 meios-lances aleatórios,
cores trocadas), `results/engine-nuvem-fairy1450-*.json`:

| Versão | Winrate | V/E/D | ms/lance |
|---|---|---|---|
| g0 | 53,5% | 321/107/172 | 7,3 |
| **g0 + `reply_top=6 reply_mode=expect`** | **63,2%** | 379/77/144 | 13,6 |
| g0 + `reply_top=10 reply_mode=expect` | 61,3% | 368/83/149 | 16,2 |
| g0 + `reply_top=6` (`min`) | 59,7% | 358/88/154 | 13,7 |
| g0 + `reply_top=4 reply_mode=expect` | 59,2% | 355/89/156 | 15,3 |
| g0 + `reply_top=100` (todos, `min`) | 61,7% | 370/71/159 | 23,3 |

Emparelhado jogo a jogo contra o g0 (95%): **6/expect +9,7 ± 5,6 pp (z = 3,4)**; todos/min
+8,2 ± 5,7 (z = 2,8); 10/expect +7,8 ± 5,6 (z = 2,8); 6/min +6,2 ± 5,6 (z = 2,2);
4/expect +5,7 ± 5,5 (z = 2,0). Todas ganham ao g0; 6/expect é a melhor e das mais baratas.
(Os ms/lance das 3 do meio foram medidos com a máquina mais carregada.)

Contra Stockfish 1800 (400 jogos, `results/engine-nuvem-sf1800-*.json`): g0 5,5% (22/77/301),
6/expect 6,75% (27/69/304): +1,2 ± 3,4 pp, sem diferença visível (o g0 quase nunca ganha ali).

Contra Fairy 1700 (600 jogos, `--seed 1700`): g0 17,5% (105/129/366), 6/expect **23,5%**
(141/109/350): **+6,0 ± 4,7 pp (z = 2,5)**. O ganho mantém-se contra um adversário mais forte.

Peso das respostas (`reply_weight`, Fairy 1450, 6/expect): 0,5 → 57,2% (+3,7 ± 5,3 vs g0);
**1 → 63,2%**; 2 → 53,3% (−0,2 ± 5,5 vs g0; −9,8 ± 5,5 vs peso 1). O peso 1 é o certo.
Menos derrotas por mate (172 → 144) e menos repetições (90 → 56). O g0 aqui deu o mesmo
que no PC (53,5% vs 55/54,5%), por isso a nuvem mede igual.

O juiz (300 posições, `results/judge-nuvem-respostas.json`) quase não vê diferença:
6/expect −3,8 ± 4,1 cp, todos/min −3,5 ± 4,0 cp, 6/min +0,2 ± 3,3 cp; asneiras ≥200 cp
12,7% → 10,7–11,3%. Os jogos veem-na; o juiz, com 300 posições, não chega para isto.
O juiz agora mostra a diferença emparelhada e o erro (`vs g0 ... +- ...`) e aceita
definições por versão: `nome=ficheiro.pt@reply_top=6,reply_mode=expect`.

Custo: ~+6 ms por lance (quase tudo é a rede nas ~200 posições de resposta), conta no
duelo de tempo. Por decidir com o Martim: ligar por omissão / num checkpoint próprio.

## Ideias ainda não testadas (não lances sem ele pedir)

- Taxa de aprendizagem muito menor / menos updates por jogo, parando no primeiro sinal
  de descida (a avaliação a cada 250 já guarda o melhor).
- Aprender só quando a revisão e o resultado concordam (lance mau num jogo perdido).

## Rumo a 1800 (2026-10-07, nuvem)

Meta do Martim: rating 1800 (referência: Stockfish 11 a `UCI_Elo` 1800, 2 ms por lance). Medições
contra Fairy 1700 (mais sensível: o g0 ganha 17,5%), 600 jogos, mesmas aberturas (`--seed 1700`),
checkpoint `claudio-grande-g0-respostas.pt` + `--set`:

| Versão | Winrate | vs base (95%) | ms/lance |
|---|---|---|---|
| g0 (sem respostas) | 17,5% | | 7 |
| **base** (respostas) | 24,5% | | 14 |
| + `mate_defence=3` | 21,7% | sem ganho | 18 |
| + `tactic_depth=5` | 32,7% | **+8,2 ± 5,0** | 59 |
| + `tactic_depth=6` | **35,8%** | **+11,3 ± 5,0** (vs td5 +3,2 ± 5,0) | 291 |
| + `tactic_depth=5 reply_depth=3` | 17,8% | −14,8 ± 4,9 vs td5: **o 3.º meio-lance estraga** | 128 |

`tactic_depth` (cálculo de material depois do lance pensado) é o que mais rende; 7 já custa ~1 s por
lance. O `reply_depth=3` (opção nova) piora: provavelmente conta o material duas vezes (as trocas
já entram na raiz) e o máximo sobre ~35 respostas próprias é otimista. Fica desligado.

Contra Stockfish 1800, as mesmas 10 aberturas (`--seed 1800`, ±30 pp, só indicativo): g0 0/2/8,
respostas 1/3/6, `tactic_depth=7` + `reply_depth=3` 0/1/9, **`tactic_depth=7` 2/0/8** (1,2 s/lance).
**400 jogos contra Stockfish 1800** (`--seed 1800`, mesmas aberturas):

| Versão | V/E/D | Winrate | Pontos (empate ½) | ~Elo (escala SF 1800) |
|---|---|---|---|---|
| g0 | 22/77/301 | 5,5% | 15,1% | ~1500 |
| respostas | 27/69/304 | 6,75% | 15,4% | ~1500 |
| **respostas + `tactic_depth=5`** | **61/86/253** | **15,2%** | **26,0%** | **~1620** |

+8,5 ± 4,3 pp de vitórias vs respostas (z = 3,9). Para 1800 faltam ~180 Elo (50% dos pontos).
Esta versão está em `claudio-grande-g0-forte.pt`.

Afinações em cima de respostas + `tactic_depth=5` (Fairy 1700, 600 jogos; td5 = 32,7%):
`tactic_top=6` 33,0% (+0,3), `reply_top=10` 32,7% (0), `exchange_depth=7` 32,3% (−0,3),
`tactic_rounds=16` 35,0% (+2,3), `tactic_rounds=8` 38,2% (+5,5 ± 5,1, z = 2,1) - mas contra
Stockfish 1800 (400 jogos) `tactic_rounds=8` deu 11,2% (45/84/271) vs 15,2%: −4,0 ± 4,7. Era
sorte (4 testes, um com z = 2 é de esperar). Nenhuma destas fica.

`tactic_depth=6` contra Stockfish 1800 (400 jogos): 13,0% (52/85/263), 23,6% dos pontos, vs td5
15,2%: −2,2 ± 4,5, e 283 ms/lance. Mais fundo que 5 já não rende: o cálculo de material está esgotado.

`leaf_weight` (opção nova: a rede no fim da linha principal do cálculo de material, cortada à vez
do adversário), Fairy 1700, 600 jogos, sobre o forte (32,7%): peso 1 → 18,8% (−13,8 ± 4,7),
peso 0,5 → 31,7% (−1,0 ± 5,4). Piora, como o `reply_depth=3`: **a rede não julga bem posições mais
fundas que 2 meios-lances** (ou no meio de trocas, ou com a memória sem ver a linha). Desligado.

Conclusão (2026-10-07): o cálculo à volta da rede rendeu ~+120 Elo (g0 ~1500 → forte ~1620 na
escala do Stockfish 1800) e parou. O que falta para 1800 é uma rede melhor.

Treino autorizado pelo Martim (2026-10-07, só na nuvem): `train_grande.py --learn search`
(aprende com o próprio cálculo, estilo TD-Leaf; ver o docstring). Contra Fairy 1700, `--lr 5e-5
--anchor 1e-3 --updates 2 --search-rows 512`, parou aos 3100 jogos (limite de tempo da sessão).
Avaliações (300 jogos, sem aberturas aleatórias): 22,3 / 18,0 / 19,3 / 24,3 / 21,7 / 20,7% aos
500..3000. O melhor (2000 jogos), emparelhado com a base: 24,8% vs 24,5% (+0,3 ± 4,8): **nem
melhor nem pior**. A rede aprende os alvos (erro 0,0039 → 0,0014), mas o valor a 1 meio-lance
já concorda com o de 2 meios-lances: ensinar isso não traz conhecimento novo. Os pesos ficaram
na nuvem (`claudio/train-busca/`, não versionado).

## Tempo por profundidade (2026-10-08, nuvem)

`tactic_depth` (meios-lances calculados depois do lance que pensa jogar), resto igual ao forte,
as mesmas 100 posições reais (40 na profundidade 8), 1 núcleo, ms por lance:

| tactic_depth | média | mediana | 90% | pior | Fairy 1700 (600 j.) | Stockfish 1800 (400 j.) |
|---|---|---|---|---|---|---|
| 1 | 13 | 10 | 23 | 45 | | |
| 2 | 10 | 10 | 13 | 21 | | |
| 3 (omissão) | 12 | 11 | 18 | 28 | 24,5% | 6,75% |
| 4 | 22 | 16 | 42 | 140 | | |
| **5 (forte)** | **69** | 53 | 152 | 364 | **32,7%** | **15,2%** |
| 6 | 272 | 125 | 745 | 1746 | 35,8% | 13,0% |
| 7 | 1142 | 695 | 3617 | 5263 | | 2/0/8 em 10 |
| 8 | 2601 | 2888 | 5414 | 6037 | | |

Até 3 o tempo é quase só a rede (~10 ms); cada meio-lance a mais depois custa ~3-4x. Aos 8 o
limite de nós (2M) corta muitas vezes e o cálculo deixa de contar. O ponto ótimo medido é o 5.

## Torneio de mutações, módulos e passagem para a GPU (2026-10-08)

Resumo completo e comandos em [README-CODEX.md](README-CODEX.md). Em curto: o torneio p3
(10 gerações, profundidade 3) deu campeões cada vez mais fortes entre Claudios (G10 > G9 > G8 >
… > original); tendência: mais peso ao material e mais candidatos a fundo. Prolongar a tendência
("gerações artificiais") ajuda até ~G15-G16 e piora a partir do G20. `claudio-campeao.pt` = G10 +
módulos novos (`scan_depth`, `ahead_draws`, `simplify_weight`). Falta medir campeão e G15 contra
motores e treinar na GPU (o learner na CPU da nuvem faz ~5 s/passo).
