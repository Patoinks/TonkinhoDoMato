# TonkAI — tempo, qualidade e sacrifícios

Análise de 22/09/2026 sobre `C:\Users\Martim\Desktop\Github\TonkAI`, projeto de xadrez em Python/PyTorch. A fase de análise abaixo preservou o código de produção e checkpoints; os scripts `perf_results/audit_*20260922*` documentam essas experiências isoladas.

Atualização após autorização para implementar: a codificação em lote e a remoção da cópia redundante/grafo desnecessário do learner foram integradas. Um candidato inicial mediu **+6,25% de jogos/s**, com partidas, pesos, otimizador, RNG e métricas idênticos. O agrupamento adicional de métricas foi retirado após uma medição isolada mostrar benefício incremental pequeno e variável. A implementação final e as respetivas medições estão em `PERFORMANCE.md` e `perf_results/implementation_*`. Também foi acrescentado o registo da vitória mais longa, sem mudar decisões ou aprendizagem. Os checkpoints existentes continuam preservados. As secções seguintes mantêm as observações da auditoria inicial; o encoder e as duas pequenas simplificações já passaram de propostas a código de produção.

O ensaio final da versão simplificada mediu **12,849 → 13,675 jogos/s (+6,43%)**, novamente com equivalência de partidas, pesos, otimizador, RNG e métricas. Ver `perf_results/implementation_all_20260922T155237_2564cfc8.json`. O PGN da vitória mais longa será guardado automaticamente em `history/longest_external_win.pgn` quando ocorrer uma vitória nas próximas partidas locais contra TinyOpponent ou humanos.

## Conclusão

O objetivo útil é **melhor aprendizagem por minuto de treino**. Há margem para acelerar mantendo a rede, os alvos e as amostras. O modelo já tem uma cabeça de valores contextuais das peças; a avaliação da posição ainda está numa fase inicial. Conforme a orientação do utilizador, as derrotas são uma parte natural dessa aprendizagem: não são motivo, por si só, para facilitar o adversário, impor valores ou alterar os resultados usados no treino.

Prioridade imediata: otimizações de execução equivalentes, mantendo o percurso de aprendizagem. Dois processos de partidas foram a melhor configuração de velocidade medida; depois disso, o gargalo passa a ser o treino, especialmente preparação de dados e transferências. Os diagnósticos de qualidade abaixo servem para acompanhar a evolução, não como exigência de intervenção nesta fase.

## Medições novas

Ambiente confirmado: RTX 5060 Ti, Python 3.13.14, PyTorch 2.11.0+cu128, NumPy 2.5.2, seis threads PyTorch no processo principal.

Cada configuração começou com os mesmos pesos de `deployed_5020games.pt` e o mesmo corpus de 70 partidas. Duas repetições, ordem invertida, 15 partidas de aquecimento e 120 medidas por janela, sempre com treino. CPU para inferência, GPU para aprendizagem, temperatura 0,1, 1 atualização/2048 amostras por jogo, adversário fixo depth=2/nodes=1000/randomness=0,35/quiescence ligada, refresh=20. Tempos excluem arranque dos processos, aquecimento, encerramento e escrita de checkpoints. Partidas e ordem de aprendizagem divergem com múltiplos processos.

| Processos de partidas | Jogos/s, duas repetições | Candidatos avaliados/s | Média de meios-lances/jogo |
|---|---:|---:|---:|
| 1 | 12,46–12,91 | 18.548–19.216 | 50,59 |
| 2 | 27,45–31,32 | 43.244–49.211 | 53,42–53,71 |
| 4 | 26,13–26,39 | 38.714–40.252 | 50,30–51,44 |

São 720 partidas medidas: **0 vitórias, 2 empates, 718 derrotas**. Servem para medir execução nesta carga, não para demonstrar que o modelo ficou melhor. Não extrapolar estes jogos de cerca de 50 meios-lances para um currículo de partidas de 100–200 meios-lances.

Com dois processos, `training` consumiu 94,7–98,0% do tempo total medido; publicação de pesos, aproximadamente 0,12–0,14%. Acrescentar processos ou publicar pesos menos vezes dificilmente resolve esse gargalo. Quatro processos aumentaram o custo de inferência agregado e não melhoraram a produção nesta experiência.

Artefactos: `perf_results/audit_20260922.py` e `.json`. Os hashes dos checkpoints foram iguais antes/depois.

## Onde ganhar velocidade preservando o modelo

As otimizações anteriores já estão instaladas: `pack_many`, amostragem apenas dos candidatos válidos, reutilização dos arrays dos testes de remoção de peças, inferência em CPU e processos de partidas. Não devem ser contabilizadas novamente como propostas novas.

**Ganho novo demonstrado no ciclo completo:** o protótipo de codificação em lote passou de **12,75 para 13,62 jogos/s em média (+6,79%)** com um processo. Foram quatro ensaios na ordem original/protótipo/protótipo/original, 60 partidas medidas cada, oito de aquecimento, mesmo corpus e pesos iniciais. Todos produziram os mesmos 3209 meios-lances, 91.926 candidatos, 60 atualizações, hashes de partidas e parâmetros finais, e estado RNG. É equivalência comprovada nestes ensaios; não uma promessa sobre toda a execução futura. O protótipo continua isolado em `perf_results/audit_encoder_loop_20260922.py`, com resultados no JSON homónimo.

Este ganho não se deve somar automaticamente ao ganho dos processos paralelos: quando o learner já ocupa quase todo o tempo, acelerar as partidas pode sobretudo reduzir consumo de CPU.

| Prioridade | Alteração candidata | Evidência e limite |
|---|---|---|
| 1 | Preparar e transferir os lotes de treino com menos cópias | `replay.py:123` expande entradas compactas para float32 na CPU; `trainer.py:169` transfere-as. Cada posição passa de 99 para 3128 bytes. Transferir compactado e expandir na GPU reduz esse payload em 31,6 vezes; isso **não** significa 31,6 vezes mais jogos/s. Precisa de medição e comparação exata dos inputs, incluindo relógio. |
| 2 | Adiar as leituras de métricas da GPU e agrupar transferências | `trainer.py:152` lê os valores das peças antes da restante aprendizagem; `:202` converte três losses separadamente. As métricas não determinam a jogada nem o gradiente. Há margem para mover estas leituras para um ponto comum, preservando os valores e a ordem. Ganho ainda não medido. |
| 3 | Codificar todos os tabuleiros candidatos em lote | Protótipo mantém `python-chess.push/pop`, ordem e regras; expande bitboards numa operação NumPy por lote. Mediana **0,1655 → 0,1237 ms/lote**, menos **25,2%** nesse componente. Foram iguais byte a byte 51.456 candidatos, em 2502 comparações, incluindo cores, relógios, roque, en passant, promoções e históricos. Benefício total depende de a CPU das partidas limitar o treino. |
| 4 | Eliminar pequenas cópias/grafos desnecessários | `trainer.py:133` usa `x[rows].clone()` embora a indexação avançada já produza uma cópia; `:97` constrói o grafo de uma utilidade que depois é destacada do gradiente. São candidatos menores, a aceitar apenas após comparar gradientes/pesos. |

A expansão na GPU deve usar a mesma representação float32 do relógio, por exemplo uma tabela derivada do NumPy atual, para evitar diferenças de arredondamento. Alterar a forma de sortear peças pode manter a distribuição mas mudar o fluxo aleatório; não entra na categoria de equivalência exata.

A documentação oficial do [PyTorch sobre desempenho](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide) identifica conversões CPU/GPU e extrações de escalares como pontos de sincronização. A [documentação de multiprocessing da versão 2.11](https://docs.pytorch.org/docs/2.11/notes/multiprocessing.html) explica a competição entre processos e threads. São fundamentos para os candidatos, não medições deste projeto.

## O que ainda não prova “sem perder qualidade”

O default passou de quatro atualizações Adam com 512 amostras para uma com 2048. A contagem de amostras é igual, mas o percurso de otimização não é. `perf_equivalence.py` valida a configuração antiga em cinco partidas/20 updates, não a força do novo calendário.

Com vários processos, as partidas usam pesos de versões anteriores e chegam em ordem variável. Isso pode compensar em força por minuto, mas só uma experiência de aprendizagem responde. Os checkpoints antigos ainda retomam o calendário antigo e inferência no learner, salvo overrides explícitos (`config.py:74`).

O currículo e o replay não são preservados no checkpoint; ao retomar, o currículo é recriado. A dificuldade efetiva também não aparece no histórico JSONL, pois é aplicada diretamente ao adversário. Guardar rung, janela de resultados e versão dos pesos por partida tornaria as comparações e a continuação do treino mais fiáveis.

## Estado inicial da aprendizagem

**Poucas vitórias.** O histórico disponível termina em 4000 jogos com **1 vitória / 779 empates / 3220 derrotas**. `latest.pt` é um checkpoint de 2500 jogos, com **1/466/2033**, não o fim desse histórico. Todas as posições recebem o resultado final (`trainer.py:57`); com este desequilíbrio, a rede tem muito poucos exemplos de conversão de vantagem em vitória. O valor das peças deriva dessa mesma rede e herda a limitação.

**Um mate imediato pode perder para uma jogada normal.** Na posição `r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4`, `python-chess` confirma que `Qxf7#` termina o jogo:

| Checkpoint | Rank do mate entre 42 lances | P(derrota) prevista para o mate | Lance escolhido |
|---|---:|---:|---|
| latest, 2500 jogos | 15 | 90,09% | Qe2 |
| deployed, 5020 jogos | 35 | 99,9986% | Qg4 |

Noutro final KQvK ambos encontram um mate, mas continuam a atribuir-lhe probabilidades incorretas. O teste inclui três FENs por checkpoint, sendo dois simétricos: é um diagnóstico concreto, não uma estimativa estatística da força geral.

A seleção em `agent.py:237` consulta a rede para os filhos; o resultado terminal só é aplicado depois de jogar (`:200`). Usar o resultado exato dos filhos terminais seria uma possível alteração futura de comportamento, mas fica fora da recomendação imediata: o diagnóstico permite observar se o modelo aprende estes mates por experiência. Nenhum resultado ou lance foi imposto.

Para acompanhar compensação de longo prazo, observar a evolução das previsões e escolhas com mais treino. Se futuramente houver estagnação persistente, podem estudar-se currículo, mistura de adversários ou posições iniciais variadas como experiências separadas. A escassez atual de vitórias, isoladamente, não estabelece essa estagnação nem justifica mudar já o treino.

## Valores por peça e o risco pretendido

A arquitetura já calcula, para cada peça, aproximadamente:

`contribuição = U(posição) − U(posição com essa peça retirada)`

O tipo, casa, lado e contexto completo entram no PieceHead. Um cavalo bloqueado pode, portanto, contribuir menos que um peão passado. A jogada continua a ser decidida pela avaliação da posição inteira; não é necessário que os valores individuais somem essa avaliação.

Há três limitações:

1. Os valores estão na escala `U = P(vitória) + 0,1 P(empate) − P(derrota)`, **não em peões**. “−0,5” nesta escala não significa perder meio peão. Uma escala visual equivalente a peões exigiria calibração e não deveria tornar-se uma tabela rígida de treino.
2. Retirar uma peça só do vetor cria uma experiência hipotética, por vezes fora das posições vistas em treino, e pode deixar flags de roque incoerentes. A diferença mede a opinião da própria rede, não uma contribuição causal comprovada. Melhorar primeiro o ValueHead e verificar estabilidade destes alvos é mais importante que aumentar o peso da loss das peças.
3. Os reis são excluídos do treino deste valor, embora a função de apresentação devolva números para eles. As médias atuais também misturam próprias e adversárias, dificultando a interpretação.

O registo de sacrifícios em `agent.py:338` apenas verifica uma captura e compara o valor da peça que captura com a capturada. Não confirma que a primeira foi perdida, ignora ofertas sem captura e compara contribuições calculadas da mesma perspetiva como se fossem valores dos respetivos donos. Uma captura gratuita pode entrar no contador.

Recomendo separar **oferta potencial**, **peça efetivamente perdida** e **resultado da continuação**. Guardar a sequência real, contribuição de cada lado, W/D/L antes/depois e resultado final. Para representar a ameaça adversária na perspetiva TonkAI pode usar-se o negativo da sua contribuição; isso não equivale necessariamente à utilidade do adversário, devido ao peso positivo dos empates. Nunca usar o contador de sacrifícios como recompensa.

No teu exemplo, há duas situações diferentes:

- O cavalo vale convencionalmente 3, mas naquele contexto a compensação faz a troca ser boa: o modelo deve poder escolhê-la por melhorar a posição, sem qualquer penalização fixa por material.
- A troca continua ligeiramente pior na avaliação global, mas oferece mais hipóteses de ganhar: isso é uma preferência explícita por risco. Pode usar um limite `U(melhor) − U(candidato) ≤ ε` e, dentro desse conjunto, favorecer maior `P(vitória)`. `ε` deve estar na escala calibrada da rede, não ser confundido com meio peão.

Atualmente existe apenas exploração pela temperatura; aumentar a temperatura distribui aleatoriedade pelos lances e não cria risco calculado. Guardar W/D/L torna possível definir preferências de risco, mas a seleção atual reduz-as a uma utilidade média. A distinção entre retorno esperado e risco é também tratada neste [trabalho sobre aprendizagem com distribuições de retorno](https://arxiv.org/abs/2212.14743).

Aceitar menor utilidade esperada para tentar ganhar mais pode ser um estilo desejado. Não pode simultaneamente ser prometido como preservação automática do desempenho médio. Perder material com compensação e aceitar uma posição objetivamente pior são decisões diferentes.

## Critério para aceitar alterações

- Otimizações de execução: mesmos inputs compactados, candidatos, rótulos, RNG, gradientes e pesos em ensaios determinísticos; depois medir tempo total com replay aquecido.
- Mudanças de aprendizagem ou risco: comparar com igual tempo de treino e também igual orçamento de amostras, várias sementes, adversários fixos, aberturas e cores emparelhadas. Reportar W/D/L e incerteza, além de calibração, mates e posições de sacrifício com continuação validada.
- Acompanhar jogos/s, meios-lances/s, candidatos/s, duração das partidas, frequência de vitórias e tipos de empate. Candidatos/s não significa amostras únicas aprendidas/s.
- `evaluate.py` já congela os pesos, mas ainda não executa esta comparação completa. Top1 do Predict mede previsão do adversário, não força da política.

Ordem recomendada: codificação em lote com equivalência já medida; otimizações equivalentes de cópias e transferências do treino; comparação de dois processos ao longo da aprendizagem; leitura coerente das peças e sacrifícios; posteriormente tolerância explícita ao risco. Preservar o espaço para aprender com as derrotas.

Reprodução: `python -B -u perf_results/audit_20260922.py`; `python -B perf_results/audit_encoder_20260922.py --benchmark`; `python -B -u perf_results/audit_encoder_loop_20260922.py`; `python -B perf_results/audit_quality_20260922.py`. Os scripts de treino só alteram modelos em memória e os seus resultados de auditoria.
