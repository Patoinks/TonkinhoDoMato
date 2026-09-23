# Execução da rede e estatísticas desktop — 2026-09-23

As cinco ideias têm caminhos experimentais opt-in, testes e medições. A seleção
para a campanha é **CPU + compilação existente + pesquisa agrupada**. Não ativar
todas as opções em conjunto: três não confirmaram ganhos, e CUDA Graphs ganhou
no update isolado mas perdeu no teste integrado com oito atores.

## Segurança e reprodução

Baseline: commit `09d4d1f16ad5e2eb0fc6fe376d981bc1914bbafc`, já com as alterações
do Claude. `perf_results/execution-fastpaths-20260923/head.zip` guarda o código;
`input.pt` guarda o checkpoint completo (replay, optimizer, RNG e pesos).
SHA-256 do checkpoint: `7c1d34d16929c3542a6af31569dd58dc29b57bda92cbb040ca4117b1c824617e`.
Os ensaios não escreveram no checkpoint original nem publicaram modelos.

O runner `scripts/benchmark_execution_paths.py` lê exclusivamente essa cópia.
Exemplo: `python scripts/benchmark_execution_paths.py --mode baseline --updates 100`.
Executar os modos sequencialmente, sem outro benchmark/treino em simultâneo.
Os resultados JSON ficam no diretório do checkpoint isolado; os pesos/ZIP
permanecem apenas locais. Cada modo de updates aquece 10 passos e mede 100.

## Resultados

| Caminho | Medição | Decisão |
|---|---|---|
| CPU original, 100 updates | 4,303 / 4,268 s | Referência eager; não inclui torch.compile |
| Compactar posições do encoder | 4,110 / 4,290 s | Pequeno ganho não consistente; desligado |
| CUDA eager, 100 updates | 5,531 s | Mais lento; desligado |
| CUDA Graphs, 100 updates | 2,605 / 2,607 s | ~1,64x no update isolado; experimental |
| Ator original, 288 posições | 1,583 / 1,578 s | Referência |
| Primeira camada incremental | 1,604 s | Sem ganho; desligado |
| Pesquisa agrupada, 288 posições | 1,462 / 1,475 s | ~1,08x na avaliação; selecionado |
| Broker, 800 pedidos de 30 posições, mesma versão | 0,279 s vs 0,082 s local serial | Custo de IPC superior à poupança; desligado |
| Broker, versões distintas | 0,334 s vs 0,081 s local serial | Desligado |

Os 100 updates de cada variante processaram **24 375 posições válidas**, com as
mesmas sequências, máscaras e RNG. A perda final ficou muito próxima, mas os
pesos não são bit-identical: diferença máxima ~0,00101 em compactação e ~0,000946
em CUDA Graphs após 10+100 updates. Isto exige distinguir equivalência matemática
de igualdade numérica; não constitui um ensaio de força/Elo a longo prazo.

Pesquisa agrupada e encoder incremental: zero mudanças de argmax nas 288
posições; diferença máxima de scores `1,1920929e-7`; mesmos 2175 nós e 24 decisões
com pesquisa. O teste não cobre todas as posições possíveis.

### Integração com atores

O teste usa oito atores, 64 jogos medidos, mistura 75/20/4/1 e dois updates/jogo,
depois de aquecimento. Os pesos dos atores são fixos para aproximar o trabalho
entre variantes, enquanto o learner recebe experiências e aprende normalmente.
Não é uma campanha real com snapshots sucessivos. Motores UCI limitados por
tempo e ordem de conclusão tornam os percursos não perfeitamente determinísticos.

- CPU eager: 15,097 / 16,835 s; 4,24 / 3,80 jogos/s.
- CPU eager + pesquisa agrupada: 15,486 s; 4,13 jogos/s.
- CUDA Graphs + pesquisa agrupada: 18,431 s antes de mover a inferência das
  experiências para CPU; 19,595 / 17,632 s depois (3,27 / 3,63 jogos/s).
- No último teste GPU: learner 6,152 s; espera por atores 6,638 s; ingestão
  3,988 s; experiências 0,381 s. Na variante CPU agrupada: learner 8,654 s;
  espera 2,455 s; ingestão 3,588 s; experiências 0,354 s.
- Portanto **não extrapolar o 1,64x do update para jogos/s**. O caminho GPU
  permanece disponível para experiências futuras, mas não é escolhido aqui.
- Broker também passou um smoke real com dois processos, oito jogos, 16 updates
  e 5295 pedidos. O ensaio de dois atores não é comparável ao de oito atores.

## O que cada opção preserva

- `--compact-sequences`: compacta entradas dos encoders e repõe o layout
  original. Padding produz `encoder(0)`, não um latent arbitrariamente zero:
  os biases também influenciam diagnósticos/prioridades. Recorrência e cabeças
  continuam com layout original. Não elimina posições reais nem reduz memória.
- `--incremental-encoder`: primeira camada FP32 calculada a partir da posição
  principal e deltas de features. Inclui relógio, roque, en passant e promoções;
  reconstrói a base em cada chamada, sem acumulação entre jogos/pesos.
- `--batch-search`: conserva orçamento, ordem e cobertura dos ramos, resultados
  terminais e segurança do rei. Softmax continua independente em cada ramo.
- `--inference-broker`: agrupa o encoder de atores por hash exato dos pesos;
  snapshots imutáveis e referências por jogo. Memória recorrente fica no ator.
  O serviço atual é CPU; não altera a frequência de atualização de políticas.
- `--learner-device cuda --cuda-graphs`: captura forward/backward recorrentes e
  unroll do target. Perdas dinâmicas, optimizer e replay continuam eager.
  Preserva detach do burn-in e clipping. Não pode combinar-se com compactação
  dinâmica ou `--compile-learner`. O PyTorch emite um aviso de streams durante
  captura; os testes de outputs/gradientes e updates passam, sem suprimir o aviso.
  Experiências curtas usam cópia CPU atual dos pesos, atualizada após mudanças.

Nenhuma opção reduz dimensões, ensemble, horizonte, frequência de experiências,
updates por jogo ou precisão para FP16/BF16. Nenhum adversário foi otimizado.

## App e arranque

O launcher conserva os parâmetros do Claude: oito atores, 100 mil jogos,
75% self-play, 20% Tiny, 4% Fairy nominal 1000 e 1% Stockfish 11 nominal 1500.
`scripts/start_v3_mix.ps1 -BatchSearch` mantém compilação CPU. `-CudaGraphs` é
uma alternativa explícita, não o default. `-Resume` conserva a mesma campanha.

A app mantém score/V/E/D por adversário, ETA, duração, motivos de empate e
gráficos. A nova aba **Todas as estatísticas** mostra todos os campos dos logs,
configuração e resumos da campanha; filtra por nome ou valor. Zero, null e campos
novos são preservados. Campos `execution` mostram opções ativas, tempos,
updates e posições aprendidas na sessão, e memória CUDA quando aplicável.

A aplicação foi reconstruída pelo script existente; o atalho do ambiente de
trabalho aponta para o executável atualizado. Capturas e recibos da validação
ficam junto dos benchmarks. Os resultados por adversário mantêm a distinção
entre self-play e totais externos; os campos crus identificam o último registo.

Validação final: **445 testes passaram em 67,62 s**, incluindo forward/backward
CUDA, refresh dos pesos das experiências, isolamento de versões no broker,
jogadas especiais, pesquisa e leitura de todas as métricas. Um aviso CUDA de
streams permanece documentado acima; esse caminho não será ativado na campanha.


Arranque Windows: a primeira tentativa terminou com erro no probe do MSVC
(`UnicodeDecodeError` no output localizado de `cl /help`). O launcher passou a
configurar `chcp 65001` e a aplicar VSLANG/PYTHONUTF8 depois de vcvars. O probe
`torch._inductor.cpp_builder.get_cpp_compiler()` passou no processo oculto,
com stdout `cl` e stderr vazio. A campanha é retomada do checkpoint de erro,
conservando o jogo já ingerido e o seu replay; não se apaga a primeira sessão.


Validação ao vivo: campanha `cognitive-20260923T030055Z-60fa3fc5`, PID Python
32280, código de arranque `71b1033`. A app instalada passou o smoke com estado
`running`: 476 jogos no primeiro recibo e 701 no segundo, quatro adversários,
354 campos e 11 métricas de execução. A captura `tracker-live.png` mostra o
score ao vivo; `tracker-all.png` mostra a nova aba antes do arranque. A segunda
captura ao vivo estava tapada por outra janela e foi descartada; o seu recibo
JSON continua a documentar os valores reais lidos pelos widgets.
