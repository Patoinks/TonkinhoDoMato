# V3 preparada para o próximo adversário

Esta versão é opt-in. Nenhum processo, peso ou parâmetro do treino V2 em curso
foi substituído. A entrada é `train_cognitive.py`; o formato novo é
`tonkai-cognitive-v3`, com saídas por omissão em `checkpoints/cognitive`.
O Elo do próximo adversário é um argumento obrigatório, não uma decisão automática.

## Os nove módulos selecionados

| Escolha | Implementação e utilização real |
|---|---|
| 1 — Memória associativa | Pesquisa até seis vizinhos num arquivo de 512 experiências. A chave combina estrutura de ocupação, estado das regras e um encoder congelado. Distingue de quem é a vez. A mistura com o valor neural é limitada a 0,12 e depende de semelhança, jogos independentes e confiança. |
| 3 — Contrafactuais | A cada 20 jogos, regressa a uma decisão incerta, simula a ação original e uma alternativa legal, contra respostas reais do adversário externo. Cada ramo dura até oito meios-lances; ambos alimentam o replay sem contar como partidas reais. |
| 4 — Informação | A exploração usa uma aproximação de ganho de evidência: discordância entre críticos, poucos exemplos independentes e baixa contradição nos resultados. O incentivo diminui com repetição e ruído. Não é uma implementação Bayesiana completa de VIME. |
| 6 — Esforço adaptativo | Mais análise perante pouca evidência, erro histórico elevado, perigo, xeque ou empate entre opções. Caso contrário usa a via rápida. Máximo de 96 posições adicionais avaliadas por decisão, três ações candidatas. |
| 8 — Verificação da confiança | Regista resíduos entre previsões feitas antes de aprender a partida e retornos de sobrevivência realmente observados. Concordância entre críticos não elimina um historial de erro. Censura e simulações não alimentam estes dados como resultados reais. |
| 9 — Memórias protegidas | Arquivo separado do replay FIFO, com retenção por erro/surpresa e diversidade; até 24 episódios completos são preservados para revisão periódica. A política de retenção não oferece bónus por vitória. |
| 10 — Robustez | Analisa várias respostas adversárias e inclui a pior encontrada, mesmo que o preditor a considere improvável. Combina média ponderada com pior resposta e compara apenas as candidatas analisadas. Regista respostas não exploradas quando o limite de nós é atingido. |
| 11 — Habilidades | Descobre sequências curtas recorrentes com retorno superior ao previsto, exige suporte de pelo menos dois jogos e verifica toda a sequência legal antes de propor reutilização. A sequência serve de desempate limitado, não força uma jogada pior. Uma resposta inesperada cancela o plano. |
| 12 — Generalização | Varia legalmente a última resposta do adversário, testa a mesma ação numa posição próxima e mede a alteração de sobrevivência. Os testes ajustam a confiança das memórias/habilidades. Regista separadamente resultados observados e estimados. |

As habilidades são sequências episódicas reutilizáveis, não conceitos humanos
como “criar uma fuga”, nem opções aprendidas pelo algoritmo DIAYN completo.
Confiança é erro de um valor descontado contínuo, não probabilidade calibrada
de vitória. A pesquisa é de duas meias-jogadas (ação e resposta), não um motor
completo de busca nem uma prova de segurança contra todas as respostas.

Experiências censuradas usam bootstrap, explicitamente marcado no JSONL por
`observed=false`. O campo `both_observed` diferencia comparações com dois
desfechos reais. Um teste de perturbação baseado em bootstrap altera confiança
com peso menor: não demonstra generalização real por si só.

## Preservação do objetivo e da informação

O objetivo continua a ser sobrevivência. Vitória e empate têm o mesmo alvo;
não se introduziram valores fixos de material, avaliações de Stockfish ou bónus
por vencer. Self-play e população de adversários continuam desativados nesta
entrada: todas as partidas reais e os ramos experimentais usam o motor externo.

O checkpoint preserva os pesos/otimizador/replay V2 e acrescenta arquivo,
episódios protegidos, erros de confiança, habilidades, respetiva confiança,
encoder de pesquisa e contadores de experiências. Migrar V2 inicia esses
módulos vazios, porque o replay antigo não guardou as previsões originais: não
se inventa evidência histórica de calibração. Memórias passam a crescer com
novas partidas reais.

Os atores recebem cópias imutáveis dos pesos e conhecimento entre partidas.
As experiências são executadas por uma instância separada do adversário. Não
interrompem jogos em curso nem modificam a memória real de outro ator.
Partidas em voo não são serializadas em checkpoints intermédios; a paragem
por `STOP` ou Ctrl+C drena as já submetidas e guarda depois.

## Uso quando o treino atual acabar

Exemplo **apenas se o próximo adversário também for 1600 nominal**:

```powershell
python train_cognitive.py --from-survival checkpoints/survival/latest.pt --opponent-elo 1600 --games 20000 --workers 4
```

Para outro nível suportado, passar explicitamente o Elo pretendido. O motor é
Stockfish 11 limitado a 20 ms por resposta; o Elo continua nominal neste ritmo.
Não foi iniciado nenhum novo treino de produção com este comando.

Retoma V3:

```powershell
python train_cognitive.py --resume checkpoints/cognitive/latest.pt --opponent-elo 1600 --games 1000
```

O argumento `--from-survival` recusa sobrescrever um checkpoint existente. A
pasta `checkpoints/survival` está bloqueada como destino de V3. A versão web
continua independente: este formato não deve ser passado ao exportador legado.
Usar como fonte o `latest.pt` de produção ainda marcado `running` também é
bloqueado: aguarda a gravação final `completed` ou uma paragem segura `stopped`.

## Validação em 22/09/2026

- Testes dos módulos: recuperação contextual, suporte por jogos independentes,
  conflito de evidência, erros partilhados pelos críticos, retenção limitada,
  requisitos/cancelamento de habilidades, vias rápida e lenta, teto de nós,
  resposta de mate improvável, isolamento de memória, experiências legais,
  objetivos e retoma dos pesos/replay/arquivo.
- A suite geral passou com 283 testes na primeira revisão; depois das últimas
  otimizações e proteções foram executados os testes focados V3, incluindo
  equivalência das regras de término e proteção contra ativação prematura.
- Smoke real com dois atores: quatro jogos completos, quatro pares
  contrafactuais e quatro testes de generalização; arquivo com 64 experiências,
  20 sequências candidatas e 23 recuperações utilizadas. Retoma subsequente de
  dois jogos passou, preservando o arquivo e acrescentando experiências.
- Um primeiro ensaio detetou bloqueio no encerramento do processo UCI. Foi
  corrigido com encerramento explícito em cada ator antes do shutdown de Python;
  os ensaios de migração e retoma seguintes encerraram e gravaram corretamente.
- Benchmark de 16 posições idênticas, duas repetições medidas, pesos iguais,
  CPU/uma thread, treino de produção concorrente: V2 média 1,26 ms versus V3
  8,36 ms na avaliação de candidatos, teto de 96 nós observado. O V3 anterior
  à otimização exata da verificação de repetição media aproximadamente 34 ms.
- Estes números são custo local de inferência, não jogos/s do próximo treino.
  Os ensaios não demonstram melhoria de Elo ou de sobrevivência. A política
  mudou deliberadamente; apenas 10 das 32 escolhas medidas coincidiram com V2.

Artefactos: `perf_results/cognitive_policy_benchmark_20260922.py/.json` e
`perf_results/cognitive-validation/history`. Todos os testes/smokes usam saídas
separadas. O treino principal manteve o mesmo PID e continuou a acrescentar jogos.
