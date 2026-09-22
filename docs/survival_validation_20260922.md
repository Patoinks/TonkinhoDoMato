# Validação V2 em 22/09/2026

O treino legado terminou naturalmente: 20.000 jogos nesta sessão, 22.500 jogos
acumulados no checkpoint. Não foi parado nem teve o adversário substituído.

## Testes

- Suite Python completa: 257 testes passaram antes das últimas alterações CLI.
- Sete testes CLI adicionais passaram: retoma e configuração do motor, paragem
  por ficheiro, drenagem dos atores, proteção contra sobrescrita e histórico PGN
  não fatal.
- Depois de evitar a reconstrução redundante do tabuleiro do adversário neural,
  os 24 testes de integração temporal e do pool foram repetidos e passaram.
- Ensaios reais: quatro jogos seriais, retoma com quatro jogos paralelos e 24
  jogos com configuração completa, quatro atores e aprendizagem ativa.
- Os testes comprovam execução, gradientes, semântica dos alvos e retoma; não
  demonstram aumento de Elo, calibração da incerteza ou aprendizagem duradoura.

## Desempenho

Stockfish 11 nominal 1600, 20 ms por resposta, uma thread por processo. Ensaio
de atores sem learner, após o treino legado terminar:

| Atores | Jogos/s | Meios-lances/s |
|---:|---:|---:|
| 1 | 2,30 | 82,25 |
| 2 | 5,24 | 151,85 |
| 4 | 7,95 | 222,55 |

Oito jogos por condição. Os jogos tiveram comprimentos diferentes, pois o RNG
do Stockfish não é controlável por UCI. O ganho normalizado por meios-lances
dos quatro atores foi 2,71x. Arranque dos processos medido separadamente.

Learner: sequências de 48, burn-in 16, lote oito, corpus idêntico de 972 estados,
três passos de aquecimento e dez medidos, uma thread PyTorch:

| Dispositivo | Média por atualização |
|---|---:|
| CPU | 55,07 ms |
| RTX 5060 Ti / CUDA | 157,26 ms |

CPU foi 2,86x mais rápido nesta rede pequena e recorrência em passos curtos.
Perdas finitas, mesmas chaves de amostras; diferença máxima de loss 9,54e-7.
Não se afirma equivalência bit a bit CPU/CUDA. Escolha: CPU, uma thread no
learner, quatro atores persistentes com uma thread cada.

Ensaio completo de 24 jogos com pesos legados finais importados: 5.171
meios-lances, 44 atualizações, 9,21 s incluindo arranque/encerramento/checkpoint,
2,61 jogos/s. Média 215,46 meios-lances por jogo; mistura observada nesta amostra
pequena: sete Stockfish, catorze históricos, três cópias atuais. Não comparar
esse valor diretamente com os cerca de 52 meios-lances do treino legado. Este
ensaio antecede a otimização equivalente do histórico do adversário neural.

Artefactos reproduzíveis:

- `perf_results/survival_actor_benchmark_20260922.py` e `.json`.
- `perf_results/survival_learner_benchmark_20260922.py` e `.json`.
- `perf_results/survival-integrated-benchmark/history/survival-20260922T165839Z-4d774c22.jsonl`.

Não se reduziu o orçamento de pesquisa do adversário para obter estes ganhos.
