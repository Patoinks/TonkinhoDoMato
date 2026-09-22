# Rodent: otimização e campanha de 100 mil jogos

O perfil do ator V3 identificou trabalho repetido nas trocas e no mapa do rei.
As trocas agora reutilizam os valores da posição e a primeira lista de
recapturas; o mapa do rei usa geometria pré-calculada e bitboards, verificando
cravadas apenas nas peças que alcançam a zona relevante. Não se alteraram os
pesos, fórmulas, profundidade de troca, orçamento de busca ou objetivo.

31 posições com história e checkpoint iguais: média de 30,68 para 23,49 ms por
decisão (menos 23,4%). Todas as pontuações e escolhas ficaram exatamente iguais.
Comparação adicional em 500 posições legais: mapas das peças e dos reis iguais
à versão anterior, incluindo profundidades de troca 1, 2 e 6. Passaram 76 testes
Python e 21 referências Python/navegador.

Benchmarks de treino reais contra o Rodent IV, 1000 Elo nominal, 2 ms/jogada,
com updates e experiências ativos, 48 jogos por configuração:

| Atores | Jogos/s com arranque | Jogos/s após primeira vaga | Posições/s |
| --- | --- | --- | --- |
| 4 otimizados | 2,35 | 2,68 | 122,79 |
| 6 otimizados | 3,05 | 3,76 | 153,75 |
| 8 otimizados | 3,27 | 4,46 | 161,73 |
| 8 anteriores | 2,86 | 3,72 | 155,34 |

Escolhidos 8 atores. Estes ensaios são curtos, com partidas de comprimentos
diferentes e Rodent sem semente controlável; os jogos/s não isolam o ganho de
código nem comprovam força. A comparação das mesmas posições é a evidência
direta do ganho sem alteração das avaliações. Os benchmarks ficaram em memória
e não substituíram o checkpoint de produção nem contam para os 100 mil jogos.

`scripts/start_rodent_100k.ps1` lança a campanha a partir do checkpoint V3
concluído de 500 jogos. `--new-campaign` mantém pesos, otimizador, replay e
experiências, reiniciando apenas o contador/limite da campanha e o relatório das
primeiras cinco aberturas. Exige outro diretório e recusa fontes ainda em treino.
A campanha tem 100000 jogos adicionais, só contra Rodent, sem self-play.

Mantêm-se dois updates por jogo, máximo de 300 meios-lances, experiências e
generalizações a cada 20 jogos, checkpoints a cada 100 e ambos os módulos de
proteção. O progresso é registado a cada 25 jogos. O gancho existente publica
o modelo no site após conclusão normal; paragens/falhas não publicam.

Para retomar uma interrupção, usar `--resume` apontado ao `latest.pt` desta
campanha, sem `--new-campaign`; o limite persistente impede ultrapassar 100000.
O launcher aceita `-Resume`. Resultados e motivos de empate acumulam entre
retomas. Aos 200 jogos houve um bloqueio temporário de ficheiro do Windows;
o checkpoint final preservou os 200. A gravação atómica agora tenta novamente
durante até cinco segundos perante `PermissionError`, mantendo o checkpoint
anterior intacto e continuando a reportar bloqueios persistentes como falha.
