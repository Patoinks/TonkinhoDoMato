# Treino V2: sobrevivência e sequências

O treino V2 é separado de `train.py`: preserva o modelo e os checkpoints legados.
`train_survival.py --init-from checkpoints/latest.pt` importa os pesos compatíveis
do encoder, predição de respostas e valores contextuais das peças. Inicializa
memória, sobrevivência e previsão futura novas; não importa o crítico W/D/L nem
o otimizador antigo. Não há garantia de aumento imediato da força de jogo.

## Objetivo

O alvo mede sobrevivência descontada, numa escala de zero a um. Numa derrota,
uma posição a `r` meios-lances do fim recebe `1 - gamma**r`, com `gamma=0.985`.
A posição final perdida recebe zero. Uma vitória e um empate concluídos recebem
exatamente o mesmo alvo: um. **Não existe prémio extra por ganhar**, nem valores
de peças ou avaliações de Stockfish nos rótulos.

O limite artificial de 300 meios-lances é censura: utiliza retorno de até oito
passos mais bootstrap de uma rede alvo, sem fabricar um empate ou sucesso. Os
resultados W/D/L continuam nos relatórios e na gravação da maior vitória externa,
mas não são uma função de recompensa. Este objetivo favorece evitar a derrota,
incluindo aceitar empates; não equivale a maximizar Elo ou probabilidade de ganhar.

## Capacidades implementadas

| Pedido | Implementação ativa |
|---|---|
| Memória temporal | GRU de 64 unidades observa cada posição realmente jogada. Candidatos hipotéticos têm memórias independentes. Undo, troca de ramo e transposições reconstroem o histórico correto. |
| Previsão do futuro | Quatro cabeças preveem representações a 1, 2, 4 e 8 meios-lances. O encoder alvo é uma cópia fixa da inicialização: não pode colapsar aproximando os alvos das previsões. |
| Erro como aprendizagem | Erro das respostas legais, erro futuro e erro do crítico treinam as respetivas cabeças; surpresa inicia prioridades de replay e os erros atualizam-nas. |
| Curiosidade | RND: erro de previsão de uma rede aleatória fixa dá um bónus limitado a 0,025 na exploração. O preditor aprende; o alvo fica congelado. O bónus não altera os alvos de sobrevivência. |
| Incerteza | Cinco críticos com conjuntos bootstrap diferentes. Dispersão entre críticos estima incerteza epistémica; outra saída aprende a variância residual. São estimativas, ainda sem calibração estatística. Só a dispersão pode acrescentar um pequeno incentivo exploratório. |
| Crédito longo | O desfecho real supervisiona todas as posições anteriores, incluindo decisões mais de 20 jogadas antes. Gradientes atravessam janelas de 48 meios-lances, com 16 de burn-in. Isto atribui crédito estatístico, não prova causalidade de uma jogada. |
| Diversidade | Cores alternadas aleatoriamente, cabeça bootstrap escolhida por partida, temperaturas variadas, mistura fixa de adversários e sementes independentes por ator. |
| Adversários antigos | Até oito snapshots congelados, guardados a cada 100 jogos, com âncora inicial e retenção de idades distintas. Escolha uniforme entre os snapshots retidos; vitórias não controlam retenção nem promoção. |

O replay mantém episódios completos compactados. Nenhuma janela atravessa jogos.
Um piso uniforme de 20% e prioridades limitadas evitam monopolização por casos
surpreendentes. Pesos de importância corrigem a amostragem dos inícios das janelas.
O burn-in é uma aproximação finita ao histórico anterior à janela; não é memória
infinita nem uma explicação textual do que aconteceu seis jogadas antes.

Valores das peças continuam aprendidos pela diferença entre a sobrevivência
estimada com a peça e uma posição contrafactual sem ela, mantendo o contexto
anterior. Os reis são excluídos. As unidades são sobrevivência, não peões ou
centipeões. Um sacrifício pode ser escolhido por aumentar sobrevivência prevista
ou por exploração; não existe uma penalização fixa por perder um cavalo.

## Adversários e desempenho

Mistura padrão: 50% Stockfish externo, 30% snapshots históricos, 20% cópia atual
congelada. Cada ator tem memória e processo UCI próprios, persistentes; o learner
é único. Os pesos mudam entre partidas, nunca no meio do jogo de um ator.

Stockfish 11 é pré-NNUE, cerca de 1,2 MB. Configuração: `UCI_LimitStrength=true`,
`UCI_Elo=1600`, uma thread, 16 MiB de hash, 20 ms por resposta. **1600 é nominal**:
a calibração original usa 60 s + 0,6 s e não mede a força neste orçamento curto.
Ver [proveniência, instalação e ensaio](opponent_1600.md).

Mais jogos/s pode significar apenas derrotas mais curtas. Comparar também
meios-lances de sobrevivência contra o mesmo adversário, posições aprendidas/s,
perda de previsão e tempo de aprendizagem. Mudanças de adversário, tamanho de
lote, extensão de jogo ou objetivo tornam comparações diretas de jogos/s enganosas.
O paralelismo altera a ordem de dados; não há promessa de identidade bit a bit.

## Executar e retomar

```powershell
python train_survival.py --init-from checkpoints/latest.pt --games 20000 --workers 4 --device cpu
python train_survival.py --resume checkpoints/survival/latest.pt --games 20000
```

O checkpoint V2 inclui pesos, rede alvo, encoder futuro fixo, otimizador, replay,
prioridades, população e estados RNG. O Stockfish não expõe uma semente UCI, e a
ordem de conclusão dos atores paralelos depende do sistema: a execução completa
não é deterministicamente reproduzível, mesmo com o learner/replay restaurados.
Partidas ainda em curso num checkpoint intermédio não fazem parte do replay.

Cada execução cria JSONL e `history/survival-.../launch.json`, com PID, fonte,
configuração, tempos e caminho de paragem. Criar o ficheiro `STOP` indicado no
launch termina após as partidas já em fila e guarda o estado. Ctrl+C pede a mesma
paragem. Um erro regista `failed`, nunca `completed`.

Os checkpoints têm `format=tonkai-survival-v2`. O exportador e o runtime web
legados não suportam esta arquitetura. Publicar o checkpoint final legado é uma
operação separada; não substituir o binário público por um V2 sem implementar
e validar também memória, novos pesos e política no browser.

## Referências de métodos

- [Recurrent Experience Replay / R2D2](https://openreview.net/forum?id=r1lyTjAqYX): replay de sequências e burn-in.
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952): prioridades com amostragem e correção.
- [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894): novidade por erro de um alvo aleatório.
- [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621): diversidade de estimadores e exploração.

Esta implementação adapta essas ideias; não reproduz integralmente os algoritmos
ou os resultados publicados.
