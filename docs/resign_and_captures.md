# V3: desistência, alvo de capturas e K/D

Tudo opt-in. Sem `--resign` e com `--capture-target-weight 0` (omissão) a V3
comporta-se exatamente como antes; a V2 nunca é afetada.

```
python train_cognitive.py --resume <v3.pt> --new-campaign --checkpoint-dir <novo> \
    --opponent-elo 1000 --resign --capture-target-weight 0.05
```

## Desistência (`--resign`)

Decidida **só pela estimativa crua da rede** (`choice.mean`), nunca pelos scores
já ajustados pelas heurísticas (king/piece safety, busca, memória), nunca por
material, nunca por avaliação de motor. Desiste quando até o melhor lance prevê
morte em ≤ 10 meios-lances (objetivo survival) ou pontuação esperada ≤ 0,05
(objetivo victory), durante 3 decisões seguidas, nunca antes do meio-lance 40.
Nunca desiste com mate ou empate disponível no tabuleiro (regras, não heurística).

Começa em **modo sombra** (só regista "teria desistido"). Liga a sério apenas com
≥ 200 disparos verificados, ≤ 5% de desistências erradas **e ≥ 10% de jogos não
perdidos**. A última condição impede a armadilha da value head saturada: a perder
quase tudo, a verificação sozinha aprovaria desistir no primeiro lance. Em modo
ativo, 10% dos jogos que desistiriam são jogados até ao fim para verificação; se
as desistências erradas passarem 5%, o limiar aperta sozinho.

Um jogo desistido fica **cortado** (bootstrap), como o limite de meios-lances —
nunca é rotulado como uma morte inventada.

## Alvo auxiliar de capturas (`--capture-target-weight`)

A `piece_head` ganha uma saída extra (`capture_out`, lida da mesma camada
escondida): quantas capturas **aquela peça, naquela casa** vai fazer no resto do
jogo. É o que aconteceu de facto, contextual e aprendido. Não é recompensa e não
entra na escolha do lance. Checkpoints sem a cabeça ganham-na ao carregar (o
otimizador é ajustado); desligar remove-a.

## Medição (sempre ligada na V3)

`captures` nas métricas: kills, mortes e K/D por tipo de peça, dos dois lados; e
o teste da hipótese "capturar uma peça quente ganha jogos": quando a TonkAI
captura uma peça que já tinha feito ≥ `hot_piece_kills` capturas nesse jogo,
como acaba o jogo, comparado com os jogos sem essa captura.

## O bónus que saiu

`capture_scores` (bónus de `0,08 × tanh(ganho/3)` somado ao score do lance, com
objetivo victory) foi removido: era recompensa por captura na decisão.
