# V3: desistência, alvo de capturas, K/D, objetivo shaped e vitórias

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
morte em ≤ 10 meios-lances (objetivo survival), pontuação esperada ≤ 0,05
(objetivo victory) ou, com shaped, `(1−w)(1−γ¹⁰) + w·0,25` ≈ 0,195, durante 3 decisões seguidas, nunca antes do meio-lance 40.
Nunca desiste com mate ou empate disponível no tabuleiro (regras, não heurística).

Começa em **modo sombra** (só regista "teria desistido"). Liga a sério apenas com
≥ 200 disparos verificados, ≤ 5% de desistências erradas **e ≥ 10% de jogos não
perdidos** (`--resign-min-non-loss`; a 0 esta última condição desliga-se). A última condição impede a armadilha da value head saturada: a perder
quase tudo, a verificação sozinha aprovaria desistir no primeiro lance. Em modo
ativo, 10% dos jogos que desistiriam são jogados até ao fim para verificação; se
as desistências erradas passarem 5%, o limiar aperta sozinho.

Um jogo desistido **é uma derrota a sério** (como no AlphaZero): termina, conta
como derrota nas métricas, alvo 0, e o resultado do PGN é do adversário. Não entra
no histórico de vitórias.

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

## Objetivo shaped (`--objective shaped --victory-weight 0.5`)

A definição de vitória do dono: baixar a segurança do rei adversário, subir a do
nosso, atacar peças inimigas de K/D alto e proteger as nossas. Em cada estado a
TonkAI regista uma medida em [0, 1]:

    proxy = 0,5 + 0,25·(segurança do nosso rei − do rei adversário)
                + 0,25·(risco das peças inimigas − risco das nossas)

com o risco pesado pelo valor das peças. O alvo de treino é
`(1−w)·sobrevivência + w·proxy`; o fim do jogo manda sempre: vitória 1, empate
`1 − w/2`, derrota 0. Na decisão, a continuação usa a mesma mistura e um mate
imediato é sempre jogado.

## Valor das peças com K/D (`--kd-weight 0.75`)

O K/D médio de cada tipo de peça junta os dois lados (TonkAI e adversário) e é
dividido pelo do peão. O valor de uma peça passa a ser
`0,75·K/D do tipo + 0,25·valor contextual da rede`. Só entra quando há ≥ 50
mortes por tipo; o rei continua fora.

## Histórico de todas as vitórias

Cada vitória real (mate ou fim de jogo, nunca uma desistência) é acrescentada a
`history/wins/wins.pgn` e indexada em `history/wins/wins.json`. Com
`web_publish.local.json` ativo, o treino lança `scripts/publish_wins.py` em
segundo plano, no máximo uma vez a cada 10 minutos e mais uma no fim. O script
copia os PGN para o site e faz deploy só dos ficheiros estáticos. Recusa fazer
deploy se o `index.html` que está no ar não for igual ao `dist` local, para não
fazer rollback do site. Na página /tonkai, "Todas as vitórias" lista-as e abre
cada uma no leitor.
