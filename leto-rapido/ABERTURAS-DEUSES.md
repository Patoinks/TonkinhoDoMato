# Aberturas dos deuses vs personalidade (2026-10-19)

Ficam quatro deuses: Nezha, Loki, Anúbis, Hades (retirados: Rá, Amaterasu, Susanoo, Wukong, Zeus, Tsukuyomi).

## Cada lance do repertório, pelo estilo de cada deus (`aberturas_estilo.py`)

Percentil do lance entre todos os lances legais da posição (1,0 = o preferido desse estilo):

| repertório \ estilo | Nezha | Loki | Anúbis | Hades |
| --- | ---: | ---: | ---: | ---: |
| Nezha | **0,61** | 0,59 | 0,48 | 0,39 |
| Loki | 0,54 | 0,42 | **0,63** | 0,50 |
| Anúbis | 0,57 | 0,48 | **0,65** | 0,54 |
| Hades | **0,61** | 0,46 | 0,53 | 0,55 |

## Perfil da abertura (meios-lances 0-19) vs resto do jogo, 100 jogos vs Stockfish 2250

Correlação dos cinco índices: Nezha +0,86, Hades +0,41, Loki -0,19, Anúbis -0,55 (Leto neutro +0,58).

## Cobertura

Só ~10% dos lances próprios dos primeiros 20 meios-lances vêm do repertório do deus (média de 1 lance de livro
por jogo; 16-25 jogos em 100 sem nenhum): o Stockfish sai das 4 linhas de cada deus logo no início, e a abertura
é quase toda o livro do Leto e a janela de estilo.

## Conclusão

As aberturas não são 100% equivalentes à personalidade: só a do Nezha bate; Loki e Anúbis contradizem o estilo,
Hades em parte. E pesam pouco, porque cobrem poucas posições. `gerar_aberturas.py` (repertório escolhido lance a lance
pelo estilo) não foi adotado: os sinais de estilo são grosseiros demais para os primeiros lances (todos 1.Cf3).
