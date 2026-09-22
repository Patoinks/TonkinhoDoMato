# Treino V3: 500 jogos contra Rodent IV

Adversário: Rodent IV 0.33, UCI_Elo=1000, UCI_LimitStrength=true,
Threads=1, Hash=16 MiB, orçamento solicitado de 2 ms/jogada.
É Elo nominal: a força efetiva neste controlo ultracurto não foi calibrada.
Em 60 jogadas legais locais, o tempo médio observado foi 15.22 ms (máximo
16.39 ms), incluindo IPC e a espera do motor/Windows. Não é uma garantia de
1000 Elo FIDE, Chess.com ou Lichess. Não usa NNUE; não foram instalados
livros junto do executável usado no treino.

Fonte oficial: https://github.com/nescitus/rodent-iv
Commit: e8d84c8c8c189a1cf4eb27c578fc573af3b916d2 (GPL-3.0).
O código do autor aceita UCI_Elo de 800 a 2800 em `sources/src/uci_options.cpp`.
Foi compilado localmente em Release/x64 com MSVC v145 e Windows SDK 10.0;
a única correção de compatibilidade foi incluir `<chrono>` antes de `<thread>`
em `uci.cpp`. Fontes e licença ficam em `engines/rodent-iv/`.
Executável: `engines/rodent-iv/rodent-iv.exe`.
SHA256: ef5457912612f47f52d3f083b0d851ed681ccf9c9896e8250140cf07b8043a0c.

`train_cognitive.py --games 500 --game-cap 500` limita a campanha a 500
partidas reais, mesmo após `--resume`. Continua 100% contra o adversário
externo, com os nove módulos V3. Ramificações experimentais não são partidas
extra nem contam como vitórias/empates/derrotas.

Os cinco primeiros jogos são identificados pela ordem de envio aos atores,
independentemente da ordem de conclusão. São guardados no diretório do
checkpoint em `first-five.pgn` e `opening-report.json`. Comparam-se a primeira
jogada própria e as sequências dos primeiros dez meios-lances, separando as
cores. Não são atribuídos nomes ECO sem uma base de aberturas. Cinco jogos
com exploração são uma amostra pequena, não uma prova de política fixa.

O treino V2 anterior deixou de ter processo ativo sem gravar conclusão:
último checkpoint com 19184 jogos aprendidos, último log com 17280 jogos no
segmento externo. A migração usa uma cópia preservada desse checkpoint,
mantendo o estado original `running` para não inventar uma conclusão.
