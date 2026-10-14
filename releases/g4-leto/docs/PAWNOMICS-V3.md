# Pawnomics v3 — G4 Leto, 14 outubro 2026

O site continua a usar o G4 Leto e os mesmos pesos `d8f9edad5a5352ec86ef50048794dbe22d2b0b6ab354d8366ff38f6190df5c3f`.
Esta alteração é no cálculo e na escolha de jogadas de peões. Não treina a rede nem altera o checkpoint Python.
O tabuleiro Chessground, a página, os estilos, o histórico PGN e o modo treino foram preservados.

## Cálculo

- `pawnBoard.ts`: movimentos legais, incluindo capturas, promoção e en passant; verifica a segurança do rei.
- `pawnomicsV3.ts`: trocas legais com peças cravadas, rotas com capturas e ocupação atualizada, preparação de interceções e aproximação do rei antes da promoção. Um peão em perigo não ganha um tempo fictício para escapar quando é o adversário a jogar.
- Peças próprias bloqueiam a rota enquanto não se moverem. Uma captura pode abrir uma rota nova. Um avanço para um bloqueio de peões não conta como promoção; o progresso parcial mantém a escala do peão.
- Os valores continuam a vir do G4. A resistência material à conversão varia continuamente, sem tabelas fixas por fase ou por fila.
- Finais só com reis e peões, até seis peças e com um peão avançado, têm busca conjunta até seis meios-lances, limitada a 384 nós. Usa apenas a última iteração concluída e reconhece afogamento, repetição no ramo e o limite de 50 jogadas. Continua a ser uma estimativa quando o orçamento não resolve o final.

## Rei e peão contra rei

`pawn-kpk-v3.bin` contém resultados WDL e distância até à próxima captura ou movimento de peão, derivados das tabelas Syzygy locais. São regras de finais resolvidos, sem rótulos para aprendizagem da rede.

- 196608 bytes; 165676 posições canónicas válidas.
- SHA-256: `387b83763281256416c8a5103d4599401febd491c15ebd3cdac3fc598aa657c2`.
- Normalização por reflexão dos ficheiros e troca de cores.
- O carregamento verifica tamanho e checksum antes de ativar a tabela.
- A escolha preserva vitória/empate/derrota, considera promoção capturável e afogamento, e prioriza um movimento de peão que reinicia o relógio de 50 jogadas. Perto desse limite, resultados cuja distância não seja segura voltam ao cálculo normal.
- Regeneração: `python scripts/generate-pawn-proofs.py --syzygy PATH --output public/tonkai/pawn-kpk-v3.bin`.

A definição de WDL/DTZ e a possível diferença de um meio-lance na distância estão documentadas em [python-chess/Syzygy](https://python-chess.readthedocs.io/en/latest/syzygy.html).

## Validação

Amostra independente, seed 801749, com Syzygy como juiz:

- 128 posições de rei e peão contra rei: todas as jogadas legais coincidiram com os resultados do juiz; a escolha preservou sempre o melhor resultado.
- G4 completo em 80 finais: versão anterior 77/80, nova 79/80; duas melhorias e nenhuma regressão nesta amostra.
- 32 finais ganhos jogados contra defesa ótima independente: 32/32 chegaram à promoção mantendo vitória, até 29 meios-lances; sem empate por repetição ou 50 jogadas.
- 36 posições de partidas existentes: todas as escolhas legais, sem alteração do tabuleiro ou histórico. Custo mediano observado: 358 ms antes e 381 ms depois; depende do equipamento e da posição.
- Nas oito escolhas alteradas desta amostra de partidas, Stockfish 11, com um thread e 100000 nós por sucessor, avaliou uma melhoria acima de 25 cp, quatro perdas acima de 25 cp e três diferenças até 25 cp. A média dessas oito diferenças foi -23.875 cp. Isto não demonstra aumento geral de força ou de Elo; as correções de legalidade e os finais têm evidência mais forte.
- 36 testes de Pawnomics, compatibilidade Leto, PGN e treino passaram, além de TypeScript, ESLint dos ficheiros alterados e build Vite.
- A página React foi validada em jsdom para guardar/descarregar PGN, recuar, navegar e continuar variantes. O renderer Chessground real também é montado nessa verificação; não é uma inspeção visual num navegador.

Os relatórios completos desta execução estão em `C:/Users/Martim/Documents/Codex/artifacts/pawnomics-20261014`.
A configuração `claudio_config.pawn_version = 3` e o descritor `pawn_proofs` ativam o módulo no site. Sem essa opção, o caminho anterior continua disponível para comparações.

## Publicação

`scripts/publish-tonkai-interface.mjs --pawnomics --prepare|--publish|--verify` preserva os ficheiros publicados fora desta mudança. Aceita alterar apenas o HTML de entrada e os manifestos Leto, acrescentar o novo bundle e a tabela de finais. Verifica que pesos, arquitetura, valores, regras nativas, abertura e restantes parâmetros continuam iguais à versão publicada anteriormente.
