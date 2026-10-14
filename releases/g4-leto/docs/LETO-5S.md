# G4 Leto — até 5 segundos por lance

O tabuleiro usa o mesmo G4 Leto, checkpoint, regras nativas e Pawnomics v3. A página passa a pedir o lance a um worker dedicado, mantendo o tabuleiro e os controlos disponíveis durante o cálculo.

O cálculo começa por uma busca curta que fornece uma jogada de reserva, depois executa o orçamento normal do G4 e buscas progressivamente mais profundas, com mais nós e mais candidatos. Não há uma espera artificial para preencher cinco segundos. Aberturas do repertório, finais resolvidos, lances forçados e mates comprovados podem responder antes.

O limite público é 5000 ms. O controlador interrompe o worker 50 ms antes desse teto para deixar margem à entrega da resposta. A inicialização do worker também conta no orçamento de um pedido. No limite, usa apenas a última busca concluída; se o dispositivo não tiver concluído nenhuma, conserva uma jogada legal de recurso.

O cancelamento interrompe o cálculo ao recuar, navegar, abrir o histórico, desistir ou iniciar outra partida. Pedidos e respostas levam a identidade da posição, e uma resposta antiga não pode mover uma partida diferente. A memória recorrente é reconstruída com o histórico completo; a calibração do G4 passa entre lances e sobrevive à reinicialização do worker, sem acumular ajustes por cada aprofundamento. As variantes de treino têm estado separado.

Ficheiros centrais:

- `src/tonkai/letoThinking.ts`: orçamento, cancelamento, última escolha e calibração.
- `src/tonkai/leto.worker.ts`: modelo dedicado e reconstrução da partida.
- `src/tonkai/letoTimedSearch.ts`: aprofundamento progressivo com os valores do G4.
- `src/pages/TonkaiPage.tsx` e `src/tonkai/game.ts`: pedido assíncrono e aplicação de uma jogada válida à posição original.

Validação desta execução:

- 46 testes passaram: prazo de cálculo, cancelamento, resultados tardios ou ilegais, memória/calibração, cópia do modelo, Pawnomics, PGN e treino.
- TypeScript, ESLint dos módulos alterados e build Vite passaram. Os três avisos existentes de dependências de `revision` na página decorrem do objeto `Game` mutável; essas dependências foram mantidas.
- O worker do bundle de produção foi executado com o WASM e a rede reais. Quatro posições de partidas completaram profundidade 10, contra a profundidade fixa 6 anterior, com quatro passagens concluídas antes do teto. As escolhas foram legais e o histórico permaneceu igual.
- Uma posição inicial de abertura e um final de rei e peão responderam cedo. O final preservou o resultado exato.
- A thread principal continuou a emitir eventos de teste durante as buscas. Este ensaio usa `worker_threads` com adaptação do transporte de ficheiros; não é uma inspeção visual num navegador.
- O worker publicado também foi executado em Chrome headless isolado, com os ficheiros públicos reais e sem interação com a interface: abertura em 44 ms, duas posições de partida em aproximadamente 4968 ms, profundidade 10, jogadas legais, histórico preservado e zero erros. Durante o cálculo, os eventos da thread principal continuaram a correr, com intervalo máximo observado de 32.3 ms.
- A página React em jsdom manteve os fluxos de PGN, histórico, recuo e continuação de variantes. O componente Chessground e os estilos foram preservados.

Relatórios: `C:/Users/Martim/Documents/Codex/artifacts/leto-thinking-20261014`.
Isto demonstra mais cálculo dentro do orçamento; não constitui uma medição de ganho de Elo.

A [API de workers](https://developer.mozilla.org/en-US/docs/Web/API/Worker/terminate) permite terminar o cálculo mesmo enquanto o WASM está numa chamada síncrona. O tempo de entrega ao ecrã também depende do agendamento do navegador e do sistema operativo.
