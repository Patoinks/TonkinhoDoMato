# G16 v2 no site

Publicado a 9 de outubro de 2026: https://anw-games.web.app/tonkai

O site usa o pacote `claudio/v2/models/g16-v2.pt`, variante `all`, nascido do
checkpoint 600 (bloco 0501-0600), SHA256
`5a2681e23cafb6f889e58c35cfd12ff6a32332d5acfdb75b733b832a2c51e18f`.
Esta publicacao nao iniciou nem retomou treino. Os pesos publicados sao congelados.

## Motor e validacao

O C++ original de `claudio/fast/claudio_fast.cpp` e o MatePlanner de
`claudio/v2/native/extra.cpp` sao compilados para WebAssembly. O gerador retira
os bindings Python e funcoes auxiliares nao usadas, preservando os corpos da
pesquisa de material, mates, regras, limites de nos e geracao de lances.
Nao altera os ficheiros reservados do outro agente.

A decisao em TypeScript segue a ordem do G16 v2: rede compacta com residual,
memoria GRU real, livro proprio, seguranca dos reis, trocas e valores por fase,
varrimento, respostas ponderadas pela propria rede, politica na ordem das
raizes, pesquisa PVS, aprofundamento critico e verificacao de mates.
O planeador de conversao conserva os limites de 4/8 lances e 40000/400000 nos.
Stockfish continua a ser apenas a avaliacao separada que ja existia na interface.

Comparacao com Python: **102 decisoes, 0 lances diferentes**; 41 reordenacoes
pela politica, 9 entradas do livro e 3 mates encontrados pelo planeador.
Erro maximo da memoria GRU: 0.0000012811. A aritmetica da rede no navegador
pode diferir em arredondamento; este teste nao demonstra equivalencia universal
nem um limite estatistico de 2% de perda de forca.

O navegador verifica SHA256 da rede, do loader e do WASM antes de jogar.
Rede exportada: `6af0464105a510e413f43b6eb5d2300a921b7e75890547134b17d816bd328985`.
O fim da partida usa as regras nativas, incluindo reclamacoes de repeticao e
50 lances, material insuficiente e a prova conservadora de peoes bloqueados.
Um empate vale 0.5. Limites artificiais nao sao apresentados como empates.

Verificacoes: build TypeScript/Vite; suite web 71 testes passaram/2 ignorados;
22 testes especificos v2 passaram. A suite Python geral teve 574 passaram,
8 ignorados e a falha preexistente `test_no_classical_piece_value_table_in_the_package`
no fallback de `tonkai/cognitive_agent.py`, sem alteracoes nesse ficheiro.
Smoke publico: partidas com brancas e pretas (8/9 meios-lances, terminadas
por desistencia no teste), relogios e incremento, avaliacoes escondidas por
defeito/toggle, layout movel e ausencia de erros JavaScript. Nao sao resultados
de treino nem confrontos para medir Elo; o teste bloqueia escritas no Firestore.

## Reproduzir

`claudio/v2/web/build.py` recebe `--emsdk` e `--out`; compila com C++20,
O3 e `-ffp-contract=off`. O SDK usado foi Emscripten 6.0.12.
`export.py --bundle claudio/v2/models/g16-v2.pt --out <assets>` escreve pesos
e manifestos. O exportador rejeita configuracoes que o port nao suporta.
`reference.py` gera as referencias reais a partir de partidas proprias.

`claudio/v2/web/site/` guarda os ficheiros de integracao do checkout de release
`ANW-TONKAI-RELEASE-20260922`, sem fazer commit das alteracoes preexistentes
nesse checkout. `scripts/verify-g16-v2.ts` compara o motor de navegador com
`validation/python-reference.json`; `smoke-g16-v2.mjs` verifica o tabuleiro
local/publico. `assets/` guarda o export pronto a usar.

A publicacao seletiva reutiliza todos os ficheiros do Hosting atual, acrescenta
os assets do G16 v2 e uma entrada `/tonkai/index.html`. O HTML inicial conserva
o bundle das outras paginas e ganha apenas a navegacao que recarrega ao entrar
no xadrez. A pagina do xadrez recarrega ao sair para manter as outras paginas.
O publisher revalida a versao live antes de ativar e preserva um snapshot para
comparar os ficheiros. Nao apaga versoes nem assets antigos.
