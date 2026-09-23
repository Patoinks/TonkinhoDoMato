Tarefa do GPT-6 Sol: pesquisa exata de trocas (capturas) no tonkai_fast

Contexto: TonkAI é uma IA de xadrez em Python e PyTorch, com regras em C++ (pybind11) em `tonkai_fast/fast.cpp`. Queremos um FACTO calculado pelas regras: quanto material o lado a jogar consegue ganhar à força só com capturas, usando os valores de peças que o próprio TonkAI mede. É o mesmo espírito da pesquisa exata de mates (`struct MateSearch` e `Board::forced_mate` no mesmo ficheiro). Lê esse código antes de começares.

## O que implementar

1. Em `tonkai_fast/fast.cpp`, uma pesquisa de capturas (quiescência) exata:
   - `q(p)` para o lado a jogar em p:
     - se NÃO está em xeque: `q = max(0, max sobre capturas legais c de (valor(vítima de c) + ganho_promoção(c) − q(p depois de c)))`. O 0 é poder recusar trocar (stand-pat).
     - se ESTÁ em xeque: sem stand-pat. Considera TODAS as respostas legais: capturas contam o valor da vítima, as outras contam 0. `q = max sobre respostas r de (ganho(r) − q(p depois de r))`. Sem respostas legais é mate, e q = −MATE (constante grande, por exemplo 1000).
   - en passant conta como captura de um peão. ganho_promoção = valor(peça promovida) − valor(peão).
   - `values` é um vetor de 6 doubles indexado pelo tipo de peça (PAWN..QUEEN, os índices do código). O rei nunca é capturado, porque só há lances legais.
   - `max_depth` limita as meias-jogadas de capturas. Ao chegar ao fundo, devolve o stand-pat (0), ou −MATE se for mate.
   - `node_limit`: conta nós como o MateSearch; se passar o limite, aborta e marca o resultado como incompleto.
   - Alpha-beta com ordem MVV-LVA. O resultado tem de ser EXATAMENTE o mesmo que sem poda quando não há aborto.
2. Métodos novos no `Board` (binding pybind11):
   - `exchange_gain(values, max_depth, node_limit) -> (gain: float, complete: bool, nodes: int)` = q da posição atual.
   - `exchange_after_moves(codes, values, max_depth, node_limit) -> list[float]`: para cada código de lance (o formato de `fast_rules.code`), faz o lance e devolve o que o ADVERSÁRIO ganha à força a seguir (q para o adversário). Não altera o tabuleiro.
3. Referência em Python com python-chess, com a MESMA semântica: `tonkai/exchange.py`, funções `exchange_gain(board, values, max_depth)` e `exchange_after_moves(board, moves, values, max_depth)`. Acrescenta wrappers em `tonkai/fast_rules.py` só se for preciso, sem mexer no que lá existe.
4. Testes em `tests/test_exchange_search.py`:
   - posições feitas à mão com resposta conhecida: peça pendurada; troca igual (ganho 0); troca com recaptura de maior valor; raio-x/bateria; en passant; promoção com captura; xeque durante a troca; mate durante a troca;
   - C++ igual à referência Python (tolerância 1e-9) em ~300 posições tiradas de `benchmarks/benchmark-v3.1-vs-checkpoint1-500.pgn`, com os valores {1:1, 2:3.9, 3:4.6, 4:5.2, 5:10.3}, max_depth 8;
   - o tabuleiro fica intacto depois de cada chamada.
5. Compila com `python setup.py build_ext --inplace`. Mede o tempo médio de `exchange_after_moves` em posições reais (todas as jogadas legais, max_depth 8) e escreve o número no relatório.

## Regras (obrigatórias)

- Mexe APENAS em: `tonkai_fast/fast.cpp`, `tonkai/exchange.py` (novo), `tonkai/fast_rules.py` (só acrescentar), `tests/test_exchange_search.py` (novo) e o relatório.
- NÃO faças git commit, checkout, reset, stash nem restore. NÃO arranques treinos nem matches. NÃO mexas em checkpoints nem em `history/`.
- Antes de acabar, corre `python -m pytest -q -o addopts=''`: a suite completa tem de passar (hoje são 505 testes).
- Escreve o relatório em `docs/tasks/RESULTADO-SOL-EXCHANGE.md`:
  - o que fizeste;
  - a assinatura exata da API;
  - o output dos testes;
  - o tempo medido;
  - qualquer caso onde tiveste de decidir algo não especificado.
