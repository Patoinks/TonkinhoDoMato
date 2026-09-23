# Resultado: pesquisa exata de trocas

## Implementação

- `tonkai_fast/fast.cpp`: pesquisa de capturas por negamax com alpha-beta e ordem MVV-LVA. Fora de xeque, permite stand-pat e só continua por capturas legais. Em xeque, considera todas as respostas legais; sem resposta, devolve `-1000.0`. En passant vale um peão e a promoção soma `valor(promovida) - valor(peão)`.
- `tonkai/exchange.py`: referência independente com python-chess e a mesma semântica.
- `tests/test_exchange_search.py`: nove posições manuais (incluindo raio-x, en passant, promoção e mate), limites, API em lote, horizonte e 300 posições do PGN indicado. O tabuleiro é verificado após cada chamada.
- Não foi necessário alterar `tonkai/fast_rules.py`: o `code(move)` e o construtor `board(position)` existentes servem diretamente a nova API.

## API exata

No objeto `tonkai_fast.Board`:

```python
exchange_gain(values: list[float], max_depth: int, node_limit: int) -> tuple[float, bool, int]
exchange_after_moves(codes: list[int], values: list[float], max_depth: int, node_limit: int) -> list[float]
```

`values` tem seis elementos, com o índice 0 reservado e os tipos `PAWN=1` até `QUEEN=5`. `codes` usa `tonkai.fast_rules.code(move)`. Cada elemento de `exchange_after_moves` é o ganho do adversário **após** o lance correspondente; o tabuleiro original não muda.

Na referência `tonkai.exchange`:

```python
exchange_gain(board: chess.Board, values: list[float], max_depth: int) -> float
exchange_after_moves(board: chess.Board, moves: list[chess.Move], values: list[float], max_depth: int) -> list[float]
```

## Validação

`python setup.py build_ext --inplace` terminou com código 0.

`python -m pytest -q -o addopts=''`:

```text
520 passed, 1 warning in 198.48s (0:03:18)
```

O aviso veio de `tests/test_cuda_learner.py` e refere uma divergência de streams no `AccumulateGrad` do PyTorch. A comparação C++/Python passou em 300 posições, uma por cada um dos primeiros 300 jogos de `benchmarks/benchmark-v3.1-vs-checkpoint1-500.pgn`, após `8 + 2 * (índice % 8)` meias-jogadas, com `values=[0, 1, 3.9, 4.6, 5.2, 10.3]`, `max_depth=8`, `node_limit=10_000_000` e tolerância absoluta `1e-9`.

## Tempo

Medi `exchange_after_moves` na extensão final, nas posições de 20 jogos desse PGN, com a mesma escolha de meia-jogada, **todas** as jogadas legais por posição, profundidade 8, os valores acima e limite de 10 milhões de nós por jogada. Foram 60 chamadas (3 repetições por posição), com 630 jogadas legais por passagem: média **0,889 ms por chamada**, mediana **0,341 ms**, mínimo **0,041 ms**, máximo **4,254 ms**. Medição local com `time.perf_counter()` e sem a suite a correr em paralelo.

## Decisões onde a tarefa não especificava o comportamento

- `node_limit` aplica-se a cada pesquisa de `exchange_gain`; em `exchange_after_moves`, cada lance recebe o seu próprio limite. Como `exchange_after_moves` não devolve um sinal `complete`, lança `RuntimeError` se qualquer pesquisa exceder o limite. `exchange_gain` devolve `(0.0, False, nodes)` quando aborta; o zero nesse caso não é um resultado exato.
- Um nó é contado na entrada da função, incluindo a raiz e as folhas, como em `MateSearch`. Ao exceder o limite, `nodes` pode ser `node_limit + 1`.
- No horizonte (`max_depth=0`), ainda se deteta mate; outras posições devolvem 0. Sem lances e sem xeque, o resultado também é 0. Não se aplicam regras de repetição ou de cinquenta lances nesta pesquisa limitada a capturas.
- Vetores com comprimento diferente de seis, limites negativos e códigos de lances ilegais são rejeitados. A ordem da lista devolvida segue exatamente a ordem de `codes`.

Não foi feito commit, nem foram iniciados treinos ou matches, conforme a tarefa.
