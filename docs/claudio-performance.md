# Claudio: otimização local, 2026-10-08

Base: `d5e6fda`, branch `claude/cloud-credits-100-usage-bcttvt`. Trabalho numa
worktree separada, mantendo intacta a pasta de trabalho anterior e os checkpoints.
Nenhum treino nem geração de livro foi lançado nesta otimização.

## Alterações

- O cálculo tático usa a mesma posição sucessora para validar e pesquisar o lance.
  Antes copiava a posição e executava o lance duas vezes.
- O ganho material de cada candidato é calculado uma vez e acompanha o candidato
  na ordenação estável, em vez de ser recalculado na pesquisa.
- A memória da tabela de transposições é reutilizada por thread. Uma geração
  distinta por chamada torna a tabela logicamente vazia: não reutiliza avaliações,
  movimentos nem limites de uma pesquisa anterior. O contador tem limpeza no overflow.
- Compilação AVX2 opcional em Windows/MSVC, verificada neste Ryzen 5 7500F,
  conservando `/fp:strict`. O build portátil continua a ser o padrão.

Profundidades, parâmetros dos módulos, limites de nós, ordem dos candidatos,
objetivos de aprendizagem e pesos dos modelos foram preservados.

## Medição

Repetição das mesmas posições reais do BO20 G15 treinado vs G10. Quatro execuções
por modelo, na ordem original/otimizado/otimizado/original (ABBA), um thread CPU,
prioridade BelowNormal. GPU não usada. Cada decisão conservou a memória GRU do jogo.

| Modelo | Amostra por execução | Original, s de cálculo/jogo | Otimizado | Poupança |
|---|---|---:|---:|---:|
| G15 após 100 jogos vs SF1800 | 8 jogos, 943 decisões | 2,3854 | 2,2172 | 0,1682 s; 7,05% |
| G16 com módulos disponíveis | 1 jogo, 128 decisões | 4,1761 | 3,9573 | 0,2188 s; 5,24% |

Todos os lances coincidiram nas 1 071 posições distintas destas amostras e nas
repetições. Estes tempos são de cálculo do jogador em posições gravadas, com um
thread; não são o tempo total de um novo BO20 com dois workers. A amostra G16 é
pequena e o livro ainda não foi gerado. Não demonstram aumento de força.

## Verificação

- 540 comparações diretas de pesquisa nativa com a DLL original: resultados
  idênticos, incluindo valores, linhas principais, limites e interrupções NaN.
  Incluem posições com roque, en passant, promoção e cheque, dois conjuntos de
  valores de peças e limites de 1, 32 e 4 096 nós.
- 27 testes Claudio aprovados, incluindo novos testes de isolamento entre
  chamadas que mudam valores de peças ou limite de nós.
- Suite geral: 574 aprovados, 8 ignorados, 1 falha antiga em
  `tests/test_model.py::test_no_classical_piece_value_table_in_the_package`,
  já documentada no README antes desta alteração. O código TonkAI partilhado não
  foi alterado. As extensões da raiz precisaram de compilação na worktree nova.
- `git diff --check` passou.

## Compilar neste PC

Na pasta `claudio/fast`, com Python 3.13 e MSVC:

```powershell
$env:CLAUDIO_NATIVE_ARCH = 'AVX2'
python setup.py build_ext --inplace --force
Remove-Item Env:\CLAUDIO_NATIVE_ARCH
```

Para o build portátil, deixar `CLAUDIO_NATIVE_ARCH` ausente. O binário AVX2 é
local para Windows x64/Python 3.13 e CPU com AVX2; não substituir os binários da
nuvem por este ficheiro. O código fonte conserva o build Linux padrão.

Os scripts normais `grupos_finais.py`, `vs_engine.py`, `train_grande.py` e
`experiencia.py` usam a extensão desta worktree. Usar 2 workers e prioridade baixa
para respeitar a preferência atual de carga leve do Martim. Não gerar o livro
nem treinar automaticamente.

Resultados e scripts de reprodução deste ensaio foram guardados na pasta
`Documents/Codex/2026-10-08/b/outputs` e em `work`, respetivamente.

# Claudio G16 3,8× mais rápido, lances idênticos (Claude na nuvem, 2026-10-08)

Base: os PRs #1–#3 do ChatGPT (merge `0f9d3d2`), autorizado pelo Martim. Só o C++ mudou
(`claudio/fast/claudio_fast.cpp`); a rede, o Python e as definições do G16 ficaram iguais.

## Medição (ABBA, mesmo núcleo, 1 thread, `claudio/bench_velocidade.py`)

| G16, jogos 0–6 do BO30 (1497 decisões) | C++ original `d5e6fda` | Agora |
|---|---:|---:|
| ms por decisão | 168,1 | 44,5 |
| s por jogo | 41,9 | 11,1 |
| jogos/s por núcleo | 0,024 | 0,090 |
| lances diferentes (8 execuções) | – | 0 |

Os 30 jogos completos (5848 decisões) também deram 0 lances diferentes. Instruções do motor
(callgrind, 260 chamadas reais): original 33,8 mil milhões, PRs #1–#3 23,7, agora 8,8.
Números em `claudio/results/velocidade-g16.json`.

## Alterações (todas devolvem exatamente o mesmo: valores, ordem dos lances, linhas, nós)

- Legalidade pelas cravações e xeques (`Pos::guard`/`legal_by`), sem fazer o lance; roque e en
  passant continuam a fazê-lo. `any_legal` tenta primeiro os passos do rei.
- `type_at` sem ciclo, `attacked()` com saída antecipada, `make()` sabendo que peça está em cada casa.
- Quiescência: se "não capturar" já chega ao beta, devolve 0 sem gerar capturas.
- Pesquisa funda: ordenação preguiçosa (o 1.º lance por seleção, o resto só se o nó precisar), e na
  fronteira um sucessor que não pode estar em xeque e corta logo conta os seus 2 nós sem ser feito.
- Mates: geração preguiçosa (pára no primeiro lance que resolve), sem alocações, e no último lance só
  se fazem lances que podem dar xeque (filtro conservador: descobertos, linhas livres, cavalo/peão).
- Geração de lances num buffer da pilha (até 16 peças por lado; acima disso, a forma antiga).

## Verificação

- `claudio/test_fast_exactness.py`: respostas gravadas do C++ original em 281 posições (jogos reais
  do G16, aleatórias, especiais e ilegais), 6171 chamadas + contagens exatas de nós. Três versões
  sabotadas de propósito falham o teste. Gravar outra vez: `--record --build PASTA`.
- Caça adversarial com 6 agentes: só diferenças em tabuleiros impossíveis (dois reis da mesma cor,
  en passant sem peão, xeque duplo resolvido por en passant de um cavalo), todas corrigidas.
  Pré-condição que fica: cada casa tem no máximo uma peça e uma cor (o que o python-chess garante).

## No PC (Windows/MSVC)

    cd claudio\fast; python setup.py build_ext --inplace --force; cd ..\..
    python -m pytest claudio/test_fast_exactness.py -q
    python claudio/bench_velocidade.py --a git:d5e6fda --b claudio/fast --games 0:6 --rounds 2
