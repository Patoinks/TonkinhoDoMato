# Aceleração C++ do Tonkinho do Mato

O módulo `tonkai_fast` acelera as regras do ator V3: proteção do rei,
trocas/recapturas, finais e reclamações de empate, e codificação de posições
candidatas. A codificação também é usada na previsão da resposta adversária e
nos exemplos de resposta do learner. Os alvos auxiliares de capturas futuras
também têm implementação C++, preservando a identidade das peças em roques,
en passant e promoções.
Preserva a ordem das jogadas do python-chess e as mesmas operações numéricas.

A rede, gradientes e otimizador continuam em PyTorch. O recolhimento de partidas,
replay, experiências e controlo do treino continuam em Python. Esta integração
não altera recompensas, valores das peças, pesos, adversário ou orçamento de busca.
Não é uma conversão integral da aplicação para um executável C++.

## Compilar no Windows

Com Python e Visual Studio C++ Build Tools instalados:

```powershell
python -m pip install -r requirements.txt
python setup.py build_ext --inplace
```

Usar o mesmo Python que executa o treino. São geradas duas extensões na raiz:
`tiny_opponent` (adversário) e `tonkai_fast` (regras V3). O módulo V3 exige C++20
e desativa contração/reassociação de operações em vírgula flutuante.
Fechar processos que tenham as extensões carregadas antes de as recompilar;
o Windows não permite substituir uma DLL em uso.

Alternativa CMake, numa Developer PowerShell/Command Prompt do Visual Studio:

```powershell
cmake -S tiny_opponent -B build/cmake-native -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake-native --config Release
```

Se existirem vários Pythons, passar `-DPython_EXECUTABLE=C:/caminho/python.exe`.
Ambas as extensões são compiladas por este caminho.

## Seleção e diagnóstico

Por defeito, usa C++ se a extensão importar e Python se não estiver instalada.
`launch.json` e as métricas de `train_cognitive.py` incluem `rules_backend`.
Para exigir C++ e falhar imediatamente se não estiver disponível:

```powershell
$env:TONKAI_RULES_BACKEND = 'cpp'
```

`python` força a referência; `auto` repõe a seleção automática. Os processos
Windows de recolha de partidas herdam esta variável. Esta opção muda a execução
das regras, não o objetivo do modelo. A extensão suporta xadrez normal; recusa
explicitamente Chess960 para não aplicar regras de roque incorretas.

## Verificação e benchmark

```powershell
python -m pytest
python scripts/benchmark_cpp.py --checkpoint checkpoints/cognitive-rodent-100k-20260922/latest.pt --games 6 --output perf_results/cpp-completion-20260923.json
```

O benchmark faz aquecimento e quatro ensaios Python/C++/C++/Python, sempre a
partir do mesmo checkpoint e sementes. Executa jogos, aprendizagem e experiências
agendadas em memória. Compara exatamente decisões, pontuações, sequências,
previsões, pesos, otimizador, replay, memórias e estados aleatórios.
Não grava checkpoints, não publica no site e não conta para a campanha.

Usa um ator CPU e TinyOpponent determinístico para tornar a comparação repetível.
Os jogos/s deste teste não são uma previsão do rendimento com oito atores/Rodent,
nem uma avaliação de Elo. O relatório JSON contém condições e hash da origem.

### Resultado local de 23/09/2026

Compilação MSVC por setuptools e por CMake concluída; suíte completa:
**398 testes passaram**. Inclui partidas completas e probabilidades de resposta
iguais entre as duas implementações, nos objetivos survival/shaped/victory.

No checkpoint `7c1d34d16929c3542a6af31569dd58dc29b57bda92cbb040ca4117b1c824617e`,
seis jogos por ensaio, 421 meios-lances, 12 updates e duas experiências:

| Ensaio | Jogos/s | Meios-lances/s |
| --- | ---: | ---: |
| Python A1 | 0,869 | 60,97 |
| C++ B1 | 1,960 | 137,55 |
| C++ B2 | 1,927 | 135,19 |
| Python A2 | 0,855 | 59,97 |

Ganho médio **2,255×** contra a referência Python. Este ganho inclui o port do
Claude e a conclusão da integração; não é um ganho adicional de 2,255× sobre
a versão que o Claude já tinha acelerado. Todos os quatro ensaios produziram
estado de aprendizagem exatamente igual, e o checkpoint original ficou intacto.
Dados: `perf_results/cpp-completion-20260923.json`.

Também passou um ensaio de integração com dois processos Windows reais usando
C++ e Rodent 800 nominal: duas partidas limitadas a 12 meios-lances, encodings
conferidos contra Python e encerramento normal dos atores/engines. Este ensaio
curto valida a integração, não o rendimento de uma campanha longa.
Dados: `perf_results/cpp-pool-smoke-20260923.json`.

### Segunda otimização: remover trabalho repetido

Além das regras já convertidas, esta etapa:

- usa finais nativos no ciclo de jogo e na medida posicional;
- valida sequências recuperadas com cópias do tabuleiro C++ que mantêm o histórico;
- reutiliza as representações fixas de tipo/casa/cor das peças, invalidando a
  cache quando os pesos mudam; mantém o mesmo tamanho e ordem dos cálculos da rede;
- guarda a redundância das memórias enquanto o arquivo permanece inalterado;
- serializa diretamente o estado para os atores, sem cópias intermédias nem
  replay protegido, que pertence exclusivamente ao learner;
- produz encodings de respostas e alvos futuros de capturas em C++.

Não agrupa a avaliação de peças de posições diferentes: essa alternativa foi
descartada porque alterava alguns bits dos resultados em vírgula flutuante.

Comparação adicional C++ anterior/novo/novo/anterior, 12 jogos por ensaio,
819 meios-lances, 24 updates e duas experiências em cada execução:

| Implementação | Jogos/s (média de dois ensaios) |
| --- | ---: |
| C++ anterior | 1,974 |
| C++ otimizado | 2,312 |

Ganho adicional **17,1%**, com hash idêntico de todo o estado de aprendizagem,
trajetórias, decisões e estados aleatórios. O checkpoint original não foi alterado.
Dados: `perf_results/cpp-phase2-final.json`. A cópia da fonte de referência está
em `perf_results/cpp-phase2-baseline`; ambos usam as mesmas operações nativas
originais, com as novas funções disponíveis apenas na versão otimizada.

Para repetir a comparação ou medir o ciclo com Rodent:

```powershell
python scripts/benchmark_cpp_iteration.py --baseline perf_results/cpp-phase2-baseline --checkpoint checkpoints/cognitive-rodent-100k-20260922/latest.pt --games 12 --output perf_results/native-comparison.json
python scripts/benchmark_rodent_pool.py --checkpoint checkpoints/cognitive-rodent-100k-20260922/latest.pt --launch history/cognitive-20260922T220738Z-f7509778/launch.json --workers 8 --games 64 --output perf_results/rodent-pool.json
```

O segundo teste lê também as definições de desistência da campanha e inclui
aprendizagem, medições das peças e experiências. Não escreve checkpoints nem
histórico da campanha. Rodent por tempo e a ordem de conclusão dos atores são
não determinísticos: partidas e duração podem mudar entre ensaios. O rendimento
depois da primeira vaga exclui arranque dos atores, mas inclui a drenagem final.
Não se deve interpretar diferenças entre esses jogos como ganho de Elo.

Verificação final desta etapa: **409 testes passaram**. O teste do payload
confirma que atualizar os pesos e memórias depois de `submit` não altera o
snapshot recebido pelo ator. O teste da cache de retenção compara rejeições,
substituições e carregamento de estado contra o recálculo integral.

Ensaio Rodent 800 nominal, 2 ms, oito atores CPU, 64 jogos por versão e
128 updates, com o mesmo checkpoint e parâmetros:

| Medida | C++ anterior | C++ otimizado |
| --- | ---: | ---: |
| Jogos/s incluindo arranque | 3,734 | 4,433 |
| Jogos/s depois da primeira vaga | 4,904 | 5,967 |
| Meios-lances por jogo | 57,53 | 62,17 |
| Tempo no learner (s) | 12,573 | 9,772 |

A diferença observada após a primeira vaga foi **21,7%**, mas este é um ensaio
curto com partidas diferentes, não uma comparação determinística nem garantia
para uma campanha longa. Os dois checkpoints permaneceram intactos e os atores
e engines encerraram normalmente. Dados:
`perf_results/cpp-phase2-rodent-before.json` e
`perf_results/cpp-phase2-rodent-final.json`.

### Terceira otimização: preservar aprendizagem e eliminar cálculos sem uso

A referência desta etapa é a versão já otimizada da segunda etapa, guardada
em `perf_results/cpp-phase3-baseline`.

- A memória recorrente compara os campos da posição diretamente, em vez de
  formatar FEN a cada observação. Histórico, roque, en passant legal e relógios
  continuam a distinguir posições e substituições de tabuleiro.
- A reconstrução de sequências para aprender respostas passa por C++ com
  validação de cada jogada. A ordem das respostas continua a ser a do python-chess.
- Xeques das respostas candidatas são calculados em C++ antes da mesma ordenação.
  O número de nós, as alternativas selecionadas e o desempate não mudam.
- As medições de mobilidade e ameaças usam contagens inteiras em C++, com as
  mesmas identidades de peças em roques, en passant e promoções.
- As médias da memória reutilizam o denominador comum, preservando os tipos
  numéricos, a multiplicação e a ordem de soma de `numpy.average`.
- O learner evita produzir saídas recorrentes que nunca são consumidas por
  nenhuma perda. Continua a processar todo o burn-in e a janela de aprendizagem;
  a rede target continua a processar também todo o horizonte futuro. A forma e
  o tamanho do batch do encoder permanecem iguais.

Mantêm-se as **48 posições de aprendizagem, 16 de burn-in, duas atualizações
por jogo**, amostras de replay, exploração, adversário e recompensas do checkpoint.
Não há quantização nem alteração de arquitetura ou dos pesos de produção.

A suíte completa passou **413 testes**. Um teste adicional compara exatamente
as saídas utilizadas e todos os gradientes de entrada/parâmetros entre o unroll
integral e o otimizado, incluindo padding e dependências de longo prazo. Outros
testes comparam FEN, ordem de jogadas, xeques e estatísticas nativas com Python.

Comparação alternada da versão anterior/atual/atual/anterior, com 20 jogos,
1491 meios-lances, 741 decisões, 40 atualizações e duas experiências por ensaio:

| Versão | Jogos/s, média |
| --- | ---: |
| Final da segunda etapa | 2,212 |
| Final da terceira etapa | 2,365 |

Ganho adicional medido **6,9%**. Os quatro ensaios deram o mesmo SHA-256 de
trajetórias e estado completo de aprendizagem:
`b5c20d6446df685a24cf8e0bd2e5680ac08587bc217675b567aa865f6485bbac`.
O checksum do checkpoint de produção também permaneceu igual.
Dados: `perf_results/cpp-phase3-verified.json`.

Ensaio de integração desta etapa com oito atores e Rodent, **128 jogos por
versão**, 256 updates e 14 experiências:

| Medida | Versão anterior | Versão atual |
| --- | ---: | ---: |
| Jogos/s, incluindo arranque | 4,996 | 5,322 |
| Jogos/s, depois da primeira vaga | 5,895 | 6,369 |
| Meios-lances/jogo | 61,51 | 58,60 |
| Meios-lances/s, incluindo arranque | 307,31 | 311,86 |

As partidas diferem, como esperado num adversário por tempo e recolha assíncrona.
Parte do aumento de jogos/s vem de partidas mais curtas; por isso, a evidência
controlada do ganho de execução é a comparação determinística de 6,9% acima.
Estes ensaios não medem melhoria de força e não contam para a campanha.
Dados: `perf_results/cpp-phase3-rodent-before.json` e
`perf_results/cpp-phase3-rodent-after.json`.
