# G16 v2 — checkpoint 600

Versão experimental baseada exclusivamente em `claudio-grande-600.pt` do bloco local de 1000 jogos. O original foi preservado. Empate vale 0,5 na decisão e na classificação.

## Modelos

`models/g16-v2.pt` contém os pesos do checkpoint 600, a rede compacta, a política de escolha aprendida, o livro de experiência e as definições dos módulos.

- `all`: todos os módulos, incluindo a política aprendida forçada (variante pedida pelo Martim).
- `full`: mesma rede compacta, mas a política só intervém se passar a calibração. Nesta experiência a calibração conservou a ordem original.
- `teacher`: módulos v2 com a rede original de 809 533 parâmetros, para separar o efeito da compactação.
- `finish`: só as otimizações de execução e o conversor de finais.
- `exact`: só as otimizações de execução, para exigir os mesmos lances do original.

A rede compacta tem 510 525 parâmetros. Foi treinada nos encodings, contextos, memória e avaliações do próprio checkpoint 600; nenhuma avaliação do Stockfish ensinou o modelo. A política aprende preferências da própria pesquisa, usando as avaliações/cálculos já disponíveis na raiz como referência. O livro foi feito com 80 partidas do próprio jogador; só os seus lances contam como alvos positivos.

## Jogar por UCI

Compilar o módulo adicional (o C++ original não é alterado):

```powershell
Set-Location claudio/v2/native
python setup.py build_ext --inplace --force
Set-Location ../../..
python claudio/v2/uci.py --checkpoint claudio/v2/models/g16-v2.pt --variant all
```

O build Windows usa AVX2 e `/fp:strict`, validado no PC do Martim. Em Linux usa `-ffp-contract=off`. Os módulos congelados em `claudio/base` e o módulo original em `claudio/fast` também têm de estar compilados, como nos scripts normais do Claudio.

O UCI devolve `bestmove 0000` com a razão do resultado quando a partida já terminou, em vez de continuar a mover peças.

## Módulos

1. Testes de fim de jogo e repetição agrupados em C++, mantendo o histórico completo.
2. Cabeças da rede preparadas uma vez para inferência; trajetória, encodings, latents e estados GRU guardados para preparar a aprendizagem sem repetir toda a passagem. `inference.cached_surprises` é usado na geração de experiência.
3. Política própria de ordenação, com a ordem original como referência e uma correção aprendida. `all` força essa correção; `full` respeita a validação. Não remove candidatos legais da pesquisa original.
4. Rede compacta formada a partir dos próprios pesos e ensinada pelo próprio G16. A arquitetura recorrente, os valores das peças e o modelo de segurança do rei são preservados.
5. Livro próprio com a regra Bom: vitória e posição pelo menos equivalente à vista contrária pelo próprio modelo, 12 meios-lances depois. O lookup corrige a diferença entre o en passant legal do livro e o en passant bruto do FEN nativo. O atalho rápido requer evidência estatística e verificações táticas; não foi acionado nesta amostra por falta de evidência suficiente.
6. Conversão de finais com busca AND/OR de mate: todos os lances legais, incluindo peões distantes. Primeiro uma busca curta, depois um orçamento maior apenas nos casos elegíveis. A vantagem material só sugere onde investir cálculo; só uma prova de mate escolhe a continuação especial.

## Fim das partidas

Mate, afogamento, insuficiência de material, repetição tripla/quintupla e regras dos 50/75 lances são reconhecidos. Os pedidos de empate disponíveis são aceites pela rotina de partidas. Há ainda uma prova conservadora para posições mortas com peões permanentemente bloqueados: só reis e peões, sem qualquer possibilidade de mover/capturar peões ou de um rei alcançar um peão.

Isto não é um reconhecedor completo de todas as posições mortas possíveis. Dois cavalos contra rei não são adjudicados automaticamente como insuficientes. O limite de 300 meios-lances continua a ser `max_plies`/inacabado, nunca um empate fictício.

Pontos = V + 0,5 E. O Score principal divide pelos jogos agendados; inacabados não somam pontos. Os dados incluem também o Score só das partidas concluídas. Estes testes não estabelecem um limite universal de 2% de perda.

## Reproduzir a comparação

```powershell
python claudio/v2/evaluate.py --mode sf --checkpoint claudio/v2/models/source-600.pt --assets claudio/v2/models --out claudio/results/v2-comparison --engine CAMINHO_STOCKFISH11 --games 100 --workers 2 --variants baseline,all,full,teacher
python claudio/v2/benchmark.py --checkpoint claudio/v2/models/source-600.pt --assets claudio/v2/models --positions 96
```

O benchmark requer a experiência completa de geração. Os logs/PGNs e os resultados desta experiência estão em `claudio/results/v2-checkpoint600/`. Para gerar novamente os dados de aprendizagem:

```powershell
python claudio/v2/prepare.py collect --checkpoint claudio/v2/models/source-600.pt --out EXPERIMENTO --engine CAMINHO_STOCKFISH11 --games 80 --workers 2
python claudio/v2/prepare.py book --checkpoint claudio/v2/models/source-600.pt --out EXPERIMENTO
python claudio/v2/prepare.py dataset --checkpoint claudio/v2/models/source-600.pt --out EXPERIMENTO --positions 900
python claudio/v2/prepare.py learn --checkpoint claudio/v2/models/source-600.pt --out EXPERIMENTO --steps 1200 --policy-steps 600
python claudio/v2/calibrate.py --assets EXPERIMENTO
```

Testes específicos: `python -m pytest claudio/v2/test_v2.py -q -o addopts=''` com `G16_V2_SOURCE` a apontar para o checkpoint real. A suite geral conserva a falha preexistente em `tests/test_model.py`, pelo fallback clássico em `tonkai/cognitive_agent.py:945`; o v2 não usa essa tabela para os seus valores de peças.
