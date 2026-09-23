# Ensaios adicionais da rede — 23 de setembro de 2026

Resultado: **nenhuma variante ativada no treino normal**. As seis ideias foram
implementadas e medidas numa cópia isolada. As duas variantes mais rápidas
alteraram a trajetória numérica; as que preservaram exatamente a aprendizagem
não demonstraram um ganho estável no teste de confirmação.

O Claude foi avisado em `WORKING.md` antes de começar e recebeu uma atualização
com os resultados. Nenhum ficheiro reservado por ele foi editado. Estes ensaios
não pararam nem lançaram campanhas, não publicaram modelos e não alteraram o
checkpoint de produção.

## Checkpoint de segurança

Diretório local: `perf_results/network-20260923/`.

- `head.zip`: código versionado no commit `2e272cd5299b6dc7d2498cceebfbf614aa87d8e4`.
- `working-tree.patch`: alterações locais existentes no início, incluindo trabalho
  ainda não gravado do outro agente; não foi feito commit dessas alterações.
- `baseline/`: cópia estável dos 39 ficheiros de implementação usados pelos ensaios.
- `input.pt`: cópia dos pesos, optimizer, replay e RNG da campanha.
- `checkpoint.json`: origem, commit e checksums individuais da cópia.

SHA-256 dos pesos originais e da cópia:
`7c1d34d16929c3542a6af31569dd58dc29b57bda92cbb040ca4117b1c824617e`.
O manifesto e resultados pequenos ficam versionados; pesos e arquivo completo
ficam apenas no disco local. O snapshot inclui as alterações locais do Claude
que já existiam nesse instante; não é uma cópia das suas alterações posteriores.

## Método

Cada ensaio usa um processo novo, CPU com uma thread Torch, o mesmo código
guardado, o mesmo checkpoint e o mesmo replay. Corre 10 updates de aquecimento
e depois mede 100 updates. A primeira ronda percorre as variantes pela ordem
normal e depois inversa, com a referência no início e no fim. Confirmações usam
referência/variante/variante/referência. Não há adversário nestas medições.

A comparação inclui rede, target, optimizer, prioridades e RNG do replay,
conhecimento e todas as métricas/perdas dos updates. Apenas os flags de execução
`foreach`/`fused` do optimizer são normalizados na comparação; os momentos e
parâmetros não são normalizados. As trajetórias retidas do replay não mudam num
ensaio de updates, pelo que o comparador guarda os seus IDs/prioridades em vez de
duplicar todo o conteúdo imutável. Diferenças nos pesos não são uma medição de
perda de força, mas também não demonstram que a qualidade ficou igual.

## Resultados

Valores são médias de duas execuções de 100 updates, após aquecimento.
Ganho = tempo de referência / tempo da variante − 1.

| Variante | Tempo (s) | Ganho no update | Estado exatamente igual? |
| --- | ---: | ---: | --- |
| Referência da primeira ronda | 4,634 | — | Sim |
| Projeção de entrada da GRU em lote | 4,485 | +3,34% | Não |
| Cache do encoder congelado | 4,735 | −2,12% | Sim neste ensaio |
| Valores target só nos endpoints usados | 4,716 | −1,73% | Não |
| AdamW `foreach` + EMA em lote | 4,691 | −1,20% | Sim |
| Perda de respostas em lote | 4,556 | +1,72% | Sim |
| Buffer de unpack reutilizado | 4,528 | +2,34% | Sim |

Confirmação das duas últimas juntas: **4,416 s de referência → 4,418 s**, ganho
**−0,02%**, estado exatamente igual. Os pequenos ganhos individuais não se
confirmaram no ciclo completo combinado; não devem ser somados.

Ensaio adicional de AdamW `fused` em FP32 no CPU: **4,426 → 4,276 s**, ganho
**3,51%**, mas sem equivalência exata. Após 110 updates, a maior diferença absoluta
num parâmetro foi 0,001557; na GRU foi 0,001782 e no target seletivo 0,000401.
As diferenças aparecem mesmo com a mesma amostragem/RNG. Não houve quantização,
redução de updates, encurtamento de memória ou alteração das recompensas.

Não há aqui uma medição de jogos/s ou de Elo. A escolha é manter os protótipos
desativados e voltar a medir sobre o perfil final do Claude, se estes componentes
continuarem relevantes.

## Implementação e reprodução

`tonkai/network_fastpaths.py` contém os helpers experimentais; não é importado
pelo treino normal. `scripts/benchmark_network_paths.py` copia a referência para
`variants/` e altera apenas essas cópias, com substituições verificadas. A cache
de encoder é apenas experimental e deve ser descartada se os pesos do encoder
forem recarregados; o benchmark usa um trainer novo em cada processo.

Com o checkpoint local acima presente, reservar os outputs em `WORKING.md` e correr:

```powershell
python scripts/benchmark_network_paths.py --updates 100
python scripts/benchmark_network_paths.py --updates 100 --modes reply+buffers --output combined.json
python scripts/benchmark_network_paths.py --updates 100 --modes fused --output fused.json
python -m pytest tests/test_network_fastpaths.py -q
```

Resultados brutos: `results.json`, `combined.json` e `fused.json` no diretório
do checkpoint de segurança. Os cinco testes novos verificam unpack exato e
reutilização, perda/gradientes por sequência, GRU com burn-in e padding em
float64, seleção dos endpoints e chaves/evicção da cache congelada.

Validação final: `python -m pytest -q -o addopts=''` passou **432 testes em
66,13 segundos**. Os checksums do checkpoint original e da cópia foram novamente
comparados no fim e permaneceram iguais.
