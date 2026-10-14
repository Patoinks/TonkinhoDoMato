# G4 Leto

Versão preservada a 14 de outubro de 2026 no repositório da IA de xadrez TonkAI.

Inclui o checkpoint original G4T2 Leto v3, a implementação de referência Python/C++, os módulos do motor web atual, os pesos exportados, o repertório e as provas de finais. O código e os recursos correspondem ao G4 Leto publicado no site. `provenance.json` regista a origem e o SHA256 de cada ficheiro copiado.

## Conteúdo

| Caminho | Conteúdo |
|---|---|
| `models/g4t2-leto-v3.pt` | Checkpoint original, com rede, segurança do rei, valores próprios e configuração |
| `python/claudio/` | Decisão do Leto, aberturas, Pawnomics de referência, carregamento, UCI e scripts de treino |
| `python/claudio/base/` | Base congelada do TonkAI de que esta versão depende |
| `python/claudio/fast/` | Fonte C++ e build do cálculo nativo Python |
| `web/src/tonkai/` | Motor TypeScript: rede recorrente, valores contextuais, segurança do rei, regras, aberturas, Pawnomics v3, finais e pesquisa iterativa |
| `web/native/g4-rules.cpp` | Fonte C++ completa das regras WASM usadas pelo motor web |
| `web/public/tonkai/` | Manifestos, pesos, WASM, livro de aberturas e tabela KPK |
| `web/tests/` | Testes do Leto, Pawnomics v3, pesquisa limitada, histórico e variantes de treino |

O Python preserva a referência original do checkpoint. As melhorias posteriores de Pawnomics v3 e o limite de cinco segundos estão na implementação **web**; não foram retroportadas para o Python nem alteraram o checkpoint.

## Verificar os pesos e ficheiros

A partir desta pasta:

```sh
python verify.py
python verify.py --checkpoint
```

A segunda opção requer PyTorch e verifica que os tensores do checkpoint original são exatamente os mesmos do binário exportado. SHA256 dos pesos web: `d8f9edad5a5352ec86ef50048794dbe22d2b0b6ab354d8366ff38f6190df5c3f`.

## Python

Instalar as dependências do repositório com `python -m pip install -r ../../requirements.txt`. Compilar os dois módulos com um compilador C++20:

```sh
cd python/claudio/base
python setup_claudio.py build_ext --inplace
cd ../fast
python setup.py build_ext --inplace
cd ../../..
python python/claudio/uci.py --checkpoint models/g4t2-leto-v3.pt
```

O UCI Python original decide com o orçamento de pesquisa do checkpoint; não implementa `go movetime`. O limite temporal de cinco segundos é aplicado pelo worker web. Os scripts de treino estão preservados para referência e requerem um adversário configurado em `CLAUDIO_STOCKFISH` ou `CLAUDIO_FAIRY`.

## Motor web

```sh
cd web
npm ci
npm test
npm run typecheck
npm run build
```

`index.ts` exporta o motor e o cliente de pesquisa. Servir `public/tonkai` em `/tonkai` e carregar `loadNetwork('/tonkai/leto.json')`. O worker executa a mesma rede e aprofunda a pesquisa até ao limite, mantendo a última iteração concluída. Livro, lances forçados e provas podem terminar antes. `public/tonkai/leto.json` fixa a configuração, os hashes e os recursos.

Para recompilar o WASM, usar Emscripten com `native/g4-rules.cpp`, C++20, `-O3 -ffp-contract=off --bind -sMODULARIZE=1 -sEXPORT_ES6=1 -sALLOW_MEMORY_GROWTH=1 -sENVIRONMENT=web,node -sSTACK_SIZE=8388608 -sINITIAL_MEMORY=67108864 -sDISABLE_EXCEPTION_CATCHING=0`; uma recompilação exige atualizar os hashes do manifesto. Os binários validados estão incluídos.
