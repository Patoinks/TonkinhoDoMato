# Preparação da publicação do checkpoint final legado

Este roteiro não autoriza nem executa a paragem de processos. A preparação
feita nesta tarefa não alterou os pesos públicos, `dist`, treino ou Hosting.

## Estado verificado em 22/09/2026

- Treino legado: PID **30676**, `train.py --resume checkpoints/latest.pt --games
  20000`, iniciado com 2500 jogos. Checkpoint final esperado: **22500 jogos**;
  métricas do JSONL: **20000 jogos nesta sessão**.
- Histórico: `history/20260922T162109Z-94d17041.jsonl`.
- Lançamento: `history/launch-20260922T162105Z/launch.json`.
- Publicador: PID **21812**, `node scripts/publish-tonkai-training-status.mjs
  --history ... --launch ... --pid 30676 --watch --interval 60000 --deploy`.
- Publicação segura: `C:\Users\Martim\Desktop\Github\ANW-TONKAI-RELEASE-20260922`,
  branch `codex/tonkai-history-release`, base e `origin/main` ainda
  `cde26d7d004cc9039c8dd29bef433df1eb703863` nesta verificação.
- Desenvolvimento: `C:\Users\Martim\Desktop\Github\ANW-TONKAI-PAGE`. Preservar
  todas as alterações. Não copiar a árvore inteira: contém trabalho anterior.
- Instruções aplicáveis: `C:\Users\Martim\AGENTS.md` pede preservar alterações,
  escopo estreito, `apply_patch` para editar e validação apropriada. Não foram
  encontrados AGENTS adicionais nas duas árvores web ou no TonkAI.

## Evitar a corrida com o publicador

O publicador escreve `public/tonkai/training-status.json` e a cópia em `dist`.
Quando o conteúdo muda, executa **`firebase deploy --only hosting` de todo o
`dist` da release**, sem exportar checkpoint nem fazer build. Portanto pode
republicar pesos antigos ou um diretório que está a ser reconstruído.

Não substituir assets nem fazer build/deploy enquanto ele estiver ativo.
Após o PID 30676 terminar, o loop publica o estado final e **sai naturalmente**.
Esperar também pelo fim dos descendentes `cmd.exe`/`node.exe` de Firebase. A
presença de `final=true` no JSONL, isoladamente, não prova o fim do processo:
`train.py` ainda faz a gravação final e os dumps depois dessa linha.

Inspeção, sem terminar processos:

```powershell
Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -in @(30676, 21812) -or $_.ParentProcessId -eq 21812 -or
    ($_.Name -eq 'node.exe' -and $_.CommandLine -match 'firebase.*deploy')
} | Select-Object ProcessId, ParentProcessId, Name, CommandLine
```

Se o publicador não sair depois do treino, confirmar primeiro a identidade do
PID 21812, a ausência do PID 30676 e de um deploy filho em curso. Só então a
coordenação pode terminar **apenas o publicador identificado** com
`Stop-Process -Id 21812`; não usar `taskkill /T`, matar todos os Node/Python nem
parar o treino. Não foi executada nenhuma destas paragens nesta preparação.

## Exportação local protegida

Quando treino e publicador tiverem terminado:

```powershell
Set-Location 'C:\Users\Martim\Desktop\Github\TonkAI'
& .\scripts\prepare_legacy_web_rollout.ps1
```

O script rejeita uma sessão incompleta, processos ainda ativos, checkpoint
inexistente, contagens inesperadas e cópia instável. Cria uma pasta única em
`perf_results/export/legacy-final-<UTC>/`, com:

- `checkpoint.pt`: cópia estável do checkpoint final.
- `tonkai.bin`, `tonkai.json`, `golden.json`, `ENCODING.md`: exportação Python.
- `rollout.json`: hashes e proveniência; marca browser/deploy como não feitos.

**Não copia ficheiros para o website, não publica e não para processos.** A
existência do checkpoint é obrigatória: `export_web.py` sozinho inicializa um
modelo aleatório se receber um caminho inexistente.

## Atualizar e verificar a release

Substituir `<pasta emitida pelo script>` pelo caminho realmente obtido. Antes de
executar, confirmar de novo que nenhum publicador/deploy está ativo e que
`origin/main`/Hosting não receberam trabalho alheio entretanto.

```powershell
$releaseRoot = 'C:\Users\Martim\Desktop\Github\ANW-TONKAI-RELEASE-20260922'
$exportRoot = '<pasta emitida pelo script>'
$trainerRoot = 'C:\Users\Martim\Desktop\Github\TonkAI'
foreach ($name in @('tonkai.bin', 'tonkai.json', 'golden.json', 'ENCODING.md')) {
    Copy-Item -LiteralPath (Join-Path $exportRoot $name) -Destination (Join-Path $releaseRoot "public\tonkai\$name")
}
Set-Location $releaseRoot
node --import tsx --test tests/tonkai-golden.test.ts
if ($LASTEXITCODE -ne 0) { throw 'Golden gate failed; do not publish.' }
npm test
if ($LASTEXITCODE -ne 0) { throw 'Node tests failed; do not publish.' }
node scripts/publish-tonkai-training-status.mjs --history "$trainerRoot\history\20260922T162109Z-94d17041.jsonl" --launch "$trainerRoot\history\launch-20260922T162105Z\launch.json" --pid 30676 --state completed
if ($LASTEXITCODE -ne 0) { throw 'Final status preparation failed.' }
node scripts/sync-tonkai-win.mjs --source "$trainerRoot\history\longest_external_win.pgn"
if ($LASTEXITCODE -ne 0) { throw 'Longest-win synchronization failed.' }
npm run build
if ($LASTEXITCODE -ne 0) { throw 'Build failed; do not publish.' }
```

O teste golden compara nove posições, 42 candidatos e 20 previsões com Python,
incluindo logits/utilidades/probabilidades e SHA256. Não afrouxar tolerâncias ou
alterar o vencedor esperado para esconder uma divergência. Saturação/empates
podem exigir diagnóstico numérico real, sem mudar a política do modelo.

O browser **não contém um hash antigo fixo**: `src/tonkai/network.ts` compara o
SHA256 de `tonkai.bin` com `tonkai.json`. O golden também compara o hash desses
ficheiros com `golden.json`. Atualizar os quatro artefactos juntos; não alterar
o teste de checksum nem desativar a verificação. Os assets usam revalidação de
cache. Uma partida já aberta conserva a instância anterior até recarregar.

O teste browser na release ainda é a versão antiga que espera painéis removidos.
Antes da validação final, transferir **só o teste atualizado** de
`ANW-TONKAI-PAGE/tests/browser/tonkai.spec.ts` para a release (e quaisquer fixtures
que uma comparação mostre necessárias); não substituir o runtime por ficheiros
do dev. Depois:

```powershell
$env:CI = '1'
npm run test:browser
if ($LASTEXITCODE -ne 0) { throw 'Browser validation failed; do not publish.' }
```

O servidor de testes usa a porta 4175. `CI=1` evita reutilizar silenciosamente
um Vite de outra árvore; se a porta estiver ocupada, usar outro porto/configuração
de teste ou coordenar o servidor existente. Não matar processos não identificados.

Conferir hashes de `dist/tonkai/tonkai.bin` e `public/tonkai/tonkai.bin`, manifesto
com `games_trained=22500`, JSON de treino `completed`, e lista de assets fora
TonkAI preservada. O texto do hash em `TONKAI_WEB.md` deve ser atualizado para o
novo hash real, sem mudança na política de inferência.

## Publicação, apenas na etapa coordenada

Depois dos testes e da confirmação da release concreta:

```powershell
Set-Location 'C:\Users\Martim\Desktop\Github\ANW-TONKAI-RELEASE-20260922'
firebase deploy --only hosting --project anw-games --non-interactive
```

Validar no site `/tonkai/tonkai.json`, `/tonkai/tonkai.bin` e
`/tonkai/training-status.json`, com cache-busting e hash dos bytes obtidos.
Confirmar a rota `/tonkai`, uma jogada legal e a ausência de erros de checksum.
Não publicar regras Firestore. Não retomar o publicador antigo.

O treino novo Stockfish1600 deve usar saídas/checkpoints próprios. A publicação
do checkpoint legado deve ficar concluída antes da passagem para esse treino;
o exportador atual é exclusivo da arquitetura legada, não da V2.
