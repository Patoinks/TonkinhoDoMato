# Prepare an immutable legacy export after completion. Never deploys or stops a process.
[CmdletBinding()]
param(
    [string]$TrainingRoot = (Join-Path $PSScriptRoot '..'),
    [string]$RunId = '20260922T162109Z-94d17041',
    [string]$LaunchId = 'launch-20260922T162105Z'
)

$ErrorActionPreference = 'Stop'
$trainingDirectory = [IO.Path]::GetFullPath($TrainingRoot)
$launchDirectory = Join-Path $trainingDirectory "history\$LaunchId"
$launch = Get-Content -LiteralPath (Join-Path $launchDirectory 'launch.json') -Raw | ConvertFrom-Json
$historyPath = Join-Path $trainingDirectory "history\$RunId.jsonl"
$rows = Get-Content -LiteralPath $historyPath | ForEach-Object { $_ | ConvertFrom-Json }
$finalMetrics = $rows | Where-Object kind -eq 'metrics' | Select-Object -Last 1
if (!$finalMetrics -or !$finalMetrics.final -or $finalMetrics.games -ne $launch.additional_games) {
    throw 'The requested legacy run has not recorded all planned games and final=true.'
}

$processes = @(Get-CimInstance Win32_Process)
if ($processes | Where-Object ProcessId -eq $launch.pid) {
    throw "Training PID $($launch.pid) still exists. Wait for normal process exit."
}
$publishers = @($processes | Where-Object {
    $_.Name -eq 'node.exe' -and $_.CommandLine -match 'publish-tonkai-training-status\.mjs'
})
if ($publishers.Count) {
    throw "A status publisher is still running (PID $($publishers.ProcessId -join ', ')). Wait for its final deployment and normal exit."
}
$deployments = @($processes | Where-Object {
    $_.Name -in @('node.exe', 'cmd.exe') -and $_.CommandLine -match 'firebase' -and
    $_.CommandLine -match 'deploy' -and $_.CommandLine -match 'hosting'
})
if ($deployments.Count) {
    throw "A Hosting deployment is still running (PID $($deployments.ProcessId -join ', '))."
}

$checkpointPath = Join-Path $trainingDirectory 'checkpoints\latest.pt'
if (!(Test-Path -LiteralPath $checkpointPath -PathType Leaf)) {
    throw "Missing checkpoint: $checkpointPath. export_web.py alone would export random weights."
}
$expectedGames = [int]$launch.resume_games + [int]$launch.additional_games
$expectedSessionGames = [int]$launch.additional_games
$pythonPath = $launch.python
if (!(Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "The run's Python executable is missing: $pythonPath"
}
$exportDirectory = Join-Path $trainingDirectory ("perf_results\export\legacy-final-" + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ'))
New-Item -ItemType Directory -Path $exportDirectory | Out-Null
$snapshotPath = Join-Path $exportDirectory 'checkpoint.pt'
$beforeHash = (Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256).Hash
Copy-Item -LiteralPath $checkpointPath -Destination $snapshotPath
$snapshotHash = (Get-FileHash -LiteralPath $snapshotPath -Algorithm SHA256).Hash
$afterHash = (Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256).Hash
if ($beforeHash -ne $snapshotHash -or $afterHash -ne $snapshotHash) {
    throw 'The checkpoint changed while copying; do not use this export directory.'
}

$validateCheckpoint = @'
import json, sys, torch
payload = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
assert payload.get('games') == int(sys.argv[2]), 'Unexpected cumulative games'
assert payload.get('metrics', {}).get('games') == int(sys.argv[3]), 'Unexpected session games'
assert 'encoder.0.weight' in payload['model'], 'Not a legacy model'
print(json.dumps({'games': payload['games'], 'steps': payload['steps'], 'session_games': payload['metrics']['games']}))
'@
& $pythonPath -c $validateCheckpoint $snapshotPath $expectedGames $expectedSessionGames
if ($LASTEXITCODE -ne 0) { throw 'Final checkpoint validation failed.' }

Push-Location $trainingDirectory
try {
    & $pythonPath (Join-Path $trainingDirectory 'export_web.py') --checkpoint $snapshotPath --out $exportDirectory
    if ($LASTEXITCODE -ne 0) { throw 'Legacy browser export failed.' }
}
finally { Pop-Location }

$manifest = Get-Content -LiteralPath (Join-Path $exportDirectory 'tonkai.json') -Raw | ConvertFrom-Json
$golden = Get-Content -LiteralPath (Join-Path $exportDirectory 'golden.json') -Raw | ConvertFrom-Json
$binaryPath = Join-Path $exportDirectory 'tonkai.bin'
$binaryHash = (Get-FileHash -LiteralPath $binaryPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($manifest.sha256 -ne $binaryHash -or $golden.sha256 -ne $binaryHash -or
    $manifest.bytes -ne (Get-Item -LiteralPath $binaryPath).Length -or
    $manifest.games_trained -ne $expectedGames -or $manifest.input_size -ne 782 -or
    $golden.positions.Count -ne 9 -or $golden.move_ranking.moves.Count -ne 42 -or
    $golden.predict_distribution.moves.Count -ne 20) {
    throw 'Export contract/hash validation failed.'
}

$record = [ordered]@{
    prepared_utc = [DateTime]::UtcNow.ToString('o')
    history = $historyPath
    launch = (Join-Path $launchDirectory 'launch.json')
    checkpoint = $snapshotPath
    checkpoint_sha256 = $snapshotHash.ToLowerInvariant()
    games_trained = $expectedGames
    session_games = $expectedSessionGames
    model_sha256 = $binaryHash
    export_directory = $exportDirectory
    browser_equivalence_tested = $false
    deployed = $false
}
$record | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $exportDirectory 'rollout.json') -Encoding utf8
Write-Output "Prepared only: $exportDirectory"
Write-Output 'Next: copy the four web export files together, run the golden/browser gates, then build and deploy Hosting separately.'
