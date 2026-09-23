# Launch V3 training with the opponent mixture, detached.
#   75% self-play, 20% TinyOpponent, 4% Fairy-Stockfish 1000, 1% Stockfish 11 1500
# A NEW campaign from the stopped Rodent campaign (its checkpoint is left untouched).
# --compile-learner needs MSVC on PATH, so the run starts inside vcvars64 with
# English compiler output (PyTorch cannot decode localised messages).
# A completed campaign auto-publishes the model to the site; history/<run>/STOP
# stops gracefully without publishing.
param([int]$Workers = 8, [int]$Games = 100000, [switch]$Resume, [switch]$NoCompile, [switch]$CudaGraphs, [switch]$BatchSearch)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = 'C:\Users\Martim\AppData\Local\Programs\Python\Python313\python.exe'
$vcvars = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'
$campaign = 'checkpoints\cognitive-mix-20260923'
$active = Get-CimInstance Win32_Process -Filter "name='python.exe'" |
    Where-Object { $_.CommandLine -match 'train_(cognitive|survival)\.py' }
if ($active) { throw 'Another training process is still running.' }
if ((Test-Path (Join-Path $root "$campaign\latest.pt")) -and -not $Resume) {
    throw 'The campaign exists already: use -Resume.'
}
$source = if ($Resume) { "$campaign\latest.pt" } else { 'checkpoints\cognitive-rodent-100k-20260922\latest.pt' }
$arguments = @('train_cognitive.py', '--resume', $source,
    '--opponent-path', 'engines\stockfish-11\stockfish-11-win\Windows\stockfish_20011801_x64_bmi2.exe',
    '--opponent-elo', '1500', '--opponent-move-time', '0.002',
    '--mix-self', '0.75', '--mix-tiny', '0.20', '--mix-fairy', '0.04',
    '--fairy-path', 'engines\fairy-stockfish\fairy-stockfish_x86-64-bmi2.exe', '--fairy-elo', '1000',
    '--games', "$Games", '--game-cap', "$Games", '--workers', "$Workers",
    '--checkpoint-dir', $campaign, '--log-every', '25',
    '--resign', '--resign-min-non-loss', '0', '--capture-target-weight', '0.05',
    '--objective', 'shaped', '--victory-weight', '0.5', '--kd-weight', '0.75')
if (-not $Resume) { $arguments += '--new-campaign' }
if ($CudaGraphs) { $arguments += @('--learner-device', 'cuda', '--cuda-graphs') }
elseif (-not $NoCompile) { $arguments += '--compile-learner' }
if ($BatchSearch) { $arguments += '--batch-search' }
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$stdout = Join-Path $root "history\mix-$stamp.stdout.log"
$stderr = Join-Path $root "history\mix-$stamp.stderr.log"
$quoted = ($arguments | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' '
$command = "set VSLANG=1033&& set PYTHONUTF8=1&& call `"$vcvars`" >nul 2>nul && `"$python`" $quoted"
$process = Start-Process -FilePath 'cmd.exe' -ArgumentList '/d', '/c', "`"$command`"" -WorkingDirectory $root `
    -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
[pscustomobject]@{ pid = $process.Id; stdout = $stdout; stderr = $stderr; arguments = $arguments
    startedAt = (Get-Date).ToUniversalTime().ToString('o') } | ConvertTo-Json -Depth 3 |
    Tee-Object -FilePath (Join-Path $root 'history\mix-process.json')
