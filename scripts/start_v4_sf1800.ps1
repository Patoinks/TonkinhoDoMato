# Launch v4 training, detached, below-normal priority: every game against Stockfish 11 at -Elo (default 1800),
# with the learned King Safety model, mate calculation and piece values v2.
# Resumes the mix campaign; 20 000 games per run.
# --compile-learner needs MSVC on PATH, so the run starts inside vcvars64 with
# UTF-8 console output (the localised MSVC help otherwise breaks PyTorch decoding).
# A completed campaign auto-publishes the model to the site; history/<run>/STOP
# stops gracefully without publishing.
param([int]$Workers = 6, [int]$Games = 20000, [int]$Elo = 1800, [double]$ThreatExplore = 0, [int]$MateLabels = 0, [int]$ExchangeDepth = 0, [double]$TradeExplore = 0, [int]$ExchangeLabels = 0, [switch]$NoCompile, [switch]$CudaGraphs, [switch]$BatchSearch)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = 'C:\Users\Martim\AppData\Local\Programs\Python\Python313\python.exe'
$vcvars = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'
$campaign = 'checkpoints\cognitive-mix-20260923'
$active = Get-CimInstance Win32_Process -Filter "name='python.exe'" |
    Where-Object { $_.CommandLine -match 'train_(cognitive|survival)\.py' }
if ($active) { throw 'Another training process is still running.' }
$source = "$campaign\latest.pt"
$arguments = @('train_cognitive.py', '--resume', $source,
    '--opponent-path', 'engines\stockfish-11\stockfish-11-win\Windows\stockfish_20011801_x64_bmi2.exe',
    '--opponent-elo', "$Elo", '--opponent-move-time', '0.002',
    '--mix-self', '0', '--mix-tiny', '0', '--mix-fairy', '0',
    '--fairy-path', 'engines\fairy-stockfish\fairy-stockfish_x86-64-bmi2.exe', '--fairy-elo', '1000',
    '--games', "$Games", '--game-cap', '100000', '--workers', "$Workers",
    '--checkpoint-dir', $campaign, '--log-every', '25',
    '--resign', '--resign-min-non-loss', '0', '--capture-target-weight', '0.05',
    '--objective', 'shaped', '--victory-weight', '0.5', '--kd-weight', '0.75',
    '--king-safety-model', 'checkpoints\king_safety\latest.pt',
    '--mate-search-moves', '3', '--mate-defense-moves', '2',
    '--threat-explore', "$ThreatExplore", '--mate-labels', "$MateLabels",
    '--exchange-depth', "$ExchangeDepth", '--trade-explore', "$TradeExplore", '--exchange-labels', "$ExchangeLabels")
if ($CudaGraphs) { $arguments += @('--learner-device', 'cuda', '--cuda-graphs') }
elseif (-not $NoCompile) { $arguments += '--compile-learner' }
if ($BatchSearch) { $arguments += '--batch-search' }
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$stdout = Join-Path $root "history\v4-sf$Elo-$stamp.stdout.log"
$stderr = Join-Path $root "history\v4-sf$Elo-$stamp.stderr.log"
$quoted = ($arguments | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' '
$command = "chcp 65001 >nul&& call `"$vcvars`" >nul 2>nul && set VSLANG=1033&& set PYTHONUTF8=1&& start `"`" /b /belownormal /wait `"$python`" $quoted"
$process = Start-Process -FilePath 'cmd.exe' -ArgumentList '/d', '/c', "`"$command`"" -WorkingDirectory $root `
    -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
[pscustomobject]@{ pid = $process.Id; stdout = $stdout; stderr = $stderr; arguments = $arguments
    startedAt = (Get-Date).ToUniversalTime().ToString('o') } | ConvertTo-Json -Depth 3 |
    Tee-Object -FilePath (Join-Path $root 'history\v4-process.json')
