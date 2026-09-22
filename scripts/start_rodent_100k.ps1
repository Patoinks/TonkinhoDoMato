param([switch]$Resume)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonExecutable = 'C:\Users\Martim\AppData\Local\Programs\Python\Python313\python.exe'
$campaignDirectory = Join-Path $projectRoot 'checkpoints\cognitive-rodent-100k-20260922'
if ((Test-Path -LiteralPath $campaignDirectory) -and -not $Resume) {
    throw 'Campaign directory already exists. Resume its latest.pt instead of starting another campaign.'
}
if ($Resume -and -not (Test-Path -LiteralPath (Join-Path $campaignDirectory 'latest.pt'))) {
    throw 'Cannot resume: the campaign checkpoint does not exist.'
}
$activeTraining = Get-CimInstance Win32_Process -Filter "name='python.exe'" |
    Where-Object { $_.CommandLine -match 'train_(cognitive|survival)\.py' }
if ($activeTraining) { throw 'Another cognitive/survival training process is still running.' }
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$stdout = Join-Path $projectRoot "history\rodent-100k-$stamp.stdout.log"
$stderr = Join-Path $projectRoot "history\rodent-100k-$stamp.stderr.log"
$sourceCheckpoint = if ($Resume) { 'checkpoints/cognitive-rodent-100k-20260922/latest.pt' } else { 'checkpoints/cognitive-1000-500/latest.pt' }
$arguments = @('train_cognitive.py', '--resume', $sourceCheckpoint,
    '--opponent-path', 'engines/rodent-iv/rodent-iv.exe', '--opponent-elo', '1000',
    '--opponent-move-time', '0.002', '--games', '100000', '--game-cap', '100000', '--workers', '8',
    '--checkpoint-dir', 'checkpoints/cognitive-rodent-100k-20260922', '--log-every', '25')
if (-not $Resume) { $arguments += '--new-campaign' }
$trainingProcess = Start-Process -FilePath $pythonExecutable -ArgumentList $arguments -WorkingDirectory $projectRoot `
    -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$receipt = [pscustomobject]@{ pid = $trainingProcess.Id; checkpoint = (Join-Path $campaignDirectory 'latest.pt');
    stdout = $stdout; stderr = $stderr; arguments = $arguments; startedAt = (Get-Date).ToUniversalTime().ToString('o') }
$receipt | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $projectRoot 'history\rodent-100k-process.json') -Encoding utf8
$receipt | ConvertTo-Json -Depth 3
