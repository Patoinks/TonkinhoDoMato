param(
    [string]$A='G16', [string]$B='G15Treinado', [string[]]$Players,
    [ValidateRange(1,10000)][int]$Games=15,
    [ValidateRange(1,6)][int]$Workers=2,
    [int]$Seed=2027, [int]$RandomPlies=4, [int]$MaxPlies=300,
    [switch]$List, [switch]$Status, [switch]$Stop, [switch]$Resume,
    [switch]$Watch, [switch]$All, [switch]$AllowConcurrent
)
$ErrorActionPreference='Stop'
$taskTimer=[Diagnostics.Stopwatch]::StartNew()
$taskPython=$env:TONKAI_PYTHON
$taskPythonPrefix=@()
if ([string]::IsNullOrWhiteSpace($taskPython)) {
    $taskPython=Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'
    if (!(Test-Path -LiteralPath $taskPython)) {
        $taskLauncher=Get-Command py.exe -ErrorAction SilentlyContinue
        if (!$taskLauncher) { throw 'Python 3.13 nao encontrado. Define TONKAI_PYTHON com o caminho do python.exe.' }
        $taskPython=$taskLauncher.Source
        $taskPythonPrefix=@('-3.13')
    }
}
$taskEntry=Join-Path $PSScriptRoot 'battle_run.py'
$taskLatest=Join-Path $PSScriptRoot 'LATEST.txt'
if ($List) {
    (Get-Content -LiteralPath (Join-Path $PSScriptRoot 'registry.json') -Raw | ConvertFrom-Json).aliases
    return
}
if ($Status -or $Stop -or $Resume) {
    if (!(Test-Path -LiteralPath $taskLatest)) { throw 'Ainda nao existe batalha nesta pipeline.' }
    $taskRun=(Get-Content -LiteralPath $taskLatest -Raw).Trim()
    if ($Status) {
        Get-Content -LiteralPath (Join-Path $taskRun 'status.json') -Raw | ConvertFrom-Json
        return
    }
    if ($Stop) {
        New-Item -Path (Join-Path $taskRun 'STOP') -ItemType File -Force | Out-Null
        Write-Output 'Pedido de paragem enviado; resultados concluídos ficam guardados.'
        return
    }
}
$taskActive=Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*battle_run.py*--execute*' }
if ($taskActive -and !$AllowConcurrent) { throw 'Ja existe uma batalha ativa. Usa -Status ou -Stop; paralelo exige -AllowConcurrent.' }
if (!$Resume) {
    $taskSelection=if ($All) { 'all' } elseif ($Players) { $Players -join ',' } else { "$A,$B" }
    $taskTimezone=[TimeZoneInfo]::FindSystemTimeZoneById('GMT Standard Time')
    $taskDate=[TimeZoneInfo]::ConvertTimeFromUtc([DateTime]::UtcNow,$taskTimezone).ToString('yyyy.MM.dd')
    $taskPrepared=& $taskPython @taskPythonPrefix $taskEntry --prepare --players $taskSelection --games $Games --workers $Workers --seed $Seed --random-plies $RandomPlies --max-plies $MaxPlies --date $taskDate
    if ($LASTEXITCODE -ne 0) { throw 'Preparacao falhou; ver erro acima.' }
    $taskInfo=$taskPrepared | ConvertFrom-Json
    $taskRun=$taskInfo.run
}
$taskArgs=@($taskPythonPrefix)+@('-u',('"{0}"' -f $taskEntry),'--execute',('"{0}"' -f $taskRun))
if ($Resume) { $taskArgs+='--resume' }
$taskProcess=Start-Process -FilePath $taskPython -ArgumentList $taskArgs -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRun 'run.log') -RedirectStandardError (Join-Path $taskRun 'errors.log')
$taskProcess.PriorityClass='BelowNormal'
$taskTimer.Stop()
[pscustomobject]@{PID=$taskProcess.Id; Run=$taskRun; DispatchSeconds=[Math]::Round($taskTimer.Elapsed.TotalSeconds,3); StatusFile=(Join-Path $taskRun 'status.json'); Table=(Join-Path $taskRun 'standings.md')}
if ($Watch) {
    $taskPrevious=-1
    do {
        Start-Sleep -Seconds 1
        try { $taskState=Get-Content -LiteralPath (Join-Path $taskRun 'status.json') -Raw | ConvertFrom-Json } catch { continue }
        if ($taskState.completed_games -ne $taskPrevious) {
            Write-Output ('{0}/{1} jogos - {2}' -f $taskState.completed_games,$taskState.total_games,$taskState.state)
            $taskPrevious=$taskState.completed_games
        }
    } while ($taskState.state -in @('prepared','running','stopping'))
    Get-Content -LiteralPath (Join-Path $taskRun 'standings.md')
}
