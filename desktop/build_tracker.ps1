$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonExecutable = 'C:\Users\Martim\AppData\Local\Programs\Python\Python313\python.exe'
Push-Location $projectRoot
try {
    & $pythonExecutable -m PyInstaller --noconfirm --windowed --onedir --name 'Tonkinho do Mato - Treino' `
        --distpath dist/tonkinho-tracker-build --workpath build/tonkinho-tracker --specpath desktop `
        --exclude-module numpy --add-data "$PSScriptRoot\tonkinho-do-mato.png;." desktop/tonkinho_tracker.py
    if ($LASTEXITCODE -ne 0) { throw 'Tracker build failed.' }
    $appDirectory = Join-Path $projectRoot 'dist\tonkinho-tracker-build\Tonkinho do Mato - Treino'
    @{ project = $projectRoot } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $appDirectory 'tracker-config.json') -Encoding utf8
    $shell = New-Object -ComObject WScript.Shell
    $shortcutPath = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Tonkinho do Mato - Treino.lnk'
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = Join-Path $appDirectory 'Tonkinho do Mato - Treino.exe'
    $shortcut.WorkingDirectory = $appDirectory
    $shortcut.Description = 'Progresso local do treino do Tonkinho do Mato'
    $shortcut.Save()
    Write-Output $shortcutPath
} finally { Pop-Location }
