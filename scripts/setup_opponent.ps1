# Stockfish 11 Windows archive, linked by https://stockfishchess.org/download/.
[CmdletBinding()]
param(
    [string]$Destination = (Join-Path $PSScriptRoot '..\engines\stockfish-11')
)

$ErrorActionPreference = 'Stop'
$archiveUrl = 'https://drive.google.com/uc?export=download&id=1it9o-AF6jb55z02Xvu-Uu387Q_CPa0Ed'
$archiveHash = 'D5A6149D4174E4F8A96011F67CB08F929E135D367E20D83456A4FF29E1E5133A'
$binaryHash = '95CCC04678193F8F8E7E595DE49A3E5270F5E853AA89A147061D65A3DBE59D81'
$targetDirectory = [IO.Path]::GetFullPath($Destination)
$archivePath = Join-Path (Split-Path $targetDirectory -Parent) 'stockfish-11-win.zip'
$binaryPath = Join-Path $targetDirectory 'stockfish-11-win\Windows\stockfish_20011801_x64.exe'

function Assert-Hash([string]$File, [string]$Expected) {
    if ((Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash -ne $Expected) {
        throw "SHA256 mismatch: $File. No unverified executable will be started."
    }
}

New-Item -ItemType Directory -Path (Split-Path $targetDirectory -Parent) -Force | Out-Null
if (!(Test-Path -LiteralPath $archivePath)) {
    $temporaryDownload = "$archivePath.$([Guid]::NewGuid().ToString('N')).download"
    try {
        Invoke-WebRequest -Uri $archiveUrl -OutFile $temporaryDownload
        Assert-Hash $temporaryDownload $archiveHash
        Move-Item -LiteralPath $temporaryDownload -Destination $archivePath
    }
    finally {
        if (Test-Path -LiteralPath $temporaryDownload) {
            Remove-Item -LiteralPath $temporaryDownload
        }
    }
}
Assert-Hash $archivePath $archiveHash
if (!(Test-Path -LiteralPath $targetDirectory)) {
    # Preserve the archive's GPL license and matching source alongside the binary.
    Expand-Archive -LiteralPath $archivePath -DestinationPath $targetDirectory
}
Assert-Hash $binaryPath $binaryHash
Write-Output "Stockfish 11 (Windows x64): $binaryPath"
Write-Output "SHA256: $binaryHash"
Write-Output 'UCI_LimitStrength=true, UCI_Elo=1600; Elo is nominal at fast move limits.'
