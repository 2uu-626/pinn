$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "remote.config.ps1")

& (Join-Path $PSScriptRoot "sync-to-remote.ps1")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$Target = "$RemoteUser@$RemoteHost"
$RunScript = @"
`$ProgressPreference = 'SilentlyContinue'
Set-Location -Path '$RemotePath'
$RunCommand
"@
$EncodedRunScript = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($RunScript))
$RemoteCommand = "powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand $EncodedRunScript"

Write-Host "Running on ${Target}:$RemotePath"
& ssh $Target $RemoteCommand
