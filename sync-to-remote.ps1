$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "remote.config.ps1")

$ProjectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$Target = "$RemoteUser@$RemoteHost"
$ArchiveName = "bladed-sync.zip"
$ArchivePath = Join-Path $env:TEMP $ArchiveName

$Connection = Test-NetConnection $RemoteHost -Port 22 -WarningAction SilentlyContinue
if (-not $Connection.TcpTestSucceeded) {
    throw "Cannot connect to $RemoteHost:22. Enable OpenSSH Server on the remote Windows machine first."
}

if (Test-Path -LiteralPath $ArchivePath) {
    Remove-Item -LiteralPath $ArchivePath -Force
}

$TarArgs = @("-a", "-cf", $ArchivePath)
foreach ($Item in $TarExclude) {
    $TarArgs += "--exclude=$Item"
}
$TarArgs += @("-C", $ProjectRoot, ".")

Write-Host "Packaging $ProjectRoot"
& tar @TarArgs
if ($LASTEXITCODE -ne 0) {
    throw "tar failed with exit code $LASTEXITCODE"
}

Write-Host "Uploading archive to $Target"
& scp $ArchivePath "${Target}:$ArchiveName"
if ($LASTEXITCODE -ne 0) {
    throw "scp failed with exit code $LASTEXITCODE"
}

$DeployScript = @"
`$ProgressPreference = 'SilentlyContinue'
New-Item -ItemType Directory -Force -Path '$RemotePath' | Out-Null
Expand-Archive -Force -Path (Join-Path `$HOME '$ArchiveName') -DestinationPath '$RemotePath'
Remove-Item -Force (Join-Path `$HOME '$ArchiveName')
"@
$EncodedDeployScript = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($DeployScript))
$RemoteCommand = "powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand $EncodedDeployScript"

Write-Host "Expanding archive on ${Target}:$RemotePath"
& ssh $Target $RemoteCommand
if ($LASTEXITCODE -ne 0) {
    throw "ssh deploy command failed with exit code $LASTEXITCODE"
}

Write-Host "Synced to ${Target}:$RemotePath"
