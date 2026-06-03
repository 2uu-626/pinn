$RemoteUser = "admin"
$RemoteHost = "192.168.99.56"
$RemotePath = "C:/Users/admin/Documents/bladed"

# Change this to the command you want to run on the remote Windows machine.
$RunCommand = "python demo.py"

# Files and folders skipped when packaging code for the remote machine.
$TarExclude = @(
    ".git",
    ".git/*",
    ".venv",
    ".venv/*",
    "venv",
    "venv/*",
    "env",
    "env/*",
    "__pycache__",
    "*/__pycache__/*",
    ".pytest_cache",
    ".pytest_cache/*",
    "outputs",
    "outputs/*",
    "output",
    "output/*",
    "results",
    "results/*",
    "checkpoints",
    "checkpoints/*",
    "*.pyc",
    "*.pyo",
    "*.log",
    "*.tmp",
    "bladed-sync-*.zip"
)

$LocalOverride = Join-Path $PSScriptRoot "remote.config.local.ps1"
if (Test-Path -LiteralPath $LocalOverride) {
    . $LocalOverride
}
