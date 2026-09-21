param(
    [ValidateSet('local', 'worker')][string]$Mode = 'local',
    [string]$Model = 'D:/Go/Server/models/kata1-tf3-b11c768-s11001M-d5973M.bin.gz',
    [string]$Binary = '',
    [string]$Output = '',
    [string[]]$Threads = @('8', '4', '12', '16'),
    [ValidateRange(1,4096)][int]$Capacity = 32,
    [switch]$Ownership,
    [switch]$DryRun,
    [switch]$SmokeOnly
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'tuning_python.ps1')
$pythonExe = Resolve-RustGoTunePython -NeedsGrpc ($Mode -eq 'worker' -and -not $DryRun)
$tuneArgs = @((Join-Path $PSScriptRoot 'tune_runtime.py'), '--mode', $Mode,
    '--model', $Model, '--threads', ($Threads -join ','), '--capacity', "$Capacity")
if ($Binary) { $tuneArgs += @('--binary', $Binary) }
if ($Output) { $tuneArgs += @('--output', $Output) }
if ($Ownership) { $tuneArgs += '--ownership' }
if ($DryRun) { $tuneArgs += '--dry-run' }
if ($SmokeOnly) { $tuneArgs += '--smoke-only' }
& $pythonExe @tuneArgs
exit $LASTEXITCODE
