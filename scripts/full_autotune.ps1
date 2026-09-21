param(
    [Parameter(Mandatory=$true)][string]$Model,
    [ValidateSet('worker', 'local')][string]$Mode = 'worker',
    [string]$Binary = '',
    [string]$Reference = '',
    [string]$CppWorker = '',
    [string]$CppConfig = '',
    [string]$Output = '',
    [ValidateRange(1,128)][int]$Capacity = 32,
    [string[]]$Batches = @('8', '4', '12', '14', '16'),
    [string[]]$Threads = @('8', '4', '12', '16'),
    [string[]]$Groups = @(),
    [switch]$Ownership,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'tuning_python.ps1')
$pythonExe = Resolve-RustGoTunePython -NeedsGrpc $true
$tuneArgs = @((Join-Path $PSScriptRoot 'full_autotune.py'), '--model', $Model, '--mode', $Mode,
    '--capacity', "$Capacity", '--batches', ($Batches -join ','), '--threads', ($Threads -join ','))
if ($Binary) { $tuneArgs += @('--binary', $Binary) }
if ($Reference) { $tuneArgs += @('--reference', $Reference) }
if ($CppWorker) { $tuneArgs += @('--cpp-worker', $CppWorker) }
if ($CppConfig) { $tuneArgs += @('--cpp-config', $CppConfig) }
if ($Output) { $tuneArgs += @('--output', $Output) }
if ($Groups.Count) { $tuneArgs += @('--groups', ($Groups -join ',')) }
if ($Ownership) { $tuneArgs += '--ownership' }
if ($DryRun) { $tuneArgs += '--dry-run' }
& $pythonExe @tuneArgs
exit $LASTEXITCODE
