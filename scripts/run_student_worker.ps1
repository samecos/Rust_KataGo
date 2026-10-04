<#
.SYNOPSIS
Start the packaged CUDA student as a standard inference Worker.
.EXAMPLE
powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\code\Rust_KataGo\scripts\run_student_worker.ps1 -PrintCommand
.EXAMPLE
./scripts/run_student_worker.ps1 -Server 127.0.0.1:50051 -Capacity 64
.NOTES
PrintCommand validates the files and prints the executable and arguments only.
The engine recognizes the model format from its contents through --model.
#>
[CmdletBinding()]
param(
    [string] $Server = '127.0.0.1:50051',
    [string] $WorkerId = 'student-dense-5070ti-tuned',
    [string] $Model = '',
    [string] $Config = '',
    [string] $Executable = '',
    [int] $Capacity = 64,
    [switch] $PrintCommand
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
$taskRepository = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Model)) { $Model = Join-Path $taskRepository 'models\student-playable-20261003\dense.rgmodel' }
if ([string]::IsNullOrWhiteSpace($Config)) { $Config = Join-Path $taskRepository 'configs\worker_student_cuda.cfg' }
if ([string]::IsNullOrWhiteSpace($Executable)) { $Executable = Join-Path $taskRepository 'target\student-worker-tuning-20261004-r1\optimized-build\release\katago-rs.exe' }

try {
    if ($Capacity -le 0) { throw 'Worker capacity must be positive.' }
    if ([string]::IsNullOrWhiteSpace($Server)) { throw 'Server address must not be empty.' }
    if ([string]::IsNullOrWhiteSpace($WorkerId)) { throw 'Worker ID must not be empty.' }
    foreach ($taskPath in @($Executable, $Config, $Model)) {
        if (-not (Test-Path -LiteralPath $taskPath -PathType Leaf)) {
            throw "Required Worker file is missing: $taskPath"
        }
    }
    $taskExecutable = (Resolve-Path -LiteralPath $Executable).ProviderPath
    $taskConfig = (Resolve-Path -LiteralPath $Config).ProviderPath
    $taskModel = (Resolve-Path -LiteralPath $Model).ProviderPath
    $taskArguments = @('nnworker', '--server', $Server, '--worker-id', $WorkerId,
        '--model', $taskModel, '--config', $taskConfig, '--capacity', "$Capacity")
    if ($PrintCommand) {
        [ordered]@{ executable = $taskExecutable; arguments = $taskArguments } | ConvertTo-Json -Depth 3
        exit 0
    }
    & $taskExecutable @taskArguments
    exit $LASTEXITCODE
}
catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
