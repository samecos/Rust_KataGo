<#
.SYNOPSIS
Start the packaged CUDA student as a standard GTP engine.
.EXAMPLE
powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\code\Rust_KataGo\scripts\play_student_gtp.ps1
.EXAMPLE
./scripts/play_student_gtp.ps1 -Model D:\models\dense.rgmodel
.NOTES
Use the executable and arguments printed by -PrintCommand in Sabaki or Lizzie.
Normal execution reserves stdout for the engine's GTP replies.
#>
[CmdletBinding()]
param(
    [string] $Model = '',
    [string] $Config = '',
    [string] $Executable = '',
    [switch] $PrintCommand
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
$taskRepository = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Model)) { $Model = Join-Path $taskRepository 'models\student-playable-20261003\dense.rgmodel' }
if ([string]::IsNullOrWhiteSpace($Config)) { $Config = Join-Path $taskRepository 'configs\gtp_student_play.cfg' }
if ([string]::IsNullOrWhiteSpace($Executable)) { $Executable = Join-Path $taskRepository 'target\student-playable-build-r1\release\katago-rs.exe' }

try {
    foreach ($taskPath in @($Executable, $Config, $Model)) {
        if (-not (Test-Path -LiteralPath $taskPath -PathType Leaf)) {
            throw "Required engine file is missing: $taskPath"
        }
    }
    $taskExecutable = (Resolve-Path -LiteralPath $Executable).ProviderPath
    $taskConfig = (Resolve-Path -LiteralPath $Config).ProviderPath
    $taskModel = (Resolve-Path -LiteralPath $Model).ProviderPath
    $taskArguments = @('gtp', '--config', $taskConfig, '--model', $taskModel)
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
