# Shared dependency bootstrap for both tuning entry points. No GPU operations.
function Resolve-RustGoTunePython {
    param([bool]$NeedsGrpc = $true)
    $tuningRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
    $candidates = @(
        (Join-Path $tuningRoot '.venv-runtime-tune/Scripts/python.exe'),
        (Join-Path $tuningRoot '.venv/Scripts/python.exe'),
        'D:/Go/Server/worker/.venv-windows/Scripts/python.exe'
    )
    $systemPython = Get-Command python -ErrorAction SilentlyContinue
    if ($systemPython) { $candidates += $systemPython.Source }
    $bootstrapPython = $null
    $probe = @'
import importlib.util as util
import inspect
import sys
available = all(util.find_spec(name) is not None for name in ('grpc', 'grpc_tools', 'google', 'google.protobuf'))
if available:
    from google.protobuf.json_format import MessageToDict
    available = 'always_print_fields_with_no_presence' in inspect.signature(MessageToDict).parameters
sys.exit(0 if available else 1)
'@
    foreach ($candidate in $candidates) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
        & $candidate -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" 2>$null
        if ($LASTEXITCODE -ne 0) { continue }
        if (-not $bootstrapPython) { $bootstrapPython = $candidate }
        if ($NeedsGrpc) {
            & $candidate -c $probe 2>$null
            if ($LASTEXITCODE -ne 0) { continue }
        }
        return $candidate
    }
    if (-not $bootstrapPython) { throw 'Python 3.11+ is required. Install Python, then rerun.' }
    $venvPath = Join-Path $tuningRoot '.venv-runtime-tune'
    Write-Host 'Preparing an isolated Python environment for CUDA tuning...'
    & $bootstrapPython -m venv $venvPath | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create the tuning Python environment.' }
    $pythonExe = Join-Path $venvPath 'Scripts/python.exe'
    & $pythonExe -m pip install -r (Join-Path $PSScriptRoot 'requirements-runtime-tune.txt') | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'Failed to install tuning dependencies. Check network access and rerun.' }
    return $pythonExe
}
