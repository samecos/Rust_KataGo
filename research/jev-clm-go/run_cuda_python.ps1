# Run the isolated CUDA environment without mixing its cuDNN with toolkit DLLs.
# The PATH change is confined to this process and restored when Python exits.
$ErrorActionPreference = 'Stop'
$PythonArguments = $args
$PythonPath = Join-Path $PSScriptRoot '.venv-cuda/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw 'Create .venv-cuda and install requirements-cuda.txt first.'
}
$OriginalCudaPythonPath = $env:PATH
$OriginalCudaIsolationRecord = $env:RUST_KATAGO_SYSTEM_CUDNN_PATHS
$RemovedCudaDirectories = [System.Collections.Generic.List[string]]::new()
$KeptCudaDirectories = [System.Collections.Generic.List[string]]::new()
try {
    foreach ($Directory in ($OriginalCudaPythonPath -split ';')) {
        if ($Directory -and (
            (Test-Path -LiteralPath (Join-Path $Directory 'cudnn64_9.dll') -PathType Leaf) -or
            (Test-Path -LiteralPath (Join-Path $Directory 'cudnn_engines_tensor_ir64_9.dll') -PathType Leaf)
        )) {
            $RemovedCudaDirectories.Add($Directory)
        } else {
            $KeptCudaDirectories.Add($Directory)
        }
    }
    $env:PATH = $KeptCudaDirectories -join ';'
    $env:RUST_KATAGO_SYSTEM_CUDNN_PATHS = $RemovedCudaDirectories -join ';'
    & $PythonPath @PythonArguments
    $CudaPythonExitCode = $LASTEXITCODE
} finally {
    $env:PATH = $OriginalCudaPythonPath
    $env:RUST_KATAGO_SYSTEM_CUDNN_PATHS = $OriginalCudaIsolationRecord
}
exit $CudaPythonExitCode
