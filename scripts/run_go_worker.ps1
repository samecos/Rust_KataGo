param(
    [string]$Server = "127.0.0.1:50051",
    [string]$WorkerId = "rustgo-5070ti",
    [string]$Model = "D:/Go/Server/models/kata1-tf3-b11c768-s11001M-d5973M.bin.gz",
    [string]$Config = "",
    [ValidateRange(1,4096)][int]$Capacity = 32,
    [string]$ModelSha256 = "",
    [switch]$Once
)
$ErrorActionPreference = "Stop"
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$binary = Join-Path $repoRoot "target/release/katago-rs.exe"
if (-not (Test-Path -LiteralPath $binary -PathType Leaf)) {
    throw "Build first: cargo build -p katago --features cuda --release"
}
if ([string]::IsNullOrEmpty($Config)) {
    $Config = Join-Path $repoRoot "configs/worker_cuda.cfg"
}
if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) { throw "Model not found: $Model" }
if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) { throw "Config not found: $Config" }
$workerArgs = @("nnworker", "--server", $Server, "--worker-id", $WorkerId,
    "--model", $Model, "--config", $Config, "--capacity", "$Capacity")
if ($ModelSha256) { $workerArgs += @("--model-sha256", $ModelSha256) }
if ($Once) { $workerArgs += "--once" }
& $binary @workerArgs
exit $LASTEXITCODE
