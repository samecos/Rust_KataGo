param(
    [ValidateSet('b11', 'b15')]
    [string]$Model,
    [ValidateSet('gtp', 'analysis', 'worker')]
    [string]$Mode = 'gtp',
    [string]$Server = '',
    [string]$WorkerId = '',
    [ValidateSet(1, 32)]
    [int]$Capacity = 32
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
# Keep this a simple script: Parameter(Mandatory) would enable advanced-script
# binding and reject the unbound GTP/analysis lines carried by $input.
if ([string]::IsNullOrWhiteSpace($Model)) {
    throw '-Model is required (b11 or b15).'
}
$Model = $Model.ToLowerInvariant()
$Mode = $Mode.ToLowerInvariant()

# This manifest records experimental INT8 numerical checks and paired ABBA
# results. It is not an FP16 certification plan and does not certify playing
# strength or restore the lossy INT8 model's original FP32 accuracy.
# This launcher only reads the manifest, binaries, models and configurations.
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$manifestPath = Join-Path $repoRoot 'docs/int8-gemm-evidence-20260923.json'

function Get-RequiredProperty {
    param($Object, [string]$Name, [string]$Context)
    if ($null -eq $Object) {
        throw "Missing manifest object: $Context"
    }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property -or $null -eq $property.Value) {
        throw "Missing manifest field: $Context.$Name"
    }
    return $property.Value
}

function Resolve-ManifestFile {
    param([string]$Value, [string]$Label, [switch]$RepositoryRelative)
    if ([string]::IsNullOrWhiteSpace($Value)) {
        throw "Empty manifest path: $Label"
    }
    $rooted = [System.IO.Path]::IsPathRooted($Value)
    if ($RepositoryRelative -and $rooted) {
        throw "Manifest $Label must be relative to the repository."
    }
    $path = if ($rooted) {
        [System.IO.Path]::GetFullPath($Value)
    } else {
        [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Value))
    }
    $repoPrefix = $repoRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    if ($RepositoryRelative -and -not $path.StartsWith($repoPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Manifest $Label must stay inside the repository."
    }
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing ${Label}: $path"
    }
    return $path
}

function Assert-ManifestHash {
    param([string]$Path, [string]$Expected, [string]$Label)
    if ($Expected -notmatch '^[0-9a-fA-F]{64}$') {
        throw "Invalid SHA-256 in manifest: $Label"
    }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    if (-not [string]::Equals($actual, $Expected, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "SHA-256 mismatch for ${Label}; refusing to use unverified files: $Path"
    }
}

if ($Mode -eq 'worker' -and [string]::IsNullOrWhiteSpace($Server)) {
    throw '-Server is required when -Mode worker is selected.'
}
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Missing INT8 experiment evidence manifest: $manifestPath"
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$schema = Get-RequiredProperty $manifest 'schema' 'manifest'
if ($schema -ne 'int8-gemm-study-evidence-v1') {
    throw "Unsupported INT8 experiment manifest schema: $schema"
}
$profiles = Get-RequiredProperty $manifest 'profiles' 'manifest'
$profile = Get-RequiredProperty $profiles $Model 'profiles'
$performance = Get-RequiredProperty $profile 'performance' "profiles.$Model"
$performanceKey = if ($Mode -eq 'worker') { "worker_c$Capacity" } else { 'local' }
$result = Get-RequiredProperty $performance $performanceKey "profiles.$Model.performance"
$accepted = Get-RequiredProperty $result 'accepted' "profiles.$Model.performance.$performanceKey"
if ($accepted -isnot [bool] -or -not $accepted) {
    throw "INT8 candidate was not accepted for model=$Model mode=$performanceKey; launch refused."
}

# Bind numerical acceptance to this mode's selected variant. A passing full
# candidate must not stand in for the independently measured light candidate.
$variantName = Get-RequiredProperty $result 'variant' "profiles.$Model.performance.$performanceKey"
if ($variantName -cnotin @('full', 'light')) {
    throw "Unknown selected INT8 variant: $variantName"
}
$variants = Get-RequiredProperty $profile 'variants' "profiles.$Model"
$variantEvidence = Get-RequiredProperty $variants $variantName "profiles.$Model.variants"
$variantAccuracy = @(Get-RequiredProperty $variantEvidence 'accuracy' "profiles.$Model.variants.$variantName")
foreach ($window in @(1, 32)) {
    $profileName = "on-w$window"
    $numericRows = @($variantAccuracy | Where-Object {
        (Get-RequiredProperty $_ 'profile' 'variant.accuracy') -ceq $profileName
    })
    if ($numericRows.Count -ne 1) {
        throw "Missing or duplicate numerical evidence for $Model/$variantName/$profileName."
    }
    $row = $numericRows[0]
    $context = "profiles.$Model.variants.$variantName.accuracy.$profileName"
    if ((Get-RequiredProperty $row 'compared_to' $context) -cne 'same-window FP16') {
        throw "Numerical reference is not same-window FP16: $context"
    }
    $gate = Get-RequiredProperty $row 'user_gate' $context
    $metrics = Get-RequiredProperty $row 'metrics' $context
    $gateResult = Get-RequiredProperty $gate 'result' "$context.user_gate"
    $maximum = [double](Get-RequiredProperty $gate 'max_abs' "$context.user_gate")
    $limit = [double](Get-RequiredProperty $gate 'limit_abs' "$context.user_gate")
    $measuredMaximum = [double](Get-RequiredProperty $metrics 'win_probability_max_abs' "$context.metrics")
    $caseCount = [int](Get-RequiredProperty $metrics 'cases' "$context.metrics")
    if ($gateResult -cne 'PASS' -or $limit -ne 0.06 -or $caseCount -ne 128 -or
        [double]::IsNaN($maximum) -or [double]::IsInfinity($maximum) -or
        $maximum -lt 0 -or $maximum -gt 0.06 -or $measuredMaximum -ne $maximum) {
        throw "INT8 numerical gate failed for $Model/$variantName/$profileName; expected maximum FP16 win-probability deviation <= 0.06."
    }
}

$binaryPath = Resolve-ManifestFile -Value (Get-RequiredProperty $manifest 'binary' 'manifest') -Label 'binary' -RepositoryRelative
$modelPath = Resolve-ManifestFile -Value (Get-RequiredProperty $profile 'model' "profiles.$Model") -Label 'model'
Assert-ManifestHash $binaryPath (Get-RequiredProperty $manifest 'binary_sha256' 'manifest') 'binary'
Assert-ManifestHash $modelPath (Get-RequiredProperty $profile 'model_sha256' "profiles.$Model") 'model'

# Each accepted model/mode result may select its own measured environment.
# Older manifests without that field retain the manifest-level candidate.
$environmentProperty = $result.PSObject.Properties['environment']
$candidateEnvironment = if ($null -ne $environmentProperty) {
    $environmentProperty.Value
} else {
    Get-RequiredProperty $manifest 'candidate_environment' 'manifest'
}
if ($candidateEnvironment -isnot [System.Management.Automation.PSCustomObject]) {
    throw 'The selected candidate environment must be a JSON object.'
}
$candidateValues = @{}
foreach ($entry in $candidateEnvironment.PSObject.Properties) {
    if ($entry.Name -notmatch '^KATAGO_[A-Z0-9_]+$' -or $entry.Value -isnot [string]) {
        throw 'The selected candidate environment must contain only KATAGO_* keys with string values.'
    }
    $candidateValues[$entry.Name] = $entry.Value
}
if (-not $candidateValues.ContainsKey('KATAGO_CUDA_INT8_GEMM_TUNE') -or $candidateValues['KATAGO_CUDA_INT8_GEMM_TUNE'] -notin @('0', '1')) {
    throw 'The candidate environment must explicitly set KATAGO_CUDA_INT8_GEMM_TUNE to 0 or 1.'
}
$expectedGemmTune = if ($variantName -ceq 'full') { '1' } else { '0' }
if ($candidateValues['KATAGO_CUDA_INT8_GEMM_TUNE'] -cne $expectedGemmTune) {
    throw "Selected variant $variantName disagrees with its GEMM tuning environment."
}
if (-not $candidateValues.ContainsKey('KATAGO_CUDA_ATTN_TILE') -or $candidateValues['KATAGO_CUDA_ATTN_TILE'] -cne 'q64') {
    throw 'The candidate environment must explicitly set KATAGO_CUDA_ATTN_TILE=q64.'
}

$configRelative = if ($Mode -eq 'worker') { 'configs/worker_int8.cfg' } else { 'configs/gtp_int8.cfg' }
$configPath = Resolve-ManifestFile -Value $configRelative -Label 'config' -RepositoryRelative
$configHashes = Get-RequiredProperty $manifest 'config_sha256' 'manifest'
Assert-ManifestHash $configPath (Get-RequiredProperty $configHashes $configRelative 'manifest.config_sha256') 'config'
$command = if ($Mode -eq 'worker') { 'nnworker' } else { $Mode }
$overrides = 'nnBackend=cudaint8backend,cudaInt8Scope=ffn,cudaInt8MinFfnWidth=0,nnMaxBatchSize=8,numNNServerThreadsPerModel=1,numSearchThreads=8'
$engineArguments = @($command, '--model', $modelPath, '--config', $configPath, '--override-config', $overrides)
if ($Mode -eq 'worker') {
    if ([string]::IsNullOrWhiteSpace($WorkerId)) {
        $WorkerId = "rustgo-$Model-int8-tuned-c$Capacity"
    }
    $engineArguments += @('--server', $Server, '--worker-id', $WorkerId, '--capacity', "$Capacity")
}

# Only process-scoped KATAGO_* variables are touched. The child inherits this
# isolated candidate environment; finally restores the caller's complete
# KATAGO_* snapshot, including values that were intentionally removed here.
# All other environment variables remain unchanged throughout the launch.
$processScope = [System.EnvironmentVariableTarget]::Process
$originalKataEnvironment = @{}
foreach ($entry in [System.Environment]::GetEnvironmentVariables($processScope).GetEnumerator()) {
    if ([string]$entry.Key -like 'KATAGO_*') {
        $originalKataEnvironment[[string]$entry.Key] = [string]$entry.Value
    }
}
$engineExitCode = 1
try {
    foreach ($name in $originalKataEnvironment.Keys) {
        [System.Environment]::SetEnvironmentVariable($name, $null, $processScope)
    }
    foreach ($name in $candidateValues.Keys) {
        [System.Environment]::SetEnvironmentVariable($name, $candidateValues[$name], $processScope)
    }
    $startupNote = if ($candidateValues['KATAGO_CUDA_INT8_GEMM_TUNE'] -eq '1') {
        'initial tuning may take about 90 seconds for B15.'
    } else {
        'lightweight q64 profile; GEMM tuning disabled.'
    }
    $workerTargetNote = if ($Mode -eq 'worker') {
        " server=$Server worker_id=$WorkerId capacity=$Capacity;"
    } else { '' }
    [Console]::Error.WriteLine("Starting experimental INT8 candidate: model=$Model mode=$Mode profile=$performanceKey;$workerTargetNote $startupNote")
    if ($Mode -eq 'worker') {
        [Console]::Error.WriteLine('This script starts only the Worker; the Server must already be listening at the specified gRPC address.')
    }
    if ($MyInvocation.ExpectingInput) {
        $input | & $binaryPath @engineArguments
    } elseif ([Console]::IsInputRedirected) {
        # pwsh -File does not automatically pass its redirected console input
        # to the native child. Forward each line immediately so GUI clients can
        # keep stdin open while waiting for individual GTP/analysis replies.
        & {
            while ($null -ne ($redirectedInputLine = [Console]::ReadLine())) {
                $redirectedInputLine
            }
        } | & $binaryPath @engineArguments
    } else {
        & $binaryPath @engineArguments
    }
    $engineExitCode = $LASTEXITCODE
} finally {
    foreach ($entry in [System.Environment]::GetEnvironmentVariables($processScope).GetEnumerator()) {
        if ([string]$entry.Key -like 'KATAGO_*') {
            [System.Environment]::SetEnvironmentVariable([string]$entry.Key, $null, $processScope)
        }
    }
    foreach ($name in $originalKataEnvironment.Keys) {
        [System.Environment]::SetEnvironmentVariable($name, $originalKataEnvironment[$name], $processScope)
    }
}
exit $engineExitCode
