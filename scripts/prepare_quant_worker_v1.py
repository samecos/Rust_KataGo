"""Prepare an original-protocol Worker command, without starting a Worker.

Only the two existing B15 FP16/INT8 recipes are supported. The new executable
does CPU quant-inspect and CUDA metadata-only cuda-fingerprint; no weights are
uploaded and no inference/benchmark is requested. Its predicted quant identity
must still pass the real backend's cudaQuantExpectedProfile gate on later load.
No old speed certificate is transferred. This file has no Worker run mode.
"""
from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
MODEL = ("models/b15-ffn-pruned-a8.bin.gz", 307758816,
         "3f216ee88226ee49ca826eaa5f7e1e9982ff48a96625914bfc4e5ad20ee095a7")
GRAPH = "6f0bb7abf426bf331cee6007490d221acc6f0b3b150d22a87c952d1777577ea1"
OLD_BINARY = "9a1f42f7083fe65cd1aed717e33b5d2f4a776f533af6b30b84cc4b85d3a6272f"
CANDIDATES = {
    "b15-b3": (3,
        ("target/unified-quant-speed-only-extra-r1/configs/b15-b3-mix-3-q64.cfg", 448,
         "f547ed57213062587c3c5c06cb0f409b94469940b3ae534b32b07ae3333009bb"),
        ("target/unified-quant-mixture-proposals-real-r1/b15-fdb168f273ba6b7f.recipe.json", 24871,
         "e0f3eb1df21fb39742857e18d93ff70e2787ec088839cc6d27ef787a6f6496eb")),
    "b15-b8": (8,
        ("target/unified-quant-speed-only-extra-r1/configs/b15-b8-cost-only-q64.cfg", 440,
         "54a463a4a9c3b38504873d0c8a4d4d0270df19651286c998bb331c1dca6a4332"),
        ("target/unified-quant-speed-cost-recipes-r1/b15-b8-cost-only.recipe.json", 24871,
         "afc46f4ab576a00552f0a3cef7db9379cf8937ad4ee108b6fb8e92f76666884b")),
}
# Deliberately fail if the inspected algorithm changes. These Sources must also
# occur byte-for-byte in the supplied successful new build plan's source list.
ALGORITHM = {
    "crates/kata_nn/src/backends/cuda.rs": "c8678174e132964f067be14ac30039ec0ef19b08cd3b746a788b0c8656787ac3",
    "crates/kata_nn/src/backends/cuda_exec.rs": "2dde8ec8d0bae76324167358cf91487b4e187236045d0938a6e9199e2ad65c50",
    "crates/kata_nn/src/tactic_plan.rs": "9705ace29d68099660ac8dc6eb22ef24d527fa170e1b8a59f766f01a03d48b5f",
    "crates/kata_nn/src/eval.rs": "fab55ef98f3fbb2531c80cab6f06487ae009392fa615b005f47bfc9791c70d3e",
    "crates/kata_nn/src/quantization_plan.rs": "df7d4e59fd774bda06bdb7d4233bee994dfc45bbfecd78f865afff03a8af96f9",
    "crates/katago/src/cmd/cuda_fingerprint.rs": "7e8bf6ebfc256678fc1f190cfce87efad5e5c173523be0146664dc260357cddb",
    "crates/katago/src/cmd/quant_inspect.rs": "6420f89f44a2bc38cd1c0c209955b3ff2e8a5d8082d8c5333a513e3b87253e7f",
    "Cargo.toml": "82a978fabf092a9d9bedfa99cf05172782dc796dc1256eca99e3c8bc6ba14d5b",
}
WORKER_PROTOCOL = {
    "crates/kata_worker/proto/worker.proto": "8a43d78331e64040f44e2c0fe49456d6d75c9ea88b11b111e03473fb48741e5e",
    "crates/kata_worker/src/client.rs": "c6ad8810bf59d4b88ded3042de387ac6fcc9df25481be45551cc7d6140aa23a0",
    "crates/kata_worker/src/evaluator.rs": "d150dbba8d17ad1c02bfc42b2456bdf76b840657c00f5a780b8d8773d4e3bff5",
}
TACTIC_KEYS = tuple("KATAGO_CUDA_" + suffix for suffix in (
    "ATTN", "ATTN_TILE", "CUBLASLT", "CUBLASLT_RANK", "DUALFFN", "FUSION",
    "GEMM_LAYOUT", "MXFP8_SCALE_CLEAR_FUSION", "NOGRAPH", "NOPIPELINE", "PADBATCH",
    "RESIDUAL_ALGO", "RMS", "SPLITK", "STEM_MAP3D", "GATE_ROWQUAD_R1",
    "QKV_IMMUTABLE_R1", "QKV_CLASSIC_N128_R1", "OUTPROJ_CLASSIC_N128_R1",
    "FFN_COMPACT_R1", "T32", "T64N32",
)) + ("KATAGO_NN_BATCH_WINDOW_US", "KATAGO_CUDA_INT8_GEMM_TUNE", "KATAGO_CUDA_INT8_RMS_FUSION")
OVERRIDES = {"KATAGO_CUDA_" + k: v for k, v in {
    "ATTN": "fa2", "ATTN_TILE": "q64", "CUBLASLT": "1", "CUBLASLT_RANK": "heuristic",
    "DUALFFN": "0", "FUSION": "none", "GEMM_LAYOUT": "tn", "NOGRAPH": "0",
    "NOPIPELINE": "0", "PADBATCH": "0", "SPLITK": "0", "INT8_GEMM_TUNE": "0",
    "INT8_RMS_FUSION": "1", "BATCH_TRACE": "0",
}.items()}
EXECUTION_KEYS = ("nnMaxBatchSize", "numNNServerThreadsPerModel", "nnUseFP16", "nnUseNHWC")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def no_duplicates(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def decode(raw):
    return json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicates,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def remaining(deadline):
    value = (deadline - time.monotonic_ns()) / 1e9
    require(value > 0, "original preparation deadline exceeded")
    return value


def source(path, deadline=None):
    path = Path(path).resolve(strict=True)
    require(path.is_file(), f"not a file: {path}")
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while True:
            if deadline is not None:
                remaining(deadline)
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            count += len(block)
    if deadline is not None:
        remaining(deadline)
    return {"path": str(path), "bytes": count, "sha256": digest.hexdigest()}


class Inventory:
    def __init__(self, deadline):
        self.deadline, self.records = deadline, {}

    def add(self, path, sha=None, size=None):
        item = source(path, self.deadline)
        require(sha is None or item["sha256"] == sha, f"Source SHA mismatch: {path}")
        require(size is None or item["bytes"] == size, f"Source length mismatch: {path}")
        key = os.path.normcase(item["path"])
        require(key not in self.records or self.records[key] == item, f"Source changed: {path}")
        self.records[key] = item
        return item

    def pinned(self, item):
        require(set(item) == {"path", "bytes", "sha256"}, "invalid Source")
        return self.add(item["path"], item["sha256"], item["bytes"])

    def raw(self, item, maximum=32 * 1024 * 1024):
        require(item["bytes"] <= maximum, "oversized metadata Source")
        remaining(self.deadline)
        raw = Path(item["path"]).read_bytes()
        require(len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"],
                "Source changed while reading")
        remaining(self.deadline)
        return raw

    def recheck(self):
        for item in list(self.records.values()):
            require(source(item["path"], self.deadline) == item, "Source changed at final gate")


def fixed_source(inventory, pin):
    path, length, digest = pin
    return inventory.add(ROOT / path, digest, length)


def compact(value, *, ordered=False):
    # The inspected serde_json::Value contract has only integers, strings,
    # booleans, null, arrays and objects. Disallow float formatting differences.
    def check(item):
        if isinstance(item, dict):
            require(all(isinstance(k, str) for k in item), "nonstring JSON key")
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        else:
            require(item is None or type(item) in (bool, int, str), "unsupported JSON scalar")
            if type(item) is int:
                require(-(1 << 63) <= item < (1 << 64), "JSON integer out of Rust range")
    check(value)
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=not ordered, separators=(",", ":")).encode("utf-8")


def recipe_identity(export, model_sha, recipe_sha):
    require(export.get("schema") == "rustgo-recipe-identity-export" and export.get("version") == 1,
            "wrong recipe identity export")
    require(export.get("model_sha256") == model_sha and export.get("graph_sha256") == GRAPH,
            "model/graph differs from fixed B15")
    require(export.get("source_recipe_file_sha256") == recipe_sha and
            export.get("canonical_encoding") == "serde-json-compact-ordered-v1",
            "recipe input/encoding mismatch")
    envelope = export["canonical_envelope"]
    require(list(envelope) == ["schema", "version", "quantization_semantics_version",
                              "model_sha256", "graph_sha256", "projections"],
            "only the ordered V1 FP16/INT8 canonical envelope is supported")
    require(envelope["schema"] == "rustgo-precision-recipe" and envelope["version"] == 1 and
            envelope["quantization_semantics_version"] == 1 and
            envelope["model_sha256"] == model_sha and envelope["graph_sha256"] == GRAPH,
            "invalid canonical recipe header")
    entries = envelope["projections"]
    require(isinstance(entries, list) and 0 < len(entries) <= 4096, "invalid projection list")
    ids = []
    for entry in entries:
        require(list(entry) == ["id", "expected_n", "expected_k", "precision"], "projection field order")
        require(entry["precision"] in ("fp16", "int8"), "MXFP8 and new precisions are unsupported")
        require(isinstance(entry["id"], str) and all(type(entry[k]) is int and entry[k] > 0
                for k in ("expected_n", "expected_k")), "invalid projection")
        ids.append(entry["id"])
    require(ids == sorted(set(ids)), "canonical projections are not unique and sorted")
    digest = hashlib.sha256(compact(envelope, ordered=True)).hexdigest()
    require(export.get("recipe_sha256") == digest, "canonical recipe digest mismatch")
    return digest


def selected_environment(values, gpu_uuid):
    require(isinstance(values, dict) and all(isinstance(k, str) and isinstance(v, str)
            and k and "=" not in k and "\0" not in k + v for k, v in values.items()),
            "invalid frozen environment")
    folded = [k.upper() for k in values]
    require(len(folded) == len(set(folded)), "case-colliding Windows environment keys")
    require(re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", gpu_uuid),
            "expected one complete GPU UUID")
    result = {k: v for k, v in values.items()
              if not k.upper().startswith("KATAGO_") and k.upper() != "CUDA_VISIBLE_DEVICES"}
    result.update(OVERRIDES)
    result["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    return result


def config_values(raw, batch, recipe):
    values = {}
    for line in raw.decode("utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        require(text.count("=") == 1, "unsupported config syntax")
        key, value = (p.strip() for p in text.split("=", 1))
        require(key and value and key not in values, "duplicate/empty config entry")
        values[key] = value
    require(values.get("nnBackend") == "cudaquantbackend" and
            values.get("nnMaxBatchSize") == str(batch) and
            values.get("numNNServerThreadsPerModel") == "1", "unsupported backend/batch/thread config")
    require(Path(values.get("cudaQuantPlan", "")).resolve() == Path(recipe).resolve(), "wrong recipe path")
    require(not any(k.startswith(("cudaTacticPlan", "cudaInt8", "cudaQuantExpectedProfile"))
                    for k in values), "config already carries incompatible pins")
    require("nnUseFP16" not in values and "nnUseNHWC" not in values,
            "only the original optional execution config is supported")
    # NnEvaluator overrides the first two to resolved decimal strings. The
    # fixed Worker setup resolves them to batch/1. It does NOT rename useFP16 or
    # inputsUseNHWC into nnUseFP16/nnUseNHWC; those hash slots remain null.
    return {key: values.get(key) for key in EXECUTION_KEYS}


def profile_contract(model_sha, recipe_sha, binary_sha, fingerprint, driver_version, env, execution):
    require(type(driver_version) is int and 0 < driver_version < (1 << 31), "bad CUDA driver API version")
    device = {key: fingerprint[key] for key in
              ("gpu_name", "compute_capability", "sm_count", "l2_cache_bytes")}
    require(device == {"gpu_name": "NVIDIA GeForce RTX 5070 Ti", "compute_capability": "12.0",
                       "sm_count": 70, "l2_cache_bytes": 50331648}, "unexpected candidate device")
    require(isinstance(fingerprint["backend_build"], dict) and fingerprint["backend_build"], "missing build fingerprint")
    require(all(env.get(k) == v for k, v in OVERRIDES.items()), "candidate tactics changed")
    require(all(k in OVERRIDES or env.get(k) is None for k in TACTIC_KEYS), "unexpected optional tactic")
    contract = {"schema": "rustgo-quantized-execution-v1", "model_sha256": model_sha,
                "recipe_sha256": recipe_sha, "executable_sha256": binary_sha,
                "backend_build": fingerprint["backend_build"], "device": device,
                "driver_version": driver_version, "tactics": {key: env.get(key) for key in TACTIC_KEYS},
                "actual_execution": {"mxfp8_scale_clear_fusion": False},
                "execution_config": execution}
    digest = hashlib.sha256(compact(contract)).hexdigest()
    return contract, "rustgo-quant-v1:" + digest


def publish(path, raw):
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path, value):
    publish(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n")


def launcher_script(command_source):
    # A literal Base64 record avoids interpolating any path into PowerShell code.
    encoded = base64.b64encode(compact(command_source)).decode("ascii")
    template = r'''# Prepared original-protocol Worker. This script is never run by preparation.
# Foreground, one process. Default reconnects normally; -Once disables reconnect.
# Local quant ExpectedProfile is mandatory; the original Server routes model SHA.
# No speed/accuracy certification and no authority to replace a production Worker.
param([switch]$Once)
$ErrorActionPreference = 'Stop'
$boundCommand = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('COMMAND_SOURCE_BASE64')) | ConvertFrom-Json
function Read-BoundJson($record) {
    $raw = [IO.File]::ReadAllBytes([string]$record.path)
    if ($raw.Length -gt 4194304 -or $raw.Length -ne [long]$record.bytes) { throw 'JSON Source length mismatch' }
    $hasher = [Security.Cryptography.SHA256]::Create()
    try { $digest = [BitConverter]::ToString($hasher.ComputeHash($raw)).Replace('-', '').ToLowerInvariant() }
    finally { $hasher.Dispose() }
    if ($digest -cne [string]$record.sha256) { throw 'JSON Source SHA mismatch' }
    return [Text.Encoding]::UTF8.GetString($raw) | ConvertFrom-Json
}
function Assert-BoundFile($record) {
    $stream = [IO.File]::OpenRead([string]$record.path)
    $hasher = [Security.Cryptography.SHA256]::Create()
    try {
        if ($stream.Length -ne [long]$record.bytes) { throw 'File Source length mismatch' }
        $digest = [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '').ToLowerInvariant()
        if ($digest -cne [string]$record.sha256) { throw 'File Source SHA mismatch' }
    } finally { $hasher.Dispose(); $stream.Dispose() }
}
$workerCommand = Read-BoundJson $boundCommand
if ($workerCommand.schema -cne 'rustgo-original-protocol-worker-command-v1') { throw 'Wrong command schema' }
Assert-BoundFile $workerCommand.binary
Assert-BoundFile $workerCommand.model
Assert-BoundFile $workerCommand.config
Assert-BoundFile $workerCommand.recipe
$workerEnvironment = Read-BoundJson $workerCommand.environment
if ($workerEnvironment.schema -cne 'rustgo-explicit-worker-environment-v1') { throw 'Wrong environment schema' }
$workerArguments = @($workerCommand.argv)
if ($workerArguments.Count -ne 15 -or $workerArguments[0] -cne $workerCommand.binary.path -or $workerArguments[1] -cne 'nnworker' -or $workerArguments[-1] -cne '--once') { throw 'Unexpected Worker argv' }
$workerExecutable = [string]$workerArguments[0]
if ($Once) { $workerArguments = @($workerArguments[1..($workerArguments.Count - 1)]) }
else { $workerArguments = @($workerArguments[1..($workerArguments.Count - 2)]) }
$savedEnvironment = [Environment]::GetEnvironmentVariables('Process')
$savedLocation = Get-Location
$workerExit = $null
try {
    foreach ($name in @([Environment]::GetEnvironmentVariables('Process').Keys)) { [Environment]::SetEnvironmentVariable([string]$name, $null, 'Process') }
    foreach ($property in $workerEnvironment.values.PSObject.Properties) { [Environment]::SetEnvironmentVariable($property.Name, [string]$property.Value, 'Process') }
    Set-Location -LiteralPath ([string]$workerCommand.cwd)
    & $workerExecutable @workerArguments
    $workerExit = $LASTEXITCODE
} finally {
    foreach ($name in @([Environment]::GetEnvironmentVariables('Process').Keys)) { [Environment]::SetEnvironmentVariable([string]$name, $null, 'Process') }
    foreach ($name in $savedEnvironment.Keys) { [Environment]::SetEnvironmentVariable([string]$name, [string]$savedEnvironment[$name], 'Process') }
    Set-Location -LiteralPath $savedLocation.Path
}
if ($null -eq $workerExit) { throw 'No actual Worker exit code observed' }
exit $workerExit
'''
    return template.replace("COMMAND_SOURCE_BASE64", encoded).encode("utf-8")


def metadata_process(command, env, output, name, deadline):
    require(command[1] in ("quant-inspect", "cuda-fingerprint"), "only metadata commands may execute")
    stdout, stderr = output / f"{name}.stdout.txt", output / f"{name}.stderr.txt"
    child = None
    record = {"argv": command, "pid": None, "actual_exit_code": None, "timeout": False,
              "error": None, "cleanup_error": None}
    # Leave a fixed 5 seconds inside the original deadline for kill+wait.
    work_seconds = remaining(deadline) - 5
    require(work_seconds > 0, "insufficient original metadata cleanup budget")
    try:
        with stdout.open("xb") as out, stderr.open("xb") as err:
            child = subprocess.Popen(command, env=env, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
            record["pid"] = child.pid
            try:
                record["actual_exit_code"] = child.wait(timeout=work_seconds)
            except subprocess.TimeoutExpired:
                record["timeout"] = True
                raise
        require(record["actual_exit_code"] == 0, f"{name} returned nonzero")
        remaining(deadline)
    except BaseException as error:
        record["error"] = repr(error)
        raise
    finally:
        if child is not None and child.poll() is None:
            try:
                child.kill()
                record["actual_exit_code"] = child.wait(timeout=remaining(deadline))
            except BaseException as error:
                record["cleanup_error"] = repr(error)
        write_json(output / f"{name}.process.json", record)


def driver_api_version():
    """Only driver version metadata; no cuInit, context, model, or kernel call."""
    require(os.name == "nt", "Windows CUDA driver metadata is required")
    # System32-only DLL search; do not accept a project-directory nvcuda.dll.
    library = ctypes.WinDLL("nvcuda.dll", winmode=0x00000800)
    query = library.cuDriverGetVersion
    query.argtypes, query.restype = [ctypes.POINTER(ctypes.c_int)], ctypes.c_int
    version = ctypes.c_int()
    status = query(ctypes.byref(version))
    require(status == 0 and version.value > 0, f"cuDriverGetVersion failed: {status}")
    return version.value


def validate_build(build, binary):
    require(build.get("schema") == "rustgo-typed-worker-native-build-result-v1", "wrong build receipt schema")
    require(all(type(build.get(key)) is int and build[key] == 0
                for key in ("actual_exit_code", "tool_actual_exit_code")) and
            build.get("sources_unchanged") is True and build.get("root_terminal_before_deadline") is True
            and not build.get("error") and not build.get("deadline_exceeded"), "build lacks normal observed terminal")
    require(build.get("status") == "BUILT_NOT_GPU_QUALIFIED" and build.get("binary") == binary,
            "build/binary mismatch or unsupported build status")
    require(build.get("original_wire_protocol") is True and build.get("new_binary_speed_qualified") is False,
            "build must bind original wire without transferring old speed qualification")


def prepare(args):
    require(os.name == "nt", "this preparation targets Windows")
    require(10 <= args.timeout_seconds <= 300, "metadata timeout must be 10..300 seconds")
    deadline = time.monotonic_ns() + int(args.timeout_seconds * 1e9)
    if args.deadline_monotonic_ns is not None:
        deadline = min(deadline, args.deadline_monotonic_ns)
    inventory = Inventory(deadline)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema": "rustgo-original-protocol-worker-preparation-v1", "status": "FAILED",
              "error": None, "source_recheck_error": None, "maximum_model_starts": 0,
              "actual_model_starts": 0, "gpu_model_upload_or_inference": False,
              "metadata_cuda_context_created": None, "new_binary_speed_qualified": False,
              "old_speed_certificate_applies_to_new_binary": False,
              "accuracy": "USER_VERIFICATION_NOT_EVALUATED", "deadline_monotonic_ns": deadline}
    failure = None
    try:
        self_source = inventory.add(__file__)
        binary = inventory.add(args.binary, args.binary_sha256)
        require(binary["sha256"] != OLD_BINARY, "old measured binary must use its unchanged old entry")
        build_source = inventory.add(args.build_result, args.build_result_sha256)
        build = decode(inventory.raw(build_source))
        validate_build(build, binary)
        build_plan_source = inventory.pinned(build["plan"])
        build_plan = decode(inventory.raw(build_plan_source))
        require(build_plan.get("schema") == "rustgo-typed-worker-native-build-plan-v1", "wrong build plan schema")
        build_sources = build_plan["sources"]
        require(isinstance(build_sources, list), "missing build source inventory")
        algorithm_sources = []
        for relative, digest in ALGORITHM.items():
            item = inventory.add(ROOT / relative, digest)
            require(item in build_sources, f"algorithm Source absent from build plan: {relative}")
            algorithm_sources.append(item)
        worker_protocol_sources = []
        for relative, digest in WORKER_PROTOCOL.items():
            item = inventory.add(ROOT / relative, digest)
            require(item in build_sources, f"original Worker protocol Source absent from build plan: {relative}")
            worker_protocol_sources.append(item)
        batch, config_pin, recipe_pin = CANDIDATES[args.candidate]
        model, config, recipe = (fixed_source(inventory, pin) for pin in (MODEL, config_pin, recipe_pin))
        config_raw = inventory.raw(config)
        execution = config_values(config_raw, batch, recipe["path"])
        environment_source = inventory.add(args.environment, args.environment_sha256)
        environment_object = decode(inventory.raw(environment_source))
        require(environment_object.get("schema") == "rustgo-fixed-cpu-environment-v1", "unsupported base environment")
        env = selected_environment(environment_object["values"], args.gpu_uuid)
        require(re.fullmatch(r"[A-Za-z0-9._-]{1,96}", args.worker_id), "invalid Worker ID")
        require(re.fullmatch(r"https?://[^\s/]+(?::[0-9]+)?", args.server), "expected gRPC HTTP(S) authority")
        inspect_path = output / "recipe-inspection"
        metadata_process([binary["path"], "quant-inspect", "--model", model["path"], "--recipe",
                          recipe["path"], "--output", str(inspect_path)], env, output, "quant-inspect", deadline)
        identity_source = inventory.add(inspect_path / "recipe-identity.json")
        copied_recipe = inventory.add(inspect_path / "source-recipe.json", recipe["sha256"], recipe["bytes"])
        canonical = recipe_identity(decode(inventory.raw(identity_source)), model["sha256"], recipe["sha256"])
        version_before = driver_api_version()
        remaining(deadline)
        metadata_process([binary["path"], "cuda-fingerprint"], env, output, "cuda-fingerprint", deadline)
        result["metadata_cuda_context_created"] = True
        fingerprint_source = inventory.add(output / "cuda-fingerprint.stdout.txt")
        fingerprint = decode(inventory.raw(fingerprint_source))
        version_after = driver_api_version()
        require(version_before == version_after, "CUDA driver API version changed across metadata query")
        contract, expected = profile_contract(model["sha256"], canonical, binary["sha256"], fingerprint,
                                               version_after, env, execution)
        config_output = output / "worker.cfg"
        publish(config_output, config_raw.rstrip(b"\r\n") +
                f"\ncudaQuantExpectedProfile = {expected}\n".encode("utf-8"))
        config_output_source = inventory.add(config_output)
        environment_output = output / "worker-environment.json"
        write_json(environment_output, {"schema": "rustgo-explicit-worker-environment-v1", "values": env})
        environment_output_source = inventory.add(environment_output)
        write_json(output / "profile-contract.json", contract)
        profile_contract_source = inventory.add(output / "profile-contract.json")
        command = [binary["path"], "nnworker", "--server", args.server, "--worker-id", args.worker_id,
                   "--model", model["path"], "--model-sha256", model["sha256"],
                   "--config", str(config_output), "--capacity", "32", "--once"]
        command_record = {"schema": "rustgo-original-protocol-worker-command-v1", "argv": command,
                          "cwd": str(ROOT), "environment": environment_output_source,
                          "binary": binary, "model": model, "recipe": recipe, "config": config_output_source,
                          "expected_local_profile": expected,
                          "server_enforces_execution_profile": False, "automatic_retry": False,
                          "execution_authorized_by_preparation": False}
        write_json(output / "worker-command.json", command_record)
        command_source = inventory.add(output / "worker-command.json")
        publish(output / "launch-worker.ps1", launcher_script(command_source))
        launcher_source = inventory.add(output / "launch-worker.ps1")
        result.update(status="PREPARED_NOT_MODEL_STARTED", preparer=self_source, candidate=args.candidate,
                      binary=binary, build_result=build_source, build_plan=build_plan_source,
                      model=model, recipe=recipe, original_config=config, input_environment=environment_source,
                      algorithm_sources=algorithm_sources, worker_protocol_sources=worker_protocol_sources,
                      recipe_identity=identity_source,
                      copied_recipe=copied_recipe, canonical_recipe_sha256=canonical,
                      cuda_fingerprint=fingerprint_source, driver_version=version_after,
                      profile_contract=profile_contract_source, expected_local_profile=expected,
                      config=config_output_source, environment=environment_output_source, command=command_source,
                      launcher=launcher_source, launcher_default="Foreground persistent original Worker; -Once disables reconnect.",
                      selected_gpu_uuid=args.gpu_uuid, gpu_uuid_independently_observed=False,
                      protocol="ORIGINAL_V1_MODEL_SHA_ROUTING", server_enforces_execution_profile=False,
                      runtime_profile_agreement="NOT_OBSERVED_UNTIL_ACTUAL_WORKER_LOAD",
                      native_budget="No model start consumed here; external one-shot runner retains its original 95/96 budget.")
    except BaseException as error:
        failure = error
        result["status"], result["error"] = "FAILED", repr(error)
    try:
        inventory.recheck()
    except BaseException as error:
        result["source_recheck_error"] = repr(error)
        result["status"] = "FAILED"
        failure = failure or error
    result["scope"] = {"sourcefreeze": list(inventory.records.values()),
                       "build_inventory": "Only selected algorithm and original Worker protocol Sources were rehashed here; the build receipt owns the full build closure.",
                       "metadata_only": True, "speed_transfer": False}
    result["finished_monotonic_ns"] = time.monotonic_ns()
    write_json(output / "preparation.json", result)
    # A JSON written before this final boundary is not itself proof of actual0.
    remaining(deadline)
    if failure is not None:
        raise failure
    return source(output / "preparation.json", deadline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "build-result", "environment"):
        parser.add_argument("--" + name, required=True, type=Path)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--deadline-monotonic-ns", type=int)
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
