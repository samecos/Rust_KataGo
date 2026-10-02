"""Optional, experimental legacy CUDA execution receipts; no certification.

The module never invents a unified recipe/profile. Historical fingerprints lack
driver API and physical GPU UUID observations; both remain explicitly unknown.
Only the caller may decide to invoke probe_fingerprint (which starts one owned
cuda-fingerprint subprocess). All other functions are CPU/file validation only.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rustgo-legacy-quantization-execution-spec-v1"
SCOPE = "EXPERIMENTAL_EXECUTION_EVIDENCE"
SPEC_FILE = "legacy-execution-spec.json"
INT8_ENVIRONMENT = {"KATAGO_CUDA_ATTN_TILE": "q64", "KATAGO_CUDA_INT8_GEMM_TUNE": "0"}
CONFIG_KEYS = {"rules", "komi", "nnBackend", "nnMaxBatchSize", "numNNServerThreadsPerModel", "nnCacheSizePowerOfTwo",
               "nnMutexPoolSizePowerOfTwo", "numSearchThreads", "maxVisits", "ponderingEnabled", "cudaTacticPlan",
               "cudaInt8Scope", "cudaInt8MinFfnWidth"}
TARGET_KEYS = ("gpu_name", "compute_capability", "sm_count", "l2_cache_bytes", "model_sha256")


def require(value, message):
    if not value:
        raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def bad(value):
        raise ValueError(f"nonfinite JSON: {value}")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def fields(value, required, optional=()):
    require(type(value) is dict and set(required) <= set(value) <= set(required) | set(optional), "missing/unknown contract fields")


def check_sha(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA256")


def file_spec(value):
    fields(value, {"path", "sha256"})
    require(type(value["path"]) is str and Path(value["path"]).is_absolute(), "source paths must be absolute")
    check_sha(value["sha256"])


def parse_config(raw):
    values = {}
    for line in raw.decode("utf-8").splitlines():
        body = line.partition("#")[0].strip()
        if not body:
            continue
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_-]*)\s*=\s*(.+)", body)
        require(match is not None and match[1] in CONFIG_KEYS and match[1] not in values, "unsupported/duplicate/indirect legacy config")
        key, value = match[1], match[2].strip()
        if value.startswith('"'):
            require(value.endswith('"') and not any(c in value[1:-1] for c in '\\"'), "ambiguous config quoting")
            value = value[1:-1]
        else:
            require('"' not in value and "'" not in value, "ambiguous config value")
        require(value and "\x00" not in value, "empty/invalid config value")
        values[key] = value
    for key, low, high in (("nnMaxBatchSize", 1, 64), ("numNNServerThreadsPerModel", 1, 1024),
                           ("nnCacheSizePowerOfTwo", -1, 48), ("nnMutexPoolSizePowerOfTwo", -1, 24)):
        require(key in values and re.fullmatch(r"-?\d+", values[key]) and low <= int(values[key]) <= high, f"invalid explicit {key}")
    return values


def normalized_config(values, plan_sha):
    result = dict(values)
    if "cudaTacticPlan" in result:
        require(plan_sha is not None, "unbound config tactic plan")
        result["cudaTacticPlan"] = {"sha256": plan_sha}
    for key in ("nnMaxBatchSize", "numNNServerThreadsPerModel", "nnCacheSizePowerOfTwo", "nnMutexPoolSizePowerOfTwo",
                "numSearchThreads", "maxVisits", "cudaInt8MinFfnWidth"):
        if key in result:
            require(re.fullmatch(r"-?\d+", result[key]), f"noninteger {key}")
            result[key] = int(result[key])
    return result


def decode_spec(raw, config_raw, plan_raw):
    spec = strict_json(raw)
    fields(spec, {"schema", "backend", "binary", "model", "config", "plan", "batch", "environment"}, {"label"})
    require(spec["schema"] == SCHEMA and spec["backend"] in ("cudabackend", "cudaint8backend"), "unsupported legacy spec")
    if "label" in spec:
        require(type(spec["label"]) is str and len(spec["label"]) <= 256, "invalid label")
    for name in ("binary", "model", "config"):
        file_spec(spec[name])
    require(type(spec["batch"]) is int and 1 <= spec["batch"] <= 64, "invalid batch")
    require(type(spec["environment"]) is dict, "environment must be explicit")
    require(sha(config_raw) == spec["config"]["sha256"], "source config SHA differs")
    values = parse_config(config_raw)
    require(values.get("nnBackend") == spec["backend"] and int(values["nnMaxBatchSize"]) == spec["batch"], "config backend/batch differs")
    plan = None
    if spec["backend"] == "cudabackend":
        require(spec["environment"] == {} and spec["plan"] is not None and "cudaTacticPlan" in values,
                "legacy FP16 requires original plan and empty explicit environment")
        require("cudaInt8Scope" not in values and "cudaInt8MinFfnWidth" not in values, "FP16 spec contains INT8 config")
        file_spec(spec["plan"])
        require(plan_raw is not None and sha(plan_raw) == spec["plan"]["sha256"], "source plan SHA differs")
        plan = strict_json(plan_raw)
        require(plan.get("schema") == 2 and plan.get("kind") == "cuda-tactic-plan" and type(plan.get("plan_id")) is str
                and plan["plan_id"] and type(plan.get("backend_build")) is dict and type(plan.get("target")) is dict,
                "legacy FP16 requires original schema2 plan identity")
        require(plan["target"].get("model_sha256") == spec["model"]["sha256"], "plan targets another model")
        require(plan["target"].get("max_batch_size", spec["batch"]) == spec["batch"], "plan batch binding differs")
        selection = plan.get("selection", {})
        claimed_binaries = (selection.get("binary", {}).get("sha256"), selection.get("binary_sha256"))
        require(all(value is None or value == spec["binary"]["sha256"] for value in claimed_binaries), "plan historical binary binding differs")
    else:
        require(spec["plan"] is None and plan_raw is None and "cudaTacticPlan" not in values, "INT8 cannot inherit an FP16 plan")
        require(spec["environment"] == INT8_ENVIRONMENT, "INT8 requires exactly explicit q64/tune0")
        require(values.get("cudaInt8Scope") == "ffn" and values.get("cudaInt8MinFfnWidth") == "0", "INT8 scope/minimum width must be ffn/0")
    semantic = dict(schema=SCHEMA, backend=spec["backend"], binary_sha256=spec["binary"]["sha256"], model_sha256=spec["model"]["sha256"],
                    config=normalized_config(values, spec["plan"]["sha256"] if spec["plan"] else None), batch=spec["batch"],
                    environment=spec["environment"], identity_scope=SCOPE)
    return dict(raw=raw, spec=spec, semantic=semantic, spec_id=sha(compact(semantic)), backend=spec["backend"], environment=spec["environment"],
                binary=spec["binary"]["path"], model=spec["model"]["path"], config=spec["config"]["path"], config_values=values, plan=plan)


def load_spec(path, *, binary=None, model=None, config=None, backend=None, batch=None):
    path = Path(path).resolve()
    raw = path.read_bytes()
    spec = strict_json(raw)
    # Validate the file table before using any externally supplied paths.
    for name in ("binary", "model", "config"):
        file_spec(spec[name])
    overrides = dict(binary=binary, model=model, config=config)
    actual = {}
    for name, override in overrides.items():
        original = Path(spec[name]["path"]).resolve()
        require(original.is_file() and file_sha(original) == spec[name]["sha256"], f"immutable source {name} changed")
        actual[name] = Path(override).resolve() if override is not None else original
        require(actual[name].is_file() and file_sha(actual[name]) == spec[name]["sha256"], f"provided {name} SHA differs")
    plan_raw = None
    if spec.get("plan") is not None:
        file_spec(spec["plan"])
        plan_raw = Path(spec["plan"]["path"]).read_bytes()
    loaded = decode_spec(raw, actual["config"].read_bytes(), plan_raw)
    if loaded["plan"] is not None:
        value = Path(loaded["config_values"]["cudaTacticPlan"])
        effective = (value if value.is_absolute() else ROOT / value).resolve()
        require(effective == Path(spec["plan"]["path"]).resolve(), "source config does not reference the declared original plan")
    require(backend is None or backend == loaded["backend"], "CLI backend differs from spec")
    require(batch is None or batch == loaded["spec"]["batch"], "CLI batch differs from spec")
    loaded.update({name: str(value) for name, value in actual.items()})
    loaded["source_path"] = str(path)
    loaded["files"] = [{"path": str(actual[name]), "sha256": spec[name]["sha256"]} for name in actual]
    if spec["plan"]:
        loaded["files"].append(spec["plan"])
    return loaded


def artifact(directory, path):
    raw = path.read_bytes()
    return dict(file=path.relative_to(directory).as_posix(), bytes=len(raw), sha256=sha(raw))


def bound(directory, item):
    fields(item, {"file", "sha256"}, {"bytes", "spec_id"})
    name = item["file"]
    require(type(name) is str and name and "\\" not in name and ":" not in name and not Path(name).is_absolute()
            and all(part not in ("", ".", "..") for part in name.split("/")), "unsafe execution artifact path")
    path = directory / name
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(directory.resolve())
            and not any(p.is_symlink() for p in path.parents if p.is_relative_to(directory)), "missing/nonlocal execution artifact")
    raw = path.read_bytes()
    require(sha(raw) == item["sha256"] and ("bytes" not in item or len(raw) == item["bytes"]), "execution artifact hash/size differs")
    return raw


def freeze_spec(loaded, directory):
    directory = Path(directory).resolve()
    write_new(directory / SPEC_FILE, loaded["raw"])
    return dict(**artifact(directory, directory / SPEC_FILE), spec_id=loaded["spec_id"])


def apply_environment(loaded, environment):
    return {**{k: v for k, v in environment.items() if not k.upper().startswith("KATAGO_")}, **loaded["environment"]}


def checked_environment(loaded, environment):
    explicit = {k: v for k, v in environment.items() if k.upper().startswith("KATAGO_")}
    require(explicit == loaded["environment"], "actual environment differs from explicit legacy tactics")
    device = {key: environment.get(key) for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")}
    return explicit, device


def validate_fingerprint(fingerprint, loaded):
    require(type(fingerprint) is dict and set(TARGET_KEYS) | {"architecture", "backend_build"} <= fingerprint.keys(), "incomplete CUDA fingerprint")
    require(fingerprint["model_sha256"] == loaded["spec"]["model"]["sha256"], "fingerprint model mismatch")
    require(type(fingerprint["gpu_name"]) is str and fingerprint["gpu_name"] and re.fullmatch(r"\d+\.\d+", fingerprint["compute_capability"])
            and fingerprint["architecture"] == "sm_" + fingerprint["compute_capability"].replace(".", "")
            and all(type(fingerprint[k]) is int and fingerprint[k] > 0 for k in ("sm_count", "l2_cache_bytes"))
            and type(fingerprint["backend_build"]) is dict and fingerprint["backend_build"].get("kernel_build_id"), "invalid CUDA fingerprint")
    if loaded["plan"] is not None:
        plan = loaded["plan"]
        require(all(plan["target"].get(key) == fingerprint[key] for key in TARGET_KEYS)
                and plan["backend_build"] == fingerprint["backend_build"], "original plan device/backend build binding mismatch")
    else:
        architecture = fingerprint.get("model_architecture", {})
        widths = architecture.get("ffn_hidden")
        require(architecture.get("format") == "native_tf3_v17" and type(widths) is list and widths
                and all(type(width) is int and width > 0 for width in widths), "INT8 fingerprint lacks actual model FFN widths")
    return fingerprint


def performance_clock():
    """Clock shared with drive_window; Python 3.11 Windows monotonic differs."""
    clock = time.get_clock_info("perf_counter")
    return dict(api="time.perf_counter_ns", implementation=clock.implementation,
                monotonic=clock.monotonic, adjustable=clock.adjustable,
                resolution_seconds=clock.resolution)


def probe_interval(directory, descriptor, *, require_performance_clock=False):
    """Read a hash-bound interval; unmarked historical receipts stay numeric-only."""
    directory = Path(directory).resolve()
    command_raw = bound(directory, descriptor["command"])
    command = strict_json(command_raw)
    result = strict_json(bound(directory, descriptor["result"]))
    require(command.get("schema") == "rustgo-legacy-fingerprint-command-v1"
            and result.get("schema") == "rustgo-legacy-fingerprint-result-v1"
            and result.get("command_sha256") == sha(command_raw), "invalid fingerprint interval binding")
    require(("timing_clock" in command) == ("timing_clock" in result), "partial fingerprint timing clock")
    clock = None
    if "timing_clock" in command:
        clock = command["timing_clock"]
        fields(clock, {"api", "implementation", "monotonic", "adjustable", "resolution_seconds"})
        require(clock["api"] == "time.perf_counter_ns"
                and type(clock["implementation"]) is str and clock["implementation"]
                and clock["monotonic"] is True and clock["adjustable"] is False
                and type(clock["resolution_seconds"]) in (int, float)
                and math.isfinite(clock["resolution_seconds"]) and clock["resolution_seconds"] > 0,
                "invalid fingerprint performance clock")
        require(clock == result["timing_clock"], "fingerprint timing clock changed within probe")
    require(not require_performance_clock or clock is not None,
            "historical unmarked probe cannot prove performance clock ordering")
    start, end = command.get("started_ns"), result.get("ended_ns")
    require(type(start) is int and type(end) is int and 0 < start <= end, "invalid fingerprint probe interval")
    return dict(timing_clock=clock, started_ns=start, ended_ns=end)


def probe_fingerprint(loaded, directory, label, *, environment, runner=None, timeout=300):
    require(label in ("before", "after"), "fingerprint label must be before/after")
    require(type(timeout) in (int, float) and math.isfinite(timeout) and 0 < timeout <= 3600, "invalid probe timeout")
    directory = Path(directory).resolve()
    output = directory / f"{label}-fingerprint"
    require(not output.exists(), "fingerprint probe cannot overwrite or retry")
    explicit, device = checked_environment(loaded, environment)
    for source in loaded["files"]:
        require(file_sha(source["path"]) == source["sha256"], "source changed before fingerprint")
    output.mkdir()
    command = dict(schema="rustgo-legacy-fingerprint-command-v1", argv=[loaded["binary"], "cuda-fingerprint", "--model", loaded["model"]],
                   cwd=str(ROOT), environment=explicit, device_environment=device, spec_id=loaded["spec_id"], timeout_seconds=timeout,
                   label=label, timing_clock=performance_clock(), started_ns=time.perf_counter_ns())
    write_new(output / "command.json", command)  # precedes any subprocess.
    result, error, stdout, stderr = None, None, b"", b""
    try:
        result = (runner or subprocess.run)(command["argv"], cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            timeout=timeout, check=False, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        stdout, stderr = result.stdout, result.stderr
        require(type(stdout) is bytes and type(stderr) is bytes and result.returncode == 0, "cuda-fingerprint subprocess failed")
        validate_fingerprint(strict_json(stdout), loaded)
        for source in loaded["files"]:
            require(file_sha(source["path"]) == source["sha256"], "source changed during fingerprint")
    except BaseException as caught:
        error = caught
        stdout = getattr(caught, "stdout", None) or stdout
        stderr = getattr(caught, "stderr", None) or stderr
    write_new(output / "stdout.json", stdout)
    write_new(output / "stderr.log", stderr)
    write_new(output / "result.json", dict(schema="rustgo-legacy-fingerprint-result-v1", status="FAILED" if error else "PASS",
              returncode=result.returncode if result else None, error=str(error) if error else None,
              timing_clock=command["timing_clock"], ended_ns=time.perf_counter_ns(),
              command_sha256=file_sha(output / "command.json"), stdout_sha256=sha(stdout), stderr_sha256=sha(stderr)))
    if error:
        raise error
    return dict(schema="rustgo-legacy-fingerprint-evidence-v1", label=label,
                **{key: artifact(directory, output / name) for key, name in
                   (("command", "command.json"), ("stdout", "stdout.json"), ("stderr", "stderr.log"), ("result", "result.json"))})


def verify_probe(directory, descriptor, loaded, *, binary, model):
    fields(descriptor, {"schema", "label", "command", "stdout", "stderr", "result"})
    require(descriptor["schema"] == "rustgo-legacy-fingerprint-evidence-v1" and descriptor["label"] in ("before", "after"), "invalid fingerprint receipt")
    files = {name: bound(directory, descriptor[name]) for name in ("command", "stdout", "stderr", "result")}
    require(all(descriptor[key]["file"] == f"{descriptor['label']}-fingerprint/{name}" for key, name in
                (("command", "command.json"), ("stdout", "stdout.json"), ("stderr", "stderr.log"), ("result", "result.json"))), "fingerprint paths do not bind the labeled probe")
    command, result = strict_json(files["command"]), strict_json(files["result"])
    require(command["schema"] == "rustgo-legacy-fingerprint-command-v1" and command["argv"] == [str(binary), "cuda-fingerprint", "--model", str(model)]
            and command["cwd"] == str(ROOT) and command["environment"] == loaded["environment"] and command["spec_id"] == loaded["spec_id"]
            and command["label"] == descriptor["label"],
            "fingerprint command/binary/model/environment identity differs")
    require(type(command["device_environment"]) is dict and set(command["device_environment"]) == {"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"}
            and all(value is None or type(value) is str for value in command["device_environment"].values()), "invalid device selection receipt")
    require(result["schema"] == "rustgo-legacy-fingerprint-result-v1" and result["status"] == "PASS" and result["returncode"] == 0
            and result["error"] is None and all(result[key + "_sha256"] == sha(files[key]) for key in ("command", "stdout", "stderr")),
            "fingerprint result is failed or not bound to raw files")
    interval = probe_interval(directory, descriptor)
    return validate_fingerprint(strict_json(files["stdout"]), loaded), command["device_environment"], (interval["started_ns"], interval["ended_ns"])


def hello_fields(hello):
    require(type(hello) is dict and type(hello.get("backend_info")) is str, "missing actual Worker Hello")
    fields_out = {}
    for part in hello["backend_info"].split(";"):
        if "=" in part:
            key, value = part.strip().split("=", 1)
            require(key not in fields_out, "duplicate Worker Hello backend identity")
            fields_out[key] = value
    return fields_out


def actual_execution(loaded, fingerprint, hello, log, effective_plan_path=None):
    # Raw log bytes remain hash-bound; match line records consistently for both
    # Windows CRLF and Unix LF writers.
    log = log.replace("\r\n", "\n").replace("\r", "\n")
    fields_out = hello_fields(hello)
    require(hello.get("model_sha256") == loaded["spec"]["model"]["sha256"] and fields_out.get("backend") == loaded["backend"], "actual Worker model/backend differs")
    require("inference-profile" not in fields_out, "legacy Worker cannot borrow a unified inference profile")
    installed = re.findall(r"cudaTacticPlan '([^']+)' installed \(plan id: ([^\r\n)]+)\)", log)
    if loaded["backend"] == "cudabackend":
        require(installed and {(str(Path(path).resolve()), plan_id) for path, plan_id in installed}
                == {(str(Path(effective_plan_path).resolve()), loaded["plan"]["plan_id"])}, "original FP16 plan loader installation marker missing/conflicting")
        require("[cuda-int8]" not in log and fields_out.get("precision") != "W8A8-mixed", "FP16 receipt contains INT8 execution")
        return dict(backend="cudabackend", plan_id=loaded["plan"]["plan_id"], plan_sha256=loaded["spec"]["plan"]["sha256"],
                    loader="INSTALLED_WITH_ORIGINAL_DEVICE_BUILD_CHECKS")
    require(not installed, "INT8 actual execution inherited a legacy FP16 plan")
    require(all(fields_out.get(key) == value for key, value in
                (("precision", "W8A8-mixed"), ("int8-scope", "ffn"), ("int8-min-ffn-width", "0"), ("quantization", "w8a8-row-out-rne-v1"))),
            "actual INT8 Hello semantics differ")
    tune_lines = re.findall(r"(?m)^.*\[cuda-tactic\] name=int8_gemm_tune ([^\r\n]+)$", log)
    require(tune_lines and set(tune_lines) == {"enabled=0 candidate_pool=16 default=0"}, "actual INT8 tune0 marker missing/conflicting")
    attention = re.findall(r"(?m)^.*\[cuda-tactic\] name=attention ([^\r\n]+)$", log)
    require(attention and set(attention) == {"requested=fa2 launch=fa2 effective=fa2 tile=q64"}, "actual q64 attention marker missing/conflicting")
    expected_layers = len(fingerprint["model_architecture"]["ffn_hidden"])
    expected = (f"precision=W8A8 accumulation=INT32 scope=ffn min_ffn_width=0 int8_ffn_layers={expected_layers} "
                f"quantization=w8a8-row-out-rne-v1 model_sha256={loaded['spec']['model']['sha256']} "
                "heads=FP16/FP32 activation_scaling=dynamic-per-row calibration=none")
    precision = re.findall(r"(?m)^.*\[cuda-int8\] (precision=[^\r\n]+)$", log)
    require(precision and set(precision) == {expected}, "actual INT8 precision/model/layer count marker missing/conflicting")
    launches = re.findall(r"(?m)^.*\[cuda-int8\] (launch=[^\r\n]+)$", log)
    pattern = r"launch=cublaslt-int8 accumulator=int32 imma=true numerical_flags=0x([0-9a-fA-F]+) M=(\d+) N=(\d+) K=(\d+)"
    require(launches and all((match := re.fullmatch(pattern, line)) and int(match[1], 16) & 4 != 0
                            and all(int(match[i]) > 0 for i in (2, 3, 4)) for line in launches), "actual INT8 IMMA launch marker missing/conflicting")
    return dict(backend="cudaint8backend", precision="W8A8", accumulation="INT32", scope="ffn", min_ffn_width=0,
                int8_ffn_layers=expected_layers, quantization="w8a8-row-out-rne-v1", attention="fa2/q64", gemm_tune=0,
                gemm="cublaslt-int8", imma=True, numerical_flags=sorted({int(re.fullmatch(pattern, line)[1], 16) for line in launches}))


def collection_sources(directory, report):
    legacy = report["legacy_execution"]
    raw = bound(directory, legacy["spec"])
    spec = strict_json(raw)
    entries = report.get("artifacts", [])
    names = [entry["file"] for entry in entries]
    require(len(names) == len(set(names)), "duplicate collection artifacts")
    artifacts = {entry["file"]: entry for entry in entries}
    require({SPEC_FILE, "source-worker.cfg", "worker.cfg", "worker.log"} <= artifacts.keys(), "legacy collection must bind spec/source config/effective config/worker log")
    require(bound(directory, artifacts[SPEC_FILE]) == raw, "collection spec artifact differs")
    source_config = bound(directory, artifacts["source-worker.cfg"])
    effective_config = bound(directory, artifacts["worker.cfg"])
    log = bound(directory, artifacts["worker.log"]).decode("utf-8", errors="strict")
    plan_raw, effective_plan = None, None
    runtimes = report.get("runtime_artifacts", [])
    if spec["plan"] is not None:
        require(len(runtimes) == 1 and runtimes[0]["config_key"] == "cudaTacticPlan", "legacy FP16 requires exactly its bound original plan")
        item = runtimes[0]
        require(item["file"] in artifacts, "frozen runtime plan is not a collection artifact")
        plan_raw = bound(directory, artifacts[item["file"]])
        require(sha(plan_raw) == item["sha256"] == spec["plan"]["sha256"], "effective plan bytes changed")
        effective_plan = parse_config(effective_config)["cudaTacticPlan"]
        require(Path(effective_plan).is_absolute(), "effective frozen plan path must be absolute")
        command = report.get("command", [])
        require(type(command) is list and command.count("--config") == 1 and command.index("--config") + 1 < len(command), "collection must bind its actual Worker config command")
        actual_config = Path(command[command.index("--config") + 1])
        require(actual_config.is_absolute() and actual_config.name == "worker.cfg"
                and Path(effective_plan).resolve() == (actual_config.parent / item["file"]).resolve(), "effective config points outside its bound runtime plan")
    else:
        require(runtimes == [], "unexpected INT8 runtime artifact")
    loaded = decode_spec(raw, source_config, plan_raw)
    require(legacy["spec"]["spec_id"] == loaded["spec_id"], "frozen spec semantic identity differs")
    values = parse_config(effective_config)
    require(normalized_config(values, spec["plan"]["sha256"] if spec["plan"] else None) == loaded["semantic"]["config"], "effective config changes execution semantics")
    require(sha(effective_config) == report["config_sha256"] and effective_config.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n") == report["config_text"], "effective config report identity differs")
    require(report["binary_sha256"] == spec["binary"]["sha256"] and report["model_sha256"] == spec["model"]["sha256"]
            and report["backend"] == spec["backend"] and report["batch"] == spec["batch"]
            and report.get("actual_profile") is None and report.get("recipe_file_sha256") is None,
            "legacy collection borrows another binary/model/backend/recipe identity")
    require(report.get("controlled_environment") == loaded["environment"], "collection declared tactics differ from spec")
    return loaded, log, effective_plan


def identity_from_evidence(directory, legacy, loaded, hello, log, binary, model, effective_plan):
    require(legacy["before"]["label"] == "before" and legacy["after"]["label"] == "after", "fingerprint before/after labels differ")
    before, before_env, before_time = verify_probe(directory, legacy["before"], loaded, binary=binary, model=model)
    after, after_env, after_time = verify_probe(directory, legacy["after"], loaded, binary=binary, model=model)
    require(compact(before) == compact(after) and before_env == after_env, "fingerprint/device environment changed across execution")
    require(probe_interval(directory, legacy["before"])["timing_clock"]
            == probe_interval(directory, legacy["after"])["timing_clock"], "fingerprint timing clock changed across execution")
    require(after_time[0] > before_time[1], "fingerprint probes overlap or replay the same interval")
    execution = actual_execution(loaded, before, hello, log, effective_plan)
    contract = dict(schema="rustgo-observed-legacy-execution-v1", evidence_scope=SCOPE, spec_id=loaded["spec_id"],
                    fingerprint=before, device_environment=before_env, actual_execution=execution,
                    driver_api_version=None, gpu_uuid=None, logical_cuda_device=0,
                    limitations=["driver API version and physical GPU UUID are not observed", "not a unified runtime profile or original FP32 certification"])
    observed = "rustgo-legacy-evidence-v1:" + sha(compact(contract))
    require("observed_execution_id" not in legacy or legacy["observed_execution_id"] == observed, "saved observed execution identity differs from raw evidence")
    paths = [legacy[side][name]["file"] for side in ("before", "after") for name in ("command", "stdout", "stderr", "result")]
    return dict(**contract, observed_execution_id=observed, artifact_paths=paths)


def validate_collection(directory, report):
    directory = Path(directory).resolve()
    require(report.get("split") in ("calibration", "selection"), "legacy selection evidence rejects holdout before output access")
    loaded, log, effective_plan = collection_sources(directory, report)
    result = identity_from_evidence(directory, report["legacy_execution"], loaded, report["hello"], log,
                                    report["binary"], report["model"], effective_plan)
    result["artifact_paths"] = [SPEC_FILE, "source-worker.cfg", "worker.cfg", "worker.log",
                               *[item["file"] for item in report.get("runtime_artifacts", [])], *result["artifact_paths"]]
    return result


def validate_arm(directory, arm_report, collection_directory, collection_report):
    directory, collection_directory = Path(directory).resolve(), Path(collection_directory).resolve()
    collection = validate_collection(collection_directory, collection_report)
    loaded, _, effective_plan = collection_sources(collection_directory, collection_report)
    require(arm_report["environment"] == loaded["environment"], "ABBA environment differs from numerical evidence")
    require(arm_report.get("batch_capacity") == collection_report["batch"]
            and arm_report.get("protocol_capacity") == collection_report["capacity"]
            and arm_report.get("concurrency") == collection_report["request_window"], "ABBA batch/window/capacity differs from numerical evidence")
    files = arm_report["files"]
    runtime_count = 4 if loaded["plan"] else 3
    require(len(files) >= runtime_count and files[0]["sha256"] == loaded["spec"]["binary"]["sha256"]
            and files[1]["sha256"] == loaded["spec"]["model"]["sha256"] and files[2]["sha256"] == collection_report["config_sha256"]
            and Path(files[2]["path"]).resolve() == collection_directory / "worker.cfg", "ABBA runtime files differ from collection")
    if loaded["plan"]:
        require(files[3]["sha256"] == loaded["spec"]["plan"]["sha256"]
                and Path(files[3]["path"]).resolve() == Path(effective_plan).resolve(), "ABBA actual plan dependency differs")
    allowed = {str((collection_directory / name).resolve()): file_sha(collection_directory / name) for name in collection["artifact_paths"]}
    for item in files[runtime_count:]:
        require(allowed.get(str(Path(item["path"]).resolve())) == item["sha256"], "ABBA extra dependency is not bound collection evidence")
    # The benchmark also validates the isolated command, exact request stream,
    # counter deltas and raw timings. This module validates execution identity.
    logs = [item for item in arm_report.get("artifacts", []) if item["file"] == "worker.log"]
    require(len(logs) == 1, "ABBA must bind exactly one actual Worker log")
    log = bound(directory, logs[0]).decode("utf-8", errors="strict")
    result = identity_from_evidence(directory, arm_report["legacy_execution"], loaded, arm_report["hello"],
                                    log, files[0]["path"], files[1]["path"], effective_plan)
    require(result["observed_execution_id"] == collection["observed_execution_id"], "ABBA observed execution differs from numerical collection")
    result["artifact_paths"].append("worker.log")
    return result
