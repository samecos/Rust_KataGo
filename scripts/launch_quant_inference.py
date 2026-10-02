"""Launch an explicitly measured local inference profile, or an explicit fallback.

prepare (the default) checks metadata and nvidia-smi only. eval measures the
evaluator with C32; gtp is an ordinary local GTP instance, not a search speed
claim. This launcher does not connect nnworker to a shared server.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import types


ACCURACY = "USER_VERIFICATION_NOT_EVALUATED"
PROFILE = re.compile(r"rustgo-quant-v1:[0-9a-f]{64}\Z")
ENV_KEYS = frozenset("KATAGO_CUDA_" + key for key in (
    "ATTN", "ATTN_TILE", "CUBLASLT", "CUBLASLT_RANK", "DUALFFN", "FUSION",
    "GEMM_LAYOUT", "NOGRAPH", "NOPIPELINE", "PADBATCH", "SPLITK",
    "INT8_GEMM_TUNE", "INT8_RMS_FUSION", "BATCH_TRACE"))
GPU_QUERY = "index,name,uuid,compute_cap,driver_version"
OLD_GPU_QUERY = "name,uuid,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used"


def require(value, message):
    if not value:
        raise ValueError(message)


def source(path):
    path = Path(path).resolve(strict=True)
    require(path.is_file(), "Source is not a file")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return dict(path=str(path), bytes=size, sha256=digest.hexdigest())


class Inventory:
    def __init__(self):
        self.records = {}

    def verify(self, record):
        require(isinstance(record, dict) and set(record) == {"path", "bytes", "sha256"},
                "invalid Source fields")
        require(isinstance(record["path"], str) and Path(record["path"]).is_absolute()
                and type(record["bytes"]) is int and record["bytes"] >= 0
                and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]), "invalid Source")
        actual = source(record["path"])
        require(actual == record, "Source changed: " + record["path"])
        key = os.path.normcase(actual["path"])
        require(key not in self.records or self.records[key] == record, "conflicting Source")
        self.records[key] = dict(record)
        return actual

    def capture(self, path):
        return self.verify(source(path))

    def raw(self, record):
        self.verify(record)
        require(record["bytes"] <= 4 * 1024 * 1024, "metadata too large")
        raw = Path(record["path"]).read_bytes()
        require(len(raw) == record["bytes"] and hashlib.sha256(raw).hexdigest() == record["sha256"],
                "metadata changed while reading")
        return raw

    def read(self, record):
        return json.loads(self.raw(record))

    def recheck(self):
        for record in list(self.records.values()):
            self.verify(record)


def query_gpu(index):
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=" + GPU_QUERY, "--format=csv,noheader,nounits", "-i", str(index)],
        capture_output=True, text=True, timeout=5,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    require(completed.returncode == 0, "GPU query failed")
    rows = list(csv.reader(io.StringIO(completed.stdout.strip())))
    require(len(rows) == 1 and len(rows[0]) == 5, "expected exactly one GPU")
    got_index, name, uuid, capability, driver = (s.strip() for s in rows[0])
    require(got_index == str(index) and uuid.startswith("GPU-"), "GPU identity unavailable")
    return dict(index=index, name=name, uuid=uuid, compute_capability=capability, driver=driver)


def measured_gpu(inventory, manifest, profile):
    """Bind the selected four arms in each round, including all eight telemetry pairs."""
    identities = set()
    for round_name in ("formal", "confirmation"):
        report = inventory.read(manifest[round_name + "_report"])
        require(report["schema"] == "rustgo-quant-speed-report-v1"
                and report["status"] == "COMPLETE_SPEED_ONLY", "incomplete measured report")
        require(report["comparisons"][profile["group"]] == profile[round_name]
                and profile[round_name]["speed_pass"] is True, "unconfirmed speed profile")
        plan = inventory.read(report["plan"])
        require(plan["schema"] == "rustgo-quant-speed-plan-v1" and plan["workers"] == 32,
                "wrong measured load")
        for name in ("binary", "runner", "environment", "source_nnbench"):
            require(plan[name] == manifest[name], "report lineage changed: " + name)
        for role in ("baseline", "candidate"):
            matches = [v for v in plan["variants"] if v["id"] == profile[role]["id"]]
            require(len(matches) == 1, "ambiguous measured variant")
            variant = dict(matches[0])
            variant.setdefault("binary", plan["binary"])
            require(variant == profile[role], "measured variant differs from handoff")
        runs = [r for r in report["runs"] if r.get("comparison") == profile["group"]]
        require([r["stage"] for r in runs] == ["A1", "B1", "B2", "A2"], "incomplete ABBA")
        for run, role in zip(runs, ("baseline", "candidate", "candidate", "baseline")):
            require(run["status"] == "COMPLETE_SPEED_SAMPLE" and run["actual_exit"] == 0
                    and run["cleanup_errors"] == [] and run["flavor"] == role
                    and run["variant"] == profile[role]["id"], "unfinished or wrong measured arm")
            expected = profile["inference_profile"] if role == "candidate" else None
            require(run["metrics"]["inference_profile"] == expected, "measured profile mismatch")
            for side in ("before", "after"):
                telemetry = run[side]
                require(telemetry["returncode"] == 0 and telemetry["query"] == OLD_GPU_QUERY,
                        "missing measured GPU identity")
                rows = list(csv.reader(io.StringIO(telemetry["output"])))
                require(len(rows) == 1 and len(rows[0]) == 9, "ambiguous measured GPU")
                identities.add(tuple(s.strip() for s in rows[0][:3]))
    require(len(identities) == 1, "measured GPU changed")
    name, uuid, driver = identities.pop()
    require(name == manifest["device"] and driver == manifest["driver"]
            and uuid.startswith("GPU-"), "handoff GPU differs from measurement")
    return dict(name=name, uuid=uuid, driver=driver,
                compute_capability=manifest["compute_capability"])


def candidate_environment(inventory, manifest, candidate, uuid):
    environment = inventory.read(manifest["environment"])["values"]
    require(isinstance(environment, dict) and all(isinstance(k, str) and isinstance(v, str)
                                                for k, v in environment.items()), "invalid frozen environment")
    overrides = candidate["environment"]
    require(set(overrides) == ENV_KEYS and all(isinstance(v, str) for v in overrides.values()),
            "expected fourteen frozen KATAGO overrides")
    environment = {k: v for k, v in environment.items()
                   if not k.upper().startswith("KATAGO_") and k.upper() != "CUDA_VISIBLE_DEVICES"}
    environment.update(overrides)
    environment["CUDA_VISIBLE_DEVICES"] = uuid
    return environment


def locked_config(raw, candidate, profile):
    values = {}
    for line in raw.decode("utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        require("=" in line, "unrecognized candidate config line")
        key, value = (s.strip() for s in line.split("=", 1))
        require(key not in values, "duplicate candidate config key")
        values[key] = value
    require(values.get("nnBackend") == "cudaquantbackend"
            and values.get("nnMaxBatchSize") == str(candidate["batch_cap"])
            and values.get("cudaQuantPlan") == candidate["recipe"]["path"], "wrong candidate config")
    require(not ({"cudaQuantExpectedProfile", "cudaTacticPlan", "cudaInt8Scope", "cudaInt8MinFfnWidth"}
                 & values.keys()), "conflicting candidate config")
    require(PROFILE.fullmatch(profile), "invalid expected profile")
    newline = b"\r\n" if b"\r\n" in raw else b"\n"
    return raw + (b"" if raw.endswith(b"\n") else newline) + b"cudaQuantExpectedProfile = " + profile.encode("ascii") + newline


def select(args, inventory, gpu_query=query_gpu):
    manifest_source = inventory.capture(args.profiles)
    require(manifest_source["sha256"] == args.profiles_sha256, "wrong profiles digest")
    manifest = inventory.read(manifest_source)
    require(manifest["schema"] == "rustgo-speed-profile-handoff-v1"
            and manifest["status"] == "COMPLETE_SPEED_ONLY"
            and manifest["accuracy"] == ACCURACY and manifest["workers"] == 32
            and manifest["reset_inherited_katago_environment"] is True, "invalid speed handoff")
    model = inventory.capture(args.model)
    candidates = [p for p in manifest["profiles"]
                  if p["candidate"]["batch_cap"] == args.batch_cap
                  and all(p["candidate"]["model"][k] == model[k] for k in ("bytes", "sha256"))]
    require(len(candidates) <= 1, "ambiguous speed profile")
    result = dict(selection="NO_MATCH", profiles=manifest_source, model=model,
                  measured_speed_applicable=False, reason="unmeasured model, batch cap, or capacity")
    if candidates and args.capacity == 32:
        profile = candidates[0]
        require(profile["speed_confirmed"] is True and profile["accuracy"] == ACCURACY,
                "profile is not speed confirmed")
        measured = measured_gpu(inventory, manifest, profile)
        try:
            gpu = gpu_query(args.gpu_index)
        except (OSError, ValueError, subprocess.SubprocessError):
            gpu = None
        result.update(gpu=gpu, measured_gpu=measured, reason="GPU identity or driver is not measured")
        if gpu is not None and gpu.get("index") == args.gpu_index and all(gpu.get(k) == v for k, v in measured.items()):
            candidate = profile["candidate"]
            require(candidate["backend"] == "cudaquantbackend" and candidate["role"] == "candidate"
                    and candidate["batch_cap"] in (3, 8) and candidate["binary"] == manifest["binary"],
                    "invalid matched candidate")
            # Once matched, any broken original is an error, never a reason to fall back.
            for name in ("binary", "runner", "environment", "source_nnbench"):
                inventory.verify(manifest[name])
            for name in ("binary", "model", "config", "recipe"):
                inventory.verify(candidate[name])
            config = locked_config(inventory.raw(candidate["config"]), candidate, profile["inference_profile"])
            environment = candidate_environment(inventory, manifest, candidate, gpu["uuid"])
            result.update(selection="MEASURED_PROFILE", reason="exact measured model, GPU, driver and load",
                          measured_speed_applicable=True, group=profile["group"], variant=candidate["id"],
                          binary=candidate["binary"], original_config=candidate["config"], recipe=candidate["recipe"],
                          expected_profile=profile["inference_profile"], runner=manifest["runner"],
                          environment_source=manifest["environment"], katago_override_keys=sorted(ENV_KEYS),
                          cuda_visible_device=gpu["uuid"], backend="cudaquantbackend")
            return result, environment, config
    if args.fallback_binary is not None:
        result.update(selection="EXPLICIT_FALLBACK", binary=inventory.capture(args.fallback_binary),
                      original_config=inventory.capture(args.fallback_config),
                      environment_policy="caller environment preserved; no measured speed claim")
    return result, None, None


def load_parser(inventory, record):
    raw = inventory.raw(record)
    module = types.ModuleType("_bound_quant_speed_parser")
    module.__file__ = record["path"]
    exec(compile(raw, record["path"], "exec"), module.__dict__)
    return module.parse_eval


def save(path, record):
    # Receipts contain no environment values. Native stdout/stderr are separate logs.
    raw = (json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with path.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def execute(args, inventory, selection, environment, config, record):
    binary = selection["binary"]["path"]
    if config is not None:
        config_path = args.output / "selected.cfg"
        with config_path.open("xb") as stream:
            stream.write(config)
            stream.flush()
            os.fsync(stream.fileno())
        record["effective_config"] = inventory.capture(config_path)
    else:
        record["effective_config"] = selection["original_config"]
    if args.mode == "prepare":
        record["status"] = "PREPARED"
        return
    command = [binary, "gtp"] if args.mode == "gtp" else [binary, "nnbench", "--mode", "eval"]
    command += ["--model", selection["model"]["path"], "--config", record["effective_config"]["path"]]
    if args.mode == "eval":
        command += ["--batch", str(args.batch_cap), "--workers", str(args.capacity),
                    "--iterations", str(args.iterations), "--warmup", str(args.warmup)]
    inventory.recheck()
    record.update(command=command, status="STARTING")
    save(args.output / "receipt.json", record)
    process = None
    streams = []
    started = time.monotonic()
    timeout = args.timeout_seconds if args.timeout_seconds is not None else (300 if args.mode == "eval" else None)
    deadline = started + timeout if timeout is not None else None
    record["timeout_seconds"] = timeout
    try:
        options = dict(env=environment, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if args.mode == "eval":
            for name in ("stdout", "stderr"):
                streams.append((args.output / (name + ".log")).open("xb"))
            options.update(stdin=subprocess.DEVNULL, stdout=streams[0], stderr=streams[1])
        require(deadline is None or time.monotonic() < deadline, "deadline expired before native start")
        process = subprocess.Popen(command, **options)
        record.update(pid=process.pid, status="RUNNING")
        save(args.output / "receipt.json", record)
        remaining = None if deadline is None else max(0, deadline - time.monotonic())
        record["actual_exit"] = process.wait(timeout=remaining)
        require(record["actual_exit"] == 0, "native process returned nonzero")
        require(deadline is None or time.monotonic() <= deadline, "native wait exceeded deadline")
    finally:
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except Exception as error:
                record["cleanup_errors"].append("kill: " + type(error).__name__)
            try:
                record["cleanup_exit"] = process.wait(timeout=10)
            except Exception as error:
                record["cleanup_errors"].append("wait: " + type(error).__name__)
        for stream in streams:
            try:
                stream.close()
            except Exception as error:
                record["cleanup_errors"].append("log close: " + type(error).__name__)
        record["process_wall_seconds"] = time.monotonic() - started
    require(not record["cleanup_errors"], "native cleanup failed")
    if args.mode == "eval" and selection["selection"] == "MEASURED_PROFILE":
        variant = dict(model=selection["model"], batch_cap=args.batch_cap, backend=selection["backend"])
        metrics = load_parser(inventory, selection["runner"])(
            (args.output / "stdout.log").read_text(encoding="utf-8"), variant,
            args.iterations, args.warmup, args.capacity)
        require(metrics["inference_profile"] == selection["expected_profile"], "runtime profile differs from measured profile")
        record["metrics"] = metrics
    require(deadline is None or time.monotonic() <= deadline, "native result processing exceeded deadline")
    record["status"] = "COMPLETED"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--profiles", required=True, type=Path)
    result.add_argument("--profiles-sha256", required=True)
    result.add_argument("--model", required=True, type=Path)
    result.add_argument("--batch-cap", required=True, type=int)
    result.add_argument("--capacity", type=int, default=32)
    result.add_argument("--gpu-index", type=int, default=0)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--mode", choices=("prepare", "eval", "gtp"), default="prepare")
    result.add_argument("--iterations", type=int, default=1024)
    result.add_argument("--warmup", type=int, default=128)
    result.add_argument("--timeout-seconds", type=float)
    result.add_argument("--fallback-binary", type=Path)
    result.add_argument("--fallback-config", type=Path)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    require(0 < args.batch_cap <= 1024 and 0 < args.capacity <= 4096 and args.gpu_index >= 0, "invalid load or device")
    require(1 <= args.iterations <= 20000 and 1 <= args.warmup <= 1000, "invalid iteration count")
    require(args.timeout_seconds is None or 0 < args.timeout_seconds <= 86400, "invalid process timeout")
    require((args.fallback_binary is None) == (args.fallback_config is None), "fallback binary and config must be supplied together")
    args.output = args.output.resolve()
    args.output.mkdir(exist_ok=False)
    inventory = Inventory()
    record = dict(schema="rustgo-local-quant-inference-receipt-v1", mode=args.mode, status="VALIDATING",
                  accuracy=ACCURACY, batch_cap=args.batch_cap, capacity=args.capacity, gpu_index=args.gpu_index,
                  native_started=False, actual_exit=None, cleanup_exit=None, cleanup_errors=[],
                  gtp_search_speed="NOT_MEASURED", shared_worker_server="NOT_SUPPORTED", no_retries=True)
    exit_code = 1
    try:
        record["launcher"] = inventory.capture(__file__)
        selection, environment, config = select(args, inventory, gpu_query=query_gpu)
        record["selection"] = selection
        if selection["selection"] == "NO_MATCH":
            record.update(status="NO_MATCH", action="Use the original invocation or supply both explicit fallback files.")
            exit_code = 2
        else:
            execute(args, inventory, selection, environment, config, record)
            exit_code = 0
    except BaseException as error:
        record.update(status="FAILED_NO_RETRY", error=type(error).__name__ + ": " + str(error))
    finally:
        record["native_started"] = "pid" in record
        record["sources_before"] = list(inventory.records.values())
        try:
            inventory.recheck()
            record["sources_unchanged"] = True
        except BaseException as error:
            record.update(status="FAILED_NO_RETRY", sources_unchanged=False,
                          source_recheck_error=type(error).__name__ + ": " + str(error))
            exit_code = 1
        save(args.output / "receipt.json", record)
    print(record["status"] + ": " + str(args.output / "receipt.json"), file=sys.stderr)
    if record["status"] == "NO_MATCH":
        print(record["action"], file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
