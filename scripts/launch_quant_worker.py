"""Prepare or run one local typed Worker from a completed rebuilt-binary ABBA.

The default prepare mode queries device metadata but starts no model. Only this
report's rebuilt candidate can qualify; old independent confirmation is never
transferred to a new binary. A run uses --once and never restarts its child.
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


COMMON_SHA256 = "1e99f8e46ebab7b4de1fe1c570bdabd043a967b32b5aca37dea71e9d9341f8e8"
COMMON_BYTES = 19680


def require(condition, message):
    if not condition:
        raise ValueError(message)


def common_module():
    path = Path(__file__).with_name("launch_quant_inference.py").resolve(strict=True)
    raw = path.read_bytes()
    require(len(raw) == COMMON_BYTES and hashlib.sha256(raw).hexdigest() == COMMON_SHA256,
            "pinned inference launcher helper changed")
    module = types.ModuleType("_bound_inference_launcher")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def bound_module(inventory, record):
    module = types.ModuleType("_bound_speed_parser")
    module.__file__ = record["path"]
    exec(compile(inventory.raw(record), record["path"], "exec"), module.__dict__)
    return module


def query_gpu(common, index):
    return common.query_gpu(index)


def read_inputs(args, common, inventory):
    plan_source = inventory.capture(args.plan)
    report_source = inventory.capture(args.report)
    require(plan_source["sha256"] == args.plan_sha256, "wrong explicit plan digest")
    require(report_source["sha256"] == args.report_sha256, "wrong explicit report digest")
    plan, report = inventory.read(plan_source), inventory.read(report_source)
    require(plan["schema"] == "rustgo-quant-speed-plan-v1" and plan["workers"] == 32
            and plan["accuracy"] == common.ACCURACY
            and plan["old_speed_certificate_applies_to_new_binary"] is False,
            "expected a speed-only rebuilt-binary plan")
    require(report["schema"] == "rustgo-quant-speed-report-v1"
            and report["status"] == "COMPLETE_SPEED_ONLY" and report["plan"] == plan_source
            and report["accuracy"] == common.ACCURACY, "report is not a complete result of the explicit plan")
    schedule = inventory.read(report["schedule"])
    require(schedule["schema"] == "rustgo-speed-abba-schedule-v1"
            and schedule["handoff"] == plan["handoff"], "wrong rebuilt schedule lineage")
    require(len(schedule["runs"]) == len(report["runs"]) == 8
            and plan["maximum_model_starts"] == 8, "expected exactly two completed four-arm groups")
    require(type(plan["inherited_native_starts"]) is int and plan["inherited_native_starts"] >= 0
            and plan["inherited_native_starts"] + 8 <= plan["cumulative_native_limit"] <= 96,
            "rebuilt schedule exceeds the declared cumulative budget")
    require(Path(plan["runner"]["path"]).resolve() == Path(__file__).with_name("benchmark_quant_speed.py").resolve(),
            "wrong speed parser source")
    runner = bound_module(inventory, plan["runner"])
    variants = {v["id"]: v for v in plan["variants"]}
    require(len(variants) == len(plan["variants"]), "duplicate variant")
    runner.validate_schedule(schedule, variants)
    for index, (scheduled, observed) in enumerate(zip(schedule["runs"], report["runs"]), 1):
        require(observed["index"] == index and all(observed[key] == scheduled[key]
                for key in ("variant", "comparison", "flavor", "stage")), "report order differs from schedule")
        require(observed["status"] == "COMPLETE_SPEED_SAMPLE" and observed["actual_exit"] == 0
                and observed["cleanup_errors"] == [], "unfinished measured arm")
    groups = {item["comparison"] for item in schedule["runs"]}
    require(len(groups) == 2 and set(report["comparisons"]) == groups, "wrong comparison groups")
    for group in groups:
        observed = [r for r in report["runs"] if r["comparison"] == group]
        require(runner.summarize(observed) == report["comparisons"][group], "reported ABBA statistics differ")
    build = inventory.read(plan["build_result"])
    require(build["schema"] == "rustgo-typed-worker-native-build-result-v1"
            and build["status"] == "BUILT_NOT_GPU_QUALIFIED" and build["actual_exit_code"] == 0
            and build["sources_unchanged"] is True and build["binary"] == plan["binary"]
            and build.get("tool_actual_exit_code") == 0 and build.get("root_terminal_before_deadline") is True
            and not build.get("error") and not build.get("deadline_exceeded"), "unverified rebuilt executable")
    # This old handoff supplies only the retained hardware descriptor and Source
    # lineage. Its old performance results are never used to qualify the new binary.
    handoff = inventory.read(plan["handoff"])
    require(handoff["schema"] == "rustgo-speed-profile-handoff-v1" and handoff["workers"] == 32
            and handoff["compute_capability"] == "12.0" and handoff["driver"] == "610.88",
            "wrong retained hardware descriptor")
    return plan, report, schedule, variants, runner, handoff, plan_source, report_source


def check_measured_group(args, common, inventory, data, candidate):
    plan, report, schedule, variants, runner, handoff, _, report_source = data
    group = candidate["group"]
    measured = [r for r in report["runs"] if r["comparison"] == group]
    ordered = [s for s in schedule["runs"] if s["comparison"] == group]
    require([r["stage"] for r in measured] == ["A1", "B1", "B2", "A2"], "wrong ABBA stages")
    profiles, gpu_ids = set(), set()
    report_directory = Path(report_source["path"]).parent
    for observed, requested in zip(measured, ordered):
        variant = variants[observed["variant"]]
        require(variant["batch_cap"] == args.batch_cap and variant["model"] == candidate["model"],
                "measured model/load differs")
        expected_command = [variant.get("binary", plan["binary"])["path"], "nnbench", "--mode", "eval",
                            "--model", variant["model"]["path"], "--config", variant["config"]["path"],
                            "--batch", str(variant["batch_cap"]), "--workers", "32",
                            "--iterations", str(requested["iterations"]), "--warmup", str(requested["warmup"])]
        require(observed["command"] == expected_command, "measured native command differs")
        require(re.fullmatch(r"[a-zA-Z0-9_-]+", variant["id"]), "unsafe measured log name")
        log = (report_directory / f"{observed['index']:03d}-{variant['id']}.stdout.log").resolve(strict=True)
        require(log.parent == report_directory, "measured log escaped report directory")
        log_source = inventory.capture(log)
        metrics = runner.parse_eval(inventory.raw(log_source).decode("utf-8"), variant,
                                    requested["iterations"], requested["warmup"], 32)
        require(metrics == observed["metrics"], "measured counters/profile do not match actual log")
        for key in ("binary", "model", "config", "recipe", "tactic_plan"):
            record = variant.get("binary", plan["binary"]) if key == "binary" else variant.get(key)
            if record:
                inventory.verify(record)
        if observed["flavor"] == "candidate":
            require(observed["variant"] == candidate["id"] and common.PROFILE.fullmatch(metrics["inference_profile"]),
                    "wrong rebuilt candidate profile")
            profiles.add(metrics["inference_profile"])
        for side in ("before", "after"):
            telemetry = observed[side]
            require(telemetry["returncode"] == 0 and telemetry["query"] == common.OLD_GPU_QUERY,
                    "missing actual measured GPU telemetry")
            rows = list(csv.reader(io.StringIO(telemetry["output"])))
            require(len(rows) == 1 and len(rows[0]) == 9, "ambiguous measured GPU telemetry")
            gpu_ids.add(tuple(value.strip() for value in rows[0][:3]))
    require(len(profiles) == len(gpu_ids) == 1, "measured profile or GPU changed")
    name, uuid, driver = gpu_ids.pop()
    require(name == handoff["device"] and driver == "610.88" and uuid.startswith("GPU-"),
            "measured GPU differs from the retained device descriptor")
    return profiles.pop(), dict(name=name, uuid=uuid, driver=driver, compute_capability="12.0")


def select(args, common, inventory):
    data = read_inputs(args, common, inventory)
    plan, report, _, variants, _, _, plan_source, report_source = data
    model = inventory.capture(args.model)
    result = dict(selection="NO_MATCH", model=model, plan=plan_source, report=report_source,
                  speed_basis="THIS_REBUILT_BINARY_ABBA_ONLY", independent_confirmation_for_this_binary=False,
                  old_confirmation_applies=False, measured_speed_applicable=False,
                  reason="model, cap, capacity, or speed-pass group is not measured")
    matches = [v for v in variants.values() if v["role"] == "candidate"
               and v["batch_cap"] == args.batch_cap
               and all(v["model"][key] == model[key] for key in ("bytes", "sha256"))]
    require(len(matches) <= 1, "ambiguous candidate selection")
    if matches and args.capacity == 32 and report["comparisons"][matches[0]["group"]]["speed_pass"] is True:
        candidate = matches[0]
        require(candidate["backend"] == "cudaquantbackend" and candidate["binary"] == plan["binary"]
                and candidate["batch_cap"] in (3, 8)
                and candidate["previous_measured_candidate"]["binary"]["sha256"] != candidate["binary"]["sha256"],
                "expected a changed-binary quantized candidate")
        profile, measured_gpu = check_measured_group(args, common, inventory, data, candidate)
        try:
            gpu = query_gpu(common, args.gpu_index)
        except (ValueError, OSError, subprocess.SubprocessError):
            gpu = None
        result.update(gpu=gpu, measured_gpu=measured_gpu, reason="GPU identity/driver was not measured")
        if gpu is not None and gpu.get("index") == args.gpu_index and all(gpu.get(k) == v for k, v in measured_gpu.items()):
            for key in ("environment", "source_nnbench", "binary"):
                inventory.verify(plan[key])
            config = common.locked_config(inventory.raw(candidate["config"]), candidate, profile)
            environment = common.candidate_environment(inventory, plan, candidate, gpu["uuid"])
            result.update(selection="MEASURED_REBUILT_PROFILE", reason="exact rebuilt measurement and current GPU",
                          measured_speed_applicable=True, group=candidate["group"], variant=candidate["id"],
                          expected_profile=profile, binary=candidate["binary"], original_config=candidate["config"],
                          recipe=candidate["recipe"], build_result=plan["build_result"], environment_source=plan["environment"],
                          katago_override_keys=sorted(common.ENV_KEYS), cuda_visible_device=gpu["uuid"],
                          speed_comparison=report["comparisons"][candidate["group"]])
            return result, environment, config
    if args.fallback_binary is not None:
        result.update(selection="EXPLICIT_LEGACY_FALLBACK", binary=inventory.capture(args.fallback_binary),
                      original_config=inventory.capture(args.fallback_config),
                      environment_policy="caller environment and original config preserved",
                      protocol_identity="not inferred; caller must supply its original legacy-compatible Worker")
    return result, None, None


def write_exclusive(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def run_owned(args, common, inventory, command, environment, receipt):
    process = None
    streams = []
    deadline = time.monotonic() + args.timeout_seconds
    started = time.monotonic()
    try:
        for name in ("stdout", "stderr"):
            streams.append((args.output / (name + ".log")).open("xb"))
        require(time.monotonic() < deadline, "worker deadline expired before start")
        process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=streams[0], stderr=streams[1],
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        receipt.update(pid=process.pid, native_started=True, status="RUNNING")
        common.save(args.output / "receipt.json", receipt)
        receipt["actual_exit"] = process.wait(timeout=max(0, deadline - time.monotonic()))
        require(receipt["actual_exit"] == 0, "Worker returned nonzero; no retry or fallback")
        require(time.monotonic() <= deadline, "Worker wait exceeded deadline")
    finally:
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except Exception as error:
                receipt["cleanup_errors"].append("kill: " + type(error).__name__)
            try:
                receipt["cleanup_exit"] = process.wait(timeout=10)
            except Exception as error:
                receipt["cleanup_errors"].append("wait: " + type(error).__name__)
        for stream in streams:
            try:
                stream.close()
            except Exception as error:
                receipt["cleanup_errors"].append("log close: " + type(error).__name__)
        receipt["process_wall_seconds"] = time.monotonic() - started
        receipt["logs"] = [inventory.capture(args.output / (name + ".log")) for name in ("stdout", "stderr")
                           if (args.output / (name + ".log")).is_file()]
    require(not receipt["cleanup_errors"], "Worker cleanup failed")
    receipt["status"] = "COMPLETED"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "plan"):
        result.add_argument("--" + name, required=True, type=Path)
        result.add_argument("--" + name + "-sha256", required=True)
    result.add_argument("--model", required=True, type=Path)
    result.add_argument("--batch-cap", required=True, type=int)
    result.add_argument("--capacity", type=int, default=32)
    result.add_argument("--gpu-index", type=int, default=0)
    result.add_argument("--server", required=True)
    result.add_argument("--worker-id", required=True)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--mode", choices=("prepare", "run"), default="prepare")
    result.add_argument("--timeout-seconds", type=float, default=300)
    result.add_argument("--fallback-binary", type=Path)
    result.add_argument("--fallback-config", type=Path)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    require(1 <= args.batch_cap <= 1024 and 1 <= args.capacity <= 4096 and args.gpu_index >= 0, "invalid load/device")
    require(0 < args.timeout_seconds <= 86400, "invalid explicit Worker timeout")
    require(0 < len(args.worker_id.encode("utf-8")) <= 128 and args.server.strip(), "server and worker ID are required")
    require((args.fallback_binary is None) == (args.fallback_config is None), "both fallback files are required")
    common = common_module()
    args.output = args.output.resolve()
    args.output.mkdir(exist_ok=False)
    inventory = common.Inventory()
    receipt = dict(schema="rustgo-local-typed-worker-receipt-v1", mode=args.mode, status="VALIDATING",
                   accuracy=common.ACCURACY, native_started=False, actual_exit=None, cleanup_exit=None,
                   cleanup_errors=[], no_retries=True, timeout_seconds=args.timeout_seconds,
                   worker_search_speed="NOT_MEASURED", typed_ack_observed_by_launcher=False)
    code = 1
    try:
        launcher = inventory.capture(__file__)
        helper = inventory.capture(common.__file__)
        require(helper["sha256"] == COMMON_SHA256 and helper["bytes"] == COMMON_BYTES, "helper Source changed")
        selected, environment, config = select(args, common, inventory)
        receipt["selection"] = selected
        if selected["selection"] == "NO_MATCH":
            receipt.update(status="NO_MATCH", action="Use the original Worker invocation or provide both explicit fallback files.")
            code = 2
        else:
            if config is not None:
                write_exclusive(args.output / "selected.cfg", config)
                effective_config = inventory.capture(args.output / "selected.cfg")
            else:
                effective_config = selected["original_config"]
            command = [selected["binary"]["path"], "nnworker", "--server", args.server,
                       "--worker-id", args.worker_id, "--model", selected["model"]["path"],
                       "--model-sha256", selected["model"]["sha256"], "--config", effective_config["path"],
                       "--capacity", str(args.capacity), "--once"]
            inventory.recheck()
            manifest = dict(schema="rustgo-local-typed-worker-manifest-v1", selection=selected,
                            launcher=launcher, helper=helper, effective_config=effective_config, command=command,
                            requested_batch_cap=args.batch_cap, capacity=args.capacity, gpu_index=args.gpu_index,
                            server=args.server, worker_id=args.worker_id, sources=list(inventory.records.values()),
                            no_retries=True, environment_values_not_serialized=True,
                            typed_ack_policy="exact loaded-profile ACK required by the typed Worker; not separately observed here",
                            speed_scope="NnEvaluator C32 measurement only; not a Worker RPC or search speed measurement",
                            accuracy=common.ACCURACY)
            write_exclusive(args.output / "manifest.json", json_bytes(manifest))
            receipt["manifest"] = inventory.capture(args.output / "manifest.json")
            receipt["status"] = "PREPARED"
            if args.mode == "run":
                inventory.recheck()
                run_owned(args, common, inventory, command, environment, receipt)
            code = 0
    except BaseException as error:
        receipt.update(status="FAILED_NO_RETRY", error=type(error).__name__ + ": " + str(error))
    finally:
        receipt["sources_before"] = list(inventory.records.values())
        try:
            inventory.recheck()
            receipt["sources_unchanged"] = True
        except BaseException as error:
            receipt.update(status="FAILED_NO_RETRY", sources_unchanged=False,
                           source_recheck_error=type(error).__name__ + ": " + str(error))
            code = 1
        common.save(args.output / "receipt.json", receipt)
    print(receipt["status"] + ": " + str(args.output / "receipt.json"), file=sys.stderr)
    if receipt["status"] == "NO_MATCH":
        print(receipt["action"], file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
