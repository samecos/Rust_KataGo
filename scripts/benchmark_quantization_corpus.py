#!/usr/bin/env python3
"""Measure complete, numerically checked corpus requests with sequential ABBA.

Consumes frozen collections, revalidates their exact protobuf outputs and
recomputes both arms' error against the explicit same-model FP16 reference.
Uses the existing continuously replenished RPC timing loop. All source requests
are prepared before timing; gRPC serialization, network, replay, evaluator and
response decoding remain inside the measured RPC boundary. Disk/JSON writing,
startup, Hello, warmup, settled heartbeats and Drain are outside it.

This pairwise measurement never publishes a recipe or declares its baseline the
best deployment. NOT_READY/non-selection input requires --smoke and produces
diagnostic results only. A final selection pipeline must additionally bind the
best validated deployment baselines, unopened holdout and reload evidence.
"""

import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from benchmark_workers import PreparedRequest, drive_window, numeric_heartbeat, read_process_cpu
from collect_quantization_corpus import ROOT, IMPORTED_CONFIG_KEYS
from compare_quantization_corpus import load_collection, compare
from compare_worker_outputs import file_hash, message_dict
from tune_runtime import clean_environment, decide
from validate_quantized_backend import PROFILE_LINE, save_json, strict_json


ORDER = ("baseline", "candidate", "candidate", "baseline")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_numeric_collection_modes(modes, smoke):
    require(set(modes) == {"reference", "baseline", "candidate"}, "all three numerical collection modes are required")
    require(all(mode in ("chunked", "continuous") for mode in modes.values()), "unsupported numerical collection window_mode")
    require(smoke or all(mode == "continuous" for mode in modes.values()),
            "formal ABBA requires continuous numerical reference/baseline/candidate collections; old chunked evidence is --smoke only")


def prepare_requests(protocol, reference, warmup_count, cycles, timeout):
    ordered = sorted(reference["positions"].values(), key=lambda p: p["request"].task_id)
    template_count = len(ordered)
    prepared, digest = [], hashlib.sha256()
    for index in range(warmup_count + template_count * cycles):
        # Begin the timed sequence at the first original request, regardless of
        # the warmup length; preserve every Position/EvalParameters byte.
        position_index = index if index < warmup_count else index - warmup_count
        source = ordered[position_index % template_count]["request"]
        request = protocol.pb.EvalRequest()
        request.CopyFrom(source)
        require(request.parameters.skip_cache, "every timed request must skip cache")
        request.task_id = index + 1
        request.session_id = "quant-corpus-abba-v1"
        request.generation = 1
        request.lease_ms = math.ceil(timeout * 1000)
        request.ClearField("input_hash")
        request.input_hash = hashlib.sha256(request.SerializeToString(deterministic=True)).digest()
        message = protocol.pb.ServerMessage(evaluate=request)
        raw = message.SerializeToString(deterministic=True)
        digest.update(len(raw).to_bytes(8, "little") + raw)
        prepared.append(PreparedRequest(message, (
            request.task_id, request.generation, request.session_id,
            request.input_hash, request.model_sha256,
        )))
    return prepared[:warmup_count], prepared[warmup_count:], digest.hexdigest()


def arm_environment(report):
    environment = clean_environment()
    declared = report.get("controlled_environment", {})
    for key, value in declared.items():
        require(isinstance(key, str) and key.startswith("KATAGO_") and isinstance(value, str),
                "invalid recorded environment")
        if key == "KATAGO_CUDA_BATCH_TRACE":
            continue  # diagnostic only, deliberately excluded from profile identity
        require(not any(s in key for s in ("DEBUG", "DUMP", "PROFILE")),
                f"profiling/debug collection cannot supply timing environment: {key}")
        require(not (key == "KATAGO_CUDA_INT8_GEMM_TUNE" and value != "0"),
                "unpersisted runtime INT8 algorithm tuning is not a frozen plan")
        require(not (key == "KATAGO_CUDA_CUBLASLT_RANK" and value == "time"),
                "unpersisted runtime algorithm timing is not a frozen plan")
        environment[key] = value
    return environment


def runtime_files(collection, binary_override):
    directory, data = collection
    report = data["report"]
    binary = (binary_override or Path(report["binary"])).resolve()
    model = Path(report["model"]).resolve()
    files = [(binary, report["binary_sha256"]), (model, report["model_sha256"]),
             (directory / "worker.cfg", report["config_sha256"])]
    # Validate the files actually named by the config, not merely copied files
    # listed by a manifest. Moving a collection must not leave a legacy config
    # pointing to an unchecked old absolute tactic-plan path.
    values = {}
    allowed = IMPORTED_CONFIG_KEYS | {"cudaQuantPlan", "cudaQuantExpectedProfile"}
    for line in (directory / "worker.cfg").read_text(encoding="utf-8").splitlines():
        body = line.partition("#")[0].strip()
        if not body:
            continue
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_-]*)\s*=\s*(.+)", body)
        require(match is not None and match[1] in allowed and match[1] not in values,
                "unsupported, indexed, duplicate or indirect timing config")
        key, value = match[1], match[2].strip()
        if value.startswith('"'):
            require(value.endswith('"') and not any(c in value[1:-1] for c in '\\"'), "ambiguous config quoting")
            value = value[1:-1]
        else:
            require('"' not in value and "'" not in value, "ambiguous config value")
        values[key] = value
    require(values.get("nnBackend") == report["backend"] and values.get("nnMaxBatchSize") == str(report["batch"]),
            "timing config backend/batch differs from numerical collection")
    artifacts = {}
    for artifact in report.get("runtime_artifacts", []):
        key = artifact["config_key"]
        require(key in ("cudaTacticPlan", "cudaQuantPlan") and key not in artifacts, "unknown/duplicate runtime artifact")
        path = Path(artifact.get("effective_path", artifact.get("file", artifact.get("path", ""))))
        require(str(path) not in ("", "."), "runtime artifact has no effective path")
        if not path.is_absolute():
            path = directory / path
        artifacts[key] = (path.resolve(), artifact["sha256"])
    if report.get("recipe_file_sha256"):
        fallback = ((directory / "recipe.json").resolve(), report["recipe_file_sha256"])
        require(artifacts.get("cudaQuantPlan", fallback) == fallback, "recipe artifact identity differs")
        artifacts["cudaQuantPlan"] = fallback  # supports older generated collections
    configured = {key for key in ("cudaTacticPlan", "cudaQuantPlan") if key in values}
    require(configured == set(artifacts), "config execution files and frozen runtime artifacts differ")
    for key in configured:
        actual = Path(values[key])
        actual = (actual if actual.is_absolute() else ROOT / actual).resolve()
        bound, expected = artifacts[key]
        require(actual == bound, f"{key} config still points outside its frozen artifact: {actual}")
        files.append((actual, expected))
    if "legacy_execution" in report:
        from legacy_quantization_execution import validate_collection
        observed = validate_collection(directory, report)
        # Revalidate nested probe evidence before every arm and after timing,
        # in addition to the effective model/config/plan files above.
        for artifact in observed["artifact_paths"]:
            path = (directory / artifact).resolve()
            if all(path != existing.resolve() for existing, _ in files):
                files.append((path, file_hash(path)))
    return binary, model, files


def check_files(files):
    for path, expected in files:
        require(path.is_file() and file_hash(path) == expected, f"runtime file changed: {path}")


def check_hello(hello, report, capacity):
    require(hello.protocol_version == 1 and hello.input_profile == "katago-eval-v1"
            and hello.model_sha256 == report["model_sha256"] and hello.max_in_flight == capacity
            and hello.model_version == report["hello"]["model_version"] and hello.max_board_size == 19
            and hello.supports_ownership and hello.supports_friendly_pass_search,
            "timed Worker Hello differs from the numerically checked contract")
    require(re.search(rf"(?:^|;\s*)backend={re.escape(report['backend'])}(?:;|$)", hello.backend_info),
            "timed Worker backend mismatch")
    if report.get("actual_profile"):
        require(re.findall(r"(?:^|;\s*)inference-profile=([^;\s]+)", hello.backend_info)
                == [report["actual_profile"]["inference_profile_id"]],
                "timed Worker precision/execution profile differs from numerical collection")


def run_arm(flavor, index, collection, binary_override, args, protocol, warmup, measured):
    from worker_protocol_tools import WorkerHarness
    from legacy_quantization_execution import performance_clock

    directory = args.output / f"{index + 1:02d}-{flavor}"
    directory.mkdir(exist_ok=False)
    source, data = collection
    prior = data["report"]
    binary, model, files = runtime_files(collection, binary_override)
    check_files(files)
    environment = arm_environment(prior)
    harness = WorkerHarness(protocol)
    command = [str(binary), "nnworker", "--server", f"127.0.0.1:{harness.port}",
               "--worker-id", f"quant-corpus-abba-{index}", "--capacity", str(args.capacity),
               "--once", "--model", str(model), "--model-sha256", prior["model_sha256"],
               "--config", str(source / "worker.cfg")]
    report = dict(status="RUNNING", arm=flavor, order_index=index, command=command, timing_clock=performance_clock(),
                  collection_report_sha256=data["report_sha256"],
                  files=[dict(path=str(p), sha256=h) for p, h in files],
                  environment={k: v for k, v in environment.items() if k.startswith("KATAGO_")},
                  batch_capacity=prior["batch"], protocol_capacity=args.capacity,
                  concurrency=args.concurrency, warmup_requests=len(warmup), measured_requests=len(measured))
    process = None
    try:
        loaded_legacy = None
        if "legacy_execution" in prior:
            import legacy_quantization_execution as legacy_execution
            loaded_legacy = legacy_execution.load_spec(
                source / "legacy-execution-spec.json", binary=binary, model=model,
                backend=prior["backend"], batch=prior["batch"])
            report["legacy_execution"] = dict(before=legacy_execution.probe_fingerprint(
                loaded_legacy, directory, "before", environment=environment,
                timeout=args.startup_timeout))
        with (directory / "worker.log").open("xb") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            peer = harness.accept(process, args.startup_timeout)
            check_hello(peer.hello, prior, args.capacity)
            report["hello"] = message_dict(peer.hello)
            drive_window(peer, warmup, args.concurrency, args.task_timeout, record=False)
            before = numeric_heartbeat(peer.settled(len(warmup), args.task_timeout))
            cpu_before = read_process_cpu(process)
            measurement = drive_window(peer, measured, args.concurrency, args.task_timeout, record=True)
            cpu_after = read_process_cpu(process)
            after = numeric_heartbeat(peer.settled(len(warmup) + len(measured), args.task_timeout))
            delta = {k: after[k] - before[k] for k in after if k != "in_flight"}
            count = len(measured)
            require(before["in_flight"] == after["in_flight"] == 0
                    and delta["failed_requests"] == 0 and delta["completed_requests"] == delta["nn_rows"] == count
                    and 1 <= delta["nn_batches"] <= count,
                    f"timed work is not exactly complete uncached NN inference: {delta}")
            average = count / delta["nn_batches"]
            require(1 <= average <= prior["batch"], "logical batch average exceeds physical capacity")
            measurement.update(nn_rows_per_second=count / measurement["elapsed_seconds"],
                               nn_batches_per_second=delta["nn_batches"] / measurement["elapsed_seconds"],
                               mean_logical_rows_per_batch=average,
                               physical_batch_distribution="NOT_OBSERVED_IN_UNTRACED_TIMING",
                               worker_cpu_seconds=None if cpu_before is None or cpu_after is None else cpu_after - cpu_before)
            measurement["harness_cpu_core_equivalents"] = measurement["harness_cpu_seconds"] / measurement["elapsed_seconds"]
            report.update(before_heartbeat=before, after_heartbeat=after, counter_deltas=delta, measurement=measurement)
            peer.outgoing.put(protocol.pb.ServerMessage(drain=protocol.pb.Drain(reason="corpus ABBA arm complete")))
            require(process.wait(timeout=30) == 0, "timed Worker failed to Drain cleanly")
            report["drained"] = True
        if prior.get("actual_profile"):
            p = prior["actual_profile"]
            logged = set(PROFILE_LINE.findall((directory / "worker.log").read_text(encoding="utf-8", errors="replace")))
            require(logged == {(p["model_sha256"], p["graph_sha256"], p["recipe_sha256"], p["inference_profile_id"])},
                    "timed Worker log does not bind the validated precision/execution identity")
        check_files(files)
        report["status"] = "PASS"
        if loaded_legacy is not None:
            report["legacy_execution"]["after"] = legacy_execution.probe_fingerprint(
                loaded_legacy, directory, "after", environment=environment,
                timeout=args.startup_timeout)
            report["artifacts"] = [dict(file="worker.log", sha256=file_hash(directory / "worker.log"))]
            observed = legacy_execution.validate_arm(directory, report, source, prior)
            report["legacy_execution"]["observed_execution_id"] = observed["observed_execution_id"]
        return report
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        harness.close()
        if "measurement" in report:
            rows = report["measurement"].pop("timing_rows", [])
            save_json(directory / "timings.json", dict(columns=["round_trip_us", "worker_elapsed_us", "queue_us",
                      "context_us", "evaluator_us", "refill_delay_us"], rows=rows))
        save_json(directory / "report.json", report)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("reference", "baseline", "candidate", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--baseline-binary", type=Path, help="Relocated byte-identical numerical collection binary")
    parser.add_argument("--candidate-binary", type=Path, help="Relocated byte-identical numerical collection binary")
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--cycles", type=int, default=1, help="Complete measured corpus repetitions, no truncation")
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--metric", choices=("rpc_throughput", "p95_latency"), default="rpc_throughput")
    parser.add_argument("--smoke", action="store_true", help="Permit incomplete/non-selection corpus; never selection evidence")
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--task-timeout", type=float, default=180)
    args = parser.parse_args()
    for name in ("reference", "baseline", "candidate", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if not 1 <= args.concurrency <= min(args.capacity, 256) or not 1 <= args.capacity <= 4096:
        parser.error("concurrency must be 1..256 and <= protocol capacity 1..4096")
    if not 1 <= args.cycles <= 1000 or args.warmup < args.concurrency or (not args.smoke and args.warmup < 128):
        parser.error("cycles must be 1..1000; warmup must cover concurrency and normally be >=128")
    if any(not math.isfinite(v) or v <= 0 or v > 3600 for v in (args.startup_timeout, args.task_timeout)):
        parser.error("timeouts must be finite in (0,3600]")
    if args.output.exists():
        parser.error("output must be a new directory")
    return args


def main():
    from worker_protocol_tools import Protocol
    from legacy_quantization_execution import performance_clock

    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema="rustgo-quantization-corpus-abba-v1", status="RUNNING", order=list(ORDER), timing_clock=performance_clock(),
                  production_certified=False, publish_allowed=False, deployment_adopted=False,
                  baseline_optimality="NOT_ESTABLISHED_BY_PAIRWISE_MEASUREMENT", smoke=args.smoke,
                  primary_metric=args.metric,
                  minimum_improvement=0.01, maximum_relative_spread=0.05, runs=[],
                  timing_boundary="initial dispatch through final result receipt; includes RPC serialization/decoding, "
                                  "queue, replay and evaluator; excludes startup/warmup/settled/Drain/report IO",
                  numeric_scope="complete frozen collections revalidated before timing; timed outputs checked for "
                                "identity/error only; no claim of bitwise pairing for varying physical batches",
                  collection_timing_equivalence="Continuous collection refills before payload accounting and buffers full outputs; "
                    "the benchmark discards outputs after identity/error checks. Matching refill mode is not identical timing "
                    "and does not prove identical physical batches across runs. Chunked evidence is allowed only for smoke.",
                  memory_metrics="NOT_COLLECTED", holdout_outputs_read=0)
    protocol = None
    try:
        # Reject holdout from metadata before parsing any saved output bytes.
        modes = {}
        for name in ("reference", "baseline", "candidate"):
            directory = getattr(args, name)
            metadata = strict_json((directory / "report.json").read_bytes())
            require(metadata.get("split") in ("calibration", "selection"),
                    "only calibration/selection collections may be opened; holdout is forbidden")
            modes[name] = metadata.get("window_mode", "chunked")
        report["numeric_collection_window_modes"] = modes
        validate_numeric_collection_modes(modes, args.smoke)
        protocol = Protocol(ROOT / "crates/kata_worker/proto/worker.proto")
        reference = load_collection(args.reference, protocol)
        arms = {name: (getattr(args, name), load_collection(getattr(args, name), protocol,
                         full_reference=reference["recipe"])) for name in ("baseline", "candidate")}
        report["reference_report_sha256"] = reference["report_sha256"]
        report["numerical_checks"] = {}
        for name, (_, data) in arms.items():
            numeric = compare(reference, data)
            require(numeric["experimental_gate"]["result"] == "PASS", f"{name} failed full numerical prerequisite")
            numeric.pop("cases")
            report["numerical_checks"][name] = numeric
            r = data["report"]
            require(r["request_window"] == args.concurrency and r["capacity"] == args.capacity,
                    f"{name} timing window/capacity lacks matching numerical collection")
            qualified = r["corpus_status"] == "READY" and r["split"] == "selection"
            require(qualified or args.smoke, "NOT_READY/non-selection input requires explicit --smoke")
            require(r["split"] != "holdout", "holdout outputs must never be consumed by a selection benchmark")
            check_files(runtime_files(arms[name], getattr(args, f"{name}_binary"))[2])
        require(reference["report"]["split"] != "holdout", "holdout cannot supply selection reference")
        require(len(reference["positions"]) * args.cycles + args.warmup <= 1_000_000,
                "request budget exceeds one million")
        warmup, measured, request_sha = prepare_requests(protocol, reference, args.warmup, args.cycles, args.task_timeout)
        require(len(measured) >= (1 if args.smoke else 1024), "normal timing requires >=1024 complete-corpus requests")
        require(len(measured) + len(warmup) <= 1_000_000, "request budget exceeds one million")
        report.update(prepared_request_sha256=request_sha, semantic_positions=len(reference["positions"]),
                      cycles=args.cycles, warmup_requests=len(warmup), measured_requests=len(measured),
                      protocol_sha256=file_hash(ROOT / "crates/kata_worker/proto/worker.proto"),
                      source_scripts={name: file_hash(ROOT / "scripts" / name) for name in
                                      ("benchmark_quantization_corpus.py", "benchmark_workers.py", "compare_quantization_corpus.py",
                                       "collect_quantization_corpus.py", "worker_protocol_tools.py", "tune_runtime.py",
                                       "compare_worker_outputs.py", "validate_quantized_backend.py",
                                       "legacy_quantization_execution.py")})
        save_json(args.output / "report.json", report)
        for index, name in enumerate(ORDER):
            check_files([(ROOT / "scripts" / key, value) for key, value in report["source_scripts"].items()])
            check_files([(ROOT / "crates/kata_worker/proto/worker.proto", report["protocol_sha256"])])
            print(f"starting {index + 1}/4 {name} C={args.concurrency}", flush=True)
            run = run_arm(name, index, arms[name], getattr(args, f"{name}_binary"), args, protocol, warmup, measured)
            report["runs"].append(run)
            save_json(args.output / "report.json", report)
            print(f"{name}: {run['measurement']['rpc_requests_per_second']:.3f} RPC/s", flush=True)
        check_files([(ROOT / "scripts" / key, value) for key, value in report["source_scripts"].items()])
        check_files([(ROOT / "crates/kata_worker/proto/worker.proto", report["protocol_sha256"])])
        scores = [run["measurement"]["rpc_requests_per_second"] if args.metric == "rpc_throughput"
                  else 1.0 / run["measurement"]["timings"]["round_trip_us"]["p95"] for run in report["runs"]]
        decision = decide([scores[0], scores[3]], scores[1:3], 0.01, 0.05)
        decision["score_units"] = "requests/second" if args.metric == "rpc_throughput" else "inverse_microseconds"
        if args.metric == "p95_latency":
            decision.update(baseline_p95_geomean_us=1 / decision["baseline_geomean"],
                            candidate_p95_geomean_us=1 / decision["candidate_geomean"],
                            latency_reduction=1 - decision["baseline_geomean"] / decision["candidate_geomean"],
                            threshold_definition="baseline P95 / candidate P95 >=1.01 (speed ratio), not 1% latency reduction")
        # Statistical pairwise acceptance is not deployment adoption; expose its
        # exact scope instead of relabeling a weak/default baseline as best.
        decision["meets_pairwise_speed_and_stability_limits"] = decision.pop("accepted")
        report.update(status="SMOKE_ONLY" if args.smoke else "PAIRWISE_MEASURED_REQUIRES_SELECTION_REVIEW",
                      pairwise=decision, required_before_adoption=["best validated baseline evidence for this workload",
                      "independent confirmation ABBA", "frozen recipe/plan and unopened holdout pass",
                      "fresh-process reload and local-search verification", "memory/resource evidence"])
        return 0
    except Exception as error:
        report.update(status="FAILED", error=str(error))
        print(f"FAILED: {error}", file=sys.stderr)
        return 1
    finally:
        if protocol is not None:
            protocol.close()
        save_json(args.output / "report.json", report)


if __name__ == "__main__":
    raise SystemExit(main())
