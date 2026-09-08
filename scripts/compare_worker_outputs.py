#!/usr/bin/env python3
"""Compare C++ and Rust real NN workers with identical semantic protocol inputs.

Default coverage: 16 fixed histories x all 8 symmetries, ownership included and
cache skipped, using an explicitly FP32 C++ CUDA reference. Each request completes
before the next is submitted by default; --request-window tests concurrent work.
C++ finishes and drains before Rust starts, so GPU inference never overlaps.
These are postprocessed white-perspective outputs, not raw network logits.

C++ deployment FP16 should be measured separately with --cpp-config. Its own
FP16/FP32 differences can exceed the fixed gates; do not widen them for a pass.

Requires grpcio/grpcio-tools. --reference-only saves a reusable C++ baseline;
--reuse-reference skips C++ after checking model/schema/request fingerprints.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from google.protobuf.json_format import MessageToDict
from worker_protocol_tools import Protocol, WorkerHarness

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6"
SCALAR_GATES = {
    "white_win_prob": (0.0025, 0.0),
    "white_loss_prob": (0.0025, 0.0),
    "white_no_result_prob": (0.0025, 0.0),
    "white_score_mean": (0.1, 0.002),
    "white_score_mean_sq": (0.5, 0.003),
    "white_lead": (0.1, 0.002),
    "var_time_left": (0.5, 0.003),
    "shortterm_winloss_error": (0.005, 0.003),
    "shortterm_score_error": (0.05, 0.003),
}
ARRAY_GATES = {"policy": (0.005, 0.0), "ownership": (0.005, 0.0)}


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def message_dict(message):
    return MessageToDict(message, preserving_proto_field_name=True,
                         always_print_fields_with_no_presence=True)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def build_requests(pb, fixture, model_hash, limit, policy_optimism=None, policy_temperature=None):
    requests = []
    for position in fixture["positions"]:
        for symmetry in range(8):
            moves = position["moves"]
            parameters = dict(symmetry=symmetry, policy_temperature=1.0, policy_optimism=0.0,
                              draw_equivalent_wins_for_white=0.5, max_history=1000,
                              include_ownership=True, skip_cache=True)
            parameters.update(position.get("parameters", {}))
            if policy_optimism is not None:
                parameters["policy_optimism"] = policy_optimism
            if policy_temperature is not None:
                parameters["policy_temperature"] = policy_temperature
            request = pb.EvalRequest(
                task_id=len(requests) + 1, generation=1, session_id="same-model-parity",
                model_sha256=model_hash, lease_ms=120_000,
                position=pb.Position(board_size=19, komi=7.5, rules="chinese",
                                     initial_player=pb.BLACK,
                                     next_player=pb.BLACK if len(moves) % 2 == 0 else pb.WHITE,
                                     moves=[pb.Move(color=color, vertex=vertex) for color, vertex in moves]),
                parameters=pb.EvalParameters(**parameters))
            request.input_hash = hashlib.sha256(request.SerializeToString(deterministic=True)).digest()
            requests.append((f"{position['name']}-sym{symmetry}", request))
    return requests[:limit] if limit else requests


def validate_result(result, request, hello):
    for field in ("task_id", "generation", "session_id", "input_hash", "model_sha256"):
        if getattr(result, field) != getattr(request, field):
            raise AssertionError(("result identity mismatch", field))
    if result.error_code or not result.HasField("output"):
        raise AssertionError(("evaluation failed", result.error_code, result.error_message))
    output = result.output
    if len(output.policy) != 362 or len(output.ownership) != 361:
        raise AssertionError(("shape mismatch", len(output.policy), len(output.ownership)))
    if output.has_shortterm_error != hello.supports_shortterm_error:
        raise AssertionError("shortterm output disagrees with Hello")
    for field in SCALAR_GATES:
        if not math.isfinite(getattr(output, field)):
            raise AssertionError(("non-finite output", field))
    if not all(math.isfinite(value) for value in output.policy):
        raise AssertionError("non-finite policy")
    if abs(sum(max(value, 0) for value in output.policy) - 1.0) > 1e-4:
        raise AssertionError("policy is not normalized")
    if abs(output.white_win_prob + output.white_loss_prob + output.white_no_result_prob - 1.0) > 1e-5:
        raise AssertionError("value probabilities are not normalized")
    if not all(math.isfinite(value) and -1.0001 <= value <= 1.0001 for value in output.ownership):
        raise AssertionError("invalid ownership")
    if not all(result.HasField(field) for field in ("queue_us", "context_us", "evaluator_us")):
        raise AssertionError("missing evaluation stage timings")


def collect_worker(label, binary, config, args, protocol, requests, fingerprints):
    directory = args.output / label
    directory.mkdir(exist_ok=True)
    harness = WorkerHarness(protocol)
    flag = "-" if label == "cpp" else "--"
    command = [str(binary.resolve()), "nnworker", f"{flag}server", f"127.0.0.1:{harness.port}",
               f"{flag}worker-id", f"parity-{label}", f"{flag}capacity", str(args.request_window), f"{flag}once",
               f"{flag}model", str(args.model.resolve()), f"{flag}config", str(config.resolve()),
               f"{flag}model-sha256", fingerprints["model_sha256"]]
    report = dict(fingerprints, worker=label, command=command, binary_sha256=file_hash(binary),
                  config_sha256=file_hash(config), config_text=config.read_text(encoding="utf-8"),
                  physical_request_window=args.request_window, results=[], status="RUNNING")
    process = None
    started = time.monotonic()
    try:
        with (directory / "worker.log").open("wb") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            peer = harness.accept(process, args.startup_timeout)
            hello = peer.hello
            if (hello.model_sha256 != fingerprints["model_sha256"] or hello.protocol_version != 1
                    or hello.input_profile != "katago-eval-v1" or hello.max_in_flight != args.request_window
                    or not hello.supports_ownership or not hello.supports_friendly_pass_search):
                raise AssertionError(("incompatible Hello", hello))
            report["hello"] = message_dict(hello)
            completed = {}
            for start in range(0, len(requests), args.request_window):
                window = requests[start:start + args.request_window]
                pending = {request.task_id: (name, request) for name, request in window}
                for name, request in window:
                    peer.outgoing.put(protocol.pb.ServerMessage(evaluate=request))
                deadline = time.monotonic() + args.task_timeout
                while pending:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"request window timed out: {sorted(pending)}")
                    message = peer.receive(max(0.001, deadline - time.monotonic()))
                    if message.HasField("heartbeat"):
                        continue
                    if not message.HasField("result") or message.result.task_id not in pending:
                        raise AssertionError("unexpected, duplicate, or stale result in request window")
                    result = message.result
                    name, request = pending.pop(result.task_id)
                    validate_result(result, request, hello)
                    (directory / f"{request.task_id:03d}.pb").write_bytes(result.SerializeToString(deterministic=True))
                    completed[result.task_id] = dict(name=name, result=message_dict(result))
                    # Stable request order permits comparison across completion
                    # orders and windows while retaining every exact task ID.
                    report["results"] = [completed[request.task_id] for _, request in requests
                                         if request.task_id in completed]
                    if len(completed) % 8 == 0 or len(completed) == len(requests):
                        print(f"{label}: {len(completed)}/{len(requests)} {name}", flush=True)
                        write_json(directory / "outputs.json", report)
            heartbeat = peer.settled(len(requests))
            if heartbeat.nn_rows < len(requests) or heartbeat.nn_batches == 0:
                raise AssertionError(("missing real NN statistics", heartbeat))
            report["final_heartbeat"] = message_dict(heartbeat)
            report["heartbeats_seen"] = len(peer.heartbeats)
            peer.outgoing.put(protocol.pb.ServerMessage(drain=protocol.pb.Drain(reason="numeric baseline complete")))
            if process.wait(timeout=30) != 0:
                raise AssertionError(f"{label} did not exit cleanly after Drain")
            report.update(status="PASS", drained=True, elapsed_seconds=time.monotonic() - started)
            print(f"{label}: all outputs saved; Drain completed", flush=True)
    except Exception as error:
        report.update(status="FAIL", error=str(error), elapsed_seconds=time.monotonic() - started)
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
        write_json(directory / "outputs.json", report)
    return report


def compare_outputs(reference, actual):
    metrics = {field: dict(max_abs=0.0, sum_squared=0.0, count=0, violations=0, worst_case=None)
               for field in (*SCALAR_GATES, *ARRAY_GATES)}
    issues = []
    top1_matches = 0
    top1_near_ties = 0
    for ref_case, rust_case in zip(reference["results"], actual["results"], strict=True):
        name = ref_case["name"]
        if name != rust_case["name"]:
            raise AssertionError("reference request order changed")
        ref = ref_case["result"]["output"]
        got = rust_case["result"]["output"]
        if ref["has_shortterm_error"] != got["has_shortterm_error"]:
            issues.append(dict(case=name, field="has_shortterm_error"))
        for field, (absolute, relative) in {**SCALAR_GATES, **ARRAY_GATES}.items():
            pairs = zip(ref[field], got[field], strict=True) if field in ARRAY_GATES else [(ref[field], got[field])]
            metric = metrics[field]
            for index, (expected, value) in enumerate(pairs):
                # Negative policy entries encode legality, not probabilities.
                if field == "policy" and (expected < 0 or value < 0):
                    if (expected < 0) != (value < 0):
                        metric["violations"] += 1
                        issues.append(dict(case=name, field="policy_legality", index=index,
                                           reference=expected, rust=value))
                    continue
                difference = abs(expected - value)
                metric["count"] += 1
                metric["sum_squared"] += difference * difference
                if difference > metric["max_abs"]:
                    metric.update(max_abs=difference, worst_case=name, worst_index=index,
                                  reference=expected, rust=value)
                if difference > absolute + relative * abs(expected):
                    metric["violations"] += 1
            if metric["violations"] and not any(item.get("field") == field for item in issues):
                issues.append(dict(case=name, field=field))
        ref_top = max(range(362), key=lambda i: ref["policy"][i])
        got_top = max(range(362), key=lambda i: got["policy"][i])
        top1_matches += ref_top == got_top
        if ref_top != got_top:
            ordered = sorted(ref["policy"], reverse=True)
            near_tie = ordered[0] - ordered[1] < 2 * ARRAY_GATES["policy"][0]
            top1_near_ties += near_tie
            issues.append(dict(case=name, field="policy_top1", reference=ref_top, rust=got_top,
                               reference_margin=ordered[0] - ordered[1], near_tie=near_tie))
    for field, metric in metrics.items():
        metric["rmse"] = math.sqrt(metric.pop("sum_squared") / max(metric["count"], 1))
        metric["absolute_tolerance"], metric["relative_tolerance"] = {**SCALAR_GATES, **ARRAY_GATES}[field]
    return dict(result="PASS" if not issues else "FAIL", fields=metrics, issues=issues,
                policy_top1_matches=top1_matches, policy_top1_total=len(reference["results"]),
                policy_top1_mismatches_near_ties=top1_near_ties,
                scope="Identical semantic request bytes and model bytes; postprocessed outputs in white perspective. "
                      "Feature construction is included end-to-end; raw tensors are not independently compared.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-sha256", default=MODEL_SHA256, help="Expected hash of the shared TF3 model bytes")
    parser.add_argument("--cpp-worker", type=Path, default=Path("D:/Go/Server/worker/build-windows-CUDA/Release/katago.exe"))
    parser.add_argument("--cpp-config", type=Path, default=ROOT / "configs/worker_cpp_fp32.cfg")
    parser.add_argument("--rust-worker", type=Path, default=ROOT / "target/release/katago-rs.exe")
    parser.add_argument("--rust-config", type=Path, default=ROOT / "configs/worker_cuda.cfg")
    parser.add_argument("--proto", type=Path, default=ROOT / "crates/kata_worker/proto/worker.proto")
    parser.add_argument("--fixtures", type=Path, default=ROOT / "scripts/fixtures/worker_positions.json")
    parser.add_argument("--output", type=Path, default=ROOT / "target/worker-parity-tf3")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--reuse-reference", type=Path, help="Previously collected cpp/outputs.json")
    parser.add_argument("--case-limit", type=int, default=0, help="Diagnostic subset; zero runs all 128 cases")
    parser.add_argument("--case-name", action="append", help="Exact case name for targeted diagnostics; repeatable")
    parser.add_argument("--request-window", type=int, default=1, help="Outstanding requests/advertised capacity, 1..32")
    parser.add_argument("--policy-optimism", type=float, default=None,
                        help="Override every request's optimism (0..1) for separate numerical acceptance; default baseline unchanged")
    parser.add_argument("--policy-temperature", type=float, default=None,
                        help="Override every request's policy temperature (0.001..100) for separate numerical acceptance; default baseline unchanged")
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--task-timeout", type=float, default=120)
    args = parser.parse_args()
    if args.reference_only and args.reuse_reference:
        parser.error("--reference-only and --reuse-reference are mutually exclusive")
    if not 0 <= args.case_limit <= 128:
        parser.error("--case-limit must be 0..128")
    if args.case_name and args.case_limit:
        parser.error("use either --case-name or --case-limit")
    if not 1 <= args.request_window <= 32:
        parser.error("--request-window must be 1..32")
    if args.policy_optimism is not None and not 0 <= args.policy_optimism <= 1:
        parser.error("--policy-optimism must be finite and in 0..1")
    if args.policy_temperature is not None and not 0.001 <= args.policy_temperature <= 100:
        parser.error("--policy-temperature must be finite and in 0.001..100")
    args.output.mkdir(parents=True, exist_ok=True)
    model_hash = file_hash(args.model)
    if model_hash != args.model_sha256.lower():
        raise AssertionError(f"model SHA256 mismatch: {model_hash}")
    protocol = Protocol(args.proto)
    try:
        fixture = json.loads(args.fixtures.read_text(encoding="utf-8"))
        requests = build_requests(protocol.pb, fixture, model_hash, args.case_limit,
                                  args.policy_optimism, args.policy_temperature)
        if args.case_name:
            selected = set(args.case_name)
            missing = selected - {name for name, _ in requests}
            if missing:
                parser.error(f"unknown case names: {sorted(missing)}")
            requests = [(name, request) for name, request in requests if name in selected]
        digest = hashlib.sha256()
        request_directory = args.output / "requests"
        request_directory.mkdir(exist_ok=True)
        for name, request in requests:
            raw = request.SerializeToString(deterministic=True)
            digest.update(len(raw).to_bytes(8, "little") + raw)
            (request_directory / f"{request.task_id:03d}.pb").write_bytes(raw)
        fingerprints = dict(model_sha256=model_hash, schema_sha256=file_hash(args.proto),
                            fixtures_sha256=file_hash(args.fixtures), requests_sha256=digest.hexdigest(),
                            cases=len(requests))
        write_json(args.output / "manifest.json", dict(fingerprints, fixture=fixture,
                   physical_request_window=args.request_window,
                   tolerance_note="Fixed before measurement. |Rust-C++| <= absolute + relative*|C++|; "
                                  "policy top-1 requires exact agreement, including reported near ties.",
                   tolerances={**SCALAR_GATES, **ARRAY_GATES}))
        if args.reuse_reference:
            reference = json.loads(args.reuse_reference.read_text(encoding="utf-8"))
            if reference.get("status") != "PASS":
                raise AssertionError("reference collection did not pass")
            for name, value in fingerprints.items():
                if reference.get(name) != value:
                    raise AssertionError(f"reference fingerprint mismatch: {name}")
        else:
            reference = collect_worker("cpp", args.cpp_worker, args.cpp_config, args, protocol, requests, fingerprints)
        if args.reference_only:
            print(f"REFERENCE READY: {args.output / 'cpp/outputs.json'}", flush=True)
            return
        actual = collect_worker("rust", args.rust_worker, args.rust_config, args, protocol, requests, fingerprints)
        if reference["hello"]["model_version"] != actual["hello"]["model_version"]:
            raise AssertionError("workers disagree on the loaded model version")
        report = dict(fingerprints, **compare_outputs(reference, actual),
                      reference_request_window=reference.get("physical_request_window", 1),
                      rust_request_window=args.request_window,
                      reference_backend=reference["hello"]["backend_info"],
                      reference_config=reference["config_text"])
        write_json(args.output / "comparison.json", report)
        for field, metric in report["fields"].items():
            print(f"{field:28s} max_abs={metric['max_abs']:.6g} rmse={metric['rmse']:.6g} "
                  f"violations={metric['violations']}")
        print(f"policy top-1: {report['policy_top1_matches']}/{report['policy_top1_total']}")
        print(f"RESULT: {report['result']}; {args.output / 'comparison.json'}")
        if report["result"] != "PASS":
            raise SystemExit(1)
    finally:
        protocol.close()


if __name__ == "__main__":
    main()
