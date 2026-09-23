#!/usr/bin/env python3
"""Validate and measure same-build INT8 GEMM tuning without certifying INT8.

Run accuracy first, then worker and/or local with --accuracy-report. Every stage
uses a new output directory. A/B changes KATAGO_CUDA_INT8_GEMM_TUNE=0/1 and,
optionally, the candidate attention tile. FFN INT8, width threshold 0, B8 and
one NN server thread are fixed. No production config or tactic plan is installed.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
from types import SimpleNamespace

import autotune_reference as reference_io
import benchmark_workers as worker
from compare_worker_outputs import build_requests, collect_worker, compare_outputs, file_hash, write_json
from tune_runtime import ROOT, clean_environment, decide, parse_benchmark
from validate_int8_backend import quantization_metrics

TOGGLE = "KATAGO_CUDA_INT8_GEMM_TUNE"
SCHEMA = "int8-gemm-tuning-r3"
WINDOWS = (1, 32)
BATCH = 8
MAX_FP16_WIN_ERROR = 0.06
BASELINE_ENVIRONMENT = {TOGGLE: "0", "KATAGO_CUDA_ATTN_TILE": "q128"}


def configuration(backend="cudaint8backend", threads=8):
    text = (f"rules=chinese\nkomi=7.5\nnnMaxBatchSize={BATCH}\n"
            "numNNServerThreadsPerModel=1\nnnCacheSizePowerOfTwo=18\n"
            f"nnMutexPoolSizePowerOfTwo=14\nnumSearchThreads={threads}\n"
            f"nnBackend={backend}\n")
    if backend == "cudaint8backend":
        text += "cudaInt8Scope=ffn\ncudaInt8MinFfnWidth=0\n"
    return text


def candidate_environment(tile, gemm_tune=1):
    values = {TOGGLE: str(1 if gemm_tune is None else gemm_tune)}
    if tile is not None:
        values["KATAGO_CUDA_ATTN_TILE"] = tile
    return values


def environment(tuned, tile=None, gemm_tune=1):
    values = candidate_environment(tile, gemm_tune) if tuned else BASELINE_ENVIRONMENT
    return {**clean_environment(), **values}


def verify_identity(args, report):
    if file_hash(args.binary) != report["binary_sha256"] or file_hash(args.model) != report["model_sha256"]:
        raise AssertionError("binary or model changed during this validation")


def verify_paths(log, tuned, tile=None, gemm_tune=1):
    if "launch=cublaslt-int8" not in log:
        raise AssertionError("missing actual INT8 GEMM execution marker")
    expected_gemm_tune = gemm_tune if tuned else 0
    if f"name=int8_gemm_tune enabled={expected_gemm_tune} " not in log:
        raise AssertionError("missing INT8 GEMM tuning path marker")
    expected_tile = tile if tuned and tile else "q128"
    pattern = (r"\[cuda-tactic\] name=attention requested=fa2 launch=fa2 effective=fa2 tile="
               + re.escape(expected_tile) + r"(?:\s|$)")
    if not re.search(pattern, log) or "fallback=q128" in log:
        raise AssertionError(f"actual attention tile is not {expected_tile}, or fallback occurred")


def verify_int8(outputs, log_path, tuned, tile=None, gemm_tune=1):
    info = outputs["hello"]["backend_info"]
    if not all(part in info for part in ("backend=cudaint8backend", "int8-scope=ffn;",
                                         "int8-min-ffn-width=0;", "quantization=w8a8-row-out-rne-v1")):
        raise AssertionError("Worker did not report the required INT8 precision identity")
    log = log_path.read_text(encoding="utf-8", errors="replace")
    verify_paths(log, tuned, tile, gemm_tune)


def output_payloads(directory, protocol, requests):
    values = []
    for _, request in requests:
        result = protocol.pb.EvalResult()
        result.ParseFromString((directory / f"{request.task_id:03d}.pb").read_bytes())
        values.append(result.output.SerializeToString(deterministic=True))
    return values


def payload_hash(values):
    digest = hashlib.sha256()
    for raw in values:
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def accuracy(args, report):
    reference_path = args.reference or reference_io.load_bundled(report["model_sha256"])
    if reference_path is None:
        raise ValueError("no matching C++ FP32 reference; provide --reference")
    protocol = worker.Protocol(ROOT / "crates/kata_worker/proto/worker.proto")
    try:
        fixture = json.loads((ROOT / "scripts/fixtures/worker_positions.json").read_text(encoding="utf-8"))
        requests = build_requests(protocol.pb, fixture, report["model_sha256"], 0)
        fingerprints = reference_io.fingerprints(requests, report["model_sha256"])
        reference = reference_io.validate_reference(reference_io.read_reference(reference_path), fingerprints, requests, protocol)
        report.update(reference=str(Path(reference_path).resolve()), reference_sha256=file_hash(reference_path),
                      fingerprints=fingerprints, results=[], equivalence=[])
        for window in WINDOWS:
            options = SimpleNamespace(output=args.output, model=args.model, request_window=window,
                                      startup_timeout=args.timeout, task_timeout=args.timeout)
            collected = {}
            payloads = {}
            for profile, backend, tuned in (("fp16", "cudabackend", False),
                                            ("off", "cudaint8backend", False),
                                            ("on", "cudaint8backend", True)):
                verify_identity(args, report)
                label = f"{profile}-w{window}"
                config = args.output / f"{label}.cfg"
                config.write_text(configuration(backend), encoding="utf-8")
                outputs = collect_worker(label, args.binary, config, options, protocol,
                                         requests, fingerprints, environment(tuned, args.candidate_attention_tile, args.candidate_gemm_tune))
                original = compare_outputs(reference, outputs)
                if profile == "fp16" and original["result"] != "PASS":
                    raise AssertionError(f"FP16 W{window} control failed original C++ FP32 gates")
                if profile != "fp16":
                    verify_int8(outputs, args.output / label / "worker.log", tuned, args.candidate_attention_tile, args.candidate_gemm_tune)
                payloads[profile] = output_payloads(args.output / label, protocol, requests)
                collected[profile] = outputs
                artifact = args.output / label / "outputs.json"
                fp16_metrics = quantization_metrics(collected["fp16"], outputs)
                fp16_win_gate = dict(result="PASS" if fp16_metrics["win_probability_max_abs"] <= MAX_FP16_WIN_ERROR else "FAIL",
                                     max_abs=fp16_metrics["win_probability_max_abs"],
                                     max_percentage_points=100 * fp16_metrics["win_probability_max_abs"],
                                     limit_abs=MAX_FP16_WIN_ERROR, limit_percentage_points=6,
                                     reference=f"fp16-w{window}",
                                     scope="maximum over the 128 tested semantic/symmetry requests; not an all-position guarantee")
                result = dict(profile=label, original_fp32_gate=original,
                              accuracy=quantization_metrics(reference, outputs),
                              fp16_accuracy=fp16_metrics, fp16_win_probability_gate=fp16_win_gate,
                              outputs=str(artifact.resolve()), outputs_sha256=file_hash(artifact),
                              output_payloads_sha256=payload_hash(payloads[profile]),
                              environment=candidate_environment(args.candidate_attention_tile, args.candidate_gemm_tune) if tuned else dict(BASELINE_ENVIRONMENT))
                report["results"].append(result)
                write_json(args.output / "report.json", report)
                print(json.dumps(dict(profile=label, original_fp32_gate=original["result"],
                                      accuracy=result["accuracy"], fp16_accuracy=fp16_metrics,
                                      fp16_win_probability_gate=fp16_win_gate), ensure_ascii=False), flush=True)
                if fp16_win_gate["result"] != "PASS":
                    raise AssertionError(f"{label}: maximum win-probability deviation from FP16 exceeds 6 percentage points")
            same = [name for (name, _), a, b in zip(requests, payloads["off"], payloads["on"], strict=True) if a == b]
            equivalence = dict(window=window, exact_matches=len(same), cases=len(requests),
                               all_exact=len(same) == len(requests),
                               mismatches=[name for name, _ in requests if name not in same],
                               outputs_comparison=compare_outputs(collected["off"], collected["on"]),
                               exact_output_gate_required=window == 1 and args.candidate_attention_tile is None,
                               interpretation=("serial B1 exact-output gate" if args.candidate_attention_tile is None else
                                   "serial B1 difference report: combined attention candidate uses the independent FP16 6pp gate") if window == 1 else
                                   "report only: asynchronous W32 batching can differ; fixed raw B8 is checked separately")
            report["equivalence"].append(equivalence)
            write_json(args.output / "report.json", report)
            if window == 1 and args.candidate_attention_tile is None and not equivalence["all_exact"]:
                raise AssertionError("serial INT8 off/on output payloads are not bitwise identical")
        report["performance_preconditions_passed"] = True
    finally:
        protocol.close()


def require_accuracy(args, report):
    if args.accuracy_report is None:
        raise ValueError("performance requires --accuracy-report from this exact binary/model")
    evidence = json.loads(args.accuracy_report.read_text(encoding="utf-8"))
    # Early r3 reports predate this option and always enabled GEMM tuning.
    measured_gemm_tune = evidence.get("candidate_gemm_tune", 1)
    if type(measured_gemm_tune) is not int or measured_gemm_tune not in (0, 1):
        raise ValueError("accuracy report has an invalid candidate GEMM tuning mode")
    if args.candidate_gemm_tune is not None and args.candidate_gemm_tune != measured_gemm_tune:
        raise ValueError("candidate GEMM tuning mode disagrees with accuracy report")
    args.candidate_gemm_tune = measured_gemm_tune
    measured_tile = evidence.get("candidate_attention_tile")
    if measured_tile not in (None, "q64", "q64-serial"):
        raise ValueError("accuracy report has an invalid candidate attention tile")
    if args.candidate_attention_tile is not None and args.candidate_attention_tile != measured_tile:
        raise ValueError("candidate attention tile disagrees with accuracy report")
    args.candidate_attention_tile = measured_tile
    expected_environment = candidate_environment(measured_tile, measured_gemm_tune)
    if evidence.get("candidate_environment") != expected_environment:
        raise ValueError("accuracy report candidate environment mismatch")
    if evidence.get("baseline_environment") != BASELINE_ENVIRONMENT:
        raise ValueError("accuracy report baseline environment mismatch")
    expected = dict(schema=SCHEMA, stage="accuracy", status="COLLECTED", batch=BATCH,
                    scope="ffn", min_ffn_width=0, binary_sha256=report["binary_sha256"],
                    model_sha256=report["model_sha256"], performance_preconditions_passed=True,
                    max_fp16_win_probability_error=MAX_FP16_WIN_ERROR)
    for key, value in expected.items():
        if evidence.get(key) != value:
            raise ValueError(f"accuracy report mismatch or incomplete: {key}")
    for window in WINDOWS:
        for profile in ("fp16", "off", "on"):
            rows = [row for row in evidence["results"] if row["profile"] == f"{profile}-w{window}"]
            if len(rows) != 1:
                raise ValueError(f"missing/duplicate accuracy profile {profile}-w{window}")
            row = rows[0]
            expected_row_environment = expected_environment if profile == "on" else BASELINE_ENVIRONMENT
            if row.get("environment") != expected_row_environment:
                raise ValueError(f"{profile}-w{window}: accuracy environment does not match this candidate")
            if file_hash(row["outputs"]) != row["outputs_sha256"]:
                raise ValueError("accuracy output artifact hash mismatch")
            if profile == "fp16" and row["original_fp32_gate"]["result"] != "PASS":
                raise ValueError("FP16 original numerical gate did not pass")
            gate = row.get("fp16_win_probability_gate", {})
            if (gate.get("result") != "PASS" or gate.get("limit_abs") != MAX_FP16_WIN_ERROR
                    or not 0 <= gate.get("max_abs", float("inf")) <= MAX_FP16_WIN_ERROR
                    or row.get("fp16_accuracy", {}).get("win_probability_max_abs") != gate["max_abs"]):
                raise ValueError(f"{profile}-w{window}: missing/failed independent 6 percentage-point FP16 win gate")
        equivalence = [row for row in evidence["equivalence"] if row["window"] == window]
        if len(equivalence) != 1 or equivalence[0]["cases"] != 128:
            raise ValueError("missing complete off/on equivalence coverage")
        if window == 1 and measured_tile is None and not equivalence[0]["all_exact"]:
            raise ValueError("serial INT8 baseline/candidate exact-output gate did not pass")
    if file_hash(evidence["reference"]) != evidence["reference_sha256"]:
        raise ValueError("reference artifact changed since accuracy collection")
    report.update(accuracy_report=str(args.accuracy_report.resolve()),
                  accuracy_report_sha256=file_hash(args.accuracy_report),
                  candidate_gemm_tune=measured_gemm_tune,
                  candidate_attention_tile=measured_tile, candidate_environment=expected_environment,
                  accuracy_results=evidence["results"], equivalence=evidence["equivalence"])


def worker_performance(args, report):
    require_accuracy(args, report)
    config = args.output / "int8.cfg"
    config.write_text(configuration(), encoding="utf-8")
    protocol = worker.Protocol(ROOT / "crates/kata_worker/proto/worker.proto")
    try:
        fixture = json.loads((ROOT / "scripts/fixtures/worker_positions.json").read_text(encoding="utf-8"))
        warmup, measured, digest, templates = worker.prepare_requests(
            protocol, fixture, report["model_sha256"], 256, args.requests, "uncached", False)
        options = SimpleNamespace(output=args.output, model=args.model, model_sha256=report["model_sha256"],
            rust_worker=args.binary, baseline_worker=args.binary, rust_config=config, baseline_config=config,
            rust_environment=candidate_environment(args.candidate_attention_tile, args.candidate_gemm_tune), baseline_environment=dict(BASELINE_ENVIRONMENT),
            workload="uncached", sample_gpu=False, gpu_index=0,
            startup_timeout=args.timeout, task_timeout=args.timeout)
        report.update(requests=args.requests, warmup_requests=256, requests_sha256=digest,
                      semantic_templates=templates, ownership=False, runs=[], comparisons=[])
        for capacity in WINDOWS:
            samples = []
            for index, flavor in enumerate(("baseline", "rust", "rust", "baseline")):
                verify_identity(args, report)
                run = worker.run_worker(flavor, index, capacity, options, protocol, warmup, measured, clean_environment())
                directory = args.output / f"c{capacity:03d}-{index + 1:02d}-{flavor}"
                verify_int8(run, directory / "worker.log", flavor == "rust", args.candidate_attention_tile, args.candidate_gemm_tune)
                measurement = run["measurement"]
                samples.append(measurement["rpc_requests_per_second"])
                report["runs"].append(dict(capacity=capacity, flavor=flavor,
                    rpc_per_second=samples[-1], nn_rows_per_second=measurement["nn_rows_per_second"],
                    average_nn_batch=measurement["average_nn_batch"],
                    round_trip_us=measurement["timings"]["round_trip_us"],
                    run_report=str((directory / "report.json").resolve())))
                write_json(args.output / "report.json", report)
                print(f"C{capacity} {flavor}: {samples[-1]:.2f} uncached RPC/s", flush=True)
            report["comparisons"].append(dict(capacity=capacity, metric="uncached RPC/s", samples=samples,
                **decide([samples[0], samples[3]], samples[1:3], 0.01, 0.05)))
            write_json(args.output / "report.json", report)
            print(json.dumps(report["comparisons"][-1]), flush=True)
    finally:
        protocol.close()


def local_performance(args, report):
    require_accuracy(args, report)
    report.update(threads=args.threads, positions=args.positions, visits=args.visits, runs=[], comparisons=[],
                  metric="actual NN rows/s", secondary_metric="MCTS visits/s")
    for threads in args.threads:
        config = args.output / f"int8-t{threads}.cfg"
        config.write_text(configuration(threads=threads), encoding="utf-8")
        samples, nn_samples = [], []
        for index, tuned in enumerate((False, True, True, False)):
            verify_identity(args, report)
            label = f"t{threads:02d}-{index + 1:02d}-{'on' if tuned else 'off'}"
            command = [str(args.binary.resolve()), "benchmark", "--model", str(args.model.resolve()),
                       "--config", str(config.resolve()), "-v", str(args.visits), "-n", str(args.positions),
                       "-t", str(threads), "--fixed-batch-size", str(BATCH)]
            run_environment = candidate_environment(args.candidate_attention_tile, args.candidate_gemm_tune) if tuned else dict(BASELINE_ENVIRONMENT)
            write_json(args.output / f"{label}.command.json", dict(command=command, environment=run_environment))
            stdout_path, stderr_path = args.output / f"{label}.stdout.log", args.output / f"{label}.stderr.log"
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                completed = subprocess.run(command, cwd=ROOT, env=environment(tuned, args.candidate_attention_tile, args.candidate_gemm_tune), text=True,
                    stdout=stdout, stderr=stderr, timeout=args.timeout,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if completed.returncode:
                raise RuntimeError(f"{label}: benchmark exited {completed.returncode}; see full logs")
            text = stdout_path.read_text(encoding="utf-8")
            logs = text + stderr_path.read_text(encoding="utf-8")
            verify_paths(logs, tuned, args.candidate_attention_tile, args.candidate_gemm_tune)
            metrics = parse_benchmark(text, threads, args.positions)
            samples.append(metrics["score"])
            nn_samples.append(metrics["nn_rows_per_second"])
            report["runs"].append(dict(threads=threads, tuned=tuned, label=label,
                                      visits_per_second=metrics["score"], **metrics))
            write_json(args.output / "report.json", report)
            print(f"{label}: {metrics['score']:.2f} visits/s; {metrics['nn_rows_per_second']:.2f} NN rows/s", flush=True)
        report["comparisons"].append(dict(threads=threads, metric="actual NN rows/s", samples=nn_samples,
            **decide([nn_samples[0], nn_samples[3]], nn_samples[1:3], 0.01, 0.05),
            visits_per_second=dict(samples=samples,
                **decide([samples[0], samples[3]], samples[1:3], 0.01, 0.05))))
        write_json(args.output / "report.json", report)
        print(json.dumps(report["comparisons"][-1]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("accuracy", "worker", "local"), required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--accuracy-report", type=Path)
    parser.add_argument("--candidate-attention-tile", choices=("q64", "q64-serial"),
                        help="optional combined candidate; performance stages inherit this from their accuracy report")
    parser.add_argument("--candidate-gemm-tune", type=int, choices=(0, 1),
                        help="candidate GEMM tuning: accuracy defaults to 1; performance inherits its accuracy report")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=4096)
    parser.add_argument("--threads", default="1,8")
    parser.add_argument("--positions", type=int, default=4)
    parser.add_argument("--visits", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.stage == "accuracy" and args.candidate_gemm_tune is None:
        args.candidate_gemm_tune = 1
    try:
        args.threads = [int(value) for value in args.threads.split(",")]
    except ValueError:
        parser.error("threads must be a comma-separated integer list")
    if (not args.threads or len(set(args.threads)) != len(args.threads) or min(args.threads) < 1
            or max(args.threads) > 64 or args.requests < 1024 or args.positions < 1
            or args.visits < 32 or args.timeout < 60):
        parser.error("invalid threads, requests, positions, visits or timeout")
    if not args.binary.is_file() or not args.model.is_file():
        parser.error("binary and model must be existing files")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema=SCHEMA, stage=args.stage, status="RUNNING", batch=BATCH, lanes=1,
                  scope="ffn", min_ffn_width=0, graph=True, quantization="w8a8-row-out-rne-v1",
                  model=str(args.model.resolve()), model_sha256=file_hash(args.model),
                  binary=str(args.binary.resolve()), binary_sha256=file_hash(args.binary),
                  production_certified=False, playing_strength="NOT_MEASURED",
                  max_fp16_win_probability_error=MAX_FP16_WIN_ERROR,
                  candidate_attention_tile=args.candidate_attention_tile,
                  candidate_gemm_tune=args.candidate_gemm_tune,
                  candidate_environment=candidate_environment(args.candidate_attention_tile, args.candidate_gemm_tune),
                  baseline_environment=dict(BASELINE_ENVIRONMENT),
                  numerical_scope="Original FP32 gates are reported separately. User acceptance requires tested maximum win-probability deviation from FP16 <= 6 percentage points; policy and score error are reported. INT8 remains a separate lossy profile.",
                  toggle=TOGGLE, cleared_environment_keys=sorted(set(os.environ) - set(clean_environment())))
    write_json(args.output / "report.json", report)
    try:
        {"accuracy": accuracy, "worker": worker_performance, "local": local_performance}[args.stage](args, report)
        verify_identity(args, report)
        report["status"] = "COLLECTED"
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        raise
    finally:
        write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
