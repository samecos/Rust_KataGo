#!/usr/bin/env python3
"""ABBA real uncached Worker timings after collecting INT8 accuracy evidence.

This records a speed/accuracy tradeoff, not FP32 certification or playing
strength. Uses the existing loopback Worker harness and checks settled NN
counters so cache hits and dropped requests cannot inflate throughput.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import benchmark_workers as worker
from compare_worker_outputs import file_hash, write_json
from tune_runtime import ROOT, clean_environment, decide


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--accuracy-report", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--scope", choices=("ffn", "transformer"), default="ffn")
    ap.add_argument("--concurrency", default="1,32")
    ap.add_argument("--requests", type=int, default=4096)
    ap.add_argument("--baseline-binary", type=Path)
    ap.add_argument("--baseline-config", type=Path)
    args = ap.parse_args()
    capacities = [int(v) for v in args.concurrency.split(",")]
    if not capacities or min(capacities) < 1 or max(capacities) > 128 or args.requests < 1024:
        ap.error("concurrency must be 1..128 and requests >= 1024")
    if bool(args.baseline_binary) != bool(args.baseline_config):
        ap.error("provide both baseline-binary and baseline-config")
    accuracy = json.loads(args.accuracy_report.read_text(encoding="utf-8"))
    sha, binary_sha = file_hash(args.model), file_hash(args.binary)
    if accuracy["status"] != "COLLECTED" or accuracy["model_sha256"] != sha or accuracy["binary_sha256"] != binary_sha:
        ap.error("accuracy report must match this exact model and binary")
    if not 1 <= accuracy["batch"] <= 16:
        ap.error("the reused Worker timing harness supports physical batches up to 16")
    for capacity in capacities:
        for profile in (f"fp16-w{capacity}", f"{args.scope}-w{capacity}"):
            if not any(row["profile"] == profile for row in accuracy["results"]):
                ap.error(f"missing accuracy coverage for {profile}")
    args.output.mkdir(parents=True, exist_ok=False)
    base = (f"rules=chinese\nkomi=7.5\nnnMaxBatchSize={accuracy['batch']}\n"
            "numNNServerThreadsPerModel=1\nnnCacheSizePowerOfTwo=10\nnnMutexPoolSizePowerOfTwo=8\n")
    fp16_cfg, int8_cfg = args.output / "fp16.cfg", args.output / "int8.cfg"
    fp16_cfg.write_text(base + "nnBackend=cudabackend\n", encoding="utf-8")
    min_width = accuracy.get("min_ffn_width", 0)
    int8_cfg.write_text(base + f"nnBackend=cudaint8backend\ncudaInt8Scope={args.scope}\ncudaInt8MinFfnWidth={min_width}\n", encoding="utf-8")
    protocol = worker.Protocol(ROOT / "crates/kata_worker/proto/worker.proto")
    fixture = json.loads((ROOT / "scripts/fixtures/worker_positions.json").read_text(encoding="utf-8"))
    warmup, measured, digest, templates = worker.prepare_requests(protocol, fixture, sha, 256, args.requests, "uncached", False)
    options = SimpleNamespace(output=args.output, model=args.model, model_sha256=sha,
        rust_worker=args.binary, rust_config=int8_cfg,
        baseline_worker=args.baseline_binary or args.binary,
        baseline_config=args.baseline_config or fp16_cfg,
        rust_environment={}, baseline_environment={}, workload="uncached", sample_gpu=False,
        gpu_index=0, startup_timeout=180, task_timeout=180)
    report = dict(status="RUNNING", production_certified=False, playing_strength="NOT_MEASURED",
        scope=args.scope, min_ffn_width=min_width, model_sha256=sha, binary_sha256=binary_sha,
        baseline_binary_sha256=file_hash(options.baseline_worker),
        baseline_kind="external-config" if args.baseline_binary else "same-build-fp16-default",
        accuracy_report=str(args.accuracy_report.resolve()), accuracy_report_sha256=file_hash(args.accuracy_report),
        accuracy_results=accuracy["results"], batch=accuracy["batch"], requests=args.requests,
        requests_sha256=digest, semantic_templates=templates, ownership=False, runs=[], comparisons=[])
    try:
        environment = clean_environment()
        for capacity in capacities:
            values = []
            for index, flavor in enumerate(("baseline", "rust", "rust", "baseline")):
                run = worker.run_worker(flavor, index, capacity, options, protocol, warmup, measured, environment)
                if flavor == "rust" and f"int8-scope={args.scope}" not in run["hello"]["backend_info"]:
                    raise AssertionError("Worker did not report the requested INT8 precision")
                if flavor == "rust" and min_width and f"int8-min-ffn-width={min_width};" not in run["hello"]["backend_info"]:
                    raise AssertionError("Worker did not report the requested INT8 layer selection")
                m = run["measurement"]
                value = m["rpc_requests_per_second"]
                values.append(value)
                report["runs"].append(dict(capacity=capacity, flavor=flavor, rpc_per_second=value,
                    average_nn_batch=m["average_nn_batch"], nn_rows_per_second=m["nn_rows_per_second"]))
                write_json(args.output / "report.json", report)
                print(f"C{capacity} {flavor}: {value:.2f} uncached RPC/s", flush=True)
            result = decide([values[0], values[3]], values[1:3], 0.01, 0.05)
            report["comparisons"].append(dict(capacity=capacity, **result))
            print(json.dumps(report["comparisons"][-1]), flush=True)
        report["status"] = "COLLECTED"
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        raise
    finally:
        protocol.close()
        write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
