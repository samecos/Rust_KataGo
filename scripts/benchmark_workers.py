#!/usr/bin/env python3
"""Benchmark real C++/Rust TF3 workers with a continuously replenished gRPC window.

Default: concurrency 32, 256 warmup requests, 1024 measured requests, uncached
inference, C++/Rust/Rust/C++ process order. Use --concurrency 1,8,16,32,64,128
for a sweep. The two deployment configs must explicitly use nnMaxBatchSize=16.
C++ defaults to its production FP16 config, not the FP32 numerical reference.

All requests are constructed before any measurement. The timed loop validates
identity and records small timing tuples; it performs no JSON conversion or disk
writes. Each completion immediately releases one replacement request. Startup,
warmup, final heartbeat collection, Drain and report writing are not timed.

RPC requests/s and actual NN rows/s are separate. --workload cached deliberately
reuses the fixed positions with the cache enabled; it is not a pure NN benchmark.
No worker processes overlap. Run only when other GPU benchmarks/builds are idle.
Workload PASS is separate from performance_status: comparisons require at least
two samples per side and the predeclared maximum relative spread on both sides.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import queue
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time

from compare_worker_outputs import (
    MODEL_SHA256, ROOT, build_requests, file_hash, message_dict, write_json,
)
from worker_protocol_tools import Protocol, WorkerHarness


@dataclass(frozen=True)
class PreparedRequest:
    message: object
    identity: tuple


def prepare_requests(protocol, fixture, model_hash, warmup_count, measured_count, workload, ownership):
    templates = build_requests(protocol.pb, fixture, model_hash, 0)
    prepared = []
    digest = hashlib.sha256()
    for index in range(warmup_count + measured_count):
        name, template = templates[index % len(templates)]
        request = protocol.pb.EvalRequest()
        request.CopyFrom(template)
        request.task_id = index + 1
        request.session_id = "same-model-worker-throughput"
        request.parameters.skip_cache = workload == "uncached"
        request.parameters.include_ownership = ownership
        request.input_hash = b""
        request.input_hash = hashlib.sha256(request.SerializeToString(deterministic=True)).digest()
        message = protocol.pb.ServerMessage(evaluate=request)
        raw = message.SerializeToString(deterministic=True)
        digest.update(len(raw).to_bytes(8, "little") + raw)
        prepared.append(PreparedRequest(message, (
            request.task_id, request.generation, request.session_id,
            request.input_hash, request.model_sha256,
        )))
    return prepared[:warmup_count], prepared[warmup_count:], digest.hexdigest(), len(templates)


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def distribution(values):
    values = [value for value in values if value is not None]
    if not values:
        return dict(count=0, mean=None, p50=None, p95=None, maximum=None)
    return dict(count=len(values), mean=statistics.fmean(values), p50=percentile(values, 0.5),
                p95=percentile(values, 0.95), maximum=max(values))


def read_process_cpu(process):
    # Optional observation only. No dependency is needed for the benchmark.
    try:
        import psutil
        times = psutil.Process(process.pid).cpu_times()
        return times.user + times.system
    except (ImportError, OSError):
        return None
    except Exception:
        return None


class GpuSampler:
    """Optional persistent nvidia-smi sampler, disabled by default."""
    def __init__(self, enabled, gpu_index):
        self.process = None
        self.thread = None
        self.samples = []
        self.error = None
        if not enabled:
            return
        binary = shutil.which("nvidia-smi")
        if binary is None:
            self.error = "nvidia-smi unavailable"
            return
        command = [binary, f"--id={gpu_index}",
                   "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm",
                   "--format=csv,noheader,nounits", "--loop-ms=500"]
        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            text=True, bufsize=1,
                                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)

            def consume():
                for line in self.process.stdout:
                    self.samples.append((time.perf_counter_ns(), line.strip()))

            self.thread = threading.Thread(target=consume, daemon=True)
            self.thread.start()
        except OSError as error:
            self.error = str(error)

    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
            if self.thread:
                self.thread.join(timeout=2)
            self.process.stdout.close()

    def report(self, started_ns, ended_ns):
        names = ("gpu_percent", "memory_percent", "memory_mib", "power_watts", "temperature_c", "sm_mhz")
        parsed = []
        for when, line in self.samples:
            if not started_ns <= when <= ended_ns:
                continue
            fields = line.split(",")
            if len(fields) != len(names):
                self.error = self.error or line
                continue
            values = {}
            for name, value in zip(names, fields):
                try:
                    values[name] = float(value.strip())
                except ValueError:
                    values[name] = None
            parsed.append(values)
        return dict(error=self.error, count=len(parsed),
                    fields={name: distribution([sample[name] for sample in parsed]) for name in names})


def receive_message(peer, deadline_ns):
    """Queue read without JSON/protobuf copies; EOF and idle deadlines are bounded."""
    while True:
        left = (deadline_ns - time.perf_counter_ns()) / 1e9
        if left <= 0:
            raise TimeoutError("worker result deadline exceeded")
        try:
            return peer.received.get(timeout=min(left, 0.25))
        except queue.Empty:
            if peer.closed.is_set():
                raise RuntimeError(f"worker stream closed: {peer.reader_error}")


def drive_window(peer, prepared, concurrency, task_timeout, record):
    """Maintain C outstanding tasks, replacing each result before bookkeeping."""
    total = len(prepared)
    pending = {}
    next_index = 0
    completed = 0
    timings = []
    completions_ns = []
    heartbeat_count = 0
    deadline_delta = int(task_timeout * 1e9)
    cpu_started_ns = time.process_time_ns()
    started_ns = time.perf_counter_ns()

    def dispatch(index):
        task = prepared[index]
        sent_ns = time.perf_counter_ns()
        pending[task.identity[0]] = (task, sent_ns)
        peer.outgoing.put(task.message)

    while next_index < min(concurrency, total):
        dispatch(next_index)
        next_index += 1
    while pending:
        # At most 128 entries; avoiding a global timeout lets long runs remain
        # live while still rejecting one lost request among progressing tasks.
        oldest_sent = min(entry[1] for entry in pending.values())
        message = receive_message(peer, oldest_sent + deadline_delta)
        received_ns = time.perf_counter_ns()
        if message.HasField("heartbeat"):
            heartbeat_count += 1
            if message.heartbeat.failed_requests:
                raise AssertionError(("worker heartbeat reported failures", message.heartbeat.failed_requests))
            continue
        if not message.HasField("result"):
            raise AssertionError("unexpected worker message after Hello")
        result = message.result
        active = pending.pop(result.task_id, None)
        if active is None:
            raise AssertionError(("duplicate/stale result", result.task_id))
        task, sent_ns = active
        identity = (result.task_id, result.generation, result.session_id, result.input_hash, result.model_sha256)
        if identity != task.identity:
            raise AssertionError(("result identity mismatch", result.task_id))
        if result.error_code or not result.HasField("output"):
            raise AssertionError(("evaluation failed", result.task_id, result.error_code, result.error_message))

        refill_delay_us = None
        if next_index < total:
            dispatch(next_index)
            next_index += 1
            refill_delay_us = (time.perf_counter_ns() - received_ns) / 1000.0
        completed += 1
        if record:
            # Small scalars only. Discard the large output message immediately.
            timings.append((
                (received_ns - sent_ns) / 1000.0, result.elapsed_us,
                result.queue_us if result.HasField("queue_us") else None,
                result.context_us if result.HasField("context_us") else None,
                result.evaluator_us if result.HasField("evaluator_us") else None,
                refill_delay_us,
            ))
            completions_ns.append(received_ns)
    ended_ns = received_ns
    elapsed_seconds = (ended_ns - started_ns) / 1e9
    result = dict(completed=completed, elapsed_seconds=elapsed_seconds,
                  rpc_requests_per_second=completed / elapsed_seconds,
                  heartbeat_count=heartbeat_count, started_ns=started_ns, ended_ns=ended_ns,
                  harness_cpu_seconds=(time.process_time_ns() - cpu_started_ns) / 1e9)
    if record:
        names = ("round_trip_us", "worker_elapsed_us", "queue_us", "context_us", "evaluator_us", "refill_delay_us")
        result["timings"] = {name: distribution([row[index] for row in timings]) for index, name in enumerate(names)}
        if total > 2 * concurrency:
            interval = (completions_ns[total - concurrency - 1] - completions_ns[concurrency - 1]) / 1e9
            result["steady_rpc_requests_per_second"] = (total - 2 * concurrency) / interval
        result["timing_rows"] = timings
    return result


def numeric_heartbeat(heartbeat):
    return {name: int(getattr(heartbeat, name)) for name in
            ("in_flight", "completed_requests", "failed_requests", "nn_rows", "nn_batches")}


def run_worker(flavor, index, concurrency, args, protocol, warmup, measured, base_environment):
    directory = args.output / f"c{concurrency:03d}-{index + 1:02d}-{flavor}"
    directory.mkdir(parents=True, exist_ok=True)
    binary = (args.cpp_worker if flavor == "cpp" else
              args.baseline_worker if flavor == "baseline" else args.rust_worker)
    config = (args.cpp_config if flavor == "cpp" else
              args.baseline_config if flavor == "baseline" else args.rust_config)
    environment = dict(base_environment)
    if flavor != "cpp":
        environment.update(args.rust_environment)
    if flavor == "baseline":
        environment.update(args.baseline_environment)
    harness = WorkerHarness(protocol)
    flag = "-" if flavor == "cpp" else "--"
    command = [str(binary.resolve()), "nnworker", f"{flag}server", f"127.0.0.1:{harness.port}",
               f"{flag}worker-id", f"bench-{flavor}-{index}", f"{flag}capacity", str(concurrency),
               f"{flag}once", f"{flag}model", str(args.model.resolve()),
               f"{flag}config", str(config.resolve()), f"{flag}model-sha256", args.model_sha256]
    report = dict(status="RUNNING", worker=flavor, order_index=index, concurrency=concurrency,
                  command=command, model_sha256=args.model_sha256, binary_sha256=file_hash(binary),
                  config_sha256=file_hash(config), config_text=config.read_text(encoding="utf-8"),
                  environment={key: value for key, value in sorted(environment.items())
                               if key.startswith("KATAGO_CUDA_") or key in args.rust_environment
                               or key in args.baseline_environment},
                  workload=args.workload, warmup_requests=len(warmup), measured_requests=len(measured))
    process = None
    sampler = None
    try:
        with (directory / "worker.log").open("wb") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            peer = harness.accept(process, args.startup_timeout)
            hello = peer.hello
            if (hello.protocol_version != 1 or hello.input_profile != "katago-eval-v1"
                    or hello.model_sha256 != args.model_sha256 or hello.max_in_flight != concurrency
                    or hello.max_board_size < 19 or not hello.supports_friendly_pass_search):
                raise AssertionError(("incompatible Hello", hello))
            report["hello"] = message_dict(hello)
            drive_window(peer, warmup, concurrency, args.task_timeout, record=False)
            before = numeric_heartbeat(peer.settled(len(warmup), args.task_timeout))
            sampler = GpuSampler(args.sample_gpu, args.gpu_index)
            worker_cpu_before = read_process_cpu(process)
            measurement = drive_window(peer, measured, concurrency, args.task_timeout, record=True)
            worker_cpu_after = read_process_cpu(process)
            sampler.stop()
            after = numeric_heartbeat(peer.settled(len(warmup) + len(measured), args.task_timeout))
            delta = {key: after[key] - before[key] for key in after if key != "in_flight"}
            if delta["completed_requests"] != len(measured) or delta["failed_requests"] != 0:
                raise AssertionError(("completion/failure counters disagree", delta))
            if before["in_flight"] != 0 or after["in_flight"] != 0:
                raise AssertionError("NN counter boundaries were not idle")
            if args.workload == "uncached" and delta["nn_rows"] != len(measured):
                raise AssertionError(("uncached work was not actual NN inference", delta, len(measured)))
            if not 0 <= delta["nn_rows"] <= len(measured) or not 0 <= delta["nn_batches"] <= delta["nn_rows"]:
                raise AssertionError(("invalid NN counter deltas", delta))
            average_batch = delta["nn_rows"] / delta["nn_batches"] if delta["nn_batches"] else None
            if average_batch is not None and not 1 <= average_batch <= 16:
                raise AssertionError(("observed average batch exceeds agreed model batch limit", average_batch))
            elapsed = measurement["elapsed_seconds"]
            measurement.update(
                nn_rows_per_second=delta["nn_rows"] / elapsed,
                nn_batches_per_second=delta["nn_batches"] / elapsed,
                average_nn_batch=average_batch,
                nn_rows_per_rpc=delta["nn_rows"] / len(measured),
                observed_non_nn_fraction=1 - delta["nn_rows"] / len(measured),
                worker_cpu_seconds=(worker_cpu_after - worker_cpu_before)
                                   if worker_cpu_before is not None and worker_cpu_after is not None else None,
            )
            measurement["harness_cpu_core_equivalents"] = measurement["harness_cpu_seconds"] / elapsed
            if measurement["worker_cpu_seconds"] is not None:
                measurement["worker_cpu_core_equivalents"] = measurement["worker_cpu_seconds"] / elapsed
            report.update(before_heartbeat=before, after_heartbeat=after, counter_deltas=delta,
                          measurement=measurement,
                          gpu=sampler.report(measurement["started_ns"], measurement["ended_ns"]))
            peer.outgoing.put(protocol.pb.ServerMessage(drain=protocol.pb.Drain(reason="throughput measurement complete")))
            if process.wait(timeout=30) != 0:
                raise AssertionError("worker failed to exit cleanly after Drain")
            report.update(status="PASS", drained=True)
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        raise
    finally:
        if sampler:
            sampler.stop()
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
            write_json(directory / "timings.json", dict(
                columns=["round_trip_us", "worker_elapsed_us", "queue_us", "context_us", "evaluator_us", "refill_delay_us"],
                rows=rows))
        write_json(directory / "report.json", report)
    return report


def parse_environment(values, parser):
    environment = {}
    for text in values:
        key, separator, value = text.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or "\x00" in value:
            parser.error("--rust-env requires KEY=VALUE")
        environment[key] = value
    return environment


def check_batch_size(path):
    text = path.read_text(encoding="utf-8")
    values = re.findall(r"(?m)^\s*nnMaxBatchSize\s*=\s*(\d+)\s*(?:#.*)?$", text)
    if values != ["16"] or re.search(r"(?m)^\s*nnMaxBatchSize\d+\s*=", text):
        raise ValueError(f"{path}: benchmark requires exactly one explicit nnMaxBatchSize = 16 and no per-model override")


def relative_spread_limit(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("maximum-relative-spread must be finite in 0..0.1") from None
    if not math.isfinite(value) or not 0 <= value <= 0.1:
        raise argparse.ArgumentTypeError("maximum-relative-spread must be finite in 0..0.1")
    return value


def combined_performance_status(statuses):
    statuses = list(statuses)
    if "UNSTABLE_DIAGNOSTIC_ONLY" in statuses:
        return "UNSTABLE_DIAGNOSTIC_ONLY"
    if not statuses or any(status != "STABLE" for status in statuses):
        return "INSUFFICIENT_REPEATS_DIAGNOSTIC_ONLY"
    return "STABLE"


def create_report(path, report):
    # Exclusive creation also prevents simultaneous invocations from replacing
    # the same evidence. Later writes update only this invocation's report.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


def summarize(runs, maximum_relative_spread=0.05):
    maximum_relative_spread = relative_spread_limit(maximum_relative_spread)
    summaries = []
    for concurrency in sorted({run["concurrency"] for run in runs}):
        entry = dict(concurrency=concurrency, maximum_relative_spread=maximum_relative_spread,
                     workers={}, comparisons={})
        for flavor in ("cpp", "baseline", "rust"):
            group = [run["measurement"] for run in runs
                     if run["concurrency"] == concurrency and run["worker"] == flavor and run["status"] == "PASS"]
            if not group:
                continue
            rates = [run["rpc_requests_per_second"] for run in group]
            nn_rates = [run["nn_rows_per_second"] for run in group]
            entry["workers"][flavor] = dict(
                runs=len(group), rpc_geomean=statistics.geometric_mean(rates),
                rpc_min=min(rates), rpc_max=max(rates),
                rpc_relative_spread=max(rates) / min(rates) - 1,
                nn_rows_mean=statistics.fmean(nn_rates),
                average_batch_mean=statistics.fmean([run["average_nn_batch"] for run in group
                                                    if run["average_nn_batch"] is not None])
                                   if any(run["average_nn_batch"] is not None for run in group) else None,
            )
        for reference in ("cpp", "baseline"):
            if reference in entry["workers"] and "rust" in entry["workers"]:
                candidate = entry["workers"]["rust"]
                baseline = entry["workers"][reference]
                if min(candidate["runs"], baseline["runs"]) < 2:
                    status = "INSUFFICIENT_REPEATS_DIAGNOSTIC_ONLY"
                elif any(side["rpc_max"] > side["rpc_min"] * (1 + maximum_relative_spread)
                         for side in (candidate, baseline)):
                    status = "UNSTABLE_DIAGNOSTIC_ONLY"
                else:
                    status = "STABLE"
                entry["comparisons"][reference] = dict(performance_status=status)
                prefix = "" if status == "STABLE" else "diagnostic_only_"
                entry[f"{prefix}rust_over_{reference}_rpc_ratio"] = candidate["rpc_geomean"] / baseline["rpc_geomean"]
        entry["performance_status"] = combined_performance_status(
            comparison["performance_status"] for comparison in entry["comparisons"].values())
        summaries.append(entry)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("D:/Go/Server/models/kata1-tf3-b11c768-s11001M-d5973M.bin.gz"))
    parser.add_argument("--model-sha256", default=MODEL_SHA256)
    parser.add_argument("--cpp-worker", type=Path, default=Path("D:/Go/Server/worker/build-windows-CUDA/Release/katago.exe"))
    parser.add_argument("--cpp-config", type=Path, default=Path("D:/Go/Server/worker/cuda-5070ti.cfg"))
    parser.add_argument("--rust-worker", type=Path, default=ROOT / "target/release/katago-rs.exe")
    parser.add_argument("--baseline-worker", type=Path, help="Optional Rust baseline binary")
    parser.add_argument("--baseline-config", type=Path, help="Baseline config; defaults to --rust-config")
    parser.add_argument("--rust-config", type=Path, default=ROOT / "configs/worker_tf3_sm120.cfg")
    parser.add_argument("--proto", type=Path, default=ROOT / "crates/kata_worker/proto/worker.proto")
    parser.add_argument("--fixtures", type=Path, default=ROOT / "scripts/fixtures/worker_positions.json")
    parser.add_argument("--output", type=Path, default=ROOT / "target/worker-benchmark")
    parser.add_argument("--concurrency", default="32", help="One value or comma-separated sweep from 1,8,16,32,64,128")
    parser.add_argument("--order", default="cpp,rust,rust,cpp", help="Sequential cpp/rust/baseline process order; default ABBA")
    parser.add_argument("--requests", type=int, default=1024, help="Measured requests per process, at least 1024")
    parser.add_argument("--warmup", type=int, default=256, help="Warmup count, at least 128 and current concurrency")
    parser.add_argument("--workload", choices=("uncached", "cached"), default="uncached")
    parser.add_argument("--ownership", action="store_true", help="Include ownership outputs; default matches Go Server")
    parser.add_argument("--rust-env", action="append", default=[], metavar="KEY=VALUE", help="Rust-only tactic/environment override; repeatable")
    parser.add_argument("--baseline-env", action="append", default=[], metavar="KEY=VALUE", help="Baseline-only overrides applied after --rust-env; repeatable")
    parser.add_argument("--sample-gpu", action="store_true", help="Optional nvidia-smi sampling, disabled by default")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU for optional sampling only; worker configs select devices")
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--task-timeout", type=float, default=120)
    parser.add_argument("--maximum-relative-spread", type=relative_spread_limit, default=0.05,
                        help="Predeclared max/min - 1 limit for each comparison side (default 0.05)")
    args = parser.parse_args()
    try:
        concurrencies = [int(value) for value in args.concurrency.split(",")]
    except ValueError:
        parser.error("--concurrency must contain integers")
    if not concurrencies or len(set(concurrencies)) != len(concurrencies) or any(value not in (1, 8, 16, 32, 64, 128) for value in concurrencies):
        parser.error("--concurrency must be distinct values from 1,8,16,32,64,128")
    order = args.order.split(",")
    if not order or any(flavor not in ("cpp", "rust", "baseline") for flavor in order):
        parser.error("--order contains only cpp,rust,baseline")
    if "baseline" in order and args.baseline_worker is None:
        parser.error("--order baseline requires --baseline-worker")
    if args.requests < 1024 or args.warmup < 0:
        parser.error("--requests must be at least 1024; --warmup must be nonnegative")
    if args.startup_timeout <= 0 or args.task_timeout <= 0:
        parser.error("timeouts must be positive")
    args.rust_environment = parse_environment(args.rust_env, parser)
    args.baseline_environment = parse_environment(args.baseline_env, parser)
    args.baseline_config = args.baseline_config or args.rust_config
    args.model_sha256 = args.model_sha256.lower()
    if file_hash(args.model) != args.model_sha256:
        raise AssertionError("shared model SHA256 mismatch")
    for flavor in set(order):
        check_batch_size(args.cpp_config if flavor == "cpp" else
                         args.baseline_config if flavor == "baseline" else args.rust_config)
    args.output.mkdir(parents=True, exist_ok=True)
    fixture = json.loads(args.fixtures.read_text(encoding="utf-8"))
    warmup_count = max(args.warmup, 128, max(concurrencies))
    report = dict(status="RUNNING", performance_status="RUNNING",
                  maximum_relative_spread=args.maximum_relative_spread,
                  model_sha256=args.model_sha256, schema_sha256=file_hash(args.proto),
                  fixtures_sha256=file_hash(args.fixtures), concurrency=concurrencies, order=order,
                  workload=args.workload, ownership=args.ownership, physical_max_batch=16,
                  warmup_requests=warmup_count, measured_requests=args.requests,
                  host=platform.platform(), python=sys.version, runs=[],
                  semantics="RPC throughput includes request queue/network, replay and evaluator; actual NN rows/s "
                            "uses settled heartbeat counter deltas. Timed work excludes startup, warmup, final "
                            "heartbeat waiting and Drain, but includes initial fill and trailing window drain. "
                            "steady_rpc_requests_per_second trims first/last concurrency completions. "
                            "Optional resource sampling may perturb performance and is reported explicitly.",
                  sample_gpu=args.sample_gpu)
    report_path = args.output / "report.json"
    create_report(report_path, report)
    protocol = None
    try:
        protocol = Protocol(args.proto)
        warmup, measured, digest, template_count = prepare_requests(
            protocol, fixture, args.model_sha256, warmup_count, args.requests, args.workload, args.ownership)
        report.update(requests_sha256=digest, semantic_templates=template_count)
        base_environment = dict(os.environ)
        print(" C  run worker    RPC/s      NN rows/s   avgBatch   RTT p50/p95 ms   evaluator p50 ms", flush=True)
        for concurrency in concurrencies:
            for index, flavor in enumerate(order):
                print(f"starting C={concurrency} run={index + 1}/{len(order)} {flavor}", flush=True)
                result = run_worker(flavor, index, concurrency, args, protocol, warmup, measured, base_environment)
                report["runs"].append(result)
                measurement = result["measurement"]
                batch = measurement["average_nn_batch"]
                rtt = measurement["timings"]["round_trip_us"]
                evaluator = measurement["timings"]["evaluator_us"]["p50"]
                batch_text = f"{batch:.2f}" if batch is not None else "n/a"
                evaluator_text = f"{evaluator / 1000:.2f}" if evaluator is not None else "n/a"
                print(f"{concurrency:3d} {index + 1:3d} {flavor:6s} {measurement['rpc_requests_per_second']:10.1f} "
                      f"{measurement['nn_rows_per_second']:10.1f} {batch_text:>10s} "
                      f"{rtt['p50'] / 1000:7.2f}/{rtt['p95'] / 1000:7.2f} "
                      f"{evaluator_text:>12s}", flush=True)
                report["summaries"] = summarize(report["runs"], args.maximum_relative_spread)
                report["performance_status"] = combined_performance_status(
                    summary["performance_status"] for summary in report["summaries"])
                write_json(report_path, report)
        report["status"] = "PASS"
    except Exception as error:
        report.update(status="FAIL", performance_status="NOT_COMPLETED", error=str(error))
        raise
    finally:
        if protocol is not None:
            protocol.close()
        write_json(report_path, report)
    for summary in report["summaries"]:
        for reference, label in (("cpp", "C++"), ("baseline", "Rust baseline")):
            key = f"rust_over_{reference}_rpc_ratio"
            if key in summary:
                print(f"C={summary['concurrency']}: Rust/{label} geometric-mean RPC throughput "
                      f"{summary[key]:.4f}x", flush=True)
            diagnostic_key = f"diagnostic_only_{key}"
            if diagnostic_key in summary:
                print(f"C={summary['concurrency']}: diagnostic_only Rust/{label} geometric-mean RPC throughput "
                      f"{summary[diagnostic_key]:.4f}x; "
                      f"{summary['comparisons'][reference]['performance_status']}", flush=True)
    print(f"RESULT: PASS; performance_status={report['performance_status']}; {report_path}", flush=True)


if __name__ == "__main__":
    main()
