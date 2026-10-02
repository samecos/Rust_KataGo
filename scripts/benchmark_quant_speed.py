"""Speed-only, sequential evaluator benchmarking; no accuracy comparisons.

Consumes a small explicit plan of existing binaries/models/recipes/configs.
Each run uses a single batch cap and bypasses the NN result cache. Evidence is
written to a fresh directory; previous experiments are never restarted.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def verify(record):
    require(source(record["path"]) == record, "source changed: " + record["path"])


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def parse_eval(stdout, variant, iterations, warmup, workers):
    headers = re.findall(r"^nnbench\(eval\): model=(.*?) device=(\d+) iterations=(\d+) warmup=(\d+)$",
                         stdout, re.M)
    require(len(headers) == 1, "expected exactly one evaluator header")
    model, devices, got_iterations, got_warmup = headers[0]
    require(Path(model).resolve() == Path(variant["model"]["path"]).resolve(), "wrong model")
    require((int(got_iterations), int(got_warmup)) == (iterations, warmup), "wrong iteration counts")
    require(int(devices) == 1, "this plan benchmarks exactly one GPU")
    pattern = (r"^\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*(\d+)\s*\|"
               r"\s*(\d+)\s*\|\s*([\d.]+)\s*\(workers (\d+), physicalMax (\d+)\)\s*$")
    rows = re.findall(pattern, stdout, re.M)
    require(len(rows) == 1, "expected exactly one batch measurement")
    cap, cost, rate, count, batches, average, clients, physical_max = rows[0]
    cap, count, batches, clients, physical_max = map(int, (cap, count, batches, clients, physical_max))
    cost, rate, average = map(float, (cost, rate, average))
    require(cap == physical_max == variant["batch_cap"] and clients == workers, "wrong batch or workers")
    require(count == workers * iterations and 1 <= batches <= count, "incomplete request counters")
    require(all(math.isfinite(x) and x > 0 for x in (cost, rate, average)), "invalid timing")
    require(1 <= count / batches <= cap, "invalid actual batch average")
    require(abs(average - count / batches) <= 0.0051, "rounded batch average mismatch")
    # Both metrics come from the same clock interval, but the CLI rounds them.
    rate_low, rate_high = max(rate - 0.050001, 1e-12), rate + 0.050001
    from_rate = (count / rate_high, count / rate_low)
    from_cost = (max(cost - 0.00050001, 0) * batches / 1000,
                 (cost + 0.00050001) * batches / 1000)
    require(max(from_rate[0], from_cost[0]) <= min(from_rate[1], from_cost[1]),
            "inconsistent rounded wall timing")
    profiles = re.findall(r"^\[cuda-quant\] inference_profile=([^\s]+) validation=unverified$", stdout, re.M)
    if variant["backend"] == "cudaquantbackend":
        require(len(profiles) == 1 and re.fullmatch(r"rustgo-quant-v1:[0-9a-f]{64}", profiles[0]),
                "missing quantized inference profile")
    else:
        require(not profiles, "unexpected quantized profile on legacy backend")
    return dict(nn_rows=count, nn_batches=batches, actual_average_batch=count / batches,
                all_logical_batches_full=count == batches * cap,
                nn_evals_per_second=rate, wall_ms_per_completed_batch=cost,
                rounded_derived_elapsed_seconds=count / rate,
                inference_profile=profiles[0] if profiles else None,
                single_request_latency_percentiles="NOT_MEASURED",
                physical_batch_distribution="NOT_TRACED")


def summarize(runs, minimum_gain=0.01, maximum_spread=0.05):
    require([r["flavor"] for r in runs] == ["baseline", "candidate", "candidate", "baseline"],
            "ABBA must be complete and in order")
    result = {}
    for name in ("baseline", "candidate"):
        identities = {r["metrics"]["inference_profile"] for r in runs if r["flavor"] == name}
        require(len(identities) == 1, "inference profile changed between repeated arms")
        rates = [r["metrics"]["nn_evals_per_second"] for r in runs if r["flavor"] == name]
        require(len(rates) == 2 and all(math.isfinite(x) and x > 0 for x in rates), "invalid ABBA samples")
        result[name] = dict(samples=rates, geometric_mean=statistics.geometric_mean(rates),
                            relative_spread=max(rates) / min(rates) - 1)
    ratio = result["candidate"]["geometric_mean"] / result["baseline"]["geometric_mean"]
    stable = all(result[k]["relative_spread"] <= maximum_spread for k in ("baseline", "candidate"))
    separation = min(result["candidate"]["samples"]) / max(result["baseline"]["samples"])
    result.update(candidate_over_baseline=ratio, throughput_gain_percent=(ratio - 1) * 100,
                  worst_observed_candidate_over_best_baseline=separation,
                  stable=stable, point_estimate_pass=stable and ratio >= 1 + minimum_gain,
                  speed_pass=stable and ratio >= 1 + minimum_gain and separation > 1,
                  accuracy="USER_VERIFICATION_NOT_EVALUATED")
    return result


def validate_schedule(schedule, variants):
    for item in schedule["runs"]:
        require(item["variant"] in variants and 1 <= item["iterations"] <= 20000 and 1 <= item["warmup"] <= 1000,
                "invalid measurement request")
    groups = {r.get("comparison") for r in schedule["runs"]} - {None}
    for group in groups:
        items = [r for r in schedule["runs"] if r.get("comparison") == group]
        require([r["flavor"] for r in items] == ["baseline", "candidate", "candidate", "baseline"],
                "comparison requires ordered ABBA")
        require(items[0]["variant"] == items[3]["variant"] and items[1]["variant"] == items[2]["variant"],
                "comparison changed variant between repeated arms")
        identities = [(variants[r["variant"]]["model"], variants[r["variant"]]["batch_cap"],
                       r["iterations"], r["warmup"]) for r in items]
        require(all(x == identities[0] for x in identities), "comparison changed model/batch/measurement length")
        require(items[0]["variant"] != items[1]["variant"], "comparison must use distinct configurations")


def telemetry():
    query = "name,uuid,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used"
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=5,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        return dict(returncode=result.returncode, query=query, output=result.stdout.strip())
    except Exception as error:
        return dict(error=str(error))


def execute(plan, item, variant, output, index, environment):
    require(time.monotonic_ns() < plan["deadline_monotonic_ns"], "plan deadline expired")
    for key in ("binary", "model", "config", "recipe", "tactic_plan"):
        record = variant.get("binary", plan["binary"]) if key == "binary" else variant.get(key)
        if record:
            verify(record)
    child_environment = dict(environment)
    # Do not let inherited experimental tactics override this plan.
    for key in list(child_environment):
        if key.upper().startswith("KATAGO_"):
            del child_environment[key]
    child_environment.update(variant["environment"])
    workers, iterations, warmup = plan["workers"], item["iterations"], item["warmup"]
    command = [variant.get("binary", plan["binary"])["path"], "nnbench", "--mode", "eval", "--model", variant["model"]["path"],
               "--config", variant["config"]["path"], "--batch", str(variant["batch_cap"]),
               "--workers", str(workers), "--iterations", str(iterations), "--warmup", str(warmup)]
    stem = f"{index:03d}-{variant['id']}"
    record = dict(index=index, variant=variant["id"], flavor=item.get("flavor", "screen"),
                  comparison=item.get("comparison"), stage=item.get("stage"), command=command,
                  before=telemetry(), status="STARTING", accuracy="USER_VERIFICATION_NOT_EVALUATED")
    receipt = output / (stem + ".json")
    save(receipt, record)
    start = time.monotonic_ns()
    deadline = min(plan["deadline_monotonic_ns"], start + plan["per_run_timeout_seconds"] * 1_000_000_000)
    require(start < deadline, "deadline expired before native start")
    process = None
    pending_error = None
    try:
        with (output / (stem + ".stdout.log")).open("xb") as stdout, (output / (stem + ".stderr.log")).open("xb") as stderr:
            process = subprocess.Popen(command, env=child_environment, stdout=stdout, stderr=stderr,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            record.update(pid=process.pid, started_monotonic_ns=start, status="RUNNING")
            save(receipt, record)
            try:
                actual_exit = process.wait(timeout=max(0.001, (deadline - time.monotonic_ns()) / 1e9))
                record["actual_exit"] = actual_exit
            except subprocess.TimeoutExpired:
                record["timed_out"] = True
                raise
        require(record["actual_exit"] == 0, "native benchmark returned nonzero")
        record["metrics"] = parse_eval((output / (stem + ".stdout.log")).read_text(encoding="utf-8"),
                                       variant, iterations, warmup, workers)
        require(time.monotonic_ns() <= deadline, "completed after run deadline")
        for key in ("binary", "model", "config", "recipe", "tactic_plan"):
            pin = variant.get("binary", plan["binary"]) if key == "binary" else variant.get(key)
            if pin:
                verify(pin)
        record["status"] = "COMPLETE_SPEED_SAMPLE"
    except BaseException as error:
        record.update(status="FAILED_NO_RETRY", error=f"{type(error).__name__}: {error}")
        pending_error = error
    finally:
        cleanup_errors = []
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except Exception as error:
                cleanup_errors.append("kill: " + str(error))
            try:
                record["cleanup_exit"] = process.wait(timeout=10)
            except Exception as error:
                cleanup_errors.append("cleanup wait: " + str(error))
        record.update(process_wall_seconds=(time.monotonic_ns() - start) / 1e9, after=telemetry())
        record["cleanup_errors"] = cleanup_errors
        if cleanup_errors or time.monotonic_ns() > deadline:
            record["status"] = "FAILED_NO_RETRY"
            record["final_error"] = "cleanup failure or run deadline exceeded"
            if pending_error is None:
                pending_error = RuntimeError(record["final_error"])
        save(receipt, record)
    if pending_error is not None:
        raise pending_error
    require(time.monotonic_ns() <= deadline, "receipt completed after run deadline")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--schedule-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(source(args.plan)["sha256"] == args.plan_sha256, "wrong plan digest")
    require(source(args.schedule)["sha256"] == args.schedule_sha256, "wrong schedule digest")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    require(plan["schema"] == "rustgo-quant-speed-plan-v1", "wrong plan schema")
    verify(plan["runner"])
    require(Path(plan["runner"]["path"]).resolve() == Path(__file__).resolve(), "wrong runner")
    verify(plan["environment"])
    verify(plan["source_nnbench"])
    environment = json.loads(Path(plan["environment"]["path"]).read_text(encoding="utf-8"))["values"]
    schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
    require(0 < len(schedule["runs"]) <= plan["maximum_model_starts"], "schedule exceeds start budget")
    variants = {v["id"]: v for v in plan["variants"]}
    require(len(variants) == len(plan["variants"]), "duplicate variant identity")
    validate_schedule(schedule, variants)
    # Each stage is a separate finite schedule and fresh evidence directory.
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema="rustgo-quant-speed-report-v1", status="RUNNING", plan=source(args.plan),
                  schedule=source(args.schedule), accuracy="USER_VERIFICATION_NOT_EVALUATED",
                  measurement="NnEvaluator request wall throughput; excludes model loading/warmup; includes encoding, queue, CUDA completion and decoding",
                  single_request_latency_percentiles="NOT_MEASURED", runs=[], comparisons={})
    report_path = args.output / "report.json"
    try:
        save(report_path, report)
        for index, item in enumerate(schedule["runs"], 1):
            print(f"starting {index}/{len(schedule['runs'])}: {item['variant']} {item.get('flavor', 'screen')}", flush=True)
            run = execute(plan, item, variants[item["variant"]], args.output, index, environment)
            report["runs"].append(run)
            save(report_path, report)
            print(f"complete: {run['metrics']['nn_evals_per_second']:.1f} rows/s; avg batch {run['metrics']['actual_average_batch']:.3f}", flush=True)
        groups = sorted({r["comparison"] for r in report["runs"] if r["comparison"] is not None})
        for group in groups:
            samples = [r for r in report["runs"] if r["comparison"] == group]
            report["comparisons"][group] = summarize(samples)
        report["status"] = "COMPLETE_SPEED_ONLY"
        require(time.monotonic_ns() <= plan["deadline_monotonic_ns"], "report completed after plan deadline")
        return 0
    except BaseException as error:
        report.update(status="FAILED_STOPPED_NO_RETRY", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(report_path, report)
        if time.monotonic_ns() > plan["deadline_monotonic_ns"]:
            report.update(status="FAILED_STOPPED_NO_RETRY", finalization_error="report write exceeded plan deadline")
            save(report_path, report)
            raise TimeoutError(report["finalization_error"])


if __name__ == "__main__":
    raise SystemExit(main())
