#!/usr/bin/env python3
"""Select a RustGo runtime configuration using existing, matched CUDA profiles.

Local mode measures MCTS visits/s. Worker mode measures real, uncached gRPC
requests/s against a temporary loopback harness, with no production Server.
No new kernel tactics or certification identities are generated here.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("D:/Go/Server/models/kata1-tf3-b11c768-s11001M-d5973M.bin.gz")
# Whole, previously validated profiles. Do not enumerate archive/experimental
# plans, mix their overrides, or sweep batch/lane sizes outside their scope.
PROFILES = (
    ("tf3-daily", "worker-tf3-sm120.json", 16, 1),
    ("tf3-c32", "worker-tf3-sm120-c32.json", 16, 1),
    ("tf3-throughput", "worker-tf3-sm120-throughput.json", 14, 2),
    ("onnx-q64", "best-tactic-plan.json", 16, 1),
    ("onnx-q64-wsl", "best-tactic-plan-sm120-q64-wsl.json", 16, 1),
)
TARGET_KEYS = ("gpu_name", "compute_capability", "sm_count", "l2_cache_bytes", "model_sha256")
BENCH_RE = re.compile(
    r"numSearchThreads\s*=\s*(\d+):\s*(\d+)\s*/\s*(\d+) positions, "
    r"visits/s = ([0-9.]+), nnEvals/s = ([0-9.]+), "
    r"nnBatches/s = ([0-9.]+), avgBatchSize = ([0-9.]+)")


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def clean_environment():
    return {key: value for key, value in os.environ.items() if not key.upper().startswith("KATAGO_")}


def match_plan(plan, fingerprint):
    if plan.get("schema") != 2 or plan.get("kind") != "cuda-tactic-plan":
        return False
    target = plan.get("target", {})
    return (all(target.get(key) == fingerprint[key] for key in TARGET_KEYS)
            and plan.get("backend_build") == fingerprint["backend_build"])


def discover_profiles(fingerprint):
    accepted, skipped = [], []
    for name, filename, batch, lanes in PROFILES:
        path = ROOT / "plans" / filename
        if not path.is_file():
            skipped.append(dict(profile=name, reason="plan file missing"))
            continue
        plan = json.loads(path.read_text(encoding="utf-8"))
        if not match_plan(plan, fingerprint):
            skipped.append(dict(profile=name, reason="model/device/backend identity mismatch"))
            continue
        accepted.append(dict(name=name, plan=str(path), plan_sha256=file_hash(path), batch=batch, lanes=lanes))
    if not accepted:
        # Compatibility baseline only: no claim of kernel certification and no
        # automatic activation of dormant tactics on a new model or machine.
        accepted.append(dict(name="cuda-default", plan=None, batch=16, lanes=1))
    return accepted, skipped


def parse_benchmark(output, threads, positions):
    matches = [m for m in BENCH_RE.finditer(output)
               if int(m[1]) == threads and int(m[2]) == positions and int(m[3]) == positions]
    if not matches:
        raise RuntimeError("benchmark has no complete result for the requested threads/positions")
    row = matches[-1]
    result = dict(score=float(row[4]), nn_rows_per_second=float(row[5]), average_nn_batch=float(row[7]))
    if not all(math.isfinite(value) and value > 0 for value in result.values()):
        raise RuntimeError("benchmark returned invalid or zero metrics")
    return result


def decide(a, b, minimum, spread):
    if len(a) != 2 or len(b) != 2 or any(not math.isfinite(v) or v <= 0 for v in a + b):
        raise ValueError("ABBA requires two positive finite samples per side")
    a_geo, b_geo = statistics.geometric_mean(a), statistics.geometric_mean(b)
    a_spread, b_spread = max(a) / min(a) - 1, max(b) / min(b) - 1
    stable = a_spread <= spread and b_spread <= spread
    return dict(accepted=stable and b_geo >= a_geo * (1 + minimum),
                stable=stable, baseline_geomean=a_geo, candidate_geomean=b_geo,
                improvement=b_geo / a_geo - 1, baseline_spread=a_spread, candidate_spread=b_spread)


def validate_gtp_smoke(output):
    # Current RustGo serializes IDs as '= 1', while other GTP engines use '=1'.
    replies = re.findall(r"^([=?])\s*(\d+)[ \t]*([^\r\n]*)", output, re.M)
    if len(replies) != 4 or [(status, number) for status, number, _ in replies] != [
            ("=", str(i)) for i in range(1, 5)]:
        raise RuntimeError("GTP smoke returned a missing/error response")
    if not re.fullmatch(r"(?:[A-HJ-T](?:[1-9]|1[0-9])|pass|resign)", replies[2][2].strip(), re.I):
        raise RuntimeError("GTP smoke genmove returned an invalid move")


def config_text(profile, threads, visits):
    lines = ["# Generated by scripts/tune_runtime.py; see report.json for scope and evidence.",
             "rules = chinese", "komi = 7.5", "nnBackend = cudabackend",
             f"nnMaxBatchSize = {profile['batch']}", f"numNNServerThreadsPerModel = {profile['lanes']}",
             "nnCacheSizePowerOfTwo = 18", "nnMutexPoolSizePowerOfTwo = 14",
             f"numSearchThreads = {threads}", f"maxVisits = {visits}"]
    if profile["plan"]:
        plan_path = Path(profile["plan"]).as_posix()
        if any(char in plan_path for char in "#\n\r"):
            raise ValueError("plan path cannot contain a config comment or newline")
        lines.append(f"cudaTacticPlan = {plan_path}")
    return "\n".join(lines) + "\n"


def ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def launch_script(binary, model, config, mode, capacity, model_sha):
    options = [mode, "--model", str(model), "--config", str(config)]
    if mode == "nnworker":
        options += ["--capacity", str(capacity), "--model-sha256", model_sha]
    elif mode == "analysis":
        options += ["--analysis-threads", "1"]
    command = "& " + " ".join(ps_quote(v) for v in [binary, *options]) + " @args"
    # Preserve the caller's environment after the child exits. This makes the
    # launch use the same clean tactic environment as the selection run.
    return ("# Append engine arguments, e.g. --server HOST:50051 --worker-id gpu-1\n"
            "$ErrorActionPreference = 'Stop'\n$savedTactics = @{}\n"
            "Get-ChildItem Env: | Where-Object { $_.Name -like 'KATAGO_*' } | ForEach-Object {\n"
            "  $savedTactics[$_.Name] = $_.Value\n  Remove-Item -LiteralPath ('Env:' + $_.Name)\n}\n"
            f"Push-Location -LiteralPath {ps_quote(ROOT)}\ntry {{\n"
            f"  if ($MyInvocation.ExpectingInput) {{\n    $input | {command}\n  }} else {{\n    {command}\n  }}\n"
            "  $engineExitCode = $LASTEXITCODE\n} finally {\n  Pop-Location\n"
            "  foreach ($key in $savedTactics.Keys) { Set-Item -LiteralPath ('Env:' + $key) -Value $savedTactics[$key] }\n}\n"
            "exit $engineExitCode\n")


class Tuner:
    def __init__(self, args, output):
        self.args, self.output = args, output
        self.environment = clean_environment()
        self.counter = 0
        self.protocol = None
        self.report = dict(status="RUNNING", mode=args.mode, runs=[], comparisons=[],
                           arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                           cleared_environment_keys=sorted(set(os.environ) - set(self.environment)),
                           metric="MCTS visits/s" if args.mode == "local" else "uncached loopback gRPC requests/s",
                           numerical_scope="Reuse matched, previously validated whole profiles; no new numerical certification.")

    def save(self):
        save_json(self.output / "report.json", self.report)

    def run_command(self, command, label, input_text=None):
        self.counter += 1
        prefix = self.output / f"{self.counter:03d}-{label}"
        save_json(prefix.with_suffix(".command.json"), command)
        with prefix.with_suffix(".stdout.log").open("w", encoding="utf-8") as stdout, \
                prefix.with_suffix(".stderr.log").open("w", encoding="utf-8") as stderr:
            result = subprocess.run(command, cwd=ROOT, env=self.environment, input=input_text,
                                    text=True, encoding="utf-8", stdout=stdout, stderr=stderr,
                                    timeout=self.args.timeout,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if result.returncode:
            raise RuntimeError(f"{label} failed (exit {result.returncode}); see {prefix}.stderr.log")
        return prefix.with_suffix(".stdout.log").read_text(encoding="utf-8")

    def prepare_worker(self):
        try:
            import benchmark_workers as worker
        except ImportError as error:
            raise RuntimeError("Worker tuning needs grpcio and grpcio-tools. Use scripts/tune_rustgo.ps1 "
                               "or install scripts/requirements-runtime-tune.txt with this Python.") from error
        self.worker = worker
        self.protocol = worker.Protocol(ROOT / "crates/kata_worker/proto/worker.proto")
        fixture_path = ROOT / "scripts/fixtures/worker_positions.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        self.warmup, self.measured, digest, count = worker.prepare_requests(
            self.protocol, fixture, self.report["fingerprint"]["model_sha256"],
            max(128, self.args.capacity), self.args.requests, "uncached", self.args.ownership)
        self.report.update(requests_sha256=digest, semantic_templates=count,
                           fixture_sha256=file_hash(fixture_path),
                           proto_sha256=file_hash(ROOT / "crates/kata_worker/proto/worker.proto"))

    def candidate(self, profile, threads):
        name = f"{profile['name']}-t{threads}"
        cfg = self.output / f"candidate-{name}.cfg"
        cfg.write_text(config_text(profile, threads, self.args.max_visits), encoding="utf-8")
        return dict(name=name, config=str(cfg), profile=profile, threads=threads)

    def measure(self, candidate):
        args = self.args
        print(f"  measuring {candidate['name']} ...", flush=True)
        if args.mode == "local":
            output = self.run_command([
                str(args.binary), "benchmark", "--model", str(args.model), "--config", candidate["config"],
                "-v", str(args.visits), "-n", str(args.positions), "-t", str(candidate["threads"]),
                "--fixed-batch-size", str(candidate["profile"]["batch"])], "search")
            metrics = parse_benchmark(output, candidate["threads"], args.positions)
        else:
            self.counter += 1
            worker_args = SimpleNamespace(
                output=self.output, rust_worker=args.binary, rust_config=Path(candidate["config"]),
                model=args.model, model_sha256=self.report["fingerprint"]["model_sha256"],
                rust_environment={}, baseline_environment={}, workload="uncached",
                sample_gpu=False, gpu_index=0, startup_timeout=args.timeout, task_timeout=args.timeout)
            result = self.worker.run_worker("rust", self.counter, args.capacity, worker_args,
                                            self.protocol, self.warmup, self.measured, self.environment)
            measured = result["measurement"]
            metrics = dict(score=measured["rpc_requests_per_second"],
                           nn_rows_per_second=measured["nn_rows_per_second"],
                           average_nn_batch=measured["average_nn_batch"],
                           rtt_us=measured["timings"]["round_trip_us"])
        self.report["runs"].append(dict(candidate=candidate["name"], **metrics))
        self.save()
        print(f"    {metrics['score']:.2f} {self.report['metric']}; avgBatch={metrics['average_nn_batch']:.2f}", flush=True)
        return metrics["score"]

    def compare(self, baseline, candidate, stage):
        samples = [self.measure(item) for item in (baseline, candidate, candidate, baseline)]
        decision = decide([samples[0], samples[3]], samples[1:3], self.args.min_improvement, self.args.max_spread)
        self.report["comparisons"].append(dict(stage=stage, baseline=baseline["name"],
                                                candidate=candidate["name"], samples=samples, **decision))
        self.save()
        print(f"  {stage}: {'ACCEPT' if decision['accepted'] else 'KEEP BASELINE'}; "
              f"delta={decision['improvement']:+.2%}; stable={decision['stable']}", flush=True)
        return decision["accepted"]

    def smoke_local(self, candidate):
        common = ["--model", str(self.args.model), "--config", candidate["config"]]
        output = self.run_command([str(self.args.binary), "gtp", *common], "gtp-smoke",
                                  "1 boardsize 19\n2 clear_board\n3 genmove B\n4 quit\n")
        validate_gtp_smoke(output)
        query = dict(id="runtime-tune-smoke", moves=[["B", "D4"], ["W", "Q16"]],
                     rules="chinese", komi=7.5, boardXSize=19, boardYSize=19,
                     maxVisits=32, includeOwnership=True)
        output = self.run_command([str(self.args.binary), "analysis", *common, "--analysis-threads", "1"],
                                  "analysis-smoke", json.dumps(query) + "\n")
        rows = [json.loads(line) for line in output.splitlines() if line.strip()]
        if (not rows or any("error" in row for row in rows)
                or not any(row.get("id") == query["id"] and row.get("moveInfos")
                           and "rootInfo" in row and not row.get("isDuringSearch", False) for row in rows)):
            raise RuntimeError("JSON analysis smoke returned no complete analysis")
        self.report["smoke"] = "GTP genmove and JSON analysis PASS"

    def execute(self):
        args = self.args
        self.report["binary_sha256"] = file_hash(args.binary)
        fingerprint = json.loads(self.run_command([str(args.binary), "cuda-fingerprint", "--model", str(args.model)], "fingerprint"))
        for key in (*TARGET_KEYS, "backend_build"):
            if key not in fingerprint:
                raise RuntimeError(f"cuda-fingerprint lacks {key}; use a CUDA release binary")
        if file_hash(args.model) != fingerprint["model_sha256"]:
            raise RuntimeError("model hash disagrees with cuda-fingerprint")
        self.report["fingerprint"] = fingerprint
        profiles, skipped = discover_profiles(fingerprint)
        self.report.update(profiles=profiles, skipped_profiles=skipped)
        print(f"GPU: {fingerprint['gpu_name']}; model: {fingerprint['model_sha256']}", flush=True)
        print("Profiles: " + ", ".join(p["name"] for p in profiles), flush=True)
        if profiles[0]["plan"] is None:
            print("No matching certified plan: using CUDA defaults; kernel optimization is unavailable for this identity.", flush=True)
        for profile in profiles:
            if profile["plan"]:
                destination = self.output / Path(profile["plan"]).name
                shutil.copyfile(profile["plan"], destination)
                profile["source_plan"] = profile["plan"]
                profile["plan"] = str(destination)
        initial_threads = args.threads[0] if args.mode == "local" else 1
        candidates = [self.candidate(profile, initial_threads) for profile in profiles]
        self.report["candidate_count"] = len(profiles) * (len(args.threads) if args.mode == "local" else 1)
        self.save()
        if args.dry_run:
            self.report["status"] = "DRY_RUN"
            return
        if args.mode == "worker":
            self.prepare_worker()
        initial = winner = candidates[0]
        # Exhaust the small approved profile x search-thread grid. ABBA compares
        # each challenger with the current winner; final confirmation is against
        # the original baseline, so incremental choices cannot hide a regression.
        if args.mode == "local":
            candidates = [self.candidate(p, t) for p in profiles for t in args.threads]
        if args.smoke_only:
            for candidate in candidates:
                self.measure(candidate)
            if args.mode == "local":
                self.smoke_local(initial)
            self.report["status"] = "SMOKE_ONLY"
            return
        if len(candidates) == 1:
            samples = [self.measure(initial), self.measure(initial)]
            self.report["baseline_samples"] = samples
            self.report["baseline_stable"] = max(samples) / min(samples) - 1 <= args.max_spread
        else:
            for candidate in candidates[1:]:
                if self.compare(winner, candidate, "selection"):
                    winner = candidate
        if winner["name"] != initial["name"] and not self.compare(initial, winner, "final-confirmation"):
            winner = initial
        if args.mode == "local":
            self.smoke_local(winner)
        else:
            self.measure(winner)  # Final serialized CFG through real NN/RPC + Drain.
            self.report["smoke"] = "Real uncached gRPC, NN counters and Drain PASS"
        # Refuse publishing a run during which its model/binary/plan changed.
        if file_hash(args.binary) != self.report["binary_sha256"] or file_hash(args.model) != fingerprint["model_sha256"]:
            raise RuntimeError("binary or model changed during tuning")
        for profile in profiles:
            if profile["plan"] and file_hash(profile["plan"]) != profile["plan_sha256"]:
                raise RuntimeError("copied plan changed during tuning")
        config = self.output / "rustgo.cfg"
        temporary = self.output / "rustgo.cfg.tmp"
        shutil.copyfile(winner["config"], temporary)
        modes = ("gtp", "analysis") if args.mode == "local" else ("nnworker",)
        for mode in modes:
            (self.output / f"run-{mode}.ps1").write_text(
                launch_script(args.binary, args.model, config, mode, args.capacity, fingerprint["model_sha256"]), encoding="utf-8-sig")
        temporary.replace(config)
        self.report.update(status="PASS", winner=winner, config=str(config),
                           selection_status="IMPROVED" if winner["name"] != initial["name"] else "BASELINE_RETAINED")
        print(f"Config: {config}\nLaunchers: {', '.join('run-' + mode + '.ps1' for mode in modes)}", flush=True)


def thread_list(value):
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("threads must be comma-separated integers") from error
    if not values or len(set(values)) != len(values) or any(v < 1 or v > 256 for v in values):
        raise argparse.ArgumentTypeError("threads must be distinct integers in 1..256")
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local", "worker"), default="local")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release" / ("katago-rs.exe" if os.name == "nt" else "katago-rs"))
    parser.add_argument("--output", type=Path, help="New output directory; existing directories are never overwritten")
    parser.add_argument("--threads", type=thread_list, default=thread_list("8,4,12,16"), help="Local search grid; first value is the baseline")
    parser.add_argument("--capacity", type=int, default=32, help="Worker's sustained request window, held fixed for fair comparison")
    parser.add_argument("--visits", type=int, default=800, help="Visits per local benchmark position")
    parser.add_argument("--positions", type=int, default=3)
    parser.add_argument("--max-visits", type=int, default=800, help="Search limit written to the output config")
    parser.add_argument("--requests", type=int, default=1024, help="Timed uncached requests per Worker process")
    parser.add_argument("--ownership", action="store_true", help="Include ownership in the Worker workload")
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument("--max-spread", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=600)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", help="Read fingerprint and list candidates; no inference or deployable CFG")
    group.add_argument("--smoke-only", action="store_true", help="Exercise candidates/protocols once; no performance selection or deployable CFG")
    args = parser.parse_args(argv)
    if not (1 <= args.capacity <= 4096 and 1 <= args.positions <= 100 and args.max_visits >= 1
            and args.visits >= (1 if args.smoke_only else 400) and args.requests >= 1024):
        parser.error("capacity 1..4096, positions 1..100, max-visits >=1, visits >=400 (or smoke-only), requests >=1024 required")
    if not (math.isfinite(args.timeout) and args.timeout > 0 and 0.01 <= args.min_improvement <= 1
            and 0 < args.max_spread <= 0.05):
        parser.error("positive timeout, min-improvement in [0.01,1], max-spread in (0,0.05] required")
    args.binary, args.model = args.binary.resolve(), args.model.resolve()
    for path in (args.binary, args.model):
        if not path.is_file():
            parser.error(f"File missing: {path}")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = (args.output or ROOT / "autotune-output" / f"{stamp}-{args.mode}").resolve()
    if output.exists():
        parser.error(f"Output directory already exists: {output}; choose a new path")
    output.mkdir(parents=True)
    tuner = Tuner(args, output)
    try:
        tuner.execute()
    except (Exception, KeyboardInterrupt) as error:
        tuner.report.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAIL", error=str(error))
        print(f"Tuning failed: {error}", file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        tuner.save()
        if tuner.protocol:
            tuner.protocol.close()
        print(f"Report: {output / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
