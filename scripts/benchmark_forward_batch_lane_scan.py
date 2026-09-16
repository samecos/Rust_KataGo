"""Diagnostic-only WSL TF3 forward scan: B12..16, each with one then two lanes.

Run only in an exclusive GPU window. This never certifies or changes a tactic.
Each case runs once in the pre-registered order; no retries replace samples.
Event median-rate sums and synchronized forward wall rates are separate metrics,
neither of which is Worker or end-to-end production throughput.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import time

import benchmark_fork_parity as parity

BINARY = "/mnt/d/code/Rust_KataGo/target/fork-parity-20260908/fp16-rne-r1/rust-fp16-rne-r1-wsl"
BINARY_SHA = "c7ce7400a100c995bc612249a7e06c6017ae33a008a147a46723af43334ca9d1"
MODEL = "/mnt/d/Go/Server/models/" + parity.MODEL.name
CASES = tuple((batch, lanes) for batch in range(12, 17) for lanes in (1, 2))
TACTICS = {"KATAGO_CUDA_DUALFFN": "1", "KATAGO_CUDA_ATTN_TILE": "q64-serial",
           "KATAGO_CUDA_NOGRAPH": "1", "KATAGO_CUDA_GEMM_LAYOUT": "tn",
           "KATAGO_CUDA_RESIDUAL_ALGO": "heuristic"}
POSITION_GENERATION = "one empty board repeated across the physical batch; Black to play"
HASH_ENCODING = "SHA256 of contiguous little-endian f32 bytes, before device conversion"


def case_spec(batch, lanes):
    parity.require(type(batch) is int and type(lanes) is int and (batch, lanes) in CASES,
                   "case is outside the pre-registered B12..16/S1..2 scan")
    return dict(case_id=f"B{batch}-S{lanes}", batch=batch, lanes=lanes,
                handles=list(range(1, lanes + 1)))


def env_prefix(inherited):
    keys = sorted({key for key in inherited if key.startswith("KATAGO_CUDA_")}
                  | {"KATAGO_NN_BATCH_WINDOW_US"})
    return ["env", *[item for key in keys for item in ("-u", key)],
            "LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64:/usr/lib/wsl/lib",
            *[f"{key}={value}" for key, value in TACTICS.items()]]


def build_command(binary, batch, lanes, *, warmup, iterations, inherited):
    case = case_spec(batch, lanes)
    # --handles contains lane IDs, not counts: --handles 2 would still be one lane.
    return parity.wsl(shlex.join([*env_prefix(inherited), binary, "nnbench", "--model", MODEL,
        "--mode", "kernel", "--input", "empty", "--batch", str(batch),
        "--handles", ",".join(map(str, case["handles"])), "--timing", "cuda-event",
        "--warmup", str(warmup), "--iterations", str(iterations), "--json"]))


def validate_build(build):
    parity.require(isinstance(build, dict), "missing backend build")
    for key in ("kernel_build_id", "cuda_compiler", "cutlass_version", "cutlass_commit"):
        parity.require(isinstance(build.get(key), str) and build[key], f"missing build {key}")
    parity.require(type(build.get("fp16_encoding_revision")) is int
                   and build["fp16_encoding_revision"] == 1, "corrected FP16 encoding revision 1 required")
    parity.require(type(build.get("cublaslt_version")) is int
                   and build["cublaslt_version"] == 130600, "cuBLASLt 130600 required")
    capabilities = build.get("capabilities", {})
    parity.require(capabilities.get("dual_ffn") is True and capabilities.get("attention_q64_serial") is True,
                   "DualFFN/q64-serial build capabilities missing")


def validate_fingerprint(data):
    expected = dict(gpu_name="NVIDIA GeForce RTX 5070 Ti", architecture="sm_120",
                    compute_capability="12.0", sm_count=70, l2_cache_bytes=50331648,
                    model_sha256=parity.MODEL_SHA)
    parity.require(all(data.get(key) == value for key, value in expected.items()),
                   "GPU/model fingerprint mismatch")
    validate_build(data.get("backend_build"))


def validate_payload(data, log, *, batch, lanes, iterations, warmup):
    case = case_spec(batch, lanes)
    parity.require(data.get("command") == "nnbench" and data.get("mode") == "kernel"
                   and data.get("timing") == "cuda-event" and data.get("kernel_only") is True,
                   "mode/timing mismatch")
    parity.require(parity.same_path(data.get("model"), MODEL) and data.get("model_sha256") == parity.MODEL_SHA,
                   "model path/hash mismatch")
    parity.require(data.get("input") == "empty" and data.get("iterations") == iterations
                   and data.get("warmup") == warmup, "input/iterations/warmup mismatch")
    parity.require(data.get("requested_batches") == [batch] and data.get("handles") == case["handles"]
                   and all(type(value) is int for value in data["handles"]), "batch/lane request mismatch")
    parity.require(data.get("device_target") == "sm_120", "device target mismatch")
    validate_build(data.get("build"))
    actual = {key: value for key, value in data.get("tactics", {}).items() if value is not None}
    parity.require(actual == TACTICS, "effective tactics mismatch or unexpected inherited tactic")
    description = data.get("input_description", {})
    expected_description = {"kind": "empty", "inputs_version": 7, "board": [19, 19],
        "spatial": "B,22,19,19 NCHW f32", "global": "B,19 f32", "input_hash_encoding": HASH_ENCODING,
        "position_generation": POSITION_GENERATION, "rules": parity.EMPTY_RULES, "encore_phase": 0,
        "effective_symmetry": 0, "policy_optimism": 0.0, "misc_params": parity.EMPTY_MISC}
    parity.require(all(description.get(key) == value for key, value in expected_description.items()),
                   "empty-board feature description mismatch")
    rows = data.get("results", [])
    parity.require(len(rows) == 1 and rows[0].get("batch") == batch, "result batch mismatch")
    row = rows[0]
    input_hashes = {}
    for key in ("spatial_input_sha256", "global_input_sha256"):
        parity.require(isinstance(row.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", row[key]),
                       f"missing/invalid {key}")
        input_hashes[key] = row[key]
    samples = row.get("handles", [])
    parity.require([sample.get("handle") for sample in samples] == case["handles"]
                   and all(type(sample.get("handle")) is int for sample in samples),
                   "result lane IDs mismatch")
    rates, elapsed, medians = [], [], []
    for sample in samples:
        values = sample.get("cuda_event_ms", [])
        parity.require(sample.get("iterations") == iterations and len(values) == iterations,
                       "lane event sample count mismatch")
        median = statistics.median([parity.positive(value, "event sample") for value in values])
        rate = batch * 1000.0 / median
        parity.matching_rate(sample.get("cuda_event_median_ms"), median, "lane event median")
        parity.matching_rate(sample.get("cuda_event_median_nn_evals_per_s"), rate, "lane median rate")
        lane_elapsed = parity.positive(sample.get("elapsed_ms"), "lane wall duration")
        parity.matching_rate(sample.get("per_batch_ms"), lane_elapsed / iterations, "lane wall ms per batch")
        parity.matching_rate(sample.get("nn_evals_per_s"), batch * iterations * 1000 / lane_elapsed,
                             "lane wall rate")
        rates.append(rate); medians.append(median); elapsed.append(lane_elapsed)
    event_sum = sum(rates)
    parity.matching_rate(row.get("sum_median_nn_evals_per_s"), event_sum, "event median-rate sum")
    wall_ms = parity.positive(row.get("wall_elapsed_ms"), "forward wall envelope")
    parity.require(wall_ms + 1e-6 >= max(elapsed), "wall envelope is shorter than a lane duration")
    if lanes == 1:
        parity.matching_rate(wall_ms, elapsed[0], "single-lane wall envelope")
    wall_rate = batch * lanes * iterations * 1000 / wall_ms
    parity.matching_rate(row.get("wall_nn_evals_per_s"), wall_rate, "forward wall rate")
    markers = [line for line in log.splitlines() if line.startswith("[cuda-tactic] ")]
    for expression in (r"name=gemm_layout launch=tn(?:\s|$)", r"name=dual_ffn launch=fused(?:\s|$)",
                       r"tile=q64-serial(?:\s|$)", r"name=cublaslt_rank engine=heuristic(?:\s|$)",
                       r"name=gemm kind=residual engine=cublaslt(?:\s|$)"):
        parity.require(any(re.search(expression, line) for line in markers), f"required path missing: {expression}")
    parity.require(not any(re.search(r"name=residual_algo(?:\s|$)|name=cublaslt_rank engine=time(?:\s|$)|"
                                    r"name=graph .*launch=graph(?:\s|$)", line) for line in markers),
                   "unexpected residual preset, timed ranking or graph path")
    return dict(event_median_rate_sum=event_sum, forward_wall_rate=wall_rate, forward_wall_elapsed_ms=wall_ms,
                lane_median_ms=medians, input_hashes=input_hashes, path_markers=markers)


def bind_input_and_build(data, metrics, *, batch, build, inputs):
    parity.require(data["build"] == build, "backend build/library changed between preflight/cases")
    identity = dict(hashes=metrics["input_hashes"], description=data["input_description"])
    parity.require(batch not in inputs or inputs[batch] == identity,
                   "input hashes/description differ between lane counts for the same physical batch")
    inputs[batch] = identity


def summarize(runs):
    parity.require([(run["batch"], run["lanes"]) for run in runs] == list(CASES)
                   and all(run.get("completion") == "COMPLETE" for run in runs),
                   "ranking requires the complete pre-registered order without replacements")
    def rank(metric):
        return [dict(case_id=run["case_id"], batch=run["batch"], lanes=run["lanes"],
                     value=parity.positive(run["metrics"][metric], metric))
                for run in sorted(runs, key=lambda run: -run["metrics"][metric])]
    event, wall = rank("event_median_rate_sum"), rank("forward_wall_rate")
    event_ids, wall_ids = [r["case_id"] for r in event], [r["case_id"] for r in wall]
    return dict(event_ranking=event, forward_wall_ranking=wall,
                rankings_disagree=event_ids != wall_ids, top_choice_disagrees=event_ids[0] != wall_ids[0],
                rank_comparison=[dict(case_id=case_id, event_rank=event_ids.index(case_id) + 1,
                                      wall_rank=wall_ids.index(case_id) + 1) for case_id in event_ids],
                next_step="Review both rankings and pre-register 1-2 configurations for same-binary numerical checks and fixed ABBA, then Worker validation. This single-pass scan certifies nothing.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", default=BINARY, help="immutable absolute WSL binary path")
    p.add_argument("--binary-sha256", default=BINARY_SHA, help="pre-registered SHA256; never inferred for a replacement binary")
    p.add_argument("--warmup", type=int, default=80)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--output", required=True, type=Path, help="new evidence directory; existing directories are rejected")
    args = p.parse_args(argv)
    if not args.binary.startswith("/") or not re.fullmatch(r"[0-9a-f]{64}", args.binary_sha256):
        p.error("require an absolute WSL binary path and lowercase SHA256")
    if args.warmup < 1 or not 100 <= args.iterations <= 10000:
        p.error("require warmup >= 1 and iterations in 100..10000")
    return args


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / "report.json"
    sources = [Path(__file__).resolve(), Path(parity.__file__).resolve()]
    report = dict(schema=1, status="DIAGNOSTIC", completion="RUNNING", production_eligible=False,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), binary=args.binary,
        binary_sha256=args.binary_sha256, model=MODEL, model_sha256=parity.MODEL_SHA,
        input="empty", tactics=TACTICS, warmup=args.warmup, iterations=args.iterations,
        pre_registered_cases=[case_spec(*case) for case in CASES], repetitions_per_case=1,
        metric_contract=dict(event="sum over lanes of batch / median(per-forward CUDA event seconds)",
            wall="total timed rows / forward envelope from earliest lane start to latest lane end",
            boundary="warmup, H2D/D2H and CUDA graph are outside this kernel-only measurement; neither metric is Worker/E2E throughput",
            stability="single-pass diagnostic: no stability gate, no baseline replacement, no performance certification"),
        tool_sources={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}, runs=[])
    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    def run_logged(command, stem, record):
        record["command"] = command
        def save_streams(stdout, stderr):
            for suffix, content in (("stdout", stdout), ("stderr", stderr)):
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="replace")
                path = args.output / f"{stem}.{suffix}.log"
                path.write_text(content or "", encoding="utf-8")
                record[suffix + "_log"] = str(path)
                record[suffix + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        started = time.perf_counter()
        try:
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=600)
        except subprocess.TimeoutExpired as error:
            save_streams(error.stdout, error.stderr)
            record["completion"] = "TIMEOUT"
            raise
        finally:
            record["process_elapsed_seconds"] = time.perf_counter() - started
        record["returncode"] = result.returncode
        save_streams(result.stdout, result.stderr)
        parity.require(result.returncode == 0, f"command failed: see {stem}.stderr.log")
        return result
    try:
        save()
        parity.require(parity.wsl_sha(args.binary) == args.binary_sha256, "binary SHA differs from registration")
        parity.require(parity.wsl_sha(MODEL) == parity.MODEL_SHA, "model SHA mismatch")
        inherited = parity.capture(parity.wsl("compgen -A variable KATAGO_CUDA_ || true")).splitlines()
        report["preflight"] = {}
        command = parity.wsl(shlex.join([*env_prefix(inherited), args.binary, "cuda-fingerprint", "--model", MODEL]))
        preflight = run_logged(command, "00-fingerprint", report["preflight"])
        fingerprint = json.loads(preflight.stdout)
        report["preflight"]["result"] = fingerprint
        validate_fingerprint(fingerprint)
        inputs = {}
        for index, (batch, lanes) in enumerate(CASES, 1):
            run = case_spec(batch, lanes)
            run["completion"] = "RUNNING"
            report["runs"].append(run); save()
            parity.require(parity.wsl_sha(args.binary) == args.binary_sha256, "binary changed during scan")
            command = build_command(args.binary, batch, lanes, warmup=args.warmup,
                                    iterations=args.iterations, inherited=inherited)
            print(f"diagnostic {index}/10 {run['case_id']}", flush=True)
            result = run_logged(command, f"{index:02d}-{run['case_id']}", run)
            data = parity.parse_report(result.stdout)
            run["result"] = data  # Preserve even an invalid payload before its assertions.
            metrics = validate_payload(data, result.stdout + "\n" + result.stderr, batch=batch,
                                       lanes=lanes, iterations=args.iterations, warmup=args.warmup)
            bind_input_and_build(data, metrics, batch=batch, build=fingerprint["backend_build"], inputs=inputs)
            parity.require(parity.wsl_sha(args.binary) == args.binary_sha256, "binary changed while running a case")
            run.update(completion="COMPLETE", metrics=metrics); save()
            print(f"{run['case_id']}: event sum {metrics['event_median_rate_sum']:.2f}; forward wall {metrics['forward_wall_rate']:.2f}", flush=True)
        parity.require(parity.wsl_sha(MODEL) == parity.MODEL_SHA, "model changed during scan")
        report.update(completion="COMPLETE", summary=summarize(report["runs"]))
        print(json.dumps(report["summary"], indent=2), flush=True)
    except Exception as error:
        report.update(completion="FAILED", error=str(error))
        if report["runs"] and report["runs"][-1]["completion"] == "RUNNING":
            report["runs"][-1].update(completion="FAILED", error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
