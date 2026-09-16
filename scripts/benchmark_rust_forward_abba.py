"""Compare WSL Rust binaries/tactics using the Fork per-lane event contract.

The same TF3 bytes and semantic empty-board inputs are used throughout
baseline/candidate/candidate/baseline. B14/S2 is the default; B16/S2 is optional.
Both sides require the corrected FP16 encoding and cuBLASLt 130600. Only the
explicit residual preset may differ between tactics. All raw samples and wall
metrics remain available; compilation and other GPU work must be idle.
"""
import argparse
import datetime
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import subprocess

import benchmark_fork_parity as parity

MODEL = "/mnt/d/Go/Server/models/" + parity.MODEL.name
BASE_TACTICS = {"KATAGO_CUDA_DUALFFN": "1", "KATAGO_CUDA_ATTN_TILE": "q64-serial",
                "KATAGO_CUDA_NOGRAPH": "1", "KATAGO_CUDA_GEMM_LAYOUT": "tn"}
RESIDUAL_KEY = "KATAGO_CUDA_RESIDUAL_ALGO"
PRESET = "tf3_5070ti_r1"
INIT_PRESET = "tf3_5070ti_init_r1"
RESIDUAL_CHOICES = ("heuristic", PRESET, INIT_PRESET)


def tactics_for(residual_algo):
    parity.require(residual_algo in RESIDUAL_CHOICES, "unregistered residual algorithm")
    return {**BASE_TACTICS, RESIDUAL_KEY: residual_algo}


def validate_comparison(hashes, tactics):
    parity.require(hashes["baseline"] != hashes["candidate"]
                   or tactics["baseline"] != tactics["candidate"],
                   "baseline and candidate have the same binary SHA and effective tactics")


def build_command(binary, tactics, *, batch, warmup, iterations, inherited):
    parity.require(batch in (14, 16), "batch must be 14 or 16")
    parity.require(tactics == tactics_for(tactics.get(RESIDUAL_KEY)), "unexpected tactic configuration")
    # Explicit env assignments follow all removals, so an inherited preset can
    # neither contaminate heuristic nor silently replace the requested preset.
    keys = sorted({key for key in inherited if key.startswith("KATAGO_CUDA_")}
                  | {"KATAGO_NN_BATCH_WINDOW_US"})
    unset = [item for key in keys for item in ("-u", key)]
    return parity.wsl(shlex.join([
        "env", *unset, "LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64:/usr/lib/wsl/lib",
        *[f"{key}={value}" for key, value in tactics.items()], binary,
        "nnbench", "--model", MODEL, "--mode", "kernel", "--input", "empty",
        "--batch", str(batch), "--handles", "1,2", "--timing", "cuda-event",
        "--warmup", str(warmup), "--iterations", str(iterations), "--json",
    ]))


def residual_markers(log, *, batch, residual_algo):
    parity.require(batch in (14, 16), "batch must be 14 or 16")
    parity.require(residual_algo in RESIDUAL_CHOICES, "unregistered residual algorithm")
    expected = ({("5054", "384", "1152", "1")} if batch == 14 else
                {("5776", "384", "1152", "3"), ("5776", "384", "384", "2")})
    seen, markers = set(), []
    for line in log.splitlines():
        if not line.startswith("[cuda-tactic] ") or not re.search(r"(?:^|\s)name=residual_algo(?:\s|$)", line):
            continue
        parity.require(residual_algo != "heuristic", "heuristic run executed a residual preset")
        pairs = re.findall(r"(?:^|\s)([A-Za-z0-9_]+)=([^\s]+)", line)
        fields = dict(pairs)
        parity.require(len(fields) == len(pairs), "duplicate residual marker fields")
        required = dict(engine=residual_algo, layout="tn", output="f32", beta="1",
                        cublaslt_version="130600", algorithm_identity="matched")
        if residual_algo == INIT_PRESET:
            required["construction"] = "public_init"
            # This number identifies where the experiment was registered. The
            # public-init implementation does not select a heuristic pool entry.
            index_key = "source_filtered_index"
            parity.require("filtered_index" not in fields,
                           "public-init marker must not claim a runtime heuristic filtered_index")
        else:
            index_key = "filtered_index"
            parity.require("construction" not in fields and "source_filtered_index" not in fields,
                           "fixed heuristic marker must not claim public-init construction or source index")
        parity.require(all(fields.get(key) == value for key, value in required.items()),
                       f"residual marker contract mismatch: {line}")
        shape = tuple(fields.get(key) for key in ("m", "n", "k", index_key))
        parity.require(shape in expected, f"unexpected residual shape/index: {shape}")
        seen.add(shape)
        markers.append(fields)
    if residual_algo != "heuristic":
        parity.require(seen == expected, f"missing residual marker(s): {sorted(expected - seen)}")
    return markers


def validate_result(data, log, *, batch, iterations, warmup, tactics):
    """Same event arithmetic as G0, parameterized locally to leave G0 unchanged."""
    parity.require(data.get("command") == "nnbench" and data.get("mode") == "kernel"
                   and data.get("timing") == "cuda-event" and data.get("kernel_only") is True,
                   "Rust benchmark mode/timing mismatch")
    parity.require(parity.same_path(data.get("model"), MODEL) and data.get("model_sha256") == parity.MODEL_SHA,
                   "Rust model path/hash mismatch")
    parity.require(data.get("input") == "empty" and data.get("warmup") == warmup
                   and data.get("iterations") == iterations, "Rust input/warmup/iteration mismatch")
    parity.require(data.get("requested_batches") == [batch] and data.get("handles") == [1, 2],
                   "Rust batch/handle request mismatch")
    parity.require(data.get("device_target") == "sm_120", "Rust device target mismatch")
    build = data.get("build", {})
    parity.require(isinstance(build.get("cuda_compiler"), str) and build["cuda_compiler"],
                   "Rust CUDA compiler provenance missing")
    parity.require(type(build.get("fp16_encoding_revision")) is int and build["fp16_encoding_revision"] == 1,
                   "Rust corrected fp16_encoding_revision=1 required")
    parity.require(type(build.get("cublaslt_version")) is int and build["cublaslt_version"] == 130600,
                   "Rust cublaslt_version=130600 required")
    actual_tactics = {key: value for key, value in data.get("tactics", {}).items() if value is not None}
    parity.require(actual_tactics == tactics, "Rust effective tactics differ from requested tactics")
    description = data.get("input_description", {})
    parity.require(description.get("kind") == "empty" and description.get("inputs_version") == 7
                   and description.get("board") == [19, 19] and description.get("rules") == parity.EMPTY_RULES
                   and description.get("encore_phase") == 0 and description.get("effective_symmetry") == 0
                   and description.get("policy_optimism") == 0.0 and description.get("misc_params") == parity.EMPTY_MISC,
                   "Rust empty-board input description mismatch")
    rows = data.get("results", [])
    parity.require(len(rows) == 1 and rows[0].get("batch") == batch, "Rust result batch mismatch")
    row = rows[0]
    for key in ("spatial_input_sha256", "global_input_sha256"):
        parity.require(isinstance(row.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", row[key]),
                       f"Rust {key} missing/invalid")
    handles = row.get("handles", [])
    parity.require([handle.get("handle") for handle in handles] == [1, 2], "Rust result lane IDs mismatch")
    rates = []
    for handle in handles:
        values = handle.get("cuda_event_ms", [])
        parity.require(handle.get("iterations") == iterations and len(values) == iterations,
                       "Rust lane event sample count mismatch")
        median = statistics.median([parity.positive(value, "Rust event time") for value in values])
        parity.matching_rate(handle.get("cuda_event_median_ms"), median, "Rust lane median")
        rate = batch * 1000.0 / median
        parity.matching_rate(handle.get("cuda_event_median_nn_evals_per_s"), rate, "Rust lane median-rate")
        rates.append(rate)
    parity.matching_rate(row.get("sum_median_nn_evals_per_s"), sum(rates), "Rust median-rate sum")
    parity.positive(row.get("wall_nn_evals_per_s"), "Rust wall throughput")
    for pattern, label in [(r"name=gemm_layout launch=tn(?:\s|$)", "TN layout"),
                           (r"name=dual_ffn launch=fused(?:\s|$)", "DualFFN"),
                           (r"tile=q64-serial(?:\s|$)", "serial attention")]:
        parity.require(re.search(pattern, log), f"Rust {label} did not execute")
    markers = residual_markers(log, batch=batch, residual_algo=tactics[RESIDUAL_KEY])
    return sum(rates), markers


def summarize(runs, maximum_relative_spread):
    summary = {}
    stable = [run["flavor"] for run in runs] == ["baseline", "candidate", "candidate", "baseline"]
    for flavor in ("baseline", "candidate"):
        rates = [parity.positive(run["event_rate_sum"], "run rate") for run in runs if run["flavor"] == flavor]
        parity.require(rates, f"no {flavor} samples")
        spread = max(rates) / min(rates) - 1
        summary[flavor] = dict(samples=rates, geomean=statistics.geometric_mean(rates), relative_spread=spread)
        stable &= len(rates) == 2 and spread <= maximum_relative_spread
    ratio = summary["candidate"]["geomean"] / summary["baseline"]["geomean"]
    summary["candidate_over_baseline_event_ratio" if stable else "diagnostic_only_event_ratio"] = ratio
    return stable, summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="WSL path to immutable baseline binary")
    parser.add_argument("--candidate", default="/mnt/d/code/Rust_KataGo/target/cudarocmopt-wsl/release/katago-rs")
    parser.add_argument("--baseline-residual-algo", choices=RESIDUAL_CHOICES, default="heuristic")
    parser.add_argument("--candidate-residual-algo", choices=RESIDUAL_CHOICES, default="heuristic")
    parser.add_argument("--batch", type=int, choices=(14, 16), default=14)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=80)
    parser.add_argument("--maximum-relative-spread", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.iterations < 100 or args.warmup < 1:
        parser.error("require iterations >= 100 and warmup >= 1")
    if not math.isfinite(args.maximum_relative_spread) or not 0 <= args.maximum_relative_spread <= 0.1:
        parser.error("maximum-relative-spread must be finite in 0..0.1")
    return args


def main():
    args = parse_args()
    parity.require(parity.wsl_sha(MODEL) == parity.MODEL_SHA, "TF3 model identity mismatch")
    binaries = {"baseline": args.baseline, "candidate": args.candidate}
    hashes = {name: parity.wsl_sha(path) for name, path in binaries.items()}
    tactics = {"baseline": tactics_for(args.baseline_residual_algo),
               "candidate": tactics_for(args.candidate_residual_algo)}
    validate_comparison(hashes, tactics)
    inherited = parity.capture(parity.wsl("compgen -A variable KATAGO_CUDA_ || true")).splitlines()
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    parity.require(not report_path.exists(), "refuse to overwrite existing benchmark evidence")
    report = dict(schema=2, status="RUNNING", utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  binaries=binaries, binary_sha256=hashes, model_sha256=parity.MODEL_SHA,
                  physical_batch=args.batch, streams=2, iterations=args.iterations, warmup=args.warmup,
                  input="empty", tactics=tactics, maximum_relative_spread=args.maximum_relative_spread,
                  required_fp16_encoding_revision=1, required_cublaslt_version=130600,
                  comparison_metric="sum of per-lane batch / median(CUDA event forward seconds)",
                  path_caveat="One-shot residual markers establish executed shapes/algorithms, not per-lane execution counts.",
                  residual_index_contract={
                      PRESET: "filtered_index is the actual workspace-filtered heuristic selection",
                      INIT_PRESET: "source_filtered_index records candidate provenance; construction uses public AlgoInit/configuration APIs",
                  },
                  order=["baseline", "candidate", "candidate", "baseline"], runs=[])
    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    identity = None
    builds = {}
    try:
        save()
        for index, flavor in enumerate(report["order"]):
            parity.require(parity.wsl_sha(binaries[flavor]) == hashes[flavor], "binary changed during ABBA")
            command = build_command(binaries[flavor], tactics[flavor], batch=args.batch,
                                    warmup=args.warmup, iterations=args.iterations, inherited=inherited)
            print(f"starting {index + 1}/4 {flavor}", flush=True)
            result = subprocess.run(command, text=True, capture_output=True, encoding="utf-8", timeout=600)
            stem = args.output / f"{index + 1}-{flavor}"
            stem.with_suffix(".stdout.log").write_text(result.stdout, encoding="utf-8")
            stem.with_suffix(".stderr.log").write_text(result.stderr, encoding="utf-8")
            parity.require(result.returncode == 0, f"{flavor} failed: see {stem}.stderr.log")
            data = parity.parse_report(result.stdout)
            rate, markers = validate_result(data, result.stdout + "\n" + result.stderr, batch=args.batch,
                                            iterations=args.iterations, warmup=args.warmup, tactics=tactics[flavor])
            build_key = hashes[flavor]
            parity.require(build_key not in builds or builds[build_key] == data["build"],
                           "same binary reported a different build/library fingerprint")
            builds[build_key] = data["build"]
            row = data["results"][0]
            current_identity = (row["spatial_input_sha256"], row["global_input_sha256"], data["input_description"])
            parity.require(identity is None or current_identity == identity, "input differs between binaries/runs")
            identity = current_identity
            report["runs"].append(dict(flavor=flavor, command=command, effective_tactics=tactics[flavor],
                                       residual_markers=markers, event_rate_sum=rate,
                                       wall_rate=row["wall_nn_evals_per_s"], result=data))
            save()
            print(f"{flavor}: events {rate:.1f}; wall {row['wall_nn_evals_per_s']:.1f}", flush=True)
        stable, summary = summarize(report["runs"], args.maximum_relative_spread)
        report.update(status="STABLE" if stable else "UNSTABLE_DIAGNOSTIC_ONLY", stable=stable, summary=summary)
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
