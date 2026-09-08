"""Compare two WSL Rust binaries using the Fork B14/S2 event timing contract.

The same TF3 bytes, semantic empty-board inputs and tactics are used throughout
baseline/candidate/candidate/baseline. All samples and both wall/event metrics
remain available; compilation and other GPU work must be idle during the run.
"""
import argparse
import datetime
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess

import benchmark_fork_parity as parity

MODEL = "/mnt/d/Go/Server/models/" + parity.MODEL.name
TACTICS = {"KATAGO_CUDA_DUALFFN": "1", "KATAGO_CUDA_ATTN_TILE": "q64-serial",
           "KATAGO_CUDA_NOGRAPH": "1", "KATAGO_CUDA_GEMM_LAYOUT": "tn"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="WSL path to immutable baseline binary")
    parser.add_argument("--candidate", default="/mnt/d/code/Rust_KataGo/target/cudarocmopt-wsl/release/katago-rs")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=80)
    parser.add_argument("--maximum-relative-spread", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 100 or args.warmup < 1:
        parser.error("require iterations >= 100 and warmup >= 1")
    if not math.isfinite(args.maximum_relative_spread) or not 0 <= args.maximum_relative_spread <= 0.1:
        parser.error("maximum-relative-spread must be finite in 0..0.1")
    parity.require(parity.wsl_sha(MODEL) == parity.MODEL_SHA, "TF3 model identity mismatch")
    binaries = {"baseline": args.baseline, "candidate": args.candidate}
    hashes = {name: parity.wsl_sha(path) for name, path in binaries.items()}
    parity.require(len(set(hashes.values())) == 2, "baseline and candidate binaries must differ")
    inherited = parity.capture(parity.wsl("compgen -A variable KATAGO_CUDA_ || true")).splitlines()
    unset = [item for key in inherited for item in ("-u", key)] + ["-u", "KATAGO_NN_BATCH_WINDOW_US"]
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    parity.require(not report_path.exists(), "refuse to overwrite existing benchmark evidence")
    report = dict(schema=1, status="RUNNING", utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  binaries=binaries, binary_sha256=hashes, model_sha256=parity.MODEL_SHA,
                  physical_batch=14, streams=2, iterations=args.iterations, warmup=args.warmup,
                  input="empty", tactics=TACTICS, maximum_relative_spread=args.maximum_relative_spread,
                  order=["baseline", "candidate", "candidate", "baseline"], runs=[])
    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    identity = None
    try:
        save()
        for index, flavor in enumerate(report["order"]):
            parity.require(parity.wsl_sha(binaries[flavor]) == hashes[flavor], "binary changed during ABBA")
            command = parity.wsl(shlex.join([
                "env", *unset, "LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64:/usr/lib/wsl/lib",
                *[f"{key}={value}" for key, value in TACTICS.items()], binaries[flavor],
                "nnbench", "--model", MODEL, "--mode", "kernel", "--input", "empty",
                "--batch", "14", "--handles", "1,2", "--timing", "cuda-event",
                "--warmup", str(args.warmup), "--iterations", str(args.iterations), "--json",
            ]))
            print(f"starting {index + 1}/4 {flavor}", flush=True)
            result = subprocess.run(command, text=True, capture_output=True, encoding="utf-8", timeout=600)
            stem = args.output / f"{index + 1}-{flavor}"
            stem.with_suffix(".stdout.log").write_text(result.stdout, encoding="utf-8")
            stem.with_suffix(".stderr.log").write_text(result.stderr, encoding="utf-8")
            parity.require(result.returncode == 0, f"{flavor} failed: see {stem}.stderr.log")
            data = parity.parse_report(result.stdout)
            rate = parity.rust_event_rate(data, result.stderr, iterations=args.iterations,
                                         warmup=args.warmup, input_kind="empty", model=MODEL, tactics=TACTICS)
            row = data["results"][0]
            current_identity = (row["spatial_input_sha256"], row["global_input_sha256"], data["input_description"])
            parity.require(identity is None or current_identity == identity, "input differs between binaries/runs")
            identity = current_identity
            report["runs"].append(dict(flavor=flavor, command=command, event_rate_sum=rate,
                                       wall_rate=row["wall_nn_evals_per_s"], result=data))
            save()
            print(f"{flavor}: events {rate:.1f}; wall {row['wall_nn_evals_per_s']:.1f}", flush=True)
        summary = {}
        stable = True
        for flavor in binaries:
            rates = [run["event_rate_sum"] for run in report["runs"] if run["flavor"] == flavor]
            spread = max(rates) / min(rates) - 1
            summary[flavor] = dict(samples=rates, geomean=statistics.geometric_mean(rates), relative_spread=spread)
            stable &= len(rates) == 2 and spread <= args.maximum_relative_spread
        ratio = summary["candidate"]["geomean"] / summary["baseline"]["geomean"]
        summary["candidate_over_baseline_event_ratio" if stable else "diagnostic_only_event_ratio"] = ratio
        report.update(status="STABLE" if stable else "UNSTABLE_DIAGNOSTIC_ONLY", stable=stable, summary=summary)
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
