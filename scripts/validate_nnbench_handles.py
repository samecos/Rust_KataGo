"""Validate synchronized multi-handle CUDA nnbench measurements.

Runs the release ``nnbench`` direct/kernel paths repeatedly, preserves the raw
JSON and stderr for every run, and checks two properties:

* one-handle wall throughput agrees with its per-handle throughput within 1%;
* repeated two-handle wall throughput has coefficient of variation below 1%.

This validates the benchmark tool and concurrent CUDA resource ownership. It
does not claim that multiple handles improve the production eval/search path.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def parse_csv(value: str) -> list[str]:
    values = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    return values


def parse_handle_sets(value: str) -> list[str]:
    # Semicolon separates test cases; commas identify handles within a case.
    values = [piece.strip() for piece in value.split(";") if piece.strip()]
    for handles in values:
        ids = parse_csv(handles)
        if any(not item.isdigit() or int(item) <= 0 for item in ids):
            raise argparse.ArgumentTypeError(f"invalid handle set: {handles}")
    if not values:
        raise argparse.ArgumentTypeError("handle set list must not be empty")
    return values


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary",
        type=Path,
        default=ROOT / "target" / "release" / "katago-rs.exe",
    )
    parser.add_argument("--model", type=Path, default=Path("D:/code/b11fix.onnx"))
    parser.add_argument("--batches", default="1,16")
    parser.add_argument("--modes", default="direct,kernel")
    parser.add_argument(
        "--handle-sets",
        default="1;1,2",
        help="semicolon-separated cases, for example '1;1,2'",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--tactic",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="CUDA tactic environment override; may be repeated",
    )
    parser.add_argument("--max-cv", type=float, default=0.01)
    parser.add_argument("--max-single-delta", type=float, default=0.01)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "target"
        / "cudarocmopt-validation"
        / f"v5-handles-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    args = parser.parse_args()

    binary = args.binary.resolve()
    model = args.model.resolve()
    batches = parse_csv(args.batches)
    modes = parse_csv(args.modes)
    handle_sets = parse_handle_sets(args.handle_sets)
    if any(mode not in {"direct", "kernel"} for mode in modes):
        parser.error("--modes supports only direct,kernel")
    if args.repeats < 2 or args.warmup < 2 or args.iterations <= 0:
        parser.error("repeats >=2, warmup >=2, and iterations >0 are required")
    if not binary.exists() or not model.exists():
        parser.error(f"missing binary/model: {binary}, {model}")

    tactic_env: dict[str, str] = {}
    for item in args.tactic:
        if "=" not in item:
            parser.error(f"invalid --tactic {item!r}, expected KEY=VALUE")
        key, value = item.split("=", 1)
        if not key.startswith("KATAGO_CUDA_") or not value:
            parser.error(f"invalid --tactic {item!r}")
        tactic_env[key] = value

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for mode in modes:
        for handles in handle_sets:
            label = handles.replace(",", "-")
            for repeat in range(1, args.repeats + 1):
                cmd = [
                    str(binary),
                    "nnbench",
                    "--model",
                    str(model),
                    "--override-config",
                    "nnBackend=cudabackend",
                    "--mode",
                    mode,
                    "--batch",
                    ",".join(batches),
                    "--handles",
                    handles,
                    "--warmup",
                    str(args.warmup),
                    "--iterations",
                    str(args.iterations),
                    "--json",
                ]
                env = os.environ.copy()
                env.update(tactic_env)
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=900,
                    env=env,
                )
                stem = f"{mode}-h{label}-r{repeat}"
                (out_dir / f"{stem}.stderr.log").write_text(
                    proc.stderr, encoding="utf-8"
                )
                (out_dir / f"{stem}.stdout.json").write_text(
                    proc.stdout, encoding="utf-8"
                )
                if proc.returncode != 0:
                    print(f"FAIL {stem}: exit {proc.returncode}")
                    print(proc.stderr.rstrip())
                    return 1
                try:
                    report = json.loads(proc.stdout)
                except json.JSONDecodeError as error:
                    print(f"FAIL {stem}: stdout is not JSON: {error}")
                    return 1
                records.append(
                    {
                        "mode": mode,
                        "handles": handles,
                        "repeat": repeat,
                        "report": report,
                    }
                )
                rates = ", ".join(
                    f"B{result['batch']}={result['wall_nn_evals_per_s']:.1f}"
                    for result in report["results"]
                )
                print(f"PASS {stem}: {rates}")

    failures: list[str] = []
    groups: list[dict] = []
    for mode in modes:
        for handles in handle_sets:
            matching = [
                record
                for record in records
                if record["mode"] == mode and record["handles"] == handles
            ]
            for batch in map(int, batches):
                results = [
                    next(
                        result
                        for result in record["report"]["results"]
                        if result["batch"] == batch
                    )
                    for record in matching
                ]
                rates = [result["wall_nn_evals_per_s"] for result in results]
                mean = statistics.fmean(rates)
                cv = statistics.stdev(rates) / mean
                group = {
                    "mode": mode,
                    "handles": handles,
                    "batch": batch,
                    "samples": rates,
                    "geometric_mean": geometric_mean(rates),
                    "coefficient_of_variation": cv,
                }
                if len(parse_csv(handles)) == 1:
                    deltas = []
                    for result in results:
                        per_handle = result["handles"][0]["nn_evals_per_s"]
                        deltas.append(abs(result["wall_nn_evals_per_s"] / per_handle - 1.0))
                    group["single_handle_wall_deltas"] = deltas
                    if max(deltas) > args.max_single_delta:
                        failures.append(
                            f"{mode} handles={handles} B{batch} max wall/per-handle "
                            f"delta {max(deltas):.3%} > {args.max_single_delta:.3%}"
                        )
                elif cv > args.max_cv:
                    failures.append(
                        f"{mode} handles={handles} B{batch} CV {cv:.3%} > "
                        f"{args.max_cv:.3%}"
                    )
                groups.append(group)
                print(
                    f"STAT {mode:6} handles={handles:3} B{batch:<2} "
                    f"gmean={group['geometric_mean']:.1f} CV={cv:.3%}"
                )

    summary = {
        "schema": 1,
        "binary": str(binary),
        "model": str(model),
        "batches": list(map(int, batches)),
        "modes": modes,
        "handle_sets": handle_sets,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "tactics": tactic_env,
        "thresholds": {
            "max_cv": args.max_cv,
            "max_single_delta": args.max_single_delta,
        },
        "groups": groups,
        "failures": failures,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"logs: {out_dir}")
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        print(f"RESULT: FAIL ({len(failures)})")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
