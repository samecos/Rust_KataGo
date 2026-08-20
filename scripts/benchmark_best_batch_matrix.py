"""Benchmark the best available per-batch CUDA throughput on C++ and Rust.

The C++ executable is the original KataGo checkout.  For each batch the
runner tests one and two NN server threads (two FP32 threads are skipped for
large batches because they exceed the 16 GiB device budget).  Rust tests one
and two direct CUDA handles.  Selection uses measured wall throughput, not a
sum of per-thread medians.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def parse_json(text: str, label: str) -> dict:
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    start = text.find("{")
    if start < 0:
        raise RuntimeError(f"{label}: no JSON object in stdout")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, dict):
        raise RuntimeError(f"{label}: JSON root is not an object")
    return value


def run(cmd: list[str], label: str, out_dir: Path, timeout: int = 1800) -> dict:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    (out_dir / f"{label}.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (out_dir / f"{label}.stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"{label}: exit {proc.returncode}\n{proc.stderr[-3000:]}")
    return parse_json(proc.stdout, label)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp-binary", type=Path, default=Path(r"D:\ExperimentalCode\KataGo\cpp\build-cuda\Release\katago.exe"))
    parser.add_argument("--cpp-config", type=Path, default=Path(r"D:\ExperimentalCode\KataGo\cpp\configs\lizzie_5070ti_wsl.cfg"))
    parser.add_argument("--cpp-model", type=Path, default=Path(r"D:\ExperimentalCode\KataGo\cpp\models\b11c768h12nbt3tflrs-fson-silu.bin.gz"))
    parser.add_argument("--rust-binary", type=Path, default=root / "target" / "release" / "katago-rs.exe")
    parser.add_argument("--rust-model", type=Path, default=Path(r"D:\code\b11fix.onnx"))
    parser.add_argument("--output-dir", type=Path, default=root / "target" / "cudnn-benchmark" / "batch-matrix-best-20260820")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    for path in (args.cpp_binary, args.cpp_config, args.cpp_model, args.rust_binary, args.rust_model):
        if not path.exists():
            parser.error(f"missing path: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cpp_records: list[dict] = []
    for precision, use_fp16 in (("fp32", "false"), ("fp16", "true")):
        for batch in BATCHES:
            thread_candidates = [1, 2]
            if precision == "fp32" and batch >= 256:
                # Two B256/B512 FP32 workspaces do not fit reliably in 16 GiB.
                thread_candidates = [1]
            for threads in thread_candidates:
                label = f"cpp-{precision}-b{batch}-t{threads}"
                cmd = [
                    str(args.cpp_binary), "benchmarknn",
                    "-config", str(args.cpp_config),
                    "-model", str(args.cpp_model),
                    "-warmup", "2", "-iterations", str(args.iterations),
                    "-batch-size", str(batch), "-json",
                    "-override-config",
                    "cudaUseFP16=" + use_fp16
                    + ",cudaUseFusedFFN=false,cudaUseMmaAttention="
                    + ("true" if precision == "fp16" else "false")
                    + ",numNNServerThreadsPerModel=" + str(threads)
                    + ",homeDataDir=" + str(args.output_dir / "cpp-cache"),
                ]
                try:
                    report = run(cmd, label, args.output_dir)
                except Exception as exc:
                    print(f"SKIP {label}: {exc}")
                    continue
                report.update({"impl": "cpp-original", "precision": precision, "batch": batch, "threads": threads})
                cpp_records.append(report)
                print(f"PASS {label}: wall_eval_s={report['actualWallNNEvalsPerSec']:.2f}")

    rust_records: list[dict] = []
    rust_env = dict(os.environ)
    rust_env.update({"KATAGO_CUDA_DUALFFN": "1", "KATAGO_CUDA_ATTN_TILE": "q64", "KATAGO_CUDA_NOGRAPH": "1"})
    for handles in ("1", "1,2"):
        label = f"rust-fp16-mixed-h{handles.replace(',', '-') }"
        cmd = [
            str(args.rust_binary), "nnbench", "--model", str(args.rust_model),
            "--override-config", "nnBackend=cudabackend,cudaDisableWarmup=true",
            "--mode", "direct", "--handles", handles,
            "--batch", ",".join(map(str, BATCHES)), "--warmup", "2",
            "--iterations", str(args.iterations), "--json",
        ]
        report = run(cmd, label, args.output_dir)
        for row in report["results"]:
            row = {**row, "impl": "rust-cuda", "precision": "fp16-storage-fp32-accum", "handles": handles}
            rust_records.append(row)
            print(f"PASS {label} b{row['batch']}: wall_eval_s={row['wall_nn_evals_per_s']:.2f}")

    best: list[dict] = []
    for precision in ("fp32", "fp16"):
        for batch in BATCHES:
            candidates = [r for r in cpp_records if r["precision"] == precision and r["batch"] == batch]
            if not candidates:
                continue
            chosen = max(candidates, key=lambda r: r["actualWallNNEvalsPerSec"])
            row = {
                "batch": batch,
                "cpp_precision": precision,
                "cpp_threads": chosen["threads"],
                "cpp_wall_evals_per_s": chosen["actualWallNNEvalsPerSec"],
                "cpp_median_evals_per_s": chosen["sumMedianNNEvalsPerSec"],
            }
            if precision == "fp16":
                rust = max((r for r in rust_records if r["batch"] == batch), key=lambda r: r["wall_nn_evals_per_s"])
                row.update({"rust_handles": rust["handles"], "rust_wall_evals_per_s": rust["wall_nn_evals_per_s"], "rust_over_cpp": rust["wall_nn_evals_per_s"] / chosen["actualWallNNEvalsPerSec"]})
            best.append(row)
    summary = {"schema": 1, "batches": BATCHES, "iterations": args.iterations, "cpp_records": cpp_records, "rust_records": rust_records, "best": best, "rust_fp32_supported": False}
    (args.output_dir / "best-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary: {args.output_dir / 'best-summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
