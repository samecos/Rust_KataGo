"""Summarize the C++ cuDNN vs Rust CUDA b11 batch matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def load_cpp(directory: Path, prefix: str) -> dict[int, dict]:
    result = {}
    for path in directory.glob(f"{prefix}-b*.stdout.log"):
        report = json.loads(path.read_text(encoding="utf-8"))
        result[int(report["batchSize"])] = report
    missing = [b for b in BATCHES if b not in result]
    if missing:
        raise SystemExit(f"missing C++ records in {directory}: {missing}")
    return result


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fp32-dir",
        type=Path,
        default=root / "target" / "cudnn-benchmark" / "batch-matrix-20260820",
    )
    parser.add_argument(
        "--fp16-dir",
        type=Path,
        default=root
        / "target"
        / "cudnn-benchmark"
        / "batch-matrix-cudnn-sdpa-final-20260820",
    )
    parser.add_argument(
        "--rust-summary",
        type=Path,
        default=root / "target" / "cudnn-benchmark" / "batch-matrix-20260820" / "summary.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    fp32 = load_cpp(args.fp32_dir, "cpp-fp32")
    fp16 = load_cpp(args.fp16_dir, "cpp-fp16")
    rust_summary = json.loads(args.rust_summary.read_text(encoding="utf-8"))
    rust = {
        int(row["batch"]): row
        for row in rust_summary["records"]
        if row.get("impl") == "rust-cuda"
    }
    missing = [b for b in BATCHES if b not in rust]
    if missing:
        raise SystemExit(f"missing Rust records: {missing}")

    rows = []
    print("batch | cpp FP32 ms/eval/s | cpp cuDNN FP16 ms/eval/s | rust mixed ms/eval/s | rust/cpp")
    print("---:|---:|---:|---:|---:")
    ratios = []
    for batch in BATCHES:
        a = fp32[batch]
        b = fp16[batch]
        r = rust[batch]
        cpp32_rate = a["perThreadNNEvalsPerSec"][0]
        cpp16_rate = b["perThreadNNEvalsPerSec"][0]
        rust_rate = r["nn_evals_per_s"]
        ratio = rust_rate / cpp16_rate
        ratios.append(ratio)
        row = {
            "batch": batch,
            "cpp_fp32_median_ms": a["perThreadMedianMs"][0],
            "cpp_fp32_evals_per_s": cpp32_rate,
            "cpp_cudnn_fp16_median_ms": b["perThreadMedianMs"][0],
            "cpp_cudnn_fp16_evals_per_s": cpp16_rate,
            "rust_mixed_median_ms": r["median_ms"],
            "rust_mixed_evals_per_s": rust_rate,
            "rust_over_cpp_cudnn_fp16": ratio,
        }
        rows.append(row)
        print(
            f"{batch} | {row['cpp_fp32_median_ms']:.3f}/{cpp32_rate:.2f} "
            f"| {row['cpp_cudnn_fp16_median_ms']:.3f}/{cpp16_rate:.2f} "
            f"| {row['rust_mixed_median_ms']:.3f}/{rust_rate:.2f} "
            f"| {ratio:.4f}"
        )

    cpp_speedups = [
        fp16[b]["perThreadNNEvalsPerSec"][0] / fp32[b]["perThreadNNEvalsPerSec"][0]
        for b in BATCHES
    ]
    summary = {
        "schema": 1,
        "batches": BATCHES,
        "rows": rows,
        "geomean_rust_over_cpp_cudnn_fp16": math.exp(
            sum(math.log(x) for x in ratios) / len(ratios)
        ),
        "geomean_cpp_cudnn_fp16_over_fp32": math.exp(
            sum(math.log(x) for x in cpp_speedups) / len(cpp_speedups)
        ),
        "rust_fp32_supported": False,
        "precision_note": "Rust cudabackend is FP16 storage with FP32 accumulation; no Rust FP32 mode or cuDNN backend exists.",
    }
    output = args.output or (args.fp16_dir / "comparison.json")
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"geomean Rust/C++ cuDNN FP16: {summary['geomean_rust_over_cpp_cudnn_fp16']:.4f}")
    print(f"geomean C++ cuDNN FP16/FP32: {summary['geomean_cpp_cudnn_fp16_over_fp32']:.4f}")
    print(f"comparison: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
