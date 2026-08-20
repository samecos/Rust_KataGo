"""Run the comparable C++ CUDA/cuDNN and Rust CUDA batch matrix.

The C++ executable consumes KataGo's native ``.bin.gz`` model and exposes
``cudaUseFP16``. The Rust CUDA executor currently consumes the exported ONNX
model and is a mixed-precision FP16-storage/FP32-accumulate implementation;
there is no Rust FP32 mode, so the report records that as unsupported instead
of silently labelling the Rust result FP32.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def iterations_for(batch: int) -> int:
    # Match C++ benchmarknn's default timed iteration count for every batch.
    # This makes the medians directly comparable and keeps high-batch samples
    # long enough to average WDDM launch jitter.
    _ = batch
    return 200


def parse_json_stdout(stdout: str, label: str) -> dict:
    # C++ emits compact JSON; Rust emits pretty-printed JSON. Accept either,
    # while ignoring terminal control sequences that can be injected by a PTY.
    cleaned = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", stdout)
    start = cleaned.find("{")
    if start >= 0:
        try:
            value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    raise RuntimeError(f"{label}: no JSON object in stdout\n{stdout[-2000:]}")


def run(cmd: list[str], env: dict[str, str] | None, label: str, out_dir: Path) -> dict:
    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    startup_s = time.perf_counter() - started
    (out_dir / f"{label}.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (out_dir / f"{label}.stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"{label}: exit {proc.returncode}\n{proc.stderr[-4000:]}"
        )
    report = parse_json_stdout(proc.stdout, label)
    report["process_wall_s"] = startup_s
    return report


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpp-binary",
        type=Path,
        default=Path(r"D:\ExperimentalCode\KataGo\cpp\build-cuda\Release\katago.exe"),
    )
    parser.add_argument(
        "--cpp-config",
        type=Path,
        default=Path(r"D:\ExperimentalCode\KataGo\cpp\configs\lizzie_5070ti_wsl.cfg"),
    )
    parser.add_argument(
        "--cpp-model",
        type=Path,
        default=Path(
            r"D:\ExperimentalCode\KataGo\cpp\models\b11c768h12nbt3tflrs-fson-silu.bin.gz"
        ),
    )
    parser.add_argument(
        "--rust-binary", type=Path, default=root / "target" / "release" / "katago-rs.exe"
    )
    parser.add_argument("--rust-model", type=Path, default=Path(r"D:\code\b11fix.onnx"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "target" / "cudnn-benchmark" / "batch-matrix-20260820",
    )
    parser.add_argument(
        "--rust-only",
        action="store_true",
        help="skip the expensive C++ matrix and append/refresh Rust records",
    )
    parser.add_argument(
        "--cpp-only",
        action="store_true",
        help="run C++ cases only; useful when rechecking a backend variant",
    )
    parser.add_argument(
        "--cpp-precisions",
        default="fp32,fp16",
        help="comma-separated C++ precisions to run (default fp32,fp16)",
    )
    parser.add_argument(
        "--cudnn-sdpa",
        action="store_true",
        help="disable the C++ custom MMA attention path and require cuDNN SDPA",
    )
    args = parser.parse_args()
    for path in (args.cpp_binary, args.cpp_config, args.cpp_model, args.rust_binary, args.rust_model):
        if not path.exists():
            parser.error(f"missing path: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    existing_summary = args.output_dir / "summary.json"
    records: list[dict] = []
    if args.rust_only:
        # The C++ matrix may already have completed before a Rust-only retry.
        # Reconstruct its records from the immutable per-case stdout logs so a
        # parser failure in one Rust case cannot erase the C++ evidence.
        for path in sorted(args.output_dir.glob("cpp-*.stdout.log")):
            report = parse_json_stdout(path.read_text(encoding="utf-8"), path.stem)
            precision = "fp16" if report["usingFP16"] else "fp32"
            records.append(
                {
                    **report,
                    "impl": "cpp-cudnn",
                    "precision": precision,
                    "batch": report["batchSize"],
                }
            )
    cpp_common = [
        str(args.cpp_binary),
        "benchmarknn",
        "-config",
        str(args.cpp_config),
        "-model",
        str(args.cpp_model),
        "-warmup",
        "2",
        "-json",
        "-override-config",
        "cudaUseFusedFFN=false,cudaUseMmaAttention="
        + ("false" if args.cudnn_sdpa else "true")
        + ",numNNServerThreadsPerModel=1,"
        f"homeDataDir={args.output_dir / 'cpp-cache'}",
    ]
    requested_precisions = [p.strip() for p in args.cpp_precisions.split(",") if p.strip()]
    precision_values = {"fp32": "false", "fp16": "true"}
    unknown = [p for p in requested_precisions if p not in precision_values]
    if unknown:
        parser.error(f"unknown --cpp-precisions value(s): {unknown}")
    for precision in (() if args.rust_only else requested_precisions):
        use_fp16 = precision_values[precision]
        for batch in BATCHES:
            label = f"cpp-{precision}-b{batch}"
            cmd = cpp_common + ["-batch-size", str(batch)]
            cmd[-1:] = [str(batch)]
            # benchmarknn accepts -batch-size before or after the override.
            cmd = cpp_common[:8] + ["-batch-size", str(batch)] + cpp_common[8:]
            cmd[-1] = (
                "cudaUseFP16="
                + use_fp16
                + ",cudaUseFusedFFN=false,cudaUseMmaAttention="
                + ("false" if args.cudnn_sdpa else "true")
                + ",numNNServerThreadsPerModel=1,"
                + f"homeDataDir={args.output_dir / 'cpp-cache'}"
            )
            report = run(cmd, None, label, args.output_dir)
            report.update({"impl": "cpp-cudnn", "precision": precision, "batch": batch})
            records.append(report)
            print(
                f"PASS {label}: median_ms={report['perThreadMedianMs'][0]:.4f} "
                f"eval_s={report['perThreadNNEvalsPerSec'][0]:.2f} "
                f"process_s={report['process_wall_s']:.1f}"
            )

    if args.cpp_only:
        summary = {
            "schema": 1,
            "date": "2026-08-20",
            "gpu_condition": "RTX 5070 Ti / SM120 / driver 610.88 / CUDA 13.3",
            "board": "19x19",
            "batches": BATCHES,
            "cpp_model": str(args.cpp_model),
            "rust_model": str(args.rust_model),
            "cpp_fused_ffn": False,
            "cpp_cudnn_sdpa": args.cudnn_sdpa,
            "rust_tactics": {"KATAGO_CUDA_DUALFFN": "1", "KATAGO_CUDA_ATTN_TILE": "q64"},
            "rust_fp32_supported": False,
            "records": records,
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"summary: {args.output_dir / 'summary.json'}")
        return 0

    rust_env = dict(__import__("os").environ)
    rust_env.update(
        {
            "KATAGO_CUDA_DUALFFN": "1",
            "KATAGO_CUDA_ATTN_TILE": "q64",
            "KATAGO_CUDA_NOGRAPH": "1",
        }
    )
    for batch in BATCHES:
        iterations = iterations_for(batch)
        label = f"rust-fp16-mixed-b{batch}"
        cmd = [
            str(args.rust_binary),
            "nnbench",
            "--model",
            str(args.rust_model),
            "--override-config",
            "nnBackend=cudabackend,cudaDisableWarmup=true",
            "--mode",
            "direct",
            "--batch",
            str(batch),
            "--warmup",
            "2",
            "--iterations",
            str(iterations),
            "--json",
        ]
        report = run(cmd, rust_env, label, args.output_dir)
        result = report["results"][0]
        handle = result["handles"][0]
        report.update(
            {
                "impl": "rust-cuda",
                "precision": "fp16-storage-fp32-accum",
                "batch": batch,
                "median_ms": handle["per_batch_ms"],
                "nn_evals_per_s": handle["nn_evals_per_s"],
            }
        )
        records.append(report)
        print(
            f"PASS {label}: per_batch_ms={handle['per_batch_ms']:.4f} "
            f"eval_s={handle['nn_evals_per_s']:.2f} "
            f"process_s={report['process_wall_s']:.1f}"
        )

    summary = {
        "schema": 1,
        "date": "2026-08-20",
        "gpu_condition": "RTX 5070 Ti / SM120 / driver 610.88 / CUDA 13.3",
        "board": "19x19",
        "batches": BATCHES,
        "cpp_model": str(args.cpp_model),
        "rust_model": str(args.rust_model),
        "cpp_fused_ffn": False,
        "rust_tactics": {"KATAGO_CUDA_DUALFFN": "1", "KATAGO_CUDA_ATTN_TILE": "q64"},
        "rust_fp32_supported": False,
        "records": records,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"summary: {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
