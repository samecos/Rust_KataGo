#!/usr/bin/env python3
"""Same-device inference benchmark for the experimental Go student models.

This measures two PyTorch models on identical 19x19 archive inputs.  It does
not measure KataGo search, the Rust CUDA backend, or playing strength.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import statistics
import sys
import time
from typing import Any
import zipfile

import numpy as np
import torch

import audit_npz
import student_models


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "data" / "2026-08-20npzs"
VARIANTS = ("dense", "compact")
MODES = ("forward_only", "packed_to_cpu_outputs")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def abba_order(round_index: int) -> tuple[str, str, str, str]:
    """Alternate ABBA and BAAB to balance which model starts a round."""
    return ("dense", "compact", "compact", "dense") if round_index % 2 == 0 else (
        "compact", "dense", "dense", "compact"
    )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is false")
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def load_input_rows(data_dir: Path, count: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Read actual full-board v7 inputs from the already extracted sample."""
    if count <= 0:
        raise ValueError("count must be positive")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"sample directory missing: {data_dir}")
    paths = sorted(data_dir.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no NPZ files in {data_dir}")
    packed_parts: list[np.ndarray] = []
    global_parts: list[np.ndarray] = []
    sources: list[dict[str, Any]] = []
    collected = 0
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            packed = audit_npz._read_member(archive, "binaryInputNCHWPacked", 512 * 1024 * 1024)
            global_input = audit_npz._read_member(archive, "globalInputNC", 512 * 1024 * 1024)
        for field, value in (("binaryInputNCHWPacked", packed), ("globalInputNC", global_input)):
            error = audit_npz._validate_array(field, value, len(packed))
            if error:
                raise ValueError(f"{path}: {error}")
        onboard = np.unpackbits(packed[:, 0, :], axis=1, bitorder="big")[:, :audit_npz.AREA]
        full_board = np.all(onboard == 1, axis=1)
        selected = np.flatnonzero(full_board)[: count - collected]
        if len(selected):
            packed_parts.append(packed[selected].copy())
            global_parts.append(global_input[selected].copy())
            sources.append({
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "selected_rows": int(len(selected)),
            })
            collected += len(selected)
        if collected >= count:
            break
    if collected < count:
        raise ValueError(f"only {collected} full-board rows in {data_dir}; need {count}")
    packed_rows = np.concatenate(packed_parts, axis=0)
    global_rows = np.concatenate(global_parts, axis=0)
    if not np.isfinite(global_rows).all():
        raise ValueError("selected global inputs contain non-finite values")
    return packed_rows, global_rows, sources


def load_input_pack(path: Path, count: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Load a small, portable copy of real inputs for another machine."""
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"packed", "global_input"}:
            raise ValueError("input pack must contain only packed and global_input")
        packed = archive["packed"]
        global_input = archive["global_input"]
    if packed.dtype != np.uint8 or packed.ndim != 3 or packed.shape[1:] != (22, 46):
        raise ValueError("input pack packed must be uint8 [N,22,46]")
    if global_input.dtype != np.float32 or global_input.shape != (len(packed), 19):
        raise ValueError("input pack global_input must be float32 [N,19]")
    if len(packed) < count or not np.isfinite(global_input).all():
        raise ValueError("input pack has too few rows or non-finite global inputs")
    onboard = np.unpackbits(packed[:, 0, :], axis=1, bitorder="big")[:, :361]
    if not np.all(onboard[:count] == 1):
        raise ValueError("input pack contains non-19x19 rows")
    return packed[:count].copy(), global_input[:count].copy(), [{
        "path": str(path.resolve()), "sha256": sha256_file(path),
        "selected_rows": count, "kind": "portable_real_input_pack",
    }]


def unpack_spatial(packed: np.ndarray) -> np.ndarray:
    bits = np.unpackbits(packed, axis=2, bitorder="big")[:, :, :audit_npz.AREA]
    return np.ascontiguousarray(bits.reshape(len(packed), 22, 19, 19), dtype=np.float32)


def load_model(path: Path, variant: str, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{variant} checkpoint missing: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("variant") != variant:
        raise ValueError(f"{path}: expected checkpoint variant {variant!r}")
    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise ValueError(f"{path}: checkpoint has no model_state dictionary")
    model = student_models.make_model(variant)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "parameters": student_models.count_parameters(model),
        "variant": variant,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "training_manifest": checkpoint.get("training_manifest"),
        "score_scale_points": checkpoint.get("score_scale_points"),
    }


def validate_outputs(outputs: Any, batch: int) -> None:
    expected = {
        "policy_logits": (batch, 362),
        "value_logits": (batch, 3),
        "score": (batch,),
        "ownership_logits": (batch, 361),
    }
    for name, shape in expected.items():
        value = getattr(outputs, name)
        if tuple(value.shape) != shape or not torch.isfinite(value).all().item():
            raise ValueError(f"invalid {name}: expected finite shape {shape}, got {tuple(value.shape)}")


def _call_forward(model: torch.nn.Module, inputs: tuple[torch.Tensor, torch.Tensor]) -> Any:
    return model(*inputs)


def _call_packed(
    model: torch.nn.Module, packed: np.ndarray, global_input: np.ndarray, device: torch.device
) -> Any:
    # Includes CPU bit unpacking, host-to-device copies, all heads' forward
    # computation, and device-to-host copies. NPZ decompression is excluded.
    spatial = torch.from_numpy(unpack_spatial(packed)).to(device)
    global_tensor = torch.from_numpy(np.ascontiguousarray(global_input.copy())).to(device)
    outputs = model(spatial, global_tensor)
    return tuple(value.to("cpu") for value in outputs)


def measure_block(
    model: torch.nn.Module,
    mode: str,
    packed: np.ndarray,
    global_input: np.ndarray,
    device: torch.device,
    iterations: int,
    device_inputs: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> float:
    """Return wall milliseconds per complete batch after an explicit sync."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    with torch.inference_mode():
        synchronize(device)
        start = time.perf_counter_ns()
        for _ in range(iterations):
            if mode == "forward_only":
                if device_inputs is None:
                    raise ValueError("forward_only requires preloaded device inputs")
                _call_forward(model, device_inputs)
            elif mode == "packed_to_cpu_outputs":
                _call_packed(model, packed, global_input, device)
            else:
                raise ValueError(f"unknown mode: {mode}")
            # Synchronize every invocation so this measures per-call wall
            # latency rather than only amortized asynchronous launch time.
            synchronize(device)
        synchronize(device)
        elapsed_ns = time.perf_counter_ns() - start
    return elapsed_ns / 1_000_000 / iterations


def summarize_samples(samples: dict[str, list[float]], batch: int) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for variant in VARIANTS:
        timings = samples[variant]
        median_ms = statistics.median(timings)
        summary[variant] = {
            "per_block_ms_per_batch": timings,
            "median_ms_per_batch": median_ms,
            "positions_per_second": batch * 1000 / median_ms,
        }
    summary["compact_speedup_vs_dense"] = (
        summary["dense"]["median_ms_per_batch"]
        / summary["compact"]["median_ms_per_batch"]
    )
    return summary


def benchmark_pair(
    models: dict[str, torch.nn.Module],
    packed_pool: np.ndarray,
    global_pool: np.ndarray,
    device: torch.device,
    batches: list[int],
    rounds: int,
    iterations: int,
    warmup: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    with torch.inference_mode():
        for batch in batches:
            packed = np.ascontiguousarray(packed_pool[:batch])
            global_input = np.ascontiguousarray(global_pool[:batch])
            device_inputs = (
                torch.from_numpy(unpack_spatial(packed)).to(device),
                torch.from_numpy(global_input.copy()).to(device),
            )
            for mode in MODES:
                samples = {variant: [] for variant in VARIANTS}
                blocks: list[dict[str, Any]] = []
                for variant in VARIANTS:
                    validate_outputs(models[variant](*device_inputs), batch)
                    if warmup:
                        measure_block(
                            models[variant], mode, packed, global_input, device,
                            warmup, device_inputs,
                        )
                for round_index in range(rounds):
                    for slot, variant in enumerate(abba_order(round_index)):
                        duration = measure_block(
                            models[variant], mode, packed, global_input, device,
                            iterations, device_inputs,
                        )
                        samples[variant].append(duration)
                        blocks.append({
                            "round": round_index + 1,
                            "slot": slot + 1,
                            "variant": variant,
                            "ms_per_batch": duration,
                        })
                results[f"batch_{batch}/{mode}"] = {
                    "batch": batch,
                    "mode": mode,
                    "blocks": blocks,
                    "summary": summarize_samples(samples, batch),
                }
    return results


def parse_batches(value: str) -> list[int]:
    try:
        batches = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("batches must be comma-separated integers") from error
    if not batches or any(batch <= 0 for batch in batches) or len(set(batches)) != len(batches):
        raise argparse.ArgumentTypeError("batches must be distinct positive integers")
    return batches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-checkpoint", type=Path, required=True)
    parser.add_argument("--compact-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--input-pack", type=Path,
                        help="portable NPZ containing real packed/global inputs; overrides --data-dir")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--batches", type=parse_batches, default=[1, 8, 32])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.rounds <= 0 or args.iterations <= 0 or args.warmup < 0:
        raise ValueError("rounds and iterations must be positive; warmup must be nonnegative")
    device = resolve_device(args.device)
    packed, global_input, sources = (
        load_input_pack(args.input_pack, max(args.batches)) if args.input_pack is not None
        else load_input_rows(args.data_dir, max(args.batches))
    )
    loaded = {variant: load_model(getattr(args, f"{variant}_checkpoint"), variant, device)
              for variant in VARIANTS}
    models = {variant: item[0] for variant, item in loaded.items()}
    model_info = {variant: item[1] for variant, item in loaded.items()}
    training_ids = {model_info[variant]["training_manifest"] for variant in VARIANTS}
    score_scales = {model_info[variant]["score_scale_points"] for variant in VARIANTS}
    if len(training_ids) != 1 or None in training_ids or score_scales != {20.0}:
        raise ValueError("checkpoints must share a training manifest and the 20-point score contract")
    measurements = benchmark_pair(
        models, packed, global_input, device, args.batches,
        args.rounds, args.iterations, args.warmup,
    )
    return {
        "schema": "go_student_inference_speed_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "PyTorch student inference only; no Rust/CUDA engine backend, search, or Go playing-strength measurement",
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "requested_device": args.device,
            "selected_device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
            "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "protocol": {
            "batches": args.batches,
            "rounds": args.rounds,
            "iterations_per_block": args.iterations,
            "warmup_iterations_per_model_mode_batch": args.warmup,
            "round_orders": [abba_order(index) for index in range(args.rounds)],
            "forward_only": "device-resident inputs, four-head model forward, wall time with per-invocation device sync",
            "packed_to_cpu_outputs": "cached packed input, CPU bit unpack, CPU-to-device, four-head model forward, all outputs to CPU, wall time with per-invocation device sync; excludes NPZ decompression",
        },
        "input": {
            "data_dir": str(args.data_dir.resolve()) if args.input_pack is None else None,
            "input_pack": str(args.input_pack.resolve()) if args.input_pack is not None else None,
            "rows": int(len(packed)),
            "shape_packed": list(packed.shape),
            "shape_global": list(global_input.shape),
            "sources": sources,
        },
        "models": model_info,
        "measurements": measurements,
        "rtx_5070_ti_rerun": (
            "python benchmark_student.py --dense-checkpoint <dense.pt> "
            "--compact-checkpoint <compact.pt> --input-pack <benchmark-inputs.npz> "
            "--output <rtx-result.json> --device cuda --batches 1,8,32 "
            f"--rounds {args.rounds} --iterations {args.iterations} --warmup {args.warmup}"
        ),
    }


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    for key, measurement in result["measurements"].items():
        summary = measurement["summary"]
        print(f"{key}: dense {summary['dense']['median_ms_per_batch']:.3f} ms, "
              f"compact {summary['compact']['median_ms_per_batch']:.3f} ms, "
              f"compact speedup {summary['compact_speedup_vs_dense']:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
