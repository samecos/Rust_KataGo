#!/usr/bin/env python3
"""Write FP32 PyTorch student outputs and NCHW binary inputs for CUDA parity.

Random/synthetic fixtures only check the implementation, not model strength.
An optional audited benchmark input pack adds previously collected real inputs
without downloading any training data. TF32 is disabled for the reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

from export_student import atomic_write, load_exported_model, sha256_file


HEADS = ("policy_logits", "value_logits", "score", "ownership_logits")


def input_pool(count: int, seed: int, input_pack: Path | None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if input_pack is None:
        rng = np.random.default_rng(seed)
        spatial = rng.integers(0, 2, size=(count, 22, 19, 19), dtype=np.uint8).astype(np.float32)
        global_input = rng.normal(0.0, 0.25, size=(count, 19)).astype(np.float32)
        return spatial, global_input, {"kind": "fixed synthetic implementation fixture", "seed": seed}
    before_input = sha256_file(input_pack)
    with np.load(input_pack, allow_pickle=False) as archive:
        if set(archive.files) != {"packed", "global_input"}:
            raise ValueError("input pack must contain only packed and global_input")
        packed = archive["packed"]
        global_input = archive["global_input"]
    if packed.dtype != np.uint8 or packed.ndim != 3 or packed.shape[1:] != (22, 46):
        raise ValueError("input pack packed must be uint8 [N,22,46]")
    if global_input.dtype != np.float32 or global_input.shape != (len(packed), 19):
        raise ValueError("input pack global_input must be float32 [N,19]")
    if len(packed) < count or not np.isfinite(global_input).all():
        raise ValueError("input pack has insufficient rows or non-finite global inputs")
    if sha256_file(input_pack) != before_input:
        raise ValueError("input pack changed while loading")
    spatial = np.unpackbits(packed[:count], axis=2, bitorder="big")[:, :, :361].reshape(count, 22, 19, 19).astype(np.float32)
    return spatial, np.ascontiguousarray(global_input[:count]), {
        "kind": "existing benchmark input pack", "path": str(input_pack.resolve()), "sha256": before_input,
    }


def write_reference(model_path: Path, output_dir: Path, batches: list[int], *, seed: int = 29,
                    input_pack: Path | None = None, device_name: str = "cpu") -> dict[str, Any]:
    if not batches or any(type(batch) is not int or batch <= 0 for batch in batches) or len(set(batches)) != len(batches):
        raise ValueError("batch sizes must be unique positive integers")
    if device_name not in ("cpu", "cuda"):
        raise ValueError("reference device must be cpu or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA reference requested but unavailable")
    if device_name == "cuda":
        from cuda_environment import isolate_windows_cudnn
        isolate_windows_cudnn(torch)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    before = sha256_file(model_path)
    model, header = load_exported_model(model_path)
    device = torch.device(device_name)
    model.to(device)
    spatial, global_input, source = input_pool(max(batches), seed, input_pack)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "rust_go_student_reference_v1", "model": str(model_path.resolve()), "model_sha256": before,
        "variant": header["variant"], "input_version": header["input_version"], "score_scale": header["score_scale"],
        "device": str(device), "torch": torch.__version__, "tf32": False, "precision": "FP32",
        "input_source": source, "cases": [],
        "semantics": "raw mover-perspective logits; score is in 20-point units; engine postprocessing is separate",
    }
    with torch.no_grad():
        for batch in batches:
            prefix = f"batch-{batch}"
            spatial_path = output_dir / f"{prefix}-spatial.f32"
            global_path = output_dir / f"{prefix}-global.f32"
            reference_path = output_dir / f"{prefix}-reference.json"
            atomic_write(spatial_path, spatial[:batch].astype("<f4", copy=False).tobytes(order="C"))
            atomic_write(global_path, global_input[:batch].astype("<f4", copy=False).tobytes(order="C"))
            outputs = model(torch.from_numpy(spatial[:batch]).to(device), torch.from_numpy(global_input[:batch]).to(device))
            reference: dict[str, Any] = {"batch": batch, "model_sha256": before, "shapes": {}}
            for name in HEADS:
                array = getattr(outputs, name).detach().cpu().numpy()
                if array.dtype != np.float32 or not np.isfinite(array).all():
                    raise FloatingPointError(f"non-finite or non-FP32 {name}")
                reference["shapes"][name] = list(array.shape)
                reference[name] = array.reshape(-1).tolist()
            atomic_write(reference_path, (json.dumps(reference, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))
            manifest["cases"].append({
                "batch": batch, "spatial": str(spatial_path.resolve()), "spatial_sha256": sha256_file(spatial_path),
                "spatial_shape": [batch, 22, 19, 19], "global_input": str(global_path.resolve()),
                "global_input_sha256": sha256_file(global_path), "global_shape": [batch, 19],
                "reference": str(reference_path.resolve()), "reference_sha256": sha256_file(reference_path),
                "binary_layout": "little-endian FP32, contiguous NCHW spatial and NC global",
            })
    if sha256_file(model_path) != before:
        raise ValueError("model changed during reference generation")
    manifest_path = output_dir / "manifest.json"
    atomic_write(manifest_path, (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="validated .rgmodel export")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 3, 8])
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--input-pack", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    try:
        manifest = write_reference(args.model, args.out_dir, args.batches, seed=args.seed,
                                   input_pack=args.input_pack, device_name=args.device)
        print(json.dumps({"manifest": str((args.out_dir / "manifest.json").resolve()),
                          "variant": manifest["variant"], "batches": args.batches}, ensure_ascii=False))
        return 0
    except (ValueError, RuntimeError, OSError, FloatingPointError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
