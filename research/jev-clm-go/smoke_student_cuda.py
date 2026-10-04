#!/usr/bin/env python3
"""Exercise actual CUDA four-head training and epoch resume with synthetic data.

No real games or training data are downloaded. Losses only check that the
training/checkpoint chain works; they cannot establish Go model quality.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

import train_student
from export_student import read_export


def synthetic_cache(path: Path, seed: int = 37) -> None:
    """Write a small, explicitly synthetic cache satisfying the trainer contract."""
    path.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    rows = 32
    spatial = rng.integers(0, 2, (rows, 22, 361), dtype=np.uint8)
    spatial[:, 0] = 1
    data = {
        "packed": np.packbits(spatial, axis=2, bitorder="big"),
        "global_input": rng.normal(0, 0.1, (rows, 19)).astype(np.float32),
        "policy_counts": rng.integers(0, 20, (rows, 362), dtype=np.int16),
        "row_weight": np.ones(rows, dtype=np.float32),
        "value_probs": rng.dirichlet([2, 2, 1], rows).astype(np.float32),
        "value_weight": np.ones(rows, dtype=np.float32),
        "score_mean": rng.normal(0, 10, rows).astype(np.float32),
        "score_mean_weight": np.ones(rows, dtype=np.float32),
        "ownership": rng.integers(-1, 2, (rows, 19, 19), dtype=np.int8),
        "ownership_weight": np.ones(rows, dtype=np.float32),
        "split_code": np.array([0] * 16 + [1] * 8 + [2] * 8, dtype=np.uint8),
    }
    np.savez_compressed(path / "shard-00000.npz", **data)
    train_student.atomic_json(path / "manifest.json", {
        "schema": "rust_katago_multitask_cache_v1", "status": "completed",
        "source": {"kind": "synthetic CUDA pipeline smoke", "seed": seed},
        "split": {"rows": {"train": 16, "val": 8, "test": 8}},
        "scope": "random synthetic tensors, no legal replay or playing-strength evidence",
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"smoke output already exists: {args.out_dir}")
    device = train_student.resolve_device("cuda")
    cache = args.out_dir / "cache"
    synthetic_cache(cache)
    output = args.out_dir / "training"
    base = ["--cache-dir", str(cache), "--out-dir", str(output),
            "--device", "cuda", "--precision", "fp16", "--batch-size", "8",
            "--train-rows", "0", "--val-rows", "0", "--test-rows", "0"]
    first = train_student.run(train_student.parser().parse_args(base + ["--epochs", "1"]))
    first_last = {
        variant: torch.load(output / f"{variant}.last.pt", map_location="cpu", weights_only=True)
        for variant in ("dense", "compact")
    }
    final = train_student.run(train_student.parser().parse_args(base + ["--epochs", "2", "--resume"]))
    checks = {}
    for variant in ("dense", "compact"):
        current = torch.load(output / f"{variant}.last.pt", map_location="cpu", weights_only=True)
        changed = {
            name: not torch.equal(tensor, first_last[variant]["model_state"][name])
            for name, tensor in current["model_state"].items()
        }
        if not all(changed.values()) or current["epoch"] != 2:
            raise AssertionError(f"{variant}: missing resumed parameter updates")
        if not all(bool(torch.isfinite(tensor).all()) for tensor in current["model_state"].values()):
            raise AssertionError(f"{variant}: non-finite saved weights")
        if len(current["model_info"]["epochs"]) != 2 or not current["scaler_state"]:
            raise AssertionError(f"{variant}: incomplete resume history/scaler")
        steps = {int(value["step"].item()) for value in current["optimizer_state"]["state"].values()}
        if steps != {4}:
            raise AssertionError(f"{variant}: AdamW state was not resumed: {steps}")
        exported_path = output / f"{variant}.rgmodel"
        header, exported_state = read_export(exported_path)
        selected = torch.load(output / f"{variant}.pt", map_location="cpu", weights_only=True)
        if header["variant"] != variant:
            raise AssertionError(f"{variant}: export variant mismatch")
        for name, tensor in selected["model_state"].items():
            torch.testing.assert_close(exported_state[name], tensor, rtol=0, atol=0)
        checks[variant] = {
            "parameters_updated_after_resume": len(changed), "all_saved_weights_finite": True,
            "optimizer_steps": 4, "epoch": 2, "precision": "fp16",
            "checkpoint": str((output / f"{variant}.pt").resolve()),
            "checkpoint_sha256": train_student.sha256_file(output / f"{variant}.pt"),
            "engine_model": str(exported_path.resolve()),
            "engine_model_sha256": train_student.sha256_file(exported_path),
            "exported_parameters_match_best_checkpoint": True,
            "cuda_peak_allocated_bytes": final["models"][variant]["cuda_peak_allocated_bytes"],
        }
    report = {
        "schema": "rust_katago_student_cuda_training_smoke_v1", "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "synthetic tensors only; CUDA FP16 training, all heads, optimizer/scaler resume and checkpoints",
        "environment": final["environment"], "checks": checks,
        "first_training_manifest_config": first["config"],
        "final_training_manifest": str((output / "manifest.json").resolve()),
    }
    report_path = args.out_dir / "result.json"
    train_student.atomic_json(report_path, report)
    print(json.dumps({"status": "passed", "device": str(device), "result": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
