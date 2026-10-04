#!/usr/bin/env python3
"""Compare native student CUDA raw outputs with exported FP32 PyTorch golden.

Each case invokes student_cuda_dump once, retains its real return code and logs,
and checks every value in all four output heads. This tests implementation
compatibility, not playing strength or an accuracy budget against KataGo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from export_student import atomic_write, sha256_file
from student_reference import HEADS


def compare_outputs(reference: dict[str, Any], actual: dict[str, Any], *, atol: float, rtol: float,
                    max_abs: float) -> dict[str, Any]:
    if not isinstance(actual, dict):
        raise ValueError("CUDA dump must be an object with four raw output heads")
    result = {}
    for name in HEADS:
        if name not in actual or name not in reference:
            raise ValueError(f"missing {name}")
        expected = np.asarray(reference[name], dtype=np.float64).reshape(-1)
        observed = np.asarray(actual[name], dtype=np.float64).reshape(-1)
        if observed.shape != expected.shape or expected.size != int(np.prod(reference["shapes"][name])):
            raise ValueError(f"{name}: output length/shape mismatch")
        if not np.isfinite(expected).all() or not np.isfinite(observed).all():
            raise ValueError(f"{name}: non-finite output")
        error = np.abs(observed - expected)
        allowed = atol + rtol * np.abs(expected)
        violated = int(np.count_nonzero(error > allowed))
        largest = float(error.max(initial=0))
        result[name] = {"values": int(expected.size), "max_absolute_error": largest,
                        "mean_absolute_error": float(error.mean()),
                        "max_relative_error_floor_1e6": float((error / np.maximum(np.abs(expected), 1e-6)).max(initial=0)),
                        "element_tolerance_failures": violated,
                        "passed": violated == 0 and largest <= max_abs}
    return {"heads": result, "passed": all(head["passed"] for head in result.values())}


def run(executable: Path, manifest_path: Path, output_dir: Path, *, atol: float = 1e-4,
        rtol: float = 1e-4, max_abs: float = 2e-4, timeout: float = 120.0) -> dict[str, Any]:
    if any(not np.isfinite(value) or value < 0 for value in (atol, rtol, max_abs)) or not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("comparison tolerances must be finite nonnegative and timeout positive")
    executable = executable.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    before_manifest = sha256_file(manifest_path)
    before_executable = sha256_file(executable)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "rust_go_student_reference_v1" or not manifest.get("cases"):
        raise ValueError("unsupported or empty reference manifest")
    model = Path(manifest["model"])
    if sha256_file(model) != manifest["model_sha256"]:
        raise ValueError("model differs from reference manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "rust_go_student_cuda_parity_v1", "status": "running",
        "executable": str(executable), "executable_sha256": before_executable,
        "manifest": str(manifest_path), "manifest_sha256": before_manifest,
        "model": str(model), "model_sha256": manifest["model_sha256"], "variant": manifest["variant"],
        "tolerance": {"atol": atol, "rtol": rtol, "global_max_absolute_error": max_abs},
        "reference_device": manifest["device"], "reference_precision": manifest["precision"],
        "cases": [], "scope": "native CUDA raw four-head implementation parity; not model strength or inference speed",
    }
    for case in manifest["cases"]:
        batch = case["batch"]
        observation: dict[str, Any] = {"batch": batch, "passed": False}
        result["cases"].append(observation)
        try:
            for path_key, sha_key in (("spatial", "spatial_sha256"), ("global_input", "global_input_sha256"), ("reference", "reference_sha256")):
                if sha256_file(Path(case[path_key])) != case[sha_key]:
                    raise ValueError(f"{path_key} differs from reference manifest")
            if Path(case["spatial"]).stat().st_size != batch * 22 * 361 * 4 or Path(case["global_input"]).stat().st_size != batch * 19 * 4:
                raise ValueError("input binary length mismatch")
            actual_path = output_dir / f"batch-{batch}-cuda.json"
            if actual_path.exists():
                raise ValueError("CUDA output already exists; choose a fresh run directory")
            command = [str(executable), str(model), case["spatial"], case["global_input"], str(batch), str(actual_path.resolve())]
            observation["command"] = command
            started = time.perf_counter()
            process = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
            observation["seconds"] = time.perf_counter() - started
            observation["returncode"] = process.returncode
            log_path = output_dir / f"batch-{batch}-cuda.log"
            atomic_write(log_path, b"STDOUT\n" + process.stdout + b"\nSTDERR\n" + process.stderr)
            observation["log"] = str(log_path.resolve())
            observation["log_sha256"] = sha256_file(log_path)
            if process.returncode != 0:
                raise RuntimeError(f"CUDA dump returned {process.returncode}")
            reference = json.loads(Path(case["reference"]).read_text(encoding="utf-8"))
            if reference["batch"] != batch or reference["model_sha256"] != manifest["model_sha256"]:
                raise ValueError("reference model/batch differs from manifest")
            actual = json.loads(actual_path.read_text(encoding="utf-8"))
            observation.update(compare_outputs(reference, actual, atol=atol, rtol=rtol, max_abs=max_abs))
            observation["outputs"] = str(actual_path.resolve())
            observation["outputs_sha256"] = sha256_file(actual_path)
            for path_key, sha_key in (("spatial", "spatial_sha256"), ("global_input", "global_input_sha256"), ("reference", "reference_sha256")):
                if sha256_file(Path(case[path_key])) != case[sha_key]:
                    raise ValueError(f"{path_key} changed during CUDA inference")
        except (ValueError, RuntimeError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
            observation["passed"] = False
            observation["error"] = f"{type(exc).__name__}: {exc}"
    stable = (sha256_file(manifest_path) == before_manifest and sha256_file(executable) == before_executable
              and sha256_file(model) == manifest["model_sha256"])
    result["source_stable"] = stable
    result["status"] = "passed" if stable and all(case["passed"] for case in result["cases"]) else "failed"
    atomic_write(output_dir / "result.json", (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True, help="student_cuda_dump executable")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--max-abs", type=float, default=2e-4)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    try:
        result = run(args.executable, args.manifest, args.out_dir, atol=args.atol, rtol=args.rtol,
                     max_abs=args.max_abs, timeout=args.timeout)
        print(json.dumps({"status": result["status"], "result": str((args.out_dir / "result.json").resolve()),
                          "cases": len(result["cases"])}, ensure_ascii=False))
        return 0 if result["status"] == "passed" else 1
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
