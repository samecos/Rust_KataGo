#!/usr/bin/env python3
"""Export frozen student checkpoints to the engine's self-identifying CUDA format.

The engine detects the 16-byte magic, independently of the file extension.
The artifact contains a bounded JSON header and contiguous little-endian FP32
weights; no Python, Torch, or ONNX runtime is required for engine inference.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any

import numpy as np
import torch

from student_models import make_model


MAGIC = b"RustGoStudentV1\n"
FORMAT_VERSION = 1
INPUT_VERSION = 7
SCORE_SCALE = 20.0
MAX_HEADER_BYTES = 65536
MAX_FILE_BYTES = 16 * 1024 * 1024
SPECS = {"compact": (48, 4), "dense": (64, 6)}
HEADER_FIELDS = {"format_version", "input_version", "variant", "width", "blocks", "score_scale", "tensors"}
TENSOR_FIELDS = {"name", "shape", "offset", "length"}


def expected_tensor_shapes(variant: str) -> dict[str, tuple[int, ...]]:
    """The frozen v1 ABI, independent of future changes to StudentNet."""
    if variant not in SPECS:
        raise ValueError(f"unsupported student variant: {variant!r}")
    width, blocks = SPECS[variant]
    shapes = {
        "stem.weight": (width, 22, 3, 3), "stem.bias": (width,),
        "global_to_stem.weight": (width, 19), "global_to_stem.bias": (width,),
    }
    for index in range(blocks):
        for conv in ("conv1", "conv2"):
            prefix = f"blocks.{index}.{conv}"
            shapes[f"{prefix}.weight"] = (width, width, 3, 3)
            shapes[f"{prefix}.bias"] = (width,)
    shapes.update({
        "policy_board.weight": (1, width, 1, 1), "policy_board.bias": (1,),
        "policy_pass.weight": (1, width + 19), "policy_pass.bias": (1,),
        "value_features.weight": (64, width + 19), "value_features.bias": (64,),
        "value.weight": (3, 64), "value.bias": (3,),
        "score_head.weight": (1, 64), "score_head.bias": (1,),
        "ownership.weight": (1, width, 1, 1), "ownership.bias": (1,),
    })
    return shapes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def validate_state(state: Mapping[str, Any], variant: str) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping):
        raise ValueError("model_state must be a parameter mapping")
    expected = expected_tensor_shapes(variant)
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        extra = sorted(str(key) for key in set(state) - set(expected))
        raise ValueError(f"parameter set mismatch: missing={missing}, extra={extra}")
    result = {}
    for name, shape in expected.items():
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided or not tensor.is_floating_point():
            raise ValueError(f"{name}: expected a dense floating point tensor")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name}: expected shape {shape}, got {tuple(tensor.shape)}")
        converted = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not torch.isfinite(converted).all().item():
            raise ValueError(f"{name}: non-finite FP32 weight")
        result[name] = converted
    return result


def load_checkpoint(path: Path) -> tuple[str, dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must contain variant, model_state, and score_scale_points")
    variant = checkpoint.get("variant")
    if not isinstance(variant, str) or variant not in SPECS:
        raise ValueError("checkpoint has an unsupported or missing variant")
    scale = checkpoint.get("score_scale_points")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or scale != SCORE_SCALE:
        raise ValueError("checkpoint score_scale_points must be 20.0")
    width, blocks = SPECS[variant]
    for field, expected in (("width", width), ("blocks", blocks)):
        if field in checkpoint and (type(checkpoint[field]) is not int or checkpoint[field] != expected):
            raise ValueError(f"checkpoint {field} does not match frozen {variant} architecture")
    return variant, validate_state(checkpoint.get("model_state"), variant)


def _integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def read_export(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Strictly validate a v1 export and reconstruct its state dictionary."""
    size = path.stat().st_size
    if size > MAX_FILE_BYTES or size < len(MAGIC) + 4:
        raise ValueError("invalid model file size")
    with path.open("rb") as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            raise ValueError("unrecognized student model magic")
        header_length = struct.unpack("<I", stream.read(4))[0]
        if not 0 < header_length <= MAX_HEADER_BYTES or header_length > size - len(MAGIC) - 4:
            raise ValueError("invalid JSON header length")
        header_raw = stream.read(header_length)
        header = json.loads(header_raw.decode("utf-8"), object_pairs_hook=_unique_object,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON number: {value}")))
        payload = stream.read()
    if not isinstance(header, dict) or set(header) != HEADER_FIELDS:
        raise ValueError("unexpected model header fields")
    if type(header["format_version"]) is not int or header["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported format_version")
    if type(header["input_version"]) is not int or header["input_version"] != INPUT_VERSION:
        raise ValueError("unsupported input_version")
    variant = header["variant"]
    if not isinstance(variant, str) or variant not in SPECS:
        raise ValueError("unsupported variant")
    width, blocks = SPECS[variant]
    if type(header["width"]) is not int or header["width"] != width or type(header["blocks"]) is not int or header["blocks"] != blocks:
        raise ValueError("width/blocks do not match frozen architecture")
    scale = header["score_scale"]
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or scale != SCORE_SCALE:
        raise ValueError("unsupported score_scale")
    expected = expected_tensor_shapes(variant)
    entries = header["tensors"]
    if not isinstance(entries, list) or len(entries) != len(expected):
        raise ValueError("tensor count mismatch")
    state = {}
    offset = 0
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != TENSOR_FIELDS:
            raise ValueError("unexpected tensor descriptor fields")
        name = entry["name"]
        if not isinstance(name, str) or name not in expected or name in state:
            raise ValueError(f"unknown or duplicate tensor name: {name}")
        shape = expected[name]
        if entry["shape"] != list(shape) or any(type(x) is not int for x in entry["shape"]):
            raise ValueError(f"tensor shape mismatch: expected {name} {shape}")
        length = math.prod(shape)
        if _integer(entry["offset"], "offset") != offset or _integer(entry["length"], "length") != length:
            raise ValueError(f"{name}: noncontiguous offset or incorrect length")
        end = offset + 4 * length
        if end > len(payload):
            raise ValueError(f"{name}: truncated weight payload")
        array = np.frombuffer(payload[offset:end], dtype="<f4").reshape(shape)
        if not np.isfinite(array).all():
            raise ValueError(f"{name}: non-finite weight payload")
        state[name] = torch.from_numpy(array.astype(np.float32, copy=True))
        offset = end
    if offset != len(payload) or len(MAGIC) + 4 + header_length + offset != size:
        raise ValueError("payload has trailing bytes or inconsistent total length")
    return header, state


def load_exported_model(path: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    header, state = read_export(path)
    model = make_model(header["variant"])
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, header


def export_model(state: Mapping[str, Any], variant: str, output: Path, *, source: dict[str, Any] | None = None) -> dict[str, Any]:
    state = validate_state(state, variant)
    width, blocks = SPECS[variant]
    entries = []
    chunks = []
    offset = 0
    for name, tensor in state.items():
        chunk = tensor.numpy().astype("<f4", copy=False).tobytes(order="C")
        entries.append({"name": name, "shape": list(tensor.shape), "offset": offset, "length": tensor.numel()})
        chunks.append(chunk)
        offset += len(chunk)
    header = {"format_version": FORMAT_VERSION, "input_version": INPUT_VERSION, "variant": variant,
              "width": width, "blocks": blocks, "score_scale": SCORE_SCALE, "tensors": entries}
    header_bytes = json.dumps(header, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(header_bytes) > MAX_HEADER_BYTES:
        raise ValueError("model header is too large")
    content = MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + b"".join(chunks)
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("model is too large")
    atomic_write(output, content)
    verified_header, verified_state = read_export(output)
    if verified_header != header or any(not torch.equal(state[key], verified_state[key]) for key in state):
        raise RuntimeError("export readback differs from source weights")
    receipt = {
        "schema": "rust_go_student_export_receipt_v1", "status": "validated",
        "model": str(output.resolve()), "sha256": sha256_file(output), "bytes": len(content),
        "magic_bytes": len(MAGIC), "header_bytes": len(header_bytes), "payload_bytes": offset,
        "tensor_count": len(entries), "parameters": offset // 4,
        "format_version": FORMAT_VERSION, "input_version": INPUT_VERSION, "variant": variant,
        "width": width, "blocks": blocks, "score_scale": SCORE_SCALE,
        "source": source or {"kind": "in-memory state"},
        "validation": "exact parameter names/shapes, finite FP32 weights, contiguous payload, exact readback",
    }
    atomic_write(Path(str(output) + ".receipt.json"),
                 (json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return receipt


def export_checkpoint(checkpoint: Path, output: Path) -> dict[str, Any]:
    if checkpoint.resolve() == output.resolve():
        raise ValueError("output must differ from checkpoint")
    before = sha256_file(checkpoint)
    variant, state = load_checkpoint(checkpoint)
    if sha256_file(checkpoint) != before:
        raise ValueError("checkpoint changed during export")
    return export_model(state, variant, output, source={"kind": "training checkpoint", "path": str(checkpoint.resolve()), "sha256": before})


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="train_student.py checkpoint; loaded with weights_only=True")
    source.add_argument("--random-variant", choices=tuple(SPECS), help="fixed-seed untrained model for implementation tests")
    result.add_argument("--output", type=Path, required=True, help="engine model artifact (normally .rgmodel)")
    result.add_argument("--seed", type=int, default=29, help="random test model seed")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.checkpoint is not None:
            receipt = export_checkpoint(args.checkpoint, args.output)
        else:
            torch.manual_seed(args.seed)
            model = make_model(args.random_variant)
            receipt = export_model(model.state_dict(), args.random_variant, args.output,
                                   source={"kind": "untrained implementation fixture", "seed": args.seed})
        print(json.dumps(receipt, ensure_ascii=False))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
