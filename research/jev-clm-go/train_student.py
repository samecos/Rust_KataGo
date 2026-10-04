#!/usr/bin/env python3
"""Train two standalone, multi-output Go models from audited self-play shards.

This produces PyTorch student checkpoints for the student weight exporter. The
same selected game-disjoint examples, optimizer, and epoch budget are used for
both variants. The held-out test split is evaluated once, after validation
checkpoint selection. No target array is passed to the model as an input.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from student_models import count_parameters, make_model
from export_student import export_checkpoint
from cuda_environment import isolate_windows_cudnn


SCHEMA = "rust_katago_student_training_v1"
RESUME_SCHEMA = "rust_katago_student_resume_v1"
SPLIT = {"train": 0, "val": 1, "test": 2}
SCORE_SCALE = 20.0  # Network predicts score in units of 20 points.
FIELDS = (
    "packed", "global_input", "policy_counts", "row_weight",
    "value_probs", "value_weight", "score_mean", "score_mean_weight",
    "ownership", "ownership_weight", "split_code",
)
FIELD_SHAPES = {
    "packed": ((22, 46), np.uint8),
    "global_input": ((19,), np.float32),
    "policy_counts": ((362,), np.int16),
    "row_weight": ((), np.float32),
    "value_probs": ((3,), np.float32),
    "value_weight": ((), np.float32),
    "score_mean": ((), np.float32),
    "score_mean_weight": ((), np.float32),
    "ownership": ((19, 19), np.int8),
    "ownership_weight": ((), np.float32),
    "split_code": ((), np.uint8),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = ("cuda" if torch.cuda.is_available() else
                     "mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(requested)


def resolve_precision(requested: str, device: torch.device) -> str:
    precision = "fp16" if requested == "auto" and device.type == "cuda" else (
        "fp32" if requested == "auto" else requested)
    if precision != "fp32" and device.type != "cuda":
        raise ValueError("fp16/bf16 training requires CUDA; use --precision fp32 for CPU/MPS")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA device does not support bfloat16")
    return precision


def autocast_context(device: torch.device, precision: str) -> Any:
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=(torch.float16 if precision == "fp16" else torch.bfloat16))


def _validate_field(name: str, array: np.ndarray, rows: int) -> None:
    shape, dtype = FIELD_SHAPES[name]
    if array.dtype != dtype or array.shape != (rows, *shape):
        raise ValueError(f"{name}: expected {(rows, *shape)} {dtype}, got {array.shape} {array.dtype}")


def load_selected(cache_dir: Path, limits: dict[str, int], selection_seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Select rows uniformly within each game-disjoint split, then load data.

    Selection uses only split codes; no label or model metric influences it.
    The same arrays are reused for both models. The prior probe's member paths
    and game hashes must have been excluded by prepare_model_data.py.
    """
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "rust_katago_multitask_cache_v1" or manifest.get("status") not in (None, "completed"):
        raise ValueError("data cache manifest has an unsupported schema or incomplete status")
    if not isinstance(manifest.get("source"), dict) or not isinstance(manifest.get("split"), dict):
        raise ValueError("data cache manifest is missing source or split metadata")
    shards = sorted(cache_dir.glob("shard-*.npz"))
    if not shards:
        raise ValueError("data cache has no shards")
    split_codes = []
    shard_rows = []
    for path in shards:
        with np.load(path, allow_pickle=False) as shard:
            codes = shard["split_code"]
        if codes.dtype != np.uint8 or np.any(codes > 2):
            raise ValueError(f"{path}: invalid split codes")
        split_codes.append(codes)
        shard_rows.append(len(codes))
    codes_all = np.concatenate(split_codes)
    rng = np.random.default_rng(selection_seed)
    chosen_by_split: dict[str, np.ndarray] = {}
    chosen_parts = []
    for name, code in SPLIT.items():
        candidates = np.flatnonzero(codes_all == code)
        if not len(candidates):
            raise ValueError(f"empty {name} split")
        cap = limits[name]
        selected = (rng.choice(candidates, size=cap, replace=False)
                    if cap and cap < len(candidates) else candidates)
        chosen_by_split[name] = np.sort(selected)
        chosen_parts.append(selected)
    chosen = np.sort(np.concatenate(chosen_parts))
    arrays = {
        name: np.empty((len(chosen), *shape), dtype=dtype)
        for name, (shape, dtype) in FIELD_SHAPES.items()
    }
    offsets = np.cumsum([0, *shard_rows])
    copied = 0
    for shard_index, path in enumerate(shards):
        lo, hi = offsets[shard_index:shard_index + 2]
        first = int(np.searchsorted(chosen, lo))
        last = int(np.searchsorted(chosen, hi))
        if first == last:
            continue
        local = chosen[first:last] - lo
        with np.load(path, allow_pickle=False) as shard:
            rows = len(shard["split_code"])
            for name in FIELDS:
                if name not in shard:
                    raise ValueError(f"{path}: missing {name}")
                source = shard[name]
                _validate_field(name, source, rows)
                arrays[name][first:last] = source[local]
        copied += last - first
    if copied != len(chosen):
        raise RuntimeError("selected row count mismatch")
    if np.any(arrays["policy_counts"] < 0) or np.any(arrays["policy_counts"].sum(axis=1) <= 0):
        raise ValueError("invalid selected policy counts")
    if not np.all(np.isfinite(arrays["global_input"])):
        raise ValueError("non-finite model input")
    for name in ("row_weight", "value_weight", "score_mean_weight", "ownership_weight"):
        if not np.all(np.isfinite(arrays[name])) or np.any(arrays[name] < 0):
            raise ValueError(f"invalid {name}")
    if np.any(arrays["row_weight"] <= 0):
        raise ValueError("selected policy rows must have positive weight")
    if not np.all(np.isfinite(arrays["value_probs"])) or not np.all(np.isfinite(arrays["score_mean"])):
        raise ValueError("non-finite target")
    if np.any(arrays["ownership"] < -1) or np.any(arrays["ownership"] > 1):
        raise ValueError("ownership target outside [-1,1]")
    indices = {name: np.flatnonzero(arrays["split_code"] == code) for name, code in SPLIT.items()}
    info = {
        "cache_manifest": str(manifest_path.resolve()),
        "cache_manifest_sha256": sha256_file(manifest_path),
        "rows_available": {name: int(np.count_nonzero(codes_all == code)) for name, code in SPLIT.items()},
        "rows_selected": {name: int(len(indices[name])) for name in SPLIT},
        "selection_seed": selection_seed,
        "selection": "uniform without replacement by split code, before reading model targets",
        "shards": len(shards),
        "cache_shards": [{"name": path.name, "sha256": sha256_file(path)} for path in shards],
    }
    selected_digest = hashlib.sha256()
    for name in FIELDS:
        selected_digest.update(name.encode("ascii"))
        selected_digest.update(arrays[name].tobytes(order="C"))
    info["selected_data_sha256"] = selected_digest.hexdigest()
    return arrays, info


def unpack_batch(packed: np.ndarray) -> np.ndarray:
    if packed.ndim != 3 or packed.shape[1:] != (22, 46):
        raise ValueError("expected packed [N,22,46]")
    bits = np.unpackbits(packed, axis=2, bitorder="big")[:, :, :361]
    return bits.reshape(len(packed), 22, 19, 19).astype(np.float32)


def batch_tensors(data: dict[str, np.ndarray], indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    batch = {
        "spatial": torch.from_numpy(unpack_batch(data["packed"][indices])).to(device),
        "global_input": torch.from_numpy(np.ascontiguousarray(data["global_input"][indices])).to(device),
    }
    for name in ("policy_counts", "row_weight", "value_probs", "value_weight",
                 "score_mean", "score_mean_weight", "ownership", "ownership_weight"):
        batch[name] = torch.from_numpy(np.ascontiguousarray(data[name][indices]).astype(np.float32)).to(device)
    return batch


def weighted_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (values * weight).sum() / weight.sum().clamp_min(1e-12)


def loss_and_metrics(outputs: Any, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    counts = batch["policy_counts"]
    value_weight = batch["row_weight"] * batch["value_weight"]
    score_weight = batch["row_weight"] * batch["score_mean_weight"]
    ownership_weight = batch["row_weight"] * batch["ownership_weight"]
    policy_target = counts / counts.sum(dim=1, keepdim=True)
    policy_ce = -(policy_target * F.log_softmax(outputs.policy_logits.float(), dim=1)).sum(dim=1)
    policy = weighted_mean(policy_ce, batch["row_weight"])
    top1 = weighted_mean((outputs.policy_logits.argmax(1) == counts.argmax(1)).float(), batch["row_weight"])

    value_target = batch["value_probs"].clamp_min(0)
    value_target = value_target / value_target.sum(dim=1, keepdim=True).clamp_min(1e-12)
    value_prob = F.softmax(outputs.value_logits.float(), dim=1)
    value_ce_rows = -(value_target * F.log_softmax(outputs.value_logits.float(), dim=1)).sum(dim=1)
    value_ce = weighted_mean(value_ce_rows, value_weight)
    value_brier = weighted_mean(((value_prob - value_target) ** 2).sum(dim=1), value_weight)

    score_target = batch["score_mean"] / SCORE_SCALE
    score_rows = F.smooth_l1_loss(outputs.score.float(), score_target, reduction="none")
    score_loss = weighted_mean(score_rows, score_weight)
    score_mae = weighted_mean((outputs.score.float() * SCORE_SCALE - batch["score_mean"]).abs(),
                              score_weight)

    ownership_target = batch["ownership"].reshape(-1, 361)
    ownership_pred = torch.tanh(outputs.ownership_logits.float())
    ownership_rows = ((ownership_pred - ownership_target) ** 2).mean(dim=1)
    ownership_loss = weighted_mean(ownership_rows, ownership_weight)
    ownership_mae = weighted_mean((ownership_pred - ownership_target).abs().mean(dim=1),
                                  ownership_weight)

    total = policy + 0.5 * value_ce + 0.1 * score_loss + 0.2 * ownership_loss
    return {
        "total_loss": total, "policy_ce": policy, "policy_top1": top1,
        "value_ce": value_ce, "value_brier": value_brier,
        "score_loss": score_loss, "score_mae_points": score_mae,
        "ownership_loss": ownership_loss, "ownership_mae": ownership_mae,
    }


@torch.no_grad()
def evaluate(model: torch.nn.Module, data: dict[str, np.ndarray], indices: np.ndarray,
             batch_size: int, device: torch.device, precision: str = "fp32") -> dict[str, Any]:
    model.eval()
    sums: dict[str, float] = {}
    weights: dict[str, float] = {}
    for start in range(0, len(indices), batch_size):
        batch = batch_tensors(data, indices[start:start + batch_size], device)
        with autocast_context(device, precision):
            outputs = model(batch["spatial"], batch["global_input"])
        values = loss_and_metrics(outputs, batch)
        # Each head has a different target coverage. Aggregate by its own weight.
        head_weight = {
            "policy": float(batch["row_weight"].sum().item()),
            "value": float((batch["row_weight"] * batch["value_weight"]).sum().item()),
            "score": float((batch["row_weight"] * batch["score_mean_weight"]).sum().item()),
            "ownership": float((batch["row_weight"] * batch["ownership_weight"]).sum().item()),
        }
        for name, value in values.items():
            head = ("policy" if name.startswith("policy") else
                    "value" if name.startswith("value") else
                    "score" if name.startswith("score") else
                    "ownership" if name.startswith("ownership") else "policy")
            w = head_weight[head]
            if w > 0:
                sums[name] = sums.get(name, 0.0) + float(value.item()) * w
                weights[name] = weights.get(name, 0.0) + w
    out = {name: sums[name] / weights[name] for name in sums}
    required = ("policy_ce", "value_ce", "score_loss", "ownership_loss")
    if any(weights.get(name, 0.0) <= 0 for name in required):
        raise ValueError("evaluation split lacks positive weight for one or more output heads")
    # Rebuild the validation selection objective from properly aggregated heads.
    out["total_loss"] = (out["policy_ce"] + 0.5 * out.get("value_ce", 0.0)
                         + 0.1 * out.get("score_loss", 0.0)
                         + 0.2 * out.get("ownership_loss", 0.0))
    out["rows"] = int(len(indices))
    out["effective_weight"] = {
        "policy": weights["policy_ce"], "value": weights["value_ce"],
        "score": weights["score_loss"], "ownership": weights["ownership_loss"],
    }
    return out


def train_epoch(model: torch.nn.Module, data: dict[str, np.ndarray], indices: np.ndarray,
                batch_size: int, optimizer: torch.optim.Optimizer, device: torch.device,
                order_rng: np.random.Generator, precision: str = "fp32",
                scaler: Any = None, max_grad_norm: float = 10.0) -> dict[str, float]:
    model.train()
    shuffled = order_rng.permutation(indices)
    running_loss = 0.0
    rows = 0
    optimizer_steps = 0
    overflow_steps = 0
    for start in range(0, len(shuffled), batch_size):
        current = shuffled[start:start + batch_size]
        batch = batch_tensors(data, current, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, precision):
            outputs = model(batch["spatial"], batch["global_input"])
        loss = loss_and_metrics(outputs, batch)["total_loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        if scaler is not None and scaler.is_enabled():
            # GradScaler must see overflow and lower its scale. A finite FP32
            # loss may still overflow while backpropagating scaled FP16 values.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=False)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < previous_scale:
                overflow_steps += 1
            else:
                optimizer_steps += 1
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            optimizer_steps += 1
        running_loss += float(loss.item()) * len(current)
        rows += len(current)
    return {"total_loss": running_loss / rows, "rows": rows,
            "optimizer_steps": optimizer_steps, "scaled_overflow_steps": overflow_steps}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def cpu_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def learning_rate_for_epoch(args: argparse.Namespace, epoch: int) -> float:
    """Deterministic epoch schedule; resuming needs no separate scheduler state."""
    if args.lr_schedule == "constant" or args.epochs == 1:
        return args.lr
    fraction = (epoch - 1) / (args.epochs - 1)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * fraction))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0 or args.max_grad_norm <= 0:
        raise ValueError("epochs, batch-size, lr and max-grad-norm must be positive")
    if any(value < 0 for value in (args.train_rows, args.val_rows, args.test_rows)):
        raise ValueError("row caps must be nonnegative (0 means all)")
    if not args.variants or len(args.variants) != len(set(args.variants)):
        raise ValueError("variants must be nonempty and distinct")
    if args.lr_schedule == "cosine" and not 0 < args.min_lr <= args.lr:
        raise ValueError("cosine min-lr must be positive and no greater than lr")
    started = time.perf_counter()
    device = resolve_device(args.device)
    if device.type == "cuda":
        isolate_windows_cudnn(torch)
    precision = resolve_precision(args.precision, device)
    data, data_info = load_selected(args.cache_dir, {
        "train": args.train_rows, "val": args.val_rows, "test": args.test_rows,
    }, args.selection_seed)
    indices = {name: np.flatnonzero(data["split_code"] == code) for name, code in SPLIT.items()}
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.resume:
        raise FileExistsError("output directory is not empty; use a new directory or --resume")
    if args.resume and not (args.out_dir / "manifest.json").is_file():
        raise ValueError("--resume requires an existing training manifest and last checkpoints")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "device": str(device), "platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": (torch.cuda.get_device_name(device) if device.type == "cuda" else None),
            "cuda_capability": (list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None),
            "isolated_system_cudnn_directories": os.environ.get("RUST_KATAGO_SYSTEM_CUDNN_PATHS", "").split(";") if os.environ.get("RUST_KATAGO_SYSTEM_CUDNN_PATHS") else [],
        },
        "config": {
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "training_seed": args.seed, "score_scale_points": SCORE_SCALE,
            "loss": "policy_ce + 0.5*value_ce + 0.1*score_smooth_l1(score/20) + 0.2*ownership_mse(tanh)",
            "train_loss_aggregation": "mean of per-batch normalized total loss, weighted by batch row count; validation and test aggregate each head by its effective row weight",
            "selection_metric": "validation total_loss; test evaluated after final checkpoint selection for this invocation; resume performs a new test evaluation",
            "variants": args.variants, "precision": precision,
            "max_grad_norm": args.max_grad_norm,
            "lr_schedule": args.lr_schedule,
            "minimum_lr": args.min_lr if args.lr_schedule == "cosine" else args.lr,
        },
        "data": data_info,
        "models": {},
        "scope": "four-head student training with automatic native CUDA .rgmodel export; no playing-strength certification",
    }
    manifest_path = args.out_dir / "manifest.json"
    # The epoch target may be extended; all data, optimizer and numerical choices
    # stay bound to the saved run. Resume is an epoch-boundary operation.
    resume_identity = {
        "data": data_info, "seed": args.seed, "batch_size": args.batch_size,
        "lr": args.lr, "precision": precision, "max_grad_norm": args.max_grad_norm,
        "variants": args.variants, "score_scale_points": SCORE_SCALE,
    }
    # Preserve historical constant-LR identities. A cosine run fixes its whole
    # horizon before test evaluation, so changing the epoch budget changes its
    # identity rather than silently restarting or stretching the schedule.
    if args.lr_schedule == "cosine":
        resume_identity["learning_rate_schedule"] = {
            "kind": "cosine", "total_epochs": args.epochs,
            "initial_lr": args.lr, "minimum_lr": args.min_lr,
        }
    result["resume_identity"] = resume_identity
    resumed: dict[str, Any] = {}
    previous: dict[str, Any] = {}
    best_recoveries: dict[str, str | None] = {}
    if args.resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("schema") != SCHEMA or previous.get("resume_identity") != resume_identity:
            raise ValueError("resume manifest configuration or data identity mismatch")
        for variant in args.variants:
            path = args.out_dir / f"{variant}.last.pt"
            if not path.is_file():
                if previous.get("models", {}).get(variant, {}).get("epochs"):
                    raise ValueError(f"{variant}: last checkpoint is missing after a completed epoch")
                continue  # No epoch committed yet; restart that variant from its seed.
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved.get("schema") != RESUME_SCHEMA or saved.get("identity") != resume_identity or saved.get("variant") != variant:
                raise ValueError(f"{variant}: resume checkpoint configuration or data identity mismatch")
            if saved["epoch"] > args.epochs:
                raise ValueError(f"{variant}: --epochs must not precede the saved epoch {saved['epoch']}")
            selected_path = args.out_dir / f"{variant}.pt"
            actual_sha = sha256_file(selected_path) if selected_path.is_file() else None
            if actual_sha != saved["best_checkpoint_sha256"]:
                # The last epoch is authoritative: an interrupted write can
                # publish a newer best file before the last checkpoint commits.
                # Preserve that file, then restore the embedded best checkpoint.
                best_payload = saved.get("best_checkpoint")
                if not isinstance(best_payload, dict) or best_payload.get("variant") != variant or best_payload.get("epoch") != saved["best_epoch"]:
                    raise ValueError(f"{variant}: changed best checkpoint cannot be recovered from last epoch")
                best_recoveries[variant] = actual_sha
            resumed[variant] = saved
        if previous.get("status") == "completed" and len(resumed) == len(args.variants) and all(
            saved["epoch"] == args.epochs for saved in resumed.values()
        ):
            raise ValueError("--epochs must exceed the saved epoch when the whole run is already completed")
        # All resume identities are checked before recovering any files.
        for variant, actual_sha in best_recoveries.items():
            selected_path = args.out_dir / f"{variant}.pt"
            if actual_sha is not None:
                preserved = args.out_dir / f"{variant}.best-before-resume.{actual_sha}.pt"
                original = selected_path.read_bytes()
                if hashlib.sha256(original).hexdigest() != actual_sha:
                    raise ValueError(f"{variant}: best checkpoint changed during resume recovery")
                if preserved.exists() and preserved.read_bytes() != original:
                    raise ValueError(f"{variant}: recovery preservation file already differs")
                preserved.write_bytes(original)
            atomic_checkpoint(selected_path, resumed[variant]["best_checkpoint"])
    atomic_json(manifest_path, result)
    for variant in args.variants:
        previous_info = previous.get("models", {}).get(variant, {})
        if variant in resumed and resumed[variant]["epoch"] == args.epochs and "test" in previous_info:
            exported = args.out_dir / f"{variant}.rgmodel"
            if exported.is_file() and sha256_file(exported) == previous_info.get("engine_model_sha256"):
                model_info = dict(previous_info)
                model_info["resumed_skipped_completed"] = True
                result["models"][variant] = model_info
                atomic_json(manifest_path, result)
                continue
        torch.manual_seed(args.seed)
        model = make_model(variant).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))
        order_rng = np.random.default_rng(args.seed)
        model_info: dict[str, Any] = {"parameters": count_parameters(model), "epochs": []}
        best_loss = float("inf")
        best_epoch = None
        checkpoint_path = args.out_dir / f"{variant}.pt"
        last_path = args.out_dir / f"{variant}.last.pt"
        start_epoch = 1
        if variant in resumed:
            saved = resumed[variant]
            model.load_state_dict(saved["model_state"], strict=True)
            optimizer.load_state_dict(saved["optimizer_state"])
            scaler.load_state_dict(saved["scaler_state"])
            order_rng.bit_generator.state = saved["order_rng_state"]
            torch.set_rng_state(saved["torch_rng_state"])
            if device.type == "cuda":
                torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
            model_info = saved["model_info"]
            # Test evaluation belongs to the old stopping point, not the new one.
            for key in ("test", "checkpoint", "checkpoint_sha256", "training_seconds"):
                model_info.pop(key, None)
            best_loss = saved["best_validation_total_loss"]
            best_epoch = saved["best_epoch"]
            start_epoch = saved["epoch"] + 1
            model_info["resumed_from_epoch"] = saved["epoch"]
            if variant in best_recoveries:
                model_info["recovered_best_from_last_epoch"] = True
                model_info["best_before_resume_sha256"] = best_recoveries[variant]
        model_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(start_epoch, args.epochs + 1):
            current_lr = learning_rate_for_epoch(args, epoch)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr
            train = train_epoch(model, data, indices["train"], args.batch_size,
                                optimizer, device, order_rng, precision, scaler, args.max_grad_norm)
            validation = evaluate(model, data, indices["val"], args.batch_size, device, precision)
            if not np.isfinite(train["total_loss"]) or not np.isfinite(validation["total_loss"]):
                raise FloatingPointError(f"{variant}: non-finite loss")
            model_info["epochs"].append({"epoch": epoch, "learning_rate": current_lr,
                                         "train": train, "validation": validation})
            print(f"[student] {variant} epoch={epoch}/{args.epochs} "
                  f"train={train['total_loss']:.4f} val={validation['total_loss']:.4f} "
                  f"policy_ce={validation['policy_ce']:.4f} lr={current_lr:.8g}", file=sys.stderr, flush=True)
            if validation["total_loss"] < best_loss:
                best_loss, best_epoch = validation["total_loss"], epoch
                atomic_checkpoint(checkpoint_path, {
                    "variant": variant, "model_state": cpu_model_state(model),
                    "score_scale_points": SCORE_SCALE, "epoch": epoch,
                    "training_manifest": str(manifest_path.resolve()),
                    "input_contract": "spatial float32 [B,22,19,19] from MSB packed; global float32 [B,19]",
                })
            model_info["best_epoch"] = best_epoch
            model_info["best_validation_total_loss"] = best_loss
            result["models"][variant] = model_info
            atomic_checkpoint(last_path, {
                "schema": RESUME_SCHEMA, "identity": resume_identity,
                "variant": variant, "epoch": epoch, "model_state": cpu_model_state(model),
                "optimizer_state": optimizer.state_dict(), "scaler_state": scaler.state_dict(),
                "order_rng_state": order_rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "best_epoch": best_epoch, "best_validation_total_loss": best_loss,
                "best_checkpoint_sha256": sha256_file(checkpoint_path), "model_info": model_info,
                "best_checkpoint": torch.load(checkpoint_path, map_location="cpu", weights_only=True),
            })
            atomic_json(manifest_path, result)
        selected = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(selected["model_state"])
        model_info["test"] = evaluate(model, data, indices["test"], args.batch_size, device, precision)
        model_info["checkpoint"] = str(checkpoint_path.resolve())
        model_info["checkpoint_sha256"] = sha256_file(checkpoint_path)
        model_info["resume_checkpoint"] = str(last_path.resolve())
        model_info["resume_checkpoint_sha256"] = sha256_file(last_path)
        engine_model_path = args.out_dir / f"{variant}.rgmodel"
        export_receipt = export_checkpoint(checkpoint_path, engine_model_path)
        model_info["engine_model"] = str(engine_model_path.resolve())
        model_info["engine_model_sha256"] = sha256_file(engine_model_path)
        model_info["engine_model_bytes"] = export_receipt["bytes"]
        model_info["engine_export_receipt"] = str(engine_model_path.with_suffix(engine_model_path.suffix + ".receipt.json").resolve())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            model_info["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        model_info["training_seconds"] = time.perf_counter() - model_started
        result["models"][variant] = model_info
        atomic_json(manifest_path, result)
    result["status"] = "completed"
    result["duration_seconds"] = time.perf_counter() - started
    atomic_json(manifest_path, result)
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cuda",
                   help="default CUDA; unavailable CUDA is an error, never a CPU fallback")
    p.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto",
                   help="auto uses scaled FP16 on CUDA and FP32 elsewhere; losses remain FP32")
    p.add_argument("--variants", nargs="+", choices=("dense", "compact"), default=["dense", "compact"])
    p.add_argument("--resume", action="store_true", help="restore last complete epoch in out-dir; epochs is the new total")
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--train-rows", type=int, default=200000, help="0 means all")
    p.add_argument("--val-rows", type=int, default=20000, help="0 means all")
    p.add_argument("--test-rows", type=int, default=20000, help="0 means all")
    p.add_argument("--selection-seed", type=int, default=20260927)
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant",
                   help="constant preserves existing behavior; cosine fixes the full epoch budget for resume")
    p.add_argument("--min-lr", type=float, default=0.00001,
                   help="final epoch learning rate for cosine (positive and <= lr)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = run(args)
    except (ValueError, RuntimeError, OSError, FloatingPointError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "manifest": str(args.out_dir / "manifest.json"),
                      "duration_seconds": result["duration_seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
