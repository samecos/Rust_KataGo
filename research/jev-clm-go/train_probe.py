#!/usr/bin/env python3
"""Small, independent policy-target probe for v7 19x19 self-play NPZ files.

This is a deliberately limited comparison: a tiny CNN and a normalized
state/action similarity head versus the same CNN and an equal-parameter
two-linear-layer policy head. Both predict all 362 policy slots from visit
targets. No legal-move replay, principal variation encoding, B11 weight
sharing, engine integration, or playing-strength claim is made here.

Only packed uint8 spatial inputs, float32 global inputs, int16 policy target0,
float32 row weights, and split codes are kept after loading. Spatial planes
are unpacked in each minibatch. globalTargetsNC is used for labels, weights,
and grouping only; it is never passed to a model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any
import zipfile

import numpy as np

import audit_npz
from cuda_environment import isolate_windows_cudnn

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    torch = None
    nn = None
    F = None


SPLIT_CODE = {"train": 0, "val": 1, "test": 2}
PROBE_SCHEMA = "rust_katago_clm_policy_probe_v1"
INPUT_CHANNELS = 22
GLOBAL_CHANNELS = 19
TRUNK_FEATURES = 32 * 5 * 5 + GLOBAL_CHANNELS


@dataclass
class ProbeData:
    packed: np.ndarray
    global_input: np.ndarray
    target: np.ndarray
    row_weight: np.ndarray
    split_code: np.ndarray
    split_seed: str
    games_per_split: dict[str, int]
    audit: dict[str, Any]

    def indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.split_code == SPLIT_CODE[split])

    def summary(self) -> dict[str, Any]:
        return {
            "eligible_rows": int(len(self.split_code)),
            "rows_per_split": {name: int(np.count_nonzero(self.split_code == code))
                               for name, code in SPLIT_CODE.items()},
            "games_per_split": self.games_per_split,
            "split_seed": self.split_seed,
            "cached_arrays": {
                "packed_uint8_shape": list(self.packed.shape),
                "global_float32_shape": list(self.global_input.shape),
                "target0_int16_shape": list(self.target.shape),
                "row_weight_float32_shape": list(self.row_weight.shape),
            },
        }


def split_codes_for_hashes(
    game_hashes: list[tuple[int, int]], split_seed: str
) -> np.ndarray:
    """Use audit_npz's exact game-level SHA-256 splitter, with a game cache."""
    by_game: dict[tuple[int, int], int] = {}
    codes = np.empty(len(game_hashes), dtype=np.uint8)
    for row, game_hash in enumerate(game_hashes):
        if game_hash not in by_game:
            by_game[game_hash] = SPLIT_CODE[audit_npz._split_for_game(game_hash, split_seed)]
        codes[row] = by_game[game_hash]
    return codes


def _read_checked(
    archive: zipfile.ZipFile, name: str, limit_bytes: int, rows: int
) -> np.ndarray:
    array = audit_npz._read_member(archive, name, limit_bytes)
    problem = audit_npz._validate_array(name, array, rows)
    if problem:
        raise ValueError(problem)
    return array


def _full_board_mask(packed: np.ndarray) -> np.ndarray:
    onboard = np.unpackbits(packed[:, 0, :], axis=1, bitorder="big")[:, :audit_npz.AREA]
    return np.all(onboard == 1, axis=1)


def _eligible_mask(
    global_target: np.ndarray, policy: np.ndarray,
    packed: np.ndarray | None = None, full_board_only: bool = False,
) -> np.ndarray:
    if np.any(~np.isfinite(global_target[:, 25])):
        raise ValueError("globalTargetsNC[25] contains non-finite row weights")
    if np.any(global_target[:, 25] < 0):
        raise ValueError("globalTargetsNC[25] contains negative row weights")
    if np.any(policy < 0):
        raise ValueError("policyTargetsNCMove contains negative visit counts")
    target0 = policy[:, 0, :]
    eligible = ((global_target[:, 26] == 1) & (global_target[:, 25] > 0)
                & (target0.sum(axis=1, dtype=np.int64) > 0))
    if full_board_only:
        if packed is None:
            raise ValueError("packed input is required for full-board selection")
        eligible &= _full_board_mask(packed)
    return eligible


def _audit_error_message(report: dict[str, Any]) -> str:
    problems = list(report.get("errors", []))
    for item in report.get("files", []):
        problems.extend(f"{item['path']}: {e}" for e in item.get("errors", []))
    return "; ".join(problems[:8]) or "unknown NPZ audit error"


def load_probe_data(
    paths: list[Path], split_seed: str = "clm-v1", max_entry_mib: int = 512,
    allow_missing_qvalue: bool = False, full_board_only: bool = False,
) -> ProbeData:
    """Validate real v7 archives first, then cache only the four input/label arrays.

    Each ZIP is read twice after the audit: first to count eligible rows so
    arrays can be allocated exactly once, then to fill those arrays. This
    avoids a dataset-wide concatenate copy.
    """
    audit = audit_npz.audit_paths(
        paths, max_entry_mib=max_entry_mib, split_seed=split_seed,
        allow_missing_qvalue=allow_missing_qvalue, full_board_only=full_board_only,
    )
    if audit["status"] != "ok":
        raise ValueError("NPZ audit failed: " + _audit_error_message(audit))
    files = [(Path(item["path"]), int(item["rows"])) for item in audit["files"]]
    limit_bytes = max_entry_mib * 1024 * 1024
    eligible_per_file: list[int] = []
    for path, rows in files:
        with zipfile.ZipFile(path) as archive:
            global_target = _read_checked(archive, "globalTargetsNC", limit_bytes, rows)
            policy = _read_checked(archive, "policyTargetsNCMove", limit_bytes, rows)
            packed_file = (_read_checked(archive, "binaryInputNCHWPacked", limit_bytes, rows)
                           if full_board_only else None)
            eligible_per_file.append(int(_eligible_mask(
                global_target, policy, packed_file, full_board_only
            ).sum()))
    total = sum(eligible_per_file)
    if total == 0:
        raise ValueError("no rows have valid policy target0 and positive row weight")

    packed = np.empty((total, INPUT_CHANNELS, audit_npz.PACKED_AREA), dtype=np.uint8)
    global_input = np.empty((total, GLOBAL_CHANNELS), dtype=np.float32)
    target = np.empty((total, audit_npz.POLICY_SIZE), dtype=np.int16)
    row_weight = np.empty(total, dtype=np.float32)
    split_code = np.empty(total, dtype=np.uint8)
    game_sets = {name: set() for name in SPLIT_CODE}
    cursor = 0
    for (path, rows), expected_count in zip(files, eligible_per_file):
        with zipfile.ZipFile(path) as archive:
            global_target = _read_checked(archive, "globalTargetsNC", limit_bytes, rows)
            policy = _read_checked(archive, "policyTargetsNCMove", limit_bytes, rows)
            packed_file = (_read_checked(archive, "binaryInputNCHWPacked", limit_bytes, rows)
                           if full_board_only else None)
            selected = np.flatnonzero(_eligible_mask(
                global_target, policy, packed_file, full_board_only
            ))
            if len(selected) != expected_count:
                raise ValueError(f"{path}: eligible row count changed between passes")
            if not len(selected):
                continue
            if packed_file is None:
                packed_file = _read_checked(archive, "binaryInputNCHWPacked", limit_bytes, rows)
            global_file = _read_checked(archive, "globalInputNC", limit_bytes, rows)
            end = cursor + len(selected)
            packed[cursor:end] = packed_file[selected]
            global_input[cursor:end] = global_file[selected]
            target[cursor:end] = policy[selected, 0, :]
            row_weight[cursor:end] = global_target[selected, 25]
            hashes, bad_hashes = audit_npz._game_hashes(global_target[selected])
            if bad_hashes:
                raise ValueError(f"{path}: {bad_hashes} selected rows have invalid game hashes")
            selected_hashes = [value for value in hashes if value is not None]
            codes = split_codes_for_hashes(selected_hashes, split_seed)
            split_code[cursor:end] = codes
            for game_hash, code in zip(selected_hashes, codes):
                game_sets[audit_npz.SPLIT_NAMES[int(code)]].add(game_hash)
            cursor = end
    if cursor != total:
        raise RuntimeError("dataset allocation count does not match filled rows")
    if any(game_sets[a] & game_sets[b] for a in SPLIT_CODE for b in SPLIT_CODE if a < b):
        raise RuntimeError("game hash leaked across splits")
    return ProbeData(
        packed=packed,
        global_input=global_input,
        target=target,
        row_weight=row_weight,
        split_code=split_code,
        split_seed=split_seed,
        games_per_split={name: len(games) for name, games in game_sets.items()},
        audit=audit,
    )


def unpack_batch(packed: np.ndarray) -> np.ndarray:
    """MSB-first v7 unpacking; discard the seven padding bits per plane."""
    bits = np.unpackbits(packed, axis=2, bitorder="big")[:, :, :audit_npz.AREA]
    return bits.reshape(len(packed), INPUT_CHANNELS, 19, 19).astype(np.float32)


if torch is not None:
    class TinyTrunk(nn.Module):
        """Same compact CNN architecture and initialization for both arms."""

        def __init__(self) -> None:
            super().__init__()
            self.convs = nn.Sequential(
                nn.Conv2d(INPUT_CHANNELS, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
            )

        def forward(self, spatial: torch.Tensor, global_input: torch.Tensor) -> torch.Tensor:
            # MPS cannot adaptively pool 19x19 to 4x4; this fixed pool covers
            # every board edge and yields 5x5 on both CPU and MPS.
            pooled = F.avg_pool2d(self.convs(spatial), kernel_size=4,
                                  stride=4, padding=1).flatten(1)
            return torch.cat((pooled, global_input), dim=1)


    class SimilarityHead(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.state_proj = nn.Linear(TRUNK_FEATURES, width, bias=False)
            self.action_vectors = nn.Parameter(torch.empty(audit_npz.POLICY_SIZE, width))
            self.logit_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))
            nn.init.xavier_uniform_(self.action_vectors)

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            state = F.normalize(self.state_proj(features), dim=1)
            actions = F.normalize(self.action_vectors, dim=1)
            return (state @ actions.T) * self.logit_scale.exp().clamp(max=100)


    class LinearPolicyHead(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.state_proj = nn.Linear(TRUNK_FEATURES, width, bias=False)
            self.output = nn.Linear(width, audit_npz.POLICY_SIZE, bias=False)
            self.logit_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return self.output(self.state_proj(features)) * self.logit_scale.exp().clamp(max=100)


    class ProbeNet(nn.Module):
        def __init__(self, head_kind: str, width: int) -> None:
            super().__init__()
            self.trunk = TinyTrunk()
            self.head = SimilarityHead(width) if head_kind == "similarity" else LinearPolicyHead(width)

        def forward(self, spatial: torch.Tensor, global_input: torch.Tensor) -> torch.Tensor:
            return self.head(self.trunk(spatial, global_input))


    def make_paired_models(seed: int, width: int) -> tuple[ProbeNet, ProbeNet]:
        """Give both arms exactly the same starting tensors and parameter count."""
        torch.manual_seed(seed)
        similarity = ProbeNet("similarity", width)
        linear = ProbeNet("linear", width)
        linear.trunk.load_state_dict(similarity.trunk.state_dict())
        linear.head.state_proj.load_state_dict(similarity.head.state_proj.state_dict())
        with torch.no_grad():
            linear.head.output.weight.copy_(similarity.head.action_vectors)
            linear.head.logit_scale.copy_(similarity.head.logit_scale)
        if head_parameter_count(similarity) != head_parameter_count(linear):
            raise RuntimeError("comparison heads have unequal parameter counts")
        return similarity, linear


    def head_parameter_count(model: ProbeNet) -> int:
        return sum(parameter.numel() for parameter in model.head.parameters())


    def _batch_tensors(
        data: ProbeData, row_indices: np.ndarray, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial = torch.from_numpy(unpack_batch(data.packed[row_indices])).to(device)
        global_input = torch.from_numpy(data.global_input[row_indices].copy()).to(device)
        target = torch.from_numpy(data.target[row_indices].astype(np.float32)).to(device)
        weight = torch.from_numpy(data.row_weight[row_indices].copy()).to(device)
        return spatial, global_input, target, weight


    def _loss_terms(
        logits: torch.Tensor, counts: torch.Tensor, weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probabilities = counts / counts.sum(dim=1, keepdim=True)
        log_probabilities = F.log_softmax(logits.float(), dim=1)
        cross_entropy = -(probabilities * log_probabilities).sum(dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=1)
        divergence = (cross_entropy - entropy).clamp_min(0)
        agreement = (logits.argmax(dim=1) == counts.argmax(dim=1)).float()
        total_weight = weights.sum()
        return (
            (cross_entropy * weights).sum() / total_weight,
            (divergence * weights).sum() / total_weight,
            (agreement * weights).sum() / total_weight,
        )


    def train_epoch(
        model: ProbeNet,
        data: ProbeData,
        row_indices: np.ndarray,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        rng: np.random.Generator,
    ) -> float:
        model.train()
        order = rng.permutation(row_indices)
        weighted_sum = 0.0
        weight_sum = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            spatial, global_input, target, weight = _batch_tensors(data, indices, device)
            optimizer.zero_grad(set_to_none=True)
            cross_entropy, _, _ = _loss_terms(model(spatial, global_input), target, weight)
            if not torch.isfinite(cross_entropy):
                raise FloatingPointError("training loss is non-finite")
            cross_entropy.backward()
            optimizer.step()
            batch_weight = float(weight.sum().item())
            weighted_sum += float(cross_entropy.item()) * batch_weight
            weight_sum += batch_weight
        return weighted_sum / weight_sum


    @torch.no_grad()
    def evaluate(
        model: ProbeNet, data: ProbeData, row_indices: np.ndarray,
        batch_size: int, device: torch.device
    ) -> dict[str, float | int]:
        model.eval()
        totals = np.zeros(3, dtype=np.float64)
        total_weight = 0.0
        for start in range(0, len(row_indices), batch_size):
            indices = row_indices[start:start + batch_size]
            spatial, global_input, target, weight = _batch_tensors(data, indices, device)
            metrics = _loss_terms(model(spatial, global_input), target, weight)
            batch_weight = float(weight.sum().item())
            totals += np.array([float(metric.item()) for metric in metrics]) * batch_weight
            total_weight += batch_weight
        if total_weight <= 0:
            raise ValueError("evaluation split has no positive row weight")
        values = totals / total_weight
        return {
            "cross_entropy": float(values[0]),
            "kl_teacher_to_model": float(values[1]),
            "top1_teacher_agreement": float(values[2]),
            "rows": int(len(row_indices)),
            "row_weight_sum": total_weight,
        }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parse_seeds(text: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("provide distinct comma-separated seeds")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="v7 NPZ files or directories")
    parser.add_argument("--out-dir", required=True, type=Path, help="directory for manifest.json")
    parser.add_argument("--split-seed", default="clm-v1", help="stable game-hash split seed")
    parser.add_argument("--seeds", type=_parse_seeds, default=[11, 29, 47],
                        help="distinct training seeds, comma-separated (default: 11,29,47)")
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cuda",
                        help="default CUDA; unavailable CUDA is an error; auto explicitly allows fallback")
    parser.add_argument("--max-entry-mib", type=int, default=512)
    parser.add_argument("--allow-missing-qvalue", action="store_true",
                        help="allow NPZ without Q labels for this policy-only probe")
    parser.add_argument("--full-board-only", action="store_true",
                        help="exclude non-19x19 rows, required for mixed-board data")
    parser.add_argument("--inspect-only", action="store_true",
                        help="audit and report eligible data without requiring torch")
    return parser


def resolve_device(requested: str) -> "torch.device":
    if torch is None:
        raise RuntimeError("PyTorch (torch) is required to select a training device")
    if requested == "auto":
        requested = ("cuda" if torch.cuda.is_available() else
                     "mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS device requested but torch.backends.mps.is_available() is false")
    if requested == "cuda":
        isolate_windows_cudnn(torch)
    return torch.device(requested)


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_started = time.perf_counter()
    if args.head_dim <= 0 or args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        raise ValueError("head-dim, epochs, batch-size, and lr must be positive")
    if args.max_entry_mib <= 0:
        raise ValueError("max-entry-mib must be positive")
    if not args.inspect_only and torch is None:
        raise RuntimeError(
            "PyTorch (torch) is required for training but is not installed. "
            "Run --inspect-only for NPZ and split inspection, or install torch "
            "in a separate training environment."
        )
    data = load_probe_data(
        args.paths, args.split_seed, args.max_entry_mib,
        allow_missing_qvalue=args.allow_missing_qvalue,
        full_board_only=args.full_board_only,
    )
    result: dict[str, Any] = {
        "schema": PROBE_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "inspected" if args.inspect_only else "running",
        "scope": "restricted 362-slot visit-target probe; no legal replay, full CLM, or Go strength claim",
        "config": {
            "split_seed": args.split_seed, "training_seeds": args.seeds,
            "head_dim": args.head_dim, "epochs": args.epochs,
            "batch_size": args.batch_size, "lr": args.lr, "device": args.device,
            "max_entry_mib": args.max_entry_mib,
            "allow_missing_qvalue": args.allow_missing_qvalue,
            "full_board_only": args.full_board_only,
        },
        "data": data.summary(),
        "source_files": [
            {"path": item["path"], "sha256": item["sha256"], "rows": item["rows"]}
            for item in data.audit["files"]
        ],
        "audit": {
            "schema": data.audit["schema"],
            "status": data.audit["status"],
            "duplicate_game_turn_rows": data.audit["split"]["duplicate_game_turn_rows"],
        },
        "selection_policy": "best epoch by validation cross-entropy for each seed/arm; test once after selection",
        "primary_metric": "held-out test cross_entropy; lower is better; paired by training seed",
    }
    if args.inspect_only:
        return result
    assert torch is not None
    indices = {name: data.indices(name) for name in SPLIT_CODE}
    empty = [name for name, values in indices.items() if not len(values)]
    if empty:
        raise ValueError(
            "game-hash split has no eligible rows in " + ", ".join(empty)
            + "; collect more independent games or change the predeclared split seed"
        )
    device = resolve_device(args.device)
    result["config"]["selected_device"] = str(device)
    result["environment"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "machine": platform.machine(),
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
        "isolated_system_cudnn_directories": [part for part in os.environ.get("RUST_KATAGO_SYSTEM_CUDNN_PATHS", "").split(";") if part],
    }
    results: list[dict[str, Any]] = []
    for seed in args.seeds:
        similarity, linear = make_paired_models(seed, args.head_dim)
        for arm, model in (("similarity", similarity), ("linear", linear)):
            arm_started = time.perf_counter()
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            rng = np.random.default_rng(seed)
            best_epoch = -1
            best_val: dict[str, float | int] | None = None
            best_state: dict[str, torch.Tensor] | None = None
            for epoch in range(1, args.epochs + 1):
                train_ce = train_epoch(
                    model, data, indices["train"], args.batch_size, optimizer, device, rng
                )
                val = evaluate(model, data, indices["val"], args.batch_size, device)
                print(
                    f"[probe] seed={seed} arm={arm} epoch={epoch}/{args.epochs} "
                    f"train_ce={train_ce:.6f} val_ce={val['cross_entropy']:.6f}",
                    file=sys.stderr, flush=True,
                )
                if not np.isfinite(train_ce) or not np.isfinite(val["cross_entropy"]):
                    raise FloatingPointError(f"{arm} seed {seed}: non-finite training/validation loss")
                if best_val is None or val["cross_entropy"] < best_val["cross_entropy"]:
                    best_epoch, best_val = epoch, val
                    best_state = {name: value.detach().cpu().clone()
                                  for name, value in model.state_dict().items()}
            assert best_state is not None and best_val is not None
            model.load_state_dict(best_state)
            test = evaluate(model, data, indices["test"], args.batch_size, device)
            results.append({
                "seed": seed, "arm": arm, "head_parameters": head_parameter_count(model),
                "trunk_parameters": sum(p.numel() for p in model.trunk.parameters()),
                "best_epoch": best_epoch, "validation": best_val, "test": test,
                "duration_seconds": time.perf_counter() - arm_started,
            })
            model.to("cpu")
    result["runs"] = results
    result["summary"] = {}
    for arm in ("similarity", "linear"):
        arm_runs = [entry for entry in results if entry["arm"] == arm]
        result["summary"][arm] = {}
        for split in ("validation", "test"):
            result["summary"][arm][split] = {
                metric: {
                    "mean": float(np.mean([entry[split][metric] for entry in arm_runs])),
                    "std_population": float(np.std([entry[split][metric] for entry in arm_runs])),
                }
                for metric in ("cross_entropy", "kl_teacher_to_model", "top1_teacher_agreement")
            }
    result["status"] = "completed"
    result["duration_seconds"] = time.perf_counter() - run_started
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    try:
        result = run(args)
    except (ValueError, RuntimeError, FloatingPointError, OSError, zipfile.BadZipFile) as exc:
        error = {
            "schema": PROBE_SCHEMA,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        _atomic_json(manifest_path, error)
        print(error["error"], file=sys.stderr)
        return 2
    _atomic_json(manifest_path, result)
    print(json.dumps({
        "status": result["status"],
        "eligible_rows": result["data"]["eligible_rows"],
        "manifest": str(manifest_path),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
