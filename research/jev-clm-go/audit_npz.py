#!/usr/bin/env python3
"""Read-only audit of v7-input, v2/v3-target 19x19 self-play archives.

The repository writes ZIP members with bare names, each containing a complete
.npy payload. This tool also accepts conventional ``name.npy`` members. It
does not construct training examples or treat policy targets as legal masks.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any
import zipfile

import numpy as np


AREA = 19 * 19
POLICY_SIZE = AREA + 1
PACKED_AREA = (AREA + 7) // 8
SCORE_DISTR_SIZE = 2 * AREA + 120
SCHEMA: dict[str, tuple[tuple[int, ...], str]] = {
    "binaryInputNCHWPacked": ((22, PACKED_AREA), "u1"),
    "globalInputNC": ((19,), "f4"),
    "policyTargetsNCMove": ((2, POLICY_SIZE), "i2"),
    "globalTargetsNC": ((64,), "f4"),
    "scoreDistrN": ((SCORE_DISTR_SIZE,), "i1"),
    "valueTargetsNCHW": ((5, 19, 19), "i1"),
    "qValueTargetsNCMove": ((3, POLICY_SIZE), "i2"),
}
GLOBAL_TARGET_VERSION_BY_WIDTH = {64: 2, 80: 3}
HASH_RADICES = (1 << 22, 1 << 22, 1 << 20) * 2
SPLIT_NAMES = ("train", "val", "test")


def _split_for_game(game_hash: tuple[int, int], seed: str) -> str:
    material = b"RustKataGo-CLM-split-v1\0" + seed.encode("utf-8") + b"\0"
    material += game_hash[0].to_bytes(8, "little")
    material += game_hash[1].to_bytes(8, "little")
    bucket = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") * 10_000 // (1 << 64)
    return "train" if bucket < 8_000 else "val" if bucket < 9_000 else "test"


def _read_member(archive: zipfile.ZipFile, name: str, limit_bytes: int) -> np.ndarray:
    candidates = [n for n in (name, name + ".npy") if n in archive.namelist()]
    if len(candidates) != 1:
        raise ValueError(f"{name}: expected exactly one bare or .npy member, found {len(candidates)}")
    member = candidates[0]
    info = archive.getinfo(member)
    if info.file_size > limit_bytes:
        raise ValueError(f"{member}: {info.file_size} uncompressed bytes exceed per-member limit")
    with archive.open(info) as stream:
        payload = stream.read(limit_bytes + 1)
    if len(payload) > limit_bytes:
        raise ValueError(f"{member}: decompressed payload exceeds per-member limit")
    if not payload.startswith(b"\x93NUMPY"):
        raise ValueError(f"{member}: ZIP member has no .npy magic")
    array = np.load(io.BytesIO(payload), allow_pickle=False)
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{member}: expected a NumPy array")
    return array


def _validate_array(name: str, array: np.ndarray, expected_rows: int | None) -> str | None:
    tail, dtype = SCHEMA[name]
    if name == "globalTargetsNC":
        if array.ndim != 2 or array.shape[1] not in GLOBAL_TARGET_VERSION_BY_WIDTH:
            return f"{name}: expected [N,64] or [N,80], got {list(array.shape)}"
    elif array.ndim != len(tail) + 1 or array.shape[1:] != tail:
        return f"{name}: expected [N,{','.join(map(str, tail))}], got {list(array.shape)}"
    if array.dtype.kind + str(array.dtype.itemsize) != dtype:
        return f"{name}: expected dtype {dtype}, got {array.dtype}"
    if expected_rows is not None and array.shape[0] != expected_rows:
        return f"{name}: row count {array.shape[0]} differs from {expected_rows}"
    return None


def _game_hashes(global_targets: np.ndarray) -> tuple[list[tuple[int, int] | None], int]:
    chunks = global_targets[:, 41:47].astype(np.float64, copy=False)
    finite = np.isfinite(chunks).all(axis=1)
    integral = (chunks == np.floor(chunks)).all(axis=1)
    in_range = np.ones(len(chunks), dtype=bool)
    for column, radix in enumerate(HASH_RADICES):
        in_range &= (chunks[:, column] >= 0) & (chunks[:, column] < radix)
    valid = finite & integral & in_range
    result: list[tuple[int, int] | None] = []
    for row, okay in zip(chunks, valid):
        if not okay:
            result.append(None)
            continue
        a, b, c, d, e, f = (int(x) for x in row)
        result.append((a | b << 22 | c << 44, d | e << 22 | f << 44))
    return result, int((~valid).sum())


def _flag_count(values: np.ndarray, name: str, errors: list[str]) -> dict[str, int]:
    good = (values == 0) | (values == 1)
    bad = int((~good).sum())
    if bad:
        errors.append(f"{name}: {bad} rows have values outside {{0,1}}")
    return {"true": int((values == 1).sum()), "false": int((values == 0).sum()), "invalid": bad}


def audit_file(
    path: Path, limit_bytes: int, seed: str,
    allow_missing_qvalue: bool = False, full_board_only: bool = False,
) -> tuple[dict[str, Any], Counter, Counter]:
    report: dict[str, Any] = {"path": str(path), "status": "error", "errors": [], "warnings": []}
    game_rows: Counter = Counter()
    game_turns: Counter = Counter()
    try:
        file_hash = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                file_hash.update(chunk)
        report["sha256"] = file_hash.hexdigest()
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("duplicate ZIP member names")
            arrays: dict[str, np.ndarray] = {}
            rows: int | None = None
            for name in SCHEMA:
                if (name == "qValueTargetsNCMove" and allow_missing_qvalue
                        and name not in names and name + ".npy" not in names):
                    report["warnings"].append(
                        "qValueTargetsNCMove is absent; Q-based candidate analysis is unavailable"
                    )
                    continue
                array = _read_member(archive, name, limit_bytes)
                error = _validate_array(name, array, rows)
                if error:
                    raise ValueError(error)
                if rows is None:
                    rows = int(array.shape[0])
                arrays[name] = array
            if "metadataInputNC" in names or "metadataInputNC.npy" in names:
                meta = _read_member(archive, "metadataInputNC", limit_bytes)
                if meta.ndim != 2 or meta.shape != (rows, 192) or meta.dtype != np.dtype("float32"):
                    raise ValueError(f"metadataInputNC: expected float32 [N,192], got {meta.dtype} {list(meta.shape)}")
                report["warnings"].append("metadata input is present; native TF3 no-metadata deployment does not consume it")
            assert rows is not None
            report["rows"] = rows
            report["arrays"] = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()}
            if rows == 0:
                report["warnings"].append("archive contains zero rows")

            spatial = arrays["binaryInputNCHWPacked"]
            padding_nonzero = 0
            off_board_plane_rows = 0
            stone_overlap_rows = 0
            full_board_mask = np.zeros(rows, dtype=bool)
            for start in range(0, rows, 4096):
                bits = np.unpackbits(spatial[start:start + 4096], axis=2, bitorder="big")
                padding_nonzero += int(np.count_nonzero(bits[:, :, AREA:]))
                bits = bits[:, :, :AREA]
                board_mask = np.all(bits[:, 0, :] == 1, axis=1)
                full_board_mask[start:start + len(board_mask)] = board_mask
                off_board_plane_rows += int(np.count_nonzero(~board_mask))
                stone_overlap_rows += int(np.count_nonzero(np.any((bits[:, 1, :] & bits[:, 2, :]) != 0, axis=1)))
            report["spatial_msb"] = {
                "unpacked_positions": AREA,
                "nonzero_padding_bits": padding_nonzero,
                "full_19x19_rows": int(full_board_mask.sum()),
                "rows_with_nonfull_onboard_plane": off_board_plane_rows,
                "rows_with_both_stone_colors_on_one_point": stone_overlap_rows,
            }
            if padding_nonzero or stone_overlap_rows or (off_board_plane_rows and not full_board_only):
                report["errors"].append("MSB-unpacked binary features violate v7 19x19 invariants")
            if off_board_plane_rows and full_board_only:
                report["warnings"].append(
                    f"{off_board_plane_rows} non-19x19 rows excluded from game-level split and probe"
                )

            global_input = arrays["globalInputNC"]
            target = arrays["globalTargetsNC"]
            nonfinite_input = int(np.count_nonzero(~np.isfinite(global_input)))
            nonfinite_target = int(np.count_nonzero(~np.isfinite(target)))
            report["nonfinite"] = {"globalInputNC": nonfinite_input, "globalTargetsNC": nonfinite_target}
            if nonfinite_input or nonfinite_target:
                report["errors"].append("input or global target contains non-finite values")
            expected_version = GLOBAL_TARGET_VERSION_BY_WIDTH[target.shape[1]]
            report["target_schema_version"] = expected_version
            bad_version = int(np.count_nonzero(target[:, 63] != expected_version))
            report["target_schema_version_invalid_rows"] = bad_version
            if bad_version:
                report["errors"].append(
                    f"globalTargetsNC[63]: {bad_version} rows are not schema version {expected_version}"
                )
            if expected_version == 3:
                report["v3_reanalyzed"] = _flag_count(
                    target[:, 64], "globalTargetsNC[64]", report["errors"]
                )

            policy0 = _flag_count(target[:, 26], "globalTargetsNC[26]", report["errors"])
            policy1 = _flag_count(target[:, 28], "globalTargetsNC[28]", report["errors"])
            future = _flag_count(target[:, 33], "globalTargetsNC[33]", report["errors"])
            report["policy"] = {
                "current_target_available": policy0,
                "next_target_available": policy1,
                "note": "policyTargetsNCMove stores visit targets or a uniform fallback; nonzero entries are not a legal-move mask",
            }
            policy_targets = arrays["policyTargetsNCMove"]
            negative_policy_values = int(np.count_nonzero(policy_targets < 0))
            empty_valid_current = int(np.count_nonzero(
                (target[:, 26] == 1) & (policy_targets[:, 0, :].sum(axis=1, dtype=np.int64) <= 0)
            ))
            empty_valid_next = int(np.count_nonzero(
                (target[:, 28] == 1) & (policy_targets[:, 1, :].sum(axis=1, dtype=np.int64) <= 0)
            ))
            report["policy"]["negative_target_values"] = negative_policy_values
            report["policy"]["available_current_with_zero_mass"] = empty_valid_current
            report["policy"]["available_next_with_zero_mass"] = empty_valid_next
            if negative_policy_values or empty_valid_current or empty_valid_next:
                report["errors"].append("policy targets contain negative counts or a valid target has no mass")
            future_values = arrays["valueTargetsNCHW"][:, 2:4]
            invalid_future_values = int(np.count_nonzero((future_values < -1) | (future_values > 1)))
            false_future_nonzero_rows = int(np.count_nonzero(np.any(future_values[target[:, 33] == 0] != 0, axis=(1, 2, 3))))
            report["future_boards"] = {
                "target_available": future,
                "coverage_fraction": future["true"] / rows if rows else 0.0,
                "potential_side_rows": future["false"],
                "potential_side_fraction": future["false"] / rows if rows else 0.0,
                "invalid_occupancy_values": invalid_future_values,
                "unavailable_rows_with_nonzero_future_channels": false_future_nonzero_rows,
                "horizons_moves": [8, 32],
                "note": "both horizons share flag [33]; near the end of a game each target is clipped to the final board; [33]=0 is a side-position clue, not a definitive label",
            }
            if invalid_future_values or false_future_nonzero_rows:
                report["errors"].append("future board channels violate occupancy or availability invariants")

            hashes, bad_hash = _game_hashes(target)
            turns = target[:, 51]
            valid_turns = np.isfinite(turns) & (turns >= 0) & (turns == np.floor(turns))
            bad_turns = int(np.count_nonzero(~valid_turns))
            if bad_hash or bad_turns:
                report["errors"].append(f"invalid game hash rows={bad_hash}, invalid turn rows={bad_turns}")
            selected_mask = full_board_mask if full_board_only else np.ones(rows, dtype=bool)
            for game, turn, valid_turn, selected in zip(hashes, turns, valid_turns, selected_mask):
                if game is not None and selected:
                    game_rows[game] += 1
                    if valid_turn:
                        game_turns[(game, int(turn))] += 1
            report["game_hash"] = {
                "valid_rows": rows - bad_hash,
                "invalid_rows": bad_hash,
                "invalid_turn_rows": bad_turns,
                "selected_full_board_rows": int(selected_mask.sum()),
                "unique_games": len(game_rows),
                "duplicate_game_turn_rows": sum(n - 1 for n in game_turns.values() if n > 1),
                "split_rows": dict(Counter({name: sum(count for game, count in game_rows.items() if _split_for_game(game, seed) == name) for name in SPLIT_NAMES})),
            }
            if bad_hash:
                report["warnings"].append("rows with invalid game hash are excluded from hash-based splitting")
            report["status"] = "ok" if not report["errors"] else "error"
    except Exception as exc:
        # A malformed archive must produce a file-level error, never abort the
        # audit of other files. KeyboardInterrupt/SystemExit still propagate.
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    return report, game_rows, game_turns


def audit_paths(
    paths: list[Path], max_entry_mib: int = 512, split_seed: str = "clm-v1",
    allow_missing_qvalue: bool = False, full_board_only: bool = False,
) -> dict[str, Any]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.npz")))
        else:
            files.append(path)
    files = sorted(set(files))
    reports: list[dict[str, Any]] = []
    all_game_rows: Counter = Counter()
    all_game_turns: Counter = Counter()
    for path in files:
        report, games, turns = audit_file(
            path, max_entry_mib * 1024 * 1024, split_seed,
            allow_missing_qvalue=allow_missing_qvalue, full_board_only=full_board_only,
        )
        reports.append(report)
        if report["status"] == "ok":
            all_game_rows.update(games)
            all_game_turns.update(turns)
    split_rows = {name: 0 for name in SPLIT_NAMES}
    split_games = {name: 0 for name in SPLIT_NAMES}
    for game, count in all_game_rows.items():
        split = _split_for_game(game, split_seed)
        split_rows[split] += count
        split_games[split] += 1
    total_rows = sum(split_rows.values())
    total_games = sum(split_games.values())
    return {
        "schema": "rust_katago_v7_19_npz_audit_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "options": {
            "allow_missing_qvalue": allow_missing_qvalue,
            "full_board_only": full_board_only,
        },
        "status": "ok" if files and all(r["status"] == "ok" for r in reports) else "error",
        "files_found": len(files),
        "errors": [] if files else ["no .npz files found"],
        "files_ok": sum(r["status"] == "ok" for r in reports),
        "files_error": sum(r["status"] != "ok" for r in reports),
        "files": reports,
        "split": {
            "method": "SHA-256 of original 128-bit game hash plus seed; 80/10/10 thresholds",
            "seed": split_seed,
            "valid_rows": total_rows,
            "unique_games": total_games,
            "rows": split_rows,
            "games": split_games,
            "row_fraction": {name: split_rows[name] / total_rows if total_rows else 0.0 for name in SPLIT_NAMES},
            "game_fraction": {name: split_games[name] / total_games if total_games else 0.0 for name in SPLIT_NAMES},
            "duplicate_game_turn_rows": sum(n - 1 for n in all_game_turns.values() if n > 1),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help=".npz files or directories (searched recursively)")
    parser.add_argument("--max-entry-mib", type=int, default=512, help="maximum uncompressed size of each ZIP member")
    parser.add_argument("--split-seed", default="clm-v1", help="stable game-level split seed")
    parser.add_argument("--allow-missing-qvalue", action="store_true",
                        help="permit absent Q array for the policy-only probe; report its absence")
    parser.add_argument("--full-board-only", action="store_true",
                        help="exclude non-19x19 rows from split counts; keep other audit checks")
    parser.add_argument("--output", type=Path, help="write JSON report here; stdout if omitted")
    args = parser.parse_args(argv)
    if args.max_entry_mib <= 0:
        parser.error("--max-entry-mib must be positive")
    result = audit_paths(
        args.paths, args.max_entry_mib, args.split_seed,
        allow_missing_qvalue=args.allow_missing_qvalue,
        full_board_only=args.full_board_only,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
