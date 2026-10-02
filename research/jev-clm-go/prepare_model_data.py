#!/usr/bin/env python3
"""Stream 19x19 KataGo self-play NPZs into compact, game-split model shards.

Only ``packed`` and ``global_input`` are inference inputs. All other arrays
are labels, weights, or bookkeeping and must never be passed to the model.
The current player is the player to move (C++ ``nextPlayer``). A prior sample
manifest excludes files; --prior-sample-dir also excludes every matching game.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from typing import Any, Iterator
import zipfile

import numpy as np

import audit_npz
import census_tar


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "data"
SCHEMA = "rust_katago_multitask_cache_v1"
MAX_NPZ_BYTES = 128 * 1024 * 1024
MAX_ZIP_ARRAY_BYTES = 64 * 1024 * 1024
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}
REQUIRED = (
    "binaryInputNCHWPacked", "globalInputNC", "policyTargetsNCMove",
    "globalTargetsNC", "scoreDistrN", "valueTargetsNCHW",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str) -> PurePosixPath:
    if not value or value.startswith("/") or "\\" in value or "\x00" in value:
        raise ValueError(f"unsafe relative path: {value!r}")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"unsafe relative path: {value!r}")
    return PurePosixPath(*parts)


def _archive_root(path: Path) -> str:
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if path.name.endswith(suffix):
            return path.name[:-len(suffix)]
    raise ValueError("archive must end in .tar, .tar.gz, or .tgz")


def _load_exclusion_manifest(path: Path | None) -> tuple[dict[str, Any] | None, set[str], set[str]]:
    if path is None:
        return None, set(), set()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "rust_katago_clm_tar_sample_v1":
        raise ValueError("unsupported prior sample manifest schema")
    selected = manifest.get("selected_files")
    if not isinstance(selected, list) or not selected:
        raise ValueError("prior sample manifest has no selected_files")
    archive_paths: set[str] = set()
    extracted_paths: set[str] = set()
    for entry in selected:
        archive_name = str(_safe_relative(entry["archive_path"]))
        extracted_name = str(_safe_relative(entry["extracted_path"]))
        if archive_name in archive_paths or extracted_name in extracted_paths:
            raise ValueError("duplicate path in prior sample manifest")
        archive_paths.add(archive_name)
        extracted_paths.add(extracted_name)
    return manifest, archive_paths, extracted_paths


def _read_npz(blob: bytes) -> dict[str, np.ndarray]:
    if len(blob) > MAX_NPZ_BYTES:
        raise ValueError(f"NPZ exceeds {MAX_NPZ_BYTES} bytes")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate ZIP member names")
        allowed = set(REQUIRED) | {name + ".npy" for name in REQUIRED}
        allowed |= {"qValueTargetsNCMove", "qValueTargetsNCMove.npy",
                    "metadataInputNC", "metadataInputNC.npy"}
        if set(names) - allowed:
            raise ValueError(f"unexpected NPZ members: {sorted(set(names) - allowed)}")
        arrays: dict[str, np.ndarray] = {}
        rows: int | None = None
        for name in REQUIRED:
            array = audit_npz._read_member(archive, name, MAX_ZIP_ARRAY_BYTES)
            problem = audit_npz._validate_array(name, array, rows)
            if problem:
                raise ValueError(problem)
            rows = len(array) if rows is None else rows
            arrays[name] = array
    assert rows is not None
    target = arrays["globalTargetsNC"]
    expected_version = audit_npz.GLOBAL_TARGET_VERSION_BY_WIDTH[target.shape[1]]
    if np.any(target[:, 63] != expected_version):
        raise ValueError("global target width and schema version disagree")
    if not np.isfinite(target).all() or not np.isfinite(arrays["globalInputNC"]).all():
        raise ValueError("non-finite global input or target")
    if np.any(arrays["policyTargetsNCMove"] < 0):
        raise ValueError("negative policy visit count")
    if np.any((target[:, 25] < 0) | (target[:, 27] < 0)):
        raise ValueError("negative row or ownership weight")
    if np.any((target[:, 35] < 0) | (target[:, 35] > 1)):
        raise ValueError("invalid value weight complement")
    return arrays


def _prior_game_hashes(sample_dir: Path, manifest: dict[str, Any]) -> set[tuple[int, int]]:
    root = sample_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("prior sample directory is not a directory")
    result: set[tuple[int, int]] = set()
    for entry in manifest["selected_files"]:
        rel = _safe_relative(entry["extracted_path"])
        path = sample_dir.joinpath(*rel.parts)
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError(f"unsafe prior sample file: {path}")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"prior sample SHA-256 mismatch: {path}")
        with zipfile.ZipFile(path) as archive:
            target = audit_npz._read_member(archive, "globalTargetsNC", MAX_ZIP_ARRAY_BYTES)
        problem = audit_npz._validate_array("globalTargetsNC", target, None)
        if problem:
            raise ValueError(f"{path}: {problem}")
        hashes, bad = audit_npz._game_hashes(target)
        if bad:
            raise ValueError(f"{path}: {bad} invalid prior game hashes")
        result.update(value for value in hashes if value is not None)
    return result


def _tar_members(path: Path) -> tuple[list[tarfile.TarInfo], int]:
    root = _archive_root(path)
    files: list[tarfile.TarInfo] = []
    seen: set[str] = set()
    with tarfile.open(path, mode="r:*") as archive:
        for member in archive:
            if member.name in seen:
                raise ValueError(f"duplicate tar path: {member.name!r}")
            seen.add(member.name)
            if census_tar._check_tar_member(member, root):
                files.append(member)
    return files, len(seen)


def _directory_files(path: Path) -> list[tuple[str, Path]]:
    root = path.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("--input-dir is not a directory")
    files = []
    for file in path.rglob("*.npz"):
        if file.is_symlink() or not file.resolve(strict=True).is_relative_to(root):
            raise ValueError(f"unsafe NPZ path: {file}")
        relative = file.relative_to(path).as_posix()
        _safe_relative(relative)
        files.append((relative, file))
    return files


def _rank(name: str) -> tuple[str, str]:
    return hashlib.sha256(name.encode("utf-8")).hexdigest(), name


def _source_blobs(
    *, archive_path: Path | None, input_dir: Path | None,
    excluded_archive: set[str], excluded_relative: set[str], max_files: int | None,
) -> tuple[Iterator[tuple[str, bytes]], dict[str, Any]]:
    if (archive_path is None) == (input_dir is None):
        raise ValueError("specify exactly one of --archive or --input-dir")
    if max_files is not None and max_files <= 0:
        raise ValueError("--max-files must be positive")
    if archive_path is not None:
        archive_path = archive_path.resolve(strict=True)
        members, all_members = _tar_members(archive_path)
        remaining = [m for m in members if m.name not in excluded_archive]
        selected = sorted(remaining, key=lambda m: _rank(m.name))
        if max_files is not None:
            selected = selected[:max_files]
        selected_names = {m.name for m in selected}
        if not selected:
            raise ValueError("no unexcluded NPZ members selected")
        source_info = {
            "kind": "archive", "path": str(archive_path),
            "bytes": archive_path.stat().st_size, "sha256": sha256_file(archive_path),
            "all_tar_members": all_members, "all_npz_files": len(members),
            "excluded_npz_files": len(members) - len(remaining),
            "selected_npz_files": len(selected),
        }

        def stream() -> Iterator[tuple[str, bytes]]:
            with tarfile.open(archive_path, mode="r|*") as archive:
                for member in archive:
                    if member.name not in selected_names:
                        continue
                    source = archive.extractfile(member)
                    if source is None:
                        raise ValueError(f"could not read tar member {member.name!r}")
                    with source:
                        blob = source.read(MAX_NPZ_BYTES + 1)
                    if len(blob) != member.size or len(blob) > MAX_NPZ_BYTES:
                        raise ValueError(f"invalid NPZ byte count: {member.name!r}")
                    yield member.name, blob

        return stream(), source_info

    assert input_dir is not None
    input_dir = input_dir.resolve(strict=True)
    files = _directory_files(input_dir)
    remaining_files = [(name, path) for name, path in files if name not in excluded_relative]
    selected_files = sorted(remaining_files, key=lambda pair: _rank(pair[0]))
    if max_files is not None:
        selected_files = selected_files[:max_files]
    if not selected_files:
        raise ValueError("no unexcluded NPZ files selected")
    source_info = {
        "kind": "directory", "path": str(input_dir), "all_npz_files": len(files),
        "excluded_npz_files": len(files) - len(remaining_files),
        "selected_npz_files": len(selected_files),
    }

    def stream_dir() -> Iterator[tuple[str, bytes]]:
        for name, path in selected_files:
            if path.stat().st_size > MAX_NPZ_BYTES:
                raise ValueError(f"NPZ exceeds {MAX_NPZ_BYTES} bytes: {path}")
            yield name, path.read_bytes()

    return stream_dir(), source_info


def _full_board(packed: np.ndarray) -> np.ndarray:
    # Exactly 361 on-board bits: 45 full bytes then one 0x80 byte.
    plane = packed[:, 0, :]
    return np.all(plane[:, :45] == 0xff, axis=1) & (plane[:, 45] == 0x80)


def _filtered_rows(
    arrays: dict[str, np.ndarray], split_seed: str,
    prior_games: set[tuple[int, int]], split_cache: dict[tuple[int, int], int],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    packed = arrays["binaryInputNCHWPacked"]
    global_input = arrays["globalInputNC"]
    target = arrays["globalTargetsNC"]
    policy = arrays["policyTargetsNCMove"][:, 0, :]
    score = arrays["scoreDistrN"]
    ownership = arrays["valueTargetsNCHW"][:, 0, :, :]
    full = _full_board(packed)
    policy_ok = (target[:, 26] == 1) & (target[:, 25] > 0) & (
        policy.sum(axis=1, dtype=np.int64) > 0
    )
    selected = full & policy_ok
    hashes, bad_hashes = audit_npz._game_hashes(target)
    if bad_hashes:
        raise ValueError(f"{bad_hashes} invalid game hashes")
    selected_indices = [i for i in np.flatnonzero(selected) if hashes[i] not in prior_games]
    counts = {
        "source_rows": len(packed), "nonfull_rows": int((~full).sum()),
        "invalid_policy_rows": int((full & ~policy_ok).sum()),
        "prior_game_rows": int(selected.sum()) - len(selected_indices),
        "selected_rows": len(selected_indices),
    }
    indices = np.asarray(selected_indices, dtype=np.int64)
    if not len(indices):
        return {}, counts
    game_hashes = [hashes[i] for i in indices]
    assert all(game is not None for game in game_hashes)
    split_codes = np.empty(len(indices), dtype=np.uint8)
    hash_array = np.empty((len(indices), 2), dtype=np.uint64)
    for j, game in enumerate(game_hashes):
        assert game is not None
        if game not in split_cache:
            split_cache[game] = SPLIT_CODE[audit_npz._split_for_game(game, split_seed)]
        split_codes[j] = split_cache[game]
        hash_array[j] = game

    complete = target[indices, 62] == 1
    value_weight = np.where(complete, 1.0 - target[indices, 35], 0.0).astype(np.float32)
    ownership_weight = np.where(complete, target[indices, 27], 0.0).astype(np.float32)
    score_weight = ownership_weight.copy()
    value = target[indices, 0:3].astype(np.float32)
    score_mean = target[indices, 3].astype(np.float32)
    score_distr = score[indices]
    own = ownership[indices]
    if np.any((value_weight > 0) & (np.abs(value.sum(axis=1) - 1.0) > 0.01)):
        raise ValueError("valid value target probabilities do not sum to 1")
    if np.any((value_weight > 0) & ((value < 0).any(axis=1) | (value > 1).any(axis=1))):
        raise ValueError("valid value target probability outside 0..1")
    if np.any((score_weight > 0) & (
        (score_distr < 0).any(axis=1) |
        (score_distr.sum(axis=1, dtype=np.int64) != 100)
    )):
        raise ValueError("valid score distribution has negative bins or sum != 100")
    if np.any((ownership_weight > 0) & ((own < -1) | (own > 1)).any(axis=(1, 2))):
        raise ValueError("valid ownership target outside -1..1")
    return {
        "packed": packed[indices].copy(),
        "global_input": global_input[indices].copy(),
        "policy_counts": policy[indices].copy(),
        "row_weight": target[indices, 25].astype(np.float32),
        "value_probs": value,
        "value_weight": value_weight,
        "score_mean": score_mean,
        "score_mean_weight": score_weight.copy(),
        "score_distr": score_distr.copy(),
        "score_weight": score_weight,
        "ownership": own.copy(),
        "ownership_weight": ownership_weight,
        "split_code": split_codes,
        "game_hash": hash_array,
    }, counts


def prepare_model_data(
    *, out_dir: Path, split_seed: str, archive_path: Path | None = None,
    input_dir: Path | None = None, exclude_manifest: Path | None = None,
    prior_sample_dir: Path | None = None, max_files: int | None = None,
    shard_rows: int = 8192,
) -> dict[str, Any]:
    if shard_rows <= 0:
        raise ValueError("--shard-rows must be positive")
    if prior_sample_dir is not None and exclude_manifest is None:
        raise ValueError("--prior-sample-dir requires --exclude-manifest")
    if out_dir.exists():
        raise FileExistsError(f"output directory already exists: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if out_dir.parent.is_symlink():
        raise ValueError("output parent must not be a symlink")
    prior_manifest, excluded_archive, excluded_relative = _load_exclusion_manifest(exclude_manifest)
    prior_games = (_prior_game_hashes(prior_sample_dir, prior_manifest)
                   if prior_sample_dir is not None and prior_manifest is not None else set())
    blobs, source_info = _source_blobs(
        archive_path=archive_path, input_dir=input_dir,
        excluded_archive=excluded_archive, excluded_relative=excluded_relative,
        max_files=max_files,
    )
    if prior_manifest is not None and archive_path is not None:
        expected = prior_manifest["source_tar"]
        if source_info["sha256"] != expected["sha256"] or source_info["bytes"] != expected["bytes"]:
            raise ValueError("prior sample manifest belongs to a different source archive")
    if prior_manifest is not None and source_info["excluded_npz_files"] != len(prior_manifest["selected_files"]):
        raise ValueError("some prior sample files were not found in the source")

    temp_dir = Path(tempfile.mkdtemp(prefix=".model-cache-", dir=out_dir.parent))
    buffer: dict[str, list[np.ndarray]] = {}
    buffer_rows = 0
    shard_index = 0
    totals: Counter[str] = Counter()
    split_rows: Counter[str] = Counter()
    split_games: dict[str, set[tuple[int, int]]] = {name: set() for name in SPLIT_CODE}
    split_cache: dict[tuple[int, int], int] = {}
    files: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal buffer_rows, shard_index
        if buffer_rows == 0:
            return
        combined = {name: np.concatenate(parts, axis=0) for name, parts in buffer.items()}
        name = f"shard-{shard_index:05d}.npz"
        np.savez_compressed(temp_dir / name, **combined)
        shard_index += 1
        buffer.clear()
        buffer_rows = 0

    try:
        for source_name, blob in blobs:
            arrays = _read_npz(blob)
            rows, counts = _filtered_rows(arrays, split_seed, prior_games, split_cache)
            totals.update(counts)
            files.append({"path": source_name, "sha256": hashlib.sha256(blob).hexdigest(), **counts})
            if not rows:
                continue
            for code, pair in zip(rows["split_code"], rows["game_hash"]):
                name = audit_npz.SPLIT_NAMES[int(code)]
                split_rows[name] += 1
                split_games[name].add((int(pair[0]), int(pair[1])))
            for name, array in rows.items():
                buffer.setdefault(name, []).append(array)
            buffer_rows += len(rows["packed"])
            if buffer_rows >= shard_rows:
                flush()
        flush()
        if totals["selected_rows"] == 0:
            raise ValueError("no full 19x19 rows with a valid policy target remained")
        if any(split_games[a] & split_games[b] for a in SPLIT_CODE for b in SPLIT_CODE if a < b):
            raise RuntimeError("game hash leaked across new cache splits")
        manifest = {
            "schema": SCHEMA, "status": "completed",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source": source_info,
            "exclusion": {
                "manifest": str(exclude_manifest) if exclude_manifest is not None else None,
                "prior_sample_dir": str(prior_sample_dir) if prior_sample_dir is not None else None,
                "prior_game_hashes": len(prior_games),
                "file_disjoint": exclude_manifest is not None,
                "game_disjoint_from_prior_sample": prior_sample_dir is not None,
            },
            "selection": {
                "max_files": max_files,
                "algorithm": "SHA-256 rank of archive member path or directory-relative path after prior-file exclusion",
                "full_19x19": "packed on-board plane exactly 361 ones",
                "policy": "globalTargetsNC[26]=1, [25]>0, target0 visit sum>0",
            },
            "split": {
                "seed": split_seed, "algorithm": "audit_npz._split_for_game: SHA-256 80/10/10 by 128-bit game hash",
                "codes": SPLIT_CODE,
                "rows": {name: split_rows[name] for name in SPLIT_CODE},
                "games": {name: len(split_games[name]) for name in SPLIT_CODE},
            },
            "labels": {
                "perspective": "player to move, called nextPlayer in KataGo writer",
                "policy_counts": "MCTS visit target0, current position; zeros do not mean illegal",
                "value_probs": "final/TD target globalTargetsNC[0:3] (win, loss, no-result); weight=(1-[35]) if [62]=1 else 0",
                "score_mean": "final/TD scalar score globalTargetsNC[3]; weight=[27] if [62]=1 else 0",
                "score_distr": "final score distribution scoreDistrN/100 at use time; weight=[27] if [62]=1 else 0; unavailable rows contain dummy bins",
                "ownership": "final ownership valueTargetsNCHW[0], +1 for player to move; weight=[27] if [62]=1 else 0",
                "row_weight": "globalTargetsNC[25] multiplies each head's weight",
                "input_only": ["packed", "global_input"],
            },
            "totals": dict(totals), "shards": shard_index, "files": files,
        }
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if out_dir.exists():
            raise FileExistsError(f"output directory appeared during preparation: {out_dir}")
        os.rename(temp_dir, out_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temp_dir)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path, help="source .tar/.tgz, streamed without extraction")
    source.add_argument("--input-dir", type=Path, help="directory of source NPZ files")
    parser.add_argument("--exclude-manifest", type=Path,
                        help="previous 1800-file sample_manifest.json; exclude exact paths")
    parser.add_argument("--prior-sample-dir", type=Path,
                        help="previous extracted sample root; also exclude matching game hashes")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="new direct child of research/jev-clm-go/data/")
    parser.add_argument("--split-seed", default="multitask-v1-aug20")
    parser.add_argument("--max-files", type=int, help="optional deterministic file budget")
    parser.add_argument("--shard-rows", type=int, default=8192)
    args = parser.parse_args()
    if args.out_dir.parent.resolve(strict=False) != DATA_ROOT.resolve(strict=False):
        parser.error(f"--out-dir must be a direct child of {DATA_ROOT}")
    try:
        manifest = prepare_model_data(
            out_dir=args.out_dir, split_seed=args.split_seed,
            archive_path=args.archive, input_dir=args.input_dir,
            exclude_manifest=args.exclude_manifest,
            prior_sample_dir=args.prior_sample_dir, max_files=args.max_files,
            shard_rows=args.shard_rows,
        )
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")
    print(json.dumps({"status": "completed", "out_dir": str(args.out_dir),
                      "rows": manifest["totals"]["selected_rows"],
                      "shards": manifest["shards"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
