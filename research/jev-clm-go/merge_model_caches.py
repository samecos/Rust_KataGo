#!/usr/bin/env python3
"""Merge official archive caches with global game splits and explicit deduplication.

Input order is precedence: put the newest archive first. A game is retained
only from its first cache, including all different positions in that cache.
Later caches' same-game rows are counted and skipped. Exact complete duplicate
rows inside the owner cache are removed. This deliberately sacrifices possible
complementary positions in later archives to avoid reanalysis/duplicate weight.

All retained games receive the requested global hash split, independently of
each source cache's prior split. Source archives, manifests and shards are read
only, SHA checked, and rechecked before publishing a fresh cache atomically.
The existing train_student.py loader can consume the resulting cache directly.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import sys
import tempfile
import time
from typing import Any
import zipfile

import numpy as np

import audit_npz
import prepare_model_data


SCHEMA = "rust_katago_multitask_cache_v1"
FIELDS = {
    "packed": ((22, 46), np.uint8), "global_input": ((19,), np.float32),
    "policy_counts": ((362,), np.int16), "row_weight": ((), np.float32),
    "value_probs": ((3,), np.float32), "value_weight": ((), np.float32),
    "score_mean": ((), np.float32), "score_mean_weight": ((), np.float32),
    "score_distr": ((842,), np.int8), "score_weight": ((), np.float32),
    "ownership": ((19, 19), np.int8), "ownership_weight": ((), np.float32),
    "split_code": ((), np.uint8), "game_hash": ((2,), np.uint64),
}
ROW_FIELDS = tuple(name for name in FIELDS if name != "split_code")
MAX_SHARD_BYTES = 512 * 1024 * 1024
MAX_ARRAY_BYTES = 128 * 1024 * 1024
MAX_SHARD_ROWS = 131072
MAX_SOURCE_GAMES = 2000000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "file_id": stat.st_ino, "device": stat.st_dev}


def read_shard(path: Path) -> dict[str, np.ndarray]:
    if path.stat().st_size > MAX_SHARD_BYTES:
        raise ValueError(f"oversized cache shard: {path}")
    with zipfile.ZipFile(path) as zipped:
        names = zipped.namelist()
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate ZIP entry in {path}")
        infos = zipped.infolist()
        if any(info.file_size > MAX_ARRAY_BYTES for info in infos) or sum(info.file_size for info in infos) > MAX_SHARD_BYTES:
            raise ValueError(f"decoded cache shard exceeds fixed memory bound: {path}")
    with np.load(path, allow_pickle=False) as shard:
        if set(shard.files) != set(FIELDS) or len(shard.files) != len(FIELDS):
            raise ValueError(f"cache shard field set mismatch: {path}")
        arrays = {name: shard[name] for name in FIELDS}
    rows = len(arrays["split_code"])
    if not 0 < rows <= MAX_SHARD_ROWS:
        raise ValueError(f"invalid cache shard row count: {path}")
    for name, (tail, dtype) in FIELDS.items():
        array = arrays[name]
        if array.dtype != dtype or array.shape != (rows, *tail):
            raise ValueError(f"{path}: {name} shape/dtype mismatch")
    if np.any(arrays["split_code"] > 2) or not prepare_model_data._full_board(arrays["packed"]).all():
        raise ValueError(f"{path}: invalid split code or non-19x19 row")
    for name in ("global_input", "value_probs", "score_mean", "row_weight", "value_weight", "score_mean_weight", "score_weight", "ownership_weight"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{path}: non-finite {name}")
    for name in ("row_weight", "value_weight", "score_mean_weight", "score_weight", "ownership_weight"):
        if np.any(arrays[name] < 0):
            raise ValueError(f"{path}: negative {name}")
    if np.any(arrays["row_weight"] <= 0) or np.any(arrays["policy_counts"] < 0) or np.any(arrays["policy_counts"].sum(axis=1, dtype=np.int64) <= 0):
        raise ValueError(f"{path}: invalid policy target/weight")
    if np.any(arrays["ownership"] < -1) or np.any(arrays["ownership"] > 1):
        raise ValueError(f"{path}: ownership outside [-1,1]")
    if not np.array_equal(arrays["score_mean_weight"], arrays["score_weight"]):
        raise ValueError(f"{path}: scalar and distribution score weights disagree")
    weighted = arrays["score_weight"] > 0
    if np.any(arrays["score_distr"][weighted] < 0) or np.any(arrays["score_distr"][weighted].sum(axis=1, dtype=np.int64) != 100):
        raise ValueError(f"{path}: invalid supervised score distribution")
    weighted = arrays["value_weight"] > 0
    value = arrays["value_probs"][weighted]
    if np.any(value < 0) or np.any(value > 1) or np.any(np.abs(value.sum(axis=1) - 1) > 0.01):
        raise ValueError(f"{path}: invalid supervised value target")
    return arrays


def row_digest(arrays: dict[str, np.ndarray], row: int) -> bytes:
    digest = hashlib.sha256(b"RustGo-multi-cache-row-v1\0")
    for name in ROW_FIELDS:
        digest.update(name.encode("ascii") + b"\0")
        digest.update(arrays[name][row].tobytes(order="C"))
    return digest.digest()


def source_inventory(cache_dirs: list[Path], out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not cache_dirs or len({path.resolve() for path in cache_dirs}) != len(cache_dirs):
        raise ValueError("source cache directories must be nonempty and distinct")
    inventory = []
    observed = {}
    archives_seen = set()
    for cache_dir in cache_dirs:
        if cache_dir.is_symlink():
            raise ValueError(f"source cache directory must not be a symlink: {cache_dir}")
        cache_dir = cache_dir.resolve(strict=True)
        if not cache_dir.is_dir() or cache_dir.is_symlink():
            raise ValueError(f"invalid source cache directory: {cache_dir}")
        if out_dir.resolve(strict=False).is_relative_to(cache_dir):
            raise ValueError("output must not be inside a source cache")
        manifest_path = cache_dir / "manifest.json"
        before_manifest = sha256_file(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA or manifest.get("status") != "completed" or manifest.get("split", {}).get("codes") != prepare_model_data.SPLIT_CODE:
            raise ValueError(f"unsupported or incomplete source cache: {cache_dir}")
        source = manifest.get("source", {})
        if source.get("kind") != "archive" or not isinstance(source.get("sha256"), str) or len(source["sha256"]) != 64:
            raise ValueError("each input cache must identify one official source archive SHA256")
        if source["sha256"] in archives_seen:
            raise ValueError("duplicate source archive SHA256: supply each archive once")
        archives_seen.add(source["sha256"])
        archive_path = Path(source["path"]).resolve(strict=True)
        if archive_path.is_symlink() or archive_path.stat().st_size != source["bytes"] or sha256_file(archive_path) != source["sha256"]:
            raise ValueError(f"source archive identity differs from cache manifest: {archive_path}")
        shards = sorted(cache_dir.glob("shard-*.npz"))
        if not shards or len(shards) != manifest.get("shards"):
            raise ValueError(f"source cache shard count mismatch: {cache_dir}")
        metadata = []
        for shard in shards:
            if shard.is_symlink() or not shard.resolve(strict=True).is_relative_to(cache_dir):
                raise ValueError(f"unsafe source shard: {shard}")
            digest = sha256_file(shard)
            metadata.append({"name": shard.name, "sha256": digest, "bytes": shard.stat().st_size})
            observed[str(shard)] = {"sha256": digest, "identity": identity(shard)}
        observed[str(manifest_path)] = {"sha256": before_manifest, "identity": identity(manifest_path)}
        observed[str(archive_path)] = {"sha256": source["sha256"], "identity": identity(archive_path)}
        inventory.append({"cache_dir": str(cache_dir), "cache_manifest": str(manifest_path), "cache_manifest_sha256": before_manifest,
                          "archive": source, "shards": metadata, "manifest": manifest})
    return inventory, observed


def merge_caches(cache_dirs: list[Path], out_dir: Path, split_seed: str, shard_rows: int = 8192) -> dict[str, Any]:
    if not isinstance(split_seed, str) or not split_seed or not 0 < shard_rows <= MAX_SHARD_ROWS:
        raise ValueError("split seed must be nonempty and shard rows within 1..131072")
    if out_dir.exists():
        raise FileExistsError(f"fresh output required; already exists: {out_dir}")
    if out_dir.parent.is_symlink():
        raise ValueError("output parent must not be a symlink")
    started = time.perf_counter()
    inventory, observed = source_inventory(cache_dirs, out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".merged-model-cache-", dir=out_dir.parent)).resolve()
    target = out_dir.resolve(strict=False)
    if temporary.parent != target.parent or not temporary.name.startswith(".merged-model-cache-"):
        raise ValueError("temporary/output directory escaped the explicitly requested destination")
    connection = sqlite3.connect(temporary / "merge-index.sqlite3")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute("CREATE TABLE games(game BLOB PRIMARY KEY, owner INTEGER NOT NULL, split INTEGER NOT NULL) WITHOUT ROWID")
    connection.execute("CREATE TABLE rows(digest BLOB PRIMARY KEY) WITHOUT ROWID")
    current_owner = 0
    output_rows = Counter()
    buffer = {name: [] for name in FIELDS}
    buffer_rows = 0
    output_shards = []
    sources = []
    row_digest_chain = hashlib.sha256(b"RustGo-multi-cache-retained-rows-v1\0")
    last_progress = started

    @lru_cache(maxsize=65536)
    def owner_and_split(game: tuple[int, int]) -> tuple[int, int]:
        key = struct.pack("<QQ", *game)
        prior = connection.execute("SELECT owner, split FROM games WHERE game=?", (key,)).fetchone()
        if prior is not None:
            return prior
        split = prepare_model_data.SPLIT_CODE[audit_npz._split_for_game(game, split_seed)]
        connection.execute("INSERT INTO games VALUES (?,?,?)", (key, current_owner, split))
        return current_owner, split

    def flush():
        nonlocal buffer_rows
        if not buffer_rows:
            return
        combined = {name: np.concatenate(parts, axis=0) for name, parts in buffer.items()}
        path = temporary / f"shard-{len(output_shards):05d}.npz"
        np.savez_compressed(path, **combined)
        output_shards.append({"name": path.name, "rows": buffer_rows, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        for parts in buffer.values():
            parts.clear()
        buffer_rows = 0

    try:
        labels = inventory[0]["manifest"]["labels"]
        for current_owner, source in enumerate(inventory):
            if source["manifest"]["labels"] != labels:
                raise ValueError("source caches declare different target/input semantics")
            cache_dir = Path(source["cache_dir"])
            original_seed = source["manifest"]["split"]["seed"]
            game_set = set()
            overlap_games = set()
            counts = Counter()

            @lru_cache(maxsize=65536)
            def original_split(game):
                return prepare_model_data.SPLIT_CODE[audit_npz._split_for_game(game, original_seed)]

            for shard_info in source["shards"]:
                path = cache_dir / shard_info["name"]
                if sha256_file(path) != shard_info["sha256"]:
                    raise ValueError(f"source shard changed before reading: {path}")
                arrays = read_shard(path)
                kept = []
                recoded = []
                for row, pair in enumerate(arrays["game_hash"]):
                    game = int(pair[0]), int(pair[1])
                    if len(game_set) >= MAX_SOURCE_GAMES and game not in game_set:
                        raise ValueError("source game inventory exceeds fixed memory bound")
                    game_set.add(game)
                    counts["input_rows"] += 1
                    if int(arrays["split_code"][row]) != original_split(game):
                        raise ValueError("source split code disagrees with its declared game hash rule")
                    owner, split = owner_and_split(game)
                    if owner != current_owner:
                        overlap_games.add(game)
                        counts["later_cache_same_game_rows_skipped"] += 1
                        continue
                    digest = row_digest(arrays, row)
                    inserted = connection.execute("INSERT OR IGNORE INTO rows VALUES (?)", (digest,)).rowcount
                    if not inserted:
                        counts["exact_complete_duplicate_rows_skipped"] += 1
                        continue
                    kept.append(row)
                    recoded.append(split)
                    row_digest_chain.update(digest)
                    counts["retained_rows"] += 1
                    counts["recoded_rows"] += int(split != int(arrays["split_code"][row]))
                    output_rows[audit_npz.SPLIT_NAMES[split]] += 1
                selected = np.asarray(kept, dtype=np.int64)
                if len(selected):
                    subset = {name: array[selected] for name, array in arrays.items()}
                    subset["split_code"] = np.asarray(recoded, dtype=np.uint8)
                    offset = 0
                    while offset < len(selected):
                        take = min(shard_rows - buffer_rows, len(selected) - offset)
                        for name in FIELDS:
                            buffer[name].append(subset[name][offset:offset + take].copy())
                        buffer_rows += take
                        offset += take
                        if buffer_rows == shard_rows:
                            flush()
                connection.commit()
                now = time.perf_counter()
                if now - last_progress >= 10:
                    print(json.dumps({"status": "merging", "source": current_owner, "shard": shard_info["name"],
                                      "source_counts": dict(counts), "output_rows": dict(output_rows),
                                      "output_shards": len(output_shards), "seconds": now - started}), flush=True)
                    last_progress = now
            declared_rows = sum(source["manifest"]["split"]["rows"].values())
            if counts["input_rows"] != declared_rows or counts["input_rows"] != source["manifest"]["totals"]["selected_rows"]:
                raise ValueError("source cache row count disagrees with manifest")
            sources.append({key: value for key, value in source.items() if key != "manifest"} | {
                "precedence": current_owner, "source_unique_games": len(game_set), "overlapping_games_skipped": len(overlap_games),
                "counts": dict(counts)})
            print(json.dumps({"source": current_owner, "cache": str(cache_dir), "counts": dict(counts)}, ensure_ascii=False), flush=True)
        flush()
        retained = sum(output_rows.values())
        if retained == 0:
            raise ValueError("merge retained no rows")
        output_games = {name: connection.execute("SELECT COUNT(*) FROM games WHERE split=?", (code,)).fetchone()[0] for name, code in prepare_model_data.SPLIT_CODE.items()}
        if connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0] != retained:
            raise RuntimeError("dedup ledger count does not match output rows")
        for path, expected in observed.items():
            if identity(Path(path)) != expected["identity"] or sha256_file(Path(path)) != expected["sha256"]:
                raise ValueError(f"source changed during merge: {path}")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.close()
        index_path = temporary / "merge-index.sqlite3"
        manifest = {
            "schema": SCHEMA, "status": "completed", "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source": {"kind": "merged_archive_caches", "sources": sources, "source_stable": True},
            "exclusion": {"manifest": None, "prior_sample_dir": None, "prior_game_hashes": 0,
                          "file_disjoint": False, "game_disjoint_from_prior_sample": False},
            "selection": {"algorithm": "input order defines per-game owner cache; retain owner-cache distinct complete rows", "max_files": None,
                          "full_19x19": "revalidated packed on-board plane exactly 361 ones", "policy": "positive row weights and nonnegative nonempty MCTS targets"},
            "split": {"seed": split_seed, "algorithm": "audit_npz._split_for_game: SHA-256 80/10/10 by 128-bit game hash; all sources globally recoded",
                      "codes": prepare_model_data.SPLIT_CODE, "rows": {name: output_rows[name] for name in prepare_model_data.SPLIT_CODE},
                      "games": output_games, "cross_split_game_intersections": 0},
            "labels": labels, "totals": {"selected_rows": retained, "source_cache_rows": sum(item["counts"]["input_rows"] for item in sources),
                                         "same_game_later_cache_rows_skipped": sum(item["counts"].get("later_cache_same_game_rows_skipped", 0) for item in sources),
                                         "exact_complete_duplicate_rows_skipped": sum(item["counts"].get("exact_complete_duplicate_rows_skipped", 0) for item in sources)},
            "deduplication": {"game_policy": "first cache wins, entire game; later complementary positions may be discarded",
                              "row_policy": "SHA256 of all fixed cache fields except split_code; remove exact complete owner-cache rows",
                              "retained_row_digest_chain_sha256": row_digest_chain.hexdigest(),
                              "index": {"path": index_path.name, "bytes": index_path.stat().st_size, "sha256": sha256_file(index_path)}},
            "shards": len(output_shards), "shard_inventory": output_shards, "files": [], "seconds": time.perf_counter() - started,
            "memory_bounds": {"decoded_shard_bytes": MAX_SHARD_BYTES, "source_games": MAX_SOURCE_GAMES, "ownership_lru_games": 65536,
                              "sqlite_cache_bytes": 64 * 1024 * 1024, "output_buffer_rows": shard_rows},
            "implementation_sha256": sha256_file(Path(__file__)),
        }
        raw_manifest = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        (temporary / "manifest.json").write_text(raw_manifest, encoding="utf-8")
        if target.exists() or temporary.resolve().parent != target.parent:
            raise FileExistsError("output appeared or verified destination changed before publication")
        temporary.rename(target)
        return manifest
    except Exception as exc:
        connection.close()
        failure = {"schema": "rust_go_cache_merge_failure_v1", "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                   "sources_completed": sources, "partial_output_shards": output_shards, "seconds": time.perf_counter() - started}
        (temporary / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise ValueError(f"merge failed; partial evidence preserved at {temporary}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dirs", type=Path, nargs="+", required=True, help="precedence order; newest source first recommended")
    parser.add_argument("--out-dir", type=Path, required=True, help="fresh destination; source caches are read only")
    parser.add_argument("--split-seed", required=True, help="one frozen global hash split seed")
    parser.add_argument("--shard-rows", type=int, default=8192)
    args = parser.parse_args(argv)
    try:
        manifest = merge_caches(args.cache_dirs, args.out_dir, args.split_seed, args.shard_rows)
        path = args.out_dir / "manifest.json"
        print(json.dumps({"status": manifest["status"], "manifest": str(path.resolve()), "manifest_sha256": sha256_file(path),
                          "rows": manifest["split"]["rows"], "games": manifest["split"]["games"], "totals": manifest["totals"]}, ensure_ascii=False))
        return 0
    except (ValueError, OSError, sqlite3.Error, KeyError, TypeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
