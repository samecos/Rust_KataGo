#!/usr/bin/env python3
"""Deterministically extract a small, auditable NPZ sample from a tar archive.

The input archive is never changed. Every member is checked before selection;
only selected regular NPZ files are copied, and output is published only after
all hashes and byte counts have been recorded. This does not parse NPZ data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "data"
CHUNK_BYTES = 1024 * 1024
DATASET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_member(member: tarfile.TarInfo, dataset_name: str) -> str | None:
    """Return the safe path below the dataset directory, or None for a dir."""
    name = member.name
    if not name or "\\" in name or "\x00" in name or name.startswith("/"):
        raise ValueError(f"unsafe tar member path: {name!r}")
    raw_parts = name.rstrip("/").split("/")
    if any(part in ("", ".", "..") for part in raw_parts):
        raise ValueError(f"unsafe tar member path: {name!r}")
    parts = PurePosixPath(name).parts
    if not parts or parts[0] != dataset_name:
        raise ValueError(f"tar member lies outside expected dataset {dataset_name!r}: {name!r}")
    if member.isdir():
        if not (len(parts) == 1 or len(parts) == 2):
            raise ValueError(f"unexpected tar directory: {name!r}")
        return None
    if not member.isfile():
        raise ValueError(f"tar member is not a regular file: {name!r}")
    if len(parts) != 3 or not parts[2].endswith(".npz"):
        raise ValueError(f"unexpected tar file path: {name!r}")
    if member.size < 0:
        raise ValueError(f"negative tar member size: {name!r}")
    return "/".join(parts[1:])


def _ranked_members(
    archive: tarfile.TarFile, dataset_name: str, count: int
) -> tuple[list[tarfile.TarInfo], int, int]:
    files: list[tarfile.TarInfo] = []
    seen_names: set[str] = set()
    member_count = 0
    for member in archive:
        member_count += 1
        relative = _checked_member(member, dataset_name)
        if member.name in seen_names:
            raise ValueError(f"duplicate tar member name: {member.name!r}")
        seen_names.add(member.name)
        if relative is not None:
            files.append(member)
    if len(files) < count:
        raise ValueError(f"requested {count} NPZ files but archive contains only {len(files)}")
    files.sort(key=lambda member: (
        hashlib.sha256(member.name.encode("utf-8")).hexdigest(), member.name
    ))
    return files[:count], member_count, len(files)


def prepare_archive(tar_path: Path, count: int, out_dir: Path) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("--count must be positive")
    if not tar_path.is_file():
        raise ValueError(f"tar archive not found: {tar_path}")
    dataset_name = tar_path.stem
    if not DATASET_NAME.fullmatch(dataset_name) or dataset_name in (".", ".."):
        raise ValueError(f"unsafe tar dataset name: {dataset_name!r}")
    expected_out = DATA_ROOT / dataset_name
    if out_dir.resolve(strict=False) != expected_out.resolve(strict=False):
        raise ValueError(f"--out-dir must be {expected_out}")
    if DATA_ROOT.is_symlink() or expected_out.is_symlink():
        raise ValueError("data directory and sample directory must not be symlinks")
    if expected_out.exists():
        raise FileExistsError(f"sample directory already exists: {expected_out}")

    tar_path = tar_path.resolve(strict=True)
    source_size = tar_path.stat().st_size
    source_sha256 = _sha256_file(tar_path)
    with tarfile.open(tar_path, mode="r:*") as archive:
        selected, member_count, npz_count = _ranked_members(archive, dataset_name, count)
        # Never let pre-existing data paths redirect a write. A temporary
        # sibling keeps partial output out of the requested sample directory.
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        if DATA_ROOT.is_symlink():
            raise ValueError("data directory must not be a symlink")
        temporary = Path(tempfile.mkdtemp(prefix=f".{dataset_name}.tmp-", dir=DATA_ROOT))
        try:
            selected_files: list[dict[str, Any]] = []
            for member in sorted(selected, key=lambda item: item.offset_data):
                relative = _checked_member(member, dataset_name)
                assert relative is not None
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                copied = 0
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"could not read tar member: {member.name!r}")
                with source, target.open("xb") as destination:
                    while block := source.read(CHUNK_BYTES):
                        destination.write(block)
                        digest.update(block)
                        copied += len(block)
                if copied != member.size:
                    raise ValueError(f"short tar member: {member.name!r} ({copied} != {member.size})")
                selected_files.append({
                    "archive_path": member.name,
                    "extracted_path": relative,
                    "bytes": copied,
                    "sha256": digest.hexdigest(),
                })
            selected_files.sort(key=lambda item: item["archive_path"])
            manifest: dict[str, Any] = {
                "schema": "rust_katago_clm_tar_sample_v1",
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "source_tar": {
                    "path": str(tar_path),
                    "bytes": source_size,
                    "sha256": source_sha256,
                    "members_total": member_count,
                    "npz_members_total": npz_count,
                },
                "selection": {
                    "count": count,
                    "algorithm": "sort regular NPZ members by (SHA-256 of UTF-8 archive member path, path); select first count",
                    "content_independent": True,
                },
                "selected_files": selected_files,
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            if expected_out.exists():
                raise FileExistsError(f"sample directory appeared during extraction: {expected_out}")
            os.rename(temporary, expected_out)
            return manifest
        except BaseException:
            shutil.rmtree(temporary)
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tar_path", type=Path, help="source .tar or .tgz archive")
    parser.add_argument("--count", type=int, default=24, help="number of NPZ members to select (default: 24)")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="must be research/jev-clm-go/data/<tar-stem>")
    args = parser.parse_args()
    try:
        manifest = prepare_archive(args.tar_path, args.count, args.out_dir)
    except (ValueError, OSError, tarfile.TarError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "status": "completed",
        "out_dir": str(DATA_ROOT / args.tar_path.stem),
        "selected_npz": len(manifest["selected_files"]),
        "source_sha256": manifest["source_tar"]["sha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
