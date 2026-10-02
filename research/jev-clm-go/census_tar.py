#!/usr/bin/env python3
"""Read-only census of a KataGo self-play tar without extracting its files.

The report records the source hash, member safety, NPZ schema, row count, and
the number of rows whose packed board-mask plane is exactly 19x19. It does not
create training data or establish model quality.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile
from typing import Any
import zipfile

import numpy as np


SCHEMA: dict[str, tuple[tuple[int, ...], str]] = {
    "binaryInputNCHWPacked": ((22, 46), "uint8"),
    "globalInputNC": ((19,), "float32"),
    "policyTargetsNCMove": ((2, 362), "int16"),
    "scoreDistrN": ((842,), "int8"),
    "valueTargetsNCHW": ((5, 19, 19), "int8"),
}
OPTIONAL_SCHEMA = {
    "qValueTargetsNCMove": ((3, 362), "int16"),
    "metadataInputNC": ((192,), "float32"),
}
GLOBAL_TARGET_WIDTHS = (64, 80)
CHUNK_BYTES = 1024 * 1024
MAX_NPZ_BYTES = 128 * 1024 * 1024
MAX_ZIP_MEMBER_BYTES = 64 * 1024 * 1024
MAX_REPORTED_ERRORS = 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_tar_member(member: tarfile.TarInfo, root: str) -> bool:
    """Validate a member path and type; return True for a regular NPZ file."""
    name = member.name
    if not name or name.startswith("/") or "\\" in name or "\x00" in name:
        raise ValueError(f"unsafe tar path: {name!r}")
    parts = name.rstrip("/").split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"unsafe tar path: {name!r}")
    if not parts or parts[0] != root:
        raise ValueError(f"path outside expected dataset directory {root!r}: {name!r}")
    if member.isdir():
        if not (parts == [root] or len(parts) == 2):
            raise ValueError(f"unexpected tar directory: {name!r}")
        return False
    if not member.isfile():
        raise ValueError(f"tar member is not a regular file: {name!r}")
    if len(parts) != 3 or not parts[2].endswith(".npz"):
        raise ValueError(f"unexpected tar file path: {name!r}")
    if not 0 <= member.size <= MAX_NPZ_BYTES:
        raise ValueError(f"NPZ member size is outside 0..{MAX_NPZ_BYTES}: {name!r}")
    return True


def _npy_header(archive: zipfile.ZipFile, name: str) -> tuple[tuple[int, ...], np.dtype]:
    info = archive.getinfo(name)
    if info.file_size > MAX_ZIP_MEMBER_BYTES:
        raise ValueError(f"ZIP member exceeds {MAX_ZIP_MEMBER_BYTES} bytes: {name!r}")
    with archive.open(info) as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported NPY version {version}: {name!r}")
    if fortran_order or dtype.hasobject:
        raise ValueError(f"Fortran-order or object array is unsupported: {name!r}")
    return shape, dtype


def _version_label(value: float) -> str:
    if not np.isfinite(value) or value != np.floor(value):
        raise ValueError(f"invalid target schema version {value!r}")
    return str(int(value))


def _inspect_npz(blob: bytes) -> tuple[int, int, tuple[str, ...], bool, list[dict[str, Any]]]:
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate ZIP member names")
        keyset = tuple(sorted(names))
        missing = sorted((set(SCHEMA) | {"globalTargetsNC"}) - set(names))
        if missing:
            raise ValueError(f"missing required NPZ members: {missing}")
        expected = SCHEMA | OPTIONAL_SCHEMA | {"globalTargetsNC": ((), "float32")}
        unknown = sorted(set(names) - set(expected))
        if unknown:
            raise ValueError(f"unexpected NPZ members: {unknown}")
        rows: int | None = None
        target_width: int | None = None
        for name in names:
            shape, dtype = _npy_header(archive, name)
            if name == "globalTargetsNC":
                if len(shape) != 2 or shape[1] not in GLOBAL_TARGET_WIDTHS or dtype != np.dtype("float32"):
                    raise ValueError(f"{name}: expected [N,64|80] float32, got {shape} {dtype}")
                target_width = int(shape[1])
            else:
                tail, wanted_dtype = expected[name]
                if len(shape) != len(tail) + 1 or shape[1:] != tail or dtype != np.dtype(wanted_dtype):
                    raise ValueError(f"{name}: expected [N,{tail}] {wanted_dtype}, got {shape} {dtype}")
            if rows is None:
                rows = int(shape[0])
            elif shape[0] != rows:
                raise ValueError(f"{name}: row count {shape[0]} differs from {rows}")
        assert rows is not None
        assert target_width is not None
        with archive.open("binaryInputNCHWPacked") as stream:
            packed = np.load(stream, allow_pickle=False)
        if packed.shape != (rows, 22, 46):
            raise ValueError("packed input changed between header and content reads")
        # 361 on-board bits in MSB order: 45 complete 0xff bytes and one 0x80.
        board_mask = packed[:, 0, :]
        full_mask = np.all(board_mask[:, :45] == 0xff, axis=1) & (board_mask[:, 45] == 0x80)
        full_19 = int(np.count_nonzero(full_mask))
        with archive.open("globalTargetsNC") as stream:
            global_targets = np.load(stream, allow_pickle=False)
        if global_targets.shape != (rows, target_width):
            raise ValueError("global targets changed between header and content reads")
        versions = global_targets[:, 63]
        groups = []
        for version in np.unique(versions):
            label = _version_label(float(version))
            selected = versions == version
            groups.append({"global_target_width": target_width,
                           "target_schema_version": label,
                           "rows": int(np.count_nonzero(selected)),
                           "full_19x19_rows": int(np.count_nonzero(selected & full_mask))})
        return rows, full_19, keyset, "qValueTargetsNCMove" not in names, groups


def census(tar_path: Path) -> dict[str, Any]:
    tar_path = tar_path.resolve(strict=True)
    if not tar_path.is_file():
        raise ValueError(f"not a regular file: {tar_path}")
    root = tar_path.name
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if root.endswith(suffix):
            root = root[:-len(suffix)]
            break
    source = {"path": str(tar_path), "bytes": tar_path.stat().st_size,
              "sha256": _sha256(tar_path)}
    tar_members = directories = npz_files = npz_bytes = rows = full_19 = missing_q = 0
    min_rows: int | None = None
    max_rows: int | None = None
    keysets: Counter[tuple[str, ...]] = Counter()
    parent_dirs: Counter[str] = Counter()
    schema_groups: dict[tuple[int, str, tuple[str, ...]], dict[str, int]] = {}
    seen: set[str] = set()
    errors: list[str] = []
    error_count = 0

    with tarfile.open(tar_path, mode="r:*") as archive:
        for member in archive:
            tar_members += 1
            try:
                if member.name in seen:
                    raise ValueError(f"duplicate tar member name: {member.name!r}")
                seen.add(member.name)
                if not _check_tar_member(member, root):
                    directories += 1
                    continue
                npz_files += 1
                npz_bytes += member.size
                parent_dirs[member.name.rsplit("/", 1)[0]] += 1
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("could not read tar file member")
                with stream:
                    blob = stream.read(MAX_NPZ_BYTES + 1)
                if len(blob) != member.size:
                    raise ValueError(f"tar member byte count {len(blob)} differs from {member.size}")
                n, full, keys, no_q, groups = _inspect_npz(blob)
                rows += n
                full_19 += full
                missing_q += int(no_q)
                min_rows = n if min_rows is None else min(min_rows, n)
                max_rows = n if max_rows is None else max(max_rows, n)
                keysets[keys] += 1
                for group in groups:
                    group_key = (group["global_target_width"], group["target_schema_version"], keys)
                    totals = schema_groups.setdefault(group_key, {
                        "files_with_rows": 0, "rows": 0, "full_19x19_rows": 0,
                    })
                    totals["files_with_rows"] += 1
                    totals["rows"] += group["rows"]
                    totals["full_19x19_rows"] += group["full_19x19_rows"]
            except (OSError, ValueError, zipfile.BadZipFile, EOFError) as exc:
                error_count += 1
                if len(errors) < MAX_REPORTED_ERRORS:
                    errors.append(f"{member.name}: {type(exc).__name__}: {exc}")

    return {
        "schema": "rust_katago_tar_census_v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok" if error_count == 0 and npz_files > 0 else "error",
        "source": source,
        "tar": {"members": tar_members, "directories": directories,
                "regular_npz_files": npz_files, "npz_bytes": npz_bytes,
                "npz_parent_directories": dict(sorted(parent_dirs.items()))},
        "npz": {
            "files_inspected": sum(keysets.values()), "total_rows": rows,
            "full_19x19_rows": full_19, "other_board_rows": rows - full_19,
            "full_19x19_fraction": full_19 / rows if rows else 0.0,
            "min_rows_per_file": min_rows, "max_rows_per_file": max_rows,
            "files_missing_qValueTargetsNCMove": missing_q,
            "keysets": [{"members": list(names), "file_count": count}
                        for names, count in sorted(keysets.items())],
            "schema_groups": [
                {"global_target_width": width, "target_schema_version": version,
                 "members": list(names), **totals}
                for (width, version, names), totals in sorted(schema_groups.items())
            ],
        },
        "error_count": error_count,
        "errors": errors,
        "note": "Only exact 19x19 packed board-mask rows are counted; no files were extracted.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tar_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = census(args.tar_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {report['npz']['files_inspected']} NPZ, "
          f"{report['npz']['total_rows']} rows, "
          f"{report['npz']['full_19x19_rows']} full 19x19 rows; {args.output}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
