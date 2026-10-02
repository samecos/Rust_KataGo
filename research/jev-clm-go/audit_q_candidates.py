#!/usr/bin/env python3
"""Read-only coverage census for KataGo per-move Q search targets.

This reports how many audited full-19x19 policy rows contain several visited
candidate moves. It does not certify the accuracy of Q estimates, make a
legal-move mask, or train a model. Visit cutoffs are exploratory filters, not
official KataGo reliability thresholds.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

import audit_npz


VISIT_CUTOFFS = (1, 8, 16, 32)
WL_GAP_SCALED = 3200  # 0.1 win-loss units after division by 32000.
OFFICIAL_WRITER = (
    "https://github.com/lightvector/KataGo/blob/"
    "352de42b663862251343d863cb6d6edb11c122cc/"
    "cpp/dataio/trainingwrite.cpp#L363-L383"
)
OFFICIAL_Q_LOSS = (
    "https://github.com/lightvector/KataGo/blob/"
    "352de42b663862251343d863cb6d6edb11c122cc/"
    "python/katago/train/metrics_pytorch.py#L81-L105"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_checked(archive: zipfile.ZipFile, name: str, rows: int, limit_bytes: int) -> np.ndarray:
    array = audit_npz._read_member(archive, name, limit_bytes)
    error = audit_npz._validate_array(name, array, rows)
    if error:
        raise ValueError(error)
    return array


def count_arrays(
    packed: np.ndarray, global_target: np.ndarray, policy: np.ndarray, q: np.ndarray,
) -> dict:
    """Count one file; reject malformed Q values before reporting coverage."""
    rows = len(global_target)
    for name, array in (
        ("binaryInputNCHWPacked", packed),
        ("globalTargetsNC", global_target),
        ("policyTargetsNCMove", policy),
        ("qValueTargetsNCMove", q),
    ):
        error = audit_npz._validate_array(name, array, rows)
        if error:
            raise ValueError(error)
    expected_version = audit_npz.GLOBAL_TARGET_VERSION_BY_WIDTH[global_target.shape[1]]
    if np.any(global_target[:, 63] != expected_version):
        raise ValueError("globalTargetsNC width and schema version disagree")
    if np.any(~np.isfinite(global_target[:, 25])) or np.any(global_target[:, 25] < 0):
        raise ValueError("non-finite or negative row weight")
    if np.any(policy < 0):
        raise ValueError("negative policy count")

    visits = q[:, 2, :]
    wl = q[:, 0, :]
    score = q[:, 1, :]
    sanity = {
        "negative_visits": int(np.count_nonzero(visits < 0)),
        "visits_above_32000": int(np.count_nonzero(visits > 32000)),
        "winloss_outside_scaled_range": int(np.count_nonzero(np.abs(wl.astype(np.int32)) > 32000)),
        "zero_visit_nonzero_winloss": int(np.count_nonzero((visits == 0) & (wl != 0))),
        "zero_visit_nonzero_score": int(np.count_nonzero((visits == 0) & (score != 0))),
    }
    if any(sanity.values()):
        raise ValueError(f"Q target sanity failed: {sanity}")

    onboard = np.unpackbits(packed[:, 0, :], axis=1, bitorder="big")[:, :audit_npz.AREA]
    full = np.all(onboard == 1, axis=1)
    valid_policy = global_target[:, 26] == 1
    weighted = global_target[:, 25] > 0
    has_mass = policy[:, 0, :].sum(axis=1, dtype=np.int64) > 0
    eligible = full & valid_policy & weighted & has_mass
    top_policy = np.argmax(policy[:, 0, :], axis=1)
    result = {
        "rows": rows,
        "target_version": expected_version,
        "full_19x19_rows": int(full.sum()),
        "eligible_rows": int(eligible.sum()),
        "eligible_top_policy_with_positive_q_visits": int(
            np.count_nonzero(eligible & (visits[np.arange(rows), top_policy] > 0))
        ),
        "eligible_rows_with_any_positive_q_visits": int(
            np.count_nonzero(eligible & np.any(visits > 0, axis=1))
        ),
        "q_sanity": sanity,
        "coverage": {},
    }
    if expected_version == 3:
        result["eligible_reanalyzed_rows"] = int(
            np.count_nonzero(eligible & (global_target[:, 64] == 1))
        )

    for cutoff in VISIT_CUTOFFS:
        accepted = visits >= cutoff
        candidate_count = accepted.sum(axis=1)
        at_least_two = eligible & (candidate_count >= 2)
        at_least_four = eligible & (candidate_count >= 4)
        low = np.where(accepted, wl, 32767).min(axis=1).astype(np.int32)
        high = np.where(accepted, wl, -32768).max(axis=1).astype(np.int32)
        separated = (high - low) >= WL_GAP_SCALED
        result["coverage"][str(cutoff)] = {
            "at_least_2_candidates": int(at_least_two.sum()),
            "at_least_4_candidates": int(at_least_four.sum()),
            "at_least_2_and_wl_gap_ge_0p1": int(np.count_nonzero(at_least_two & separated)),
            "at_least_4_and_wl_gap_ge_0p1": int(np.count_nonzero(at_least_four & separated)),
        }
    return result


def run(data_dir: Path, audit_path: Path, max_entry_mib: int = 512) -> dict:
    """Verify every source SHA against the full NPZ audit before counting."""
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "ok" or not audit.get("options", {}).get("full_board_only"):
        raise ValueError("reference audit must pass in full-board-only mode")
    paths = sorted(data_dir.rglob("*.npz"))
    if not paths:
        raise ValueError("no .npz files found")
    audited = {Path(item["path"]).resolve(): item for item in audit["files"]}
    if {path.resolve() for path in paths} != set(audited):
        raise ValueError("sample file set differs from reference audit")

    totals = Counter()
    coverage = {str(cutoff): Counter() for cutoff in VISIT_CUTOFFS}
    by_version = {"2": Counter(), "3": Counter()}
    limit_bytes = max_entry_mib * 1024 * 1024
    for path in paths:
        entry = audited[path.resolve()]
        if entry.get("status") != "ok":
            raise ValueError(f"reference audit did not approve {path}")
        if _sha256(path) != entry.get("sha256"):
            raise ValueError(f"source SHA-256 changed since reference audit: {path}")
        rows = int(entry["rows"])
        with zipfile.ZipFile(path) as archive:
            arrays = {
                name: _read_checked(archive, name, rows, limit_bytes)
                for name in (
                    "binaryInputNCHWPacked", "globalTargetsNC",
                    "policyTargetsNCMove", "qValueTargetsNCMove",
                )
            }
        item = count_arrays(
            arrays["binaryInputNCHWPacked"], arrays["globalTargetsNC"],
            arrays["policyTargetsNCMove"], arrays["qValueTargetsNCMove"],
        )
        if (item["rows"] != rows or
                item["full_19x19_rows"] != entry["spatial_msb"]["full_19x19_rows"]):
            raise ValueError(f"row or full-board count disagrees with reference audit: {path}")
        totals["files"] += 1
        for key in (
            "rows", "full_19x19_rows", "eligible_rows",
            "eligible_top_policy_with_positive_q_visits",
            "eligible_rows_with_any_positive_q_visits",
            "eligible_reanalyzed_rows",
        ):
            totals[key] += item.get(key, 0)
        version = str(item["target_version"])
        by_version[version]["files"] += 1
        by_version[version]["rows"] += item["rows"]
        by_version[version]["eligible_rows"] += item["eligible_rows"]
        for cutoff, counts in item["coverage"].items():
            coverage[cutoff].update(counts)

    return {
        "schema": "rust_katago_q_candidate_coverage_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "source": {
            "data_dir": str(data_dir),
            "reference_audit": str(audit_path),
            "reference_audit_sha256": _sha256(audit_path),
            "per_file_sha256": "verified against reference_audit.files[].sha256",
            "scope": "counts only NPZ files present in data_dir; do not extrapolate to the parent tgz",
        },
        "definition": {
            "eligible_row": "on-board plane has all 361 positions, globalTargetsNC[26]=1, globalTargetsNC[25]>0, policy target0 has positive mass",
            "candidate": "qValueTargetsNCMove[:,2,move] >= min_visits; 362 moves include pass",
            "winloss_gap": "max minus min of qValueTargetsNCMove[:,0,move] among accepted candidates >= 3200 (0.1 after /32000)",
            "source_writer": OFFICIAL_WRITER,
            "official_q_loss": OFFICIAL_Q_LOSS,
            "note": "KataGo masks zero visits and weights positive visits by sqrt(visits); cutoffs 8/16/32 here are exploratory coverage filters, not certified confidence thresholds.",
        },
        "totals": dict(totals),
        "by_target_version": {key: dict(value) for key, value in by_version.items()},
        "q_sanity": {
            "negative_visits": 0,
            "visits_above_32000": 0,
            "winloss_outside_scaled_range": 0,
            "zero_visit_nonzero_winloss": 0,
            "zero_visit_nonzero_score": 0,
            "checked_on": "all source rows and all 362 moves; any nonzero count aborts",
        },
        "coverage_by_min_visits": {key: dict(value) for key, value in coverage.items()},
        "interpretation_limit": "candidate coverage only; Q search estimates are correlated/noisy and this report does not establish hard-negative label reliability or playing strength",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--audit", type=Path, required=True, help="passing audit_npz.py JSON for the exact NPZ set")
    parser.add_argument("--max-entry-mib", type=int, default=512)
    parser.add_argument("--output", type=Path, help="write JSON here; stdout otherwise")
    args = parser.parse_args(argv)
    if args.max_entry_mib <= 0:
        parser.error("--max-entry-mib must be positive")
    result = run(args.data_dir, args.audit, args.max_entry_mib)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
