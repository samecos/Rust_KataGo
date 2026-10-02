"""Small synthetic fixtures for audit_npz; no GPU or engine process required."""

from __future__ import annotations

import io
import json
from pathlib import Path
from contextlib import redirect_stdout
import tempfile
import unittest
import zipfile

import numpy as np

import audit_npz


def fixture_arrays() -> dict[str, np.ndarray]:
    rows = 3
    binary = np.zeros((rows, 22, 19 * 19), dtype=np.uint8)
    binary[:, 0, :] = 1
    binary[0, 1, 0] = 1
    binary[1, 2, 1] = 1
    packed = np.packbits(binary, axis=2, bitorder="big")
    global_targets = np.zeros((rows, 64), dtype=np.float32)
    global_targets[:, 63] = 2
    global_targets[:, 25] = 1
    global_targets[:2, 26] = 1
    global_targets[0, 28] = 1
    global_targets[:2, 33] = 1
    global_targets[:2, 41:47] = (123, 456, 789, 10, 11, 12)
    global_targets[2, 41:47] = (7, 8, 9, 1, 2, 3)
    global_targets[:, 51] = (0, 1, 0)
    policy = np.zeros((rows, 2, 362), dtype=np.int16)
    policy[:2, 0, 0] = 10
    policy[0, 1, 1] = 8
    policy[2, 0, :] = 1  # fallback can be nonzero even when flag [26] is false
    value = np.zeros((rows, 5, 19, 19), dtype=np.int8)
    value[0, 2, 0, 0] = 1
    value[1, 3, 1, 1] = -1
    return {
        "binaryInputNCHWPacked": packed,
        "globalInputNC": np.zeros((rows, 19), dtype=np.float32),
        "policyTargetsNCMove": policy,
        "globalTargetsNC": global_targets,
        "scoreDistrN": np.zeros((rows, 842), dtype=np.int8),
        "valueTargetsNCHW": value,
        "qValueTargetsNCMove": np.zeros((rows, 3, 362), dtype=np.int16),
    }


def write_fixture(path: Path, arrays: dict[str, np.ndarray], suffix: bool = False) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            archive.writestr(name + (".npy" if suffix else ""), buffer.getvalue())


class AuditNpzTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "sample.npz"

    def test_bare_member_names_msb_and_game_level_split(self) -> None:
        write_fixture(self.path, fixture_arrays())
        result = audit_npz.audit_paths([self.path])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["split"]["valid_rows"], 3)
        self.assertEqual(result["split"]["unique_games"], 2)
        self.assertEqual(sum(result["split"]["rows"].values()), 3)
        report = result["files"][0]
        self.assertEqual(len(report["sha256"]), 64)
        self.assertEqual(report["spatial_msb"]["nonzero_padding_bits"], 0)
        self.assertEqual(report["policy"]["current_target_available"]["true"], 2)
        self.assertEqual(report["policy"]["current_target_available"]["false"], 1)
        self.assertEqual(report["future_boards"]["target_available"]["true"], 2)
        self.assertEqual(report["future_boards"]["potential_side_rows"], 1)
        # Both turns of one game must be assigned to the same split.
        games, _ = audit_npz._game_hashes(fixture_arrays()["globalTargetsNC"])
        split = audit_npz._split_for_game(games[0], "clm-v1")
        self.assertEqual(split, audit_npz._split_for_game(games[1], "clm-v1"))

    def test_conventional_member_suffix_is_accepted(self) -> None:
        write_fixture(self.path, fixture_arrays(), suffix=True)
        self.assertEqual(audit_npz.audit_paths([self.path])["status"], "ok")

    def test_missing_member_and_wrong_shape_fail_closed(self) -> None:
        arrays = fixture_arrays()
        arrays.pop("qValueTargetsNCMove")
        write_fixture(self.path, arrays)
        self.assertEqual(audit_npz.audit_paths([self.path])["status"], "error")
        self.assertIn("qValueTargetsNCMove", str(audit_npz.audit_paths([self.path])["files"][0]["errors"]))
        arrays = fixture_arrays()
        arrays["globalInputNC"] = np.zeros((3, 18), dtype=np.float32)
        write_fixture(self.path, arrays)
        self.assertEqual(audit_npz.audit_paths([self.path])["status"], "error")

    def test_nonfinite_bad_hash_and_msb_padding_are_reported(self) -> None:
        arrays = fixture_arrays()
        arrays["globalInputNC"][0, 0] = np.nan
        arrays["globalTargetsNC"][1, 41] = 1.5
        arrays["binaryInputNCHWPacked"][0, 0, -1] |= 1
        write_fixture(self.path, arrays)
        report = audit_npz.audit_paths([self.path])["files"][0]
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["nonfinite"]["globalInputNC"], 1)
        self.assertEqual(report["game_hash"]["invalid_rows"], 1)
        self.assertEqual(report["spatial_msb"]["nonzero_padding_bits"], 1)

    def test_duplicate_game_turns_across_archives(self) -> None:
        write_fixture(self.path, fixture_arrays())
        second = self.path.with_name("second.npz")
        write_fixture(second, fixture_arrays())
        result = audit_npz.audit_paths([self.path, second])
        self.assertEqual(result["split"]["duplicate_game_turn_rows"], 3)
        self.assertEqual(result["split"]["unique_games"], 2)

    def test_cli_json_and_invalid_future_mask(self) -> None:
        arrays = fixture_arrays()
        write_fixture(self.path, arrays)
        output = io.StringIO()
        with redirect_stdout(output):
            code = audit_npz.main([str(self.path)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["files_found"], 1)
        arrays["valueTargetsNCHW"][2, 2, 0, 0] = 1
        write_fixture(self.path, arrays)
        result = audit_npz.audit_paths([self.path])
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["files"][0]["future_boards"]["unavailable_rows_with_nonzero_future_channels"], 1)

    def test_valid_policy_target_requires_positive_mass(self) -> None:
        arrays = fixture_arrays()
        arrays["policyTargetsNCMove"][0, 0, :] = 0
        write_fixture(self.path, arrays)
        report = audit_npz.audit_paths([self.path])["files"][0]
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["policy"]["available_current_with_zero_mass"], 1)

    def test_explicit_policy_only_mixed_board_scope(self) -> None:
        arrays = fixture_arrays()
        arrays.pop("qValueTargetsNCMove")
        arrays["binaryInputNCHWPacked"][2, 0, 0] = 0
        write_fixture(self.path, arrays)
        self.assertEqual(audit_npz.audit_paths([self.path])["status"], "error")
        result = audit_npz.audit_paths(
            [self.path], allow_missing_qvalue=True, full_board_only=True
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["split"]["valid_rows"], 2)
        self.assertEqual(result["files"][0]["spatial_msb"]["full_19x19_rows"], 2)
        self.assertTrue(any("qValueTargetsNCMove is absent" in warning
                            for warning in result["files"][0]["warnings"]))

    def test_v3_global_targets_are_validated_by_width_and_version(self) -> None:
        arrays = fixture_arrays()
        v3 = np.zeros((3, 80), dtype=np.float32)
        v3[:, :64] = arrays["globalTargetsNC"]
        v3[:, 63] = 3
        arrays["globalTargetsNC"] = v3
        write_fixture(self.path, arrays)
        result = audit_npz.audit_paths([self.path])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["files"][0]["target_schema_version"], 3)
        v3[0, 63] = 2
        write_fixture(self.path, arrays)
        self.assertEqual(audit_npz.audit_paths([self.path])["status"], "error")


if __name__ == "__main__":
    unittest.main()
