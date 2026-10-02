#!/usr/bin/env python3
"""Targeted checks for Q coverage cutoffs and fail-closed sanity."""

import unittest

import numpy as np

import audit_q_candidates


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    packed = np.zeros((3, 22, 46), dtype=np.uint8)
    full = np.packbits(np.ones(361, dtype=np.uint8), bitorder="big")
    packed[:, 0, :] = full
    packed[2, 0, 0] = 0  # Third row is not a full 19x19 board.
    target = np.zeros((3, 64), dtype=np.float32)
    target[:, 25] = 1
    target[:, 26] = 1
    target[:, 63] = 2
    policy = np.zeros((3, 2, 362), dtype=np.int16)
    policy[:, 0, 0] = 10
    q = np.zeros((3, 3, 362), dtype=np.int16)
    q[0, 2, :4] = [8, 8, 7, 32]
    q[0, 0, :4] = [0, 3200, 8000, 0]
    q[1, 2, :4] = [16, 8, 8, 8]
    q[1, 0, :4] = [0, 100, 100, 100]
    q[2, 2, :4] = [32, 32, 32, 32]
    q[2, 0, :4] = [0, 3200, 0, 0]
    return packed, target, policy, q


class QCandidateCoverageTests(unittest.TestCase):
    def test_full_board_cutoffs_and_scaled_gap(self) -> None:
        result = audit_q_candidates.count_arrays(*fixture())
        self.assertEqual(result["eligible_rows"], 2)
        self.assertEqual(result["coverage"]["1"]["at_least_4_candidates"], 2)
        self.assertEqual(result["coverage"]["8"]["at_least_4_candidates"], 1)
        self.assertEqual(result["coverage"]["8"]["at_least_2_and_wl_gap_ge_0p1"], 1)
        self.assertEqual(result["coverage"]["16"]["at_least_2_candidates"], 0)
        self.assertEqual(result["coverage"]["32"]["at_least_4_candidates"], 0)

    def test_zero_visit_cannot_have_q_label(self) -> None:
        packed, target, policy, q = fixture()
        q[0, 0, 8] = 10
        with self.assertRaisesRegex(ValueError, "Q target sanity failed"):
            audit_q_candidates.count_arrays(packed, target, policy, q)

    def test_v3_reanalysis_column_is_counted(self) -> None:
        packed, target, policy, q = fixture()
        v3 = np.zeros((3, 80), dtype=np.float32)
        v3[:, :64] = target
        v3[:, 63] = 3
        v3[0, 64] = 1
        v3[2, 64] = 1  # Excluded by the full-board gate.
        result = audit_q_candidates.count_arrays(packed, v3, policy, q)
        self.assertEqual(result["target_version"], 3)
        self.assertEqual(result["eligible_reanalyzed_rows"], 1)


if __name__ == "__main__":
    unittest.main()
