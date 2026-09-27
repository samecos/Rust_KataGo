"""Synthetic CPU and optional Apple MPS checks for the restricted v7 policy probe."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

import audit_npz
import train_probe


def _hash_chunks(hash0: int, hash1: int) -> tuple[int, ...]:
    return (
        hash0 & ((1 << 22) - 1),
        (hash0 >> 22) & ((1 << 22) - 1),
        (hash0 >> 44) & ((1 << 20) - 1),
        hash1 & ((1 << 22) - 1),
        (hash1 >> 22) & ((1 << 22) - 1),
        (hash1 >> 44) & ((1 << 20) - 1),
    )


def synthetic_v7_arrays(games: int = 150) -> dict[str, np.ndarray]:
    rows = games * 2
    binary = np.zeros((rows, 22, audit_npz.AREA), dtype=np.uint8)
    binary[:, 0, :] = 1
    global_input = np.zeros((rows, 19), dtype=np.float32)
    global_target = np.zeros((rows, 64), dtype=np.float32)
    global_target[:, 25] = 1
    global_target[:, 26] = 1
    global_target[:, 63] = 2
    policy = np.zeros((rows, 2, audit_npz.POLICY_SIZE), dtype=np.int16)
    for game in range(games):
        game_hash = ((game + 1) * 1099511628211, (game + 7) * 16777619)
        for turn in range(2):
            row = game * 2 + turn
            binary[row, 1, (game * 3 + turn) % audit_npz.AREA] = 1
            global_input[row, 0] = game / games
            global_target[row, 41:47] = _hash_chunks(*game_hash)
            global_target[row, 51] = turn
            policy[row, 0, (game + turn) % audit_npz.POLICY_SIZE] = 10
            policy[row, 0, (game + turn + 1) % audit_npz.POLICY_SIZE] = 2
    # Valid archive rows that must not enter the supervised probe.
    global_target[1, 25] = 0
    global_target[3, 26] = 0
    policy[3, 0, :] = 1  # v7 fallback when policy target0 is unavailable
    return {
        "binaryInputNCHWPacked": np.packbits(binary, axis=2, bitorder="big"),
        "globalInputNC": global_input,
        "policyTargetsNCMove": policy,
        "globalTargetsNC": global_target,
        "scoreDistrN": np.zeros((rows, audit_npz.SCORE_DISTR_SIZE), dtype=np.int8),
        "valueTargetsNCHW": np.zeros((rows, 5, 19, 19), dtype=np.int8),
        "qValueTargetsNCMove": np.zeros((rows, 3, audit_npz.POLICY_SIZE), dtype=np.int16),
    }


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, array in arrays.items():
            payload = io.BytesIO()
            np.save(payload, array, allow_pickle=False)
            archive.writestr(name, payload.getvalue())


class TrainProbeDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.path = self.folder / "synthetic.npz"
        write_npz(self.path, synthetic_v7_arrays())

    def test_loader_filters_rows_and_keeps_each_game_in_one_split(self) -> None:
        data = train_probe.load_probe_data([self.path], split_seed="probe-test")
        self.assertEqual(len(data.split_code), 298)
        self.assertEqual(data.packed.dtype, np.uint8)
        self.assertEqual(data.global_input.dtype, np.float32)
        self.assertEqual(data.target.dtype, np.int16)
        self.assertEqual(data.row_weight.dtype, np.float32)
        self.assertEqual(sum(data.summary()["rows_per_split"].values()), 298)
        self.assertTrue(all(data.summary()["rows_per_split"][name] > 0
                            for name in audit_npz.SPLIT_NAMES))
        expected_games = [
            game for game in range(150) for turn in range(2)
            if not (game == 0 and turn == 1) and not (game == 1 and turn == 1)
        ]
        observed: dict[int, set[int]] = {}
        for game, code in zip(expected_games, data.split_code):
            observed.setdefault(game, set()).add(int(code))
            hash_pair = ((game + 1) * 1099511628211, (game + 7) * 16777619)
            expected = train_probe.SPLIT_CODE[
                audit_npz._split_for_game(hash_pair, "probe-test")
            ]
            self.assertEqual(int(code), expected)
        self.assertTrue(all(len(codes) == 1 for codes in observed.values()))
        spatial = train_probe.unpack_batch(data.packed[:1])
        self.assertEqual(spatial.shape, (1, 22, 19, 19))
        self.assertTrue(np.all(spatial[0, 0] == 1))
        self.assertEqual(spatial[0, 1, 0, 0], 1)
        self.assertEqual(data.target[0].sum(), 12)

    def test_split_mapping_is_order_independent(self) -> None:
        hashes = [(9, 10), (1, 2), (9, 10), (5, 6)]
        first = train_probe.split_codes_for_hashes(hashes, "same-seed")
        second = train_probe.split_codes_for_hashes(hashes[::-1], "same-seed")
        self.assertEqual(int(first[0]), int(first[2]))
        self.assertEqual(first.tolist(), second[::-1].tolist())

    def test_explicit_mixed_board_selection_excludes_non19_row(self) -> None:
        arrays = synthetic_v7_arrays()
        arrays.pop("qValueTargetsNCMove")
        arrays["binaryInputNCHWPacked"][0, 0, 0] = 0
        write_npz(self.path, arrays)
        with self.assertRaises(ValueError):
            train_probe.load_probe_data([self.path])
        data = train_probe.load_probe_data(
            [self.path], allow_missing_qvalue=True, full_board_only=True
        )
        self.assertEqual(len(data.split_code), 297)
        self.assertTrue(np.all(train_probe._full_board_mask(data.packed)))
        self.assertEqual(data.audit["files"][0]["spatial_msb"]["full_19x19_rows"], 299)

    def test_v3_targets_use_only_shared_policy_fields(self) -> None:
        arrays = synthetic_v7_arrays()
        v3 = np.zeros((len(arrays["globalTargetsNC"]), 80), dtype=np.float32)
        v3[:, :64] = arrays["globalTargetsNC"]
        v3[:, 63] = 3
        v3[:, 64] = 1
        v3[:, 65:] = 99  # extra targets must never become model inputs
        arrays["globalTargetsNC"] = v3
        write_npz(self.path, arrays)
        data = train_probe.load_probe_data([self.path])
        self.assertEqual(len(data.split_code), 298)
        self.assertEqual(data.global_input.shape[1], 19)
        self.assertEqual(data.audit["files"][0]["target_schema_version"], 3)

    def test_inspect_manifest_without_torch(self) -> None:
        output = self.folder / "inspect"
        with redirect_stdout(io.StringIO()):
            code = train_probe.main([
                str(self.path), "--out-dir", str(output),
                "--split-seed", "probe-test", "--inspect-only",
            ])
        self.assertEqual(code, 0)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "inspected")
        self.assertEqual(manifest["data"]["eligible_rows"], 298)
        self.assertEqual(len(manifest["source_files"][0]["sha256"]), 64)
        self.assertNotIn("runs", manifest)

    @unittest.skipUnless(train_probe.torch is None, "torch is installed")
    def test_missing_torch_explains_dependency(self) -> None:
        output = self.folder / "missing-torch"
        with redirect_stderr(io.StringIO()):
            code = train_probe.main([str(self.path), "--out-dir", str(output)])
        self.assertEqual(code, 2)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertIn("PyTorch (torch) is required", manifest["error"])


@unittest.skipIf(train_probe.torch is None, "torch is not installed")
class TrainProbeTorchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "synthetic.npz"
        write_npz(self.path, synthetic_v7_arrays(games=12))

    def _check_training_batch(self, device_name: str) -> None:
        torch = train_probe.torch
        assert torch is not None
        torch.set_num_threads(1)
        device = train_probe.resolve_device(device_name)
        data = train_probe.load_probe_data([self.path], split_seed="probe-test")
        similarity, linear = train_probe.make_paired_models(seed=17, width=16)
        self.assertEqual(train_probe.head_parameter_count(similarity),
                         train_probe.head_parameter_count(linear))
        self.assertEqual(train_probe.head_parameter_count(similarity),
                         (train_probe.TRUNK_FEATURES + audit_npz.POLICY_SIZE) * 16 + 1)
        indices = np.arange(4)
        for model in (similarity, linear):
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            loss = train_probe.train_epoch(
                model, data, indices, batch_size=4, optimizer=optimizer,
                device=device, rng=np.random.default_rng(17),
            )
            self.assertTrue(np.isfinite(loss))
            metrics = train_probe.evaluate(
                model, data, indices, batch_size=4, device=device
            )
            for key in ("cross_entropy", "kl_teacher_to_model", "top1_teacher_agreement"):
                self.assertTrue(np.isfinite(metrics[key]))

    def test_equal_head_parameters_and_one_finite_cpu_training_batch(self) -> None:
        self._check_training_batch("cpu")

    @unittest.skipUnless(train_probe.torch is not None and
                         train_probe.torch.backends.mps.is_available(),
                         "Apple MPS is unavailable")
    def test_one_finite_mps_training_batch(self) -> None:
        self.assertEqual(str(train_probe.resolve_device("auto")), "mps")
        self._check_training_batch("mps")


if __name__ == "__main__":
    unittest.main()
