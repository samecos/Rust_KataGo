"""Focused checks for the standalone, same-device student speed benchmark."""

from __future__ import annotations

import io
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import torch

import benchmark_student
import student_models


def write_input_npz(path: Path, *, rows: int = 4) -> None:
    board = np.zeros((rows, 22, 361), dtype=np.uint8)
    board[:, 0, :] = 1
    board[0, 0, 0] = 0  # This smaller-board row must be filtered out.
    board[1:, 1, 123] = 1
    arrays = {
        "binaryInputNCHWPacked": np.packbits(board, axis=2, bitorder="big"),
        "globalInputNC": np.arange(rows * 19, dtype=np.float32).reshape(rows, 19),
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in arrays.items():
            stream = io.BytesIO()
            np.save(stream, value, allow_pickle=False)
            archive.writestr(name, stream.getvalue())


class BenchmarkStudentTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.data_dir = self.folder / "data"
        self.data_dir.mkdir()
        write_input_npz(self.data_dir / "sample.npz")

    def test_abba_balances_order_and_summary(self) -> None:
        self.assertEqual(benchmark_student.abba_order(0),
                         ("dense", "compact", "compact", "dense"))
        self.assertEqual(benchmark_student.abba_order(1),
                         ("compact", "dense", "dense", "compact"))
        summary = benchmark_student.summarize_samples(
            {"dense": [4.0, 6.0], "compact": [2.0, 3.0]}, batch=8
        )
        self.assertEqual(summary["compact_speedup_vs_dense"], 2.0)
        self.assertEqual(summary["compact"]["positions_per_second"], 3200.0)

    def test_real_npz_input_loading_filters_smaller_board(self) -> None:
        packed, global_input, sources = benchmark_student.load_input_rows(self.data_dir, 3)
        self.assertEqual(packed.shape, (3, 22, 46))
        self.assertEqual(global_input.shape, (3, 19))
        self.assertEqual(global_input[0, 0], 19)
        self.assertEqual(len(sources), 1)
        self.assertEqual(len(sources[0]["sha256"]), 64)
        spatial = benchmark_student.unpack_spatial(packed)
        self.assertEqual(spatial.shape, (3, 22, 19, 19))
        self.assertTrue(np.all(spatial[:, 0] == 1))
        self.assertTrue(np.all(spatial[:, 1, 6, 9] == 1))

    def test_portable_real_input_pack_matches_original_rows(self) -> None:
        packed, global_input, _ = benchmark_student.load_input_rows(self.data_dir, 3)
        path = self.folder / "benchmark-inputs.npz"
        np.savez_compressed(path, packed=packed, global_input=global_input)
        packed_copy, global_copy, sources = benchmark_student.load_input_pack(path, 2)
        np.testing.assert_array_equal(packed_copy, packed[:2])
        np.testing.assert_array_equal(global_copy, global_input[:2])
        self.assertEqual(sources[0]["kind"], "portable_real_input_pack")
        with self.assertRaisesRegex(ValueError, "too few rows"):
            benchmark_student.load_input_pack(path, 4)

    def test_cpu_checkpoint_loading_and_both_timing_modes(self) -> None:
        paths = {}
        for variant in ("dense", "compact"):
            model = student_models.make_model(variant)
            path = self.folder / f"{variant}.pt"
            torch.save({"variant": variant, "model_state": model.state_dict()}, path)
            paths[variant] = path
        loaded = {
            variant: benchmark_student.load_model(path, variant, torch.device("cpu"))
            for variant, path in paths.items()
        }
        self.assertGreater(loaded["dense"][1]["parameters"],
                           loaded["compact"][1]["parameters"])
        packed, global_input, _ = benchmark_student.load_input_rows(self.data_dir, 2)
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            result = benchmark_student.benchmark_pair(
                {variant: pair[0] for variant, pair in loaded.items()},
                packed, global_input, torch.device("cpu"),
                batches=[1, 2], rounds=2, iterations=1, warmup=1,
            )
        finally:
            torch.set_num_threads(old_threads)
        self.assertEqual(len(result), 4)
        for measurement in result.values():
            self.assertEqual(len(measurement["blocks"]), 8)
            self.assertEqual([block["variant"] for block in measurement["blocks"][:4]],
                             list(benchmark_student.abba_order(0)))
            self.assertGreater(measurement["summary"]["dense"]["median_ms_per_batch"], 0)
            self.assertGreater(measurement["summary"]["compact"]["median_ms_per_batch"], 0)

    def test_checkpoint_variant_is_enforced(self) -> None:
        path = self.folder / "wrong.pt"
        torch.save({"variant": "compact", "model_state": {}}, path)
        with self.assertRaisesRegex(ValueError, "expected checkpoint variant"):
            benchmark_student.load_model(path, "dense", torch.device("cpu"))

    @unittest.skipUnless(torch.backends.mps.is_available(), "Apple MPS unavailable")
    def test_mps_both_timing_modes_synchronize_and_copy_outputs(self) -> None:
        models = {variant: student_models.make_model(variant).to("mps").eval()
                  for variant in ("dense", "compact")}
        packed, global_input, _ = benchmark_student.load_input_rows(self.data_dir, 1)
        result = benchmark_student.benchmark_pair(
            models, packed, global_input, torch.device("mps"),
            batches=[1], rounds=1, iterations=1, warmup=1,
        )
        self.assertEqual(set(result), {"batch_1/forward_only", "batch_1/packed_to_cpu_outputs"})


if __name__ == "__main__":
    unittest.main()
