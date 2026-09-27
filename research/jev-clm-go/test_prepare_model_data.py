"""Focused synthetic checks for the multitask data cache and holdout gates."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np

import audit_npz
import prepare_model_data


def _hash_chunks(first: int, second: int) -> tuple[int, ...]:
    return (
        first & ((1 << 22) - 1), (first >> 22) & ((1 << 22) - 1),
        (first >> 44) & ((1 << 20) - 1), second & ((1 << 22) - 1),
        (second >> 22) & ((1 << 22) - 1), (second >> 44) & ((1 << 20) - 1),
    )


def _arrays(games: list[tuple[int, int]], version: int = 2) -> dict[str, np.ndarray]:
    n = len(games)
    binary = np.zeros((n, 22, audit_npz.AREA), dtype=np.uint8)
    binary[:, 0] = 1
    target = np.zeros((n, 64 if version == 2 else 80), dtype=np.float32)
    target[:, 0] = 1.0
    target[:, 3] = 4.5
    target[:, 25] = 1.0
    target[:, 26] = 1.0
    target[:, 27] = 1.0
    target[:, 62] = 1.0
    target[:, 63] = version
    policy = np.zeros((n, 2, 362), dtype=np.int16)
    policy[:, 0, 5] = 10
    score = np.zeros((n, 842), dtype=np.int8)
    score[:, 425] = 100
    ownership = np.zeros((n, 5, 19, 19), dtype=np.int8)
    ownership[:, 0, 0, 0] = 1
    for i, game in enumerate(games):
        target[i, 41:47] = _hash_chunks(*game)
        target[i, 51] = i
    return {
        "binaryInputNCHWPacked": np.packbits(binary, axis=2, bitorder="big"),
        "globalInputNC": np.zeros((n, 19), dtype=np.float32),
        "policyTargetsNCMove": policy,
        "globalTargetsNC": target,
        "scoreDistrN": score,
        "valueTargetsNCHW": ownership,
    }


def _npz_blob(arrays: dict[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


class PrepareModelDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_tar_excludes_prior_files_and_games_and_masks_nonfinal_labels(self) -> None:
        game_a = (1, 11)
        game_b = (2, 22)
        game_c = (3, 33)
        kept = _arrays([game_a, game_b, game_c, game_c, game_c])
        kept["globalTargetsNC"][2, 26] = 0
        kept["binaryInputNCHWPacked"][3, 0, 0] = 0
        kept["globalTargetsNC"][4, 62] = 0
        prior_blob = _npz_blob(_arrays([game_b]))
        kept_blob = _npz_blob(kept)
        archive_path = self.root / "synthetic.tgz"
        prior_name = "synthetic/model/prior.npz"
        kept_name = "synthetic/model/kept.npz"
        with tarfile.open(archive_path, mode="w:gz") as archive:
            for name, blob in ((kept_name, kept_blob), (prior_name, prior_blob)):
                info = tarfile.TarInfo(name)
                info.size = len(blob)
                archive.addfile(info, io.BytesIO(blob))
        prior_dir = self.root / "prior"
        (prior_dir / "model").mkdir(parents=True)
        (prior_dir / "model" / "prior.npz").write_bytes(prior_blob)
        manifest_path = self.root / "sample_manifest.json"
        manifest_path.write_text(json.dumps({
            "schema": "rust_katago_clm_tar_sample_v1",
            "source_tar": {"bytes": archive_path.stat().st_size,
                           "sha256": prepare_model_data.sha256_file(archive_path)},
            "selected_files": [{"archive_path": prior_name,
                                "extracted_path": "model/prior.npz",
                                "sha256": hashlib.sha256(prior_blob).hexdigest()}],
        }), encoding="utf-8")
        out = self.root / "cache"
        result = prepare_model_data.prepare_model_data(
            out_dir=out, split_seed="test-multitask", archive_path=archive_path,
            exclude_manifest=manifest_path, prior_sample_dir=prior_dir,
            shard_rows=1,
        )
        self.assertEqual(result["source"]["excluded_npz_files"], 1)
        self.assertEqual(result["totals"]["source_rows"], 5)
        self.assertEqual(result["totals"]["nonfull_rows"], 1)
        self.assertEqual(result["totals"]["invalid_policy_rows"], 1)
        self.assertEqual(result["totals"]["prior_game_rows"], 1)
        self.assertEqual(result["totals"]["selected_rows"], 2)
        self.assertTrue(result["exclusion"]["game_disjoint_from_prior_sample"])
        with np.load(out / "shard-00000.npz") as shard:
            self.assertEqual(shard["packed"].shape, (2, 22, 46))
            self.assertEqual(shard["game_hash"].tolist(), [[1, 11], [3, 33]])
            self.assertEqual(shard["value_weight"].tolist(), [1.0, 0.0])
            self.assertEqual(shard["score_mean_weight"].tolist(), [1.0, 0.0])
            self.assertEqual(shard["ownership_weight"].tolist(), [1.0, 0.0])
            self.assertEqual(shard["score_mean"].tolist(), [4.5, 4.5])
            self.assertEqual(shard["value_probs"][0].tolist(), [1.0, 0.0, 0.0])
            self.assertEqual(shard["ownership"][0, 0, 0], 1)
            self.assertEqual(int(shard["split_code"][0]), prepare_model_data.SPLIT_CODE[
                audit_npz._split_for_game(game_a, "test-multitask")])
            self.assertEqual(int(shard["split_code"][1]), prepare_model_data.SPLIT_CODE[
                audit_npz._split_for_game(game_c, "test-multitask")])

    def test_directory_v3_and_score_distribution_validation(self) -> None:
        folder = self.root / "input"
        folder.mkdir()
        path = folder / "v3.npz"
        arrays = _arrays([(7, 70)], version=3)
        path.write_bytes(_npz_blob(arrays))
        out = self.root / "v3-cache"
        result = prepare_model_data.prepare_model_data(
            out_dir=out, split_seed="v3-test", input_dir=folder,
        )
        self.assertEqual(result["totals"]["selected_rows"], 1)
        with np.load(out / "shard-00000.npz") as shard:
            self.assertEqual(shard["score_distr"][0].sum(), 100)
            self.assertEqual(shard["value_probs"].shape, (1, 3))
        arrays["scoreDistrN"][0, 425] = 99
        path.write_bytes(_npz_blob(arrays))
        with self.assertRaisesRegex(ValueError, "score distribution"):
            prepare_model_data.prepare_model_data(
                out_dir=self.root / "bad-cache", split_seed="v3-test", input_dir=folder,
            )
        self.assertFalse((self.root / "bad-cache").exists())


if __name__ == "__main__":
    unittest.main()
