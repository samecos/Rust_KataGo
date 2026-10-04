"""CPU checks for global game splits, dedup precedence, and atomic source gates."""
from __future__ import annotations

from contextlib import redirect_stdout
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import audit_npz
import merge_model_caches as merge


GLOBAL_SEED = "test-global-game-split"
LABELS = {"perspective": "player to move", "input_only": ["packed", "global_input"], "policy_counts": "MCTS visits"}


def game_for_split(name):
    for number in range(1, 10000):
        game = number, number + 10000
        if audit_npz._split_for_game(game, GLOBAL_SEED) == name:
            return game
    raise AssertionError("could not find a synthetic game in requested split")


def arrays_for_rows(games, seed, markers=None):
    rows = len(games)
    arrays = {name: np.zeros((rows, *shape), dtype=dtype) for name, (shape, dtype) in merge.FIELDS.items()}
    arrays["packed"][:, 0, :45] = 255
    arrays["packed"][:, 0, 45] = 128
    arrays["policy_counts"][:, 5] = 10
    arrays["value_probs"][:, 0] = 1
    arrays["score_distr"][:, 420] = 100
    for name in ("row_weight", "value_weight", "score_mean_weight", "score_weight", "ownership_weight"):
        arrays[name][:] = 1
    arrays["game_hash"][:] = games
    arrays["split_code"][:] = [merge.prepare_model_data.SPLIT_CODE[audit_npz._split_for_game(game, seed)] for game in games]
    if markers is not None:
        arrays["global_input"][:, 0] = markers
    return arrays


class MergeCachesTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.folder = Path(self.temporary.name).resolve()
        self.assertTrue(self.folder.is_relative_to(Path(__file__).resolve().parent))
        self.addCleanup(self.temporary.cleanup)
        self.created_sources = 0

    def cache(self, arrays, seed="source-split", *, labels=None):
        self.created_sources += 1
        index = self.created_sources
        folder = self.folder / f"cache-{index}"
        folder.mkdir()
        archive = self.folder / f"source-{index}.tgz"
        archive.write_bytes(gzip.compress(f"synthetic identity {index}".encode()))
        shard = folder / "shard-00000.npz"
        np.savez_compressed(shard, **arrays)
        rows = len(arrays["split_code"])
        manifest = {"schema": merge.SCHEMA, "status": "completed", "shards": 1,
                    "source": {"kind": "archive", "path": str(archive), "bytes": archive.stat().st_size, "sha256": merge.sha256_file(archive)},
                    "split": {"seed": seed, "codes": merge.prepare_model_data.SPLIT_CODE,
                              "rows": {name: int(np.count_nonzero(arrays["split_code"] == code)) for name, code in merge.prepare_model_data.SPLIT_CODE.items()}},
                    "totals": {"selected_rows": rows}, "labels": labels or LABELS}
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return folder

    def run_merge(self, inputs, output="merged", shard_rows=2):
        with redirect_stdout(io.StringIO()):
            return merge.merge_caches(inputs, self.folder / output, GLOBAL_SEED, shard_rows)

    def output_arrays(self, output="merged"):
        files = sorted((self.folder / output).glob("shard-*.npz"))
        combined = {name: [] for name in merge.FIELDS}
        for path in files:
            with np.load(path, allow_pickle=False) as shard:
                for name in merge.FIELDS:
                    combined[name].append(shard[name])
        return {name: np.concatenate(parts) for name, parts in combined.items()}

    def test_global_split_recode_is_game_disjoint_and_sources_unchanged(self):
        games = [game_for_split(name) for name in ("train", "val", "test")]
        cache_a = self.cache(arrays_for_rows(games, "source-A"), "source-A")
        cache_b = self.cache(arrays_for_rows([(900, 901)], "source-B"), "source-B")
        before = {path: merge.sha256_file(path) for path in self.folder.rglob("*") if path.is_file()}
        manifest = self.run_merge([cache_a, cache_b])
        actual = self.output_arrays()
        for game, code in zip(actual["game_hash"], actual["split_code"], strict=True):
            expected = merge.prepare_model_data.SPLIT_CODE[audit_npz._split_for_game(tuple(int(x) for x in game), GLOBAL_SEED)]
            self.assertEqual(int(code), expected)
        self.assertEqual(manifest["split"]["cross_split_game_intersections"], 0)
        self.assertEqual(sum(manifest["split"]["games"].values()), 4)
        self.assertTrue(all(merge.sha256_file(path) == digest for path, digest in before.items()))
        self.assertEqual(manifest["source"]["sources"][0]["archive"]["sha256"], merge.sha256_file(self.folder / "source-1.tgz"))
        self.assertEqual(manifest["source"]["sources"][0]["cache_manifest_sha256"], merge.sha256_file(cache_a / "manifest.json"))
        self.assertEqual([entry["rows"] for entry in manifest["shard_inventory"]], [2, 2])

    def test_entire_game_precedence_and_exact_row_dedup_preserve_different_positions(self):
        game = game_for_split("train")
        other = game_for_split("test")
        first = self.cache(arrays_for_rows([game, game, game], "source-split", [1, 1, 2]))
        second = self.cache(arrays_for_rows([game, other], "source-split", [3, 4]))
        manifest = self.run_merge([first, second])
        actual = self.output_arrays()
        self.assertEqual(actual["global_input"][:, 0].tolist(), [1, 2, 4])
        self.assertEqual(manifest["totals"]["selected_rows"], 3)
        self.assertEqual(manifest["totals"]["same_game_later_cache_rows_skipped"], 1)
        self.assertEqual(manifest["totals"]["exact_complete_duplicate_rows_skipped"], 1)
        self.assertEqual(manifest["source"]["sources"][1]["overlapping_games_skipped"], 1)
        self.assertIn("complementary", manifest["deduplication"]["game_policy"])

    def test_complete_row_hash_ignores_prior_split_but_includes_targets(self):
        data = arrays_for_rows([(1, 2)], "source-split")
        original = merge.row_digest(data, 0)
        data["split_code"][0] = (int(data["split_code"][0]) + 1) % 3
        self.assertEqual(merge.row_digest(data, 0), original)
        data["score_mean"][0] = 1
        self.assertNotEqual(merge.row_digest(data, 0), original)

    def test_loader_consumes_merged_schema_with_reproducible_selected_data_sha(self):
        import train_student
        games = [game_for_split(name) for name in ("train", "val", "test")]
        cache = self.cache(arrays_for_rows(games, "source-split"))
        self.run_merge([cache])
        limits = {"train": 1, "val": 1, "test": 1}
        data, info = train_student.load_selected(self.folder / "merged", limits, 20261003)
        replay, replay_info = train_student.load_selected(self.folder / "merged", limits, 20261003)
        self.assertEqual(info["rows_selected"], limits)
        self.assertEqual(info["selected_data_sha256"], replay_info["selected_data_sha256"])
        for name in data:
            np.testing.assert_array_equal(data[name], replay[name])

    def test_bad_dtype_nonfinite_nonfull_or_source_split_fails_without_publication(self):
        mutations = [lambda a: a.update(global_input=a["global_input"].astype(np.float64)),
                     lambda a: a["score_mean"].__setitem__(0, float("nan")),
                     lambda a: a["packed"].__setitem__((0, 0, 0), 0),
                     lambda a: a["split_code"].__setitem__(0, (int(a["split_code"][0]) + 1) % 3)]
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=mutation):
                arrays = arrays_for_rows([(1, 2)], "source-split")
                mutation(arrays)
                cache = self.cache(arrays)
                output = f"invalid-{index}"
                with self.assertRaisesRegex(ValueError, "partial evidence preserved"):
                    self.run_merge([cache], output)
                self.assertFalse((self.folder / output).exists())
        self.assertEqual(len(list(self.folder.glob(".merged-model-cache-*/failure.json"))), len(mutations))

    def test_source_archive_and_duplicate_archive_identity_rejected(self):
        first = self.cache(arrays_for_rows([(1, 2)], "source-split"))
        second = self.cache(arrays_for_rows([(3, 4)], "source-split"))
        manifest_path = second / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        first_source = json.loads((first / "manifest.json").read_text())["source"]
        manifest["source"] = first_source
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "duplicate source archive"):
            self.run_merge([first, second])
        (self.folder / "source-1.tgz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "archive identity differs"):
            self.run_merge([first])
        self.assertFalse((self.folder / "merged").exists())

    def test_changed_source_after_read_is_detected_and_partial_files_preserved(self):
        cache = self.cache(arrays_for_rows([(1, 2)], "source-split"))
        real_read = merge.read_shard

        def mutate_after_read(path):
            arrays = real_read(path)
            changed = {name: value.copy() for name, value in arrays.items()}
            changed["global_input"][0, 0] = 10
            np.savez_compressed(path, **changed)
            return arrays

        with patch.object(merge, "read_shard", side_effect=mutate_after_read):
            with self.assertRaisesRegex(ValueError, "source changed during merge"):
                self.run_merge([cache])
        self.assertFalse((self.folder / "merged").exists())
        self.assertEqual(len(list(self.folder.glob(".merged-model-cache-*/failure.json"))), 1)
        self.assertEqual(len(list(self.folder.glob(".merged-model-cache-*/shard-*.npz"))), 1)

    def test_fresh_output_source_directory_and_semantic_gates(self):
        first = self.cache(arrays_for_rows([(1, 2)], "source-split"))
        output = self.folder / "merged"
        output.mkdir()
        marker = output / "old-evidence"
        marker.write_text("preserve")
        with self.assertRaises(FileExistsError):
            self.run_merge([first])
        self.assertEqual(marker.read_text(), "preserve")
        with self.assertRaisesRegex(ValueError, "inside a source cache"):
            merge.merge_caches([first], first / "new-output", GLOBAL_SEED)
        second = self.cache(arrays_for_rows([(3, 4)], "source-split"), labels={"perspective": "wrong"})
        with self.assertRaisesRegex(ValueError, "different target/input semantics"):
            self.run_merge([first, second], "different-labels")


if __name__ == "__main__":
    unittest.main()
