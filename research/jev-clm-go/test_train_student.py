"""Checkpoint continuity and fail-closed device/data tests for student training."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import smoke_student_cuda
import train_student
import train_probe


class StudentTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"
        smoke_student_cuda.synthetic_cache(self.cache)

    def args(self, name: str, epochs: int, resume: bool = False) -> object:
        values = ["--cache-dir", str(self.cache), "--out-dir", str(self.root / name),
                  "--device", "cpu", "--precision", "fp32", "--variants", "compact",
                  "--epochs", str(epochs), "--batch-size", "8",
                  "--train-rows", "0", "--val-rows", "0", "--test-rows", "0"]
        return train_student.parser().parse_args(values + (["--resume"] if resume else []))

    def test_epoch_resume_preserves_optimizer_and_order(self) -> None:
        train_student.run(self.args("whole", 2))
        train_student.run(self.args("split", 1))
        resumed = train_student.run(self.args("split", 2, True))
        whole = torch.load(self.root / "whole" / "compact.last.pt", weights_only=True)
        split = torch.load(self.root / "split" / "compact.last.pt", weights_only=True)
        for name in whole["model_state"]:
            torch.testing.assert_close(whole["model_state"][name], split["model_state"][name], rtol=0, atol=0)
        self.assertEqual(whole["order_rng_state"], split["order_rng_state"])
        self.assertEqual(split["epoch"], 2)
        self.assertEqual(resumed["models"]["compact"]["resumed_from_epoch"], 1)
        self.assertEqual(len(resumed["models"]["compact"]["epochs"]), 2)
        self.assertNotIn("learning_rate_schedule", split["identity"])

    def test_cosine_resume_same_total_is_exact_and_horizon_is_locked(self) -> None:
        whole_args = self.args("cosine-whole", 3)
        whole_args.lr_schedule = "cosine"
        split_args = self.args("cosine-split", 3)
        split_args.lr_schedule = "cosine"
        train_student.run(whole_args)
        original_epoch = train_student.train_epoch
        calls = 0

        def interrupt_second_epoch(*values: object, **options: object) -> object:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic interruption within fixed cosine budget")
            return original_epoch(*values, **options)

        with mock.patch.object(train_student, "train_epoch", side_effect=interrupt_second_epoch):
            with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                train_student.run(split_args)
        split_args.resume = True
        restored = train_student.run(split_args)
        whole = torch.load(self.root / "cosine-whole" / "compact.last.pt", weights_only=True)
        split = torch.load(self.root / "cosine-split" / "compact.last.pt", weights_only=True)
        for name in whole["model_state"]:
            torch.testing.assert_close(whole["model_state"][name], split["model_state"][name], rtol=0, atol=0)
        rates = [record["learning_rate"] for record in restored["models"]["compact"]["epochs"]]
        self.assertAlmostEqual(rates[0], 0.001)
        self.assertAlmostEqual(rates[1], 0.000505)
        self.assertAlmostEqual(rates[2], 0.00001)
        self.assertEqual(split["optimizer_state"]["param_groups"][0]["lr"], 0.00001)
        previous = (self.root / "cosine-split" / "manifest.json").read_bytes()
        split_args.epochs = 4
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            train_student.run(split_args)
        self.assertEqual((self.root / "cosine-split" / "manifest.json").read_bytes(), previous)

    def test_resume_rejects_changed_inputs_before_writing_manifest(self) -> None:
        train_student.run(self.args("run", 1))
        manifest = self.root / "run" / "manifest.json"
        old = manifest.read_bytes()
        shard_path = self.cache / "shard-00000.npz"
        with np.load(shard_path, allow_pickle=False) as shard:
            arrays = {name: shard[name] for name in shard.files}
        arrays["global_input"][0, 0] += 1
        np.savez_compressed(shard_path, **arrays)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            train_student.run(self.args("run", 2, True))
        self.assertEqual(manifest.read_bytes(), old)

    def test_existing_output_and_completed_epoch_are_not_overwritten(self) -> None:
        train_student.run(self.args("run", 1))
        with self.assertRaises(FileExistsError):
            train_student.run(self.args("run", 1))
        with self.assertRaisesRegex(ValueError, "must exceed"):
            train_student.run(self.args("run", 1, True))

    def test_cuda_is_default_and_unavailable_cuda_errors(self) -> None:
        self.assertEqual(train_student.parser().parse_args([
            "--cache-dir", "missing", "--out-dir", "unused"]).device, "cuda")
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA requested but unavailable"):
                train_student.resolve_device("cuda")
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            train_student.resolve_precision("fp16", torch.device("cpu"))

    def test_partial_second_variant_resumes_without_retraining_completed_first(self) -> None:
        args = self.args("partial", 1)
        args.variants = ["dense", "compact"]
        original = train_student.train_epoch
        calls = 0

        def fail_second(*values: object, **options: object) -> object:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic interruption before compact epoch commit")
            return original(*values, **options)

        with mock.patch.object(train_student, "train_epoch", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                train_student.run(args)
        dense_path = self.root / "partial" / "dense.last.pt"
        before = dense_path.read_bytes()
        args.resume = True
        result = train_student.run(args)
        self.assertEqual(dense_path.read_bytes(), before)
        self.assertTrue(result["models"]["dense"]["resumed_skipped_completed"])
        self.assertEqual(result["models"]["compact"]["epochs"][0]["epoch"], 1)
        self.assertTrue((self.root / "partial" / "compact.rgmodel").is_file())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_probe_both_heads_take_real_cuda_optimizer_steps(self) -> None:
        with np.load(self.cache / "shard-00000.npz", allow_pickle=False) as source:
            data = train_probe.ProbeData(
                packed=source["packed"], global_input=source["global_input"],
                target=source["policy_counts"], row_weight=source["row_weight"],
                split_code=source["split_code"], split_seed="synthetic",
                games_per_split={"train": 16, "val": 8, "test": 8}, audit={},
            )
        device = train_probe.resolve_device("cuda")
        for model in train_probe.make_paired_models(17, 256):
            model.to(device)
            before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            loss = train_probe.train_epoch(model, data, np.arange(8), 4, optimizer, device, np.random.default_rng(17))
            self.assertTrue(np.isfinite(loss))
            self.assertTrue(all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                                for parameter in model.parameters()))
            self.assertTrue(any(not torch.equal(before[name], parameter)
                                for name, parameter in model.named_parameters()))

    def test_resume_recovers_best_if_interrupted_before_last_checkpoint_commit(self) -> None:
        train_student.run(self.args("interrupted", 1))
        best_path = self.root / "interrupted" / "compact.pt"
        before = torch.load(best_path, weights_only=True)
        original_save = train_student.atomic_checkpoint
        original_eval = train_student.evaluate

        def force_best(*values: object, **options: object) -> object:
            metrics = original_eval(*values, **options)
            metrics["total_loss"] = 0.0  # Force publication of a new best.
            return metrics

        def interrupt_commit(path: Path, payload: object) -> None:
            if path.name == "compact.last.pt":
                raise OSError("synthetic crash after best publication")
            original_save(path, payload)

        with mock.patch.object(train_student, "evaluate", side_effect=force_best), mock.patch.object(
            train_student, "atomic_checkpoint", side_effect=interrupt_commit
        ):
            with self.assertRaisesRegex(OSError, "synthetic crash"):
                train_student.run(self.args("interrupted", 2, True))
        self.assertEqual(torch.load(best_path, weights_only=True)["epoch"], 2)
        original_epoch = train_student.train_epoch

        def verify_recovered(*values: object, **options: object) -> object:
            restored = torch.load(best_path, weights_only=True)
            self.assertEqual(restored["epoch"], 1)
            for name in before["model_state"]:
                torch.testing.assert_close(restored["model_state"][name], before["model_state"][name], rtol=0, atol=0)
            return original_epoch(*values, **options)

        with mock.patch.object(train_student, "train_epoch", side_effect=verify_recovered):
            result = train_student.run(self.args("interrupted", 2, True))
        self.assertTrue(result["models"]["compact"]["recovered_best_from_last_epoch"])
        self.assertEqual(len(list((self.root / "interrupted").glob("compact.best-before-resume.*.pt"))), 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_fp16_scaler_overflow_skips_step_and_reduces_scale(self) -> None:
        device = train_student.resolve_device("cuda")
        train_student.isolate_windows_cudnn(torch)
        data, _ = train_student.load_selected(self.cache, {"train": 0, "val": 0, "test": 0}, 29)
        model = train_student.make_model("compact").to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scaler = torch.amp.GradScaler("cuda", init_scale=1e30)
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        result = train_student.train_epoch(model, data, np.arange(8), 8, optimizer,
                                          device, np.random.default_rng(17), "fp16", scaler)
        self.assertTrue(np.isfinite(result["total_loss"]))
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertEqual(result["scaled_overflow_steps"], 1)
        self.assertLess(scaler.get_scale(), 1e30)
        self.assertTrue(all(torch.equal(before[name], tensor) for name, tensor in model.state_dict().items()))
        # The same batch can then take a finite update at a lower scale.
        scaler.update(new_scale=1.0)
        updated = train_student.train_epoch(model, data, np.arange(8), 8, optimizer,
                                           device, np.random.default_rng(17), "fp16", scaler)
        self.assertEqual(updated["optimizer_steps"], 1)
        self.assertEqual(updated["scaled_overflow_steps"], 0)


if __name__ == "__main__":
    unittest.main()
