"""CPU regressions for configuration selection and fail-closed publication."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import tune_runtime as tune


class RuntimeTuneTests(unittest.TestCase):
    def test_abba_rejects_fast_but_unstable_candidate(self):
        result = tune.decide([100, 100], [110, 130], .01, .05)
        self.assertFalse(result["stable"])
        self.assertFalse(result["accepted"])

    def test_abba_rejects_unstable_baseline(self):
        self.assertFalse(tune.decide([90, 100], [120, 120], .01, .05)["accepted"])

    def test_abba_accepts_stable_gain_and_rejects_noise_or_regression(self):
        self.assertTrue(tune.decide([100, 101], [104, 105], .01, .05)["accepted"])
        self.assertFalse(tune.decide([100, 100], [100.5, 100.5], .01, .05)["accepted"])
        self.assertFalse(tune.decide([100, 100], [95, 95], .01, .05)["accepted"])

    def test_abba_rejects_missing_invalid_samples(self):
        for values in ([0, 10], [float("nan"), 10], [float("inf"), 10], [10]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                tune.decide(values, [10, 10], .01, .05)

    def test_parser_requires_complete_requested_result(self):
        partial = "numSearchThreads = 8: 1 / 3 positions, visits/s = 200.00, nnEvals/s = 150.00, nnBatches/s = 50.00, avgBatchSize = 3.00"
        final = partial.replace("1 / 3", "3 / 3").replace("200.00", "100.00")
        self.assertEqual(tune.parse_benchmark(partial + "\n" + final, 8, 3)["score"], 100)
        for text, threads, positions in ((partial, 8, 3), (final, 4, 3), (final, 8, 1), ("", 8, 3)):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                tune.parse_benchmark(text, threads, positions)

    def test_gtp_accepts_engine_id_spacing_but_rejects_missing_errors_and_bad_moves(self):
        for prefix in ("=", "= "):
            output = f"{prefix}1 \n\n{prefix}2 \n\n{prefix}3 C16\n\n{prefix}4 \n\n"
            tune.validate_gtp_smoke(output)
            for invalid in (output.replace("C16", "garbage"), output.replace(f"{prefix}2", "?2"),
                            output.replace(f"{prefix}4", f"{prefix}5")):
                with self.assertRaises(RuntimeError):
                    tune.validate_gtp_smoke(invalid)

    def test_identity_mismatch_never_rebinds_plan(self):
        fingerprint = {key: f"test-{key}" for key in tune.TARGET_KEYS}
        fingerprint["backend_build"] = {"kernel_build_id": "exact", "artifact": "exact"}
        plan = dict(schema=2, kind="cuda-tactic-plan", target={key: fingerprint[key] for key in tune.TARGET_KEYS},
                    backend_build=copy.deepcopy(fingerprint["backend_build"]))
        self.assertTrue(tune.match_plan(plan, fingerprint))
        for key in tune.TARGET_KEYS:
            other = copy.deepcopy(plan)
            other["target"][key] = "different"
            self.assertFalse(tune.match_plan(other, fingerprint))
        other = copy.deepcopy(plan)
        other["backend_build"]["artifact"] = "different"
        self.assertFalse(tune.match_plan(other, fingerprint))

    def test_config_and_launch_support_spaces_and_quotes(self):
        profile = dict(batch=14, lanes=2, plan="C:/My Model's Folder/plan.json")
        self.assertIn("nnMaxBatchSize = 14", tune.config_text(profile, 12, 800))
        script = tune.launch_script("C:/My Model's Folder/katago.exe", "model.bin.gz", "out.cfg", "nnworker", 64, "hash")
        self.assertIn("'C:/My Model''s Folder/katago.exe'", script)
        self.assertIn("'--capacity' '64'", script)
        self.assertIn("@args", script)
        self.assertIn("$input | &", script)
        self.assertIn("finally", script)
        with self.assertRaises(ValueError):
            tune.config_text(dict(profile, plan="bad#path.json"), 8, 800)

    def test_thread_bounds_and_duplicates(self):
        self.assertEqual(tune.thread_list("8,4,12"), [8, 4, 12])
        for value in ("", "0", "257", "8,8", "x"):
            with self.subTest(value=value), self.assertRaises(Exception):
                tune.thread_list(value)

    def test_environment_does_not_override_selection(self):
        with patch.dict(tune.os.environ, {"KATAGO_CUDA_NOGRAPH": "1", "KATAGO_NN_BATCH_WINDOW_US": "99", "TEST_KEEP": "yes"}):
            env = tune.clean_environment()
            self.assertNotIn("KATAGO_CUDA_NOGRAPH", env)
            self.assertNotIn("KATAGO_NN_BATCH_WINDOW_US", env)
            self.assertEqual(env["TEST_KEEP"], "yes")

    def test_process_failure_not_parsed_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            args = type("Args", (), dict(mode="local", timeout=10))()
            runner = tune.Tuner(args, Path(directory))
            with self.assertRaises(RuntimeError):
                runner.run_command([tune.sys.executable, "-c", "print('visits/s = 999'); raise SystemExit(3)"], "bad")

    def test_failure_does_not_publish_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary, model, output = root / "binary", root / "model", root / "out"
            binary.write_text("binary")
            model.write_text("model")
            with patch.object(tune.Tuner, "execute", side_effect=RuntimeError("fixture failure")), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = tune.main(["--binary", str(binary), "--model", str(model), "--output", str(output)])
            self.assertEqual(code, 1)
            self.assertFalse((output / "rustgo.cfg").exists())
            self.assertEqual(json.loads((output / "report.json").read_text())["status"], "FAIL")
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                tune.main(["--binary", str(binary), "--model", str(model), "--output", str(output)])

    def test_final_abba_regression_returns_to_original_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary, model = root / "engine", root / "model"
            binary.write_text("fixture engine")
            model.write_text("fixture model")
            args = SimpleNamespace(mode="local", binary=binary, model=model, threads=[8, 12],
                                   max_visits=800, capacity=32, min_improvement=.01, max_spread=.05,
                                   dry_run=False, smoke_only=False)
            runner = tune.Tuner(args, root)
            fp = {key: "fixture" for key in tune.TARGET_KEYS}
            fp.update(model_sha256=tune.file_hash(model), backend_build={})
            profiles = [dict(name="cuda-default", plan=None, batch=16, lanes=1)]
            with patch.object(runner, "run_command", return_value=json.dumps(fp)), \
                    patch.object(tune, "discover_profiles", return_value=(profiles, [])), \
                    patch.object(runner, "measure", side_effect=[100, 120, 120, 100, 100, 99, 99, 100]), \
                    patch.object(runner, "smoke_local") as smoke, contextlib.redirect_stdout(io.StringIO()):
                runner.execute()
            self.assertEqual(runner.report["selection_status"], "BASELINE_RETAINED")
            self.assertTrue(runner.report["comparisons"][0]["accepted"])
            self.assertFalse(runner.report["comparisons"][1]["accepted"])
            self.assertEqual(smoke.call_args.args[0]["threads"], 8)
            self.assertIn("numSearchThreads = 8", (root / "rustgo.cfg").read_text())
            self.assertTrue((root / "run-gtp.ps1").is_file())

    def test_smoke_failure_blocks_publication_after_successful_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary, model = root / "engine", root / "model"
            binary.write_text("fixture engine")
            model.write_text("fixture model")
            args = SimpleNamespace(mode="local", binary=binary, model=model, threads=[8], max_visits=800,
                                   capacity=32, min_improvement=.01, max_spread=.05, dry_run=False, smoke_only=False)
            runner = tune.Tuner(args, root)
            fp = {key: "fixture" for key in tune.TARGET_KEYS}
            fp.update(model_sha256=tune.file_hash(model), backend_build={})
            with patch.object(runner, "run_command", return_value=json.dumps(fp)), \
                    patch.object(tune, "discover_profiles", return_value=([dict(name="basic", plan=None, batch=16, lanes=1)], [])), \
                    patch.object(runner, "measure", return_value=100), \
                    patch.object(runner, "smoke_local", side_effect=RuntimeError("bad GTP reply")), \
                    self.assertRaises(RuntimeError), contextlib.redirect_stdout(io.StringIO()):
                runner.execute()
            self.assertFalse((root / "rustgo.cfg").exists())
            self.assertFalse((root / "run-gtp.ps1").exists())


if __name__ == "__main__":
    unittest.main()
