"""CPU-only launcher tests: real small files, mocked GPU query/native Popen."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import launch_quant_inference as launch


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.gpu = dict(index=0, name="Fixture GPU", uuid="GPU-fixture", compute_capability="12.0", driver="610.88")
        self.profile = "rustgo-quant-v1:" + "a" * 64
        self.binary = self.write("native.exe", b"inert executable fixture, never executed")
        self.model = self.write("original-model.bin", b"same model contents")
        self.alias = self.write("relocated-model.bin", b"same model contents")
        self.recipe = self.write("recipe.json", b'{"fixture":true}')
        self.environment = self.json_file("environment.json", {"values": {
            "PATH": "fixture", "SECRET_SENTINEL": "DO_NOT_EMIT_THIS_SECRET",
            "Katago_STALE": "wrong", "CUDA_VISIBLE_DEVICES": "wrong"}})
        self.nnbench = self.write("nnbench.rs", b"fixture metadata")
        self.runner = launch.source(Path(__file__).with_name("benchmark_quant_speed.py"))
        self.config_raw = ("nnBackend = cudaquantbackend\r\nnnMaxBatchSize = 3\r\n"
                           "cudaQuantPlan = " + self.recipe["path"] + "\r\n").encode()
        self.config = self.write("candidate.cfg", self.config_raw)
        self.fallback_binary = self.write("fallback.exe", b"inert fallback")
        self.fallback_config = self.write("fallback.cfg", b"unchanged caller configuration")
        self.candidate = dict(id="candidate", group="b15-b3", model=self.model, binary=self.binary,
                              config=self.config, recipe=self.recipe, backend="cudaquantbackend",
                              role="candidate", batch_cap=3, environment={key: "0" for key in launch.ENV_KEYS})
        self.baseline = dict(id="baseline", group="b15-b3", model=self.model, binary=self.binary,
                             config=self.fallback_config, backend="cudaint8backend", role="baseline",
                             batch_cap=3, environment={})
        comparison = {"speed_pass": True}
        profile = dict(group="b15-b3", speed_confirmed=True, accuracy=launch.ACCURACY,
                       baseline=self.baseline, candidate=self.candidate, inference_profile=self.profile,
                       formal=comparison, confirmation=comparison)
        globals_ = dict(binary=self.binary, runner=self.runner, environment=self.environment, source_nnbench=self.nnbench)
        plan = self.json_file("plan.json", dict(schema="rustgo-quant-speed-plan-v1", workers=32,
                                               variants=[self.baseline, self.candidate], **globals_))
        telemetry = dict(returncode=0, query=launch.OLD_GPU_QUERY,
                         output="Fixture GPU, GPU-fixture, 610.88, 40, 2000, 10000, 100, 0, 10")
        runs = [dict(stage=stage, comparison="b15-b3", flavor=role, variant=role,
                     status="COMPLETE_SPEED_SAMPLE", actual_exit=0, cleanup_errors=[],
                     metrics={"inference_profile": self.profile if role == "candidate" else None},
                     before=telemetry, after=telemetry)
                for stage, role in zip(("A1", "B1", "B2", "A2"), ("baseline", "candidate", "candidate", "baseline"))]
        report = dict(schema="rustgo-quant-speed-report-v1", status="COMPLETE_SPEED_ONLY", plan=plan,
                      comparisons={"b15-b3": comparison}, runs=runs)
        self.manifest = dict(schema="rustgo-speed-profile-handoff-v1", status="COMPLETE_SPEED_ONLY",
                             accuracy=launch.ACCURACY, workers=32, reset_inherited_katago_environment=True,
                             device=self.gpu["name"], driver=self.gpu["driver"], compute_capability="12.0",
                             profiles=[profile], formal_report=self.json_file("formal.json", report),
                             confirmation_report=self.json_file("confirmation.json", report), **globals_)
        self.manifest_source = self.json_file("profiles.json", self.manifest)
        self.counter = 0

    def write(self, name, raw):
        path = self.root / name
        path.write_bytes(raw)
        return launch.source(path)

    def json_file(self, name, value):
        return self.write(name, json.dumps(value).encode())

    def argv(self, *extra):
        self.counter += 1
        return ["--profiles", self.manifest_source["path"], "--profiles-sha256", self.manifest_source["sha256"],
                "--model", self.alias["path"], "--batch-cap", "3", "--output", str(self.root / ("out" + str(self.counter))),
                *extra]

    def invoke(self, argv, gpu=None, popen=None):
        out, err = io.StringIO(), io.StringIO()
        with patch.object(launch, "query_gpu", return_value=self.gpu if gpu is None else gpu), \
                patch.object(launch.subprocess, "Popen", side_effect=popen or AssertionError("unexpected native start")), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = launch.main(argv)
        directory = Path(argv[argv.index("--output") + 1])
        receipt = json.loads((directory / "receipt.json").read_text())
        self.assertNotIn("DO_NOT_EMIT_THIS_SECRET", out.getvalue() + err.getvalue() + json.dumps(receipt))
        self.assertEqual(out.getvalue(), "")
        return code, receipt, directory

    def test_prepare_relocated_model_expected_config_clean_environment_no_mutation(self):
        before = {p: p.read_bytes() for p in self.root.iterdir() if p.is_file()}
        argv = self.argv()
        args = launch.parser().parse_args(argv)
        selection, environment, config = launch.select(args, launch.Inventory(), gpu_query=lambda index: self.gpu)
        self.assertEqual(selection["model"], self.alias)
        self.assertEqual(selection["selection"], "MEASURED_PROFILE")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "GPU-fixture")
        self.assertEqual({k for k in environment if k.upper().startswith("KATAGO_")}, launch.ENV_KEYS)
        self.assertEqual(environment["SECRET_SENTINEL"], "DO_NOT_EMIT_THIS_SECRET")
        self.assertEqual(config, self.config_raw + ("cudaQuantExpectedProfile = " + self.profile + "\r\n").encode())
        code, receipt, directory = self.invoke(argv)
        self.assertEqual(code, 0)
        self.assertEqual(receipt["status"], "PREPARED")
        self.assertFalse(receipt["native_started"])
        self.assertIsNone(receipt["actual_exit"])
        self.assertEqual((directory / "selected.cfg").read_bytes(), config)
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_mismatch_explicit_fallback_preserves_config_and_environment(self):
        changes = [{"uuid": "GPU-other"}, {"driver": "999"}, {"name": "Other GPU"},
                   {"compute_capability": "9.0"}, {"index": 1}, {}, {}, {}]
        options = [[], [], [], [], [], ["--batch-cap", "8"], ["--capacity", "16"], []]
        for index, (change, extra) in enumerate(zip(changes, options)):
            with self.subTest(index=index):
                argv = self.argv("--fallback-binary", self.fallback_binary["path"],
                                 "--fallback-config", self.fallback_config["path"], *extra)
                if index == 7:
                    other = self.write("other-model.bin", b"different model")
                    argv[argv.index("--model") + 1] = other["path"]
                gpu = dict(self.gpu, **change)
                args = launch.parser().parse_args(argv)
                selection, environment, config = launch.select(args, launch.Inventory(), gpu_query=lambda _: gpu)
                self.assertEqual(selection["selection"], "EXPLICIT_FALLBACK")
                self.assertFalse(selection["measured_speed_applicable"])
                self.assertIsNone(environment)
                self.assertIsNone(config)
                code, receipt, directory = self.invoke(argv, gpu=gpu)
                self.assertEqual(code, 0)
                self.assertEqual(receipt["effective_config"], self.fallback_config)
                self.assertFalse((directory / "selected.cfg").exists())

    def test_no_match_and_partial_fallback_are_explicit(self):
        code, receipt, _ = self.invoke(self.argv("--capacity", "1"))
        self.assertEqual(code, 2)
        self.assertEqual(receipt["status"], "NO_MATCH")
        with self.assertRaisesRegex(ValueError, "supplied together"):
            launch.main(self.argv("--fallback-binary", self.fallback_binary["path"]))

    def test_matched_tampering_is_not_fallback(self):
        for original in (self.recipe, self.config, self.binary, self.runner):
            # The real parser is never edited: a tampered Source declaration tests it instead.
            with self.subTest(original=original["path"]):
                if original == self.runner:
                    manifest = copy.deepcopy(self.manifest)
                    manifest["runner"]["sha256"] = "0" * 64
                    changed = self.json_file("bad-profiles.json", manifest)
                    argv = self.argv()
                    argv[1], argv[3] = changed["path"], changed["sha256"]
                    restore = None
                else:
                    path = Path(original["path"])
                    restore = path.read_bytes()
                    path.write_bytes(restore + b"tampered")
                    argv = self.argv()
                argv += ["--fallback-binary", self.fallback_binary["path"], "--fallback-config", self.fallback_config["path"]]
                try:
                    code, receipt, _ = self.invoke(argv)
                    self.assertEqual(code, 1)
                    self.assertEqual(receipt["status"], "FAILED_NO_RETRY")
                    self.assertFalse(receipt["native_started"])
                finally:
                    if restore is not None:
                        path.write_bytes(restore)

    def native(self, command, **options):
        self.last_command, self.last_options = command, options
        if "stdout" in options:
            stdout = ("nnbench(eval): model=" + self.alias["path"] + " device=1 iterations=2 warmup=1\n"
                      "[cuda-quant] inference_profile=" + self.profile + " validation=unverified\n"
                      " 3 | 2.909 | 1000.0 | 64 | 22 | 2.91 (workers 32, physicalMax 3)\n")
            options["stdout"].write(stdout.encode())
        process = FakeProcess()
        self.last_process = process
        return process

    def test_eval_real_parser_counts_profile_and_native_exit(self):
        code, receipt, _ = self.invoke(self.argv("--mode", "eval", "--iterations", "2", "--warmup", "1"), popen=self.native)
        self.assertEqual(code, 0)
        self.assertEqual(receipt["actual_exit"], 0)
        self.assertEqual(receipt["metrics"]["nn_rows"], 64)
        self.assertEqual(receipt["metrics"]["nn_batches"], 22)
        self.assertFalse(receipt["metrics"]["all_logical_batches_full"])
        self.assertEqual(receipt["metrics"]["single_request_latency_percentiles"], "NOT_MEASURED")
        self.assertIn(self.alias["path"], self.last_command)
        self.assertEqual(self.last_options["env"]["CUDA_VISIBLE_DEVICES"], self.gpu["uuid"])
        self.assertEqual(self.last_options["stdin"], subprocess.DEVNULL)

    def test_wrong_runtime_profile_fails_after_real_parser(self):
        def wrong(command, **options):
            process = self.native(command, **options)
            options["stdout"].seek(0)
            content = ("nnbench(eval): model=" + self.alias["path"] + " device=1 iterations=2 warmup=1\n"
                       "[cuda-quant] inference_profile=rustgo-quant-v1:" + "b" * 64 + " validation=unverified\n"
                       " 3 | 2.909 | 1000.0 | 64 | 22 | 2.91 (workers 32, physicalMax 3)\n")
            options["stdout"].write(content.encode())
            options["stdout"].truncate()
            return process
        code, receipt, _ = self.invoke(self.argv("--mode", "eval", "--iterations", "2", "--warmup", "1"), popen=wrong)
        self.assertEqual(code, 1)
        self.assertEqual(receipt["actual_exit"], 0)
        self.assertEqual(receipt["status"], "FAILED_NO_RETRY")
        self.assertIn("runtime profile", receipt["error"])

    def test_timeout_has_cleanup_exit_without_normal_exit(self):
        def timeout(command, **options):
            process = self.native(command, **options)
            process.timeout = True
            return process
        code, receipt, _ = self.invoke(self.argv("--mode", "eval", "--iterations", "2", "--warmup", "1"), popen=timeout)
        self.assertEqual(code, 1)
        self.assertIsNone(receipt["actual_exit"])
        self.assertEqual(receipt["cleanup_exit"], 9)
        self.assertTrue(self.last_process.killed)
        self.assertEqual(receipt["cleanup_errors"], [])

    def test_gtp_inherits_stdio_and_does_not_claim_search_speed(self):
        code, receipt, _ = self.invoke(self.argv("--mode", "gtp"), popen=self.native)
        self.assertEqual(code, 0)
        self.assertEqual(self.last_command[1], "gtp")
        self.assertFalse({"stdin", "stdout", "stderr"} & self.last_options.keys())
        self.assertIsNone(receipt["timeout_seconds"])
        self.assertEqual(receipt["gtp_search_speed"], "NOT_MEASURED")


class FakeProcess:
    pid = 12345

    def __init__(self):
        self.code = None
        self.timeout = False
        self.killed = False

    def wait(self, timeout=None):
        if self.timeout and not self.killed:
            raise subprocess.TimeoutExpired("inert fixture", timeout)
        self.code = 9 if self.killed else 0
        return self.code

    def poll(self):
        return self.code

    def kill(self):
        self.killed = True


if __name__ == "__main__":
    unittest.main()
