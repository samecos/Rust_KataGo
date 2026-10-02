"""CPU-only small-file tests; GPU query and native Worker are mocked."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import launch_quant_worker as worker


class WorkerLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.common = worker.common_module()
        self.runner_source = self.common.source(Path(__file__).with_name("benchmark_quant_speed.py"))
        self.runner = worker.bound_module(self.common.Inventory(), self.runner_source)
        self.gpu = dict(index=0, name="Fixture GPU", uuid="GPU-fixture", compute_capability="12.0", driver="610.88")
        self.model = self.put("original-model.bin", b"same model")
        self.alias = self.put("relocated-model.bin", b"same model")
        self.old_binary = self.put("old-native.exe", b"inert old executable")
        self.binary = self.put("new-native.exe", b"inert rebuilt executable")
        self.recipe = self.put("recipe.json", b'{"fixture":true}')
        self.fallback_config = self.put("fallback.cfg", b"caller config unchanged")
        self.environment = self.put_json("environment.json", {"values": {
            "PATH": "fixture", "SECRET_SENTINEL": "NO_ENV_VALUE_IN_RECEIPT",
            "Katago_OLD": "discard", "CUDA_VISIBLE_DEVICES": "discard"}})
        self.handoff = self.put_json("old-handoff.json", dict(schema="rustgo-speed-profile-handoff-v1",
            workers=32, device="Fixture GPU", compute_capability="12.0", driver="610.88"))
        self.build = self.put_json("build.json", dict(schema="rustgo-typed-worker-native-build-result-v1",
            status="BUILT_NOT_GPU_QUALIFIED", actual_exit_code=0, sources_unchanged=True,
            binary=self.binary, tool_actual_exit_code=0, root_terminal_before_deadline=True))
        self.variants, self.runs, self.requests = [], [], []
        self.profiles = {3: "rustgo-quant-v1:" + "a" * 64, 8: "rustgo-quant-v1:" + "b" * 64}
        self.configs = {}
        telemetry = dict(returncode=0, query=self.common.OLD_GPU_QUERY,
                         output="Fixture GPU, GPU-fixture, 610.88, 40, 1000, 1000, 100, 0, 1")
        for cap in (3, 8):
            group = "b15-b" + str(cap)
            config = self.put(group + ".cfg", ("nnBackend = cudaquantbackend\r\nnnMaxBatchSize = " + str(cap)
                         + "\r\ncudaQuantPlan = " + self.recipe["path"] + "\r\n").encode())
            self.configs[cap] = config
            baseline = dict(id=group + "-baseline", group=group, role="baseline", model=self.model,
                            batch_cap=cap, binary=self.old_binary, config=self.fallback_config,
                            backend="cudaint8backend", environment={})
            candidate = dict(id=group + "-candidate-rebuilt", group=group, role="candidate", model=self.model,
                             batch_cap=cap, binary=self.binary, config=config, recipe=self.recipe,
                             backend="cudaquantbackend", environment={k: "0" for k in self.common.ENV_KEYS},
                             previous_measured_candidate={"binary": self.old_binary})
            self.variants.extend((baseline, candidate))
            for stage, role in zip(("A1", "B1", "B2", "A2"), ("baseline", "candidate", "candidate", "baseline")):
                variant = baseline if role == "baseline" else candidate
                index = len(self.runs) + 1
                requested = dict(variant=variant["id"], comparison=group, flavor=role, stage=stage,
                                 iterations=3, warmup=1)
                self.requests.append(requested)
                rate = 1000 if role == "baseline" else 1030
                batches = 96 // cap
                cost = (96 / rate) * 1000 / batches
                stdout = ("nnbench(eval): model=" + self.model["path"] + " device=1 iterations=3 warmup=1\n")
                if role == "candidate":
                    stdout += "[cuda-quant] inference_profile=" + self.profiles[cap] + " validation=unverified\n"
                stdout += f" {cap} | {cost:.3f} | {rate:.1f} | 96 | {batches} | {float(cap):.2f} (workers 32, physicalMax {cap})\n"
                self.put(f"{index:03d}-{variant['id']}.stdout.log", stdout.encode())
                metrics = self.runner.parse_eval(stdout, variant, 3, 1, 32)
                command = [variant["binary"]["path"], "nnbench", "--mode", "eval", "--model", self.model["path"],
                           "--config", variant["config"]["path"], "--batch", str(cap), "--workers", "32",
                           "--iterations", "3", "--warmup", "1"]
                self.runs.append(dict(index=index, variant=variant["id"], comparison=group, flavor=role,
                                      stage=stage, status="COMPLETE_SPEED_SAMPLE", actual_exit=0,
                                      cleanup_errors=[], command=command, metrics=metrics,
                                      before=telemetry, after=telemetry))
        self.plan = dict(schema="rustgo-quant-speed-plan-v1", workers=32, accuracy=self.common.ACCURACY,
                         old_speed_certificate_applies_to_new_binary=False, maximum_model_starts=8,
                         inherited_native_starts=84, cumulative_native_limit=96, runner=self.runner_source,
                         variants=self.variants, binary=self.binary, build_result=self.build,
                         handoff=self.handoff, environment=self.environment,
                         source_nnbench=self.put("nnbench.rs", b"inert source fixture"))
        self.plan_source = self.put_json("plan.json", self.plan)
        self.schedule = self.put_json("schedule.json", dict(schema="rustgo-speed-abba-schedule-v1",
                                    handoff=self.handoff, runs=self.requests))
        self.report = dict(schema="rustgo-quant-speed-report-v1", status="COMPLETE_SPEED_ONLY",
                           plan=self.plan_source, schedule=self.schedule, accuracy=self.common.ACCURACY,
                           runs=self.runs, comparisons={group: self.runner.summarize([r for r in self.runs if r["comparison"] == group])
                           for group in ("b15-b3", "b15-b8")})
        self.report_source = self.put_json("report.json", self.report)
        self.counter = 0

    def put(self, name, raw):
        path = self.root / name
        path.write_bytes(raw)
        return self.common.source(path)

    def put_json(self, name, value):
        return self.put(name, json.dumps(value).encode())

    def argv(self, *extra):
        self.counter += 1
        return ["--report", self.report_source["path"], "--report-sha256", self.report_source["sha256"],
                "--plan", self.plan_source["path"], "--plan-sha256", self.plan_source["sha256"],
                "--model", self.alias["path"], "--batch-cap", "3", "--server", "127.0.0.1:50051",
                "--worker-id", "fixture-worker", "--output", str(self.root / ("out" + str(self.counter))), *extra]

    def invoke(self, argv, gpu=None, popen=None):
        out, err = io.StringIO(), io.StringIO()
        with patch.object(worker, "query_gpu", return_value=self.gpu if gpu is None else gpu), \
                patch.object(worker.subprocess, "Popen", side_effect=popen or AssertionError("no native start allowed")), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = worker.main(argv)
        directory = Path(argv[argv.index("--output") + 1])
        receipt = json.loads((directory / "receipt.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text()) if (directory / "manifest.json").exists() else None
        self.assertEqual(out.getvalue(), "")
        self.assertNotIn("NO_ENV_VALUE_IN_RECEIPT", out.getvalue() + err.getvalue() + json.dumps(receipt) + json.dumps(manifest))
        return code, receipt, manifest, directory

    def test_prepare_same_content_other_path_locked_config_clean_env_and_no_old_confirmation(self):
        before = {p: p.read_bytes() for p in self.root.iterdir() if p.is_file()}
        argv = self.argv()
        with patch.object(worker, "query_gpu", return_value=self.gpu):
            selected, environment, _ = worker.select(worker.parser().parse_args(argv), self.common, self.common.Inventory())
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "GPU-fixture")
        self.assertEqual({k for k in environment if k.upper().startswith("KATAGO_")}, self.common.ENV_KEYS)
        code, receipt, manifest, directory = self.invoke(argv)
        self.assertEqual(code, 0)
        self.assertEqual(receipt["status"], "PREPARED")
        self.assertFalse(receipt["native_started"])
        self.assertEqual(manifest["selection"]["model"], self.alias)
        self.assertEqual(manifest["selection"]["expected_profile"], self.profiles[3])
        self.assertFalse(manifest["selection"]["independent_confirmation_for_this_binary"])
        self.assertFalse(manifest["selection"]["old_confirmation_applies"])
        self.assertEqual(manifest["command"][-1], "--once")
        self.assertEqual(manifest["command"][1], "nnworker")
        self.assertIn(self.alias["sha256"], manifest["command"])
        original = Path(self.configs[3]["path"]).read_bytes()
        self.assertEqual((directory / "selected.cfg").read_bytes(), original +
                         ("cudaQuantExpectedProfile = " + self.profiles[3] + "\r\n").encode())
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_gpu_cap_capacity_mismatch_no_match_or_explicit_fallback(self):
        cases = [({"uuid": "GPU-other"}, []), ({"driver": "999"}, []),
                 ({"compute_capability": "9.0"}, []), ({"name": "Other GPU"}, []),
                 ({}, ["--batch-cap", "16"]), ({}, ["--capacity", "16"])]
        for change, extra in cases:
            with self.subTest(change=change, extra=extra):
                code, receipt, manifest, _ = self.invoke(self.argv(*extra), gpu=dict(self.gpu, **change))
                self.assertEqual(code, 2)
                self.assertEqual(receipt["status"], "NO_MATCH")
                self.assertIsNone(manifest)
                argv = self.argv(*extra, "--fallback-binary", self.old_binary["path"],
                                 "--fallback-config", self.fallback_config["path"])
                code, receipt, manifest, directory = self.invoke(argv, gpu=dict(self.gpu, **change))
                self.assertEqual(code, 0)
                self.assertEqual(manifest["effective_config"], self.fallback_config)
                self.assertFalse(manifest["selection"]["measured_speed_applicable"])
                self.assertFalse((directory / "selected.cfg").exists())

    def test_incomplete_report_and_mixed_profile_refuse_even_with_fallback(self):
        original = copy.deepcopy(self.report)
        for problem in ("RUNNING", "mixed-profile"):
            with self.subTest(problem=problem):
                altered = copy.deepcopy(original)
                if problem == "RUNNING":
                    altered["status"] = "RUNNING"
                else:
                    altered["runs"][2]["metrics"]["inference_profile"] = self.profiles[8]
                self.report_source = self.put_json("report.json", altered)
                code, receipt, manifest, _ = self.invoke(self.argv("--fallback-binary", self.old_binary["path"],
                                                       "--fallback-config", self.fallback_config["path"]))
                self.assertEqual(code, 1)
                self.assertEqual(receipt["status"], "FAILED_NO_RETRY")
                self.assertIsNone(manifest)

    def test_matched_source_tampering_never_falls_back(self):
        for record in (self.recipe, self.configs[3], self.binary):
            with self.subTest(source=record["path"]):
                path = Path(record["path"])
                raw = path.read_bytes()
                path.write_bytes(raw + b"corrupt")
                try:
                    code, receipt, _, _ = self.invoke(self.argv("--fallback-binary", self.old_binary["path"],
                                                  "--fallback-config", self.fallback_config["path"]))
                    self.assertEqual(code, 1)
                    self.assertEqual(receipt["status"], "FAILED_NO_RETRY")
                    self.assertFalse(receipt["native_started"])
                finally:
                    path.write_bytes(raw)

    def test_report_counters_must_match_log_and_no_speed_pass_means_no_match(self):
        original = copy.deepcopy(self.report)
        self.report["runs"][0]["metrics"]["nn_rows"] += 1
        self.report_source = self.put_json("report.json", self.report)
        code, receipt, _, _ = self.invoke(self.argv())
        self.assertEqual(code, 1)
        self.assertIn("actual log", receipt["error"])
        # Recompute a valid losing comparison, rather than flipping a trusted flag.
        self.report = original
        for row in self.report["runs"]:
            if row["comparison"] == "b15-b3" and row["flavor"] == "candidate":
                row["metrics"]["nn_evals_per_second"] = 999.0
        self.report["comparisons"]["b15-b3"] = self.runner.summarize(self.report["runs"][:4])
        self.report_source = self.put_json("report.json", self.report)
        code, receipt, _, _ = self.invoke(self.argv())
        self.assertEqual(code, 2)
        self.assertEqual(receipt["status"], "NO_MATCH")

    def test_owned_normal_exit_and_typed_ack_failure_no_retry(self):
        for native_exit in (0, 1):
            with self.subTest(native_exit=native_exit):
                calls = []
                def native(command, **options):
                    calls.append((command, options))
                    self.assertEqual(options["env"]["CUDA_VISIBLE_DEVICES"], "GPU-fixture")
                    options["stderr"].write(b"inert typed ACK boundary\n")
                    return FakeProcess(native_exit)
                code, receipt, manifest, _ = self.invoke(self.argv("--mode", "run"), popen=native)
                self.assertEqual(code, native_exit)
                self.assertEqual(receipt["actual_exit"], native_exit)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][0], manifest["command"])
                self.assertFalse(receipt["typed_ack_observed_by_launcher"])
                self.assertEqual(len(receipt["logs"]), 2)

    def test_owned_timeout_records_cleanup_separately(self):
        child = FakeProcess(0, timeout=True)
        code, receipt, _, _ = self.invoke(self.argv("--mode", "run"), popen=lambda *a, **k: child)
        self.assertEqual(code, 1)
        self.assertIsNone(receipt["actual_exit"])
        self.assertEqual(receipt["cleanup_exit"], 9)
        self.assertTrue(child.killed)
        self.assertEqual(receipt["cleanup_errors"], [])


class FakeProcess:
    pid = 12345
    def __init__(self, exit_code, timeout=False):
        self.exit_code = exit_code
        self.timeout = timeout
        self.killed = False
        self.code = None
    def wait(self, timeout=None):
        if self.timeout and not self.killed:
            raise subprocess.TimeoutExpired("inert Worker", timeout)
        self.code = 9 if self.killed else self.exit_code
        return self.code
    def poll(self):
        return self.code
    def kill(self):
        self.killed = True


if __name__ == "__main__":
    unittest.main()
