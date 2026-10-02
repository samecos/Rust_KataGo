"""CPU-only driver tests: synthetic protobufs and fake tool execution, no GPU."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import subprocess
import time
import unittest
from unittest.mock import patch

import collect_quantization_corpus as collection
from compare_worker_outputs import message_dict
import quantization_selection_ledger as ledger
import run_quantization_search as driver
import test_quantization_selection_ledger as fixtures


class FakeTools:
    def __init__(self, test, *, fail_phase=None, interrupt_phase=None, numeric_fail=False):
        self.test, self.fixture = test, test.fixture
        self.sources = dict(self.fixture.collections)
        self.fail_phase, self.interrupt_phase, self.numeric_fail = fail_phase, interrupt_phase, numeric_fail
        self.calls = []

    def collect(self, command):
        recipe = command["recipe_sha256"]
        directory = Path(command["output_directory"])
        source = self.sources[recipe]
        shutil.copytree(source, directory)
        report = ledger.read_json(directory / "report.json")
        cfg = (directory / "worker.cfg").read_text(encoding="utf-8").replace(str(source / "recipe.json"), str(directory / "recipe.json"))
        (directory / "worker.cfg").write_text(cfg, encoding="utf-8")
        report.update(config_text=cfg, config_sha256=ledger.planner.file_digest(directory / "worker.cfg"))
        if self.numeric_fail and recipe == self.fixture.int8:
            path = directory / "results/000001.pb"
            result = self.fixture.protocol.pb.EvalResult.FromString(path.read_bytes())
            result.output.white_win_prob, result.output.white_loss_prob = 0.7, 0.3
            raw = result.SerializeToString(deterministic=True)
            path.write_bytes(raw)
            entry = next(collection.read_jsonl((directory / "outputs.jsonl").read_bytes(), "synthetic test"))
            entry.update(result=message_dict(result), result_sha256=ledger.digest(raw),
                         output_sha256=ledger.digest(result.output.SerializeToString(deterministic=True)))
            (directory / "outputs.jsonl").write_bytes(fixtures.lines([entry]))
            report["ordered_output_hashes_sha256"] = ledger.digest((1).to_bytes(8, "little") + bytes.fromhex(entry["output_sha256"]))
        raw = (directory / "results/000001.pb").read_bytes()
        summary = dict(schema="rustgo-continuous-corpus-buffer-v1", submitted=1, received_basic_identity_checked=1,
                       heartbeat_count=0, protobuf_payload_bytes=len(raw), max_pending=1, pending_task_ids=[],
                       artifacts=[dict(task_id=1, file="results/000001.pb", bytes=len(raw), sha256=ledger.digest(raw))],
                       validation="SYNTHETIC CPU TEST ONLY")
        fixtures.put(directory / "continuous-buffer.json", summary)
        report.update(window_mode="continuous", continuous_numerically_validated_results=1,
                      collection_buffer_contract=dict(schema="rustgo-continuous-corpus-buffer-contract-v1", max_results=1,
                                                      max_protobuf_payload_bytes=256 * 1024 * 1024),
                      continuous_buffer={k: v for k, v in summary.items() if k != "artifacts"})
        names = [item["file"] for item in report["artifacts"]] + ["continuous-buffer.json"]
        report["artifacts"] = [fixtures.artifact(directory / name) for name in names]
        fixtures.put(directory / "report.json", report)
        self.fixture.collections[recipe] = directory
        if recipe == self.fixture.fp16:
            self.fixture.reference = directory

    def __call__(self, command, environment, stdout, stderr, timeout):
        events = sorted((self.test.prepared.parent / "events").iterdir())
        latest = ledger.read_json(events[-1])
        self.test.assertEqual(latest["event_type"], "ATTEMPT_STARTED")
        self.test.assertEqual(latest["payload"]["command"], command)
        self.test.assertFalse(any(k.upper().startswith("KATAGO_") for k in environment))
        self.calls.append(command)
        if command["phase"] == self.interrupt_phase:
            stdout.write(b"synthetic interrupted process output\n")
            raise KeyboardInterrupt("synthetic interruption")
        if command["phase"] == self.fail_phase:
            stderr.write(b"synthetic tool error\n")
            return 2
        if command["phase"] == "collect":
            self.collect(command)
            return 0
        if command["phase"] == "benchmark":
            self.fixture.ledger = self.test.prepared.parent / "ledger"
            source = self.fixture.make_abba()
            shutil.copytree(source, command["output_directory"])
            return 0
        old_argv = sys.argv
        sys.argv = command["argv"][3:]
        text_out, text_error = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(text_out), redirect_stderr(text_error):
                try:
                    runpy.run_path(sys.argv[0], run_name="__main__")
                    returncode = 0
                except SystemExit as error:
                    returncode = error.code or 0
        finally:
            sys.argv = old_argv
            stdout.write(text_out.getvalue().encode("utf-8"))
            stderr.write(text_error.getvalue().encode("utf-8"))
        return returncode


class DriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.LedgerTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        fixtures.LedgerTests.tearDownClass()

    def setUp(self):
        self.fixture = fixtures.LedgerTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.prepared = self.root / "driver/prepared.json"

    def prepare(self):
        return driver.prepare(self.root / "search/search-plan.json", self.root / "workload.json", self.fixture.binary,
                              1, self.root / "sgfs", self.prepared.parent)

    def test_prepare_is_cpu_only_and_freezes_all_commands(self):
        with patch.object(driver.subprocess, "Popen", side_effect=AssertionError("prepare spawned a process")), \
             patch.dict(driver.os.environ, {"KATAGO_CUDA_INT8_GEMM_TUNE": "1", "PYTHONPATH": "untrusted"}):
            result = self.prepare()
        contract = driver.validate_prepared(self.prepared, result["prepared"]["sha256"])
        self.assertEqual(result["subprocesses_started"], 0)
        self.assertEqual((result["numeric_budget"], result["performance_budget"]), (2, 1))
        self.assertNotIn("KATAGO_CUDA_INT8_GEMM_TUNE", contract["environment"])
        self.assertNotIn("PYTHONPATH", contract["environment"])
        self.assertTrue(all(Path(step["argv"][3]).is_absolute() for step in contract["commands"]))
        self.assertTrue(all("continuous" in step["argv"] for step in contract["commands"] if step["phase"] == "collect"))
        self.assertFalse((self.prepared.parent / "events").exists())

    def test_complete_fake_execution_order_and_no_adoption(self):
        self.prepare()
        fake = FakeTools(self)
        result = driver.run(self.prepared, runner=fake)
        self.assertEqual([c["phase"] for c in fake.calls], ["collect", "compare", "ledger-init", "ingest-numeric",
                         "collect", "compare", "ingest-numeric", "benchmark", "ingest-performance", "summarize"])
        self.assertEqual(result["counts"], dict(NUMERIC=2, PERFORMANCE=1))
        self.assertEqual(result["baseline_optimality"], "NOT_ESTABLISHED")
        self.assertFalse(result["publish_allowed"])
        summary = ledger.read_json(result["summary"]["path"])
        self.assertEqual(summary["scope"], "SMOKE_DIAGNOSTIC_ONLY")
        self.assertEqual(summary["conclusion"], "CANDIDATE_REQUIRES_CONFIRMATION")
        with self.assertRaisesRegex(ValueError, "single-use"):
            driver.run(self.prepared, runner=fake)

    def test_valid_numeric_failure_is_registered_and_skips_performance(self):
        self.prepare()
        fake = FakeTools(self, numeric_fail=True)
        result = driver.run(self.prepared, runner=fake)
        self.assertEqual(result["counts"], dict(NUMERIC=2, PERFORMANCE=0))
        self.assertEqual(result["numeric_gates"][self.fixture.int8], "FAIL")
        self.assertNotIn("benchmark", [step["phase"] for step in fake.calls])
        state = ledger.inspect(self.prepared.parent / "ledger")
        self.assertEqual(state["results"]["NUMERIC"][self.fixture.int8]["gate"], "FAIL")
        self.assertEqual(ledger.read_json(result["summary"]["path"])["conclusion"], "BASELINE_RETAINED")

    def test_reference_collection_failure_stops_before_ledger_and_cannot_retry(self):
        self.prepare()
        fake = FakeTools(self, fail_phase="collect")
        with self.assertRaisesRegex(ValueError, "subprocess failed"):
            driver.run(self.prepared, runner=fake)
        self.assertEqual(len(fake.calls), 1)
        result = ledger.read_json(self.prepared.parent / "result.json")
        self.assertEqual(result["counts"], dict(NUMERIC=1, PERFORMANCE=0))
        self.assertTrue(result["logs"])
        self.assertFalse((self.prepared.parent / "ledger").exists())
        with self.assertRaisesRegex(ValueError, "single-use"):
            driver.run(self.prepared, runner=fake)

    def test_reference_comparison_execution_error_stops(self):
        self.prepare()
        fake = FakeTools(self, fail_phase="compare")
        with self.assertRaises(OSError):
            driver.run(self.prepared, runner=fake)
        self.assertEqual([step["phase"] for step in fake.calls], ["collect", "compare"])
        self.assertFalse((self.prepared.parent / "ledger").exists())

    def test_interruption_keeps_attempt_logs_lock_and_blocks_resume(self):
        self.prepare()
        fake = FakeTools(self, interrupt_phase="collect")
        with self.assertRaises(KeyboardInterrupt):
            driver.run(self.prepared, runner=fake)
        result = ledger.read_json(self.prepared.parent / "result.json")
        self.assertEqual(result["counts"]["NUMERIC"], 1)
        self.assertTrue((self.prepared.parent / ".run.lock").is_file())
        self.assertIn("synthetic interrupted", Path(result["logs"][0]["path"]).read_text())
        with self.assertRaisesRegex(ValueError, "single-use"):
            driver.run(self.prepared, runner=fake)

    def test_budget_deficit_rejected_before_prepare_artifacts(self):
        work = self.fixture.workload
        work["max_numeric_evaluations"] = 1
        fixtures.put(self.root / "workload.json", work)
        with self.assertRaisesRegex(ValueError, "budget"):
            self.prepare()
        self.assertFalse(self.prepared.parent.exists())

    def test_holdout_rejected_without_reading_outputs_or_starting_process(self):
        work = self.fixture.workload
        work["split"] = "holdout"
        fixtures.put(self.root / "workload.json", work)
        with patch.object(driver.subprocess, "Popen", side_effect=AssertionError("holdout launched")):
            with self.assertRaisesRegex(ValueError, "holdout"):
                self.prepare()

    def test_source_change_rejected_before_any_attempt(self):
        self.prepare()
        self.fixture.binary.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "immutable source"):
            driver.run(self.prepared, runner=lambda *args: self.fail("changed binary ran"))
        self.assertFalse((self.prepared.parent / "events").exists())

    def test_manifest_command_change_rejected_even_without_external_anchor(self):
        self.prepare()
        prepared = ledger.read_json(self.prepared)
        prepared["commands"][0]["argv"].append("--trace-batches")
        fixtures.put(self.prepared, prepared)
        with self.assertRaisesRegex(ValueError, "commands differ"):
            driver.run(self.prepared, runner=lambda *args: self.fail("modified command ran"))

    def test_lock_is_never_stolen_and_prepared_output_never_replaced(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "NEW directory"):
            self.prepare()
        (self.prepared.parent / ".run.lock").write_text("existing process")
        with self.assertRaises(FileExistsError):
            driver.run(self.prepared, runner=lambda *args: self.fail("lock stolen"))

    def test_contract_anchor_detects_change(self):
        result = self.prepare()
        prepared = ledger.read_json(self.prepared)
        prepared["environment"]["PATH"] = "changed"
        fixtures.put(self.prepared, prepared)
        with self.assertRaisesRegex(ValueError, "expected anchor"):
            driver.run(self.prepared, result["prepared"]["sha256"], runner=lambda *args: self.fail("anchor ignored"))

    @unittest.skipUnless(os.name == "nt", "native containment checks require Windows")
    def test_native_job_success_and_timeout_kill_only_owned_descendants(self):
        # Persist CPU-only intent before launching even the unrelated sentinel.
        fixtures.put(self.root / "containment-attempt.json", dict(kind="CPU_ONLY_SELF_CREATED_PYTHON_PROCESSES",
                     scenarios=["normal helper exit with sleeping descendant", "timeout"], gpu_executions=0))
        sentinel = subprocess.Popen([sys.executable, "-E", "-s", "-c", "import time; time.sleep(60)"],
                                    creationflags=subprocess.CREATE_NO_WINDOW)
        api = driver.ctypes.WinDLL("kernel32", use_last_error=True)
        api.OpenProcess.argtypes = [driver.wintypes.DWORD, driver.wintypes.BOOL, driver.wintypes.DWORD]
        api.OpenProcess.restype = driver.wintypes.HANDLE
        api.WaitForSingleObject.argtypes = [driver.wintypes.HANDLE, driver.wintypes.DWORD]
        api.CloseHandle.argtypes = [driver.wintypes.HANDLE]
        api.IsProcessInJob.argtypes = [driver.wintypes.HANDLE, driver.wintypes.HANDLE, driver.ctypes.POINTER(driver.wintypes.BOOL)]
        assigned = []
        observations = []
        original_assign = driver.WindowsJob.assign
        def checked_assign(job, process):
            original_assign(job, process)
            contained = driver.wintypes.BOOL()
            self.assertTrue(api.IsProcessInJob(driver.wintypes.HANDLE(int(process._handle)), job.handle, driver.ctypes.byref(contained)))
            self.assertTrue(contained.value, "helper was not attached before stdin release")
            assigned.append(process.pid)
        try:
            for wait in (False, True):
                pid_path = self.root / f"owned-pids-{wait}.json"
                script = self.root / f"owned-child-{wait}.py"
                script.write_text("import json, os, subprocess, sys, time\nfrom pathlib import Path\n"
                                  "child = subprocess.Popen([sys.executable, '-E', '-s', '-c', 'import time; time.sleep(60)'], "
                                  "creationflags=subprocess.CREATE_NO_WINDOW)\n"
                                  f"Path({str(pid_path)!r}).write_text(json.dumps([os.getpid(), child.pid]))\n"
                                  + ("time.sleep(60)\n" if wait else "print('owned-output', flush=True)\nraise SystemExit(7)\n"), encoding="utf-8")
                command = dict(argv=[sys.executable, "-E", "-s", str(script)],
                               launcher_argv=[sys.executable, "-E", "-s", str(Path(driver.__file__).resolve()), "_child"])
                fixtures.put(self.root / f"containment-command-{wait}.json", command)
                with patch.object(driver.WindowsJob, "assign", checked_assign), \
                     (self.root / f"native-{wait}.stdout.log").open("xb") as out, (self.root / f"native-{wait}.stderr.log").open("xb") as err:
                    if wait:
                        with self.assertRaises(subprocess.TimeoutExpired):
                            driver.execute(command, driver.environment(), out, err, 3)
                    else:
                        self.assertEqual(driver.execute(command, driver.environment(), out, err, 10), 7)
                if not wait:
                    self.assertIn("owned-output", (self.root / f"native-{wait}.stdout.log").read_text())
                for pid in ledger.read_json(pid_path):
                    handle = api.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE, own recorded PID only.
                    if handle:
                        try:
                            self.assertEqual(api.WaitForSingleObject(handle, 5000), 0, f"owned process {pid} survived job close")
                        finally:
                            api.CloseHandle(handle)
                    else:
                        self.assertEqual(driver.ctypes.get_last_error(), 87, "cannot establish owned process exit")
                self.assertIsNone(sentinel.poll(), "unrelated sentinel was terminated")
                observations.append(dict(timeout_case=wait, owned_pids=ledger.read_json(pid_path), all_owned_exited=True,
                                         unrelated_sentinel_alive=True, helper_attached_before_release=True))
            self.assertEqual(len(assigned), 2)
            evidence = ledger.ROOT / "target/unified-quant-driver-cpu-r1" / ("native-job-" + self.root.name + ".json")
            ledger.write_new(evidence, dict(status="PASS", platform=sys.platform, pointer_bits=8 * driver.ctypes.sizeof(driver.ctypes.c_void_p),
                                           observations=observations, normal_returncode=7, normal_output_preserved=True, gpu_executions=0))
        finally:
            sentinel.terminate()
            sentinel.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
