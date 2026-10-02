"""Hermetic catalog driver tests: genuine CPU evidence validation, fake helpers."""
from copy import deepcopy
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import quantization_execution_catalog as catalog
import quantization_selection_ledger as ledger
import run_quantization_search as driver
import test_quantization_execution_ledger as fixtures


class FakeTools:
    def __init__(self, test, *, fail=None, interrupt=None, numeric_fail=False, tamper=False):
        self.test, self.fixture = test, test.fixture
        self.fail, self.interrupt, self.numeric_fail, self.tamper = fail, interrupt, numeric_fail, tamper
        self.calls = []

    def __call__(self, command, environment, stdout, stderr, timeout):
        test, fixture = self.test, self.fixture
        events = sorted((test.prepared.parent / "events").glob("*.json"))
        started = ledger.read_json(events[-1])
        test.assertEqual(started["event_type"], "ATTEMPT_STARTED")
        test.assertEqual(started["payload"]["command"], command)
        contract, registered, state = ledger.execution_replay(fixture.ledger, started["payload"]["registry_head_sha256"])
        reservation = state["reservations"][command["attempt_key"]]
        test.assertEqual(reservation["reservation_id"], started["payload"]["reservation_id"])
        test.assertEqual(registered[-1]["event_type"], "ATTEMPT_RESERVED")
        test.assertTrue((fixture.ledger / ".execution-driver.lock").is_file())
        test.assertFalse(any(k.upper().startswith("KATAGO_") for k in environment))
        self.calls.append(command)
        if command["phase"] == self.interrupt:
            stdout.write(b"synthetic contained helper interruption\n")
            raise KeyboardInterrupt("synthetic helper interrupted")
        if command["phase"] == self.fail:
            stderr.write(b"synthetic contained helper failed\n")
            return 2
        if command["phase"] == "collect":
            row = reservation["registration"]
            spec = next(s for s in fixture.cat["execution_specs"] if s["execution_spec_id"] == row["candidate_execution_spec_id"])
            execution = spec["semantic"]["execution"]
            source = (fixture.fixture.collections[execution["recipe_sha256"]] if spec["kind"] == "unified_v1_recipe"
                      else test.legacy_collections[execution["backend"]])
            destination = Path(command["output_directory"])
            shutil.copytree(source, destination)
            report = ledger.read_json(destination / "report.json")
            for name in ("worker.cfg", "worker.log"):
                path = destination / name
                path.write_text(path.read_text('utf-8').replace(str(source), str(destination)), encoding='utf-8')
            report.update(config_text=(destination / "worker.cfg").read_text('utf-8'),
                          config_sha256=ledger.planner.file_digest(destination / "worker.cfg"),
                          command=fixture.command(destination, report))
            for runtime in report.get("runtime_artifacts", []):
                for key in ("path", "effective_path"):
                    if key in runtime: runtime[key] = runtime[key].replace(str(source), str(destination))
            fixture.save_report(destination, report)
            if self.numeric_fail and spec["kind"] == "unified_v1_recipe" and execution["recipe_sha256"] == fixture.fixture.int8:
                fixture.change_win(destination, 0.7)
            if self.tamper:
                (destination / "results/000001.pb").write_bytes(b"invalid synthetic evidence")
        else:
            test.assertEqual(command["phase"], "benchmark")
            # The ledger fixture constructs raw per-arm timings/probes against
            # the currently admitted collections, never a GPU implementation.
            source = fixture.make_abba(reservation["registration"], state=state)
            shutil.copytree(source, command["output_directory"])
        return 0


class ExecutionDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ExecutionLedgerTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        fixtures.ExecutionLedgerTests.tearDownClass()

    def setUp(self):
        self.fixture = fixtures.ExecutionLedgerTests("runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.prepared = self.root / "driver/prepared-execution.json"
        self.legacy_collections = {backend:self.fixture.make_legacy(backend) for backend in ("cudabackend", "cudaint8backend")}
        self.fixture.initialize(collection_budget=4, abba_budget=5)

    def prepare(self, name="driver"):
        state = ledger.execution_inspect(self.fixture.ledger)
        self.prepared = self.root / name / "prepared-execution.json"
        result = driver.prepare_execution(self.fixture.catalog_path, self.fixture.sha, self.fixture.ledger,
            ledger.planner.file_digest(self.fixture.ledger / "contract.json"), state["head_sha256"], self.prepared.parent)
        self.prepared_sha = result["prepared"]["sha256"]
        return result

    def run_driver(self, fake=None):
        return driver.run_execution(self.prepared, self.prepared_sha, runner=fake or FakeTools(self))

    def admit_all(self):
        for row in self.fixture.cat["numeric_pairs"]:
            spec=next(s for s in self.fixture.cat["execution_specs"] if s["execution_spec_id"]==row["candidate_execution_spec_id"])
            execution=spec["semantic"]["execution"]
            directory=(self.fixture.fixture.collections[execution["recipe_sha256"]] if spec["kind"]=="unified_v1_recipe"
                       else self.legacy_collections[execution["backend"]])
            self.fixture.qualify(directory,row)

    def test_cpu_prepare_freezes_original_controls_and_does_not_reserve(self):
        before=ledger.execution_inspect(self.fixture.ledger)
        with patch.dict(os.environ,{"KATAGO_CUDA_INT8_GEMM_TUNE":"1","PYTHONPATH":"untrusted"}):
            result=self.prepare()
        prepared=driver.validate_execution_prepared(self.prepared,self.prepared_sha)
        self.assertEqual(before,ledger.execution_inspect(self.fixture.ledger))
        self.assertEqual(result["needed_attempts"],dict(COLLECTION=4,ABBA=5))
        self.assertEqual(len(prepared["commands"]),9)
        self.assertFalse(any(k.startswith("KATAGO_") or k=="PYTHONPATH" for k in prepared["environment"]))
        collect=[c for c in prepared["commands"] if c["phase"]=="collect"]
        self.assertEqual(collect[0]["argv"][collect[0]["argv"].index("--backend")+1],"cudaquantbackend")
        for command in collect:
            argv=command["argv"]
            self.assertEqual(argv[argv.index("--window-mode")+1],"continuous")
            if "--execution-spec" in argv:
                spec=ledger.read_json(argv[argv.index("--execution-spec")+1])
                self.assertEqual(argv[argv.index("--binary")+1],spec["binary"]["path"])
                self.assertEqual(argv[argv.index("--batch")+1],str(spec["batch"]))
                self.assertEqual(argv[argv.index("--config")+1],spec["config"]["path"])
        self.assertFalse((self.prepared.parent/"events").exists())

    def test_requires_existing_ledger_and_exact_anchor(self):
        with self.assertRaises(ValueError):
            driver.prepare_execution(self.fixture.catalog_path,self.fixture.sha,self.root/"absent","a"*64,"b"*64,self.root/"bad")
        self.assertFalse((self.root/"absent").exists())
        state=ledger.execution_inspect(self.fixture.ledger)
        with self.assertRaisesRegex(ValueError,"contract differs"):
            driver.prepare_execution(self.fixture.catalog_path,self.fixture.sha,self.fixture.ledger,"a"*64,state["head_sha256"],self.root/"bad")

    def test_insufficient_whole_pair_budget_rejects_prepare(self):
        registry=self.root/"small-ledger"
        state=ledger.execution_initialize(registry,4,4)
        with self.assertRaisesRegex(ValueError,"complete remaining"):
            driver.prepare_execution(self.fixture.catalog_path,self.fixture.sha,registry,ledger.planner.file_digest(registry/"contract.json"),state["head_sha256"],self.root/"small")
        self.assertFalse((self.root/"small").exists())

    def test_driver_output_cannot_be_inside_external_ledger(self):
        state=ledger.execution_inspect(self.fixture.ledger)
        with self.assertRaisesRegex(ValueError,"outside the durable ledger"):
            driver.prepare_execution(self.fixture.catalog_path,self.fixture.sha,self.fixture.ledger,
                ledger.planner.file_digest(self.fixture.ledger/"contract.json"),state["head_sha256"],self.fixture.ledger/"driver")
        self.assertFalse((self.fixture.ledger/"driver").exists())

    def test_source_or_command_tampering_rejected_before_attempt(self):
        self.prepare()
        command=next((self.prepared.parent/"commands").iterdir())
        command.write_bytes(command.read_bytes()+b" ")
        fake=FakeTools(self)
        with self.assertRaisesRegex(ValueError,"immutable source changed"):
            self.run_driver(fake)
        self.assertFalse(fake.calls)
        self.assertEqual(ledger.execution_inspect(self.fixture.ledger)["consumed_attempts"],dict(COLLECTION=0,ABBA=0))

    def test_moved_head_rejects_before_execution(self):
        self.prepare()
        self.fixture.reserve()
        fake=FakeTools(self)
        with self.assertRaisesRegex(ValueError,"head differs"):
            self.run_driver(fake)
        self.assertFalse(fake.calls)

    def test_pending_or_failed_attempt_cannot_be_reprepared(self):
        receipt=self.fixture.reserve()
        with self.assertRaisesRegex(ValueError,"pending"):
            self.prepare()
        ledger.execution_fail(self.fixture.ledger,receipt["reservation_id"],"synthetic interrupted old attempt")
        with self.assertRaisesRegex(ValueError,"already failed"):
            self.prepare()

    def test_reference_tool_failure_consumes_one_and_stops(self):
        self.prepare();fake=FakeTools(self,fail="collect")
        with self.assertRaisesRegex(ValueError,"exit status 2"):
            self.run_driver(fake)
        state=ledger.execution_inspect(self.fixture.ledger)
        self.assertEqual(len(fake.calls),1)
        self.assertEqual(state["consumed_attempts"],dict(COLLECTION=1,ABBA=0))
        self.assertEqual(len(state["failed_attempts"]),1)
        self.assertFalse(state["pending_reservations"])
        self.assertTrue((self.fixture.ledger/".execution-driver.lock").exists())
        with self.assertRaisesRegex(ValueError,"lock exists"):
            self.prepare("new-driver")

    def test_interruption_keeps_logs_and_failed_reservation(self):
        self.prepare();fake=FakeTools(self,interrupt="collect")
        with self.assertRaises(KeyboardInterrupt):self.run_driver(fake)
        result=ledger.read_json(self.prepared.parent/"result.json")
        self.assertEqual(result["status"],"FAILED_NO_RETRY")
        self.assertTrue(result["logs"])
        self.assertEqual(result["actual_new_attempts"],dict(COLLECTION=1,ABBA=0))
        self.assertEqual(len(ledger.execution_inspect(self.fixture.ledger)["failed_attempts"]),1)

    def test_invalid_raw_evidence_stops_after_first_collection(self):
        self.prepare();fake=FakeTools(self,tamper=True)
        with self.assertRaisesRegex(ValueError,"evidence rejected"):self.run_driver(fake)
        self.assertEqual(len(fake.calls),1)
        state=ledger.execution_inspect(self.fixture.ledger)
        self.assertEqual(len(state["failed_attempts"]),1)
        self.assertFalse(state["pending_reservations"])

    def test_existing_collections_are_reused_without_recollection(self):
        self.admit_all()
        prepared=self.prepare()
        self.assertEqual(prepared["needed_attempts"],dict(COLLECTION=0,ABBA=5))
        fake=FakeTools(self,fail="benchmark")
        with self.assertRaisesRegex(ValueError,"exit status 2"):self.run_driver(fake)
        self.assertEqual([c["phase"] for c in fake.calls],["benchmark"])
        self.assertEqual(ledger.execution_inspect(self.fixture.ledger)["consumed_attempts"],dict(COLLECTION=4,ABBA=1))

    def test_frozen_reused_evidence_tamper_blocks_before_new_attempt(self):
        self.admit_all();self.prepare()
        (self.fixture.fixture.reference/"results/000001.pb").write_bytes(b"changed")
        fake=FakeTools(self)
        with self.assertRaises(ValueError):self.run_driver(fake)
        self.assertFalse(fake.calls)

    def test_complete_catalog_executes_all_pairs_without_adoption(self):
        self.prepare();fake=FakeTools(self)
        result=self.run_driver(fake)
        self.assertEqual([c["phase"] for c in fake.calls],["collect"]*4+["benchmark"]*5)
        self.assertEqual(result["actual_new_attempts"],dict(COLLECTION=4,ABBA=5))
        self.assertFalse(result["publish_allowed"])
        self.assertFalse((self.fixture.ledger/".execution-driver.lock").exists())
        self.assertTrue((self.prepared.parent/".run.lock").exists())
        self.assertEqual(len(ledger.execution_inspect(self.fixture.ledger)["performance"]),5)
        second=self.prepare("second-driver")
        self.assertEqual(second["needed_attempts"],dict(COLLECTION=0,ABBA=0))
        second_fake=FakeTools(self)
        rerun=self.run_driver(second_fake)
        self.assertFalse(second_fake.calls)
        self.assertEqual(rerun["actual_new_attempts"],dict(COLLECTION=0,ABBA=0))

    def test_numeric_failure_skips_all_dependent_pairs_only(self):
        self.prepare();fake=FakeTools(self,numeric_fail=True)
        result=self.run_driver(fake)
        self.assertEqual(result["actual_new_attempts"],dict(COLLECTION=4,ABBA=2))
        self.assertEqual(sum(v=="SKIPPED_NUMERIC_FAIL" for v in result["pair_outcomes"].values()),3)
        self.assertEqual(sum(v=="FAIL" for v in result["numeric_gates"].values()),1)

    def test_observed_qualification_mismatch_rejected_before_helper(self):
        self.admit_all();self.prepare();fake=FakeTools(self)
        original=ledger.execution_reserve
        def altered(*args,**kwargs):
            result=original(*args,**kwargs)
            result=deepcopy(result)
            result["reservation"]["numeric_qualification"][0]["observed_execution_id"]="incorrect"
            return result
        with patch.object(ledger,"execution_reserve",side_effect=altered):
            with self.assertRaisesRegex(ValueError,"observed numerical identity changed"):self.run_driver(fake)
        self.assertFalse(fake.calls)
        self.assertEqual(ledger.execution_inspect(self.fixture.ledger)["consumed_attempts"],dict(COLLECTION=4,ABBA=1))


if __name__ == "__main__":
    unittest.main()
