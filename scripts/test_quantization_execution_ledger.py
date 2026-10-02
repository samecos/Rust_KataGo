"""Hermetic catalog-v2 accounting and raw receipts; no GPU or process is run."""
from copy import deepcopy
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

import quantization_execution_catalog as catalog
import quantization_selection_ledger as ledger
import legacy_quantization_execution as legacy
import compare_quantization_corpus as comparison
import benchmark_quantization_corpus as benchmark
from benchmark_workers import distribution
from tune_runtime import decide
from compare_worker_outputs import message_dict
import test_quantization_execution_catalog as catalog_fixture
import test_quantization_selection_ledger as v1_fixture
import test_legacy_quantization_execution as legacy_fixture

put, artifact, lines = v1_fixture.put, v1_fixture.artifact, v1_fixture.lines


class ExecutionLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        v1_fixture.LedgerTests.setUpClass()
        legacy_fixture.LegacyExecutionTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        v1_fixture.LedgerTests.tearDownClass()

    def setUp(self):
        self.fixture = v1_fixture.LedgerTests("runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.protocol = self.fixture.root, self.fixture.protocol
        for module, name in ((subprocess, "Popen"), (subprocess, "run"), (os, "system")):
            guard = patch.object(module, name, side_effect=AssertionError("CPU test must not launch a process"))
            guard.start()
            self.addCleanup(guard.stop)
        self.controls = [catalog_fixture.CatalogTests.make_control(self.fixture, backend) for backend in ("cudabackend", "cudaint8backend")]
        self.fingerprint = deepcopy(legacy_fixture.LegacyExecutionTests.fingerprint_original)
        self.fingerprint["model_sha256"] = self.fixture.model_sha
        # Make the synthetic FP16 original plan match the synthetic probe.
        path = self.controls[0]
        spec = ledger.read_json(path)
        plan_path = Path(spec["plan"]["path"])
        plan = ledger.read_json(plan_path)
        plan["target"].update({key: self.fingerprint[key] for key in legacy.TARGET_KEYS})
        plan["backend_build"] = self.fingerprint["backend_build"]
        put(plan_path, plan)
        spec["plan"]["sha256"] = ledger.planner.file_digest(plan_path)
        put(path, spec)
        self.work = dict(self.fixture.workload, max_numeric_evaluations=4, max_performance_evaluations=5)
        self.work_path = self.root / "catalog-workload.json"
        put(self.work_path, self.work)
        self.catalog_path = self.make_catalog("catalog")
        self.cat = ledger.read_json(self.catalog_path)
        self.sha = ledger.planner.file_digest(self.catalog_path)
        self.ledger = self.root / "execution-ledger"
        self.reference_id = self.cat["identity"]["canonical_reference_execution_spec_id"]
        for directory in self.fixture.collections.values():
            self.prepare_unified(directory)

    def make_catalog(self, name, *, work=None, batches=None, reference_batch=1):
        work_path = self.work_path
        if work is not None:
            work_path = self.root / (name + "-workload.json")
            put(work_path, work)
        catalog.create_catalog(self.root / "search/search-plan.json", work_path, self.fixture.binary,
                               batches or [1], self.controls, self.root / name, reference_batch=reference_batch,
                               sgf_root=self.root / "sgfs")
        return self.root / name / "catalog.json"

    def save_report(self, directory, report):
        report["artifacts"] = [artifact(directory / row["file"]) for row in report["artifacts"]]
        put(directory / "report.json", report)

    def continuous(self, directory, report):
        raw = (directory / "results/000001.pb").read_bytes()
        summary = dict(schema="rustgo-continuous-corpus-buffer-v1", submitted=1, received_basic_identity_checked=1,
                       heartbeat_count=0, protobuf_payload_bytes=len(raw), max_pending=1, pending_task_ids=[],
                       artifacts=[dict(task_id=1, file="results/000001.pb", bytes=len(raw), sha256=ledger.digest(raw))],
                       validation="SYNTHETIC CPU FIXTURE; NEVER GPU EVIDENCE")
        put(directory / "continuous-buffer.json", summary)
        report.update(window_mode="continuous", continuous_numerically_validated_results=1,
                      collection_buffer_contract=dict(schema="rustgo-continuous-corpus-buffer-contract-v1", max_results=1,
                                                      max_protobuf_payload_bytes=256 * 1024 * 1024),
                      continuous_buffer={key: value for key, value in summary.items() if key != "artifacts"})
        if not any(row["file"] == "continuous-buffer.json" for row in report["artifacts"]):
            report["artifacts"].append(artifact(directory / "continuous-buffer.json"))

    def command(self, directory, report):
        return [report["binary"], "nnworker", "--server", "127.0.0.1:50001", "--worker-id", "quantization-corpus-local",
                "--capacity", str(report["capacity"]), "--once", "--model", report["model"], "--model-sha256", report["model_sha256"],
                "--config", str(directory / "worker.cfg")]

    def prepare_unified(self, directory, *, batch=1):
        report = ledger.read_json(directory / "report.json")
        values = catalog.fixed_unified_config(batch, report["actual_profile"]["recipe_sha256"])
        values["cudaQuantPlan"] = str(directory / "recipe.json")
        values["cudaQuantExpectedProfile"] = report["actual_profile"]["inference_profile_id"]
        cfg = "".join(f"{key}={value}\n" for key, value in values.items())
        (directory / "worker.cfg").write_text(cfg, encoding="utf-8")
        report.update(config_text=cfg, config_sha256=ledger.planner.file_digest(directory / "worker.cfg"), batch=batch)
        report["command"] = self.command(directory, report)
        self.continuous(directory, report)
        self.save_report(directory, report)
        return directory

    def make_legacy(self, backend):
        source_spec = next(path for path in self.controls if ledger.read_json(path)["backend"] == backend)
        loaded = legacy.load_spec(source_spec)
        directory = self.root / (backend + "-collection")
        shutil.copytree(self.fixture.reference, directory)
        report = ledger.read_json(directory / "report.json")
        frozen = legacy.freeze_spec(loaded, directory)
        source = Path(loaded["config"]).read_bytes()
        (directory / "source-worker.cfg").write_bytes(source)
        effective, runtimes = source, []
        if loaded["plan"]:
            path = directory / "runtime-cudaTacticPlan.json"
            path.write_bytes(Path(loaded["spec"]["plan"]["path"]).read_bytes())
            effective = source.replace(loaded["spec"]["plan"]["path"].encode(), str(path).encode())
            runtimes = [dict(config_key="cudaTacticPlan", file=path.name, sha256=ledger.planner.file_digest(path))]
            log = f"cudaTacticPlan '{path}' installed (plan id: {loaded['plan']['plan_id']})\n"
        else:
            log = legacy_fixture.LegacyExecutionTests.int8_log_original.replace("e" * 64, self.fixture.model_sha)
        (directory / "worker.cfg").write_bytes(effective)
        (directory / "worker.log").write_text(log, encoding="utf-8")
        (directory / "recipe.json").unlink()
        info = "fixture; backend=" + backend
        if backend == "cudaint8backend":
            info += "; precision=W8A8-mixed; int8-scope=ffn; int8-min-ffn-width=0; quantization=w8a8-row-out-rne-v1"
        report.update(backend=backend, batch=8, binary=loaded["binary"], binary_sha256=loaded["spec"]["binary"]["sha256"],
                      config_text=(directory / "worker.cfg").read_text(encoding="utf-8"), config_sha256=legacy.sha(effective),
                      actual_profile=None, recipe_file_sha256=None, runtime_artifacts=runtimes,
                      controlled_environment=loaded["environment"], legacy_execution=dict(spec=frozen))
        report["hello"]["backend_info"] = info
        report["command"] = self.command(directory, report)
        report["artifacts"] = [artifact(path) for path in directory.iterdir() if path.is_file() and path.name != "report.json"]
        runner = lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, legacy.compact(self.fingerprint), b"synthetic CPU probe")
        for label in ("before", "after"):
            report["legacy_execution"][label] = legacy.probe_fingerprint(loaded, directory, label,
                        environment=legacy.apply_environment(loaded, {}), runner=runner)
        self.save_report(directory, report)
        return directory

    def initialize(self, collection_budget=8, abba_budget=8):
        ledger.execution_initialize(self.ledger, collection_budget, abba_budget)
        return ledger.execution_register_catalog(self.ledger, self.catalog_path, self.sha)

    def row(self, spec_id=None, cat=None):
        return next(row for row in (cat or self.cat)["numeric_pairs"] if row["candidate_execution_spec_id"] == (spec_id or self.reference_id))

    def reserve(self, row=None, sha=None):
        row = row or self.row()
        return ledger.execution_reserve(self.ledger, sha or self.sha, "COLLECTION", row["collection_attempt_key"])["reservation"]

    def admit(self, directory, row=None):
        reservation = self.reserve(row)
        return ledger.execution_ingest_collection(self.ledger, reservation["reservation_id"], directory)

    def qualify(self, directory, row=None):
        row = row or self.row()
        self.admit(directory, row)
        return ledger.execution_compare(self.ledger, self.sha, row["comparison_key"])

    def unified_int8_row(self):
        return next(row for row in self.cat["numeric_pairs"]
                    if row["candidate_execution_spec_id"] != self.reference_id
                    and next(spec for spec in self.cat["execution_specs"] if spec["execution_spec_id"] == row["candidate_execution_spec_id"])["kind"] == "unified_v1_recipe")

    def qualify_all(self):
        for row in self.cat["numeric_pairs"]:
            spec = next(spec for spec in self.cat["execution_specs"] if spec["execution_spec_id"] == row["candidate_execution_spec_id"])
            execution = spec["semantic"]["execution"]
            directory = self.make_legacy(execution["backend"]) if spec["kind"] == "legacy_explicit_spec" else self.fixture.collections[execution["recipe_sha256"]]
            self.qualify(directory, row)

    def make_abba(self, pair, *, state=None, candidate_ns=9000000, candidate_latency_us=None, catalog_contract=None, base_start=None):
        """Real report/probe formats with synthetic bytes; never execute a tool."""
        state = state or ledger.execution_replay(self.ledger)[2]
        cat = catalog_contract or self.cat
        work = cat["workload"]
        self.abba_counter = getattr(self, "abba_counter", 0) + 1
        directory = self.root / f"abba-{self.abba_counter}"
        directory.mkdir()
        ref_key = catalog.collection_attempt_key(cat["numeric_workload_id"], pair["reference_execution_spec_id"])
        reference = state["loaded"][ref_key]
        _, measured, prepared_sha = benchmark.prepare_requests(self.protocol, reference, work["warmup"], work["cycles"], work["task_timeout"])
        count = len(measured)
        arm_data, observations = {}, {}
        for name in ("baseline", "candidate"):
            key = catalog.collection_attempt_key(cat["numeric_workload_id"], pair[name + "_execution_spec_id"])
            arm_data[name], observations[name] = state["loaded"][key], state["collections"][key]
        report = dict(schema="rustgo-quantization-corpus-abba-v1", status="SMOKE_ONLY" if work["smoke"] else "PAIRWISE_MEASURED_REQUIRES_SELECTION_REVIEW",
                timing_clock=legacy.performance_clock(), order=list(benchmark.ORDER), holdout_outputs_read=0,
                publish_allowed=False, production_certified=False, deployment_adopted=False, primary_metric=work["metric"],
                cycles=work["cycles"], warmup_requests=work["warmup"], smoke=work["smoke"], minimum_improvement=0.01,
                maximum_relative_spread=0.05, reference_report_sha256=reference["report_sha256"],
                prepared_request_sha256=prepared_sha, measured_requests=count, semantic_positions=len(reference["positions"]),
                numeric_collection_window_modes=dict(reference="continuous", baseline="continuous", candidate="continuous"),
                runs=[], numerical_checks={}, protocol_sha256=ledger.planner.file_digest(ledger.ROOT / "crates/kata_worker/proto/worker.proto"),
                source_scripts={name: ledger.planner.file_digest(ledger.ROOT / "scripts" / name) for name in ledger.SOURCE_NAMES
                                if name not in ("quantization_selection_ledger.py", "plan_quantization_search.py")})
        for name, data in arm_data.items():
            metrics = ledger.execution_compare_data(reference, data)
            metrics.pop("comparison_binding")
            report["numerical_checks"][name] = metrics
        scores = []
        for index, name in enumerate(benchmark.ORDER):
            data, observation = arm_data[name], observations[name]
            source = Path(observation["original_directory"])
            prior = data["report"]
            arm_dir = directory / f"{index + 1:02d}-{name}"
            arm_dir.mkdir()
            shutil.copyfile(source / "worker.log", arm_dir / "worker.log")
            ns = 10000000 if name == "baseline" else (candidate_ns[index - 1] if isinstance(candidate_ns, tuple) else candidate_ns)
            start = (base_start if base_start is not None else self.abba_counter * 10000000000) + index * 1000000000
            elapsed = ns / 1e9
            rows = [[ns / 1000 / count, 10, 0, 0, 10, None] for _ in range(count)]
            if name == "candidate" and candidate_latency_us is not None:
                for row in rows:
                    row[0] = candidate_latency_us
            columns = ["round_trip_us", "worker_elapsed_us", "queue_us", "context_us", "evaluator_us", "refill_delay_us"]
            measurement = dict(started_ns=start, ended_ns=start + ns, elapsed_seconds=elapsed, completed=count,
                    rpc_requests_per_second=count / elapsed, timings={key: distribution([row[col] for row in rows]) for col, key in enumerate(columns)},
                    nn_rows_per_second=count / elapsed, nn_batches_per_second=count / elapsed, mean_logical_rows_per_batch=1.0,
                    physical_batch_distribution="NOT_OBSERVED_IN_UNTRACED_TIMING")
            before = dict(in_flight=0, completed_requests=work["warmup"], failed_requests=0, nn_rows=work["warmup"], nn_batches=work["warmup"])
            after = dict(in_flight=0, completed_requests=work["warmup"] + count, failed_requests=0, nn_rows=work["warmup"] + count, nn_batches=work["warmup"] + count)
            arm = dict(status="PASS", drained=True, arm=name, order_index=index, timing_clock=legacy.performance_clock(),
                    collection_report_sha256=data["report_sha256"], concurrency=work["concurrency"], protocol_capacity=work["capacity"],
                    warmup_requests=work["warmup"], measured_requests=count, batch_capacity=prior["batch"],
                    files=[dict(path=str(path), sha256=sha) for path, sha in benchmark.runtime_files((source, data), None)[2]],
                    environment={key: value for key, value in benchmark.arm_environment(prior).items() if key.startswith("KATAGO_")},
                    hello=prior["hello"], before_heartbeat=before, after_heartbeat=after,
                    counter_deltas={key: after[key] - before[key] for key in after if key != "in_flight"}, measurement=measurement)
            arm["command"] = [prior["binary"], "nnworker", "--server", f"127.0.0.1:{50001 + index}", "--worker-id", f"quant-corpus-abba-{index}",
                    "--capacity", str(work["capacity"]), "--once", "--model", prior["model"], "--model-sha256", prior["model_sha256"], "--config", str(source / "worker.cfg")]
            if prior["backend"] != "cudaquantbackend":
                loaded = legacy.load_spec(source / legacy.SPEC_FILE)
                runner = lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, legacy.compact(self.fingerprint), b"synthetic CPU probe")
                arm["legacy_execution"] = {}
                for label, lo, hi in (("before", start - 3, start - 2), ("after", start + ns + 2, start + ns + 3)):
                    descriptor = legacy.probe_fingerprint(loaded, arm_dir, label, environment=legacy.apply_environment(loaded, {}), runner=runner)
                    command_path = arm_dir / descriptor["command"]["file"]
                    result_path = arm_dir / descriptor["result"]["file"]
                    command, result = ledger.read_json(command_path), ledger.read_json(result_path)
                    command["started_ns"], result["ended_ns"] = lo, hi
                    put(command_path, command)
                    result["command_sha256"] = ledger.planner.file_digest(command_path)
                    put(result_path, result)
                    for key in ("command", "stdout", "stderr", "result"):
                        descriptor[key] = legacy.artifact(arm_dir, arm_dir / descriptor[key]["file"])
                    arm["legacy_execution"][label] = descriptor
                arm["artifacts"] = [artifact(arm_dir / "worker.log")]
                arm["legacy_execution"]["observed_execution_id"] = observation["observed_execution"]["observed_execution_id"]
            put(arm_dir / "report.json", arm)
            put(arm_dir / "timings.json", dict(columns=columns, rows=rows))
            report["runs"].append(arm)
            scores.append(count / elapsed if work["metric"] == "rpc_throughput" else 1 / rows[0][0])
        decision = decide([scores[0], scores[3]], scores[1:3], 0.01, 0.05)
        decision["score_units"] = "requests/second" if work["metric"] == "rpc_throughput" else "inverse_microseconds"
        if work["metric"] == "p95_latency":
            decision.update(baseline_p95_geomean_us=1 / decision["baseline_geomean"], candidate_p95_geomean_us=1 / decision["candidate_geomean"],
                    latency_reduction=1 - decision["baseline_geomean"] / decision["candidate_geomean"],
                    threshold_definition="baseline P95 / candidate P95 >=1.01 (speed ratio), not 1% latency reduction")
        decision["meets_pairwise_speed_and_stability_limits"] = decision.pop("accepted")
        report["pairwise"] = decision
        put(directory / "report.json", report)
        return directory

    def ingest_abba(self, pair, **kwargs):
        receipt = ledger.execution_reserve(self.ledger, self.sha, "ABBA", pair["pair_key"])
        return ledger.execution_ingest_abba(self.ledger, receipt["reservation"]["reservation_id"], self.make_abba(pair, **kwargs))

    def test_initialization_and_v1_schema_dispatch_are_separate(self):
        state = self.initialize()
        self.assertEqual(state["consumed_attempts"], dict(COLLECTION=0, ABBA=0))
        self.assertFalse(state["publish_allowed"])
        self.assertEqual(state["abba_result_ingestion"], "RAW_FOUR_ARM_EVIDENCE_REQUIRED")
        with self.assertRaisesRegex(ValueError, "unsupported ledger contract"):
            ledger.inspect(self.ledger)
        self.fixture.init()
        with self.assertRaisesRegex(ValueError, "v1 cannot be upgraded"):
            ledger.execution_inspect(self.fixture.ledger)

    def test_register_requires_independent_rebuild_and_anchor(self):
        ledger.execution_initialize(self.ledger, 8, 8)
        with self.assertRaisesRegex(ValueError, "expected anchor"):
            ledger.execution_register_catalog(self.ledger, self.catalog_path, "f" * 64)
        value = ledger.read_json(self.catalog_path)
        value["performance_pairs"] = []
        put(self.catalog_path, value)
        with self.assertRaisesRegex(ValueError, "independently rebuilt"):
            ledger.execution_register_catalog(self.ledger, self.catalog_path, ledger.planner.file_digest(self.catalog_path))
        self.assertEqual(ledger.execution_inspect(self.ledger)["registered_catalog_sha256"], [])

    def test_pending_and_failed_attempts_consume_without_retry(self):
        self.initialize()
        receipt = self.reserve()
        state = ledger.execution_inspect(self.ledger)
        self.assertEqual((state["consumed_attempts"]["COLLECTION"], len(state["pending_reservations"])), (1, 1))
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self.reserve()
        failed = ledger.execution_fail(self.ledger, receipt["reservation_id"], "external process interrupted")
        self.assertEqual(failed["remaining_attempts"]["COLLECTION"], 7)
        self.assertEqual(failed["pending_reservations"], [])
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self.reserve()
        with self.assertRaisesRegex(ValueError, "already terminal"):
            ledger.execution_ingest_collection(self.ledger, receipt["reservation_id"], self.fixture.reference)

    def test_cross_catalog_metric_timeout_alias_and_format_do_not_reopen_budget(self):
        self.initialize()
        self.reserve()
        other = self.make_catalog("alias", work=dict(self.work, metric="p95_latency", task_timeout=180.0, max_numeric_evaluations=8))
        other.write_text(__import__("json").dumps(ledger.read_json(other), indent=5), encoding="utf-8")
        sha = ledger.planner.file_digest(other)
        ledger.execution_register_catalog(self.ledger, other, sha)
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self.reserve(self.row(cat=ledger.read_json(other)), sha)
        self.assertEqual(ledger.execution_inspect(self.ledger)["consumed_attempts"]["COLLECTION"], 1)

    def test_changed_reference_cannot_recollect_any_existing_execution(self):
        second = self.make_catalog("another-reference", batches=[2], reference_batch=2)
        self.initialize()
        ledger.execution_register_catalog(self.ledger, second, ledger.planner.file_digest(second))
        b = ledger.read_json(second)
        shared = self.cat["identity"]["required_control_execution_spec_ids"][0]
        ar, br = self.row(shared), self.row(shared, b)
        self.assertEqual(ar["collection_attempt_key"], br["collection_attempt_key"])
        self.assertNotEqual(ar["comparison_key"], br["comparison_key"])
        receipt = self.reserve(ar)
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self.reserve(br, ledger.planner.file_digest(second))
        self.assertEqual(receipt["stage"], "COLLECTION")

    def test_new_reference_reuses_existing_legacy_collection_for_cpu_comparison(self):
        self.initialize()
        self.qualify(self.fixture.reference)
        spec = next(row for row in self.cat["execution_specs"] if row["semantic"]["execution"]["backend"] == "cudaint8backend")
        candidate_row = self.row(spec["execution_spec_id"])
        self.qualify(self.make_legacy("cudaint8backend"), candidate_row)
        other = self.make_catalog("new-reference", batches=[2], reference_batch=2)
        sha, cat = ledger.planner.file_digest(other), ledger.read_json(other)
        ledger.execution_register_catalog(self.ledger, other, sha)
        directory = self.root / "reference-b2"
        shutil.copytree(self.fixture.reference, directory)
        self.prepare_unified(directory, batch=2)
        reference_row = self.row(cat["identity"]["canonical_reference_execution_spec_id"], cat)
        reservation = self.reserve(reference_row, sha)
        ledger.execution_ingest_collection(self.ledger, reservation["reservation_id"], directory)
        candidate_row = self.row(spec["execution_spec_id"], cat)
        result = ledger.execution_compare(self.ledger, sha, candidate_row["comparison_key"])
        self.assertEqual(result["consumed_attempts"]["COLLECTION"], 3)
        self.assertEqual(result["numerical_comparisons"][candidate_row["comparison_key"]]["metrics"]["experimental_gate"]["result"], "PASS")

    def test_global_budget_applies_across_catalogs(self):
        self.initialize(collection_budget=1)
        self.reserve()
        other = self.make_catalog("other", work=dict(self.work, metric="p95_latency"))
        sha = ledger.planner.file_digest(other)
        ledger.execution_register_catalog(self.ledger, other, sha)
        with self.assertRaisesRegex(ValueError, "budget exhausted"):
            self.reserve(self.unified_int8_row(), sha)

    def test_exclusive_lock_and_stale_head_do_not_append(self):
        initial = self.initialize()
        with ledger.locked(self.ledger):
            with self.assertRaises(FileExistsError):
                self.reserve()
        self.reserve()
        with self.assertRaisesRegex(ValueError, "expected anchor"):
            ledger.execution_reserve(self.ledger, self.sha, "COLLECTION", self.unified_int8_row()["collection_attempt_key"], initial["head_sha256"])
        self.assertEqual(ledger.execution_inspect(self.ledger)["consumed_attempts"]["COLLECTION"], 1)

    def test_raw_collections_and_cpu_comparisons_are_counted_separately(self):
        self.initialize()
        state = self.admit(self.fixture.reference)
        self.assertEqual(state["numerical_comparisons"], {})
        self.assertEqual(state["observed_collections"][self.row()["collection_attempt_key"]]["numeric_gate"], "NOT_COMPARED")
        state = ledger.execution_compare(self.ledger, self.sha, self.row()["comparison_key"])
        self.assertEqual(state["numerical_comparisons"][self.row()["comparison_key"]]["metrics"]["experimental_gate"]["result"], "PASS")
        head = state["head_sha256"]
        repeated = ledger.execution_compare(self.ledger, self.sha, self.row()["comparison_key"])
        self.assertEqual((repeated["head_sha256"], repeated["consumed_attempts"]["COLLECTION"]), (head, 1))
        self.assertEqual(repeated["pending_reservations"], [])

    def test_no_unreserved_observation_or_client_pass_unlocks_abba(self):
        self.initialize()
        with self.assertRaisesRegex(ValueError, "existing reservation"):
            ledger.execution_ingest_collection(self.ledger, "f" * 64, self.fixture.reference)
        with self.assertRaisesRegex(ValueError, "requires validated"):
            ledger.execution_compare(self.ledger, self.sha, self.row()["comparison_key"])
        with self.assertRaisesRegex(ValueError, "numeric PASS"):
            ledger.execution_reserve(self.ledger, self.sha, "ABBA", self.cat["performance_pairs"][0]["pair_key"])
        self.assertEqual(ledger.execution_inspect(self.ledger)["consumed_attempts"], dict(COLLECTION=0, ABBA=0))

    def test_legacy_fp16_and_int8_raw_probes_are_admitted_and_qualified(self):
        self.initialize()
        self.qualify(self.fixture.reference)
        for backend in ("cudabackend", "cudaint8backend"):
            spec = next(row for row in self.cat["execution_specs"] if row["semantic"]["execution"]["backend"] == backend)
            row = self.row(spec["execution_spec_id"])
            state = self.qualify(self.make_legacy(backend), row)
            observation = state["observed_collections"][row["collection_attempt_key"]]["observed_execution"]
            self.assertTrue(observation["observed_execution_id"].startswith("rustgo-legacy-evidence-v1:"))
            self.assertEqual(state["numerical_comparisons"][row["comparison_key"]]["metrics"]["experimental_gate"]["result"], "PASS")
        pair = next(pair for pair in self.cat["performance_pairs"] if pair["candidate_execution_spec_id"] == self.reference_id)
        receipt = ledger.execution_reserve(self.ledger, self.sha, "ABBA", pair["pair_key"])
        self.assertEqual(len(receipt["reservation"]["numeric_qualification"]), 2)
        state = ledger.execution_fail(self.ledger, receipt["reservation"]["reservation_id"], "synthetic interrupted ABBA; no process ran")
        self.assertEqual(state["consumed_attempts"], dict(COLLECTION=3, ABBA=1))
        other = self.make_catalog("abba-alias", work=dict(self.work, metric="p95_latency"))
        sha = ledger.planner.file_digest(other)
        ledger.execution_register_catalog(self.ledger, other, sha)
        with self.assertRaisesRegex(ValueError, "already consumed"):
            ledger.execution_reserve(self.ledger, sha, "ABBA", pair["pair_key"])

    def test_abba_rejects_missing_control_and_numeric_failure(self):
        self.initialize()
        self.qualify(self.fixture.reference)
        row = self.unified_int8_row()
        directory = self.fixture.collections[self.fixture.int8]
        self.change_win(directory, 0.7)
        state = self.qualify(directory, row)
        self.assertEqual(state["numerical_comparisons"][row["comparison_key"]]["metrics"]["experimental_gate"]["result"], "FAIL")
        for pair in self.cat["performance_pairs"]:
            with self.assertRaisesRegex(ValueError, "numeric PASS"):
                ledger.execution_reserve(self.ledger, self.sha, "ABBA", pair["pair_key"])
        self.assertEqual(ledger.execution_inspect(self.ledger)["consumed_attempts"]["ABBA"], 0)

    def change_win(self, directory, win):
        result_path = directory / "results/000001.pb"
        result = self.protocol.pb.EvalResult.FromString(result_path.read_bytes())
        result.output.white_win_prob, result.output.white_loss_prob = win, 1 - win
        raw = result.SerializeToString(deterministic=True)
        result_path.write_bytes(raw)
        output = list(ledger.planner.corpus_io.read_jsonl((directory / "outputs.jsonl").read_bytes(), "outputs"))[0]
        output.update(result=message_dict(result), result_sha256=ledger.digest(raw), output_sha256=ledger.digest(result.output.SerializeToString(deterministic=True)))
        (directory / "outputs.jsonl").write_bytes(lines([output]))
        report = ledger.read_json(directory / "report.json")
        report["ordered_output_hashes_sha256"] = ledger.digest((1).to_bytes(8, "little") + bytes.fromhex(output["output_sha256"]))
        self.continuous(directory, report)
        self.save_report(directory, report)

    def test_holdout_rejected_before_any_pb_or_output_open_and_consumes(self):
        self.initialize()
        receipt = self.reserve()
        report = ledger.read_json(self.fixture.reference / "report.json")
        report["split"] = "holdout"
        put(self.fixture.reference / "report.json", report)
        original = Path.read_bytes
        def guarded(path):
            if path.suffix == ".pb" or path.name in ("outputs.jsonl", "worker.log"):
                raise AssertionError("forbidden output opened before holdout preflight")
            return original(path)
        with patch.object(Path, "read_bytes", guarded):
            with self.assertRaisesRegex(ValueError, "holdout/unknown"):
                ledger.execution_ingest_collection(self.ledger, receipt["reservation_id"], self.fixture.reference)
        state = ledger.execution_inspect(self.ledger)
        self.assertEqual((len(state["failed_attempts"]), state["consumed_attempts"]["COLLECTION"]), (1, 1))

    def test_declared_pass_wrong_config_cannot_replace_raw_identity(self):
        self.initialize()
        receipt = self.reserve()
        report = ledger.read_json(self.fixture.reference / "report.json")
        report["experimental_gate"] = dict(result="PASS")
        cfg = report["config_text"].replace("numSearchThreads=1", "numSearchThreads=2")
        (self.fixture.reference / "worker.cfg").write_text(cfg, encoding="utf-8")
        report.update(config_text=cfg, config_sha256=ledger.planner.file_digest(self.fixture.reference / "worker.cfg"))
        self.save_report(self.fixture.reference, report)
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            ledger.execution_ingest_collection(self.ledger, receipt["reservation_id"], self.fixture.reference)
        self.assertEqual(len(ledger.execution_inspect(self.ledger)["failed_attempts"]), 1)

    def test_decode_error_is_terminal_not_reingestable(self):
        from google.protobuf.message import DecodeError
        self.initialize()
        receipt = self.reserve()
        with patch.object(comparison, "load_collection", side_effect=DecodeError("synthetic corrupt protobuf")):
            with self.assertRaisesRegex(ValueError, "remains consumed"):
                ledger.execution_ingest_collection(self.ledger, receipt["reservation_id"], self.fixture.reference)
        with self.assertRaisesRegex(ValueError, "already terminal"):
            ledger.execution_ingest_collection(self.ledger, receipt["reservation_id"], self.fixture.reference)

    def test_admitted_raw_output_and_snapshot_are_revalidated(self):
        self.initialize()
        self.qualify(self.fixture.reference)
        source = self.fixture.reference / "results/000001.pb"
        source.write_bytes(source.read_bytes() + b"bad")
        with self.assertRaisesRegex(ValueError, "immutable source"):
            ledger.execution_inspect(self.ledger)

    def test_actual_request_lease_must_match_planned_timeout(self):
        directory = self.fixture.reference
        report = ledger.read_json(directory / "report.json")
        requests, request_sha = ledger.planner.corpus_io.make_requests(self.protocol, [self.fixture.record], self.fixture.model_sha,
                            "quant-corpus-" + report["source_request_file_sha256"][:24], 179000)
        request = requests[0]
        request_raw = request.SerializeToString(deterministic=True)
        (directory / "requests/000001.pb").write_bytes(request_raw)
        index = list(ledger.planner.corpus_io.read_jsonl((directory / "requests-index.jsonl").read_bytes(), "index"))[0]
        index.update(request=message_dict(request), request_sha256=ledger.digest(request_raw), input_hash=request.input_hash.hex())
        (directory / "requests-index.jsonl").write_bytes(lines([index]))
        result = self.protocol.pb.EvalResult.FromString((directory / "results/000001.pb").read_bytes())
        result.input_hash = request.input_hash
        raw = result.SerializeToString(deterministic=True)
        (directory / "results/000001.pb").write_bytes(raw)
        output = list(ledger.planner.corpus_io.read_jsonl((directory / "outputs.jsonl").read_bytes(), "outputs"))[0]
        output.update(result=message_dict(result), result_sha256=ledger.digest(raw))
        (directory / "outputs.jsonl").write_bytes(lines([output]))
        report["requests_sha256"] = request_sha
        self.continuous(directory, report)
        self.save_report(directory, report)
        self.initialize()
        receipt = self.reserve()
        with self.assertRaisesRegex(ValueError, "lease differs"):
            ledger.execution_ingest_collection(self.ledger, receipt["reservation_id"], directory)
        self.assertEqual(len(ledger.execution_inspect(self.ledger)["failed_attempts"]), 1)

    def test_rehashed_client_pass_is_rejected_by_raw_comparison_replay(self):
        self.initialize()
        state = self.qualify(self.fixture.reference)
        path = self.ledger / "events" / f"{state['event_count'] - 1:06d}.json"
        event = ledger.read_json(path)
        event["payload"]["metrics"]["experimental_gate"]["measured"] = 0.123
        event.pop("event_sha256")
        event["event_sha256"] = ledger.digest(ledger.compact(event))
        put(path, event)
        with self.assertRaisesRegex(ValueError, "recomputed raw protobufs"):
            ledger.execution_inspect(self.ledger)

    def test_event_hash_and_catalog_snapshot_tampering_fail_closed(self):
        self.initialize()
        path = next((self.ledger / "evidence").rglob("catalog.json"))
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "frozen execution evidence"):
            ledger.execution_inspect(self.ledger)

    def test_cpu_semantic_comparison_accepts_metadata_alias_but_rejects_position_change(self):
        a = comparison.load_collection(self.fixture.reference, self.protocol)
        b = comparison.load_collection(self.fixture.collections[self.fixture.int8], self.protocol, full_reference=a["recipe"])
        b["report"]["source_request_file_sha256"] = "b" * 64
        row = next(iter(b["positions"].values()))
        row["record"] = dict(row["record"], name="different provenance label")
        self.assertEqual(ledger.execution_compare_data(a, b)["experimental_gate"]["result"], "PASS")
        row["request"].parameters.symmetry = 1
        with self.assertRaisesRegex(ValueError, "ordered semantics"):
            ledger.execution_compare_data(a, b)

    def test_cli_execution_init_and_inspect_are_explicit(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["ledger", "execution-init", "--output", str(self.ledger), "--max-collection-attempts", "8", "--max-abba-attempts", "8"]), patch("sys.stdout", output):
            self.assertEqual(ledger.main(), 0)
        self.assertEqual(ledger.planner.strict_json(output.getvalue())["schema"], ledger.EXECUTION_SCHEMA)

    def change_arm(self, directory, index, change):
        report = ledger.read_json(directory / "report.json")
        arm = report["runs"][index]
        arm_dir = directory / f"{index + 1:02d}-{benchmark.ORDER[index]}"
        change(arm, arm_dir)
        put(arm_dir / "report.json", arm)
        put(directory / "report.json", report)

    def change_probe(self, directory, index, side, change):
        def update(arm, arm_dir):
            descriptor = arm["legacy_execution"][side]
            command_path, result_path = (arm_dir / descriptor[key]["file"] for key in ("command", "result"))
            command, result = ledger.read_json(command_path), ledger.read_json(result_path)
            change(command, result, arm)
            put(command_path, command)
            result["command_sha256"] = ledger.planner.file_digest(command_path)
            put(result_path, result)
            for key in ("command", "stdout", "stderr", "result"):
                descriptor[key] = legacy.artifact(arm_dir, arm_dir / descriptor[key]["file"])
        self.change_arm(directory, index, update)

    def ready_abba(self):
        self.initialize()
        self.qualify_all()

    def test_abba_full_control_matrix_is_unranked_and_never_adopted(self):
        self.ready_abba()
        for pair in self.cat["performance_pairs"]:
            state = self.ingest_abba(pair)
            self.assertEqual(state["performance"][pair["pair_key"]]["gate"], "PASS")
        self.assertEqual((len(state["performance"]), state["consumed_attempts"]["ABBA"], state["pending_reservations"]), (5, 5, []))
        head = state["head_sha256"]
        result = ledger.execution_summary(self.ledger, self.sha, self.root / "summary", head)["report"]
        self.assertEqual(result["conclusion"], "CANDIDATE_REQUIRES_INDEPENDENT_CONFIRMATION")
        self.assertEqual(len(result["positive_candidates_unranked_requiring_confirmation"]), 2)
        self.assertTrue(result["complete_finite_matrix"])
        self.assertTrue(all(len(row["controls"]) == 3 for row in result["candidates"]))
        self.assertEqual(result["scope"], "SMOKE_DIAGNOSTIC_ONLY")
        self.assertFalse(result["formal_adoption_allowed"] or result["publish_allowed"] or result["production_certified"])
        self.assertEqual(ledger.execution_inspect(self.ledger)["head_sha256"], head)
        other = self.make_catalog("changed-metric", work=dict(self.work, metric="p95_latency"))
        sha = ledger.planner.file_digest(other)
        ledger.execution_register_catalog(self.ledger, other, sha)
        changed = ledger.execution_summary(self.ledger, sha, self.root / "summary-other")["report"]
        self.assertEqual(changed["positive_candidates_unranked_requiring_confirmation"], [])
        self.assertTrue(any(cell["status"] == "DECISION_CONTEXT_MISMATCH" for row in changed["candidates"] for cell in row["controls"]))
        with self.assertRaisesRegex(ValueError, "already consumed"):
            ledger.execution_reserve(self.ledger, sha, "ABBA", self.cat["performance_pairs"][0]["pair_key"])

    def test_abba_partial_matrix_cannot_select_a_winner(self):
        self.ready_abba()
        self.ingest_abba(self.cat["performance_pairs"][0])
        result = ledger.execution_summary(self.ledger, self.sha, self.root / "partial")["report"]
        self.assertEqual(result["conclusion"], "EVIDENCE_INCOMPLETE")
        self.assertEqual(result["positive_candidates_unranked_requiring_confirmation"], [])
        self.assertFalse(result["complete_finite_matrix"])

    def test_abba_no_gain_and_unstable_are_consumed_terminal_results(self):
        self.ready_abba()
        for pair, ns, expected in ((self.cat["performance_pairs"][0], 10000000, "NO_GAIN"),
                                   (self.cat["performance_pairs"][1], (8000000, 10000000), "UNSTABLE")):
            state = self.ingest_abba(pair, candidate_ns=ns)
            self.assertEqual(state["performance"][pair["pair_key"]]["gate"], expected)
            with self.assertRaisesRegex(ValueError, "already consumed"):
                ledger.execution_reserve(self.ledger, self.sha, "ABBA", pair["pair_key"])
            reservation = ledger.execution_replay(self.ledger)[2]["reservations"][pair["pair_key"]]
            with self.assertRaisesRegex(ValueError, "already terminal"):
                ledger.execution_fail(self.ledger, reservation["reservation_id"], "cannot refund a completed negative comparison")
        self.assertEqual(state["consumed_attempts"]["ABBA"], 2)

    def test_abba_p95_uses_raw_latency_even_with_throughput_gain(self):
        self.catalog_path = self.make_catalog("p95", work=dict(self.work, metric="p95_latency"))
        self.cat, self.sha = ledger.read_json(self.catalog_path), ledger.planner.file_digest(self.catalog_path)
        self.ready_abba()
        pair = self.cat["performance_pairs"][0]
        state = self.ingest_abba(pair, candidate_ns=9000000, candidate_latency_us=11000)
        self.assertEqual(state["performance"][pair["pair_key"]]["gate"], "NO_GAIN")
        self.assertLess(state["performance"][pair["pair_key"]]["pairwise"]["improvement"], 0)

    def test_abba_invalid_raw_timing_is_failed_without_retry(self):
        self.ready_abba()
        pair = self.cat["performance_pairs"][0]
        receipt = ledger.execution_reserve(self.ledger, self.sha, "ABBA", pair["pair_key"])
        directory = self.make_abba(pair)
        path = directory / "02-candidate/timings.json"
        value = ledger.read_json(path)
        value["rows"][0][0] += 1
        put(path, value)
        with self.assertRaisesRegex(ValueError, "distribution differs"):
            ledger.execution_ingest_abba(self.ledger, receipt["reservation"]["reservation_id"], directory)
        state = ledger.execution_inspect(self.ledger)
        self.assertEqual((state["consumed_attempts"]["ABBA"], len(state["failed_attempts"]), state["performance"]), (1, 1, {}))
        with self.assertRaisesRegex(ValueError, "already terminal"):
            ledger.execution_ingest_abba(self.ledger, receipt["reservation"]["reservation_id"], directory)

    def test_abba_rejects_actual_input_counter_command_environment_and_profile_changes(self):
        self.ready_abba()
        changes = [
            (lambda arm, _: arm["counter_deltas"].update(nn_rows=2), "NN counters"),
            (lambda arm, _: arm["command"].__setitem__(7, "2"), "command differs"),
            (lambda arm, _: arm["environment"].update(KATAGO_CUDA_INT8_GEMM_TUNE="1"), "environment differs"),
            (lambda arm, _: arm.update(concurrency=2), "workload/execution"),
        ]
        for pair, (change, message) in zip(self.cat["performance_pairs"], changes):
            receipt = ledger.execution_reserve(self.ledger, self.sha, "ABBA", pair["pair_key"])
            directory = self.make_abba(pair)
            self.change_arm(directory, 0, change)
            with self.assertRaisesRegex(ValueError, message):
                ledger.execution_ingest_abba(self.ledger, receipt["reservation"]["reservation_id"], directory)
        pair = self.cat["performance_pairs"][-1]
        receipt = ledger.execution_reserve(self.ledger, self.sha, "ABBA", pair["pair_key"])
        directory = self.make_abba(pair)
        report = ledger.read_json(directory / "report.json")
        report["prepared_request_sha256"] = "f" * 64
        put(directory / "report.json", report)
        with self.assertRaisesRegex(ValueError, "ordered complete input"):
            ledger.execution_ingest_abba(self.ledger, receipt["reservation"]["reservation_id"], directory)

    def test_abba_legacy_requires_clock_markers_and_probes_outside_timing(self):
        self.ready_abba()
        pairs = [pair for pair in self.cat["performance_pairs"] if pair["baseline_execution_spec_id"] in self.cat["identity"]["required_control_execution_spec_ids"]]
        for mode, pair in zip(("overlap", "no-clock", "marker"), pairs):
            receipt = ledger.execution_reserve(self.ledger, self.sha, "ABBA", pair["pair_key"])
            directory = self.make_abba(pair)
            if mode == "overlap":
                self.change_probe(directory, 0, "before", lambda command, result, arm: result.update(ended_ns=arm["measurement"]["started_ns"]))
                expected = "overlaps the timed"
            elif mode == "no-clock":
                for side in ("before", "after"):
                    self.change_probe(directory, 0, side, lambda command, result, _: (command.pop("timing_clock"), result.pop("timing_clock")))
                expected = "clock"
            else:
                def change_log(arm, arm_dir):
                    (arm_dir / "worker.log").write_text("no actual execution marker\n", encoding="utf-8")
                    arm["artifacts"] = [artifact(arm_dir / "worker.log")]
                self.change_arm(directory, 0, change_log)
                expected = "marker"
            with self.assertRaisesRegex(ValueError, expected):
                ledger.execution_ingest_abba(self.ledger, receipt["reservation"]["reservation_id"], directory)

    def test_abba_reused_interval_and_reformatted_raw_arm_are_not_new_evidence(self):
        self.ready_abba()
        controls = self.cat["identity"]["required_control_execution_spec_ids"]
        pairs = [pair for pair in self.cat["performance_pairs"] if pair["baseline_execution_spec_id"] == controls[0]]
        self.ingest_abba(pairs[0], base_start=10000000000)
        receipt = ledger.execution_reserve(self.ledger, self.sha, "ABBA", pairs[1]["pair_key"])
        directory = self.make_abba(pairs[1], base_start=10000000000)
        path = directory / "01-baseline/timings.json"
        path.write_text(__import__("json").dumps(ledger.read_json(path), indent=5), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "already consumed|overlap previously"):
            ledger.execution_ingest_abba(self.ledger, receipt["reservation"]["reservation_id"], directory)

    def test_abba_rehashed_client_gate_and_missing_snapshot_are_recomputed(self):
        self.ready_abba()
        pair = self.cat["performance_pairs"][0]
        state = self.ingest_abba(pair, candidate_ns=10000000)
        path = self.ledger / "events" / f"{state['event_count'] - 1:06d}.json"
        original = ledger.read_json(path)
        tampered = deepcopy(original)
        tampered["payload"]["observation"]["gate"] = "PASS"
        tampered.pop("event_sha256")
        tampered["event_sha256"] = ledger.digest(ledger.compact(tampered))
        put(path, tampered)
        with self.assertRaisesRegex(ValueError, "recomputed raw evidence"):
            ledger.execution_inspect(self.ledger)
        tampered = deepcopy(original)
        tampered["evidence"].pop()
        tampered.pop("event_sha256")
        tampered["event_sha256"] = ledger.digest(ledger.compact(tampered))
        put(path, tampered)
        with self.assertRaisesRegex(ValueError, "every required frozen artifact"):
            ledger.execution_inspect(self.ledger)

    def test_abba_no_reservation_and_summary_anchor_or_existing_output_fail_closed(self):
        self.ready_abba()
        with self.assertRaisesRegex(ValueError, "existing reservation"):
            ledger.execution_ingest_abba(self.ledger, "f" * 64, self.root / "must-not-open")
        head = ledger.execution_inspect(self.ledger)["head_sha256"]
        self.ingest_abba(self.cat["performance_pairs"][0])
        with self.assertRaisesRegex(ValueError, "expected anchor"):
            ledger.execution_summary(self.ledger, self.sha, self.root / "summary", head)
        output = self.root / "already-exists"
        output.mkdir()
        with self.assertRaisesRegex(ValueError, "NEW directory"):
            ledger.execution_summary(self.ledger, self.sha, output)


if __name__ == "__main__":
    unittest.main()
