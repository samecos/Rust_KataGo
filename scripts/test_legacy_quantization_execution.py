"""Self-contained synthetic CPU receipts; never run a GPU or need old artifacts."""
from copy import deepcopy
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import legacy_quantization_execution as execution


class LegacyExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Formats reproduce the historical protocol, but these constants are
        # synthetic test input, not execution/certification evidence.
        cls.fingerprint_original = dict(architecture="sm_120", compute_capability="12.0", gpu_name="SYNTHETIC SM120 GPU",
                sm_count=70, l2_cache_bytes=50331648, model_sha256="e" * 64,
                backend_build=dict(kernel_build_id="SYNTHETIC_CPU_TEST_BUILD", compiled_sm=["120", "89"], cublaslt_version=130600),
                model_architecture=dict(format="native_tf3_v17", blocks=15, ffn_hidden=[16, 48, 512] * 15))
        cls.plan_original = dict(schema=2, kind="cuda-tactic-plan", plan_id="synthetic-fp16-plan",
                target={**{key: cls.fingerprint_original[key] for key in execution.TARGET_KEYS}, "max_batch_size": 8},
                backend_build=deepcopy(cls.fingerprint_original["backend_build"]), selection=dict(binary_sha256="d" * 64),
                apply=dict(tactic_overrides={"KATAGO_CUDA_ATTN": "fa2", "KATAGO_CUDA_ATTN_TILE": "q64"}))
        cls.fp16_log_original = "cudaTacticPlan '__SYNTHETIC_PLAN_PATH__' installed (plan id: synthetic-fp16-plan)\n"
        cls.int8_log_original = ("[cuda-tactic] name=int8_gemm_tune enabled=0 candidate_pool=16 default=0\n"
                "[synthetic timestamp] [cuda-int8] precision=W8A8 accumulation=INT32 scope=ffn min_ffn_width=0 int8_ffn_layers=45 "
                f"quantization=w8a8-row-out-rne-v1 model_sha256={'e' * 64} heads=FP16/FP32 activation_scaling=dynamic-per-row calibration=none\n"
                "[cuda-tactic] name=attention requested=fa2 launch=fa2 effective=fa2 tile=q64\n"
                "[cuda-int8] launch=cublaslt-int8 accumulator=int32 imma=true numerical_flags=0x200804 M=361 N=480 K=512\n")
        cls.target = execution.ROOT / "target/legacy-execution-cpu-tests"
        cls.target.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-", dir=self.target)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.binary, self.model = self.root / "synthetic.exe", self.root / "synthetic.bin.gz"
        self.binary.write_bytes(b"CPU SYNTHETIC EXECUTABLE NEVER RUN")
        self.model.write_bytes(b"CPU SYNTHETIC MODEL NEVER RUN")
        self.model_sha = execution.file_sha(self.model)
        self.fingerprint = deepcopy(self.fingerprint_original)
        self.fingerprint["model_sha256"] = self.model_sha
        self.plan = deepcopy(self.plan_original)
        self.plan["target"]["model_sha256"] = self.model_sha
        if "binary" in self.plan.get("selection", {}):
            self.plan["selection"]["binary"]["sha256"] = execution.file_sha(self.binary)
        if "binary_sha256" in self.plan.get("selection", {}):
            self.plan["selection"]["binary_sha256"] = execution.file_sha(self.binary)
        self.plan_path = self.root / "plan.json"
        execution.write_new(self.plan_path, self.plan)
        self.spec_path = self.root / "spec.json"

    def spec(self, backend="cudaint8backend"):
        cfg = self.root / "source.cfg"
        text = ("rules=chinese\nkomi=7.5\n" + f"nnBackend={backend}\n" +
                "nnMaxBatchSize=8\nnumNNServerThreadsPerModel=1\nnnCacheSizePowerOfTwo=18\nnnMutexPoolSizePowerOfTwo=14\n")
        text += f"cudaTacticPlan={self.plan_path}\n" if backend == "cudabackend" else "cudaInt8Scope=ffn\ncudaInt8MinFfnWidth=0\n"
        cfg.write_text(text, encoding="utf-8")
        item = lambda path: dict(path=str(path), sha256=execution.file_sha(path))
        spec = dict(schema=execution.SCHEMA, backend=backend, binary=item(self.binary), model=item(self.model), config=item(cfg),
                    plan=item(self.plan_path) if backend == "cudabackend" else None, batch=8,
                    environment={} if backend == "cudabackend" else dict(execution.INT8_ENVIRONMENT))
        execution.write_new(self.spec_path, spec)
        return execution.load_spec(self.spec_path)

    def runner(self, argv, **kwargs):
        # The command receipt must already exist before the subprocess adapter.
        directories = [path for path in self.root.rglob("command.json") if execution.strict_json(path.read_bytes())["argv"] == argv]
        self.assertTrue(directories)
        self.assertEqual(argv[1:3], ["cuda-fingerprint", "--model"])
        return subprocess.CompletedProcess(argv, 0, execution.compact(self.fingerprint), b"synthetic fingerprint fixture; CPU only")

    def make_collection(self, backend="cudaint8backend"):
        loaded = self.spec(backend)
        directory = self.root / "collection"
        directory.mkdir()
        frozen = execution.freeze_spec(loaded, directory)
        source = Path(loaded["config"]).read_bytes()
        (directory / "source-worker.cfg").write_bytes(source)
        runtimes = []
        effective = source
        plan_path = None
        if loaded["plan"]:
            plan_path = directory / "runtime-cudaTacticPlan.json"
            plan_path.write_bytes(self.plan_path.read_bytes())
            effective = source.replace(str(self.plan_path).encode(), str(plan_path).encode())
            runtimes = [dict(config_key="cudaTacticPlan", file=plan_path.name, sha256=execution.file_sha(plan_path))]
        (directory / "worker.cfg").write_bytes(effective)
        if loaded["plan"]:
            log = self.fp16_log_original.replace("__SYNTHETIC_PLAN_PATH__", str(plan_path))
        else:
            log = self.int8_log_original.replace(self.fingerprint_original["model_sha256"], self.model_sha)
        (directory / "worker.log").write_text(log, encoding="utf-8")
        backend_info = "fixture; backend=" + backend
        if not loaded["plan"]:
            backend_info += "; precision=W8A8-mixed; int8-scope=ffn; int8-min-ffn-width=0; quantization=w8a8-row-out-rne-v1"
        report = dict(split="calibration", backend=backend, binary=str(self.binary), binary_sha256=execution.file_sha(self.binary),
                      model=str(self.model), model_sha256=self.model_sha, config_sha256=execution.sha(effective), config_text=(directory / "worker.cfg").read_text(encoding="utf-8"),
                      batch=8, capacity=32, request_window=32, controlled_environment=loaded["environment"], runtime_artifacts=runtimes,
                      hello=dict(model_sha256=self.model_sha, backend_info=backend_info), legacy_execution=dict(spec=frozen),
                      command=[str(self.binary), "nnworker", "--config", str(directory / "worker.cfg")])
        report["artifacts"] = [execution.artifact(directory, path) for path in directory.iterdir() if path.is_file()]
        for label in ("before", "after"):
            report["legacy_execution"][label] = execution.probe_fingerprint(loaded, directory, label,
                        environment=execution.apply_environment(loaded, {"KATAGO_UNTRUSTED": "removed"}), runner=self.runner)
        return loaded, directory, report

    def update_artifact(self, directory, report, name):
        report["artifacts"] = [execution.artifact(directory, directory / name) if item["file"] == name else item for item in report["artifacts"]]

    def rewrite_probe(self, directory, descriptor, change):
        command_path, result_path = (directory / descriptor[key]["file"] for key in ("command", "result"))
        command, result = (execution.strict_json(path.read_bytes()) for path in (command_path, result_path))
        change(command, result)
        command_path.write_bytes(execution.compact(command))
        result["command_sha256"] = execution.file_sha(command_path)
        result_path.write_bytes(execution.compact(result))
        descriptor.update(command=execution.artifact(directory, command_path), result=execution.artifact(directory, result_path))

    def test_new_probes_use_benchmark_performance_clock(self):
        with patch.object(execution.time, "monotonic_ns", side_effect=AssertionError("wrong clock")), \
                patch.object(execution.time, "perf_counter_ns", side_effect=[100, 101, 102, 103]):
            loaded, directory, report = self.make_collection()
        before = report["legacy_execution"]["before"]
        interval = execution.probe_interval(directory, before, require_performance_clock=True)
        self.assertEqual(interval, dict(timing_clock=execution.performance_clock(), started_ns=100, ended_ns=101))
        self.assertEqual(execution.verify_probe(directory, before, loaded, binary=self.binary, model=self.model)[2], (100, 101))
        execution.validate_collection(directory, report)

    def test_old_unmarked_probes_remain_numeric_only(self):
        _, directory, report = self.make_collection()
        for label in ("before", "after"):
            self.rewrite_probe(directory, report["legacy_execution"][label],
                               lambda command, result: (command.pop("timing_clock"), result.pop("timing_clock")))
        execution.validate_collection(directory, report)
        before = report["legacy_execution"]["before"]
        self.assertIsNone(execution.probe_interval(directory, before)["timing_clock"])
        with self.assertRaisesRegex(ValueError, "unmarked probe"):
            execution.probe_interval(directory, before, require_performance_clock=True)

    def test_mixed_old_new_probe_pair_is_rejected(self):
        _, directory, report = self.make_collection()
        self.rewrite_probe(directory, report["legacy_execution"]["after"],
                           lambda command, result: (command.pop("timing_clock"), result.pop("timing_clock")))
        with self.assertRaisesRegex(ValueError, "clock changed across"):
            execution.validate_collection(directory, report)

    def test_partial_or_disagreeing_probe_clocks_are_rejected(self):
        _, directory, report = self.make_collection()
        before = report["legacy_execution"]["before"]
        self.rewrite_probe(directory, before, lambda command, result: result.pop("timing_clock"))
        with self.assertRaisesRegex(ValueError, "partial fingerprint timing clock"):
            execution.probe_interval(directory, before)
        self.rewrite_probe(directory, before, lambda command, result: result.update(
            timing_clock={**command["timing_clock"], "implementation": "different clock"}))
        with self.assertRaisesRegex(ValueError, "clock changed within"):
            execution.probe_interval(directory, before)

    def test_invalid_explicit_probe_clocks_are_rejected(self):
        _, directory, report = self.make_collection()
        before = report["legacy_execution"]["before"]
        invalid = [None, {**execution.performance_clock(), "unknown": True}]
        for field, value in (("api", "time.monotonic_ns"), ("implementation", ""), ("monotonic", 1),
                             ("adjustable", True), ("resolution_seconds", 0), ("resolution_seconds", True)):
            invalid.append({**execution.performance_clock(), field: value})
        for clock in invalid:
            with self.subTest(clock=clock):
                self.rewrite_probe(directory, before, lambda command, result:
                                   (command.update(timing_clock=clock), result.update(timing_clock=clock)))
                with self.assertRaises(ValueError):
                    execution.probe_interval(directory, before)

    def test_int8_q64_tune0_model_layers_and_imma_validate(self):
        _, directory, report = self.make_collection()
        result = execution.validate_collection(directory, report)
        self.assertEqual(result["actual_execution"]["int8_ffn_layers"], 45)
        self.assertTrue(result["actual_execution"]["imma"])
        self.assertEqual(result["evidence_scope"], execution.SCOPE)
        self.assertIsNone(result["driver_api_version"])
        self.assertIsNone(result["gpu_uuid"])
        self.assertEqual(len(result["artifact_paths"]), 12)
        report["legacy_execution"]["observed_execution_id"] = result["observed_execution_id"]
        self.assertEqual(execution.validate_collection(directory, report)["observed_execution_id"], result["observed_execution_id"])

    def test_fp16_loader_and_plan_build_binding_validate(self):
        _, directory, report = self.make_collection("cudabackend")
        result = execution.validate_collection(directory, report)
        self.assertEqual(result["actual_execution"]["plan_id"], self.plan["plan_id"])
        self.assertEqual(len(result["artifact_paths"]), 13)

    def test_spec_alias_and_plan_relocation_preserve_semantic_identity(self):
        loaded = self.spec("cudabackend")
        plan_copy = self.root / "renamed-plan.json"
        plan_copy.write_bytes(self.plan_path.read_bytes())
        config_copy = self.root / "renamed.cfg"
        config_copy.write_bytes(Path(loaded["config"]).read_bytes().replace(str(self.plan_path).encode(), str(plan_copy).encode()))
        copied = deepcopy(loaded["spec"])
        copied["label"] = "renamed alias"
        copied["plan"]["path"] = str(plan_copy)
        copied["config"] = dict(path=str(config_copy), sha256=execution.file_sha(config_copy))
        new_spec = self.root / "alias.json"
        execution.write_new(new_spec, copied)
        self.assertEqual(execution.load_spec(new_spec)["spec_id"], loaded["spec_id"])

    def test_changed_binary_config_or_model_fails_cpu_loading(self):
        self.spec()
        self.binary.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "immutable source binary"):
            execution.load_spec(self.spec_path)

    def test_int8_requires_exact_tactics_and_rejects_unknown_spec_fields(self):
        loaded = self.spec()
        value = deepcopy(loaded["spec"])
        value["environment"]["KATAGO_CUDA_INT8_GEMM_TUNE"] = "1"
        self.spec_path.write_bytes(execution.compact(value))
        with self.assertRaisesRegex(ValueError, "exactly explicit q64/tune0"):
            execution.load_spec(self.spec_path)
        value = deepcopy(loaded["spec"])
        value["recipe_sha256"] = "f" * 64
        self.spec_path.write_bytes(execution.compact(value))
        with self.assertRaisesRegex(ValueError, "unknown"):
            execution.load_spec(self.spec_path)

    def test_both_historical_plan_binary_field_forms_are_enforced(self):
        loaded = self.spec("cudabackend")
        for field in ("binary", "binary_sha256"):
            with self.subTest(field=field):
                plan = deepcopy(self.plan)
                plan["selection"]["binary"] = {"sha256": execution.file_sha(self.binary)}
                plan["selection"]["binary_sha256"] = execution.file_sha(self.binary)
                if field == "binary":
                    plan["selection"][field]["sha256"] = "f" * 64
                else:
                    plan["selection"][field] = "f" * 64
                plan_raw = execution.compact(plan)
                spec = deepcopy(loaded["spec"])
                spec["plan"]["sha256"] = execution.sha(plan_raw)
                with self.assertRaisesRegex(ValueError, "historical binary binding"):
                    execution.decode_spec(execution.compact(spec), Path(loaded["config"]).read_bytes(), plan_raw)

    def test_fp16_original_build_binding_cannot_be_replaced(self):
        loaded = self.spec("cudabackend")
        fingerprint = deepcopy(self.fingerprint)
        fingerprint["backend_build"]["kernel_build_id"] = "other-build"
        with self.assertRaisesRegex(ValueError, "device/backend build binding"):
            execution.validate_fingerprint(fingerprint, loaded)

    def test_probe_failure_and_timeout_are_preserved_without_retry(self):
        loaded = self.spec()
        directory = self.root / "failed"
        directory.mkdir()
        def fail(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"], output=b"partial", stderr=b"timeout")
        with self.assertRaises(subprocess.TimeoutExpired):
            execution.probe_fingerprint(loaded, directory, "before", environment=loaded["environment"], runner=fail)
        self.assertEqual((directory / "before-fingerprint/stdout.json").read_bytes(), b"partial")
        self.assertEqual(execution.strict_json((directory / "before-fingerprint/result.json").read_bytes())["status"], "FAILED")
        with self.assertRaisesRegex(ValueError, "overwrite or retry"):
            execution.probe_fingerprint(loaded, directory, "before", environment=loaded["environment"], runner=self.runner)

    def test_probe_rejects_inherited_tactics_before_launch(self):
        loaded = self.spec()
        with self.assertRaisesRegex(ValueError, "actual environment"):
            execution.probe_fingerprint(loaded, self.root, "before", environment={**loaded["environment"], "KATAGO_CUDA_NOGRAPH": "1"},
                                        runner=lambda *args, **kwargs: self.fail("unexpected subprocess"))

    def test_declared_q64_cannot_replace_actual_q32_or_conflicting_marker(self):
        _, directory, report = self.make_collection()
        path = directory / "worker.log"
        path.write_text(path.read_text().replace("tile=q64", "tile=q32"), encoding="utf-8")
        self.update_artifact(directory, report, "worker.log")
        with self.assertRaisesRegex(ValueError, "q64 attention"):
            execution.validate_collection(directory, report)

    def test_missing_and_conflicting_tune_markers_are_rejected(self):
        loaded = self.spec()
        log = self.int8_log_original.replace(self.fingerprint_original["model_sha256"], self.model_sha)
        hello = dict(model_sha256=self.model_sha, backend_info="backend=cudaint8backend; precision=W8A8-mixed; int8-scope=ffn; int8-min-ffn-width=0; quantization=w8a8-row-out-rne-v1")
        for altered in (log.replace("name=int8_gemm_tune", "name=removed_tune"),
                        log + "\n[cuda-tactic] name=int8_gemm_tune enabled=1 candidate_pool=16 default=0\n"):
            with self.subTest(log_tail=altered[-100:]):
                with self.assertRaisesRegex(ValueError, "tune0 marker"):
                    execution.actual_execution(loaded, self.fingerprint, hello, altered)

    def test_bound_artifacts_reject_path_escape_before_read(self):
        outside = self.root / "outside.json"
        outside.write_bytes(b"bound bytes")
        for filename in ("../outside.json", "sub/../../outside.json", str(outside)):
            with self.subTest(filename=filename):
                with self.assertRaisesRegex(ValueError, "unsafe execution artifact"):
                    execution.bound(self.root / "inside", dict(file=filename, sha256=execution.file_sha(outside)))

    def test_model_layer_count_and_imma_bit_are_actual_checks(self):
        loaded, directory, report = self.make_collection()
        original = (directory / "worker.log").read_text()
        for before, after, expected in (("int8_ffn_layers=45", "int8_ffn_layers=33", "layer count"),
                                        ("numerical_flags=0x200804", "numerical_flags=0x200800", "IMMA")):
            with self.subTest(marker=before):
                (directory / "worker.log").write_text(original.replace(before, after), encoding="utf-8")
                self.update_artifact(directory, report, "worker.log")
                with self.assertRaisesRegex(ValueError, expected):
                    execution.validate_collection(directory, report)

    def test_fp16_loader_marker_cannot_name_another_plan(self):
        _, directory, report = self.make_collection("cudabackend")
        path = directory / "worker.log"
        path.write_text(path.read_text().replace(self.plan["plan_id"], "other-plan"), encoding="utf-8")
        self.update_artifact(directory, report, "worker.log")
        with self.assertRaisesRegex(ValueError, "installation marker"):
            execution.validate_collection(directory, report)

    def test_fp16_effective_config_must_reference_bound_frozen_plan(self):
        _, directory, report = self.make_collection("cudabackend")
        path = directory / "worker.cfg"
        path.write_text(path.read_text().replace(str(directory / "runtime-cudaTacticPlan.json"), str(self.plan_path)), encoding="utf-8")
        self.update_artifact(directory, report, "worker.cfg")
        report["config_sha256"] = execution.file_sha(path)
        report["config_text"] = path.read_text()
        with self.assertRaisesRegex(ValueError, "outside its bound runtime plan"):
            execution.validate_collection(directory, report)

    def test_fingerprint_mutation_and_forged_saved_id_are_rejected(self):
        _, directory, report = self.make_collection()
        report["legacy_execution"]["observed_execution_id"] = "invented"
        with self.assertRaisesRegex(ValueError, "saved observed"):
            execution.validate_collection(directory, report)
        del report["legacy_execution"]["observed_execution_id"]
        (directory / "before-fingerprint/stdout.json").write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "hash/size"):
            execution.validate_collection(directory, report)

    def test_valid_different_before_after_fingerprints_do_not_match(self):
        loaded, directory, report = self.make_collection()
        receipt = report["legacy_execution"]["after"]
        value = deepcopy(self.fingerprint)
        value["sm_count"] += 1
        raw = execution.compact(value)
        (directory / receipt["stdout"]["file"]).write_bytes(raw)
        receipt["stdout"] = execution.artifact(directory, directory / receipt["stdout"]["file"])
        result_path = directory / receipt["result"]["file"]
        result = execution.strict_json(result_path.read_bytes())
        result["stdout_sha256"] = execution.sha(raw)
        result_path.write_bytes(execution.compact(result))
        receipt["result"] = execution.artifact(directory, result_path)
        with self.assertRaisesRegex(ValueError, "changed across"):
            execution.validate_collection(directory, report)

    def test_abba_arm_reconstructs_same_identity_and_rejects_wrong_environment(self):
        loaded, directory, report = self.make_collection()
        collection_identity = execution.validate_collection(directory, report)
        arm_dir = self.root / "arm"
        arm_dir.mkdir()
        (arm_dir / "worker.log").write_bytes((directory / "worker.log").read_bytes())
        arm = dict(environment=loaded["environment"], batch_capacity=8, protocol_capacity=32, concurrency=32, hello=report["hello"],
                   files=[dict(path=str(path), sha256=execution.file_sha(path)) for path in (self.binary, self.model, directory / "worker.cfg")],
                   artifacts=[execution.artifact(arm_dir, arm_dir / "worker.log")], legacy_execution={})
        for label in ("before", "after"):
            arm["legacy_execution"][label] = execution.probe_fingerprint(loaded, arm_dir, label, environment=loaded["environment"], runner=self.runner)
        result = execution.validate_arm(arm_dir, arm, directory, report)
        self.assertEqual(result["observed_execution_id"], collection_identity["observed_execution_id"])
        arm["environment"] = {}
        with self.assertRaisesRegex(ValueError, "ABBA environment"):
            execution.validate_arm(arm_dir, arm, directory, report)

    def test_holdout_rejected_before_fingerprint_or_worker_log_access(self):
        _, directory, report = self.make_collection()
        report["split"] = "holdout"
        (directory / "worker.log").unlink()
        with self.assertRaisesRegex(ValueError, "holdout"):
            execution.validate_collection(directory, report)


if __name__ == "__main__":
    unittest.main()
