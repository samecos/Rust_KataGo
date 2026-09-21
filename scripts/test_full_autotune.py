"""Fail-closed selection/publication tests; no GPU work."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import autotune_reference as gold
import full_autotune as full
import tune_runtime as runtime


def args(**overrides):
    values = dict(mode="local", groups=list(full.GROUPS), batches=[8, 4, 12, 14, 16], threads=[8, 12],
                  capacity=32, max_visits=800, timeout=10, min_improvement=.01, max_spread=.05,
                  binary=Path("engine"), model=Path("model"))
    values.update(overrides)
    return SimpleNamespace(**values)


def fingerprint():
    fp = {key: "fixture" for key in (*runtime.TARGET_KEYS, "architecture")}
    fp["backend_build"] = dict(capabilities=dict(dual_ffn=False, attention_q64=False, attention_q64_serial=False),
                               fp16_encoding_revision=1, kernel_build_id="exact")
    return fp


class FullAutotuneTests(unittest.TestCase):
    def test_pruned_architecture_excludes_fixed_shape_tactics(self):
        fp = fingerprint()
        fp["backend_build"]["capabilities"]["dual_ffn"] = True
        fp["model_architecture"] = dict(supports_fixed_ffn_tactics=False, supports_k384_layout=False)
        for group in ("dual_ffn", "fusion", "splitk", "layout"):
            self.assertEqual(full.candidates(group, full.state(), fp, args()), [])
        self.assertTrue(full.candidates("rms", full.state(), fp, args()))
        self.assertTrue(full.candidates("attention", full.state(), fp, args()))

    def test_capability_filtering(self):
        fp, baseline = fingerprint(), full.state()
        self.assertEqual(len(full.candidates("dual_ffn", baseline, fp, args())), 1)
        self.assertEqual(len(full.candidates("attention", baseline, fp, args())), 2)
        fp["backend_build"]["capabilities"] = dict(dual_ffn=True, attention_q64=True, attention_q64_serial=True)
        self.assertEqual(len(full.candidates("dual_ffn", baseline, fp, args())), 2)
        self.assertEqual(len(full.candidates("attention", baseline, fp, args())), 4)

    def test_catalog_and_native_batch_bounds(self):
        baseline = full.state()
        for group in full.GROUPS:
            with patch.object(full, "specialized_states", return_value=[]):
                for item in full.candidates(group, baseline, fingerprint(), args()):
                    self.assertTrue(1 <= item["batch"] <= 16)
                    self.assertIn(item["lanes"], (1, 2))
        self.assertEqual(baseline, full.state())
        for value in ("0", "17", "8,8"):
            with self.assertRaises(Exception):
                full.positive_list(value)
        with self.assertRaises(Exception):
            full.group_list("made-up-group")

    def test_strict_scheduling_dependencies(self):
        strict = full.changed(full.state(), {"ATTN": "fa4-strict-b14-r1", "NOGRAPH": "1"}, batch=14)
        for item in full.candidates("reschedule", strict, fingerprint(), args()):
            self.assertEqual(item["batch"], 14)
            self.assertEqual(item["tactics"]["KATAGO_CUDA_NOGRAPH"], "1")

    def test_plan_has_target_identity_and_does_not_alias_input(self):
        fp, baseline = fingerprint(), full.state()
        plan = full.new_plan(fp, baseline, "test-plan", {"status": "UNVALIDATED_CANDIDATE"})
        self.assertEqual(plan["target"]["model_sha256"], fp["model_sha256"])
        self.assertEqual(plan["target"]["max_batch_size"], 8)
        plan["backend_build"]["kernel_build_id"] = "mutated"
        plan["apply"]["tactic_overrides"]["KATAGO_CUDA_DUALFFN"] = "1"
        self.assertEqual(fp["backend_build"]["kernel_build_id"], "exact")
        self.assertEqual(baseline["tactics"]["KATAGO_CUDA_DUALFFN"], "0")

    def test_path_check_does_not_accept_q64_serial_as_q64(self):
        value = full.changed(full.state(), {"ATTN_TILE": "q64"})
        log = "\n".join(full.expected_markers(value))
        full.verify_paths(log, value)
        with self.assertRaises(ValueError):
            full.verify_paths(log.replace("tile=q64", "tile=q64-serial"), value)
        with self.assertRaises(ValueError):
            full.verify_paths(log + "\nprobe=fail", value)

    def test_performance_cannot_run_without_numeric(self):
        with tempfile.TemporaryDirectory() as directory:
            tuner = full.FullTuner(args(), Path(directory))
            with self.assertRaises(RuntimeError):
                tuner.measure(dict(name="untested"))

    def test_compact_dualffn_requires_probe_and_its_own_execution_marker(self):
        value = full.changed(full.state(), {"DUALFFN": "1", "FFN_COMPACT_R1": "1"})
        log = "\n".join(full.expected_markers(value))
        self.assertNotIn("name=dual_ffn launch=fused", log)
        full.verify_paths(log, value)
        for marker in ("name=dual_ffn requested=1 compiled=1 probe=pass",
                       "name=ffn_compact stage=execute effective=1"):
            with self.subTest(missing=marker), self.assertRaises(ValueError):
                full.verify_paths(log.replace(marker, ""), value)
        with self.assertRaises(ValueError):
            full.verify_paths(log, full.changed(value, {"FFN_COMPACT_R1": "0"}))

    def test_small_tile_names_match_executor_log_not_cuda_symbol_suffix(self):
        # Actual cuda_exec.rs trace names omit a _kernel suffix.
        base_log = "\n".join((
            "[cuda-tactic] name=dual_ffn launch=unfused effective=0",
            "[cuda-tactic] name=graph requested=graph launch=graph",
            "[cuda-tactic] name=rms launch=w4",
            "[cuda-tactic] name=fusion mode=none",
            "[cuda-tactic] name=attention requested=fa2 launch=fa2 effective=fa2 tile=q128",
            "[cuda-tactic] name=gemm kind=f16out engine=handwritten tile=hgemm_t64_f16out"))
        for key, tile in (("T32", "hgemm_t32"), ("T64N32", "hgemm_t64n32")):
            value = full.changed(full.state(), {"CUBLASLT": "0", key: "1"})
            full.verify_paths(base_log + "\n[cuda-tactic] name=gemm kind=residual engine=handwritten tile=" + tile, value)
            with self.assertRaises(ValueError):
                full.verify_paths(base_log, value)

    def test_candidate_process_failure_keeps_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            tuner = full.FullTuner(args(), Path(directory))
            with patch.object(tuner, "measure", side_effect=[100, RuntimeError("candidate crashed")]):
                self.assertFalse(tuner.compare_candidate({"name": "a"}, {"name": "b"}, "test"))
            self.assertEqual(tuner.report["comparisons"][-1]["status"], "CANDIDATE_FAILED")

    def test_baseline_failure_is_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            tuner = full.FullTuner(args(), Path(directory))
            with patch.object(tuner, "measure", side_effect=RuntimeError("baseline crashed")), self.assertRaises(RuntimeError):
                tuner.compare_candidate({"name": "a"}, {"name": "b"}, "test")

    def test_noisy_improvement_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            tuner = full.FullTuner(args(), Path(directory))
            with patch.object(tuner, "measure", side_effect=[100, 110, 125, 100]), contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(tuner.compare_candidate({"name": "a"}, {"name": "b"}, "test"))

    def test_failed_final_smoke_removes_publishable_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tuner = full.FullTuner(args(), root)
            tuner.fingerprint = fingerprint()
            tuner.numeric_fingerprint = dict(cases=128)
            tuner.report.update(binary_sha256="fake", reference_sha256="fake")
            candidate = tuner.materialize(full.state())
            with patch.object(tuner, "smoke_local", side_effect=RuntimeError("bad final config")), self.assertRaises(RuntimeError):
                tuner.publish(candidate, candidate)
            self.assertFalse((root / "result").exists())
            self.assertTrue((root / "staging/plan.json").is_file())
            self.assertFalse((root / "staging/READY").exists())

    def test_success_publishes_plan_cfg_and_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tuner = full.FullTuner(args(groups=["dual_ffn"]), root)
            tuner.fingerprint = fingerprint()
            tuner.numeric_fingerprint = dict(cases=128)
            tuner.report.update(binary_sha256="fake", reference_sha256="fake")
            candidate = tuner.materialize(full.state())
            with patch.object(tuner, "smoke_local"), contextlib.redirect_stdout(io.StringIO()):
                tuner.publish(candidate, candidate)
            self.assertTrue((root / "result/READY").is_file())
            plan = json.loads((root / "result/plan.json").read_text())
            self.assertEqual(plan["selection"]["search_scope"], "CUSTOM_GROUPS")
            self.assertEqual(plan["selection"]["selection_status"], "BASELINE_RETAINED")
            self.assertIn(str(root / "result/plan.json").replace("\\", "/"), (root / "result/rustgo.cfg").read_text())


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from worker_protocol_tools import Protocol
            from compare_worker_outputs import build_requests
        except ImportError:
            raise unittest.SkipTest("reference tests need the tuning grpc/protobuf environment")
        cls.protocol = Protocol(runtime.ROOT / "crates/kata_worker/proto/worker.proto")
        cls.reference = gold.read_reference(gold.load_bundled(gold.BUNDLED_MODEL))
        fixture = json.loads((runtime.ROOT / "scripts/fixtures/worker_positions.json").read_text(encoding="utf-8"))
        cls.requests = build_requests(cls.protocol.pb, fixture, gold.BUNDLED_MODEL, 0)
        cls.fp = gold.fingerprints(cls.requests, gold.BUNDLED_MODEL)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "protocol"):
            cls.protocol.close()

    def test_bundled_full_corpus(self):
        gold.validate_reference(self.reference, self.fp, self.requests, self.protocol)

    def test_rejects_false_provenance_fp16_truncation_identity_and_nan(self):
        mutations = (
            lambda r: r.update(worker="rust"),
            lambda r: r.update(config_text="cudaUseFP16 = true"),
            lambda r: r["results"].pop(),
            lambda r: r["results"][0]["result"].update(model_sha256="wrong"),
            lambda r: r["results"][0]["result"]["output"].update(white_win_prob="NaN"),
        )
        for mutate in mutations:
            reference = copy.deepcopy(self.reference)
            mutate(reference)
            with self.subTest(mutate=mutate), self.assertRaises((ValueError, AssertionError)):
                gold.validate_reference(reference, self.fp, self.requests, self.protocol)

    def test_rejects_changed_fixture_or_model(self):
        for key in ("model_sha256", "fixtures_sha256", "schema_sha256", "requests_sha256"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                gold.validate_reference(self.reference, dict(self.fp, **{key: "different"}), self.requests, self.protocol)


if __name__ == "__main__":
    unittest.main()
