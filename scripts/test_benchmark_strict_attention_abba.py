"""CPU-only measurement/identity gates; no subprocess, CUDA or build is run."""
import argparse
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

import benchmark_strict_attention_abba as abba

MODEL = Path("test-tf3.bin.gz")
BUILD = dict(fp16_encoding_revision=1, cublaslt_version=130600, cuda_compiler="13.3.73",
             strict_attention_artifact=abba.ASSET)
COMMON_LOG = "\n".join([
    "[cuda-tactic] name=gemm_layout launch=tn effective=tn requested=tn k=384",
    "[cuda-tactic] name=dual_ffn launch=fused effective=1",
    "[cuda-tactic] name=cublaslt_rank engine=heuristic",
    *[f"[cuda-tactic] name=gemm kind={k} engine=cublaslt" for k in ("plain", "f16out", "residual")],
])
BASELINE = "[cuda-tactic] name=attention requested=fa2 launch=fa2 effective=fa2 tile=q64-serial"
CANDIDATE = (f"[cuda-tactic] name=attention requested={abba.PRESET} launch=strict-aot effective={abba.PRESET} "
             f"physical_batch=14 artifact={abba.ASSET} identity=matched rope=exact-qk-only v=original-packed abi=1")


def fixture(flavor="candidate"):
    handles = []
    for lane, values, elapsed in [(1, [2.0, 4.0] * 500, 4000.0), (2, [3.0, 5.0] * 500, 4500.0)]:
        median = statistics.median(values)
        handles.append(dict(handle=lane, iterations=1000, elapsed_ms=elapsed, per_batch_ms=elapsed / 1000,
                            nn_evals_per_s=14_000_000 / elapsed, cuda_event_ms=values,
                            cuda_event_median_ms=median, cuda_event_median_nn_evals_per_s=14_000 / median))
    return dict(command="nnbench", mode="kernel", timing="cuda-event", kernel_only=True,
                model=str(MODEL), model_sha256=abba.parity.MODEL_SHA, input="empty", warmup=80, iterations=1000,
                requested_batches=[14], handles=[1, 2], device_target="sm_120", build=copy.deepcopy(BUILD),
                tactics=abba.tactics(flavor),
                input_description=dict(kind="empty", inputs_version=7, board=[19, 19], rules=abba.parity.EMPTY_RULES,
                                       misc_params=abba.parity.EMPTY_MISC, encore_phase=0, effective_symmetry=0, policy_optimism=0.0),
                results=[dict(batch=14, handles=handles, spatial_input_sha256="a" * 64, global_input_sha256="b" * 64,
                              wall_elapsed_ms=5000.0, wall_nn_evals_per_s=5600.0,
                              sum_median_nn_evals_per_s=sum(h["cuda_event_median_nn_evals_per_s"] for h in handles))])


def validate(data=None, flavor="candidate", log=None):
    return abba.validate_result(data or fixture(flavor), log or (COMMON_LOG + "\n" + (CANDIDATE if flavor == "candidate" else BASELINE)),
                                flavor, MODEL, BUILD)


class StrictAttentionTests(unittest.TestCase):
    def test_candidate_needs_no_q64_marker(self):
        self.assertNotIn("q64", CANDIDATE)
        result = validate()
        self.assertAlmostEqual(result["event_rate_sum"], 14000 / 3 + 14000 / 4)
        self.assertEqual(result["wall_rate"], 5600)

    def test_baseline_requires_serial_q64(self):
        validate(flavor="baseline")
        with self.assertRaises(ValueError):
            validate(flavor="baseline", log=COMMON_LOG + "\n" + BASELINE.replace("q64-serial", "q128"))

    def test_candidate_rejects_fallback(self):
        with self.assertRaises(ValueError):
            validate(log=COMMON_LOG + "\n" + CANDIDATE + " fallback=q64-serial")

    def test_candidate_rejects_extra_legacy_scope(self):
        with self.assertRaises(ValueError):
            validate(log=COMMON_LOG + "\n" + CANDIDATE + "\n" + BASELINE)

    def test_candidate_rejects_wrong_asset_batch_rope_and_v(self):
        for old, new in [(abba.ASSET, "sha256:" + "0" * 64), ("physical_batch=14", "physical_batch=16"),
                         ("rope=exact-qk-only", "rope=unrolled"), ("v=original-packed", "v=copied")]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                validate(log=COMMON_LOG + "\n" + CANDIDATE.replace(old, new))

    def test_rejects_layout_and_residual_candidate_contamination(self):
        for log in [COMMON_LOG.replace("effective=tn", "effective=nn_k384_b16"),
                    COMMON_LOG + "\n[cuda-tactic] name=residual_algo engine=tf3_5070ti_init_r1"]:
            with self.subTest(log=log), self.assertRaises(ValueError):
                validate(log=log + "\n" + CANDIDATE)

    def test_effective_tactics_must_match_exactly(self):
        for key, value in [("KATAGO_CUDA_RMS", "tuned"), ("KATAGO_CUDA_ATTN", "fa2")]:
            data = fixture()
            data["tactics"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(data)

    def test_all_2000_original_events_required(self):
        data = fixture()
        data["results"][0]["handles"][0]["cuda_event_ms"].pop()
        with self.assertRaises(ValueError):
            validate(data)

    def test_raw_samples_recompute_median(self):
        data = fixture()
        data["results"][0]["handles"][0]["cuda_event_ms"] = [1.0] * 501 + [100.0] * 499
        with self.assertRaises(ValueError):
            validate(data)
        h = data["results"][0]["handles"][0]
        h["cuda_event_median_ms"] = 1.0
        h["cuda_event_median_nn_evals_per_s"] = 14000.0
        data["results"][0]["sum_median_nn_evals_per_s"] = 17500.0
        self.assertEqual(validate(data)["event_rate_sum"], 17500.0)

    def test_nonfinite_or_nonpositive_events_rejected(self):
        for bad in (float("nan"), float("inf"), 0, -1, True):
            data = fixture()
            data["results"][0]["handles"][0]["cuda_event_ms"][0] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate(data)

    def test_lane_and_full_wall_arithmetic(self):
        for key in ("wall_elapsed_ms", "wall_nn_evals_per_s"):
            data = fixture()
            data["results"][0][key] *= 1.1
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(data)
        for key in ("elapsed_ms", "per_batch_ms", "nn_evals_per_s"):
            data = fixture()
            data["results"][0]["handles"][0][key] *= 1.1
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(data)

    def test_build_asset_and_encoder_binding(self):
        for key, value in [("fp16_encoding_revision", None), ("strict_attention_artifact", "wrong"),
                           ("cuda_compiler", "other-platform")]:
            data = fixture()
            data["build"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(data)

    def test_input_hash_and_physical_shape_required(self):
        for key, value in [("spatial_input_sha256", None), ("global_input_sha256", "wrong"), ("batch", 16)]:
            data = fixture()
            data["results"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(data)

    def runs(self, rates):
        return [dict(flavor=flavor, event_rate_sum=rate, wall_rate=rate) for flavor, rate in zip(abba.ORDER, rates)]

    def test_stable_gain_uses_two_arm_geomeans(self):
        summary = abba.summarize(self.runs([1000, 1020, 1020, 1000]))
        self.assertEqual(summary["performance_status"], "PASS_GAIN_GATE")
        self.assertAlmostEqual(summary["metrics"]["event_rate_sum"]["candidate_over_baseline_ratio"], 1.02)

    def test_small_stable_gain_rejected(self):
        self.assertEqual(abba.summarize(self.runs([1000, 1005, 1005, 1000]))["performance_status"], "REJECT_BELOW_MINIMUM_GAIN")

    def test_593_vs_1134_is_diagnostic_only(self):
        summary = abba.summarize(self.runs([593, 1108, 1126, 1134]))
        self.assertEqual(summary["performance_status"], "UNSTABLE_DIAGNOSTIC_ONLY")
        self.assertNotIn("candidate_over_baseline_ratio", summary["metrics"]["event_rate_sum"])
        self.assertIn("diagnostic_only_ratio", summary["metrics"]["event_rate_sum"])

    def test_full_abba_order_required(self):
        for runs in [self.runs([1000, 1020]), list(reversed(self.runs([1000, 1020, 1020, 1000])))[1:],
                     [dict(flavor="candidate", event_rate_sum=1000, wall_rate=1000)] * 4]:
            with self.subTest(runs=runs), self.assertRaises(ValueError):
                abba.summarize(runs)

    def test_environment_scrubs_all_inherited_katago_keys(self):
        actual = abba.environment("candidate", "win", {"PATH": "keep", "KATAGO_CUDA_PROFILE": "1", "katago_bad": "1"})
        self.assertEqual(actual, {"PATH": "keep", **abba.tactics("candidate")})

    def test_command_is_native_direct_no_wrapper(self):
        cmd = abba.command("native.exe", "model.bin.gz")
        self.assertEqual(cmd[:2], ["native.exe", "nnbench"])
        self.assertNotIn("wsl", cmd)
        self.assertEqual(cmd[cmd.index("--batch") + 1], "14")

    def test_worker_failed_report_cannot_start_gpu(self):
        with patch.object(abba, "read", return_value={"schema": 1, "status": "FAIL", "pass": False}), \
                patch.object(abba, "child", side_effect=AssertionError("must not launch")):
            with self.assertRaises(ValueError):
                abba.worker_evidence(Path("failed.json"), Path("worker.exe"), {})

    def test_worker_wrong_binary_or_platform_rejected(self):
        report = dict(schema=1, status="PASS_BOTH_WORKER_CPP_FP32", pass_=True)
        report.update({"pass": True, "binary_sha256": "windows-sha", "binary": {"sha256": "windows-sha"}})
        with patch.object(abba, "read", return_value=report), patch.object(abba, "sha", return_value="wsl-sha"):
            with self.assertRaisesRegex(ValueError, "another binary/platform"):
                abba.worker_evidence(Path("report.json"), Path("worker"), {})

    def test_mandatory_both_numeric_reports(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            abba.parse_args(["run", "--binary", "main.exe", "--numeric-report", "native.json", "--output", "new"])

    def test_fresh_output_refuses_existing_evidence_before_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = argparse.Namespace(output=Path(temporary))
            with patch.object(abba, "preflight", side_effect=AssertionError("must not execute")), self.assertRaises(FileExistsError):
                abba.run(args)

    def test_strict_json_rejects_duplicate_and_nonfinite(self):
        for text in ['{"command":"nnbench","command":"nnbench"}', '{"command":"nnbench","x":NaN}']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                abba.parse_nnbench(text)
        self.assertEqual(abba.parse_nnbench("log\n" + json.dumps(fixture()))["command"], "nnbench")


if __name__ == "__main__":
    unittest.main()
