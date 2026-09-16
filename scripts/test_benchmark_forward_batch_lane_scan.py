"""CPU tests of the scan protocol and independent event/wall arithmetic."""
import contextlib
import copy
import hashlib
import io
import math
import shlex
import unittest
from unittest.mock import patch

import benchmark_forward_batch_lane_scan as scan

LOG = "\n".join((
    "[cuda-tactic] name=gemm_layout launch=tn effective=tn",
    "[cuda-tactic] name=dual_ffn launch=fused effective=1",
    "[cuda-tactic] name=attention requested=fa2 launch=fa2 tile=q64-serial",
    "[cuda-tactic] name=cublaslt_rank engine=heuristic",
    "[cuda-tactic] name=gemm kind=residual engine=cublaslt",
))


def fixture(batch=12, lanes=2):
    # One outlier makes mean(event) and median(event) intentionally different;
    # the two lane wall times also differ from the common wall envelope.
    samples = [dict(handle=1, iterations=4, elapsed_ms=20.0, per_batch_ms=5.0,
                    nn_evals_per_s=batch * 200.0, cuda_event_ms=[1.0, 9.0, 2.0, 4.0],
                    cuda_event_median_ms=3.0, cuda_event_median_nn_evals_per_s=batch * 1000 / 3)]
    if lanes == 2:
        samples.append(dict(handle=2, iterations=4, elapsed_ms=120.0, per_batch_ms=30.0,
                    nn_evals_per_s=batch * 1000 / 30, cuda_event_ms=[5.0, 100.0, 7.0, 5.0],
                    cuda_event_median_ms=6.0, cuda_event_median_nn_evals_per_s=batch * 1000 / 6))
    build = dict(kernel_build_id="sha256:" + "f" * 64, cuda_compiler="13.3.73", cutlass_version="3.9.2",
                 cutlass_commit="fixture", fp16_encoding_revision=1, cublaslt_version=130600,
                 capabilities=dict(dual_ffn=True, attention_q64_serial=True))
    description = {"kind": "empty", "inputs_version": 7, "board": [19, 19],
        "spatial": "B,22,19,19 NCHW f32", "global": "B,19 f32", "input_hash_encoding": scan.HASH_ENCODING,
        "position_generation": scan.POSITION_GENERATION, "rules": copy.deepcopy(scan.parity.EMPTY_RULES),
        "encore_phase": 0, "effective_symmetry": 0, "policy_optimism": 0.0,
        "misc_params": copy.deepcopy(scan.parity.EMPTY_MISC)}
    row = dict(batch=batch, handles=samples, wall_elapsed_ms=20.0 if lanes == 1 else 140.0,
        wall_nn_evals_per_s=batch * 200.0 if lanes == 1 else batch * 400 / 7,
        sum_median_nn_evals_per_s=batch * 1000 / 3 if lanes == 1 else batch * 500.0,
        spatial_input_sha256=hashlib.sha256(f"spatial B{batch}".encode()).hexdigest(),
        global_input_sha256=hashlib.sha256(f"global B{batch}".encode()).hexdigest())
    return dict(command="nnbench", mode="kernel", timing="cuda-event", kernel_only=True,
        model=scan.MODEL, model_sha256=scan.parity.MODEL_SHA, input="empty", iterations=4, warmup=80,
        requested_batches=[batch], handles=list(range(1, lanes + 1)), device_target="sm_120",
        build=build, tactics=dict(scan.TACTICS), input_description=description, results=[row])


def validate(data, batch=12, lanes=2, log=LOG):
    return scan.validate_payload(data, log, batch=batch, lanes=lanes, iterations=4, warmup=80)


class ForwardBatchLaneScanTests(unittest.TestCase):
    def test_fixed_cases_and_lane_ids_are_not_lane_count_arguments(self):
        self.assertEqual(scan.CASES, ((12, 1), (12, 2), (13, 1), (13, 2), (14, 1),
                                     (14, 2), (15, 1), (15, 2), (16, 1), (16, 2)))
        for batch, lanes in scan.CASES:
            command = scan.build_command("/tmp/frozen binary", batch, lanes, warmup=80, iterations=200,
                inherited=["KATAGO_CUDA_RESIDUAL_ALGO", "KATAGO_CUDA_CUBLASLT_RANK", "UNRELATED"])
            words = shlex.split(command[-1])
            self.assertEqual(words[words.index("--handles") + 1], "1" if lanes == 1 else "1,2")
            self.assertEqual(words[words.index("--batch") + 1], str(batch))
            self.assertEqual(words[words.index("--mode") + 1], "kernel")
            self.assertEqual(words[words.index("--input") + 1], "empty")
            self.assertNotIn("UNRELATED", words)
            self.assertIn("/tmp/frozen binary", words)
            key = "KATAGO_CUDA_RESIDUAL_ALGO"
            self.assertEqual(words.count(key + "=heuristic"), 1)
            self.assertLess(words.index(key), words.index(key + "=heuristic"))
            self.assertEqual(words[words.index(key) - 1], "-u")
        for case in ((11, 1), (17, 1), (12, 3), (12, True)):
            with self.assertRaises(ValueError):
                scan.case_spec(*case)

    def test_default_identity_is_pre_registered_and_parameters_bounded(self):
        args = scan.parse_args(["--output", "unused"])
        self.assertEqual((args.binary, args.binary_sha256), (scan.BINARY, scan.BINARY_SHA))
        self.assertEqual((args.warmup, args.iterations), (80, 200))
        for extra in (["--binary", "relative"], ["--binary-sha256", "f" * 63], ["--warmup", "0"],
                      ["--iterations", "99"], ["--iterations", "10001"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                scan.parse_args(["--output", "unused", *extra])

    def test_median_sum_and_wall_are_independent_and_raw_events_unchanged(self):
        data = fixture()
        before = copy.deepcopy(data)
        result = validate(data)
        self.assertEqual(result["lane_median_ms"], [3.0, 6.0])
        self.assertEqual(result["event_median_rate_sum"], 6000.0)
        self.assertAlmostEqual(result["forward_wall_rate"], 96 / 0.140)
        self.assertNotEqual(result["event_median_rate_sum"], result["forward_wall_rate"])
        self.assertEqual(data, before)
        single = validate(fixture(12, 1), lanes=1)
        self.assertEqual(single["event_median_rate_sum"], 4000.0)
        self.assertEqual(single["forward_wall_rate"], 2400.0)

    def test_all_physical_batches_accept_only_the_requested_lane_payload(self):
        for batch, lanes in scan.CASES:
            validate(fixture(batch, lanes), batch, lanes)
            bad = fixture(batch, lanes)
            bad["requested_batches"] = [batch + 1]
            with self.assertRaises(ValueError):
                validate(bad, batch, lanes)
        bad = fixture(12, 1)
        bad["handles"] = [2]
        bad["results"][0]["handles"][0]["handle"] = 2
        with self.assertRaisesRegex(ValueError, "lane request"):
            validate(bad, lanes=1)

    def test_tampered_statistics_samples_and_timing_envelopes_reject(self):
        mutations = [
            lambda r: r.update(sum_median_nn_evals_per_s=9999.0),
            lambda r: r.update(wall_nn_evals_per_s=6000.0),
            lambda r: r.update(wall_elapsed_ms=10.0),
            lambda r: r["handles"][0].update(cuda_event_median_ms=4.0),
            lambda r: r["handles"][0]["cuda_event_ms"].pop(),
            lambda r: r["handles"][0]["cuda_event_ms"].__setitem__(0, math.nan),
            lambda r: r["handles"][1].update(handle=1),
            lambda r: r["handles"][0].update(handle=True),
            lambda r: r["handles"][0].update(nn_evals_per_s=1.0),
            lambda r: r["handles"][0].update(per_batch_ms=1.0),
        ]
        for i, mutate in enumerate(mutations):
            data = fixture(); mutate(data["results"][0])
            with self.subTest(i=i), self.assertRaises(ValueError):
                validate(data)

    def test_input_tactic_and_build_drift_reject_instead_of_ignoring_environment(self):
        mutations = [
            lambda d: d.update(model="/wrong/model.bin.gz"),
            lambda d: d.update(model_sha256="0" * 64),
            lambda d: d["tactics"].pop("KATAGO_CUDA_RESIDUAL_ALGO"),
            lambda d: d["tactics"].update(KATAGO_CUDA_RESIDUAL_ALGO="tf3_5070ti_r1"),
            lambda d: d["tactics"].update(KATAGO_CUDA_CUBLASLT_RANK="time"),
            lambda d: d["build"].update(fp16_encoding_revision=0),
            lambda d: d["build"].update(fp16_encoding_revision=True),
            lambda d: d["build"].update(cublaslt_version=130500),
            lambda d: d["input_description"].update(position_generation="all-zero tensors"),
            lambda d: d["input_description"]["rules"].update(komi=0),
            lambda d: d["results"][0].update(spatial_input_sha256="invalid"),
        ]
        for i, mutate in enumerate(mutations):
            data = fixture(); mutate(data)
            with self.subTest(i=i), self.assertRaises(ValueError):
                validate(data)
        for line in LOG.splitlines():
            with self.assertRaisesRegex(ValueError, "required path"):
                validate(fixture(), log=LOG.replace(line, ""))
        with self.assertRaisesRegex(ValueError, "unexpected residual preset"):
            validate(fixture(), log=LOG + "\n[cuda-tactic] name=residual_algo engine=tf3_5070ti_init_r1")

    def test_same_batch_inputs_and_exact_loaded_build_are_bound_across_lane_counts(self):
        inputs = {}; data = fixture(12, 1)
        scan.bind_input_and_build(data, validate(data, lanes=1), batch=12, build=data["build"], inputs=inputs)
        second = fixture(12, 2)
        scan.bind_input_and_build(second, validate(second), batch=12, build=data["build"], inputs=inputs)
        second["results"][0]["global_input_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "input hashes"):
            scan.bind_input_and_build(second, validate(second), batch=12, build=data["build"], inputs=inputs)
        second = fixture(13, 1)
        scan.bind_input_and_build(second, validate(second, 13, 1), batch=13, build=data["build"], inputs=inputs)
        second["build"]["cutlass_commit"] = "different"
        with self.assertRaisesRegex(ValueError, "build/library changed"):
            scan.bind_input_and_build(second, validate(second, 13, 1), batch=13, build=data["build"], inputs=inputs)

    def test_rankings_expose_metric_conflicts_without_discarding_or_certifying_samples(self):
        runs = []
        for i, case in enumerate(scan.CASES):
            runs.append({**scan.case_spec(*case), "completion": "COMPLETE",
                         "metrics": dict(event_median_rate_sum=1000 + i, forward_wall_rate=2000 - i)})
        before = copy.deepcopy(runs)
        summary = scan.summarize(runs)
        self.assertEqual(summary["event_ranking"][0]["case_id"], "B16-S2")
        self.assertEqual(summary["forward_wall_ranking"][0]["case_id"], "B12-S1")
        self.assertTrue(summary["rankings_disagree"])
        self.assertTrue(summary["top_choice_disagrees"])
        self.assertEqual(len(summary["rank_comparison"]), 10)
        self.assertEqual(runs, before)
        for bad in (runs[:-1], list(reversed(runs)), [runs[0], *runs[2:], runs[-1]]):
            with self.assertRaisesRegex(ValueError, "pre-registered order"):
                scan.summarize(bad)

    def test_gpu_preflight_rejects_other_device_and_model_without_launching_tools(self):
        fingerprint = dict(gpu_name="NVIDIA GeForce RTX 5070 Ti", architecture="sm_120",
            compute_capability="12.0", sm_count=70, l2_cache_bytes=50331648,
            model_sha256=scan.parity.MODEL_SHA, backend_build=fixture()["build"])
        with patch.object(scan.subprocess, "run", side_effect=AssertionError("CPU test launched subprocess")):
            scan.validate_fingerprint(fingerprint)
            for key, value in [("sm_count", 84), ("model_sha256", "0" * 64), ("l2_cache_bytes", 67108864)]:
                with self.assertRaises(ValueError):
                    scan.validate_fingerprint({**fingerprint, key: value})


if __name__ == "__main__":
    unittest.main()
