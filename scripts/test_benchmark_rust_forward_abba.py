"""CPU-only checks for residual ABBA controls; no subprocess/GPU is launched."""
import contextlib
import copy
import io
import json
import math
import shlex
import statistics
import unittest

import benchmark_rust_forward_abba as abba


BASE_LOG = "\n".join((
    "[cuda-tactic] name=gemm_layout launch=tn effective=tn",
    "[cuda-tactic] name=dual_ffn launch=fused effective=1",
    "[cuda-tactic] name=attention requested=fa2 launch=fa2 tile=q64-serial",
))


def preset_log(batch, residual_algo=abba.PRESET):
    shapes = [(5054, 1152, 1)] if batch == 14 else [(5776, 1152, 3), (5776, 384, 2)]
    initialized = residual_algo == abba.INIT_PRESET
    construction = "construction=public_init " if initialized else ""
    index_field = "source_filtered_index" if initialized else "filtered_index"
    return "\n".join(
        f"[cuda-tactic] name=residual_algo engine={residual_algo} {construction}m={m} n=384 k={k} "
        f"layout=tn output=f32 beta=1 {index_field}={index} cublaslt_version=130600 algorithm_identity=matched"
        for m, k, index in shapes
    )


def fixture(batch, residual_algo=abba.PRESET):
    handles = []
    for lane, values in [(1, [2.0, 4.0, 2.0, 4.0]), (2, [3.0, 5.0, 3.0, 5.0])]:
        median = statistics.median(values)
        handles.append(dict(handle=lane, iterations=4, cuda_event_ms=values,
                            cuda_event_median_ms=median,
                            cuda_event_median_nn_evals_per_s=batch * 1000.0 / median))
    return dict(command="nnbench", mode="kernel", timing="cuda-event", kernel_only=True,
                model=abba.MODEL, model_sha256=abba.parity.MODEL_SHA, input="empty", warmup=80,
                iterations=4, requested_batches=[batch], handles=[1, 2], device_target="sm_120",
                build=dict(cuda_compiler="13.3.73", fp16_encoding_revision=1, cublaslt_version=130600),
                tactics=abba.tactics_for(residual_algo),
                input_description=dict(kind="empty", inputs_version=7, board=[19, 19],
                                       rules=copy.deepcopy(abba.parity.EMPTY_RULES), encore_phase=0,
                                       effective_symmetry=0, policy_optimism=0.0,
                                       misc_params=copy.deepcopy(abba.parity.EMPTY_MISC)),
                results=[dict(batch=batch, handles=handles, spatial_input_sha256="a" * 64,
                              global_input_sha256="b" * 64, wall_nn_evals_per_s=1234.0,
                              sum_median_nn_evals_per_s=sum(h["cuda_event_median_nn_evals_per_s"] for h in handles))])


def validate(data, batch, residual_algo=abba.PRESET, log=None):
    if log is None:
        log = BASE_LOG + ("\n" + preset_log(batch, residual_algo) if residual_algo != "heuristic" else "")
    return abba.validate_result(data, log, batch=batch, iterations=4, warmup=80,
                                tactics=abba.tactics_for(residual_algo))


class RustForwardAbbaTests(unittest.TestCase):
    def test_cli_defaults_explicit_choices_and_batch(self):
        base = ["--baseline", "/tmp/rust", "--output", "unused"]
        args = abba.parse_args(base)
        self.assertEqual((args.batch, args.baseline_residual_algo, args.candidate_residual_algo),
                         (14, "heuristic", "heuristic"))
        args = abba.parse_args(base + ["--batch", "16", "--candidate-residual-algo", abba.PRESET])
        self.assertEqual((args.batch, args.candidate_residual_algo), (16, abba.PRESET))
        args = abba.parse_args(base + ["--baseline-residual-algo", abba.PRESET,
                                      "--candidate-residual-algo", abba.INIT_PRESET])
        self.assertEqual((args.baseline_residual_algo, args.candidate_residual_algo),
                         (abba.PRESET, abba.INIT_PRESET))
        for extra in (["--batch", "15"], ["--candidate-residual-algo", "time"],
                      ["--maximum-relative-spread", "nan"], ["--maximum-relative-spread", "0.11"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                abba.parse_args(base + extra)

    def test_command_explicitly_sets_each_preset_after_environment_cleanup(self):
        binary = "/tmp/with space/katago-rs"
        inherited = [abba.RESIDUAL_KEY, "KATAGO_CUDA_NOGRAPH", "KATAGO_CUDA_PROFILE", "UNRELATED"]
        for residual in abba.RESIDUAL_CHOICES:
            command = abba.build_command(binary, abba.tactics_for(residual), batch=16,
                                         warmup=80, iterations=1000, inherited=inherited)
            self.assertEqual(command[:6], ["wsl", "-d", "Ubuntu-24.04", "--", "bash", "-lc"])
            words = shlex.split(command[-1])
            self.assertIn(binary, words)
            assignment = f"{abba.RESIDUAL_KEY}={residual}"
            self.assertEqual(words.count(assignment), 1)
            self.assertGreater(words.index(assignment), words.index(abba.RESIDUAL_KEY))
            self.assertEqual(words[words.index(abba.RESIDUAL_KEY) - 1], "-u")
            self.assertNotIn("UNRELATED", words)
            for option, value in [("--batch", "16"), ("--input", "empty"), ("--handles", "1,2"),
                                  ("--mode", "kernel"), ("--timing", "cuda-event")]:
                self.assertEqual(words[words.index(option) + 1], value)

    def test_same_binary_requires_different_explicit_tactics(self):
        hashes = dict(baseline="a" * 64, candidate="a" * 64)
        for baseline in abba.RESIDUAL_CHOICES:
            for candidate in abba.RESIDUAL_CHOICES:
                tactics = dict(baseline=abba.tactics_for(baseline), candidate=abba.tactics_for(candidate))
                if baseline == candidate:
                    with self.assertRaisesRegex(ValueError, "same binary SHA and effective tactics"):
                        abba.validate_comparison(hashes, tactics)
                else:
                    abba.validate_comparison(hashes, tactics)
        hashes["candidate"] = "b" * 64
        abba.validate_comparison(hashes, tactics)

    def test_b14_and_b16_recompute_lane_medians_from_raw_events(self):
        for batch in (14, 16):
            data = fixture(batch)
            # Exercise the exact JSON parser used after the real subprocess.
            parsed = abba.parity.parse_report("diagnostic {noise}\n" + json.dumps(data, indent=2))
            rate, markers = validate(parsed, batch)
            self.assertAlmostEqual(rate, batch * 1000 / 3 + batch * 1000 / 4)
            self.assertEqual(len(markers), 1 if batch == 14 else 2)
        data = fixture(14)
        legacy_metric = abba.parity.rust_event_rate(data, BASE_LOG, iterations=4, warmup=80,
                                                   input_kind="empty", model=abba.MODEL,
                                                   tactics=abba.tactics_for(abba.PRESET))
        self.assertEqual(validate(data, 14)[0], legacy_metric)

    def test_preset_requires_every_shape_and_exact_algorithm_contract(self):
        for batch in (14, 16):
            lines = preset_log(batch).splitlines()
            for omitted in range(len(lines)):
                log = BASE_LOG + "\n" + "\n".join(line for i, line in enumerate(lines) if i != omitted)
                with self.subTest(batch=batch, omitted=omitted), self.assertRaisesRegex(ValueError, "missing residual"):
                    validate(fixture(batch), batch, log=log)
        for old, new in [("layout=tn", "layout=nn"), ("output=f32", "output=f16"), ("beta=1", "beta=0"),
                         ("filtered_index=3", "filtered_index=1"), ("m=5776", "m=5054"),
                         ("k=1152", "k=2304"), ("cublaslt_version=130600", "cublaslt_version=130500"),
                         ("algorithm_identity=matched", "algorithm_identity=unmatched"),
                         ("engine=tf3_5070ti_r1", "engine=heuristic"), ("beta=1", "beta=1 beta=1")]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                validate(fixture(16), 16, log=BASE_LOG + "\n" + preset_log(16).replace(old, new))

    def test_heuristic_cannot_silently_execute_a_preset(self):
        data = fixture(14, "heuristic")
        self.assertEqual(validate(data, 14, "heuristic")[1], [])
        for preset in (abba.PRESET, abba.INIT_PRESET):
            with self.assertRaisesRegex(ValueError, "heuristic run executed a residual preset"):
                validate(data, 14, "heuristic", log=BASE_LOG + "\n" + preset_log(14, preset))

    def test_initialized_requires_all_shapes_and_records_only_source_index(self):
        for batch in (14, 16):
            data = fixture(batch, abba.INIT_PRESET)
            rate, markers = validate(data, batch, abba.INIT_PRESET)
            self.assertAlmostEqual(rate, batch * 1000 / 3 + batch * 1000 / 4)
            self.assertEqual(len(markers), 1 if batch == 14 else 2)
            for marker in markers:
                self.assertEqual(marker["construction"], "public_init")
                self.assertIn("source_filtered_index", marker)
                self.assertNotIn("filtered_index", marker)
            lines = preset_log(batch, abba.INIT_PRESET).splitlines()
            for omitted in range(len(lines)):
                log = BASE_LOG + "\n" + "\n".join(line for i, line in enumerate(lines) if i != omitted)
                with self.subTest(batch=batch, omitted=omitted), self.assertRaisesRegex(ValueError, "missing residual"):
                    validate(data, batch, abba.INIT_PRESET, log=log)

    def test_initialized_construction_and_algorithm_contract_cannot_be_forged(self):
        marker = preset_log(16, abba.INIT_PRESET)
        for old, new in [
            ("construction=public_init ", ""),
            ("construction=public_init", "construction=heuristic"),
            ("source_filtered_index=3", "filtered_index=3"),
            ("source_filtered_index=3", "source_filtered_index=3 filtered_index=3"),
            ("source_filtered_index=3", "source_filtered_index=1"),
            ("source_filtered_index=3", "source_filtered_index=3 source_filtered_index=3"),
            ("layout=tn", "layout=nn"), ("output=f32", "output=f16"), ("beta=1", "beta=0"),
            ("m=5776", "m=5054"), ("k=1152", "k=2304"),
            ("cublaslt_version=130600", "cublaslt_version=130500"),
            ("algorithm_identity=matched", "algorithm_identity=unmatched"),
        ]:
            with self.subTest(old=old, new=new), self.assertRaises(ValueError):
                validate(fixture(16, abba.INIT_PRESET), 16, abba.INIT_PRESET,
                         log=BASE_LOG + "\n" + marker.replace(old, new))

    def test_fixed_and_initialized_markers_and_effective_tactics_are_not_interchangeable(self):
        for requested, executed in [(abba.PRESET, abba.INIT_PRESET), (abba.INIT_PRESET, abba.PRESET)]:
            with self.subTest(requested=requested), self.assertRaises(ValueError):
                validate(fixture(16, requested), 16, requested,
                         log=BASE_LOG + "\n" + preset_log(16, executed))
            with self.assertRaisesRegex(ValueError, "effective tactics"):
                validate(fixture(16, executed), 16, requested)
        for extra in ("construction=public_init", "source_filtered_index=3"):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate(fixture(16), 16, log=BASE_LOG + "\n" + preset_log(16).replace("beta=1", f"beta=1 {extra}"))

    def test_missing_or_unbound_tactic_and_library_are_rejected(self):
        mutations = [
            lambda d: d["tactics"].pop(abba.RESIDUAL_KEY),
            lambda d: d["tactics"].update({abba.RESIDUAL_KEY: "heuristic"}),
            lambda d: d["tactics"].update(KATAGO_CUDA_CUBLASLT_RANK="time"),
            lambda d: d["build"].pop("fp16_encoding_revision"),
            lambda d: d["build"].update(fp16_encoding_revision=0),
            lambda d: d["build"].update(fp16_encoding_revision=True),
            lambda d: d["build"].pop("cublaslt_version"),
            lambda d: d["build"].update(cublaslt_version=130500),
            lambda d: d.update(model="/wrong/model.bin.gz"),
            lambda d: d.update(model_sha256="0" * 64),
            lambda d: d.update(requested_batches=[14]),
            lambda d: d["input_description"]["rules"].update(komi=0),
        ]
        for index, mutate in enumerate(mutations):
            data = fixture(16)
            mutate(data)
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate(data, 16)

    def test_event_tampering_missing_samples_and_duplicate_lanes_are_rejected(self):
        for mutation in ("rate", "count", "nan", "duplicate"):
            data = fixture(16)
            row = data["results"][0]
            if mutation == "rate":
                row["sum_median_nn_evals_per_s"] *= 1.01
            elif mutation == "count":
                row["handles"][0]["cuda_event_ms"].pop()
            elif mutation == "nan":
                row["handles"][0]["cuda_event_ms"][0] = math.nan
            else:
                row["handles"][1]["handle"] = 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate(data, 16)

    def test_spread_gate_preserves_all_samples_and_rejects_incomplete_order(self):
        runs = [dict(flavor=f, event_rate_sum=v) for f, v in
                [("baseline", 1000), ("candidate", 1100), ("candidate", 1102), ("baseline", 1002)]]
        stable, summary = abba.summarize(runs, 0.05)
        self.assertTrue(stable)
        self.assertEqual(summary["candidate"]["samples"], [1100, 1102])
        self.assertIn("candidate_over_baseline_event_ratio", summary)
        runs[-1]["event_rate_sum"] = 1300
        stable, summary = abba.summarize(runs, 0.05)
        self.assertFalse(stable)
        self.assertNotIn("candidate_over_baseline_event_ratio", summary)
        self.assertIn("diagnostic_only_event_ratio", summary)
        self.assertFalse(abba.summarize(runs[:2], 0.05)[0])


if __name__ == "__main__":
    unittest.main()
