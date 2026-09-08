"""CPU-only controls for benchmark identity, raw-event arithmetic and stability."""
import copy
import json
import math
from pathlib import Path
import statistics
import unittest

import benchmark_fork_parity as parity


TACTICS = {"KATAGO_CUDA_DUALFFN": "1", "KATAGO_CUDA_ATTN_TILE": "q64-serial",
           "KATAGO_CUDA_NOGRAPH": "1", "KATAGO_CUDA_GEMM_LAYOUT": "tn"}
RUST_MODEL = "/mnt/d/Go/Server/models/" + parity.MODEL.name
RUST_LOG = "\n".join((
    "[cuda-tactic] name=gemm_layout launch=tn effective=tn",
    "[cuda-tactic] name=dual_ffn launch=fused effective=1",
    "[cuda-tactic] name=attention requested=fa2 launch=fa2 tile=q64-serial",
))


def rust_fixture():
    handles = []
    for lane, values in [(1, [2.0, 4.0, 2.0, 4.0]), (2, [3.0, 5.0, 3.0, 5.0])]:
        median = statistics.median(values)
        handles.append(dict(handle=lane, iterations=4, cuda_event_ms=values,
                            cuda_event_median_ms=median,
                            cuda_event_median_nn_evals_per_s=14000.0 / median))
    return dict(command="nnbench", mode="kernel", timing="cuda-event", kernel_only=True,
                model=RUST_MODEL, model_sha256=parity.MODEL_SHA, input="empty", warmup=80,
                iterations=4, requested_batches=[14], handles=[1, 2], device_target="sm_120",
                build=dict(cuda_compiler="13.3.73"), tactics=copy.deepcopy(TACTICS),
                input_description=dict(kind="empty", inputs_version=7, board=[19, 19],
                                       rules=copy.deepcopy(parity.EMPTY_RULES), encore_phase=0,
                                       effective_symmetry=0, policy_optimism=0.0,
                                       misc_params=copy.deepcopy(parity.EMPTY_MISC)),
                results=[dict(batch=14, handles=handles, spatial_input_sha256="a" * 64,
                              global_input_sha256="b" * 64,
                              sum_median_nn_evals_per_s=sum(x["cuda_event_median_nn_evals_per_s"] for x in handles))])


def validate_rust(data, log=RUST_LOG):
    return parity.rust_event_rate(data, log, iterations=4, warmup=80, input_kind="empty",
                                  model=RUST_MODEL, tactics=TACTICS)


def fork_fixture():
    data = dict(modelFile=parity.FORK_MODEL, batchSize=14, numServerThreads=2,
                numIterations=1000, phaseOffsetUs=0, gpuIdxs=[0],
                cudaDevices=[dict(name="NVIDIA GeForce RTX 5070 Ti", multiProcessorCount=70,
                                  computeCapabilityMajor=12, computeCapabilityMinor=0)],
                perServerMedianMs=[10.0, 20.0], perServerNNEvalsPerSec=[1400.0, 700.0],
                combinedNNEvalsPerSec=2100.0)
    log = f"Loaded CUDA tactic plan id from {parity.FORK_PLAN} fileSha256={parity.FORK_PLAN_SHA} B14 streamsPerDevice=2 evaluatorThreads=2\n"
    for lane in (0, 1):
        log += f"Cuda backend thread {lane}: Model version 17 useFP16 = true useNHWC = true\n"
        log += "\n".join("SM120 backend: " + marker for marker in parity.FORK_MARKERS) + "\n"
    return data, log


class ForkParityTests(unittest.TestCase):
    def test_parser_ignores_diagnostic_objects_and_rejects_two_results(self):
        report = rust_fixture()
        text = '{"results":"diagnostic only"}\nlog has {noise}\n' + json.dumps(report, indent=2)
        self.assertEqual(parity.parse_report(text), report)
        with self.assertRaisesRegex(ValueError, "found 2"):
            parity.parse_report(text + "\n" + json.dumps(report))

    def test_recomputes_raw_lane_medians_and_rejects_tampering(self):
        data = rust_fixture()
        self.assertAlmostEqual(validate_rust(data), 14000 / 3 + 14000 / 4)
        for mutation in ("rate", "sample_count", "nan", "duplicate_lane"):
            changed = copy.deepcopy(data)
            row = changed["results"][0]
            if mutation == "rate":
                row["sum_median_nn_evals_per_s"] *= 1.01
            elif mutation == "sample_count":
                row["handles"][0]["cuda_event_ms"].pop()
            elif mutation == "nan":
                row["handles"][0]["cuda_event_ms"][0] = math.nan
            else:
                row["handles"][1]["handle"] = 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_rust(changed)

    def test_rejects_identity_input_and_hidden_tactic_changes(self):
        mutations = (
            lambda d: d.update(model="/another/path/" + parity.MODEL.name),
            lambda d: d.update(model_sha256="0" * 64),
            lambda d: d.update(input="positions"),
            lambda d: d.update(warmup=0),
            lambda d: d["input_description"]["rules"].update(komi=0),
            lambda d: d["input_description"]["misc_params"].update(nn_policy_temperature=0.5),
            lambda d: d["tactics"].update(KATAGO_CUDA_ATTN="v3"),
        )
        for index, mutate in enumerate(mutations):
            data = rust_fixture()
            mutate(data)
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_rust(data)
        with self.assertRaisesRegex(ValueError, "serial attention"):
            validate_rust(rust_fixture(), RUST_LOG.replace("tile=q64-serial", "tile=q64-serial-disabled"))

    def test_fork_requires_bound_plan_both_lanes_and_tactic_counts(self):
        data, log = fork_fixture()
        self.assertEqual(parity.fork_event_rate(data, log, 1000), 2100.0)
        for changed in (log.replace(parity.FORK_PLAN_SHA, "0" * 64),
                        log.replace("Cuda backend thread 1:", "Cuda backend thread 3:"),
                        log.replace("SM120 backend: " + parity.FORK_MARKERS[-1], "missing", 1)):
            with self.assertRaises(ValueError):
                parity.fork_event_rate(data, changed, 1000)
        data["combinedNNEvalsPerSec"] = 2200.0
        with self.assertRaisesRegex(ValueError, "median-rate sum"):
            parity.fork_event_rate(data, log, 1000)

    def test_unstable_or_single_samples_cannot_produce_accepted_ratio(self):
        runs = [dict(flavor=f, event_rate_sum=v) for f, v in
                [("fork", 1000), ("rust", 800), ("rust", 802), ("fork", 1002)]]
        stable, summary = parity.summarize(runs, 0.1)
        self.assertTrue(stable)
        self.assertIn("rust_over_fork_event_ratio", summary)
        runs[-1]["event_rate_sum"] = 1300
        stable, summary = parity.summarize(runs, 0.1)
        self.assertFalse(stable)
        self.assertNotIn("rust_over_fork_event_ratio", summary)
        self.assertIn("diagnostic_only_rust_over_fork_event_ratio", summary)
        self.assertFalse(parity.summarize(runs[:2], 0.1)[0])

    def test_retained_actual_logs_parse_and_event_totals_recompute(self):
        directory = parity.ROOT / "target/fork-parity-20260908/fork-rust-wsl-baseline-abba"
        fork_path, rust_path = directory / "1-fork.stdout.log", directory / "2-rust.stdout.log"
        if not fork_path.exists() or not rust_path.exists():
            self.skipTest("local audit records are not committed fixtures")
        log = fork_path.read_text(encoding="utf-8")
        fork = parity.parse_report(log)
        self.assertAlmostEqual(parity.fork_event_rate(fork, log, 1000), fork["combinedNNEvalsPerSec"], places=5)
        rust = parity.parse_report(rust_path.read_text(encoding="utf-8"))
        row = rust["results"][0]
        expected = sum(14000 / statistics.median(h["cuda_event_ms"]) for h in row["handles"])
        parity.matching_rate(row["sum_median_nn_evals_per_s"], expected, "retained Rust rate")
        # The legacy run had different positions and no input/hash metadata;
        # it must not silently qualify as the new semantic-empty control.
        with self.assertRaises(ValueError):
            parity.rust_event_rate(rust, RUST_LOG, iterations=1000, warmup=80,
                                   input_kind="empty", model=RUST_MODEL, tactics=TACTICS)


if __name__ == "__main__":
    unittest.main()
