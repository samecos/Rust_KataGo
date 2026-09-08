"""CPU-only worker benchmark evidence and performance-gate regressions."""
import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest import mock

import benchmark_workers as bench


def sample(worker, rate, concurrency=64, status="PASS"):
    return dict(worker=worker, concurrency=concurrency, status=status,
                measurement=dict(rpc_requests_per_second=rate, nn_rows_per_second=rate,
                                 average_nn_batch=16,
                                 timings=dict(round_trip_us=dict(p50=100, p95=200),
                                              evaluator_us=dict(p50=90))))


def abba(baseline, candidate, concurrency=64):
    return [sample("baseline", baseline[0], concurrency),
            *[sample("rust", rate, concurrency) for rate in candidate],
            *[sample("baseline", rate, concurrency) for rate in baseline[1:]]]


class PerformanceGateTests(unittest.TestCase):
    def test_stable_repeats_publish_normal_ratio(self):
        summary, = bench.summarize(abba([1000, 1010], [1030, 1040]))
        self.assertEqual(summary["performance_status"], "STABLE")
        self.assertEqual(summary["maximum_relative_spread"], 0.05)
        self.assertAlmostEqual(summary["rust_over_baseline_rpc_ratio"],
                               statistics.geometric_mean([1030, 1040]) /
                               statistics.geometric_mean([1000, 1010]))
        self.assertNotIn("diagnostic_only_rust_over_baseline_rpc_ratio", summary)

    def test_observed_593_to_1134_baseline_is_diagnostic_only(self):
        summary, = bench.summarize(abba([593.2811918926445, 1134.6202713900334],
                                       [1107.979769847499, 1130.0435556472858]))
        self.assertEqual(summary["performance_status"], "UNSTABLE_DIAGNOSTIC_ONLY")
        self.assertGreater(summary["workers"]["baseline"]["rpc_relative_spread"], 0.9)
        self.assertIn("diagnostic_only_rust_over_baseline_rpc_ratio", summary)
        self.assertNotIn("rust_over_baseline_rpc_ratio", summary)

    def test_unstable_candidate_also_blocks_normal_ratio(self):
        summary, = bench.summarize(abba([1000, 1010], [1100, 1400]))
        self.assertEqual(summary["performance_status"], "UNSTABLE_DIAGNOSTIC_ONLY")
        self.assertNotIn("rust_over_baseline_rpc_ratio", summary)

    def test_either_side_with_one_sample_is_diagnostic_only(self):
        for baseline, candidate in [([1000], [1050, 1050]),
                                    ([1000, 1000], [1050]), ([1000], [1050])]:
            with self.subTest(baseline=baseline, candidate=candidate):
                summary, = bench.summarize(abba(baseline, candidate))
                self.assertEqual(summary["performance_status"], "INSUFFICIENT_REPEATS_DIAGNOSTIC_ONLY")
                self.assertIn("diagnostic_only_rust_over_baseline_rpc_ratio", summary)
                self.assertNotIn("rust_over_baseline_rpc_ratio", summary)

    def test_failed_runs_do_not_count_toward_repeats(self):
        runs = abba([1000, 1000], [1050, 1050])
        runs[-1]["status"] = "FAIL"
        summary, = bench.summarize(runs)
        self.assertEqual(summary["workers"]["baseline"]["runs"], 1)
        self.assertEqual(summary["performance_status"], "INSUFFICIENT_REPEATS_DIAGNOSTIC_ONLY")

    def test_repeats_from_other_concurrency_do_not_fill_missing_samples(self):
        summaries = bench.summarize(abba([1000], [1050, 1050], 32) +
                                    abba([1000, 1000], [1050], 64))
        self.assertEqual(len(summaries), 2)
        self.assertTrue(all(s["performance_status"] == "INSUFFICIENT_REPEATS_DIAGNOSTIC_ONLY"
                            for s in summaries))

    def test_declared_threshold_is_inclusive_and_zero_requires_equal_rates(self):
        for limit in (0.0, 0.05, 0.1):
            with self.subTest(limit=limit):
                runs = abba([1000, 1000 * (1 + limit)], [1100, 1100])
                summary, = bench.summarize(runs, limit)
                self.assertEqual(summary["performance_status"], "STABLE")
                runs[-1]["measurement"]["rpc_requests_per_second"] += 0.01
                summary, = bench.summarize(runs, limit)
                self.assertEqual(summary["performance_status"], "UNSTABLE_DIAGNOSTIC_ONLY")

    def test_invalid_thresholds_are_rejected_by_cli_before_any_work(self):
        for value in ("nan", "inf", "-inf", "-0.001", "0.1001", "bogus"):
            with self.subTest(value=value), mock.patch("sys.argv", [
                "benchmark_workers.py", f"--maximum-relative-spread={value}"
            ]), mock.patch.object(bench, "file_hash") as hashes, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    bench.main()
                self.assertEqual(error.exception.code, 2)
                hashes.assert_not_called()
                with self.assertRaises(argparse.ArgumentTypeError):
                    bench.summarize([], value)

    def test_no_comparison_is_not_performance_stable(self):
        summary, = bench.summarize([sample("rust", 1000), sample("rust", 1000)])
        self.assertEqual(summary["performance_status"], "INSUFFICIENT_REPEATS_DIAGNOSTIC_ONLY")
        self.assertFalse(summary["comparisons"])


class ReportLifecycleTests(unittest.TestCase):
    def test_existing_report_is_preserved_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            bench.create_report(path, {"status": "PASS", "evidence": [593, 1134]})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                bench.create_report(path, {"status": "RUNNING"})
            self.assertEqual(path.read_bytes(), original)

    def test_cli_preserves_workload_pass_and_labels_unstable_ratio(self):
        # Exercise real report serialization and final output; replace worker/
        # protocol operations so this test cannot launch processes or use a GPU.
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            output = Path(directory)
            fixture = output / "fixture.json"
            fixture.write_text("{}", encoding="utf-8")
            arguments = ["benchmark_workers.py", "--output", str(output),
                         "--fixtures", str(fixture), "--baseline-worker", "unused.exe",
                         "--concurrency", "64", "--order", "baseline,rust,rust,baseline"]
            stack.enter_context(mock.patch("sys.argv", arguments))
            stack.enter_context(mock.patch.object(bench, "file_hash", return_value=bench.MODEL_SHA256))
            stack.enter_context(mock.patch.object(bench, "check_batch_size"))
            stack.enter_context(mock.patch.object(bench.platform, "platform", return_value="CPU-test-host"))
            protocol = stack.enter_context(mock.patch.object(bench, "Protocol"))
            stack.enter_context(mock.patch.object(bench, "prepare_requests",
                                                 return_value=([None] * 128, [None] * 1024, "digest", 1)))
            pending = iter(abba([593.28, 1134.62], [1107.98, 1130.04]))

            def measured(*args):
                initial = json.loads((output / "report.json").read_text(encoding="utf-8"))
                self.assertEqual(initial["maximum_relative_spread"], 0.05)
                return next(pending)

            worker = stack.enter_context(mock.patch.object(bench, "run_worker", side_effect=measured))
            stack.enter_context(mock.patch.object(bench.subprocess, "Popen",
                                                 side_effect=AssertionError("CPU test attempted child launch")))
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                bench.main()
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["performance_status"], "UNSTABLE_DIAGNOSTIC_ONLY")
            self.assertEqual(worker.call_count, 4)
            protocol.return_value.close.assert_called_once()
            self.assertIn("diagnostic_only Rust/Rust baseline", stdout.getvalue())
            self.assertNotIn("C=64: Rust/Rust baseline", stdout.getvalue())
            self.assertIn("RESULT: PASS; performance_status=UNSTABLE_DIAGNOSTIC_ONLY", stdout.getvalue())
            original = (output / "report.json").read_bytes()
            with self.assertRaises(FileExistsError):
                bench.main()
            self.assertEqual((output / "report.json").read_bytes(), original)
            self.assertEqual(worker.call_count, 4)
            self.assertEqual(protocol.call_count, 1)


if __name__ == "__main__":
    unittest.main()
