"""CPU-only checks of legacy receipt wiring; subprocesses/GPU are mocked."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import benchmark_quantization_corpus as benchmark
import collect_quantization_corpus as collector
import compare_quantization_corpus as comparator
import legacy_quantization_execution as legacy


class LegacyIntegrationTests(unittest.TestCase):
    def test_cli_requires_explicit_legacy_config_and_rejects_tracing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {name: root / name for name in ("binary", "model", "requests", "manifest", "config", "spec")}
            for path in paths.values():
                path.write_text("{}", encoding="utf-8")
            argv = ["collect"]
            for name in ("binary", "model", "requests", "manifest"):
                argv += [f"--{name}", str(paths[name])]
            argv += ["--output", str(root / "new"), "--backend", "cudaint8backend", "--execution-spec", str(paths["spec"])]
            for extra in ([], ["--config", str(paths["config"]), "--trace-batches"]):
                with self.subTest(extra=extra), patch("sys.argv", argv + extra), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    collector.parse_args()
            with patch("sys.argv", argv + ["--config", str(paths["config"])]), \
                    patch.object(collector, "load_worker_config", return_value={}), \
                    patch.object(legacy, "load_spec", return_value={"checked": True}) as load:
                args = collector.parse_args()
            self.assertEqual(args.loaded_execution_spec, {"checked": True})
            self.assertEqual(load.call_args.kwargs["batch"], 8)
            self.assertEqual(load.call_args.kwargs["backend"], "cudaint8backend")

    def test_comparator_rejects_bad_receipt_before_reading_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = dict(schema="rustgo-quantization-collection-v1", status="COLLECTED_UNVERIFIED",
                          drained=True, backend="cudaint8backend", expected_requests=1, completed_requests=1,
                          artifacts=[], legacy_execution={})
            for name in ("model_sha256", "binary_sha256", "schema_sha256", "requests_sha256", "corpus_manifest_sha256",
                         "corpus_request_set_sha256", "source_request_file_sha256", "ordered_output_hashes_sha256"):
                report[name] = "a" * 64
            report["schema_sha256"] = collector.sha256((comparator.ROOT / "crates/kata_worker/proto/worker.proto").read_bytes())
            (root / "report.json").write_text(json.dumps(report), encoding="utf-8")
            with patch.object(comparator, "bound_artifacts", return_value={}), \
                    patch.object(comparator, "validate_collection_window_mode"), \
                    patch.object(legacy, "validate_collection", side_effect=ValueError("bad raw receipt")) as validate, \
                    self.assertRaisesRegex(ValueError, "bad raw receipt"):
                comparator.load_collection(root, None)
            validate.assert_called_once()

    def test_runtime_dependencies_include_validated_nested_probes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("binary", "model"):
                (root / name).write_bytes(name.encode())
            (root / "worker.cfg").write_text("nnBackend=cudaint8backend\nnnMaxBatchSize=8\n", encoding="utf-8")
            probe = root / "before-fingerprint/stdout.json"
            probe.parent.mkdir()
            probe.write_bytes(b"bound probe")
            report = dict(binary=str(root / "binary"), model=str(root / "model"), backend="cudaint8backend", batch=8,
                          binary_sha256=benchmark.file_hash(root / "binary"), model_sha256=benchmark.file_hash(root / "model"),
                          config_sha256=benchmark.file_hash(root / "worker.cfg"), legacy_execution={})
            with patch.object(legacy, "validate_collection", return_value={"artifact_paths": ["worker.cfg", "before-fingerprint/stdout.json"]}):
                _, _, files = benchmark.runtime_files((root, {"report": report}), None)
            self.assertEqual(len(files), 4)  # effective config is deduplicated
            benchmark.check_files(files)
            probe.write_bytes(b"tampered after qualification")
            with self.assertRaisesRegex(ValueError, "runtime file changed"):
                benchmark.check_files(files)

    def run_fake_arm(self, validation_error=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source", root / "output"
            source.mkdir()
            output.mkdir()
            events = []
            report = dict(backend="cudaint8backend", batch=8, model_sha256="a" * 64,
                          controlled_environment=legacy.INT8_ENVIRONMENT, legacy_execution={})
            args = SimpleNamespace(output=output, capacity=1, concurrency=1, startup_timeout=10, task_timeout=10)
            peer = SimpleNamespace(hello=object(), outgoing=SimpleNamespace(put=lambda _: None), settled=lambda n, _: n)
            harness = SimpleNamespace(port=12345, accept=lambda *_: peer, close=lambda: events.append("close"))
            process = SimpleNamespace(wait=lambda **_: 0, poll=lambda: 0)
            protocol = SimpleNamespace(pb=SimpleNamespace(ServerMessage=lambda **_: None, Drain=lambda **_: None))
            def fingerprint(*_, **kwargs):
                label = _[2]
                events.append(label + "-probe")
                return {"label": label}
            def drive(*_, **kwargs):
                events.append("timing" if kwargs["record"] else "warmup")
                return dict(elapsed_seconds=1.0, harness_cpu_seconds=0.1, timing_rows=[])
            def validate(*_):
                events.append("validate")
                if validation_error:
                    raise ValueError(validation_error)
                return dict(observed_execution_id="checked")
            with patch("worker_protocol_tools.WorkerHarness", return_value=harness), \
                    patch.object(benchmark, "runtime_files", return_value=(root / "binary", root / "model", [])), \
                    patch.object(benchmark.subprocess, "Popen", return_value=process), \
                    patch.object(benchmark, "check_hello"), patch.object(benchmark, "message_dict", return_value={}), \
                    patch.object(benchmark, "read_process_cpu", return_value=None), \
                    patch.object(benchmark, "drive_window", side_effect=drive), \
                    patch.object(benchmark, "numeric_heartbeat", side_effect=lambda n: dict(in_flight=0, failed_requests=0,
                        completed_requests=n, nn_rows=n, nn_batches=n)), \
                    patch.object(legacy, "load_spec", return_value={}), \
                    patch.object(legacy, "probe_fingerprint", side_effect=fingerprint), \
                    patch.object(legacy, "validate_arm", side_effect=validate):
                if validation_error:
                    with self.assertRaisesRegex(ValueError, validation_error):
                        benchmark.run_arm("baseline", 0, (source, dict(report=report, report_sha256="b" * 64)),
                                          None, args, protocol, [1], [2])
                else:
                    result = benchmark.run_arm("baseline", 0, (source, dict(report=report, report_sha256="b" * 64)),
                                               None, args, protocol, [1], [2])
                    self.assertEqual(result["legacy_execution"]["observed_execution_id"], "checked")
            saved = json.loads((output / "01-baseline/report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "FAIL" if validation_error else "PASS")
            self.assertEqual(events, ["before-probe", "warmup", "timing", "after-probe", "validate", "close"])
            self.assertEqual(saved["artifacts"][0]["file"], "worker.log")

    def test_arm_probes_and_verification_are_outside_timing(self):
        self.run_fake_arm()

    def test_arm_identity_failure_is_not_reported_as_pass(self):
        self.run_fake_arm("observed identity drift")


if __name__ == "__main__":
    unittest.main()
