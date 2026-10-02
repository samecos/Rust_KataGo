#!/usr/bin/env python3
"""CPU-only peer/clock regressions; no binary, server, CUDA or cargo is used."""

import copy
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from benchmark_quantization_corpus import arm_environment, validate_numeric_collection_modes
from collect_quantization_corpus import (
    continuous_capture, drive_continuous, save_continuous_evidence,
    parse_args, sha256, validate_collection_window_mode,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeResult:
    def __init__(self, peer, task_id):
        self.peer = peer
        self.task_id, self.generation, self.session_id, self.input_hash, self.model_sha256 = identity(task_id)
        self.error_code, self.error_message = "", ""
        self.has_output = True
        self.payload = f"retained-protobuf-{task_id}".encode()

    def HasField(self, field):
        return field == "output" and self.has_output

    def ByteSize(self):
        self.peer.events.append(("size", self.task_id, len(self.peer.sent)))
        return len(self.payload)

    def SerializeToString(self, deterministic=False):
        if not self.peer.stopped:
            raise AssertionError("serialization occurred before Worker Drain/stop")
        return self.payload


class FakeMessage:
    def __init__(self, field, payload):
        self.field = field
        setattr(self, field, payload)

    def HasField(self, field):
        return field == self.field

    def SerializeToString(self, deterministic=False):
        if self.field == "result":
            return self.result.SerializeToString(deterministic=deterministic)
        return b"retained-heartbeat"


def identity(task_id):
    return task_id, 1, "test-session", f"hash-{task_id}".encode(), "a" * 64


class FakePeer:
    def __init__(self, clock, events):
        self.clock, self.incoming = clock, list(events)
        self.outgoing = SimpleNamespace(put=self.send)
        self.sent, self.events = [], []
        self.stopped = False

    def send(self, message):
        self.sent.append(message.task_id)
        self.events.append(("send", message.task_id))

    def receive(self, seconds):
        if not self.incoming:
            self.clock.now += seconds
            raise TimeoutError("simulated lost request")
        delta, task, mutate = self.incoming.pop(0)
        if delta > seconds:
            self.clock.now += seconds
            raise TimeoutError("simulated oldest-request deadline")
        self.clock.now += delta
        if task == "heartbeat":
            return FakeMessage("heartbeat", SimpleNamespace(failed_requests=mutate or 0))
        result = FakeResult(self, task)
        if mutate:
            mutate(result)
        self.events.append(("receive", task))
        return FakeMessage("result", result)


def prepared(count):
    return [(SimpleNamespace(task_id=i), identity(i)) for i in range(1, count + 1)]


class ContinuousCollectionTests(unittest.TestCase):
    def drive(self, count, events, *, window=2, capacity=2, timeout=10, limit=1024 * 1024):
        clock = FakeClock()
        peer = FakePeer(clock, events)
        capture = continuous_capture()
        drive_continuous(peer, prepared(count), window, capacity, timeout, limit, capture, clock=clock)
        return peer, capture

    def test_out_of_order_heartbeat_all_complete_refill_before_accounting(self):
        peer, capture = self.drive(6, [(0.1, v, None) for v in ("heartbeat", 2, 1, "heartbeat", 4, 3, 6, 5)])
        self.assertEqual(peer.sent, [1, 2, 3, 4, 5, 6])
        self.assertEqual([r.task_id for r in capture["results"]], [2, 1, 4, 3, 6, 5])
        self.assertEqual(capture["heartbeat_count"], 2)
        self.assertEqual(capture["max_pending"], 2)
        self.assertEqual(capture["pending_task_ids"], [])
        sizes = [event for event in peer.events if event[0] == "size"]
        self.assertEqual([event[2] for event in sizes], [3, 4, 5, 6, 6, 6])
        self.assertFalse(peer.stopped)  # no SerializeToString was called

    def test_window_larger_than_corpus_sends_only_actual_requests(self):
        peer, capture = self.drive(2, [(0, 2, None), (0, 1, None)], window=8, capacity=8)
        self.assertEqual(peer.sent, [1, 2])
        self.assertEqual(capture["max_pending"], 2)

    def test_duplicate_stale_identity_errors_and_missing_output_fail(self):
        cases = [
            [(0, 1, None), (0, 1, None)],
            [(0, 99, None)],
            [(0, 1, lambda r: setattr(r, "input_hash", b"wrong"))],
            [(0, 1, lambda r: setattr(r, "model_sha256", "b" * 64))],
            [(0, 1, lambda r: setattr(r, "has_output", False))],
            [(0, 1, lambda r: setattr(r, "error_code", "ERROR"))],
            [(0, "heartbeat", 1)],
        ]
        for events in cases:
            with self.subTest(events=events), self.assertRaises(AssertionError):
                self.drive(3, events)

    def test_heartbeats_and_other_results_do_not_renew_oldest_deadline(self):
        clock = FakeClock()
        peer = FakePeer(clock, [(0.4, 2, None), (0.4, "heartbeat", None), (0.3, 3, None)])
        capture = continuous_capture()
        with self.assertRaises(TimeoutError):
            drive_continuous(peer, prepared(3), 2, 2, 1.0, 1024, capture, clock=clock)
        self.assertEqual(clock.now, 1.0)
        self.assertEqual([r.task_id for r in capture["results"]], [2])
        self.assertEqual(capture["pending_task_ids"], [1, 3])

    def test_window_capacity_and_duplicate_prepared_reject_before_sending(self):
        for window, capacity in [(0, 2), (3, 2), (257, 4096), (1, 0)]:
            clock = FakeClock()
            peer = FakePeer(clock, [])
            with self.subTest(window=window), self.assertRaises(ValueError):
                drive_continuous(peer, prepared(1), window, capacity, 1, 1024, continuous_capture(), clock=clock)
            self.assertEqual(peer.sent, [])
        peer = FakePeer(FakeClock(), [])
        with self.assertRaises(ValueError):
            drive_continuous(peer, prepared(1) * 2, 1, 1, 1, 1024, continuous_capture())
        self.assertEqual(peer.sent, [])

    def test_payload_budget_fails_after_refill_and_preserves_partial_evidence(self):
        clock = FakeClock()
        peer = FakePeer(clock, [(0, 1, None), (0, 2, None)])
        capture = continuous_capture()
        with self.assertRaises(ValueError):
            drive_continuous(peer, prepared(4), 2, 2, 1, 20, capture, clock=clock)
        self.assertEqual(peer.sent, [1, 2, 3, 4])
        self.assertEqual([r.task_id for r in capture["results"]], [1, 2])
        peer.stopped = True
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "results").mkdir()
            report = {"status": "FAILED"}
            save_continuous_evidence(capture, directory, report)
            self.assertEqual(report["status"], "FAILED")
            self.assertEqual(len(list((directory / "results").glob("*.pb"))), 2)
            self.assertEqual(report["continuous_buffer"]["pending_task_ids"], [3, 4])

    def test_offending_message_is_retained_on_identity_failure(self):
        clock = FakeClock()
        peer = FakePeer(clock, [(0, 1, lambda r: setattr(r, "generation", 2))])
        capture = continuous_capture()
        with self.assertRaises(AssertionError):
            drive_continuous(peer, prepared(1), 1, 1, 1, 1024, capture, clock=clock)
        self.assertIsNotNone(capture["failure_message"])
        peer.stopped = True
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "results").mkdir()
            save_continuous_evidence(capture, directory, {})
            self.assertTrue((directory / "continuous-failure-message.pb").is_file())

    def test_complete_buffer_evidence_and_tamper_checks(self):
        peer, capture = self.drive(3, [(0, 2, None), (0, 3, None), (0, 1, None)])
        peer.stopped = True
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "results").mkdir()
            report = dict(window_mode="continuous", expected_requests=3, request_window=2, capacity=2,
                          collection_buffer_contract=dict(schema="rustgo-continuous-corpus-buffer-contract-v1",
                              max_results=3, max_protobuf_payload_bytes=1024 * 1024),
                          continuous_numerically_validated_results=3)
            save_continuous_evidence(capture, directory, report)
            summary_raw = (directory / "continuous-buffer.json").read_bytes()
            report["artifacts"] = [dict(file="continuous-buffer.json", sha256=sha256(summary_raw))]
            self.assertEqual(validate_collection_window_mode(directory, report), "continuous")
            for key, value in [("continuous_numerically_validated_results", 2), ("window_mode", "chunked")]:
                altered = copy.deepcopy(report)
                altered[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_collection_window_mode(directory, altered)
            summary = json.loads(summary_raw)
            summary["pending_task_ids"] = [3]
            raw = json.dumps(summary).encode()
            (directory / "continuous-buffer.json").write_bytes(raw)
            altered = copy.deepcopy(report)
            altered["continuous_buffer"] = {k: v for k, v in summary.items() if k != "artifacts"}
            altered["artifacts"][0]["sha256"] = sha256(raw)
            with self.assertRaises(ValueError):
                validate_collection_window_mode(directory, altered)
            (directory / "continuous-buffer.json").write_bytes(summary_raw)
            (directory / "results/000002.pb").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                validate_collection_window_mode(directory, report)

    def test_old_chunked_metadata_and_formal_admission(self):
        self.assertEqual(validate_collection_window_mode(Path("unused"), {}), "chunked")
        old = dict(reference="chunked", baseline="chunked", candidate="chunked")
        validate_numeric_collection_modes(old, True)
        with self.assertRaises(ValueError):
            validate_numeric_collection_modes(old, False)
        modes = dict.fromkeys(old, "continuous")
        validate_numeric_collection_modes(modes, False)
        for key in modes:
            altered = dict(modes, **{key: "chunked"})
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_numeric_collection_modes(altered, False)
        with self.assertRaises(ValueError):
            validate_numeric_collection_modes(dict(modes, candidate="unknown"), True)

    def test_formal_admission_rejects_before_protocol_or_outputs_are_read(self):
        import benchmark_quantization_corpus as benchmark
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = {}
            for name in ("reference", "baseline", "candidate"):
                directory = root / name
                directory.mkdir()
                # Deliberately no protobufs, requests or outputs exist.
                (directory / "report.json").write_text('{"split":"selection"}', encoding="utf-8")
                directories[name] = directory
            args = SimpleNamespace(**directories, output=root / "rejected", smoke=False, metric="rpc_throughput")
            with patch.object(benchmark, "parse_args", return_value=args), \
                    patch("worker_protocol_tools.Protocol") as protocol:
                self.assertEqual(benchmark.main(), 1)
                protocol.assert_not_called()
            report = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
            self.assertIn("formal ABBA requires continuous", report["error"])
            self.assertEqual(set(report["numeric_collection_window_modes"].values()), {"chunked"})


class ExplicitMxfp8TacticTests(unittest.TestCase):
    def test_cli_and_timing_environment_preserve_only_explicit_tactic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {key: root / key for key in ("binary", "model", "requests", "manifest", "recipe")}
            for path in paths.values():
                path.write_text("{}", encoding="utf-8")
            base = ["collect"]
            for key in ("binary", "model", "requests", "manifest"):
                base.extend([f"--{key}", str(paths[key])])
            base.extend(["--output", str(root / "new-output")])
            with patch("sys.argv", base):
                self.assertIsNone(parse_args().mxfp8_scale_clear_fusion)
            for value in ("0", "1"):
                argv = base + ["--backend", "cudaquantbackend", "--recipe", str(paths["recipe"]),
                               "--mxfp8-scale-clear-fusion", value]
                with self.subTest(value=value), patch("sys.argv", argv):
                    self.assertEqual(parse_args().mxfp8_scale_clear_fusion, value)
                env = arm_environment({"controlled_environment": {
                    "KATAGO_CUDA_MXFP8_SCALE_CLEAR_FUSION": value, "KATAGO_CUDA_BATCH_TRACE": "1"}})
                self.assertEqual(env["KATAGO_CUDA_MXFP8_SCALE_CLEAR_FUSION"], value)
                self.assertNotIn("KATAGO_CUDA_BATCH_TRACE", env)
            for extra in (["--mxfp8-scale-clear-fusion", "1"],
                          ["--backend", "cudaquantbackend", "--mxfp8-scale-clear-fusion", "1"],
                          ["--backend", "cudaquantbackend", "--recipe", str(paths["recipe"]),
                           "--mxfp8-scale-clear-fusion", "2"]):
                with patch("sys.argv", base + extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args()


if __name__ == "__main__":
    unittest.main()
