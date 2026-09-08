"""CPU-only heartbeat-boundary regression tests; no worker or GPU is launched."""
from types import SimpleNamespace
import unittest

from worker_protocol_tools import Peer


class HeartbeatMessage:
    def __init__(self, *, rows=128, batches=8, completed=128, in_flight=0, failed=0):
        self.heartbeat = SimpleNamespace(
            nn_rows=rows, nn_batches=batches, completed_requests=completed,
            in_flight=in_flight, failed_requests=failed,
        )

    def HasField(self, name):
        return name == "heartbeat"


class SettledHeartbeatTests(unittest.TestCase):
    def peer(self, *messages):
        peer = Peer(None)
        for message in messages:
            peer.received.put(message)
        return peer

    def test_late_nn_rows_require_two_matching_snapshots(self):
        first = HeartbeatMessage(rows=127)
        second = HeartbeatMessage(rows=128)
        third = HeartbeatMessage(rows=128)
        peer = self.peer(first, second, third)
        result = peer.settled(128, seconds=1)
        self.assertIs(result, third.heartbeat)
        self.assertEqual(result.nn_rows, 128)
        self.assertTrue(peer.received.empty())

    def test_late_batch_count_also_restarts_stability(self):
        last = HeartbeatMessage(batches=9)
        peer = self.peer(HeartbeatMessage(batches=8), HeartbeatMessage(batches=9), last)
        self.assertIs(peer.settled(128, seconds=1), last.heartbeat)

    def test_busy_heartbeat_breaks_consecutive_idle_snapshots(self):
        last = HeartbeatMessage()
        peer = self.peer(HeartbeatMessage(), HeartbeatMessage(in_flight=1), HeartbeatMessage(), last)
        self.assertIs(peer.settled(128, seconds=1), last.heartbeat)

    def test_incomplete_heartbeat_breaks_consecutive_idle_snapshots(self):
        last = HeartbeatMessage()
        peer = self.peer(HeartbeatMessage(), HeartbeatMessage(completed=127), HeartbeatMessage(), last)
        self.assertIs(peer.settled(128, seconds=1), last.heartbeat)

    def test_worker_failure_is_not_accepted_as_stable(self):
        with self.assertRaisesRegex(AssertionError, "worker failures"):
            self.peer(HeartbeatMessage(failed=1)).settled(128, seconds=1)

    def test_single_idle_heartbeat_times_out_without_confirmation(self):
        with self.assertRaisesRegex(TimeoutError, "timed out|did not settle"):
            self.peer(HeartbeatMessage()).settled(128, seconds=0.02)

    def test_closed_stream_does_not_confirm_single_idle_heartbeat(self):
        peer = self.peer(HeartbeatMessage())
        peer.reader_error = "injected EOF"
        peer.closed.set()
        with self.assertRaisesRegex(RuntimeError, "worker stream closed: injected EOF"):
            peer.settled(128, seconds=0.02)

    def test_unexpected_result_is_rejected_while_settling(self):
        peer = self.peer(SimpleNamespace(HasField=lambda name: name == "result"))
        with self.assertRaisesRegex(AssertionError, "unexpected result"):
            peer.settled(128, seconds=1)


if __name__ == "__main__":
    unittest.main()
