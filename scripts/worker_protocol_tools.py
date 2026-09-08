"""Small real-gRPC harness shared by worker verification scripts.

Importing this module requires grpcio and grpcio-tools. Worker inference remains
in child binaries; this module never substitutes an evaluator.
"""
from concurrent.futures import ThreadPoolExecutor
import importlib
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time

import grpc
from grpc_tools import protoc


class Protocol:
    def __init__(self, schema):
        schema = Path(schema).resolve()
        self.generated = tempfile.TemporaryDirectory(prefix="rustgo-protocol-")
        code = protoc.main([
            "protoc", f"-I{schema.parent}", f"--python_out={self.generated.name}",
            f"--grpc_python_out={self.generated.name}", str(schema),
        ])
        if code:
            raise RuntimeError(f"protoc failed with status {code}")
        sys.path.insert(0, self.generated.name)
        self.pb = importlib.import_module("worker_pb2")
        self.rpc = importlib.import_module("worker_pb2_grpc")

    def close(self):
        sys.path.remove(self.generated.name)
        self.generated.cleanup()


class Peer:
    def __init__(self, context):
        self.context = context
        self.outgoing = queue.Queue()
        self.received = queue.Queue()
        self.hello = None
        self.heartbeats = []
        self.closed = threading.Event()
        self.reader_error = None

    def receive(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                message = self.received.get(timeout=min(0.2, max(0.001, deadline - time.monotonic())))
            except queue.Empty:
                if self.closed.is_set():
                    raise RuntimeError(f"worker stream closed: {self.reader_error}")
                continue
            if message.HasField("heartbeat"):
                self.heartbeats.append(message.heartbeat)
            return message
        raise TimeoutError("worker message timed out")

    def result(self, task_id, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            message = self.receive(max(0.001, deadline - time.monotonic()))
            if message.HasField("result"):
                if message.result.task_id != task_id:
                    raise AssertionError(("unexpected result identity", message.result.task_id, task_id))
                return message.result
            if not message.HasField("heartbeat"):
                raise AssertionError("unexpected worker message after Hello")
        raise TimeoutError(f"result {task_id} timed out")

    def settled(self, completed, seconds=30):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            message = self.receive(max(0.001, deadline - time.monotonic()))
            if not message.HasField("heartbeat"):
                raise AssertionError("unexpected result while awaiting final heartbeat")
            heartbeat = message.heartbeat
            if heartbeat.completed_requests >= completed and heartbeat.in_flight == 0:
                if heartbeat.failed_requests != 0:
                    raise AssertionError(("worker failures", heartbeat))
                return heartbeat
        raise TimeoutError("worker did not settle")


class WorkerHarness:
    def __init__(self, protocol):
        self.protocol = protocol
        self.connections = queue.Queue()
        harness = self

        class Service(protocol.rpc.WorkerServiceServicer):
            def Connect(self, iterator, context):
                peer = Peer(context)
                try:
                    first = next(iterator)
                    if not first.HasField("hello"):
                        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "first message must be Hello")
                    peer.hello = first.hello
                    harness.connections.put(peer)
                except (StopIteration, grpc.RpcError):
                    return

                def consume():
                    try:
                        for message in iterator:
                            peer.received.put(message)
                    except grpc.RpcError as error:
                        peer.reader_error = str(error)
                    finally:
                        peer.closed.set()
                        peer.outgoing.put(None)

                threading.Thread(target=consume, daemon=True).start()
                yield protocol.pb.ServerMessage(welcome=protocol.pb.Welcome(
                    protocol_version=1, connection_id="worker-numeric-verification"))
                while context.is_active():
                    try:
                        message = peer.outgoing.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if message is None:
                        return
                    yield message

        self.server = grpc.server(ThreadPoolExecutor(max_workers=4), options=[
            ("grpc.max_receive_message_length", 4 * 1024 * 1024),
            ("grpc.max_send_message_length", 4 * 1024 * 1024),
        ])
        protocol.rpc.add_WorkerServiceServicer_to_server(Service(), self.server)
        self.port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()

    def accept(self, process, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"worker exited during startup ({process.returncode}); inspect log")
            try:
                return self.connections.get(timeout=0.1)
            except queue.Empty:
                continue
        raise TimeoutError("worker did not send Hello")

    def close(self):
        self.server.stop(0).wait()


class DrainProxy:
    """Transparent WorkerService relay with a local test-only Drain control.

    The actual Go Server still performs registration, scheduling and validation.
    The relay only injects Drain after the smoke workload has fully settled.
    """
    def __init__(self, protocol, upstream):
        self.protocol = protocol
        self.peers = {}
        self.lock = threading.Lock()
        self.channel = grpc.insecure_channel(upstream)
        proxy = self

        class Service(protocol.rpc.WorkerServiceServicer):
            def Connect(self, iterator, context):
                outgoing = queue.Queue()
                upstream_inputs = queue.Queue()
                worker_id = None

                def upload():
                    nonlocal worker_id
                    try:
                        for message in iterator:
                            if message.HasField("hello"):
                                worker_id = message.hello.worker_id
                                with proxy.lock:
                                    proxy.peers[worker_id] = outgoing
                            upstream_inputs.put(message)
                    except grpc.RpcError:
                        pass
                    finally:
                        upstream_inputs.put(None)
                        outgoing.put(None)

                def requests():
                    while True:
                        message = upstream_inputs.get()
                        if message is None:
                            return
                        yield message

                call = protocol.rpc.WorkerServiceStub(proxy.channel).Connect(requests())

                def download():
                    try:
                        for message in call:
                            outgoing.put(message)
                    except grpc.RpcError:
                        pass
                    finally:
                        outgoing.put(None)

                threading.Thread(target=upload, daemon=True).start()
                threading.Thread(target=download, daemon=True).start()
                try:
                    while context.is_active():
                        try:
                            message = outgoing.get(timeout=0.1)
                        except queue.Empty:
                            continue
                        if message is None:
                            return
                        yield message
                finally:
                    call.cancel()
                    if worker_id:
                        with proxy.lock:
                            proxy.peers.pop(worker_id, None)

        self.server = grpc.server(ThreadPoolExecutor(max_workers=8))
        protocol.rpc.add_WorkerServiceServicer_to_server(Service(), self.server)
        self.port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()

    def drain(self, worker_ids):
        with self.lock:
            for worker_id in worker_ids:
                if worker_id not in self.peers:
                    raise AssertionError(f"worker disconnected before Drain: {worker_id}")
                self.peers[worker_id].put(self.protocol.pb.ServerMessage(
                    drain=self.protocol.pb.Drain(reason="mixed-pool verification complete")))

    def close(self):
        self.server.stop(0).wait()
        self.channel.close()
