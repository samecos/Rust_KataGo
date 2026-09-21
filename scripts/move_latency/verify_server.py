"""Isolated real-Worker Server lifecycle smoke. Never attaches to an existing service.

Requires websockets; all launched processes and logs belong to this invocation.
This is an integration smoke, not a large-tree latency/strength acceptance gate.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

from websockets.asyncio.client import connect


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


async def smoke(args, http, report):
    async with connect(f"ws://127.0.0.1:{http}/ws", max_size=8_000_000) as ws:
        sequence = 0

        async def call(kind, **fields):
            nonlocal sequence
            sequence += 1
            id_ = str(sequence)
            start = time.perf_counter()
            await ws.send(json.dumps(dict(id=id_, type=kind, **fields)))
            while True:
                value = json.loads(await asyncio.wait_for(ws.recv(), 60))
                if value.get("type") == "response" and value.get("id") == id_:
                    assert value["ok"], value
                    snapshot = value["data"]
                    report["commands"].append(dict(command=kind, fields=fields,
                        elapsed_ms=(time.perf_counter()-start)*1000, snapshot=snapshot))
                    return snapshot

        async def analyze(visits):
            snapshot = await call("analyze", enabled=True, maxVisits=visits)
            gen = snapshot["generation"]
            deadline = time.monotonic()+120
            while True:
                snapshot = json.loads(await asyncio.wait_for(ws.recv(), 60))
                assert time.monotonic() < deadline, "analysis did not finish its visit budget"
                if snapshot.get("type") == "snapshot" and snapshot["generation"] == gen:
                    assert snapshot["analysis"]["status"] != "error", snapshot
                    if snapshot["analysis"]["status"] == "finished":
                        report["searches"].append(snapshot)
                        assert snapshot["analysis"]["visits"] > 0
                        return snapshot

        await call("open")
        snapshot = await analyze(args.visits)
        assert snapshot["analysis"]["graphNodes"] > 1000, "must exercise the background drop threshold"
        # Main line keeps a searched child and its exact existing visit count.
        best = snapshot["analysis"]["candidates"][0]
        changed = await call("play", color=best["color"], index=best["index"])
        assert changed["position"] == 1
        assert changed["analysis"]["rootChange"]["retained_nodes"] > 0
        assert changed["analysis"]["inFlight"] == 0
        snapshot = await analyze(128)
        assert snapshot["analysis"]["rootChange"]["first_completion_ms"] is not None
        # Pass, undo, browsing, komi invalidation, fresh game and rapid moves.
        await call("play", color=snapshot["toPlay"], index=None)
        await call("undo")
        await call("seek", position=0)
        await call("configure", komi=6.5)
        await call("new_game")
        for i, point in enumerate([60, 300, 72, 288, 180, 181]):
            await call("play", color=1+i % 2, index=point)
        # Change roots while real NN requests may be outstanding.
        await call("analyze", enabled=True)
        await asyncio.sleep(0.03)
        await call("new_game")
        await call("analyze", enabled=False)
        snapshot = await analyze(128)
        move = await call("genmove", maxVisits=64)
        assert move["position"] == 1
        await call("analyze", enabled=False)
        deadline = time.monotonic()+15
        while True:
            snapshot = await call("snapshot")
            if snapshot["analysis"]["reclamation"]["pending_batches"] == 0:
                break
            assert time.monotonic() < deadline, "retirement did not drain"
            await asyncio.sleep(0.02)
        assert snapshot["analysis"]["inFlight"] == 0
        assert snapshot["analysis"]["reclamation"]["peak_pending_bytes"] > 0
        report["final"] = snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", type=Path, required=True)
    p.add_argument("--worker", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--worker-config", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--visits", type=int, default=4096)
    args = p.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    http, grpc = free_port(), free_port()
    assert http != grpc
    report = dict(scope="isolated real Worker lifecycle smoke; not large-tree performance acceptance",
                  server_sha256=digest(args.server), worker_sha256=digest(args.worker),
                  model_sha256=digest(args.model), config_sha256=digest(args.worker_config),
                  commands=[], searches=[])
    server = worker = None
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        with open(out/"server.log", "w") as sl, open(out/"worker.log", "w") as wl:
            server = subprocess.Popen([str(args.server.resolve()), "--http", f"127.0.0.1:{http}",
                "--grpc", f"127.0.0.1:{grpc}", "--gtp", "--model-sha256", report["model_sha256"],
                "--publish-ms", "50", "--max-in-flight", "32"], stdin=subprocess.PIPE,
                stdout=sl, stderr=subprocess.STDOUT, creationflags=flags)
            deadline = time.monotonic()+30
            while True:
                assert server.poll() is None, "isolated Server exited"
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{http}/health", timeout=1) as r:
                        report["health"] = json.load(r)
                    break
                except OSError:
                    assert time.monotonic() < deadline
                    time.sleep(0.1)
            worker = subprocess.Popen([str(args.worker.resolve()), "nnworker", "--once", "--server",
                f"127.0.0.1:{grpc}", "--worker-id", "move-latency-smoke", "--capacity", "32",
                "--model", str(args.model.resolve()), "--config", str(args.worker_config.resolve())],
                stdout=wl, stderr=subprocess.STDOUT, creationflags=flags)
            deadline = time.monotonic()+180
            while True:
                assert worker.poll() is None, "isolated Worker exited; inspect worker.log"
                with urllib.request.urlopen(f"http://127.0.0.1:{http}/api/workers", timeout=2) as r:
                    workers = json.load(r)["workers"]
                if any(w["connected"] for w in workers):
                    report["workers"] = workers
                    break
                assert time.monotonic() < deadline, "Worker startup timed out"
                time.sleep(0.1)
            asyncio.run(smoke(args, http, report))
            server.stdin.write(b"quit\n")
            server.stdin.flush()
            assert server.wait(timeout=30) == 0, "Server failed graceful shutdown"
            worker.wait(timeout=30)
            report["status"] = "passed"
    except Exception as e:
        report.update(status="failed", error=repr(e))
        raise
    finally:
        # Only terminate the exact child processes created above, if still alive.
        for process in [worker, server]:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        (out/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(status=report["status"], out=str(out))))


if __name__ == "__main__":
    main()
