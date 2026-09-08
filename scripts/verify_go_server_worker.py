#!/usr/bin/env python3
"""Exercise an isolated real Go Server + RustGo worker over HTTP and TCP GTP.

The default single-worker mode uses only Python's standard library. Its fixture is synthetic; pass
--model and --config to verify a real CUDA build. Never touches an existing
server: all three listen ports are ephemeral and only child processes are stopped.

--cpp-worker adds a real C++ worker using the same model bytes and hash. This
mixed-pool mode requires grpcio/grpcio-tools for a transparent test relay that
injects Drain after the real Go Server's HTTP/GTP workload has settled.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_until(predicate, seconds, processes):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(f"Child exited ({process.returncode}); inspect output logs")
        try:
            value = predicate()
            if value:
                return value
        except (OSError, ValueError):
            pass
        time.sleep(0.1)
    raise TimeoutError("Timed out; inspect server.log and worker.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, type=Path, help="Built go-server executable")
    parser.add_argument("--worker", type=Path, default=ROOT / "target/debug/katago-rs.exe")
    parser.add_argument("--model", type=Path, help="Real model; omitted means explicit dummy fixture")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cpp-worker", type=Path, help="Optional real C++ nnworker for a shared-model mixed pool")
    parser.add_argument("--cpp-config", type=Path, default=Path("D:/Go/Server/worker/cuda-5070ti.cfg"))
    parser.add_argument("--output", type=Path, default=ROOT / "target/worker-smoke")
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--capacity", type=int, default=32, help="Worker capacity and Server request window")
    args = parser.parse_args()
    if not 1 <= args.capacity <= 4096:
        parser.error("--capacity must be in 1..4096")
    if args.cpp_worker and (not args.model or args.capacity > 1024):
        parser.error("mixed mode requires --model and --capacity <= 1024 (C++ limit)")
    args.output.mkdir(parents=True, exist_ok=True)
    ports = set()
    while len(ports) < 3:
        ports.add(free_port())
    http, grpc, gtp = sorted(ports)
    if args.model:
        model = str(args.model.resolve())
        with open(model, "rb") as stream:
            model_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        config = args.config or ROOT / "configs/worker_cuda.cfg"
    else:
        model = "/dev/null"
        model_hash = hashlib.sha256(b"Rust_KataGo synthetic dummy worker v1").hexdigest()
        config = args.config or ROOT / "configs/worker_dummy.cfg"
    server_command = [str(args.server.resolve()), "--http", f"127.0.0.1:{http}",
                      "--grpc", f"127.0.0.1:{grpc}", "--gtp-tcp", f"127.0.0.1:{gtp}",
                      "--model-sha256", model_hash, "--max-in-flight", str(args.capacity), "--max-sessions", "2"]
    worker_command = [str(args.worker.resolve()), "nnworker", "--server", f"127.0.0.1:{grpc}",
                      "--worker-id", "rustgo-smoke", "--capacity", str(args.capacity), "--once",
                      "--model", model, "--config", str(config.resolve()), "--model-sha256", model_hash]
    if not args.model:
        worker_command.append("--allow-dummy")
    processes = []
    protocol = None
    proxy = None
    transcript = []
    report = {"synthetic": args.model is None, "model_sha256": model_hash,
              "server_command": server_command, "worker_command": worker_command}

    def get_json(path):
        with urllib.request.urlopen(f"http://127.0.0.1:{http}{path}", timeout=2) as response:
            return json.load(response)

    try:
        with ExitStack() as logs:
            server_log = logs.enter_context((args.output / "server.log").open("wb"))
            worker_log = logs.enter_context((args.output / "worker.log").open("wb"))
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            processes.append(subprocess.Popen(server_command, cwd=ROOT, stdout=server_log, stderr=subprocess.STDOUT, creationflags=flags))
            wait_until(lambda: get_json("/health"), 15, processes)
            worker_ids = {"rustgo-smoke"}
            if args.cpp_worker:
                from worker_protocol_tools import Protocol, DrainProxy
                protocol = Protocol(ROOT / "crates/kata_worker/proto/worker.proto")
                proxy = DrainProxy(protocol, f"127.0.0.1:{grpc}")
                worker_command[worker_command.index("--server") + 1] = f"127.0.0.1:{proxy.port}"
                worker_ids.add("cpp-smoke")
            processes.append(subprocess.Popen(worker_command, cwd=ROOT, stdout=worker_log, stderr=subprocess.STDOUT, creationflags=flags))
            if args.cpp_worker:
                cpp_log = logs.enter_context((args.output / "cpp-worker.log").open("wb"))
                cpp_command = [str(args.cpp_worker.resolve()), "nnworker", "-server", f"127.0.0.1:{proxy.port}",
                               "-worker-id", "cpp-smoke", "-capacity", str(args.capacity), "-once",
                               "-model", model, "-config", str(args.cpp_config.resolve()), "-model-sha256", model_hash]
                report["cpp_worker_command"] = cpp_command
                processes.append(subprocess.Popen(cpp_command, cwd=ROOT, stdout=cpp_log, stderr=subprocess.STDOUT, creationflags=flags))

            def registered_workers():
                workers = get_json("/api/workers")["workers"]
                return workers if {worker["id"] for worker in workers} == worker_ids else None

            workers = wait_until(registered_workers, args.startup_timeout, processes)
            for worker in workers:
                assert worker["model"] == model_hash, worker
                assert worker["inputProfile"] == "katago-eval-v1", worker
            report["registered_worker"] = next(worker for worker in workers if worker["id"] == "rustgo-smoke")
            report["registered_workers"] = workers
            with socket.create_connection(("127.0.0.1", gtp), timeout=15) as sock:
                stream = sock.makefile("rwb")

                def command(text):
                    stream.write((text + "\n").encode())
                    stream.flush()
                    lines = []
                    while True:
                        line = stream.readline()
                        if not line:
                            raise RuntimeError("GTP closed before response")
                        if line in (b"\n", b"\r\n"):
                            if lines:
                                break
                            continue
                        lines.append(line.decode().strip())
                    response = "\n".join(lines)
                    transcript.append({"command": text, "response": response})
                    assert response.startswith("="), (text, response)
                    return response[1:].strip()

                assert command("protocol_version") == "2"
                command("boardsize 19")
                command("komi 7.5")
                command("play b D4")
                first = command("genmove w")
                assert first and first.lower() != "resign", first
                command("undo")
                second = command("genmove w")
                assert second and second.lower() != "resign", second
                command("clear_board")
                command("play b pass")
                command("play w pass")
                command("final_score")
                command("quit")
            def settled_workers():
                workers = registered_workers()
                if not workers:
                    return None
                return workers if all(worker["completed"] > 0 and worker["inFlight"] == 0
                                      and worker["retiringInFlight"] == 0 and worker["heartbeatSequence"] > 1
                                      for worker in workers) else None

            final_workers = wait_until(settled_workers, 30 if args.cpp_worker else 15, processes)
            for worker in final_workers:
                assert worker["failures"] == 0, worker
                if args.model:
                    assert worker["nnRows"] > 0 and worker["nnBatches"] > 0, worker
            report["final_workers"] = final_workers
            if proxy:
                proxy.drain(worker_ids)
                for process in processes[1:]:
                    assert process.wait(timeout=30) == 0, "worker failed to exit cleanly after Drain"
                wait_until(lambda: get_json("/api/workers")["workers"] == [], 15, processes[:1])
                report.update(drained=True, pool_empty_after_drain=True)
            final_worker = next(worker for worker in final_workers if worker["id"] == "rustgo-smoke")
            report.update(result="PASS", final_worker=final_worker, gtp=transcript)
    except Exception as error:
        report.update(result="FAIL", error=str(error), gtp=transcript)
        raise
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        if proxy:
            proxy.close()
        if protocol:
            protocol.close()
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RESULT: PASS ({'synthetic fixture' if args.model is None else 'real model'}); {args.output / 'report.json'}")


if __name__ == "__main__":
    main()
