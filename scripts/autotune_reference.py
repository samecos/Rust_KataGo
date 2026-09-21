"""Portable C++ FP32 references for full_autotune.py (no GPU needed to read)."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

from tune_runtime import ROOT, clean_environment, file_hash

BUNDLED_MODEL = "1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6"
REFERENCE_DIR = ROOT / "scripts/fixtures/autotune"


def fingerprints(requests, model_hash):
    digest = hashlib.sha256()
    for _, request in requests:
        raw = request.SerializeToString(deterministic=True)
        digest.update(len(raw).to_bytes(8, "little") + raw)
    return dict(model_sha256=model_hash, schema_sha256=file_hash(ROOT / "crates/kata_worker/proto/worker.proto"),
                fixtures_sha256=file_hash(ROOT / "scripts/fixtures/worker_positions.json"),
                requests_sha256=digest.hexdigest(), cases=len(requests))


def read_reference(path):
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON: {value}")))


def validate_reference(reference, expected, requests, protocol):
    """Validate the *actual corpus*, not only its self-declared fingerprints."""
    from google.protobuf.json_format import ParseDict
    from compare_worker_outputs import validate_result
    if reference.get("status") != "PASS" or reference.get("worker") != "cpp":
        raise ValueError("reference must be a completed C++ FP32 collection, not Rust/dummy output")
    for key, value in expected.items():
        if reference.get(key) != value:
            raise ValueError(f"reference fingerprint mismatch: {key}")
    fp16_values = re.findall(r"^\s*cudaUseFP16\d*\s*=\s*([^#\s]+)", reference.get("config_text", ""), re.M)
    if not fp16_values or any(value.lower() != "false" for value in fp16_values):
        raise ValueError("reference config must explicitly disable CUDA FP16")
    if not re.fullmatch(r"[0-9a-f]{64}", reference.get("binary_sha256", "")):
        raise ValueError("reference lacks source binary SHA256")
    if len(reference.get("results", [])) != len(requests) or len(requests) != 128:
        raise ValueError("reference must include all 16 positions x 8 symmetries")
    hello = ParseDict(reference["hello"], protocol.pb.WorkerHello())
    if (hello.model_sha256 != expected["model_sha256"] or hello.model_version != 17
            or "cuda" not in hello.backend_info.lower() or "dummy" in hello.backend_info.lower()):
        raise ValueError("reference must come from a real native TF3 v17 C++ CUDA worker")
    for entry, (name, request) in zip(reference["results"], requests, strict=True):
        if entry["name"] != name:
            raise ValueError("reference case order/name mismatch")
        result = ParseDict(entry["result"], protocol.pb.EvalResult())
        validate_result(result, request, hello)
    return reference


def load_bundled(model_hash):
    index_path = REFERENCE_DIR / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    entry = index.get(model_hash)
    if not entry:
        return None
    path = REFERENCE_DIR / entry["file"]
    if file_hash(path) != entry["sha256"]:
        raise ValueError("bundled FP32 reference SHA256 mismatch")
    return path


def write_reference(path, reference):
    raw = json.dumps(reference, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    Path(path).write_bytes(gzip.compress(raw, mtime=0))


def collect_reference(model, binary, config, output, protocol, requests, expected, timeout):
    from compare_worker_outputs import collect_worker
    output.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(output=output, request_window=1, model=model,
                           startup_timeout=timeout, task_timeout=timeout)
    # Reject FP16 before launching an expensive reference collection.
    values = re.findall(r"^\s*cudaUseFP16\d*\s*=\s*([^#\s]+)", config.read_text(encoding="utf-8-sig"), re.M)
    if not values or any(v.lower() != "false" for v in values):
        raise ValueError("--cpp-config must explicitly set cudaUseFP16 = false")
    result = collect_worker("cpp", binary, config, args, protocol, requests, expected, clean_environment())
    validate_reference(result, expected, requests, protocol)
    return result


def main():
    parser = argparse.ArgumentParser(description="Export a reusable native TF3 C++ CUDA FP32 numerical reference")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New .json.gz file")
    parser.add_argument("--source", type=Path, help="Existing compare_worker_outputs.py cpp/outputs.json")
    parser.add_argument("--cpp-worker", type=Path)
    parser.add_argument("--cpp-config", type=Path, default=ROOT / "configs/worker_cpp_fp32.cfg")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if bool(args.source) == bool(args.cpp_worker):
        parser.error("provide exactly one of --source or --cpp-worker")
    from compare_worker_outputs import build_requests
    from worker_protocol_tools import Protocol
    protocol = Protocol(ROOT / "crates/kata_worker/proto/worker.proto")
    try:
        fixture = json.loads((ROOT / "scripts/fixtures/worker_positions.json").read_text(encoding="utf-8"))
        requests = build_requests(protocol.pb, fixture, file_hash(args.model), 0)
        expected = fingerprints(requests, file_hash(args.model))
        if args.source:
            reference = validate_reference(read_reference(args.source), expected, requests, protocol)
        else:
            reference = collect_reference(args.model.resolve(), args.cpp_worker.resolve(), args.cpp_config.resolve(),
                                          args.output.resolve().with_suffix(".collection"), protocol, requests, expected, 600)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_reference(args.output, reference)
        print(f"Reference: {args.output.resolve()}\nSHA256: {file_hash(args.output)}")
    finally:
        protocol.close()


if __name__ == "__main__":
    main()
