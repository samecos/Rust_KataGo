#!/usr/bin/env python3
"""CPU verification and explicit offline regeneration of the strict B14 bundle.

Verification and lock capture import only Python's standard library. Rebuild is
an explicit, separate command; it invokes the unchanged historical generator
with fake tensors and CUDA devices hidden. It never runs a numerical benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import sysconfig


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "crates/kata_nn/cuda-aot/fa4-strict-b14-r1"
EXPECTED_IDENTITY = {
    "revision": 1,
    "cubin_sha256": "90341b6d5c54a69e983a7e7d8b62956748785ad44c3e4e1b2734b92f3da46749",
    "helper_ptx_sha256": "3a5448f8cb639fca97bf13ce17b66f12bdc83c4875b8f146447e7c861f33191a",
    "abi_sha256": "8c8ef96685ce5dceceb1aaf0d5c1b8aa30183defd1affbd08c983c4ff5f152ae",
}
RUNTIME_FILES = {
    "cubin_sha256": "attention.sm120.cubin",
    "helper_ptx_sha256": "rope_no_vcopy.sm120.ptx",
    "abi_sha256": "abi.json",
}


def artifact_fingerprint(bundle: Path) -> str:
    """Runtime contract: actual bytes, ordered names/lengths, revision 1."""
    digest = hashlib.sha256(b"Rust_KataGo strict-attention AOT\0")
    digest.update(struct.pack("<I", 1))
    for filename in RUNTIME_FILES.values():
        name = filename.encode("utf-8")
        data = (bundle / filename).read_bytes()
        digest.update(struct.pack("<Q", len(name)))
        digest.update(name)
        digest.update(struct.pack("<Q", len(data)))
        digest.update(data)
    return "sha256:" + digest.hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_bundle(bundle: Path) -> dict:
    manifest = read_json(bundle / "manifest.json")
    if manifest["strict_attention_artifact"] != EXPECTED_IDENTITY:
        raise ValueError("bundle does not declare the measured revision-1 identity")
    fingerprint = artifact_fingerprint(bundle)
    if manifest["runtime_fingerprint"] != fingerprint:
        raise ValueError("runtime artifact fingerprint mismatch")
    for key, relative in RUNTIME_FILES.items():
        if sha256(bundle / relative) != EXPECTED_IDENTITY[key]:
            raise ValueError(f"measured runtime artifact changed: {relative}")
    declared = set()
    for item in manifest["files"]:
        relative = item["path"]
        candidate = bundle / relative
        if relative in declared or not candidate.resolve().is_relative_to(bundle.resolve()):
            raise ValueError(f"duplicate or escaping manifest path: {relative}")
        declared.add(relative)
        if candidate.stat().st_size != item["bytes"] or sha256(candidate) != item["sha256"]:
            raise ValueError(f"bundle file identity mismatch: {relative}")
    observed = {path.relative_to(bundle).as_posix() for path in bundle.rglob("*")
                if path.is_file() and path.name != "manifest.json"
                and "__pycache__" not in path.parts}
    # Nested historical manifests are evidence and must also be inventoried.
    observed.update(path.relative_to(bundle).as_posix() for path in bundle.rglob("manifest.json")
                    if path != bundle / "manifest.json")
    if observed != declared:
        raise ValueError(f"bundle inventory differs: missing={declared-observed}, extra={observed-declared}")
    abi = read_json(bundle / "abi.json")
    if abi["revision"] != 1 or abi["attention"]["qkv_shape_bshd"] != [14, 361, 12, 32]:
        raise ValueError("unexpected ABI revision/shape")
    if abi["rope_helper"]["kernel"] != "probe_rope_no_vcopy":
        raise ValueError("unexpected helper entry")
    ptx = (bundle / RUNTIME_FILES["helper_ptx_sha256"]).read_text(encoding="utf-8")
    if ".visible .entry probe_rope_no_vcopy(" not in ptx:
        raise ValueError("measured helper entry absent")
    return {"status": "CPU_BUNDLE_IDENTITY_PASS", "files": len(declared),
            "strict_attention_artifact": EXPECTED_IDENTITY,
            "runtime_fingerprint": fingerprint,
            "production_certified": False, "gpu_executed": False}


def normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def capture_toolchain(cuda_home: Path) -> dict:
    """Read package metadata and bytes; never import CUDA, Torch or CuTe."""
    if sys.platform != "linux":
        raise ValueError("the measured AOT export toolchain is Linux/WSL")
    site = Path(sysconfig.get_paths()["purelib"]).resolve()
    distributions = {}
    files = {}
    for dist in importlib.metadata.distributions():
        name = normalized(dist.metadata["Name"])
        if name in distributions:
            raise ValueError(f"ambiguous installed distribution: {name}")
        distributions[name] = dist.version
        if not (name.startswith("nvidia-cutlass-dsl") or name in {"flash-attn-4", "quack-kernels"}):
            continue
        for entry in dist.files or ():
            path = Path(dist.locate_file(entry)).resolve()
            if not path.is_file() or not path.is_relative_to(site):
                continue
            relative = path.relative_to(site).as_posix()
            if path.suffix == ".pyc" or "__pycache__" in path.parts:
                continue
            files[relative] = {"path": relative, "bytes": path.stat().st_size,
                               "sha256": sha256(path)}
    critical = {"nvidia-cutlass-dsl": "4.7.0", "quack-kernels": "0.6.4",
                "flash-attn-4": "0.0.1.dev1+katago", "torch": "2.13.0"}
    for name, version in critical.items():
        if distributions.get(name) != version:
            raise ValueError(f"not the recorded export environment: {name}")
    cuda_files = []
    for relative in ["bin/nvcc", "bin/ptxas", "nvvm/libdevice/libdevice.10.bc",
                     "nvvm/lib64/libnvvm.so.4.0.0"]:
        path = cuda_home / relative
        cuda_files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return {
        "schema": 1, "scope": "CPU snapshot of the existing measured WSL export environment",
        "rebuild_verified": False, "gpu_executed": False,
        "python": {"version": sys.version, "executable_sha256": sha256(Path(sys.executable))},
        "distributions": dict(sorted(distributions.items())),
        "codegen_package_files": [files[key] for key in sorted(files)],
        "cuda_files": cuda_files,
        "known_dependency_conflict": {
            "quack-kernels": "0.6.4 declares nvidia-cutlass-dsl==4.6.2",
            "actual_tested_nvidia-cutlass-dsl": "4.7.0",
            "installation": "Use the explicit pinned set with --no-deps; do not silently resolve or upgrade it.",
        },
        "archive_availability": {
            "flash_attn_wheel": {
                "filename": "flash_attn_4-0.0.1.dev1+katago-py3-none-any.whl",
                "sha256": "9ce90d0c89282558f53df3b0a44844c32050f4f4a1dc59f2d0ebd7a578f8366c",
                "source_commit": "145b1010051dbfd4bdc41a0ae55d495b08d7a458",
                "available_in_original_wsl_source_builds": True,
            },
            "other_wheel_archive_hashes": None,
            "limitation": "Installed codegen files are byte-locked; this is not an offline wheelhouse. Retrieve exact licensed packages before rebuilding elsewhere, then verify this lock.",
        },
    }


def verify_toolchain(bundle: Path, cuda_home: Path) -> dict:
    lock = read_json(bundle / "locks/toolchain.lock.json")
    observed = capture_toolchain(cuda_home)
    for field in ["python", "distributions", "codegen_package_files", "cuda_files"]:
        if observed[field] != lock[field]:
            raise ValueError(f"locked toolchain differs: {field}; do not overwrite the lock")
    return {"status": "CPU_TOOLCHAIN_IDENTITY_PASS", "gpu_executed": False,
            "codegen_files": len(lock["codegen_package_files"]), "rebuild_verified": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "capture-toolchain", "verify-toolchain", "rebuild"])
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--cuda-home", type=Path, default=Path("/usr/local/cuda-13.3"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    if args.command == "capture-toolchain":
        if args.output is None:
            parser.error("capture-toolchain requires a fresh --output path")
        lock = capture_toolchain(args.cuda_home)
        write_new_json(args.output, lock)
        print(json.dumps({"status": "CPU_TOOLCHAIN_CAPTURED", "files": len(lock["codegen_package_files"])}))
        return 0
    if args.command == "verify":
        print(json.dumps(verify_bundle(bundle)))
        return 0
    verify_bundle(bundle)
    checked = verify_toolchain(bundle, args.cuda_home)
    if args.command == "verify-toolchain":
        print(json.dumps(checked))
        return 0
    if args.output is None or args.output.exists():
        parser.error("rebuild requires --output naming a fresh directory; never overwrite this bundle")
    # Keep the measured generator unchanged; always override its historical
    # defaults. The relative reference snapshot is fully included in the bundle.
    command = [sys.executable, str(bundle / "source/build_fa4_strict.py"), "--build",
               "--reference-dir", str(bundle / "source/reference"),
               "--output-dir", str(args.output.resolve()), "--cuda-home", str(args.cuda_home)]
    result = subprocess.run(command, check=False)
    if result.returncode:
        return result.returncode
    generated = list(args.output.glob("*.cubin"))
    if len(generated) != 1:
        raise ValueError("expected one regenerated CUBIN")
    digest = sha256(generated[0])
    print(json.dumps({"status": "EXPORTED_NOT_CERTIFIED", "cubin_sha256": digest,
                      "matches_measured_cubin": digest == EXPECTED_IDENTITY["cubin_sha256"],
                      "warning": "Changed source paths/lineinfo may change bytes. Never replace the measured asset; a new hash needs ABI, numeric and performance revalidation."}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
