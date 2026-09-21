#!/usr/bin/env python3
"""Create a standalone FULL AUTOTUNE kit without model weights or local paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "scripts/full_autotune.py", "scripts/full_autotune.ps1", "scripts/autotune_reference.py",
    "scripts/tuning_python.ps1", "scripts/requirements-runtime-tune.txt", "scripts/tune_runtime.py",
    "scripts/validate_cuda_tactics.py", "scripts/benchmark_workers.py", "scripts/compare_worker_outputs.py",
    "scripts/worker_protocol_tools.py", "scripts/fixtures/worker_positions.json",
    "scripts/fixtures/autotune/index.json", "scripts/fixtures/autotune/tf3-b11c768-fp32.json.gz",
    "scripts/fixtures/autotune/README.md", "scripts/package_full_autotune.py",
    "crates/kata_worker/proto/worker.proto", "configs/worker_cpp_fp32.cfg",
    "docs/RustGo-FULL-AUTOTUNE.md", "docs/RustGo剪枝模型支持.md", "docs/RustGo运行模式说明.md", "docs/编译指南.md",
    "scripts/tune_rustgo.ps1", "docs/RustGo自动配置优化.md",
    "docs/RustGo性能优化结项记录.md", "docs/Go-Server-Worker.md",
]


def package(output, binary=None):
    if output.exists():
        raise ValueError("output already exists")
    sources = [(ROOT / name, name) for name in FILES]
    sources += [(path, "plans/" + path.name) for path in sorted((ROOT / "plans").glob("*.json"))
                if path.name.startswith(("worker-tf3-sm120", "best-tactic-plan"))]
    if binary:
        sources.append((binary, "target/release/katago-rs" + (".exe" if binary.suffix.lower() == ".exe" else "")))
    for path, _ in sources:
        if not path.is_file():
            raise ValueError(f"missing package input: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    manifest = dict(kind="rustgo-full-autotune-kit", includes_model=False, includes_binary=bool(binary), files={})
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for source, name in sources:
                data = source.read_bytes()
                entry = zipfile.ZipInfo.from_file(source, arcname=name)
                entry.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(entry, data)
                manifest["files"][name] = hashlib.sha256(data).hexdigest()
            archive.writestr("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            archive.writestr("START-HERE.txt", "Read docs/RustGo-FULL-AUTOTUNE.md.\n"
                             "PowerShell: ./scripts/full_autotune.ps1 -Model /path/to/model.bin.gz -Binary /path/to/katago-rs.exe\n"
                             "Python: python scripts/full_autotune.py --help\n"
                             "Model weights and CUDA runtime/driver are not included.\n")
        temporary.replace(output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    args = parser.parse_args()
    manifest = package(args.output.resolve(), args.binary.resolve() if args.binary else None)
    print(f"Kit: {args.output.resolve()} ({len(manifest['files'])} files; model not included)")


if __name__ == "__main__":
    main()
