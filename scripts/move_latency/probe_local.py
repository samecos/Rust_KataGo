"""Measure local Search lifecycle on an isolated source copy, without CUDA."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--nodes", default="10000,100000,300000")
    p.add_argument("--keep-pct", default="10,90")
    p.add_argument("--threads", default="8")
    p.add_argument("--rounds", default="1")
    args = p.parse_args()
    source = Path(__file__).resolve().parents[2]
    out = args.out_dir.resolve()
    if out.exists():
        raise SystemExit(f"Refusing to reuse output directory: {out}")
    workspace = out / "source"
    members = [d.name for d in (source / "crates").iterdir() if (d / "Cargo.toml").is_file()]
    for name in members:
        shutil.copytree(source / "crates" / name, workspace / "crates" / name,
                        ignore=shutil.ignore_patterns("target", "tests", "examples", "benches"))
    manifest = (source / "Cargo.toml").read_text(encoding="utf-8")
    (workspace / "Cargo.toml").write_text(manifest, encoding="utf-8")
    shutil.copyfile(source / "Cargo.lock", workspace / "Cargo.lock")
    search = workspace / "crates/kata_search/src/search.rs"
    original = search.read_bytes()
    code = original.decode("utf-8").replace("fn search_with_dummy()", "pub(super) fn search_with_dummy()")
    code += "\n" + Path(__file__).with_name("local_gc_probe.rs").read_text(encoding="utf-8")
    search.write_text(code, encoding="utf-8")
    metadata = {"source": str(source), "search_sha256": hashlib.sha256(original).hexdigest(),
                "nodes": args.nodes, "keep_pct": args.keep_pct, "threads": args.threads,
                "rounds": args.rounds, "scope": "synthetic graph; no model inference; production sources unchanged"}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    env = dict(os.environ, MOVE_PROBE_NODES=args.nodes, MOVE_PROBE_KEEP=args.keep_pct,
               MOVE_PROBE_THREADS=args.threads, MOVE_PROBE_ROUNDS=args.rounds)
    cmd = ["cargo", "test", "--manifest-path", str(workspace / "Cargo.toml"),
           "--target-dir", str(out / "build"), "--release", "--offline", "-p", "kata_search",
           "--lib", "move_latency_probe", "--", "--ignored", "--nocapture", "--test-threads=1"]
    with (out / "probe.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    rows = []
    for line in (out / "probe.log").read_text(encoding="utf-8").splitlines():
        if "MOVE_PROBE " in line:
            rows.append(json.loads(line.split("MOVE_PROBE ", 1)[1]))
    (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"exit_code": result.returncode, "samples": len(rows), "out": str(out)}))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
