"""Offline root-collection experiment on a COPY of Go Server's go-core crate.

Never writes to the Server checkout or an existing executable. The synthetic
graph measures collection costs, not Go strength or end-to-end UI latency.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-root", type=Path, default=Path("D:/Go/Server"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--nodes", default="10000,100000,300000")
    parser.add_argument("--keep-pct", default="10,50,90")
    parser.add_argument("--edge-slots", default="256")
    parser.add_argument("--rounds", default="2")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    if out.exists():
        raise SystemExit(f"Refusing to reuse output directory: {out}")
    source = args.server_root.resolve() / "crates/go-core"
    crate = out / "go-core-probe"
    shutil.copytree(source / "src", crate / "src")
    manifest = (source / "Cargo.toml").read_text(encoding="utf-8")
    (crate / "Cargo.toml").write_text(manifest + "\n[workspace]\n", encoding="utf-8")
    shutil.copyfile(args.server_root / "Cargo.lock", crate / "Cargo.lock")
    probe = Path(__file__).with_name("server_gc_probe.rs").read_text(encoding="utf-8")
    search = crate / "src/search.rs"
    original = search.read_bytes()
    with search.open("a", encoding="utf-8") as f:
        f.write("\n" + probe)
    metadata = {
        "source": str(source),
        "search_sha256": hashlib.sha256(original).hexdigest(),
        "nodes": args.nodes, "keep_pct": args.keep_pct,
        "edge_slots": args.edge_slots, "abba_rounds": args.rounds,
        "scope": "synthetic graph; CPU-only; no service or model contacted",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    env = dict(os.environ, MOVE_PROBE_NODES=args.nodes,
               MOVE_PROBE_KEEP=args.keep_pct, MOVE_PROBE_EDGES=args.edge_slots,
               MOVE_PROBE_ROUNDS=args.rounds)
    cmd = ["cargo", "test", "--manifest-path", str(crate / "Cargo.toml"),
           "--target-dir", str(out / "build"), "--release", "--offline", "--lib",
           "move_latency_probe", "--", "--ignored", "--nocapture", "--test-threads=1"]
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
