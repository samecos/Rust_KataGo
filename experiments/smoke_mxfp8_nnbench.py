"""Check direct/kernel MXFP8 status accounting, not performance or accuracy.

Each three-iteration run is retained. This calls only the supplied experimental
binary and never modifies a deployment, plan, service, or source recipe.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_strict_attention_abba import parse_nnbench
from tune_runtime import clean_environment


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "model", "recipe", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    for name in ("binary", "model", "recipe", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = {str(p): sha(p) for p in (args.binary, args.model, args.recipe, Path(__file__))}
    report = dict(schema="rustgo-mxfp8-nnbench-smoke-v1", status="RUNNING", files=frozen,
                  runs=[], performance_evidence=False, accuracy_certified=False,
                  deployment_adopted=False)

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(name, command):
        assert all(sha(Path(p)) == expected for p, expected in frozen.items())
        result = subprocess.run(command, cwd=ROOT, env=clean_environment(),
                                capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=180)
        (args.output / f"{name}.stdout.log").write_text(result.stdout, encoding="utf-8")
        (args.output / f"{name}.stderr.log").write_text(result.stderr, encoding="utf-8")
        assert result.returncode == 0, f"{name} failed; inspect retained logs"
        return result

    try:
        inspect = args.output / "inspection"
        run("inspect", [str(args.binary), "quant-inspect", "--model", str(args.model),
                        "--recipe", str(args.recipe), "--output", str(inspect)])
        identity = json.loads((inspect / "recipe-identity.json").read_text())
        recipe_sha = identity["recipe_sha256"]
        cases = [
            ("mx-kernel-wall", "cudaquantbackend", "kernel", "wall", "1", "1,8"),
            ("mx-kernel-events-two-handles", "cudaquantbackend", "kernel", "cuda-event", "1,2", "1,8"),
            ("mx-direct-two-handles", "cudaquantbackend", "direct", "wall", "1,2", "1,8"),
            ("fp16-control", "cudabackend", "kernel", "cuda-event", "1", "1"),
            ("int8-control", "cudaint8backend", "kernel", "cuda-event", "1", "1"),
        ]
        for name, backend, mode, timing, handles, batches in cases:
            command = [str(args.binary), "nnbench", "--model", str(args.model),
                       "--override-config", f"nnBackend={backend}", "--mode", mode,
                       "--batch", batches, "--handles", handles, "--input", "positions",
                       "--warmup", "2", "--iterations", "3", "--timing", timing, "--json"]
            mx = backend == "cudaquantbackend"
            if mx:
                command += ["--override-config", f"cudaQuantPlan={args.recipe}"]
            data = parse_nnbench(run(name, command).stdout)
            assert data["model_sha256"] == frozen[str(args.model)]
            assert data["mode"] == mode and data["timing"] == timing
            assert data["requested_batches"] == list(map(int, batches.split(",")))
            assert data["handles"] == list(map(int, handles.split(",")))
            if mx:
                assert data["recipe_sha256"] == recipe_sha
                assert data["quantization_semantics_version"] == 2
                assert data["mxfp8_quantization_version"].startswith("mxfp8-e4m3-")
            else:
                assert "mxfp8_quantization_version" not in data
            for row in data["results"]:
                assert len(row["handles"]) == len(data["handles"])
                for handle in row["handles"]:
                    assert handle["iterations"] == 3
                    if mx:
                        status = handle["quantized_status_validation"]
                        assert status["warmup_forwards"] == 2 and status["timed_forwards"] == 3
                        assert status["device_snapshot_words"] == (6 if mode == "kernel" else 0)
                    else:
                        assert "quantized_status_validation" not in handle
                    if timing == "cuda-event":
                        assert len(handle["cuda_event_ms"]) == 3
            report["runs"].append(dict(name=name, command=command, data=data))
            save()
            print(f"PASS {name}", flush=True)
        assert all(sha(Path(p)) == expected for p, expected in frozen.items())
        report["status"] = "SMOKE_ONLY"
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
