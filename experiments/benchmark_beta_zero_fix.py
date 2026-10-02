"""Bounded B1/B8 FP16 ABBA for the beta=0 correctness fix, not PTQ selection.

Requires completed original FP32 and pre-fix raw-bit gates. Leaves every run
and timing sample intact; it cannot publish a plan or replace a binary.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_strict_attention_abba import parse_nnbench
from tune_runtime import clean_environment, decide


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "model", "numeric-report", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = {key: getattr(args, key).resolve() for key in ("baseline", "candidate", "model", "numeric_report")}
    files = {str(path): digest(path) for path in paths.values()}
    for path in [Path(__file__).resolve(), ROOT / "scripts/tune_runtime.py",
                 ROOT / "scripts/benchmark_strict_attention_abba.py"]:
        files[str(path)] = digest(path)
    numeric = json.loads(paths["numeric_report"].read_text(encoding="utf-8"))
    assert numeric["schema"] == "rustgo-beta-zero-numeric-v1" and numeric["status"] == "PASS"
    assert numeric["model_sha256"] == files[str(paths["model"])]
    assert numeric["candidate_binary_sha256"] == files[str(paths["candidate"])]
    assert numeric["baseline_binary_sha256"] == files[str(paths["baseline"])]
    assert numeric["original_fp32"] == {"B1": "PASS", "B8": "PASS"}
    assert numeric["finite_output_legacy_bits"] == {"B1": "PASS", "B8": "PASS"}
    for evidence in numeric["evidence"]:
        path = Path(evidence["path"])
        assert digest(path) == evidence["sha256"]
        files[str(path)] = evidence["sha256"]
    assert files[str(paths["baseline"])] != files[str(paths["candidate"])]
    report = dict(schema="rustgo-beta-zero-forward-abba-v1", status="RUNNING", files=files,
                  order=["baseline", "candidate", "candidate", "baseline"], runs=[],
                  production_certified=False, deployment_adopted=False, publish_allowed=False,
                  scope="FP16 fixed-physical-batch correctness-fix regression; not a PTQ or search performance claim")

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def unchanged():
        for path, expected in files.items():
            assert digest(Path(path)) == expected, f"changed artifact: {path}"

    try:
        input_identity = None
        for index, arm in enumerate(report["order"]):
            unchanged()
            command = [str(paths[arm]), "nnbench", "--model", str(paths["model"]),
                       "--override-config", "nnBackend=cudabackend", "--mode", "kernel",
                       "--batch", "1,8", "--handles", "1", "--input", "positions",
                       "--warmup", "60", "--iterations", "400", "--timing", "cuda-event", "--json"]
            print(f"{index+1}/4 {arm}: B1/B8, 400 iterations", flush=True)
            result = subprocess.run(command, cwd=ROOT, env=clean_environment(), capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=600)
            prefix = args.output / f"{index}-{arm}"
            prefix.with_suffix(".stdout.log").write_text(result.stdout, encoding="utf-8")
            prefix.with_suffix(".stderr.log").write_text(result.stderr, encoding="utf-8")
            assert result.returncode == 0, f"failed arm {index}: inspect retained logs"
            data = parse_nnbench(result.stdout)
            assert data["mode"] == "kernel" and data["kernel_only"] is True and data["timing"] == "cuda-event"
            assert data["precision"] == "FP16-mixed" and data["model_sha256"] == numeric["model_sha256"]
            assert data["requested_batches"] == [1, 8] and data["handles"] == [1]
            assert data["input"] == "positions" and data["iterations"] == 400 and data["warmup"] == 60
            assert all(value is None for value in data["tactics"].values()), "inherited execution tactic"
            assert [row["batch"] for row in data["results"]] == [1, 8]
            identity = [(row["spatial_input_sha256"], row["global_input_sha256"]) for row in data["results"]]
            if input_identity is None:
                input_identity = identity
            assert identity == input_identity, "A/B semantic inputs changed"
            for row in data["results"]:
                assert len(row["handles"]) == 1
                handle = row["handles"][0]
                samples = handle["cuda_event_ms"]
                assert len(samples) == 400 and all(math.isfinite(v) and v > 0 for v in samples)
                rate = 400 * row["batch"] * 1000 / row["wall_elapsed_ms"]
                assert math.isclose(rate, row["wall_nn_evals_per_s"], rel_tol=1e-9)
            report["runs"].append(dict(arm=arm, command=command, data=data))
            save()
        unchanged()
        report["wall_decisions"] = {
            f"B{batch}": decide(
                [r["data"]["results"][j]["wall_nn_evals_per_s"] for r in report["runs"] if r["arm"] == "baseline"],
                [r["data"]["results"][j]["wall_nn_evals_per_s"] for r in report["runs"] if r["arm"] == "candidate"],
                .01, .05) for j, batch in enumerate((1, 8))}
        report["status"] = "MEASURED_NOT_DEPLOYED"
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
