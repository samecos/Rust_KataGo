#!/usr/bin/env python3
"""GTP/analysis and fail-closed configuration smoke for the INT8 backend."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess

from compare_worker_outputs import file_hash, write_json
from tune_runtime import ROOT, clean_environment, validate_gtp_smoke


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--min-ffn-width", type=int, default=0)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = [str(args.binary.resolve()), "--model", str(args.model.resolve()),
            "--config", str(ROOT / "configs/gtp_int8.cfg")]
    report = dict(status="RUNNING", binary_sha256=file_hash(args.binary), model_sha256=file_hash(args.model), cases=[])

    def run(label, mode, override, stdin, expected=None, env_extra=None):
        overrides = dict(maxVisits="8", numSearchThreads="2", cudaInt8MinFfnWidth=str(args.min_ffn_width))
        overrides.update(token.split("=", 1) for token in override.split(","))
        command = [base[0], mode, *base[1:], "--override-config", ",".join(f"{k}={v}" for k, v in overrides.items())]
        env = clean_environment()
        env.update(env_extra or {})
        result = subprocess.run(command, input=stdin, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=120, cwd=ROOT, env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        (args.output / f"{label}.stdout.log").write_text(result.stdout, encoding="utf-8")
        (args.output / f"{label}.stderr.log").write_text(result.stderr, encoding="utf-8")
        if expected:
            assert result.returncode != 0 and expected in result.stdout + result.stderr, label
        else:
            assert result.returncode == 0, label
            assert "launch=cublaslt-int8" in result.stderr + result.stdout, label
        report["cases"].append(dict(name=label, passed=True, command=command, expected_error=expected))
        return result.stdout

    try:
        for scope in ("ffn", "transformer"):
            text = run(f"gtp-{scope}", "gtp", f"cudaInt8Scope={scope}", "1 protocol_version\n2 boardsize 19\n3 genmove b\n4 quit\n")
            validate_gtp_smoke(text)
        text = run("analysis", "analysis", "cudaInt8Scope=ffn", json.dumps(dict(
            id="int8-smoke", moves=[], rules="chinese", komi=7.5, boardXSize=19, boardYSize=19,
            maxVisits=8, includeOwnership=True)) + "\n")
        replies = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
        response = next(row for row in replies if row.get("id") == "int8-smoke" and "rootInfo" in row)
        assert response["rootInfo"]["visits"] > 0 and len(response["ownership"]) == 361
        assert all(math.isfinite(v) for v in response["ownership"])
        run("bad-scope", "gtp", "cudaInt8Scope=misspelled", "quit\n", "expected ffn or transformer")
        run("fp16-plan", "gtp", "cudaTacticPlan=plans/best-tactic-plan.json", "quit\n", "cannot reuse a FP16")
        run("float-scope", "gtp", "nnBackend=cudabackend", "quit\n", "requires nnBackend=cudaint8backend")
        run("bad-width", "gtp", "cudaInt8MinFfnWidth=9", "quit\n", "must be 8-aligned")
        run("all-float", "gtp", "cudaInt8Scope=ffn,cudaInt8MinFfnWidth=8192", "quit\n", "excludes every FFN")
        run("fp16-tactic", "gtp", "cudaInt8Scope=ffn", "quit\n", "INT8 requires KATAGO_CUDA_DUALFFN=0", {"KATAGO_CUDA_DUALFFN": "1"})
        report["status"] = "PASS"
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        raise
    finally:
        write_json(args.output / "report.json", report)
    print("PASS: GTP, analysis and incompatible configuration rejection")


if __name__ == "__main__":
    main()
