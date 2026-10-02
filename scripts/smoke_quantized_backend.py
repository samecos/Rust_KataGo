#!/usr/bin/env python3
"""Exercise the unified backend through GTP/analysis and reject wrong identities.

No production service is contacted; all child processes are owned by this run.
This checks integration and admission, not quantization accuracy or performance.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import re

from compare_worker_outputs import file_hash
from tune_runtime import ROOT, clean_environment, validate_gtp_smoke
from validate_quantized_backend import (MIXED_PROJECTIONS, PROFILE_LINE, config_path,
                                       run_logged, save_json, strict_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mixed-precision", choices=("int8", "mxfp8"), default="int8",
                        help="Experimental projection route; this remains an integration smoke, not an accuracy gate")
    args = parser.parse_args()
    args.binary, args.model, args.output = (p.resolve() for p in (args.binary, args.model, args.output))
    args.output.mkdir(parents=True, exist_ok=False)
    environment = clean_environment()
    report = dict(status="RUNNING", binary_sha256=file_hash(args.binary),
                  model_sha256=file_hash(args.model), mixed_precision=args.mixed_precision,
                  cases=[], production_certified=False)
    recipe_dir = args.output / "inspection"
    run_logged([str(args.binary), "quant-inspect", "--model", str(args.model), "--output", str(recipe_dir)],
               args.output / "inspect-command", environment, 120)
    template = strict_json((recipe_dir / "fp16-recipe.json").read_bytes())

    def recipe(name, value):
        path = args.output / f"{name}.json"
        save_json(path, value)
        return path

    fp16 = recipe("fp16", template)
    mixed = copy.deepcopy(template)
    if args.mixed_precision == "mxfp8":
        mixed.update(version=2, quantization_semantics_version=2)
    for projection in mixed["projections"]:
        if projection["id"] in MIXED_PROJECTIONS:
            projection["precision"] = args.mixed_precision
    assert sum(p["precision"] == args.mixed_precision for p in mixed["projections"]) == 4
    mixed_path = recipe("mixed", mixed)

    def run(name, overrides=None, mode="gtp", stdin="1 quit\n", expected=None, extra_env=None):
        import os
        import subprocess
        options = dict(rules="chinese", komi="7.5", nnBackend="cudaquantbackend", nnMaxBatchSize="8",
                       numNNServerThreadsPerModel="1", numSearchThreads="2", maxVisits="8",
                       nnCacheSizePowerOfTwo="10", nnMutexPoolSizePowerOfTwo="8", logToStderr="true")
        options.update(overrides or {})
        cfg = args.output / f"{name}.cfg"
        cfg.write_text("".join(f"{key}={value}\n" for key, value in options.items()), encoding="utf-8")
        command = [str(args.binary), mode, "--model", str(args.model), "--config", str(cfg)]
        env = dict(environment, **(extra_env or {}))
        result = subprocess.run(command, input=stdin, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=120, cwd=ROOT, env=env,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        (args.output / f"{name}.stdout.log").write_text(result.stdout, encoding="utf-8")
        (args.output / f"{name}.stderr.log").write_text(result.stderr, encoding="utf-8")
        text = result.stdout + "\n" + result.stderr
        if expected is None:
            assert result.returncode == 0, f"{name}: exit {result.returncode}; inspect logs"
            if mode == "gtp":
                assert re.search(r"(?m)^=\s*[14]\s*$", result.stdout), f"{name}: no GTP response"
            identities = set(PROFILE_LINE.findall(text))
            assert len(identities) == 1, f"{name}: missing actual execution identity"
            identity = next(iter(identities))[-1]
        else:
            assert result.returncode != 0 and expected in text, f"{name}: wrong admission result; inspect logs"
            identity = None
        report["cases"].append(dict(name=name, passed=True, command=command, expected_error=expected,
                                    inference_profile_id=identity))
        save_json(args.output / "report.json", report)
        return result.stdout, text, identity

    try:
        commands = "1 protocol_version\n2 boardsize 19\n3 genmove b\n4 quit\n"
        out, _, builtin_id = run("builtin-fp16", stdin=commands)
        validate_gtp_smoke(out)
        _, _, explicit_id = run("explicit-fp16", dict(cudaQuantPlan=config_path(fp16)))
        assert builtin_id == explicit_id, "equivalent FP16 recipes must share canonical identity"
        out, log, mixed_id = run("mixed-gtp", dict(cudaQuantPlan=config_path(mixed_path)), stdin=commands)
        validate_gtp_smoke(out)
        assert f"launch=cublaslt-{args.mixed_precision}" in log and mixed_id != builtin_id
        if args.mixed_precision == "mxfp8":
            wrong_version = copy.deepcopy(mixed)
            wrong_version.update(version=1, quantization_semantics_version=1)
            run("mxfp8-rejects-legacy-version",
                dict(cudaQuantPlan=config_path(recipe("mxfp8-legacy-version", wrong_version))),
                expected="unsupported precision recipe schema/version/quantization semantics version")
        request = dict(id="quant-smoke", moves=[], rules="chinese", komi=7.5,
                       boardXSize=19, boardYSize=19, maxVisits=8, includeOwnership=True)
        out, _, _ = run("mixed-analysis", dict(cudaQuantPlan=config_path(mixed_path)), mode="analysis",
                        stdin=json.dumps(request) + "\n")
        replies = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
        response = next(row for row in replies if row.get("id") == "quant-smoke" and "rootInfo" in row)
        assert response["rootInfo"]["visits"] > 0 and len(response["ownership"]) == 361
        assert all(math.isfinite(v) for v in response["ownership"])
        _, _, singleton_id = run("actual-batch-one", dict(nnMaxBatchSize="1"))
        assert singleton_id != builtin_id, "actual batch must bind execution identity"
        run("wrong-batch-profile", dict(nnMaxBatchSize="1", cudaQuantExpectedProfile=builtin_id),
            expected="quantized inference profile mismatch")
        run("accepted-profile", dict(cudaQuantExpectedProfile=builtin_id))
        run("legacy-recipe", dict(nnBackend="cudabackend", cudaQuantPlan=config_path(fp16)),
            expected="requires nnBackend=cudaquantbackend")
        run("legacy-plan", dict(cudaTacticPlan="unused.json"), expected="incompatible option cudaTacticPlan")
        run("legacy-scope", dict(cudaInt8Scope="ffn"), expected="incompatible option cudaInt8Scope")
        run("missing-recipe", dict(cudaQuantPlan=config_path(args.output / "does-not-exist.json")),
            expected="cudaQuantPlan")
        wrong_model = copy.deepcopy(template)
        wrong_model["model_sha256"] = "0" * 64
        run("wrong-model", dict(cudaQuantPlan=config_path(recipe("wrong-model-recipe", wrong_model))),
            expected="model_sha256 mismatch")
        wrong_shape = copy.deepcopy(template)
        wrong_shape["projections"][0]["expected_n"] += 8
        run("wrong-shape", dict(cudaQuantPlan=config_path(recipe("wrong-shape-recipe", wrong_shape))),
            expected="shape mismatch")
        run("runtime-tuning", extra_env={"KATAGO_CUDA_INT8_GEMM_TUNE": "1"},
            expected="requires deterministic algorithm selection")
        report["status"] = "PASS_INTEGRATION_NOT_ACCURACY_CERTIFICATION"
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        raise
    finally:
        save_json(args.output / "report.json", report)
    print(f"PASS: {len(report['cases'])} GTP/analysis/identity admission cases")


if __name__ == "__main__":
    main()
