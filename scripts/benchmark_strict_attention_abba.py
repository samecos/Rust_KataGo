#!/usr/bin/env python3
"""Native Windows/WSL strict-attention B14/S2 full-forward ABBA.

Run only after both same-platform native raw and Worker C++ FP32 gates pass.
W80/N1000, ABBA, event-median rate sums, 5% spread and 1% gain are fixed before
measurement. No compilation, platform wrappers, plan changes or GPU fallback.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess

import benchmark_fork_parity as parity

ROOT = Path(__file__).resolve().parents[1]
NATIVE_ROOT = ROOT / "target/fork-parity-20260908/g2-strict-attention-integration-r1"
NATIVE_CONTRACT_SHA = "f2ca6e74a6c0846d670232ceb3a8c37932ced32a6e995265c7054b3d6768e6ba"
PRESET = "fa4-strict-b14-r1"
ASSET = "sha256:e142b91fe865445357aa754015d4ed9693ff8f9aa4a65eb631dbb9183c17fbc0"
ORDER = ["baseline", "candidate", "candidate", "baseline"]
BATCH, LANES, WARMUP, ITERATIONS = 14, 2, 80, 1000
MAX_SPREAD, MIN_GAIN = 0.05, 0.01
SETTINGS = {
    "KATAGO_CUDA_DUALFFN": "1", "KATAGO_CUDA_CUBLASLT": "1",
    "KATAGO_CUDA_RESIDUAL_ALGO": "heuristic", "KATAGO_CUDA_CUBLASLT_RANK": "heuristic",
    "KATAGO_CUDA_GEMM_LAYOUT": "tn", "KATAGO_CUDA_ATTN_TILE": "q64-serial",
    "KATAGO_CUDA_NOGRAPH": "1", "KATAGO_CUDA_PADBATCH": "0",
    "KATAGO_CUDA_NOPIPELINE": "0", "KATAGO_CUDA_SPLITK": "0",
    "KATAGO_CUDA_T32": "0", "KATAGO_CUDA_T64N32": "0", "KATAGO_CUDA_FUSION": "none",
}
require = parity.require


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, f"duplicate JSON/marker key: {key}")
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=invalid_constant)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def imported(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tactics(flavor):
    require(flavor in ("baseline", "candidate"), "unregistered arm")
    return {**SETTINGS, "KATAGO_CUDA_ATTN": "fa2" if flavor == "baseline" else PRESET}


def environment(flavor, platform, inherited=None):
    result = {k: v for k, v in (os.environ if inherited is None else inherited).items()
              if not k.upper().startswith("KATAGO_")}
    if platform == "wsl":
        result["LD_LIBRARY_PATH"] = "/usr/local/cuda-13.3/lib64:/usr/lib/wsl/lib"
    return {**result, **tactics(flavor)}


def native_evidence(path, binary):
    """Recompute frozen numerical evidence on CPU; never call native main()."""
    contract_path, runner_path = NATIVE_ROOT / "contract.json", NATIVE_ROOT / "validate_native.py"
    require(sha(contract_path) == NATIVE_CONTRACT_SHA, "native numeric contract changed")
    contract = read(contract_path)
    require(sha(runner_path) == contract["runner_sha256"], "native runner changed")
    native = imported(runner_path, "strict_attention_native_gates")
    platform = native.native_platform(binary)
    report = read(path)
    require(report.get("schema") == 1 and report.get("status") == "PASS_ALL_SIX_NATIVE_RAW_GATES" and report.get("pass") is True,
            "all six native raw gates must PASS before ABBA")
    require(report.get("platform") == platform and report.get("native_host_only") is True,
            "numeric report is not from this native platform")
    require(report.get("is_cpp_golden_reference") is False and report.get("production_certified") is False,
            "native report scope changed")
    require(report["contract_sha256"] == NATIVE_CONTRACT_SHA
            and report["runner_sha256"] == contract["runner_sha256"], "numeric provenance changed")
    require(report["strict_attention_artifact"] == contract["strict_attention_artifact"] == ASSET
            and native.artifact_identity() == ASSET, "strict AOT artifact changed")
    require(report["source_sha256"] == contract["source_sha256"], "numeric source contract differs")
    records = {}
    def bind(p, expected=None):
        p = Path(p).resolve()
        actual = sha(p)
        require(expected is None or actual == expected, f"evidence SHA mismatch: {p}")
        records[str(p)] = actual
    for p in (path, contract_path, runner_path, Path(__file__), Path(parity.__file__)):
        bind(p)
    bind(report["binary"], report["binary_sha256"])
    require(native.native_platform(Path(report["binary"])) == platform, "numeric executable platform differs")
    for name, expected in contract["source_sha256"].items():
        bind(ROOT / name, expected)
    for name in ("attention.sm120.cubin", "rope_no_vcopy.sm120.ptx", "abi.json"):
        bind(ROOT / "crates/kata_nn/cuda-aot" / PRESET / name)
    require(report["comparator"] == contract["comparator"], "numeric comparator differs")
    comparator_path = ROOT / contract["comparator"]["path"]
    bind(comparator_path, contract["comparator"]["sha256"])
    compare = imported(comparator_path, "strict_attention_raw_comparator")
    require(compare.GATES == report["absolute_gates"] == contract["absolute_gates"], "numeric gates changed")
    cache = {}
    def dump(directory):
        directory = Path(directory).resolve()
        if str(directory) not in cache:
            data = compare.load_dump(directory)
            bind(directory / "meta.json", data["meta_sha256"])
            for row in data["meta"]["rows"]:
                for item in row["files"].values():
                    bind(directory / item["path"], item["sha256"])
            cache[str(directory)] = data
        return cache[str(directory)]
    expected_labels = [f"{arm}-b{batch}-l{lanes}" for arm in ("baseline", "candidate")
                       for batch, lanes in native.SHAPES]
    require([r["label"] for r in report["runs"]] == expected_labels, "numeric six-run order/coverage differs")
    builds, devices, expected_comparisons = [], [], {}
    for run in report["runs"]:
        require(run["status"] == "PASS" and run["returncode"] == 0, "incomplete numeric child")
        candidate = run["label"].startswith("candidate-")
        batch, lanes = run["batch"], run["lanes"]
        require(run["candidate"] is candidate and run["label"] == f"{'candidate' if candidate else 'baseline'}-b{batch}-l{lanes}",
                "numeric arm/shape metadata differs")
        bind(run["log"], run["log_sha256"])
        actual = dump(run["dump"])
        require(actual["meta_sha256"] == run["meta_sha256"], "numeric meta changed")
        markers = native.markers(Path(run["log"]).read_text(encoding="utf-8", errors="replace"), batch, candidate)
        require(markers == run["markers"], "numeric recorded markers differ from log")
        old = contract["references"][platform][f"b{batch}-l{lanes}"]
        reference = dump(ROOT / old["path"])
        require(reference["meta_sha256"] == old["meta_sha256"], "old native reference changed")
        native.validate_meta(actual, batch, lanes, candidate, Path(report["binary"]), contract, reference)
        builds.append(actual["meta"]["backend_build_fingerprint"])
        devices.append(actual["meta"]["device_fingerprint"])
        targets = [("same-binary-baseline", Path(path).parent / f"baseline-b{batch}-l{lanes}", batch != 14)] if candidate else [
            ("old-same-batch", ROOT / old["path"], True)]
        targets.append(("old-b1", ROOT / contract["references"][platform]["b1-l1"]["path"], False))
        for name, ref_path, bitwise in targets:
            expected_comparisons[f"{run['label']}-vs-{name}.json"] = (actual, dump(ref_path), bitwise)
    require(len(report["comparisons"]) == 12, "numeric twelve-comparison coverage incomplete")
    seen = set()
    for entry in report["comparisons"]:
        p = Path(entry["path"])
        require(p.name in expected_comparisons and p.name not in seen, "unexpected/duplicate numeric comparison")
        seen.add(p.name)
        bind(p, entry["sha256"])
        c = read(p)
        actual, reference, bitwise = expected_comparisons[p.name]
        require(entry["pass"] is True and c["pass"] is True and c["is_cpp_golden_reference"] is False,
                "numeric comparison is not a completed internal PASS")
        require(entry["same_batch_bitwise_required"] is bitwise and c["same_batch_bitwise_required"] is bitwise,
                "native bitwise boundary changed")
        require(c["absolute_gates"] == compare.GATES and c["candidate"] == compare.summary(actual)
                and c["reference"] == compare.summary(reference), "comparison input/gate binding differs")
        for key, data in (("reference_self_consistency", reference), ("candidate_vs_reference", actual)):
            recomputed = compare.compare_rows(data, reference["canonical"])
            require(recomputed["pass"] is True and c[key] == recomputed, "raw numerical recomputation failed: " + key)
        if bitwise:
            recomputed = native.raw_bitwise(actual, reference, compare.HEADS)
            require(recomputed["pass"] is True and c["same_batch_bitwise"] == recomputed, "native bitwise recomputation failed")
    require(all(b == builds[0] for b in builds) and all(d == devices[0] for d in devices), "six native build/device fingerprints differ")
    return dict(platform=platform, build=builds[0], device=devices[0], records=records,
                report=str(Path(path).resolve()), report_sha256=sha(path), test_binary_sha256=report["binary_sha256"],
                strict_attention_artifact=ASSET, is_cpp_golden_reference=False)


def parse_nnbench(stdout):
    decoder = json.JSONDecoder(object_pairs_hook=pairs, parse_constant=invalid_constant)
    found = []
    for start in re.finditer(r"(?m)^[ \t]*\{", stdout):
        try:
            data, _ = decoder.raw_decode(stdout[start.start():].lstrip())
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("command") == "nnbench":
            found.append(data)
    require(len(found) == 1, "expected one complete nnbench report")
    return found[0]


def path_markers(log, flavor):
    parsed = [pairs(re.findall(r"([A-Za-z0-9_]+)=([^\s]+)", line.split("[cuda-tactic]", 1)[1]))
              for line in log.splitlines() if "[cuda-tactic]" in line]
    attention = [m for m in parsed if m.get("name") == "attention"]
    require(len(attention) == 1, "expected one actual fixed-B14 attention scope")
    m = attention[0]
    expected = dict(requested="fa2", launch="fa2", effective="fa2", tile="q64-serial") if flavor == "baseline" else dict(
        requested=PRESET, launch="strict-aot", effective=PRESET, physical_batch="14", artifact=ASSET,
        identity="matched", rope="exact-qk-only", v="original-packed", abi="1")
    require(all(m.get(k) == v for k, v in expected.items()) and "fallback" not in m,
            "attention did not execute the exact registered path; fallback cannot count as candidate")
    layouts = [m for m in parsed if m.get("name") == "gemm_layout"]
    require(layouts and all(m.get("launch") == m.get("effective") == m.get("requested") == "tn" for m in layouts),
            "TN layout was not used exclusively")
    require(not any(m.get("name") == "residual_algo" for m in parsed), "residual preset was active")
    require(any(m.get("name") == "cublaslt_rank" and m.get("engine") == "heuristic" for m in parsed), "heuristic rank missing")
    require(any(m.get("name") == "dual_ffn" and m.get("launch") == "fused" and m.get("effective") == "1" for m in parsed), "DualFFN did not launch")
    for kind in ("plain", "f16out", "residual"):
        require(any(m.get("name") == "gemm" and m.get("kind") == kind and m.get("engine") == "cublaslt" for m in parsed), "missing cuBLASLt " + kind)
    return parsed


def validate_result(data, log, flavor, model, build):
    require(data.get("command") == "nnbench" and data.get("mode") == "kernel" and data.get("timing") == "cuda-event"
            and data.get("kernel_only") is True, "full kernel-forward cuda-event mode required")
    require(parity.same_path(data.get("model"), model) and data.get("model_sha256") == parity.MODEL_SHA, "model differs")
    require(data.get("input") == "empty" and data.get("warmup") == WARMUP and data.get("iterations") == ITERATIONS,
            "input/W80/N1000 contract differs")
    require(data.get("requested_batches") == [BATCH] and data.get("handles") == [1, 2], "physical B14/S2 request differs")
    require(data.get("device_target") == "sm_120" and data.get("build") == build, "actual build differs from numerical gates")
    require(build.get("strict_attention_artifact") == ASSET and type(build.get("fp16_encoding_revision")) is int
            and build["fp16_encoding_revision"] == 1 and build.get("cublaslt_version") == 130600,
            "AOT/FP16/cuBLASLt contract differs")
    require({k: v for k, v in data.get("tactics", {}).items() if v is not None} == tactics(flavor), "hidden or wrong effective tactics")
    d = data.get("input_description", {})
    require(d.get("kind") == "empty" and d.get("inputs_version") == 7 and d.get("board") == [19, 19]
            and d.get("rules") == parity.EMPTY_RULES and d.get("misc_params") == parity.EMPTY_MISC
            and d.get("encore_phase") == 0 and d.get("effective_symmetry") == 0 and d.get("policy_optimism") == 0,
            "empty-board semantic input changed")
    rows = data.get("results", [])
    require(len(rows) == 1 and rows[0].get("batch") == BATCH, "actual physical batch differs")
    row = rows[0]
    for key in ("spatial_input_sha256", "global_input_sha256"):
        require(isinstance(row.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", row[key]), "invalid input SHA: " + key)
    require([h.get("handle") for h in row.get("handles", [])] == [1, 2], "actual lanes differ")
    wall = parity.positive(row.get("wall_elapsed_ms"), "full wall duration")
    rates = []
    for h in row["handles"]:
        values = h.get("cuda_event_ms", [])
        require(h.get("iterations") == ITERATIONS and len(values) == ITERATIONS, "missing original event samples")
        median = statistics.median([parity.positive(v, "raw event time") for v in values])
        parity.matching_rate(h.get("cuda_event_median_ms"), median, "lane median")
        rate = BATCH * 1000 / median
        parity.matching_rate(h.get("cuda_event_median_nn_evals_per_s"), rate, "lane median rate")
        elapsed = parity.positive(h.get("elapsed_ms"), "lane wall time")
        require(elapsed <= wall * (1 + 1e-7), "lane wall exceeds full envelope")
        parity.matching_rate(h.get("per_batch_ms"), elapsed / ITERATIONS, "lane wall batch time")
        parity.matching_rate(h.get("nn_evals_per_s"), BATCH * ITERATIONS * 1000 / elapsed, "lane wall rate")
        rates.append(rate)
    parity.matching_rate(row.get("sum_median_nn_evals_per_s"), sum(rates), "sum of lane median rates")
    wall_rate = LANES * BATCH * ITERATIONS * 1000 / wall
    parity.matching_rate(row.get("wall_nn_evals_per_s"), wall_rate, "full wall rate")
    return dict(event_rate_sum=sum(rates), wall_rate=wall_rate, markers=path_markers(log, flavor),
                input_identity=[row["spatial_input_sha256"], row["global_input_sha256"], d])


def summarize(runs):
    require([r["flavor"] for r in runs] == ORDER, "complete ABBA order required")
    metrics = {}
    for metric in ("event_rate_sum", "wall_rate"):
        arms = {}
        for flavor in ("baseline", "candidate"):
            samples = [parity.positive(r[metric], metric) for r in runs if r["flavor"] == flavor]
            require(len(samples) == 2, "two independent processes per arm required")
            arms[flavor] = dict(samples=samples, geomean=statistics.geometric_mean(samples),
                                relative_spread=max(samples) / min(samples) - 1)
        stable = all(a["relative_spread"] <= MAX_SPREAD for a in arms.values())
        ratio = arms["candidate"]["geomean"] / arms["baseline"]["geomean"]
        metrics[metric] = dict(arms=arms, stable=stable)
        metrics[metric]["candidate_over_baseline_ratio" if stable else "diagnostic_only_ratio"] = ratio
    primary = metrics["event_rate_sum"]
    status = ("UNSTABLE_DIAGNOSTIC_ONLY" if not primary["stable"] else
              "PASS_GAIN_GATE" if primary["candidate_over_baseline_ratio"] >= 1 + MIN_GAIN else "REJECT_BELOW_MINIMUM_GAIN")
    return dict(performance_status=status, primary_metric="event_rate_sum", metrics=metrics,
                maximum_relative_spread=MAX_SPREAD, minimum_gain=MIN_GAIN,
                wall_scope="separate full-forward envelope; recorded even if it disagrees with primary event metric",
                production_certified=False)


def child(command, stem, env):
    """Direct native child only; timeout kills/reaps it and preserves partial logs."""
    with stem.with_suffix(".stdout.log").open("xb") as stdout, stem.with_suffix(".stderr.log").open("xb") as stderr:
        result = subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                timeout=600, check=False)
    require(result.returncode == 0, f"native child failed ({result.returncode}): {stem}")
    return (stem.with_suffix(".stdout.log").read_text(encoding="utf-8", errors="replace"),
            stem.with_suffix(".stderr.log").read_text(encoding="utf-8", errors="replace"))


def command(binary, model):
    return list(map(str, [binary, "nnbench", "--model", model, "--mode", "kernel", "--input", "empty",
                         "--batch", BATCH, "--handles", "1,2", "--timing", "cuda-event", "--warmup", WARMUP,
                         "--iterations", ITERATIONS, "--json"]))


def worker_evidence(path, binary, native):
    report = read(path)
    require(report.get("schema") == 1 and report.get("status") == "PASS_BOTH_WORKER_CPP_FP32"
            and report.get("pass") is True, "both Worker C++ FP32 profiles must PASS before ABBA")
    require(report["binary_sha256"] == report["binary"]["sha256"] == sha(binary),
            "Worker golden gate is from another binary/platform")
    require(Path(report["binary"]["path"]).resolve() == binary.resolve(), "Worker executable path differs")
    require(report["strict_attention_artifact"] == ASSET, "Worker artifact differs")
    records = {str(Path(path).resolve()): sha(path)}
    def bind(entry):
        p = Path(entry["path"]).resolve()
        require(sha(p) == entry["sha256"], "Worker evidence SHA mismatch: " + str(p))
        records[str(p)] = entry["sha256"]
    for key in ("binary", "golden", "stage_manifest", "runtime_fingerprint"):
        bind(report[key])
    require(report["stage_manifest_sha256"] == report["stage_manifest"]["sha256"], "Worker stage SHA differs")
    stage = read(report["stage_manifest"]["path"])
    for key in ("runner", "binary", "python", "model", "golden", "fingerprint", "proto", "fixtures"):
        bind(stage[key])
    for key in ("tools", "native_evidence", "source_files"):
        for entry in stage[key]:
            bind(entry)
    require(stage["binary"] == report["binary"] and stage["golden"] == report["golden"]
            and stage["source_files"] == report["source_files"], "Worker stage/report identity differs")
    require(stage["golden"]["sha256"] == "f21a005c1927cb591f4e8eafd15dc12394bd680ef9417547f910b095bce07855"
            and stage["model"]["sha256"] == parity.MODEL_SHA, "Worker model/C++ FP32 reference differs")
    require(any(e["sha256"] == native["report_sha256"] and Path(e["path"]).resolve() == Path(native["report"])
                for e in stage["native_evidence"]), "Worker was not validated against this platform's native gate")
    require(all(native["records"].get(str(Path(e["path"]).resolve())) == e["sha256"] for e in report["source_files"]),
            "Worker and native source provenance differ")
    fingerprint = read(report["runtime_fingerprint"]["path"])
    require(fingerprint == read(stage["fingerprint"]["path"]), "Worker runtime/staged fingerprint differs")
    validate_fingerprint(fingerprint, native)
    # This gate's runner contains no import-time GPU work. Its source bytes are
    # bound above; re-use its exact plan/config/marker checks without main().
    validator = imported(stage["runner"]["path"], "strict_attention_worker_gates")
    require(validator.TACTICS == SETTINGS and validator.ASSET == ASSET, "Worker tactic contract differs")
    comparator_path = ROOT / "scripts/compare_worker_outputs.py"
    require(str(comparator_path.resolve()) in records, "Worker comparator is not bound by the staged tool hashes")
    comparator = imported(comparator_path, "strict_attention_worker_comparator")
    golden = read(report["golden"]["path"])
    require(golden["status"] == "PASS" and golden["drained"] is True and len(golden["results"]) == 128
            and re.search(r"(?m)^cudaUseFP16\s*=\s*false\s*$", golden["config_text"]), "incomplete C++ FP32 golden")
    require([p["flavor"] for p in report["profiles"]] == ["baseline", "candidate"], "Worker profile coverage/order incomplete")
    for p in report["profiles"]:
        flavor = p["flavor"]
        require(all(p.get(key) == "PASS" for key in ("status", "numeric_status", "path_status"))
                and p["binary_sha256"] == sha(binary) and p["tactics"] == tactics(flavor)
                and p["policy_top1_matches"] == p["policy_top1_total"] == 128 and p["request_window"] == 32,
                "Worker profile did not satisfy all registered gates")
        for key in ("plan", "config", "comparison", "outputs", "worker_log"):
            bind(p[key])
        require({k: p[k] for k in ("plan_id", "plan", "config")} == stage["profiles"][flavor], "Worker stage profile changed")
        validator.validate_plan(read(p["plan"]["path"]), fingerprint, flavor)
        validator.validate_config(Path(p["config"]["path"]).read_text(encoding="utf-8"), p["plan"]["path"])
        actual, comparison = read(p["outputs"]["path"]), read(p["comparison"]["path"])
        require(actual["status"] == "PASS" and actual["drained"] is True and actual["cases"] == len(actual["results"]) == 128
                and actual["binary_sha256"] == sha(binary) and actual["config_sha256"] == p["config"]["sha256"]
                and actual["physical_request_window"] == 32, "Worker collection/identity incomplete")
        require(actual["config_text"] == Path(p["config"]["path"]).read_text(encoding="utf-8"), "actual Worker config changed")
        for key in ("model_sha256", "schema_sha256", "fixtures_sha256", "requests_sha256"):
            require(actual[key] == comparison[key] == golden[key], "Worker semantic identity differs: " + key)
        recomputed = comparator.compare_outputs(golden, actual)
        require(recomputed["result"] == "PASS" and recomputed["policy_top1_matches"] == 128
                and all(comparison[k] == v for k, v in recomputed.items()), "Worker C++ FP32 recomputation failed")
        log = Path(p["worker_log"]["path"]).read_text(encoding="utf-8", errors="replace")
        require(validator.markers(log, p, flavor == "candidate") == p["paths"], "Worker actual path markers differ")
        h = actual["final_heartbeat"]
        rows, batches = int(h["nn_rows"]), int(h["nn_batches"])
        require(rows == int(h["completed_requests"]) == 128 and int(h["failed_requests"]) == int(h["in_flight"]) == 0
                and 1 <= batches <= 128 and rows <= 14 * batches, "Worker did not finish uncached and drain")
        require(p["batching"]["nn_rows"] == rows and p["batching"]["nn_batches"] == batches
                and p["batching"]["average_batch"] == rows / batches, "Worker mixed batch arithmetic differs")
    return dict(report=str(Path(path).resolve()), report_sha256=sha(path), records=records,
                binary_sha256=sha(binary), cpp_fp32_golden_pass=True, request_window=32,
                physical_batch_scope="mixed; B14 measured distribution not asserted", fingerprint=fingerprint)


def validate_fingerprint(fingerprint, native):
    require(fingerprint.get("architecture") == "sm_120" and fingerprint.get("model_sha256") == parity.MODEL_SHA,
            "main runtime model/architecture differs")
    require(fingerprint.get("backend_build") == native["build"], "main and numeric build fingerprints differ")
    require({k: fingerprint.get(k) for k in native["device"]} == native["device"], "main and numeric GPU differ")


def preflight(args):
    binary = args.binary.resolve()
    native = native_evidence(args.numeric_report.resolve(), binary)
    model = (args.model or Path("D:/Go/Server/models" if native["platform"] == "win" else "/mnt/d/Go/Server/models") / parity.MODEL.name).resolve()
    require(sha(model) == parity.MODEL_SHA, "actual model bytes differ")
    worker = worker_evidence(args.worker_numeric_report.resolve(), binary, native)
    return dict(binary=str(binary), binary_sha256=sha(binary), model=str(model), model_sha256=parity.MODEL_SHA,
                platform=native["platform"], native=native, worker=worker)


def stable_inputs(evidence):
    require(sha(evidence["binary"]) == evidence["binary_sha256"], "same main binary changed")
    require(sha(evidence["model"]) == evidence["model_sha256"], "model changed")
    for domain in ("native", "worker"):
        for path, expected in evidence[domain]["records"].items():
            require(sha(path) == expected, "frozen prerequisite changed: " + path)


def registration():
    return dict(batch=BATCH, lanes=LANES, warmup=WARMUP, iterations=ITERATIONS,
                maximum_relative_spread=MAX_SPREAD, minimum_gain=MIN_GAIN,
                metric="sum per lane 14000 / median(all 1000 original CUDA event milliseconds)",
                secondary_metric="28000 rows / full wall envelope seconds",
                artifact=ASSET, tactics={f: tactics(f) for f in ("baseline", "candidate")})


def run(args):
    # Fresh directory first, so failed preflight also leaves reviewable evidence.
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema=1, status="PREFLIGHT", workload_status="NOT_RUN", production_certified=False,
                  started_utc=datetime.now(timezone.utc).isoformat(), runs=[], order=ORDER,
                  registration=registration(),
                  scope="same-binary full kernel forward; native raw and mixed Worker C++ FP32 prerequisites are separate; no production certification",
                  process_policy="four synchronous native child processes; no WSL wrapper; native timeout kills and reaps child; partial logs retained",
                  source_binding_scope="frozen source files are build/execution provenance, not main-binary embedded source attestation")
    report_path = output / "report.json"
    write(report_path, report)
    try:
        evidence = preflight(args)
        report.update(evidence=evidence, status="RUNNING")
        write(report_path, report)
        stable_inputs(evidence)
        fp_command = [evidence["binary"], "cuda-fingerprint", "--model", evidence["model"]]
        stdout, _ = child(fp_command, output / "runtime-fingerprint", environment("baseline", evidence["platform"]))
        fingerprint = json.loads(stdout, object_pairs_hook=pairs, parse_constant=invalid_constant)
        validate_fingerprint(fingerprint, evidence["native"])
        require(fingerprint == evidence["worker"]["fingerprint"], "actual main runtime differs from Worker golden runtime")
        report["runtime_fingerprint"] = dict(command=fp_command, result=fingerprint,
                                             stdout_sha256=sha(output / "runtime-fingerprint.stdout.log"))
        write(report_path, report)
        identity = None
        for index, flavor in enumerate(ORDER, 1):
            stable_inputs(evidence)
            stem = output / f"{index}-{flavor}"
            cmd = command(evidence["binary"], evidence["model"])
            entry = dict(index=index, flavor=flavor, status="RUNNING", command=cmd,
                         tactics=tactics(flavor), stdout=str(stem.with_suffix(".stdout.log")),
                         stderr=str(stem.with_suffix(".stderr.log")))
            report["runs"].append(entry)
            write(report_path, report)
            print(f"{evidence['platform']} {index}/4 {flavor} B14/S2 W80/N1000", flush=True)
            stdout, stderr = child(cmd, stem, environment(flavor, evidence["platform"]))
            entry.update(stdout_sha256=sha(entry["stdout"]), stderr_sha256=sha(entry["stderr"]))
            data = parse_nnbench(stdout)
            checked = validate_result(data, stdout + "\n" + stderr, flavor, evidence["model"], evidence["native"]["build"])
            require(identity is None or identity == checked["input_identity"], "actual input hashes/semantics differ across ABBA")
            identity = checked["input_identity"]
            entry.update(status="PASS", result=data, **checked)
            stable_inputs(evidence)
            write(report_path, report)
            print(f"events {checked['event_rate_sum']:.3f} rows/s; full wall {checked['wall_rate']:.3f} rows/s", flush=True)
        summary = summarize(report["runs"])
        report.update(status="COMPLETED", workload_status="PASS", summary=summary)
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as error:
        report.update(status="FAIL_ABORTED_NO_LATER_PHASES", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write(report_path, report)
    return report


def verify(path):
    """CPU-only reconstruction from logs, original samples and prerequisite bytes."""
    report = read(path)
    require(report["status"] == "COMPLETED" and report["workload_status"] == "PASS", "benchmark incomplete")
    require(report["registration"] == registration() and report["order"] == ORDER
            and report["production_certified"] is False, "registered measurement scope changed")
    evidence = report["evidence"]
    args = argparse.Namespace(binary=Path(evidence["binary"]), model=Path(evidence["model"]),
                              numeric_report=Path(evidence["native"]["report"]),
                              worker_numeric_report=Path(evidence["worker"]["report"]))
    require(preflight(args) == evidence, "prerequisite reconstruction differs")
    fp = report["runtime_fingerprint"]
    p = Path(path).parent / "runtime-fingerprint.stdout.log"
    require(sha(p) == fp["stdout_sha256"] and read(p) == fp["result"] == evidence["worker"]["fingerprint"],
            "runtime fingerprint evidence changed")
    identity = None
    for i, entry in enumerate(report["runs"], 1):
        require(entry["index"] == i and entry["status"] == "PASS" and entry["command"] == command(evidence["binary"], evidence["model"]),
                "recorded run/command changed")
        require(entry["tactics"] == tactics(entry["flavor"]), "recorded run tactics changed")
        for key in ("stdout", "stderr"):
            require(sha(entry[key]) == entry[key + "_sha256"], "raw log changed")
        stdout = Path(entry["stdout"]).read_text(encoding="utf-8", errors="replace")
        stderr = Path(entry["stderr"]).read_text(encoding="utf-8", errors="replace")
        require(parse_nnbench(stdout) == entry["result"], "saved raw samples differ from original stdout")
        checked = validate_result(entry["result"], stdout + "\n" + stderr, entry["flavor"], evidence["model"], evidence["native"]["build"])
        require(all(entry[k] == v for k, v in checked.items()), "sample/median/wall reconstruction differs")
        require(identity is None or identity == checked["input_identity"], "ABBA input hashes differ")
        identity = checked["input_identity"]
    require(report["summary"] == summarize(report["runs"]), "geomean/spread/gain decision reconstruction differs")
    return dict(status="PASS_CPU_RECONSTRUCTION", report_sha256=sha(path), summary=report["summary"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("check", "run"):
        p = sub.add_parser(name)
        p.add_argument("--binary", type=Path, required=True)
        p.add_argument("--numeric-report", type=Path, required=True)
        p.add_argument("--worker-numeric-report", type=Path, required=True)
        p.add_argument("--model", type=Path)
        if name == "run":
            p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("verify")
    p.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.action == "run":
        run(args)
    elif args.action == "check":
        evidence = preflight(args)
        print(json.dumps(dict(status="PASS_CPU_PREFLIGHT_NO_GPU", platform=evidence["platform"],
                              binary_sha256=evidence["binary_sha256"], native_report_sha256=evidence["native"]["report_sha256"],
                              worker_report_sha256=evidence["worker"]["report_sha256"]), indent=2))
    else:
        print(json.dumps(verify(args.report), indent=2))


if __name__ == "__main__":
    main()
