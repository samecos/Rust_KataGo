#!/usr/bin/env python3
"""Validate unified recipes using sequential, isolated local gRPC workers.

The legacy FP16 control must first pass the unchanged C++ FP32 reference gates
at every requested window. Then collect legacy FFN INT8, unified FP16, unified
FFN INT8, an independently mixed projection recipe, and optional --recipe files.
At request window 1, unified FP16/FFN migrations require exact protobuf NNOutput
bytes against their legacy counterparts. Concurrent windows record exactness
and require the unchanged numerical gates against the corresponding legacy
backend, since request scheduling can change physical batches. Recipe errors use
the same-model FP16 baseline, with
an experimental maximum win-probability difference of 0.06; policy, score and
ownership metrics and the original FP32 gates are reported separately.

No production worker or server is contacted. Each collection owns a temporary
127.0.0.1 gRPC server and one child worker, drained before the next is launched.
This is an offline experiment, not calibration, Elo testing, or certification.
Requires grpcio/grpcio-tools and an explicit or bundled native TF3 FP32 reference.
--pad-to-batch explicitly enables the existing PADBATCH tactic for every process
and requires exact migration at every window. That tactic pads multirow batches
to capacity but retains B1 for singletons; it does not prove batch pairing.
--trace-batches records physical batch/row placement and raw output hashes for
diagnostics only. Traced runs are never performance evidence.
"""

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
WIN_PROBABILITY_MAX_ABS = 0.06
MIXED_PROJECTIONS = {
    "trunk.block00.pair00.ffn.dual",
    "trunk.block00.pair01.ffn.down",
    "trunk.block00.pair00.attention.qkv",
    "trunk.block00.pair01.attention.out",
}
PROFILE_LINE = re.compile(
    r"\[cuda-quant\]\s+model_sha256=([0-9a-f]{64})\s+"
    r"graph_sha256=([0-9a-f]{64})\s+recipe_sha256=([0-9a-f]{64})\s+"
    r"inference_profile=(\S+)\s+validation=unverified"
)


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"nonfinite JSON number: {value}")

    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)


def run_logged(command, directory, environment, timeout, input_text=None):
    directory.mkdir(parents=True, exist_ok=False)
    save_json(directory / "command.json", dict(command=command, input=input_text,
                                               timeout_seconds=timeout))
    with (directory / "stdout.log").open("wb") as stdout, \
            (directory / "stderr.log").open("wb") as stderr:
        # subprocess.run times out only this child; no process-name/service kill.
        result = subprocess.run(command, cwd=ROOT, env=environment,
                                input=None if input_text is None else input_text.encode("utf-8"),
                                stdout=stdout, stderr=stderr, timeout=timeout,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if result.returncode:
        raise RuntimeError(f"command exited {result.returncode}; see {directory}")
    return tuple((directory / name).read_text(encoding="utf-8", errors="replace")
                 for name in ("stdout.log", "stderr.log"))


def config_path(path):
    value = path.resolve().as_posix()
    if any(character in value for character in '#"\r\n'):
        raise ValueError(f"path cannot be represented unambiguously in a config: {path}")
    return f'"{value}"'


def create_recipes(args, model_sha256, environment):
    inspection = args.output / "inspection"
    run_logged([str(args.binary), "quant-inspect", "--model", str(args.model),
                "--output", str(inspection)], args.output / "quant-inspect",
               environment, args.startup_timeout)
    template = strict_json((inspection / "fp16-recipe.json").read_bytes())
    manifest = strict_json((inspection / "model-manifest.json").read_bytes())
    if template["model_sha256"] != model_sha256 or manifest["model_sha256"] != model_sha256:
        raise AssertionError("quant-inspect model identity disagrees with source bytes")
    if template["graph_sha256"] != manifest["graph_sha256"]:
        raise AssertionError("quant-inspect template/manifest graph identity mismatch")
    projections = template["projections"]
    if not projections or any(p["precision"] != "fp16" for p in projections):
        raise AssertionError("quant-inspect did not return an explicit all-FP16 template")
    ids = {p["id"] for p in projections}
    if not MIXED_PROJECTIONS <= ids:
        raise ValueError(f"model lacks mixed smoke projections: {sorted(MIXED_PROJECTIONS - ids)}")

    directory = args.output / "recipes"
    directory.mkdir()
    profiles = []
    for name in ("new-fp16", "new-ffn", "mixed"):
        recipe = copy.deepcopy(template)
        for projection in recipe["projections"]:
            if ((name == "new-ffn" and ".ffn." in projection["id"])
                    or (name == "mixed" and projection["id"] in MIXED_PROJECTIONS)):
                projection["precision"] = "int8"
        if name != "new-fp16" and not any(p["precision"] == "int8" for p in recipe["projections"]):
            raise AssertionError(f"{name}: empty INT8 selection")
        path = directory / f"{name}.json"
        save_json(path, recipe)
        profiles.append(dict(name=name, recipe=path, int8=name != "new-fp16"))
    for index, source in enumerate(args.recipe, 1):
        raw = source.read_bytes()
        recipe = strict_json(raw)
        # Preserve the caller's exact bytes; the Rust loader validates the full
        # schema, source/graph identity, projection IDs and actual dimensions.
        path = directory / f"user-{index:02d}.json"
        path.write_bytes(raw)
        profiles.append(dict(name=f"user-{index:02d}", recipe=path, source=str(source),
                             int8=any(p.get("precision") == "int8"
                                      for p in recipe.get("projections", []))))
    return profiles, manifest


def discover_profile(args, label, cfg, model_sha256, environment):
    initial_config = cfg.read_text(encoding="utf-8")
    stdout, stderr = run_logged(
        [str(args.binary), "gtp", "--model", str(args.model), "--config", str(cfg)],
        args.output / f"{label}-discovery", environment, args.startup_timeout, "1 quit\n")
    if re.search(r"(?m)^=\s*1\s*$", stdout) is None:
        raise AssertionError(f"{label}: missing successful GTP quit response")
    identities = set(PROFILE_LINE.findall(stdout + "\n" + stderr))
    if len(identities) != 1:
        raise AssertionError(f"{label}: expected one actual loaded profile, found {len(identities)}")
    model, graph, recipe, actual = identities.pop()
    if model != model_sha256 or re.fullmatch(r"rustgo-quant-v1:[0-9a-f]{64}", actual) is None:
        raise AssertionError(f"{label}: unexpected actual inference identity")
    discovery = dict(model_sha256=model, graph_sha256=graph, recipe_sha256=recipe,
                     inference_profile_id=actual, config_before_expected_profile=initial_config,
                     validation="UNVERIFIED")
    save_json(args.output / f"{label}-discovery" / "identity.json", discovery)
    # GTP and Worker see identical batch capacity/server threads and tactic
    # environment. The expectation is derived from loading, never guessed.
    cfg.write_text(initial_config + f"cudaQuantExpectedProfile={actual}\n", encoding="utf-8")
    return discovery


def exact_outputs(protocol, output_directory, expected, actual):
    mismatches = []
    expected_digest, actual_digest = hashlib.sha256(), hashlib.sha256()
    for left, right in zip(expected["results"], actual["results"], strict=True):
        if left["name"] != right["name"] or left["result"]["task_id"] != right["result"]["task_id"]:
            raise AssertionError("exact comparison request order/identity changed")
        task_id = int(left["result"]["task_id"])
        outputs = []
        for report in (expected, actual):
            raw = (output_directory / report["worker"] / f"{task_id:03d}.pb").read_bytes()
            result = protocol.pb.EvalResult.FromString(raw)
            if result.task_id != task_id or result.error_code or not result.HasField("output"):
                raise AssertionError("invalid saved protobuf result for exact comparison")
            outputs.append(result.output.SerializeToString(deterministic=True))
        for digest, raw in zip((expected_digest, actual_digest), outputs, strict=True):
            digest.update(len(raw).to_bytes(8, "little") + raw)
        if outputs[0] != outputs[1]:
            mismatches.append(dict(case=left["name"], task_id=task_id,
                                   reference_sha256=hashlib.sha256(outputs[0]).hexdigest(),
                                   candidate_sha256=hashlib.sha256(outputs[1]).hexdigest()))
    return dict(result="PASS" if not mismatches else "FAIL", cases=len(expected["results"]),
                reference_profile=expected["worker"], candidate_profile=actual["worker"],
                reference_outputs_sha256=expected_digest.hexdigest(),
                candidate_outputs_sha256=actual_digest.hexdigest(), mismatches=mismatches,
                scope="Exact deterministic protobuf NNOutput bytes, including float bits; "
                      "excludes EvalResult timings. These are postprocessed outputs, not raw logits.",
                batching_note="Same request window and capacity; actual dynamic batch scheduling "
                              "may differ between runs. Differences still fail this exact check.")


def batch_summary(outputs):
    heartbeat = outputs["final_heartbeat"]
    rows, batches = int(heartbeat["nn_rows"]), int(heartbeat["nn_batches"])
    trace = outputs.get("_batch_trace")
    return dict(nn_rows=rows, nn_batches=batches, mean_rows_per_batch=rows / batches,
                physical_batch_distribution=trace["physical_batch_distribution"] if trace else None,
                distribution_status="OBSERVED_FROM_POST_WARMUP_CUDA_BATCH_TRACE" if trace else
                                    "UNAVAILABLE: current WorkerHeartbeat reports aggregate rows/batches; "
                                    "EvalResult does not identify each request's physical batch")


def parse_batch_trace(log, expected_rows, capacity):
    warmups = list(re.finditer(r"CUDA handle warmup batches=[^\r\n]*completed[^\r\n]*", log))
    if not warmups:
        raise AssertionError("batch trace: missing completed warmup boundary; refusing to mix warmup and requests")
    tail = log[warmups[-1].end():]
    batches, by_input = [], defaultdict(list)
    distribution, logical_distribution = Counter(), Counter()
    for line in tail.splitlines():
        marker = "[cuda-batch-trace] "
        if marker not in line:
            continue
        batch = strict_json(line.split(marker, 1)[1])
        physical, logical, rows = batch.get("physical_batch"), batch.get("logical_rows"), batch.get("rows")
        if (type(physical) is not int or type(logical) is not int or not isinstance(rows, list)
                or not 1 <= logical <= physical <= capacity or len(rows) != logical):
            raise AssertionError("batch trace: invalid physical/logical dimensions")
        for index, row in enumerate(rows):
            if (row.get("row") != index or type(row.get("row")) is not int
                    or any(not isinstance(row.get(key), str)
                           or re.fullmatch(r"[0-9a-f]{64}", row[key]) is None
                           for key in ("input_sha256", "raw_output_sha256"))):
                raise AssertionError("batch trace: invalid row index/hash")
            by_input[row["input_sha256"]].append(dict(batch_index=len(batches), physical_batch=physical,
                                                      row=index, raw_output_sha256=row["raw_output_sha256"]))
        batches.append(batch)
        distribution[physical] += 1
        logical_distribution[logical] += 1
    logical_rows = sum(batch["logical_rows"] for batch in batches)
    if logical_rows != expected_rows:
        raise AssertionError(f"batch trace: expected {expected_rows} post-warmup rows, got {logical_rows}")
    return dict(schema="rustgo-cuda-batch-trace-v1", status="DIAGNOSTIC_ONLY",
                warmup_boundaries_seen=len(warmups), warmup_traces_excluded=log[:warmups[-1].end()].count("[cuda-batch-trace] "),
                logical_rows=logical_rows, observed_batches=len(batches), unique_inputs=len(by_input),
                physical_batch_distribution=dict(sorted(distribution.items())),
                logical_batch_distribution=dict(sorted(logical_distribution.items())),
                inputs=dict(sorted(by_input.items())), batches=batches,
                scope="Actual encoded inputs including symmetry/optimism and five raw output tensors; "
                      "only records following the final completed handle warmup are included.",
                performance_evidence=False)


def compare_batch_traces(reference, candidate):
    left, right = reference["inputs"], candidate["inputs"]
    shared = sorted(left.keys() & right.keys())
    matched, mismatches, repeated_inconsistency = [], [], []
    different_batch, different_row, different_context = [], [], []
    unmatched_left = unmatched_right = 0
    for input_sha in shared:
        if Counter(r["physical_batch"] for r in left[input_sha]) != Counter(r["physical_batch"] for r in right[input_sha]):
            different_batch.append(input_sha)
        if Counter(r["row"] for r in left[input_sha]) != Counter(r["row"] for r in right[input_sha]):
            different_row.append(input_sha)
        groups = []
        for entries in (left[input_sha], right[input_sha]):
            grouped = defaultdict(list)
            for entry in entries:
                grouped[(entry["physical_batch"], entry["row"])].append(entry["raw_output_sha256"])
            groups.append(grouped)
        a, b = groups
        if Counter({key: len(values) for key, values in a.items()}) != Counter({key: len(values) for key, values in b.items()}):
            different_context.append(input_sha)
        for context in sorted(a.keys() | b.keys()):
            expected, actual = a.get(context, []), b.get(context, [])
            count = min(len(expected), len(actual))
            unmatched_left += len(expected) - count
            unmatched_right += len(actual) - count
            if len(set(expected)) > 1 or len(set(actual)) > 1:
                repeated_inconsistency.append(dict(input_sha256=input_sha, physical_batch=context[0], row=context[1],
                                                   reference_raw_hashes=expected, candidate_raw_hashes=actual))
            for occurrence, (ref_hash, got_hash) in enumerate(zip(expected, actual)):
                entry = dict(input_sha256=input_sha, physical_batch=context[0], row=context[1], occurrence=occurrence,
                             reference_raw_output_sha256=ref_hash, candidate_raw_output_sha256=got_hash,
                             exact=ref_hash == got_hash)
                matched.append(entry)
                if not entry["exact"]:
                    mismatches.append(entry)
    missing = sorted(left.keys() - right.keys())
    extra = sorted(right.keys() - left.keys())
    unmatched_left += sum(len(left[key]) for key in missing)
    unmatched_right += sum(len(right[key]) for key in extra)
    result = "NO_MATCHED_ROWS" if not matched else "MISMATCH" if mismatches or repeated_inconsistency else "EXACT_FOR_MATCHED_ROWS"
    return dict(status="DIAGNOSTIC_ONLY", matched_raw_hash_result=result,
                matched_rows=len(matched), matched_exact_rows=sum(entry["exact"] for entry in matched),
                matched=matched, mismatches=mismatches, repeated_context_inconsistency=repeated_inconsistency,
                unique_shared_inputs=len(shared), missing_inputs=missing, extra_inputs=extra,
                unmatched_reference_rows=unmatched_left, unmatched_candidate_rows=unmatched_right,
                different_physical_batch_inputs=len(different_batch), different_row_inputs=len(different_row),
                different_batch_or_row_inputs=len(different_context),
                input_shas_with_different_physical_batch=different_batch,
                input_shas_with_different_row=different_row,
                input_shas_with_different_batch_or_row=different_context,
                scope="Match input SHA + physical batch + row; preserve occurrence order for duplicate keys. "
                      "No raw equivalence claim for unmatched contexts; this diagnostic does not change acceptance gates.")


def physical_batch_policy(batch, pad_to_batch):
    return dict(mode="PAD_MULTIROW_TO_CAPACITY" if pad_to_batch else "EXACT_LOGICAL_BATCH",
                configured_capacity=batch, singleton_physical_batch=1,
                multirow_physical_batch=batch if pad_to_batch else "logical batch size",
                padded_rows="copy final real row" if pad_to_batch else "none",
                tactic_overrides={"KATAGO_CUDA_PADBATCH": "1"} if pad_to_batch else {},
                per_request_batch_assignment="NOT_OBSERVED_BY_CURRENT_PROTOCOL")


def migration_requires_exact(window, batch, pad_to_batch):
    return window == 1 or batch == 1 or pad_to_batch


def check_execution(outputs, log, expected_backend, discovery=None, require_int8=False):
    info = outputs["hello"]["backend_info"]
    if re.search(rf"(?:^|;\s*)backend={re.escape(expected_backend)}(?:;|$)", info) is None:
        raise AssertionError(f"unexpected worker backend identity: {info}")
    if discovery:
        profiles = re.findall(r"(?:^|;\s*)inference-profile=([^;\s]+)", info)
        if profiles != [discovery["inference_profile_id"]]:
            raise AssertionError("worker actual inference profile differs from GTP discovery")
        loaded = set(PROFILE_LINE.findall(log))
        expected = (discovery["model_sha256"], discovery["graph_sha256"],
                    discovery["recipe_sha256"], discovery["inference_profile_id"])
        if loaded != {expected}:
            raise AssertionError("worker log does not confirm the actual model/recipe/profile identity")
    if require_int8 and "launch=cublaslt-int8" not in log:
        raise AssertionError("missing actual INT8 GEMM execution marker")
    if expected_backend == "cudaint8backend" and "int8-scope=ffn;" not in info:
        raise AssertionError("legacy control did not report FFN INT8 scope")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Validated C++ FP32 reference; otherwise use bundled model match")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are refused")
    parser.add_argument("--recipe", type=Path, action="append", default=[], help="Additional recipe JSON; repeatable")
    parser.add_argument("--batch", type=int, default=8, help="NN batch capacity, 1..64; identical for discovery and Worker")
    parser.add_argument("--windows", default="1,32", help="Unique comma-separated in-flight request windows, 1..128; must include 1")
    parser.add_argument("--pad-to-batch", action="store_true",
                        help="Diagnostic: PADBATCH=1 for all processes; multirow batches pad to capacity, "
                             "singletons stay B1; require exact migration at every window")
    parser.add_argument("--trace-batches", action="store_true",
                        help="Diagnostic: trace actual post-warmup batch/row/raw-output hashes; never performance evidence")
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--task-timeout", type=float, default=180)
    args = parser.parse_args()
    try:
        args.windows = [int(value) for value in args.windows.split(",")]
    except ValueError:
        parser.error("--windows must contain comma-separated integers")
    if (1 not in args.windows or len(args.windows) != len(set(args.windows))
            or not all(1 <= value <= 128 for value in args.windows)
            or not 1 <= args.batch <= 64):
        parser.error("batch must be 1..64; windows must be unique values in 1..128 and include 1 for mandatory exact migration")
    if not all(math.isfinite(value) and value > 0 for value in (args.startup_timeout, args.task_timeout)):
        parser.error("timeouts must be finite and positive")
    args.binary, args.model, args.output = (path.resolve() for path in (args.binary, args.model, args.output))
    args.recipe = [path.resolve() for path in args.recipe]
    if args.reference:
        args.reference = args.reference.resolve()
    for path in [args.binary, args.model, *args.recipe, *([args.reference] if args.reference else [])]:
        if not path.is_file():
            parser.error(f"file not found: {path}")
    if args.output.exists():
        parser.error(f"output directory already exists: {args.output}")
    return args


def main():
    args = parse_args()
    # --help remains usable without grpcio; execution uses the existing strict
    # collector/reference validators, not a second protocol implementation.
    import autotune_reference as reference_io
    from compare_worker_outputs import build_requests, collect_worker, compare_outputs, file_hash
    from tune_runtime import clean_environment
    from validate_int8_backend import quantization_metrics
    from worker_protocol_tools import Protocol

    model_sha = file_hash(args.model)
    reference_path = args.reference or reference_io.load_bundled(model_sha)
    if reference_path is None:
        raise ValueError("no matching C++ FP32 reference; provide --reference")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema="rustgo-quantized-validation-v1", model=str(args.model), model_sha256=model_sha,
                  binary=str(args.binary), binary_sha256=file_hash(args.binary), batch=args.batch,
                  windows=args.windows, reference=str(reference_path), reference_sha256=file_hash(reference_path),
                  same_model_fp16_win_probability_limit=WIN_PROBABILITY_MAX_ABS,
                  status="RUNNING", results=[], production_certified=False,
                  playing_strength="NOT_MEASURED", performance="NOT_MEASURED",
                  calibration="NOT_PERFORMED", server_scope="temporary isolated loopback harness only",
                  physical_batch_policy=physical_batch_policy(args.batch, args.pad_to_batch),
                  batch_trace_enabled=args.trace_batches,
                  model_identity_note="Original source SHA remains unchanged; profile expectation is local "
                                      "and is not production server capability negotiation.")
    protocol = None
    try:
        protocol = Protocol(ROOT / "crates/kata_worker/proto/worker.proto")
        fixture = strict_json((ROOT / "scripts/fixtures/worker_positions.json").read_bytes())
        requests = build_requests(protocol.pb, fixture, model_sha, 0)
        fingerprints = reference_io.fingerprints(requests, model_sha)
        report["fingerprints"] = fingerprints
        reference = reference_io.validate_reference(reference_io.read_reference(reference_path),
                                                    fingerprints, requests, protocol)
        environment = clean_environment()
        environment.update(report["physical_batch_policy"]["tactic_overrides"])
        if args.trace_batches:
            environment["KATAGO_CUDA_BATCH_TRACE"] = "1"
        report["controlled_environment"] = {key: value for key, value in environment.items() if key.upper().startswith("KATAGO_")}
        base = (f"rules=chinese\nkomi=7.5\nnnMaxBatchSize={args.batch}\n"
                "numNNServerThreadsPerModel=1\nnumSearchThreads=1\nmaxVisits=1\n"
                "ponderingEnabled=false\nlogToStderr=true\nnnCacheSizePowerOfTwo=10\n"
                "nnMutexPoolSizePowerOfTwo=8\n")

        def collect(name, window, backend, recipe=None):
            label = f"{name}-w{window}"
            cfg = args.output / f"{label}.cfg"
            text = base + f"nnBackend={backend}\n"
            if backend == "cudaint8backend":
                text += "cudaInt8Scope=ffn\ncudaInt8MinFfnWidth=0\n"
            if recipe:
                text += f"cudaQuantPlan={config_path(recipe)}\n"
            cfg.write_text(text, encoding="utf-8")
            discovery = discover_profile(args, label, cfg, model_sha, environment) if recipe else None
            options = SimpleNamespace(output=args.output, model=args.model, request_window=window,
                                      startup_timeout=args.startup_timeout, task_timeout=args.task_timeout)
            outputs = collect_worker(label, args.binary, cfg, options, protocol, requests, fingerprints, environment)
            if outputs["hello"]["model_version"] != reference["hello"]["model_version"]:
                raise AssertionError("reference and worker disagree on actual model version")
            log = (args.output / label / "worker.log").read_text(encoding="utf-8", errors="replace")
            check_execution(outputs, log, backend, discovery,
                            require_int8=backend == "cudaint8backend")
            original = compare_outputs(reference, outputs)
            save_json(args.output / label / "original-fp32-gate.json", original)
            result = dict(profile=label, original_fp32_gate=original, actual_identity=discovery,
                          batching=batch_summary(outputs),
                          original_fp32_metrics=quantization_metrics(reference, outputs))
            report["results"].append(result)
            save_json(args.output / "report.json", report)
            if args.trace_batches:
                trace = parse_batch_trace(log, len(requests), args.batch)
                trace["heartbeat_rows_match"] = trace["logical_rows"] == int(outputs["final_heartbeat"]["nn_rows"])
                trace["heartbeat_batches_match"] = trace["observed_batches"] == int(outputs["final_heartbeat"]["nn_batches"])
                outputs["_batch_trace"] = trace
                result["batching"] = batch_summary(outputs)
                result["batch_trace_file"] = str(args.output / label / "batch-trace.json")
                save_json(args.output / label / "batch-trace.json", trace)
                save_json(args.output / "report.json", report)
            return outputs, result, log

        # All controls must pass before any quantized loading/execution starts.
        baselines = {}
        for window in args.windows:
            outputs, result, _ = collect("baseline-fp16", window, "cudabackend")
            if result["original_fp32_gate"]["result"] != "PASS":
                raise AssertionError(f"baseline FP16 w{window} failed original C++ FP32 gates; stopping")
            baselines[window] = outputs

        profiles, manifest = create_recipes(args, model_sha, environment)
        report["graph_sha256"] = manifest["graph_sha256"]
        report["recipes"] = [dict(profile=p["name"], path=str(p["recipe"]),
                                  source=p.get("source"), file_sha256=file_hash(p["recipe"])) for p in profiles]
        experimental_pass = True
        for window in args.windows:
            baseline = baselines[window]
            legacy, result, _ = collect("legacy-ffn-int8", window, "cudaint8backend")
            result["same_model_fp16_metrics"] = quantization_metrics(baseline, legacy)
            for profile in profiles:
                outputs, result, log = collect(profile["name"], window, "cudaquantbackend", profile["recipe"])
                if profile["int8"] and "launch=cublaslt-int8" not in log:
                    raise AssertionError(f"{profile['name']}: missing actual INT8 GEMM execution")
                relative = quantization_metrics(baseline, outputs)
                gate = dict(result="PASS" if relative["win_probability_max_abs"] <= WIN_PROBABILITY_MAX_ABS else "FAIL",
                            metric="white_win_prob maximum absolute difference vs same-model FP16",
                            limit=WIN_PROBABILITY_MAX_ABS, measured=relative["win_probability_max_abs"],
                            scope="Fixed 128 semantic fixtures only; experimental gate, not playing strength")
                result.update(same_model_fp16_metrics=relative, same_model_fp16_experimental_gate=gate,
                              same_model_fp16_all_fields=compare_outputs(baseline, outputs))
                save_json(args.output / outputs["worker"] / "same-model-fp16.json",
                          dict(metrics=relative, experimental_gate=gate,
                               all_fields=result["same_model_fp16_all_fields"]))
                experimental_pass &= gate["result"] == "PASS"
                if profile["name"] in ("new-fp16", "new-ffn"):
                    control = baseline if profile["name"] == "new-fp16" else legacy
                    exact = exact_outputs(protocol, args.output, control, outputs)
                    # Window 1 has at most one request at a time. Batch capacity
                    # 1 also forces singleton batches at concurrent windows.
                    # Otherwise the protocol cannot prove per-request matching
                    # physical batches, even when aggregate counts happen to match.
                    # The explicit padding diagnostic demands exactness even
                    # though singleton-vs-multirow assignment remains unknown.
                    exact_required = migration_requires_exact(window, args.batch, args.pad_to_batch)
                    exact["required_for_migration_gate"] = exact_required
                    strict = compare_outputs(control, outputs)
                    migration = dict(result=exact["result"] if exact_required else strict["result"],
                                     mode="EXACT_OUTPUT_BYTES" if exact_required else "UNCHANGED_NUMERICAL_GATES",
                                     reference_profile=control["worker"],
                                     reference_batching=batch_summary(control), candidate_batching=batch_summary(outputs),
                                     physical_batch_policy=report["physical_batch_policy"],
                                     physical_batch_matching="SINGLETON_BY_CONFIGURATION" if window == 1 or args.batch == 1
                                     else "NOT_OBSERVABLE_WITH_CURRENT_PROTOCOL",
                                     exact_requirement_reason="explicit padding diagnostic" if args.pad_to_batch
                                     else "singleton configuration" if exact_required else "observed only")
                    result.update(exact_legacy_migration_observed=exact,
                                  strict_legacy_migration_gate=strict, legacy_migration_gate=migration)
                    if args.trace_batches:
                        raw = compare_batch_traces(control["_batch_trace"], outputs["_batch_trace"])
                        result["matched_batch_raw_diagnostic"] = raw
                        save_json(args.output / outputs["worker"] / "matched-batch-raw-diagnostic.json", raw)
                    save_json(args.output / outputs["worker"] / "exact-migration.json", exact)
                    save_json(args.output / outputs["worker"] / "migration-gate.json",
                              dict(gate=migration, unchanged_numerical_gates=strict))
                    if migration["result"] != "PASS":
                        raise AssertionError(f"{outputs['worker']}: legacy migration failed; see migration-gate.json")
                save_json(args.output / "report.json", report)
                print(json.dumps(dict(profile=outputs["worker"], original_fp32_gate=result["original_fp32_gate"]["result"],
                                      same_model_fp16_experimental_gate=gate, metrics=relative), ensure_ascii=False), flush=True)
        report["status"] = "EXPERIMENTAL_GATES_PASSED" if experimental_pass else "EXPERIMENTAL_GATE_FAILED"
        return 0 if experimental_pass else 1
    except Exception as error:
        report.update(status="FAILED", error=str(error))
        raise
    finally:
        if protocol is not None:
            protocol.close()
        save_json(args.output / "report.json", report)


if __name__ == "__main__":
    raise SystemExit(main())
