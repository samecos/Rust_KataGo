#!/usr/bin/env python3
"""CPU-only, append-only registration of bounded quantization experiments.

This initial ledger supports existing v1 FP16/INT8 planner seeds only. It never
runs a Worker, proposes candidates, reads holdout outputs or publishes READY.
The read-only summarize-selection command closes the finite registered set;
positive comparisons remain unranked candidates requiring independent review.
Numeric evidence is recomputed from original protobufs. ABBA
evidence is checked against those collections and its four raw timing files.
Failed attempts consume their reserved budget; there is no retry command.

The local hash chain is an audit aid, not an external signature. Keep the head
SHA printed by each command separately and use --expect-head to detect rollback.
All original sources must remain available and unchanged. A crash leaves the
exclusive lock in place for manual investigation; locks are never stolen.
"""

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace

import plan_quantization_search as planner

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rustgo-quantization-selection-ledger-v1"
ZERO = "0" * 64
require = planner.require
compact = lambda value: planner.compact(value, sorted_keys=True)
digest = planner.digest
read_json = lambda path: planner.strict_json(Path(path).read_bytes())
SOURCE_NAMES = ("quantization_selection_ledger.py", "plan_quantization_search.py",
                "collect_quantization_corpus.py", "compare_quantization_corpus.py",
                "benchmark_quantization_corpus.py", "benchmark_workers.py",
                "worker_protocol_tools.py", "compare_worker_outputs.py",
                "validate_quantized_backend.py", "tune_runtime.py", "legacy_quantization_execution.py")
WORKLOAD_FIELDS = {"schema", "split", "metric", "concurrency", "capacity", "warmup", "cycles",
                   "task_timeout", "max_numeric_evaluations", "max_performance_evaluations",
                   "minimum_improvement", "maximum_relative_spread", "smoke"}


def same(left, right):
    return compact(left) == compact(right)


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(planner.pretty(value))
        stream.flush()
        os.fsync(stream.fileno())


def file_record(path):
    path = Path(path).resolve()
    require(path.is_file(), f"missing source: {path}")
    return dict(path=str(path), bytes=path.stat().st_size, sha256=planner.file_digest(path))


def verify_record(item):
    require(same(file_record(item["path"]), item), f"immutable source changed: {item['path']}")


def safe_file(root, relative):
    return planner.local_path(root, relative)


def source_versions():
    return [file_record(ROOT / "scripts" / name) for name in SOURCE_NAMES] + [
        file_record(ROOT / "crates/kata_worker/proto/worker.proto")]


def validate_search(search_path, scratch_parent, sgf_root=None):
    """Regenerate the planner's entire contract from original inputs, not PASS text."""
    search_path = Path(search_path).resolve()
    plan = read_json(search_path)
    require(plan.get("schema") == planner.SCHEMA and plan.get("version") == 1, "unsupported search plan")
    paths, identity = plan["source_paths"], plan["identity"]
    with tempfile.TemporaryDirectory(prefix="ledger-plan-check-", dir=scratch_parent) as temp:
        rebuilt = planner.run(SimpleNamespace(
            model=Path(paths["model"]), model_manifest=Path(paths["model_manifest"]),
            fp16_template=Path(paths["fp16_template"]), corpus_manifest=Path(paths["corpus_manifest"]),
            sgf_root=sgf_root, output=Path(temp) / "rebuilt", seed=identity["seed"],
            max_candidates=identity["max_candidates"], max_finalists=identity["max_finalists"]))
    require(same(plan, rebuilt), "search plan differs from independently regenerated source contract")
    sources = [file_record(search_path)]
    for entry in plan["artifacts"]:
        path = safe_file(search_path.parent, entry["file"])
        actual = file_record(path)
        require(actual["sha256"] == entry["sha256"] and actual["bytes"] == entry["bytes"],
                "search artifact hash/size mismatch")
        sources.append(actual)
    sources.extend(file_record(path) for path in paths.values())
    corpus_path = Path(paths["corpus_manifest"])
    corpus, _, _, _, _, sgfs = planner.validate_corpus(corpus_path, sgf_root)
    sources.extend(file_record(safe_file(corpus_path.parent, item["file"])) for item in corpus["artifacts"])
    sgf_directory = (sgf_root or Path(corpus["options"]["input"])).resolve()
    sources.extend(file_record(safe_file(sgf_directory, path)) for path in sgfs)
    return plan, sources


def validate_workload(workload, plan):
    planner.object_fields(workload, WORKLOAD_FIELDS, "workload")
    require(workload["schema"] == "rustgo-quantization-ledger-workload-v1", "unsupported workload schema")
    require(workload["split"] in ("calibration", "selection"), "holdout is forbidden")
    require(workload["metric"] in ("rpc_throughput", "p95_latency"), "unsupported metric")
    require(type(workload["smoke"]) is bool, "smoke must be boolean")
    require(workload["smoke"] or (workload["split"] == "selection" and plan["corpus_status"] == "READY"),
            "NOT_READY/non-selection requires a frozen smoke workload")
    for key, maximum in (("concurrency", 256), ("capacity", 4096), ("cycles", 1000),
                         ("max_numeric_evaluations", 8), ("max_performance_evaluations", 8),
                         ("warmup", 1000000)):
        planner.positive_integer(workload[key], key, maximum)
    require(workload["concurrency"] <= workload["capacity"] and workload["warmup"] >= workload["concurrency"],
            "capacity/warmup does not cover concurrency")
    require(workload["smoke"] or workload["warmup"] >= 128, "formal warmup must be >=128")
    require(type(workload["task_timeout"]) in (int, float) and math.isfinite(workload["task_timeout"])
            and 0 < workload["task_timeout"] <= 3600, "invalid task timeout")
    require(workload["minimum_improvement"] == 0.01 and workload["maximum_relative_spread"] == 0.05,
            "initial ledger fixes improvement/spread at 1%/5%")


def initialize(search_path, workload_path, baselines_path, output, sgf_root=None):
    output = Path(output).resolve()
    require(not output.exists() and output.parent.is_dir(), "output must be a NEW directory with an existing parent")
    plan, sources = validate_search(search_path, output.parent, sgf_root)
    workload, baselines = read_json(workload_path), read_json(baselines_path)
    validate_workload(workload, plan)
    planner.object_fields(baselines, {"schema", "fp16_reference_report_sha256", "baselines"}, "baseline catalog")
    require(baselines["schema"] == "rustgo-quantization-ledger-baselines-v1", "unsupported baseline catalog")
    planner.corpus_io.check_sha(baselines["fp16_reference_report_sha256"], "FP16 reference report SHA")
    require(type(baselines["baselines"]) is list, "baseline catalog must be a list")
    known = set(plan["identity"]["candidate_recipe_sha256"])
    seen = set()
    for entry in baselines["baselines"]:
        planner.object_fields(entry, {"recipe_sha256", "collection_report_sha256"}, "baseline entry")
        planner.corpus_io.check_sha(entry["collection_report_sha256"], "baseline report SHA")
        require(entry["recipe_sha256"] in known and entry["recipe_sha256"] not in seen,
                "unknown/repeated baseline recipe; v1 only accepts registered unified seeds")
        seen.add(entry["recipe_sha256"])
    sources.extend([file_record(workload_path), file_record(baselines_path), *source_versions()])
    sources = list({s["path"]: s for s in sources}.values())
    contract = dict(schema=SCHEMA, search_id=plan["search_id"], workload_id=digest(compact(workload)),
                    plan=plan, workload=workload, baselines=baselines, sources=sources,
                    sgf_root=str(sgf_root.resolve()) if sgf_root else None,
                    limitations=["v1 seeds only; no candidate generation/retry/finalist/holdout/publication",
                                 "baseline catalog is declared, not proof of optimality or FP32 certification",
                                 "local hash chain has no external signature; retain the head SHA separately"])
    output.mkdir()
    write_new(output / "contract.json", contract)
    (output / "events").mkdir()
    (output / "evidence").mkdir()
    append_event(output, contract, [], "INITIALIZED", dict(contract_sha256=planner.file_digest(output / "contract.json")))
    return inspect(output)


@contextmanager
def locked(root):
    lock = root / ".ledger.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(f"pid={os.getpid()}\n")
    try:
        yield
    finally:
        lock.unlink()


def append_event(root, contract, events, kind, payload, evidence=None):
    event = dict(schema=SCHEMA, sequence=len(events), previous_event_sha256=events[-1]["event_sha256"] if events else ZERO,
                 search_id=contract["search_id"], workload_id=contract["workload_id"],
                 event_type=kind, evidence=evidence or [], payload=payload)
    event["event_sha256"] = digest(compact(event))
    write_new(root / "events" / f"{len(events):06d}.json", event)
    events.append(event)
    return event


def replay(root, expect_head=None):
    root = Path(root).resolve()
    require(not (root / "contract.json").is_symlink()
            and all((root / name).is_dir() and not (root / name).is_symlink() for name in ("events", "evidence")),
            "ledger control paths must be local directories/files")
    contract = read_json(root / "contract.json")
    require(contract.get("schema") == SCHEMA, "unsupported ledger contract")
    for record in contract["sources"]:
        verify_record(record)
    events = []
    paths = sorted((root / "events").iterdir())
    require(paths, "ledger has no initialization event")
    for sequence, path in enumerate(paths):
        require(path.name == f"{sequence:06d}.json" and not path.is_symlink(), "missing/noncanonical ledger event")
        event = read_json(path)
        planner.object_fields(event, {"schema", "sequence", "previous_event_sha256", "search_id", "workload_id",
                                      "event_type", "evidence", "payload", "event_sha256"}, "ledger event")
        require(event["event_type"] in {"INITIALIZED", "REQUEST_REJECTED", "NUMERIC_RESERVED", "NUMERIC_RESULT",
                                         "PERFORMANCE_RESERVED", "PERFORMANCE_RESULT"}, "unsupported ledger event")
        claimed = event.pop("event_sha256")
        require(digest(compact(event)) == claimed, "event hash mismatch")
        require(event["schema"] == SCHEMA and event["sequence"] == sequence
                and event["previous_event_sha256"] == (events[-1]["event_sha256"] if events else ZERO)
                and event["search_id"] == contract["search_id"] and event["workload_id"] == contract["workload_id"],
                "event chain/identity mismatch")
        for artifact in event["evidence"]:
            raw = safe_file(root, artifact["relative_path"]).read_bytes()
            require(len(raw) == artifact["bytes"] and digest(raw) == artifact["sha256"], "frozen evidence changed")
        event["event_sha256"] = claimed
        events.append(event)
    require(events[0]["event_type"] == "INITIALIZED"
            and events[0]["payload"]["contract_sha256"] == planner.file_digest(root / "contract.json"),
            "contract is not bound to initialization")
    require(expect_head is None or events[-1]["event_sha256"] == expect_head, "ledger head differs from expected anchor")
    return contract, events


def derive_state(contract, events):
    used = {kind: {} for kind in ("NUMERIC", "PERFORMANCE")}
    results = {kind: {} for kind in used}
    evidence_ids = set()
    for event in events:
        kind = event["event_type"]
        payload = event["payload"]
        for stage in used:
            if kind == stage + "_RESERVED":
                require(payload["recipe_sha256"] in contract["plan"]["identity"]["candidate_recipe_sha256"], "unregistered recipe in ledger")
                require(payload["recipe_sha256"] not in used[stage], "candidate attempt reused in ledger")
                require(len(used[stage]) < contract["workload"]["max_" + stage.lower() + "_evaluations"], "ledger budget exceeded")
                used[stage][payload["recipe_sha256"]] = event["sequence"]
            if kind == stage + "_RESULT":
                require(payload["recipe_sha256"] in used[stage] and payload["recipe_sha256"] not in results[stage],
                        "unreserved/repeated result in ledger")
                results[stage][payload["recipe_sha256"]] = payload
                if payload.get("evidence_id"):
                    require(payload["evidence_id"] not in evidence_ids, "evidence replay in ledger")
                    evidence_ids.add(payload["evidence_id"])
    return used, results, evidence_ids


def inspect(root, expect_head=None):
    contract, events = replay(root, expect_head)
    used, results, _ = derive_state(contract, events)
    return dict(schema=SCHEMA, status="SMOKE_ONLY" if contract["workload"]["smoke"] else "EVIDENCE_ONLY_REVIEW_REQUIRED",
                search_id=contract["search_id"], workload_id=contract["workload_id"], head_sha256=events[-1]["event_sha256"],
                event_count=len(events), unique_candidates=len(contract["plan"]["candidates"]),
                numeric_attempts=len(used["NUMERIC"]), performance_attempts=len(used["PERFORMANCE"]),
                remaining_numeric_attempts=contract["workload"]["max_numeric_evaluations"] - len(used["NUMERIC"]),
                remaining_performance_attempts=contract["workload"]["max_performance_evaluations"] - len(used["PERFORMANCE"]),
                results=results, original_fp32_gate="NOT_EVALUATED", baseline_optimality="NOT_ESTABLISHED",
                holdout_outputs_read=0, deployment_adopted=False, publish_allowed=False, production_certified=False)


def reserve(root, contract, events, stage, recipe_sha):
    used, results, seen = derive_state(contract, events)
    require(recipe_sha in contract["plan"]["identity"]["candidate_recipe_sha256"], "unregistered recipe")
    require(recipe_sha not in used[stage], "candidate already reserved; aliases/retries cannot restore budget")
    limit = contract["workload"]["max_" + stage.lower() + "_evaluations"]
    require(len(used[stage]) < limit, f"{stage.lower()} budget exhausted")
    if stage == "PERFORMANCE":
        require(results["NUMERIC"].get(recipe_sha, {}).get("gate") == "PASS", "performance requires registered numeric PASS")
    append_event(root, contract, events, stage + "_RESERVED", dict(recipe_sha256=recipe_sha))
    return results, seen


def metadata_before_outputs(directory, contract):
    metadata = read_json(directory / "report.json")
    require(metadata.get("split") in ("calibration", "selection"), "holdout/unknown outputs are forbidden before reading any output")
    require(metadata.get("window_mode", "chunked") in ("chunked", "continuous"), "unknown collection window mode")
    work, plan = contract["workload"], contract["plan"]
    identity = plan["identity"]
    require(metadata["split"] == work["split"], "collection split differs from workload")
    require(metadata["model_sha256"] == identity["source_sha256"]["model"], "cross-model evidence")
    require(metadata["corpus_manifest_sha256"] == identity["source_sha256"]["corpus_manifest"]
            and metadata["corpus_request_set_sha256"] == identity["request_set_sha256"]
            and metadata["source_request_file_sha256"] == identity[work["split"] + "_requests_sha256"],
            "collection is not the full frozen split")
    require(metadata["backend"] == "cudaquantbackend" and metadata.get("actual_profile"),
            "initial ledger requires actual unified identity; legacy qualification is not implemented")
    require(metadata["request_window"] == work["concurrency"] and metadata["capacity"] == work["capacity"],
            "collection window/capacity differs from workload")
    return metadata


def collection_files(directory, contract):
    metadata = metadata_before_outputs(directory, contract)
    names = {"report.json", "worker.log"}
    if metadata.get("recipe_file_sha256"):
        names.add("recipe.json")
    for item in metadata["artifacts"]:
        require(item["file"] in {"outputs.jsonl", "requests-index.jsonl", "worker.cfg", "recipe.json", "worker.log", "continuous-buffer.json"},
                "unknown collection artifact; only v1 unified collection files are allowed")
        names.add(item["file"])
    if metadata.get("window_mode", "chunked") == "continuous":
        require("continuous-buffer.json" in names, "continuous collection must bind its buffer evidence")
    allowed_source = {"manifest.json", "games.jsonl", metadata["split"] + ".requests.jsonl"}
    require({item["file"] for item in metadata["source_artifacts"]} == allowed_source, "unexpected source artifact; holdout is forbidden")
    names.update("source/" + name for name in allowed_source)
    count = metadata["expected_requests"]
    require(type(count) is int and 0 < count <= 100000, "invalid collection count")
    names.update(f"{kind}/{i:06d}.pb" for kind in ("requests", "results") for i in range(1, count + 1))
    for name in names:
        path = directory / name
        require(not path.is_symlink() and not any(p.is_symlink() for p in path.parents if p.is_relative_to(directory)), "evidence symlink")
        safe_file(directory, name)
    return sorted(names)


def snapshot(root, sequence, label, directory, names):
    """Copy only after metadata rejects holdout; refuse links and path escapes."""
    records = []
    directory = directory.resolve()
    for name in names:
        path = directory / name
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(directory), "missing evidence/symlink/path escape")
        relative = Path("evidence") / f"{sequence:06d}" / label / path.relative_to(directory)
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        require(target.parent.resolve().is_relative_to(root.resolve()), "snapshot destination escaped ledger")
        raw = path.read_bytes()
        with target.open("xb") as stream:
            stream.write(raw)
        require(digest(path.read_bytes()) == digest(raw), "evidence changed during snapshot")
        records.append(dict(relative_path=relative.as_posix(), bytes=len(raw), sha256=digest(raw)))
    return records


def validate_numeric(contract, recipe_sha, reference_path, candidate_path, protocol, *, runtime_paths=None):
    import compare_quantization_corpus as comparison
    import benchmark_quantization_corpus as benchmark
    # Both metadata checks precede either protobuf/output read.
    reference_meta = metadata_before_outputs(reference_path, contract)
    candidate_meta = metadata_before_outputs(candidate_path, contract)
    require(planner.file_digest(reference_path / "report.json") == contract["baselines"]["fp16_reference_report_sha256"],
            "FP16 reference differs from frozen catalog")
    require(candidate_meta["actual_profile"]["recipe_sha256"] == recipe_sha, "candidate recipe identity mismatch")
    reference = comparison.load_collection(reference_path, protocol)
    candidate = comparison.load_collection(candidate_path, protocol, full_reference=reference["recipe"])
    full_fp16 = planner.canonical_recipe(read_json(Path(contract["plan"]["source_paths"]["fp16_template"])))
    require(same(reference["recipe"], full_fp16), "FP16 reference is not the complete frozen model template")
    registered = next(c for c in contract["plan"]["candidates"] if c["recipe_sha256"] == recipe_sha)
    search_dir = Path(contract["sources"][0]["path"]).parent
    require(same(candidate["recipe"], read_json(safe_file(search_dir, registered["recipe_file"]))),
            "candidate canonical recipe differs from initial registered seed")
    # A summary reads outputs from ledger snapshots, while checking the exact
    # original runtime files that the recorded configuration actually names.
    runtime_reference, runtime_candidate = runtime_paths or (reference_path, candidate_path)
    files = []
    for path, data in ((runtime_reference, reference), (runtime_candidate, candidate)):
        runtime = benchmark.runtime_files((path, data), None)[2]
        benchmark.check_files(runtime)
        files.extend(file_record(p) for p, _ in runtime)
        benchmark.arm_environment(data["report"])
    metric = comparison.compare(reference, candidate)
    metric.pop("cases")
    execution = {key: candidate_meta.get(key) for key in
                 ("binary_sha256", "model_sha256", "config_sha256", "actual_profile", "controlled_environment",
                  "runtime_artifacts", "batch", "capacity", "request_window", "schema_sha256", "batching")}
    execution["window_mode"] = candidate_meta.get("window_mode", "chunked")
    return dict(gate=metric["experimental_gate"]["result"], metrics=metric,
                execution_id=digest(compact(execution)), execution=execution,
                reference_report_sha256=reference["report_sha256"], collection_report_sha256=candidate["report_sha256"],
                collection_directory=str(runtime_candidate), reference_directory=str(runtime_reference),
                collection_buffer_contract=candidate_meta.get("collection_buffer_contract"), runtime_sources=files), reference, candidate


def validate_performance(contract, recipe_sha, directory, results, protocol, *, validated_numeric=None):
    import benchmark_quantization_corpus as benchmark
    from benchmark_workers import distribution
    from google.protobuf.json_format import ParseDict
    from tune_runtime import decide
    report = read_json(directory / "report.json")
    work = contract["workload"]
    require(report.get("schema") == "rustgo-quantization-corpus-abba-v1"
            and report.get("status") in ("SMOKE_ONLY", "PAIRWISE_MEASURED_REQUIRES_SELECTION_REVIEW"), "incomplete ABBA")
    require(report["status"] == ("SMOKE_ONLY" if work["smoke"] else "PAIRWISE_MEASURED_REQUIRES_SELECTION_REVIEW"), "ABBA status/smoke mismatch")
    require(report["holdout_outputs_read"] == 0 and report["order"] == list(benchmark.ORDER)
            and report["publish_allowed"] is False and report["production_certified"] is False
            and report["deployment_adopted"] is False, "ABBA scope/order differs")
    require(report["primary_metric"] == work["metric"] and report["cycles"] == work["cycles"]
            and report["warmup_requests"] == work["warmup"] and report["smoke"] is work["smoke"]
            and report["minimum_improvement"] == 0.01 and report["maximum_relative_spread"] == 0.05,
            "ABBA workload/threshold differs")
    candidate_entry = results["NUMERIC"][recipe_sha]
    require(len(report["runs"]) == 4, "ABBA requires four runs")
    baseline_sha = report["runs"][0]["collection_report_sha256"]
    baselines = [entry for entry in contract["baselines"]["baselines"] if entry["collection_report_sha256"] == baseline_sha]
    require(len(baselines) == 1, "ABBA baseline is not explicitly registered")
    baseline_recipe = baselines[0]["recipe_sha256"]
    require(baseline_recipe != recipe_sha, "candidate cannot benchmark against itself")
    baseline_entry = results["NUMERIC"].get(baseline_recipe, {})
    require(baseline_entry.get("gate") == "PASS" and baseline_entry["collection_report_sha256"] == baseline_sha,
            "baseline lacks registered numerical qualification")
    arm_data = {}
    for name, recipe, entry in (("baseline", baseline_recipe, baseline_entry), ("candidate", recipe_sha, candidate_entry)):
        if validated_numeric is None:
            checked, reference, collection = validate_numeric(contract, recipe, Path(entry["reference_directory"]),
                                                             Path(entry["collection_directory"]), protocol)
        else:
            checked, reference, collection = validated_numeric[recipe]
        require(checked["gate"] == "PASS" and checked["execution_id"] == entry["execution_id"]
                and checked["collection_report_sha256"] == entry["collection_report_sha256"], "numeric evidence/execution changed")
        require(work["smoke"] or checked["execution"]["window_mode"] == "continuous",
                "formal ABBA requires continuous-window numerical collection; old/chunked evidence is diagnostic only")
        arm_data[name] = collection
        expected_metric = dict(checked["metrics"])
        require(same(report["numerical_checks"][name], expected_metric), "ABBA numeric summary differs from recomputed protobufs")
    require(report["reference_report_sha256"] == reference["report_sha256"], "ABBA reference mismatch")
    require(len(reference["positions"]) * work["cycles"] + work["warmup"] <= 1000000, "ABBA exceeds request budget")
    _, measured, prepared_sha = benchmark.prepare_requests(protocol, reference, work["warmup"], work["cycles"], work["task_timeout"])
    count = len(measured)
    require(work["smoke"] or count >= 1024, "formal ABBA requires at least 1024 complete-corpus requests")
    require(report["prepared_request_sha256"] == prepared_sha and report["measured_requests"] == count
            and report["semantic_positions"] == len(reference["positions"]), "ABBA prepared request set mismatch")
    source_expected = {name: planner.file_digest(ROOT / "scripts" / name) for name in SOURCE_NAMES
                       if name not in ("quantization_selection_ledger.py", "plan_quantization_search.py")}
    require(same(report["source_scripts"], source_expected)
            and report["protocol_sha256"] == planner.file_digest(ROOT / "crates/kata_worker/proto/worker.proto"),
            "ABBA source/protocol version differs")
    scores, run_ids, previous_end = [], [], None
    for index, name in enumerate(benchmark.ORDER):
        arm_dir = directory / f"{index + 1:02d}-{name}"
        arm = read_json(arm_dir / "report.json")
        require(same(arm, report["runs"][index]), "ABBA top-level/arm report differs")
        prior = arm_data[name]["report"]
        require(arm["status"] == "PASS" and arm["drained"] is True and arm["arm"] == name and arm["order_index"] == index
                and arm["collection_report_sha256"] == arm_data[name]["report_sha256"]
                and arm["concurrency"] == work["concurrency"] and arm["protocol_capacity"] == work["capacity"]
                and arm["warmup_requests"] == work["warmup"] and arm["measured_requests"] == count
                and arm["batch_capacity"] == prior["batch"], "ABBA arm contract mismatch")
        expected_files = benchmark.runtime_files((Path(results["NUMERIC"][baseline_recipe if name == "baseline" else recipe_sha]["collection_directory"]), arm_data[name]), None)[2]
        require(same(arm["files"], [dict(path=str(p), sha256=h) for p, h in expected_files]), "ABBA runtime files differ")
        command = arm["command"]
        require(type(command) is list and len(command) == 15 and type(command[3]) is str
                and re.fullmatch(r"127\.0\.0\.1:[0-9]{1,5}", command[3]) is not None,
                "ABBA isolated command envelope differs")
        require(command == [str(expected_files[0][0]), "nnworker", "--server", command[3], "--worker-id",
                            f"quant-corpus-abba-{index}", "--capacity", str(work["capacity"]), "--once", "--model",
                            str(expected_files[1][0]), "--model-sha256", prior["model_sha256"], "--config", str(expected_files[2][0])],
                "ABBA command does not execute the frozen runtime files")
        benchmark.check_files(expected_files)
        expected_env = {k: v for k, v in benchmark.arm_environment(prior).items() if k.startswith("KATAGO_")}
        require(same(arm["environment"], expected_env), "ABBA declared environment differs")
        benchmark.check_hello(ParseDict(arm["hello"], protocol.pb.WorkerHello()), prior, work["capacity"])
        p = prior["actual_profile"]
        logged = set(benchmark.PROFILE_LINE.findall((arm_dir / "worker.log").read_text(encoding="utf-8", errors="replace")))
        require(logged == {(p["model_sha256"], p["graph_sha256"], p["recipe_sha256"], p["inference_profile_id"])}, "ABBA actual profile log differs")
        before, after = arm["before_heartbeat"], arm["after_heartbeat"]
        delta = {k: after[k] - before[k] for k in after if k != "in_flight"}
        require(same(delta, arm["counter_deltas"]) and before["in_flight"] == after["in_flight"] == 0
                and before["completed_requests"] == before["nn_rows"] == work["warmup"] and before["failed_requests"] == 0
                and 1 <= before["nn_batches"] <= work["warmup"]
                and delta["failed_requests"] == 0 and delta["nn_rows"] == delta["completed_requests"] == count
                and 1 <= delta["nn_batches"] <= count and count / delta["nn_batches"] <= prior["batch"], "ABBA NN counters differ")
        timing = read_json(arm_dir / "timings.json")
        columns = ["round_trip_us", "worker_elapsed_us", "queue_us", "context_us", "evaluator_us", "refill_delay_us"]
        require(timing["columns"] == columns and len(timing["rows"]) == count, "ABBA timing rows incomplete")
        for row in timing["rows"]:
            require(type(row) is list and len(row) == 6 and all(v is None or (type(v) in (int, float) and math.isfinite(v) and v >= 0) for v in row)
                    and row[0] is not None and row[0] > 0, "invalid timing sample")
        measurement = arm["measurement"]
        for column, key in enumerate(columns):
            require(same(measurement["timings"][key], distribution([row[column] for row in timing["rows"]])), "timing summary differs from raw rows")
        elapsed = (measurement["ended_ns"] - measurement["started_ns"]) / 1e9
        require(elapsed > 0 and elapsed == measurement["elapsed_seconds"] and measurement["completed"] == count
                and measurement["rpc_requests_per_second"] == count / elapsed, "throughput/elapsed/count differs")
        require(previous_end is None or measurement["started_ns"] > previous_end, "ABBA replayed/overlapping run time interval")
        previous_end = measurement["ended_ns"]
        scores.append(count / elapsed if work["metric"] == "rpc_throughput" else 1 / measurement["timings"]["round_trip_us"]["p95"])
        # Exclude mutable arm labels/index/command so renaming a previously used
        # sample cannot give it a new identity in another comparison.
        run_ids.append(digest(compact([arm["collection_report_sha256"], arm["files"], arm["environment"],
                                       measurement["started_ns"], measurement["ended_ns"],
                                       digest((arm_dir / "timings.json").read_bytes()), digest((arm_dir / "worker.log").read_bytes())])))
    require(len(set(run_ids)) == 4, "ABBA replayed arm")
    decision = decide([scores[0], scores[3]], scores[1:3], 0.01, 0.05)
    for key in decision:
        reported_key = "meets_pairwise_speed_and_stability_limits" if key == "accepted" else key
        require(same(decision[key], report["pairwise"][reported_key]), "reported pairwise decision differs from raw runs")
    return dict(gate="PASS" if decision["accepted"] else ("UNSTABLE" if not decision["stable"] else "NO_GAIN"),
                pairwise=decision, run_ids=run_ids, baseline_recipe_sha256=baseline_recipe,
                baseline_optimality="NOT_ESTABLISHED", collection_report_sha256=candidate_entry["collection_report_sha256"],
                execution_id=candidate_entry["execution_id"])


def ingest(root, stage, recipe_sha, *, reference=None, candidate=None, performance=None, expect_head=None):
    root = Path(root).resolve()
    with locked(root):
        contract, events = replay(root, expect_head)
        try:
            results, seen = reserve(root, contract, events, stage, recipe_sha)
        except (ValueError, KeyError) as error:
            append_event(root, contract, events, "REQUEST_REJECTED", dict(stage=stage, recipe_sha256=recipe_sha, error=str(error)))
            raise
        payload = dict(recipe_sha256=recipe_sha, gate="INVALID_EVIDENCE")
        submission = dict(reference=str(Path(reference).resolve()) if reference is not None else None,
                          candidate=str(Path(candidate).resolve()) if candidate is not None else None,
                          performance=str(Path(performance).resolve()) if performance is not None else None)
        initial_files = []
        artifacts = []
        protocol = None
        error = None
        try:
            from worker_protocol_tools import Protocol
            protocol = Protocol(ROOT / "crates/kata_worker/proto/worker.proto")
            if stage == "NUMERIC":
                reference, candidate = Path(reference).resolve(), Path(candidate).resolve()
                # Preflight both split headers before reading either set of outputs.
                reference_names = collection_files(reference, contract)
                candidate_names = collection_files(candidate, contract)
                inputs = [("reference", reference, reference_names), ("candidate", candidate, candidate_names)]
                initial_files = [file_record(path / name) for _, path, names in inputs for name in names]
                payload.update(validate_numeric(contract, recipe_sha, reference, candidate, protocol)[0])
                evidence_id = digest(compact([stage, payload["reference_report_sha256"], payload["collection_report_sha256"]]))
            else:
                performance = Path(performance).resolve()
                names = ["report.json"] + [f"{index + 1:02d}-{name}/{filename}" for index, name in enumerate(("baseline", "candidate", "candidate", "baseline"))
                                            for filename in ("report.json", "timings.json", "worker.log")]
                inputs = [("abba", performance, names)]
                initial_files = [file_record(safe_file(performance, name)) for name in names]
                payload.update(validate_performance(contract, recipe_sha, performance, results, protocol))
                evidence_id = digest(compact([stage, payload["run_ids"]]))
                prior_runs = {run for result in results["PERFORMANCE"].values() for run in result.get("run_ids", [])}
                require(not prior_runs.intersection(payload["run_ids"]), "ABBA arm evidence was already consumed")
            require(evidence_id not in seen, "replayed evidence")
            payload["evidence_id"] = evidence_id
            for label, path, names in inputs:
                artifacts.extend(snapshot(root, len(events), label, path, names))
            # Source or config changes during verification invalidate the attempt.
            for source in initial_files + contract["sources"] + payload.get("runtime_sources", []):
                verify_record(source)
        except Exception as caught:
            error = caught
            payload = dict(recipe_sha256=recipe_sha, gate="INVALID_EVIDENCE", error=str(caught),
                           submission=submission, observed_input_hashes=initial_files)
        finally:
            if protocol is not None:
                protocol.close()
            append_event(root, contract, events, stage + "_RESULT", payload, artifacts)
        if error is not None:
            raise ValueError(str(error)) from error
    return inspect(root)


def frozen_stage_directory(root, event, label, names):
    directory = root / "evidence" / f"{event['sequence']:06d}" / label
    expected = {(directory / name).relative_to(root).as_posix() for name in names}
    actual = [item["relative_path"] for item in event["evidence"]
              if item["relative_path"].startswith(directory.relative_to(root).as_posix() + "/")]
    require(len(actual) == len(expected) and set(actual) == expected, "result does not bind every required frozen artifact")
    return directory


def summarize_selection(root, output, expect_head=None):
    """Derive a head-bound diagnostic conclusion; never mutate ledger or budgets."""
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists() and output.parent.is_dir() and not output.is_relative_to(root),
            "summary output must be a NEW directory outside the ledger with an existing parent")
    with locked(root):
        contract, events = replay(root, expect_head)
        used, results, _ = derive_state(contract, events)
        numeric, identities, result_events = {}, {}, {}
        from worker_protocol_tools import Protocol
        protocol = Protocol(ROOT / "crates/kata_worker/proto/worker.proto")
        try:
            for event in events:
                if event["event_type"] not in ("NUMERIC_RESULT", "PERFORMANCE_RESULT"):
                    continue
                payload = event["payload"]
                recipe = payload["recipe_sha256"]
                stage = event["event_type"].removesuffix("_RESULT")
                result_events[stage, recipe] = event
                if stage != "NUMERIC" or payload["gate"] == "INVALID_EVIDENCE":
                    continue
                directories = []
                # Both split headers are checked before loading either output.
                for label in ("reference", "candidate"):
                    directory = root / "evidence" / f"{event['sequence']:06d}" / label
                    directories.append(frozen_stage_directory(root, event, label, collection_files(directory, contract)))
                for record in payload["runtime_sources"]:
                    verify_record(record)
                checked = validate_numeric(contract, recipe, *directories, protocol,
                                           runtime_paths=(Path(payload["reference_directory"]), Path(payload["collection_directory"])))
                require(all(same(value, payload.get(key)) for key, value in checked[0].items()),
                        "registered numeric decision differs from frozen protobuf evidence")
                require(payload["evidence_id"] == digest(compact(["NUMERIC", checked[0]["reference_report_sha256"],
                                                                   checked[0]["collection_report_sha256"]])), "numeric evidence identity differs")
                numeric[recipe] = checked
                identities[recipe] = dict(execution_id=payload["execution_id"], execution=payload["execution"],
                                          runtime_sources=payload["runtime_sources"],
                                          numeric_event_sha256=event["event_sha256"],
                                          frozen_artifacts=event["evidence"])
            for (stage, recipe), event in result_events.items():
                payload = event["payload"]
                if stage != "PERFORMANCE" or payload["gate"] == "INVALID_EVIDENCE":
                    continue
                names = ["report.json"] + [f"{i + 1:02d}-{name}/{file}" for i, name in enumerate(("baseline", "candidate", "candidate", "baseline"))
                                           for file in ("report.json", "timings.json", "worker.log")]
                directory = frozen_stage_directory(root, event, "abba", names)
                checked = validate_performance(contract, recipe, directory, results, protocol, validated_numeric=numeric)
                require(all(same(value, payload.get(key)) for key, value in checked.items()),
                        "registered performance decision differs from frozen raw timings")
                require(payload["evidence_id"] == digest(compact(["PERFORMANCE", checked["run_ids"]])), "performance evidence identity differs")
                identities[recipe].update(performance_event_sha256=event["event_sha256"],
                                           performance_artifacts=event["evidence"])
        finally:
            protocol.close()

        work = contract["workload"]
        remaining = {stage: work["max_" + stage.lower() + "_evaluations"] - len(used[stage]) for stage in used}
        catalog = {entry["recipe_sha256"]: entry for entry in contract["baselines"]["baselines"]}
        baselines, candidates, next_steps = [], [], []
        for entry in contract["plan"]["candidates"]:
            recipe = entry["recipe_sha256"]
            number = results["NUMERIC"].get(recipe, {})
            performance = results["PERFORMANCE"].get(recipe, {})
            row = dict(name=entry["name"], recipe_sha256=recipe, identity=identities.get(recipe),
                       numeric_gate=number.get("gate", "NOT_RECORDED"), performance_gate=performance.get("gate", "NOT_RECORDED"))
            if recipe in catalog:
                qualified = number.get("gate") == "PASS" and number.get("collection_report_sha256") == catalog[recipe]["collection_report_sha256"]
                row.update(status="REGISTERED_BASELINE" if qualified else "BASELINE_EVIDENCE_INCOMPLETE",
                           optimality="NOT_ESTABLISHED", catalog=catalog[recipe])
                baselines.append(row)
                if qualified:
                    continue
                stage = "NUMERIC"
            elif number.get("gate") == "FAIL":
                row["status"] = "REJECTED_NUMERIC_GATE"
                candidates.append(row)
                continue
            elif number.get("gate") == "PASS" and performance.get("gate") in ("PASS", "NO_GAIN", "UNSTABLE"):
                row.update(status={"PASS": "CANDIDATE_REQUIRES_CONFIRMATION", "NO_GAIN": "REJECTED_NO_GAIN",
                                   "UNSTABLE": "REJECTED_UNSTABLE"}[performance["gate"]],
                           pairwise=performance["pairwise"], compared_baseline_recipe_sha256=performance["baseline_recipe_sha256"])
                candidates.append(row)
                continue
            else:
                row["status"] = "EVIDENCE_INCOMPLETE"
                candidates.append(row)
                stage = "PERFORMANCE" if number.get("gate") == "PASS" else "NUMERIC"
            attempted = recipe in used[stage]
            action = "CLOSE_WITHOUT_RETRY" if attempted or remaining[stage] == 0 else "REGISTER_PREDECLARED_EVIDENCE_IF_AVAILABLE"
            row["missing_stage"] = stage
            row["next_action"] = action
            row["reason"] = "ATTEMPT_ALREADY_CONSUMED" if attempted else ("BUDGET_EXHAUSTED" if remaining[stage] == 0 else "NOT_ATTEMPTED")
            next_steps.append(dict(recipe_sha256=recipe, stage=stage, action=action, reason=row["reason"]))

        retained = [row["recipe_sha256"] for row in baselines if row["status"] == "REGISTERED_BASELINE"]
        positive = [row["recipe_sha256"] for row in candidates if row["status"] == "CANDIDATE_REQUIRES_CONFIRMATION"]
        incomplete = not candidates or not baselines or len(retained) != len(baselines) or any(row["status"] == "EVIDENCE_INCOMPLETE" for row in candidates)
        conclusion = "EVIDENCE_INCOMPLETE" if incomplete else ("CANDIDATE_REQUIRES_CONFIRMATION" if positive else "BASELINE_RETAINED")
        qualified_corpus = not work["smoke"] and work["split"] == "selection" and contract["plan"]["corpus_status"] == "READY"
        missing = ["BEST_VALIDATED_FP16_AND_INT8_BASELINE_QUALIFICATION", "INDEPENDENT_ABBA_CONFIRMATION",
                   "FROZEN_INDEPENDENT_HOLDOUT"]
        if not qualified_corpus:
            missing.insert(0, "QUALIFIED_SELECTION_CORPUS")
        if not candidates:
            next_steps.append(dict(action="CLOSE_WITHOUT_COMPARATIVE_CLAIM", reason="NO_REGISTERED_CHALLENGER"))
        if not baselines:
            next_steps.append(dict(action="CLOSE_WITHOUT_COMPARATIVE_CLAIM", reason="NO_REGISTERED_BASELINE"))
        if not incomplete:
            next_steps.append(dict(action="RETAIN_REGISTERED_BASELINE" if not positive else
                                   ("REQUIRE_QUALIFIED_CORPUS_BEFORE_FORMAL_SELECTION" if not qualified_corpus else
                                    "REVIEW_BASELINE_QUALIFICATION_THEN_INDEPENDENT_CONFIRMATION")))
        report = dict(schema=SCHEMA, report_kind="SELECTION_SUMMARY", conclusion=conclusion,
                      scope="QUALIFIED_SELECTION_EVIDENCE_ONLY" if qualified_corpus else "SMOKE_DIAGNOSTIC_ONLY",
                      search_id=contract["search_id"], workload_id=contract["workload_id"], workload=work,
                      ledger_directory=str(root), head_sha256=events[-1]["event_sha256"], event_count=len(events),
                      contract=file_record(root / "contract.json"), source_fingerprints=contract["sources"],
                      baselines=baselines, candidates=candidates, retained_baseline_recipe_sha256=retained,
                      positive_pairwise_candidates_unranked=positive, performance_ranking=None,
                      remaining_attempts=remaining, next_steps=next_steps, missing_qualification=missing,
                      automatic_optimization_complete=False, formal_adoption_allowed=False,
                      baseline_optimality="NOT_ESTABLISHED", original_fp32_gate="NOT_EVALUATED",
                      independent_confirmation="NOT_REGISTERED", holdout_outputs_read=0,
                      deployment_adopted=False, publish_allowed=False, production_certified=False)
        # Detect changes during CPU validation before writing any conclusion.
        replay(root, events[-1]["event_sha256"])
        for data in numeric.values():
            for record in data[0]["runtime_sources"]:
                verify_record(record)
        output.mkdir()
        write_new(output / "selection-summary.json", report)
        return dict(report=report, summary_file=file_record(output / "selection-summary.json"))


EXECUTION_SCHEMA = "rustgo-quantization-execution-ledger-v2"


def execution_sources():
    # Lazy imports in this mode are intentional: catalog imports the v1 helpers.
    return [*source_versions(), file_record(ROOT / "scripts/quantization_execution_catalog.py")]


def execution_safe_file(root, relative):
    path = root / relative
    require(not path.is_symlink() and not any(parent.is_symlink() for parent in path.parents if parent.is_relative_to(root)),
            "execution evidence symlink is forbidden")
    return safe_file(root, relative)


def execution_append(root, contract, events, kind, payload, evidence=None):
    event = dict(schema=EXECUTION_SCHEMA, sequence=len(events),
                 previous_event_sha256=events[-1]["event_sha256"] if events else ZERO,
                 ledger_id=contract["ledger_id"], event_type=kind, payload=payload, evidence=evidence or [])
    event["event_sha256"] = digest(compact(event))
    write_new(root / "events" / f"{len(events):06d}.json", event)
    events.append(event)
    return event


def execution_initialize(output, max_collection_attempts, max_abba_attempts):
    output = Path(output).resolve()
    require(not output.exists() and output.parent.is_dir(), "output must be a NEW directory with an existing parent")
    for name, value in (("max_collection_attempts", max_collection_attempts), ("max_abba_attempts", max_abba_attempts)):
        planner.positive_integer(value, name, 100000)
    identity = dict(schema=EXECUTION_SCHEMA, directory=str(output),
                    budgets=dict(COLLECTION=max_collection_attempts, ABBA=max_abba_attempts), sources=execution_sources())
    contract = dict(**identity, ledger_id=digest(compact(identity)),
                    limitations=["one durable ledger; never create a new ledger to retry consumed experiments",
                                 "reservation precedes evidence admission; old reports do not prove process start order",
                                 "local hash chain is not an external signature; retain the head SHA separately",
                                 "original catalogs, sources and admitted evidence must remain unchanged and available",
                                 "raw ABBA results are recomputed; execution is performed only by an external driver",
                                 "no holdout, adoption, ranking, certification or publication"])
    output.mkdir()
    write_new(output / "contract.json", contract)
    (output / "events").mkdir()
    (output / "evidence").mkdir()
    execution_append(output, contract, [], "INITIALIZED", dict(contract_sha256=planner.file_digest(output / "contract.json")))
    return execution_inspect(output)


def execution_row(catalog, stage, key):
    field, key_field = ("numeric_pairs", "collection_attempt_key") if stage == "COLLECTION" else ("performance_pairs", "pair_key")
    require(stage in ("COLLECTION", "ABBA"), "unsupported execution stage")
    rows = [row for row in catalog[field] if row[key_field] == key]
    require(len(rows) == 1, "attempt key is not registered in this catalog")
    return rows[0]


def execution_reservation(contract, catalog_sha, catalog, stage, key):
    row = execution_row(catalog, stage, key)
    identity = dict(ledger_id=contract["ledger_id"], stage=stage, attempt_key=key)
    return dict(**identity, reservation_id=digest(compact(identity)), catalog_sha256=catalog_sha,
                workload_id=row["workload_id"], registration=row)


def execution_collection_preflight(directory, catalog, spec):
    """Reject holdout and unregistered execution before opening any output."""
    metadata = read_json(execution_safe_file(directory, "report.json"))
    require(metadata.get("split") in ("calibration", "selection"), "holdout/unknown outputs are forbidden before reading any output")
    work, execution = catalog["numeric_workload"], spec["semantic"]["execution"]
    require(metadata.get("split") == catalog["workload"]["split"], "collection split differs from catalog")
    require(metadata.get("window_mode") == "continuous", "catalog-v2 requires continuous numerical collections")
    require(metadata.get("model_sha256") == spec["semantic"]["model_sha256"]
            and metadata.get("binary_sha256") == execution["binary_sha256"]
            and metadata.get("backend") == execution["backend"] and metadata.get("batch") == execution["batch"],
            "collection model/binary/backend/batch differs from planned execution")
    require(metadata.get("request_window") == work["concurrency"] and metadata.get("capacity") == work["capacity"]
            and metadata.get("expected_requests") == work["ordered_request_semantics"]["count"],
            "collection request count/window/capacity differs from planned workload")
    require(type(metadata.get("expected_requests")) is int and 0 < metadata["expected_requests"] <= 100000,
            "invalid collection count")
    allowed_source = {"manifest.json", "games.jsonl", metadata["split"] + ".requests.jsonl"}
    require({item["file"] for item in metadata["source_artifacts"]} == allowed_source,
            "unexpected collection source artifact; holdout is forbidden")
    # All paths are screened before validators with direct Path.read_bytes calls.
    names = {"report.json", "worker.log", "worker.cfg"}
    ordinary = {"outputs.jsonl", "requests-index.jsonl", "worker.cfg", "recipe.json", "worker.log", "continuous-buffer.json"}
    if spec["kind"] == "legacy_explicit_spec":
        import legacy_quantization_execution as legacy
        ordinary |= {legacy.SPEC_FILE, "source-worker.cfg", "runtime-cudaTacticPlan.json"}
        # Probe files have a fixed layout defined by the existing legacy validator.
        observation = legacy.validate_collection(directory, metadata)
        names.update(observation["artifact_paths"])
    for item in metadata["artifacts"]:
        require(item["file"] in ordinary, "unknown catalog collection artifact")
        names.add(item["file"])
    names.update("source/" + name for name in allowed_source)
    names.update(f"{kind}/{i:06d}.pb" for kind in ("requests", "results") for i in range(1, metadata["expected_requests"] + 1))
    for name in names:
        execution_safe_file(directory, name)
    return metadata, sorted(names)


def execution_load_collection(directory, catalog, spec_id, protocol):
    """Map verified raw runtime evidence to one planned spec; never trust PASS."""
    import quantization_execution_catalog as catalogs
    import compare_quantization_corpus as comparison
    import benchmark_quantization_corpus as benchmark
    spec = next(row for row in catalog["execution_specs"] if row["execution_spec_id"] == spec_id)
    metadata, names = execution_collection_preflight(directory, catalog, spec)
    reference_id = catalog["identity"]["canonical_reference_execution_spec_id"]
    reference = next(row for row in catalog["execution_specs"] if row["execution_spec_id"] == reference_id)
    full_reference = reference["semantic"]["execution"]["precision_recipe"]
    data = comparison.load_collection(directory, protocol, full_reference=None if spec_id == reference_id else full_reference)
    ordered = sorted(data["positions"].values(), key=lambda row: row["request"].task_id)
    require(same(catalogs.ordered_request_semantics([row["record"] for row in ordered]),
                 catalog["numeric_workload"]["ordered_request_semantics"]), "actual ordered request semantics differ from catalog")
    require(all(row["request"].lease_ms == catalog["numeric_workload"]["lease_ms"] for row in ordered),
            "actual request lease differs from catalog timeout")
    execution = spec["semantic"]["execution"]
    environment = {key: value for key, value in metadata.get("controlled_environment", {}).items() if key != "KATAGO_CUDA_BATCH_TRACE"}
    require(same(environment, execution["environment"]), "actual controlled environment differs from planned execution")
    benchmark.arm_environment(metadata)
    command = metadata.get("command", [])
    require(type(command) is list and len(command) == 15 and re.fullmatch(r"127\.0\.0\.1:[0-9]+", command[3] or "") is not None,
            "collection must bind its isolated actual Worker command")
    expected_command = [metadata["binary"], "nnworker", "--server", command[3], "--worker-id", "quantization-corpus-local",
                        "--capacity", str(metadata["capacity"]), "--once", "--model", metadata["model"],
                        "--model-sha256", metadata["model_sha256"], "--config", str(directory / "worker.cfg")]
    require(command == expected_command, "collection actual Worker command differs from planned runtime files")
    _, _, runtime = benchmark.runtime_files((directory, data), None)
    benchmark.check_files(runtime)
    if spec["kind"] == "legacy_explicit_spec":
        import legacy_quantization_execution as legacy
        loaded, _, _ = legacy.collection_sources(directory, metadata)
        require(same(loaded["semantic"], execution), "observed legacy spec differs from planned execution")
        observed = data["legacy_observation"]
    else:
        require("legacy_execution" not in metadata and metadata.get("actual_profile"), "unified actual profile is missing")
        require(same(data["recipe"], execution["precision_recipe"])
                and metadata["actual_profile"]["recipe_sha256"] == execution["recipe_sha256"]
                and metadata["actual_profile"]["graph_sha256"] == spec["semantic"]["graph_sha256"],
                "actual unified recipe/graph differs from planned execution")
        # The collector owns this exact config format. Runtime path existence and
        # recipe bytes were checked above; only that path/profile is normalized.
        values = {}
        for line in metadata["config_text"].splitlines():
            body = line.partition("#")[0].strip()
            if not body:
                continue
            key, sep, value = body.partition("=")
            key, value = key.strip(), value.strip()
            require(sep and key not in values, "ambiguous unified execution config")
            values[key] = value
        require(values.pop("cudaQuantExpectedProfile", None) == metadata["actual_profile"]["inference_profile_id"],
                "unified config has no matching observed profile")
        require("cudaQuantPlan" in values, "unified config has no recipe")
        values["cudaQuantPlan"] = dict(recipe_sha256=execution["recipe_sha256"])
        expected = {key: str(value) if not isinstance(value, dict) else value for key, value in execution["configuration"].items()}
        require(same(values, expected), "actual unified configuration differs from planned execution")
        identity = dict(execution_spec_id=spec_id, actual_profile=metadata["actual_profile"], environment=environment)
        observed = dict(observed_execution_id="rustgo-unified-evidence-v2:" + digest(compact(identity)), **identity)
    original_files = [file_record(execution_safe_file(directory, name)) for name in names]
    runtime_sources = [file_record(path) for path, _ in runtime]
    result = dict(execution_spec_id=spec_id, numeric_workload_id=catalog["numeric_workload_id"],
                  report_sha256=data["report_sha256"], observed_execution=observed,
                  original_directory=str(directory), original_files=original_files, runtime_sources=runtime_sources,
                  validation="RAW_PROTOBUF_AND_EXECUTION_IDENTITY_VALIDATED", numeric_gate="NOT_COMPARED",
                  process_start_after_reservation="NOT_PROVEN_BY_COLLECTION_REPORT")
    return result, data, names


def execution_compare_data(reference, candidate):
    """Compare exact validated outputs across equivalent corpus serialization.

    Catalog keys describe ordered Position/EvalParameters, not names, JSON
    whitespace or session envelopes. Only after bytewise message equivalence
    may the existing numeric comparator receive a common metadata view.
    Original protobufs and their envelopes remain frozen and independently checked.
    """
    import compare_quantization_corpus as comparison
    left = sorted(reference["positions"].values(), key=lambda row: row["request"].task_id)
    right = sorted(candidate["positions"].values(), key=lambda row: row["request"].task_id)
    require(len(left) == len(right), "numeric comparison request count differs")
    for a, b in zip(left, right, strict=True):
        require(all(getattr(a["request"], field).SerializeToString(deterministic=True)
                    == getattr(b["request"], field).SerializeToString(deterministic=True) for field in ("position", "parameters")),
                "numeric comparison actual ordered semantics differ")
    for key in ("model_sha256", "schema_sha256", "expected_requests"):
        require(reference["report"][key] == candidate["report"][key], "numeric comparison model/schema/count differs")
    adapted = dict(candidate, report=dict(candidate["report"]), positions={})
    for key in ("corpus_manifest_sha256", "corpus_request_set_sha256", "source_request_file_sha256", "requests_sha256", "split"):
        adapted["report"][key] = reference["report"][key]
    for a, b in zip(left, right, strict=True):
        adapted["positions"][a["record"]["semantic_position_sha256"]] = dict(b, record=a["record"], request=a["request"])
    result = comparison.compare(reference, adapted)
    result.pop("cases")
    result["comparison_binding"] = "individually validated raw collections; ordered Position/EvalParameters equality"
    return result


def execution_qualification(catalog, pair, state):
    required = []
    for spec_id in pair["requires_numeric_pass_for"]:
        row = next(row for row in catalog["numeric_pairs"] if row["candidate_execution_spec_id"] == spec_id)
        comparison = state["comparisons"].get(row["comparison_key"])
        require(comparison is not None and comparison["metrics"]["experimental_gate"]["result"] == "PASS",
                "ABBA requires recomputed numeric PASS for canonical FP16 reference and both actual executions")
        observed = state["collections"][row["collection_attempt_key"]]
        required.append(dict(execution_spec_id=spec_id, comparison_key=row["comparison_key"],
                             collection_attempt_key=row["collection_attempt_key"], report_sha256=observed["report_sha256"],
                             observed_execution_id=observed["observed_execution"]["observed_execution_id"]))
    return required


def execution_abba_decision(scores, metric):
    from tune_runtime import decide
    decision = decide([scores[0], scores[3]], scores[1:3], 0.01, 0.05)
    reported = dict(decision)
    reported["score_units"] = "requests/second" if metric == "rpc_throughput" else "inverse_microseconds"
    if metric == "p95_latency":
        reported.update(baseline_p95_geomean_us=1 / decision["baseline_geomean"],
                        candidate_p95_geomean_us=1 / decision["candidate_geomean"],
                        latency_reduction=1 - decision["baseline_geomean"] / decision["candidate_geomean"],
                        threshold_definition="baseline P95 / candidate P95 >=1.01 (speed ratio), not 1% latency reduction")
    reported["meets_pairwise_speed_and_stability_limits"] = reported.pop("accepted")
    return decision, reported


def execution_abba_timing(arm, timing, count, batch, warmup):
    """Use the benchmark's distributions and decision inputs, with raw counters."""
    from benchmark_workers import distribution
    before, after = arm["before_heartbeat"], arm["after_heartbeat"]
    fields = {"in_flight", "completed_requests", "failed_requests", "nn_rows", "nn_batches"}
    require(set(before) == set(after) == fields and all(type(value) is int and value >= 0 for row in (before, after) for value in row.values()),
            "invalid ABBA heartbeat counters")
    delta = {key: after[key] - before[key] for key in fields if key != "in_flight"}
    require(same(delta, arm["counter_deltas"]) and before["in_flight"] == after["in_flight"] == 0
            and before["completed_requests"] == before["nn_rows"] == warmup and before["failed_requests"] == 0
            and 1 <= before["nn_batches"] <= warmup and warmup / before["nn_batches"] <= batch
            and delta["failed_requests"] == 0 and delta["completed_requests"] == delta["nn_rows"] == count
            and 1 <= delta["nn_batches"] <= count and count / delta["nn_batches"] <= batch,
            "ABBA NN counters differ from complete warmup/measured work")
    columns = ["round_trip_us", "worker_elapsed_us", "queue_us", "context_us", "evaluator_us", "refill_delay_us"]
    require(timing["columns"] == columns and type(timing["rows"]) is list and len(timing["rows"]) == count,
            "ABBA timing rows incomplete")
    for row in timing["rows"]:
        require(type(row) is list and len(row) == 6 and all(value is None or (type(value) in (int, float)
                and math.isfinite(value) and value >= 0) for value in row) and row[0] is not None and row[0] > 0,
                "invalid ABBA raw timing sample")
    measured = arm["measurement"]
    for index, name in enumerate(columns):
        require(same(measured["timings"][name], distribution([row[index] for row in timing["rows"]])),
                "ABBA timing distribution differs from raw rows")
    require(type(measured["started_ns"]) is int and type(measured["ended_ns"]) is int
            and 0 < measured["started_ns"] < measured["ended_ns"], "invalid ABBA measured interval")
    elapsed = (measured["ended_ns"] - measured["started_ns"]) / 1e9
    require(elapsed == measured["elapsed_seconds"] and measured["completed"] == count
            and measured["rpc_requests_per_second"] == count / elapsed, "ABBA elapsed/throughput/count differs")
    for key, expected in (("nn_rows_per_second", count / elapsed), ("nn_batches_per_second", delta["nn_batches"] / elapsed),
                          ("mean_logical_rows_per_batch", count / delta["nn_batches"])):
        require(measured[key] == expected, "ABBA derived NN throughput/batch differs")
    require(measured["physical_batch_distribution"] == "NOT_OBSERVED_IN_UNTRACED_TIMING", "ABBA cannot claim unobserved physical batches")
    return measured


def execution_load_abba(directory, catalog, reservation, state, protocol):
    """Reconstruct one reserved comparison from all four original arm receipts."""
    import quantization_execution_catalog as catalogs
    import benchmark_quantization_corpus as benchmark
    import legacy_quantization_execution as legacy
    from google.protobuf.json_format import ParseDict
    pair, work = reservation["registration"], catalog["workload"]
    require(same(reservation["numeric_qualification"], execution_qualification(catalog, pair, state)),
            "ABBA numerical prerequisites changed")
    report = read_json(execution_safe_file(directory, "report.json"))
    require(same(report.get("timing_clock"), legacy.performance_clock()), "ABBA must bind the explicit performance clock")
    require(report.get("schema") == "rustgo-quantization-corpus-abba-v1"
            and report.get("status") == ("SMOKE_ONLY" if work["smoke"] else "PAIRWISE_MEASURED_REQUIRES_SELECTION_REVIEW"),
            "incomplete ABBA or status/smoke mismatch")
    require(report["holdout_outputs_read"] == 0 and report["order"] == list(benchmark.ORDER)
            and report["publish_allowed"] is False and report["production_certified"] is False
            and report["deployment_adopted"] is False, "ABBA scope/order differs")
    require(report["primary_metric"] == work["metric"] and report["cycles"] == work["cycles"]
            and report["warmup_requests"] == work["warmup"] and report["smoke"] is work["smoke"]
            and report["minimum_improvement"] == 0.01 and report["maximum_relative_spread"] == 0.05,
            "ABBA frozen workload/threshold differs")
    require(type(report["runs"]) is list and len(report["runs"]) == 4, "ABBA requires four raw runs")
    reference_key = catalogs.collection_attempt_key(catalog["numeric_workload_id"], pair["reference_execution_spec_id"])
    reference = state["loaded"][reference_key]
    require(report["reference_report_sha256"] == reference["report_sha256"], "ABBA canonical FP16 reference differs")
    arm_data, observations = {}, {}
    for name in ("baseline", "candidate"):
        key = catalogs.collection_attempt_key(catalog["numeric_workload_id"], pair[name + "_execution_spec_id"])
        arm_data[name], observations[name] = state["loaded"][key], state["collections"][key]
        metrics = execution_compare_data(reference, arm_data[name])
        metrics.pop("comparison_binding")
        require(metrics["experimental_gate"]["result"] == "PASS" and same(metrics, report["numerical_checks"][name]),
                "ABBA numerical prerequisite differs from raw protobufs")
    require(report["numeric_collection_window_modes"] == dict(reference="continuous", baseline="continuous", candidate="continuous"),
            "ABBA numerical collection window modes differ")
    require(work["smoke"] or all(data["report"]["split"] == "selection" and data["report"]["corpus_status"] == "READY"
                                for data in [reference, *arm_data.values()]), "formal ABBA requires qualified selection collections")
    warmup, measured, prepared_sha = benchmark.prepare_requests(protocol, reference, work["warmup"], work["cycles"], work["task_timeout"])
    count = len(measured)
    require(len(warmup) + count <= 1000000 and count >= (1 if work["smoke"] else 1024), "ABBA complete-corpus request budget differs")
    require(report["prepared_request_sha256"] == prepared_sha and report["measured_requests"] == count
            and report["semantic_positions"] == len(reference["positions"]), "ABBA ordered complete input/warmup/cycles differs")
    source_expected = {name: planner.file_digest(ROOT / "scripts" / name) for name in SOURCE_NAMES
                       if name not in ("quantization_selection_ledger.py", "plan_quantization_search.py")}
    require(same(report["source_scripts"], source_expected)
            and report["protocol_sha256"] == planner.file_digest(ROOT / "crates/kata_worker/proto/worker.proto"),
            "ABBA source/protocol version differs")
    names, runtime_sources, scores, run_ids, intervals, actual = {"report.json"}, {}, [], [], [], []
    previous_end = None
    for index, name in enumerate(benchmark.ORDER):
        prefix = f"{index + 1:02d}-{name}"
        arm_dir = directory / prefix
        for filename in ("report.json", "timings.json", "worker.log"):
            names.add(prefix + "/" + filename)
            execution_safe_file(directory, prefix + "/" + filename)
        arm = read_json(arm_dir / "report.json")
        require(same(arm, report["runs"][index]), "ABBA top-level/arm report differs")
        require(same(arm.get("timing_clock"), report["timing_clock"]), "ABBA arm clock differs from top-level timing clock")
        data, observation = arm_data[name], observations[name]
        prior, source = data["report"], Path(observation["original_directory"])
        require(arm["status"] == "PASS" and arm["drained"] is True and arm["arm"] == name and arm["order_index"] == index
                and arm["collection_report_sha256"] == data["report_sha256"] and arm["concurrency"] == work["concurrency"]
                and arm["protocol_capacity"] == work["capacity"] and arm["warmup_requests"] == work["warmup"]
                and arm["measured_requests"] == count and arm["batch_capacity"] == prior["batch"], "ABBA arm workload/execution differs")
        files = benchmark.runtime_files((source, data), None)[2]
        require(same(arm["files"], [dict(path=str(path), sha256=sha) for path, sha in files]), "ABBA runtime files differ")
        benchmark.check_files(files)
        for path, _ in files:
            item = file_record(path)
            runtime_sources[item["path"]] = item
        command = arm["command"]
        require(type(command) is list and len(command) == 15 and type(command[3]) is str
                and re.fullmatch(r"127\.0\.0\.1:[0-9]{1,5}", command[3]) is not None,
                "ABBA isolated command envelope differs")
        require(command == [str(files[0][0]), "nnworker", "--server", command[3], "--worker-id", f"quant-corpus-abba-{index}",
                            "--capacity", str(work["capacity"]), "--once", "--model", str(files[1][0]), "--model-sha256",
                            prior["model_sha256"], "--config", str(files[2][0])], "ABBA command differs from frozen runtime files")
        require(same(arm["environment"], {key: value for key, value in benchmark.arm_environment(prior).items() if key.startswith("KATAGO_")}),
                "ABBA environment differs from numerical execution")
        benchmark.check_hello(ParseDict(arm["hello"], protocol.pb.WorkerHello(), ignore_unknown_fields=False), prior, work["capacity"])
        timing = read_json(arm_dir / "timings.json")
        measurement = execution_abba_timing(arm, timing, count, prior["batch"], work["warmup"])
        start, end = measurement["started_ns"], measurement["ended_ns"]
        if prior.get("actual_profile"):
            require("legacy_execution" not in arm, "unified ABBA cannot borrow legacy evidence")
            profile = prior["actual_profile"]
            logged = set(benchmark.PROFILE_LINE.findall((arm_dir / "worker.log").read_text(encoding="utf-8", errors="strict")))
            require(logged == {(profile["model_sha256"], profile["graph_sha256"], profile["recipe_sha256"], profile["inference_profile_id"])},
                    "ABBA actual unified profile log differs")
            observed = observation["observed_execution"]
        else:
            observed = legacy.validate_arm(arm_dir, arm, source, prior)
            require(observed["observed_execution_id"] == observation["observed_execution"]["observed_execution_id"],
                    "ABBA actual legacy execution differs from qualified observation")
            for filename in observed["artifact_paths"]:
                execution_safe_file(arm_dir, filename)
                names.add(prefix + "/" + filename)
            bounds = {}
            for side in ("before", "after"):
                descriptor = arm["legacy_execution"][side]
                interval = legacy.probe_interval(arm_dir, descriptor, require_performance_clock=True)
                require(same(interval["timing_clock"], report["timing_clock"]), "ABBA probe clock differs from measured clock")
                bounds[side] = (interval["started_ns"], interval["ended_ns"])
            require(bounds["before"][1] < start and end < bounds["after"][0], "legacy fingerprint probe overlaps the timed region")
            start, end = bounds["before"][0], bounds["after"][1]
        require(previous_end is None or start > previous_end, "ABBA replayed/overlapping arm intervals")
        previous_end = end
        intervals.append(dict(started_ns=start, ended_ns=end, measured_started_ns=measurement["started_ns"], measured_ended_ns=measurement["ended_ns"]))
        actual.append(dict(execution_spec_id=pair[name + "_execution_spec_id"], observed_execution_id=observed["observed_execution_id"],
                           collection_report_sha256=data["report_sha256"]))
        scores.append(count / measurement["elapsed_seconds"] if work["metric"] == "rpc_throughput" else 1 / measurement["timings"]["round_trip_us"]["p95"])
        # Exclude mutable labels, paths and JSON formatting from a timing identity.
        run_ids.append(digest(compact([observed["observed_execution_id"], measurement["started_ns"], measurement["ended_ns"], timing])))
    require(len(set(run_ids)) == 4, "ABBA arm evidence replayed")
    for prior in state["performance"].values():
        require(not set(run_ids).intersection(prior["run_ids"]), "ABBA arm evidence was already consumed")
        require(all(left["ended_ns"] < right["started_ns"] or right["ended_ns"] < left["started_ns"]
                    for left in intervals for right in prior["run_intervals"]), "ABBA intervals overlap previously consumed evidence")
    decision, expected = execution_abba_decision(scores, work["metric"])
    require(same(expected, report["pairwise"]), "ABBA pairwise geometric mean/spread/threshold differs from raw runs")
    observation = dict(gate="PASS" if decision["accepted"] else ("UNSTABLE" if not decision["stable"] else "NO_GAIN"),
                pairwise=decision, metric=work["metric"], baseline_execution_spec_id=pair["baseline_execution_spec_id"],
                candidate_execution_spec_id=pair["candidate_execution_spec_id"], reference_execution_spec_id=pair["reference_execution_spec_id"],
                run_ids=run_ids, run_intervals=intervals, actual_executions=actual, prepared_request_sha256=prepared_sha,
                report_sha256=planner.file_digest(directory / "report.json"), original_directory=str(directory),
                original_files=[file_record(execution_safe_file(directory, name)) for name in sorted(names)],
                runtime_sources=[runtime_sources[path] for path in sorted(runtime_sources)],
                process_start_after_reservation="NOT_PROVEN_BY_ABBA_REPORT", production_certified=False)
    return observation, sorted(names)


def execution_replay(root, expect_head=None):
    import quantization_execution_catalog as catalogs
    from worker_protocol_tools import Protocol
    root = Path(root).resolve()
    require(not (root / "contract.json").is_symlink()
            and all((root / name).is_dir() and not (root / name).is_symlink() for name in ("events", "evidence")),
            "ledger control paths must be local directories/files")
    contract = read_json(root / "contract.json")
    require(contract.get("schema") == EXECUTION_SCHEMA, "catalog-v2 requires a separate execution ledger; v1 cannot be upgraded")
    for record in contract["sources"]:
        verify_record(record)
    identity = {key: contract[key] for key in ("schema", "directory", "budgets", "sources")}
    require(contract["ledger_id"] == digest(compact(identity)), "execution ledger identity mismatch")
    for stage in ("COLLECTION", "ABBA"):
        planner.positive_integer(contract["budgets"][stage], "budget", 100000)
    state = dict(catalogs={}, reservations={}, collections={}, comparisons={}, performance={}, failures={}, loaded={})
    events, protocol = [], None
    try:
        for sequence, path in enumerate(sorted((root / "events").iterdir())):
            require(path.name == f"{sequence:06d}.json" and not path.is_symlink(), "missing/noncanonical execution ledger event")
            event = read_json(path)
            planner.object_fields(event, {"schema", "sequence", "previous_event_sha256", "ledger_id", "event_type", "payload", "evidence", "event_sha256"}, "execution event")
            claimed = event.pop("event_sha256")
            require(digest(compact(event)) == claimed, "execution event hash mismatch")
            require(event["schema"] == EXECUTION_SCHEMA and event["sequence"] == sequence
                    and event["ledger_id"] == contract["ledger_id"]
                    and event["previous_event_sha256"] == (events[-1]["event_sha256"] if events else ZERO), "execution event chain mismatch")
            for artifact in event["evidence"]:
                raw = execution_safe_file(root, artifact["relative_path"]).read_bytes()
                require(len(raw) == artifact["bytes"] and digest(raw) == artifact["sha256"], "frozen execution evidence changed")
            payload, kind = event["payload"], event["event_type"]
            if sequence == 0:
                require(kind == "INITIALIZED" and payload == dict(contract_sha256=planner.file_digest(root / "contract.json")),
                        "execution contract is not bound to initialization")
            elif kind == "CATALOG_REGISTERED":
                planner.object_fields(payload, {"catalog_sha256", "source", "snapshot"}, "registered catalog")
                verify_record(payload["source"])
                require(payload["catalog_sha256"] == payload["source"]["sha256"]
                        and payload["catalog_sha256"] not in state["catalogs"], "duplicate/mismatched catalog registration")
                catalog_path = execution_safe_file(root, payload["snapshot"])
                catalog = catalogs.validate_catalog(catalog_path, payload["catalog_sha256"])
                state["catalogs"][payload["catalog_sha256"]] = catalog
            elif kind == "ATTEMPT_RESERVED":
                stage, key = payload["stage"], payload["attempt_key"]
                catalog = state["catalogs"][payload["catalog_sha256"]]
                expected = execution_reservation(contract, payload["catalog_sha256"], catalog, stage, key)
                if stage == "ABBA":
                    expected["numeric_qualification"] = execution_qualification(catalog, expected["registration"], state)
                require(same(payload, expected), "reservation differs from registered execution")
                require(key not in state["reservations"], "semantic attempt key was already consumed across catalogs")
                require(sum(row["stage"] == stage for row in state["reservations"].values()) < contract["budgets"][stage], "durable execution budget exhausted")
                state["reservations"][key] = payload
            elif kind in ("ATTEMPT_FAILED", "COLLECTION_OBSERVED", "ABBA_OBSERVED"):
                key = payload["attempt_key"]
                reservation = state["reservations"][key]
                require(key not in state["failures"] and key not in state["collections"] and key not in state["performance"], "attempt is already terminal")
                require(payload["reservation_id"] == reservation["reservation_id"], "result reservation identity mismatch")
                if kind == "ATTEMPT_FAILED":
                    planner.object_fields(payload, {"attempt_key", "reservation_id", "reason"}, "failed attempt")
                    require(type(payload["reason"]) is str and 0 < len(payload["reason"]) <= 4096, "failure reason is required")
                    state["failures"][key] = payload
                elif kind == "COLLECTION_OBSERVED":
                    require(reservation["stage"] == "COLLECTION", "collection evidence cannot finish ABBA")
                    for item in [*payload["observation"]["original_files"], *payload["observation"]["runtime_sources"]]:
                        verify_record(item)
                    if protocol is None:
                        protocol = Protocol(ROOT / "crates/kata_worker/proto/worker.proto")
                    catalog = state["catalogs"][reservation["catalog_sha256"]]
                    observation, data, names = execution_load_collection(Path(payload["observation"]["original_directory"]), catalog,
                                                    reservation["registration"]["candidate_execution_spec_id"], protocol)
                    frozen_stage_directory(root, event, "collection", names)
                    require(same(payload, dict(attempt_key=key, reservation_id=reservation["reservation_id"], observation=observation)),
                            "collection observation differs from raw evidence")
                    state["collections"][key], state["loaded"][key] = observation, data
                else:
                    require(reservation["stage"] == "ABBA", "ABBA evidence cannot finish a collection")
                    for item in [*payload["observation"]["original_files"], *payload["observation"]["runtime_sources"]]:
                        verify_record(item)
                    if protocol is None:
                        protocol = Protocol(ROOT / "crates/kata_worker/proto/worker.proto")
                    catalog = state["catalogs"][reservation["catalog_sha256"]]
                    observation, names = execution_load_abba(Path(payload["observation"]["original_directory"]), catalog, reservation, state, protocol)
                    frozen_stage_directory(root, event, "abba", names)
                    require(same(payload, dict(attempt_key=key, reservation_id=reservation["reservation_id"], observation=observation)),
                            "ABBA observation differs from recomputed raw evidence")
                    state["performance"][key] = observation
            elif kind == "NUMERIC_COMPARED":
                catalog = state["catalogs"][payload["catalog_sha256"]]
                row = next(row for row in catalog["numeric_pairs"] if row["comparison_key"] == payload["comparison_key"])
                require(row["comparison_key"] not in state["comparisons"], "CPU comparison already registered")
                ref = catalogs.collection_attempt_key(catalog["numeric_workload_id"], row["reference_execution_spec_id"])
                candidate = row["collection_attempt_key"]
                metrics = execution_compare_data(state["loaded"][ref], state["loaded"][candidate])
                expected = dict(catalog_sha256=payload["catalog_sha256"], comparison_key=row["comparison_key"],
                                reference_attempt_key=ref, candidate_attempt_key=candidate, metrics=metrics)
                require(same(payload, expected), "CPU comparison differs from recomputed raw protobufs")
                state["comparisons"][row["comparison_key"]] = payload
            else:
                raise ValueError("unsupported execution ledger event")
            event["event_sha256"] = claimed
            events.append(event)
        require(events, "execution ledger has no initialization event")
        require(expect_head is None or events[-1]["event_sha256"] == expect_head, "execution ledger head differs from expected anchor")
        return contract, events, state
    finally:
        if protocol is not None:
            protocol.close()


def execution_inspect(root, expect_head=None):
    contract, events, state = execution_replay(root, expect_head)
    consumed = {stage: sum(row["stage"] == stage for row in state["reservations"].values()) for stage in ("COLLECTION", "ABBA")}
    pending = [row for key, row in state["reservations"].items() if key not in state["failures"] and key not in state["collections"] and key not in state["performance"]]
    return dict(schema=EXECUTION_SCHEMA, status="CPU_EXECUTION_REGISTRY_ONLY", ledger_id=contract["ledger_id"],
                head_sha256=events[-1]["event_sha256"], event_count=len(events),
                registered_catalog_sha256=list(state["catalogs"]), consumed_attempts=consumed,
                remaining_attempts={stage: contract["budgets"][stage] - consumed[stage] for stage in consumed},
                pending_reservations=pending, failed_attempts=state["failures"], observed_collections=state["collections"],
                numerical_comparisons=state["comparisons"], performance=state["performance"], cpu_comparisons_consume_attempts=False,
                abba_result_ingestion="RAW_FOUR_ARM_EVIDENCE_REQUIRED", execution_driver="EXTERNAL_PREREGISTERED_DRIVER_REQUIRED",
                process_start_order_proven=False, holdout_outputs_read=0, deployment_adopted=False,
                publish_allowed=False, production_certified=False, performance_ranking=None,
                limitations=contract["limitations"])


def execution_register_catalog(root, catalog_path, expect_catalog_sha, expect_head=None):
    import quantization_execution_catalog as catalogs
    root, catalog_path = Path(root).resolve(), Path(catalog_path).resolve()
    planner.corpus_io.check_sha(expect_catalog_sha, "catalog SHA anchor")
    with locked(root):
        contract, events, state = execution_replay(root, expect_head)
        catalogs.validate_catalog(catalog_path, expect_catalog_sha)
        source = file_record(catalog_path)
        if source["sha256"] in state["catalogs"]:
            return execution_inspect(root, events[-1]["event_sha256"])
        evidence = snapshot(root, len(events), "catalog", catalog_path.parent, ["catalog.json"])
        verify_record(source)
        execution_append(root, contract, events, "CATALOG_REGISTERED",
                         dict(catalog_sha256=source["sha256"], source=source, snapshot=evidence[0]["relative_path"]), evidence)
        return execution_inspect(root, events[-1]["event_sha256"])


def execution_reserve(root, catalog_sha, stage, key, expect_head=None):
    root = Path(root).resolve()
    with locked(root):
        contract, events, state = execution_replay(root, expect_head)
        require(catalog_sha in state["catalogs"], "catalog is not registered")
        require(key not in state["reservations"], "semantic attempt key was already consumed across catalogs; no retry")
        require(sum(row["stage"] == stage for row in state["reservations"].values()) < contract["budgets"].get(stage, 0), "durable execution budget exhausted")
        catalog = state["catalogs"][catalog_sha]
        payload = execution_reservation(contract, catalog_sha, catalog, stage, key)
        if stage == "ABBA":
            payload["numeric_qualification"] = execution_qualification(catalog, payload["registration"], state)
        event = execution_append(root, contract, events, "ATTEMPT_RESERVED", payload)
        # The caller may start work only after receiving this fsynced receipt.
        return dict(schema=EXECUTION_SCHEMA, status="RESERVED_BEFORE_EXTERNAL_LAUNCH", reservation=payload,
                    head_sha256=event["event_sha256"], consumed=True, retry_allowed=False,
                    execution_started=False, publish_allowed=False)


def execution_fail(root, reservation_id, reason, expect_head=None):
    root = Path(root).resolve()
    require(type(reason) is str and 0 < len(reason) <= 4096, "failure reason must contain 1..4096 characters")
    with locked(root):
        contract, events, state = execution_replay(root, expect_head)
        matches = [row for row in state["reservations"].values() if row["reservation_id"] == reservation_id]
        require(len(matches) == 1, "unknown reservation")
        key = matches[0]["attempt_key"]
        require(key not in state["collections"] and key not in state["failures"] and key not in state["performance"], "attempt is already terminal")
        execution_append(root, contract, events, "ATTEMPT_FAILED", dict(attempt_key=key, reservation_id=reservation_id, reason=reason))
        return execution_inspect(root, events[-1]["event_sha256"])


def execution_ingest_collection(root, reservation_id, directory, expect_head=None):
    from worker_protocol_tools import Protocol
    root, directory = Path(root).resolve(), Path(directory).resolve()
    with locked(root):
        contract, events, state = execution_replay(root, expect_head)
        matches = [row for row in state["reservations"].values() if row["reservation_id"] == reservation_id and row["stage"] == "COLLECTION"]
        require(len(matches) == 1, "collection requires its own existing reservation before ingestion")
        reservation = matches[0]
        key = reservation["attempt_key"]
        require(key not in state["collections"] and key not in state["failures"], "attempt is already terminal")
        protocol = None
        try:
            protocol = Protocol(ROOT / "crates/kata_worker/proto/worker.proto")
            catalog = state["catalogs"][reservation["catalog_sha256"]]
            observation, _, names = execution_load_collection(directory, catalog, reservation["registration"]["candidate_execution_spec_id"], protocol)
            evidence = snapshot(root, len(events), "collection", directory, names)
            for record in [*observation["original_files"], *observation["runtime_sources"]]:
                verify_record(record)
        except Exception as error:
            execution_append(root, contract, events, "ATTEMPT_FAILED", dict(attempt_key=key, reservation_id=reservation_id,
                                                                            reason=("evidence rejected: " + str(error))[:4096]))
            raise ValueError(f"collection evidence rejected; reserved attempt remains consumed: {error}") from error
        finally:
            if protocol is not None:
                protocol.close()
        execution_append(root, contract, events, "COLLECTION_OBSERVED", dict(attempt_key=key, reservation_id=reservation_id, observation=observation), evidence)
        return execution_inspect(root, events[-1]["event_sha256"])


def execution_compare(root, catalog_sha, comparison_key, expect_head=None):
    import quantization_execution_catalog as catalogs
    root = Path(root).resolve()
    with locked(root):
        contract, events, state = execution_replay(root, expect_head)
        require(catalog_sha in state["catalogs"], "catalog is not registered")
        catalog = state["catalogs"][catalog_sha]
        rows = [row for row in catalog["numeric_pairs"] if row["comparison_key"] == comparison_key]
        require(len(rows) == 1, "CPU comparison key is not registered")
        if comparison_key in state["comparisons"]:
            return execution_inspect(root, events[-1]["event_sha256"])
        row = rows[0]
        reference = catalogs.collection_attempt_key(catalog["numeric_workload_id"], row["reference_execution_spec_id"])
        candidate = row["collection_attempt_key"]
        require(reference in state["loaded"] and candidate in state["loaded"], "CPU comparison requires validated reference and candidate collections")
        metrics = execution_compare_data(state["loaded"][reference], state["loaded"][candidate])
        execution_append(root, contract, events, "NUMERIC_COMPARED", dict(catalog_sha256=catalog_sha, comparison_key=comparison_key,
                                    reference_attempt_key=reference, candidate_attempt_key=candidate, metrics=metrics))
        return execution_inspect(root, events[-1]["event_sha256"])


def execution_ingest_abba(root, reservation_id, directory, expect_head=None):
    from worker_protocol_tools import Protocol
    root, directory = Path(root).resolve(), Path(directory).resolve()
    with locked(root):
        contract, events, state = execution_replay(root, expect_head)
        matches = [row for row in state["reservations"].values() if row["reservation_id"] == reservation_id and row["stage"] == "ABBA"]
        require(len(matches) == 1, "ABBA requires its own existing reservation before ingestion")
        reservation = matches[0]
        key = reservation["attempt_key"]
        require(key not in state["performance"] and key not in state["failures"], "attempt is already terminal")
        protocol = None
        try:
            protocol = Protocol(ROOT / "crates/kata_worker/proto/worker.proto")
            catalog = state["catalogs"][reservation["catalog_sha256"]]
            observation, names = execution_load_abba(directory, catalog, reservation, state, protocol)
            evidence = snapshot(root, len(events), "abba", directory, names)
            for record in [*observation["original_files"], *observation["runtime_sources"], *contract["sources"]]:
                verify_record(record)
        except Exception as error:
            execution_append(root, contract, events, "ATTEMPT_FAILED", dict(attempt_key=key, reservation_id=reservation_id,
                                                                            reason=("ABBA evidence rejected: " + str(error))[:4096]))
            raise ValueError(f"ABBA evidence rejected; reserved attempt remains consumed: {error}") from error
        finally:
            if protocol is not None:
                protocol.close()
        execution_append(root, contract, events, "ABBA_OBSERVED", dict(attempt_key=key, reservation_id=reservation_id, observation=observation), evidence)
        return execution_inspect(root, events[-1]["event_sha256"])


def execution_summary(root, catalog_sha, output, expect_head=None):
    """Close the entire registered control matrix without ranking or adoption."""
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists() and output.parent.is_dir() and not output.is_relative_to(root),
            "summary output must be a NEW directory outside the ledger with an existing parent")
    with locked(root):
        contract, events, state = execution_replay(root, expect_head)
        require(catalog_sha in state["catalogs"], "catalog is not registered")
        catalog = state["catalogs"][catalog_sha]
        reference = catalog["identity"]["canonical_reference_execution_spec_id"]
        controls = [reference, *catalog["identity"]["required_control_execution_spec_ids"]]
        numerical = {row["candidate_execution_spec_id"]: state["comparisons"].get(row["comparison_key"], {}).get("metrics", {}).get("experimental_gate", {}).get("result", "NOT_RECORDED")
                     for row in catalog["numeric_pairs"]}
        candidates, incomplete = [], False
        for spec in catalog["execution_specs"]:
            if spec["kind"] != "unified_v1_recipe":
                continue
            candidate_id = spec["execution_spec_id"]
            matrix = []
            for control in controls:
                numeric_ids = {reference, control, candidate_id}
                numeric_gates = {identity: numerical[identity] for identity in sorted(numeric_ids)}
                cell = dict(baseline_execution_spec_id=control, candidate_execution_spec_id=candidate_id, numeric_gates=numeric_gates)
                if control == candidate_id:
                    cell.update(pair_key=None, status="SELF_REFERENCE" if all(value == "PASS" for value in numeric_gates.values()) else "NUMERIC_EVIDENCE_INCOMPLETE")
                else:
                    pair = next(row for row in catalog["performance_pairs"] if row["baseline_execution_spec_id"] == control
                                and row["candidate_execution_spec_id"] == candidate_id)
                    key = pair["pair_key"]
                    result = state["performance"].get(key)
                    cell["pair_key"] = key
                    if any(value == "FAIL" for value in numeric_gates.values()):
                        cell["status"] = "BLOCKED_NUMERIC_GATE"
                    elif any(value != "PASS" for value in numeric_gates.values()):
                        cell["status"] = "NUMERIC_EVIDENCE_INCOMPLETE"
                    elif result is not None:
                        # The first reservation freezes interpretation. Changing
                        # a metric/direction cannot rehabilitate a failed pair.
                        if result["metric"] != catalog["workload"]["metric"] or result["baseline_execution_spec_id"] != control or result["candidate_execution_spec_id"] != candidate_id:
                            cell["status"] = "DECISION_CONTEXT_MISMATCH"
                        else:
                            cell.update(status=result["gate"], pairwise=result["pairwise"], report_sha256=result["report_sha256"],
                                        actual_executions=result["actual_executions"])
                    elif key in state["failures"]:
                        cell["status"] = "FAILED_ATTEMPT_NO_RETRY"
                    else:
                        cell["status"] = "PENDING_RESERVED_NO_RETRY" if key in state["reservations"] else "NOT_RESERVED"
                matrix.append(cell)
            missing = any(cell["status"] in ("NUMERIC_EVIDENCE_INCOMPLETE", "PENDING_RESERVED_NO_RETRY", "NOT_RESERVED") for cell in matrix)
            incomplete |= missing
            passed = all(cell["status"] in ("PASS", "SELF_REFERENCE") for cell in matrix)
            candidates.append(dict(execution_spec_id=candidate_id, is_canonical_reference=candidate_id == reference,
                                   status="COMPLETE_MATRIX_PASSED_REQUIRES_CONFIRMATION" if passed else ("EVIDENCE_INCOMPLETE" if missing else "REJECTED_CONTROL_MATRIX"),
                                   complete_control_matrix_passed=passed, controls=matrix))
        # Until the finite matrix is closed there is no selected winner. Even a
        # closed positive matrix is unranked and needs independent confirmation.
        positive = [] if incomplete else [row["execution_spec_id"] for row in candidates if row["complete_control_matrix_passed"]]
        qualified = not catalog["workload"]["smoke"] and catalog["workload"]["split"] == "selection" and catalog["corpus_status"] == "READY"
        consumed = {stage: sum(row["stage"] == stage for row in state["reservations"].values()) for stage in ("COLLECTION", "ABBA")}
        report = dict(schema=EXECUTION_SCHEMA, report_kind="EXECUTION_SELECTION_SUMMARY", catalog_sha256=catalog_sha,
                      catalog_id=catalog["catalog_id"], ledger_id=contract["ledger_id"], head_sha256=events[-1]["event_sha256"], event_count=len(events),
                      conclusion="EVIDENCE_INCOMPLETE" if incomplete else ("CANDIDATE_REQUIRES_INDEPENDENT_CONFIRMATION" if positive else "NO_CANDIDATE_PASSED_COMPLETE_CONTROL_MATRIX"),
                      scope="QUALIFIED_SELECTION_EVIDENCE_ONLY" if qualified else "SMOKE_DIAGNOSTIC_ONLY",
                      canonical_reference_execution_spec_id=reference, required_control_execution_spec_ids=controls,
                      candidates=candidates, positive_candidates_unranked_requiring_confirmation=positive,
                      complete_finite_matrix=not incomplete, performance_ranking=None, numerical_gates=numerical,
                      remaining_attempts={stage: contract["budgets"][stage] - consumed[stage] for stage in consumed},
                      independent_confirmation="NOT_PERFORMED", original_fp32_gate="NOT_EVALUATED", playing_strength="NOT_MEASURED",
                      automatic_optimization_complete=False, formal_adoption_allowed=False, baseline_optimality="NOT_ESTABLISHED",
                      process_start_order_proven=False, holdout_outputs_read=0, deployment_adopted=False, publish_allowed=False, production_certified=False)
        execution_replay(root, events[-1]["event_sha256"])
        output.mkdir()
        write_new(output / "selection-summary.json", report)
        return dict(report=report, summary_file=file_record(output / "selection-summary.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    for name in ("search-plan", "workload", "baselines", "output"):
        init.add_argument("--" + name, type=Path, required=True)
    init.add_argument("--sgf-root", type=Path)
    execution_init = commands.add_parser("execution-init", help="initialize a separate catalog-v2 durable attempt ledger")
    execution_init.add_argument("--output", type=Path, required=True)
    execution_init.add_argument("--max-collection-attempts", type=int, required=True)
    execution_init.add_argument("--max-abba-attempts", type=int, required=True)
    for name in ("execution-inspect", "execution-register-catalog", "execution-reserve-collection", "execution-reserve-abba",
                 "execution-fail", "execution-ingest-collection", "execution-compare", "execution-ingest-abba", "execution-summary"):
        sub = commands.add_parser(name)
        sub.add_argument("--ledger", type=Path, required=True)
        sub.add_argument("--expect-head")
        if name == "execution-register-catalog":
            sub.add_argument("--catalog", type=Path, required=True)
            sub.add_argument("--expect-catalog-sha256", required=True)
        if name in ("execution-reserve-collection", "execution-reserve-abba", "execution-compare", "execution-summary"):
            sub.add_argument("--catalog-sha256", required=True)
        if name == "execution-reserve-collection":
            sub.add_argument("--collection-attempt-key", required=True)
        if name == "execution-reserve-abba":
            sub.add_argument("--pair-key", required=True)
        if name == "execution-compare":
            sub.add_argument("--comparison-key", required=True)
        if name in ("execution-fail", "execution-ingest-collection", "execution-ingest-abba"):
            sub.add_argument("--reservation-id", required=True)
        if name == "execution-fail":
            sub.add_argument("--reason", required=True)
        if name == "execution-ingest-collection":
            sub.add_argument("--collection", type=Path, required=True)
        if name == "execution-ingest-abba":
            sub.add_argument("--performance", type=Path, required=True)
        if name == "execution-summary":
            sub.add_argument("--output", type=Path, required=True)
    for name in ("status", "ingest-numeric", "ingest-performance", "summarize-selection"):
        sub = commands.add_parser(name)
        sub.add_argument("--ledger", type=Path, required=True)
        sub.add_argument("--expect-head")
        if name.startswith("ingest-"):
            sub.add_argument("--recipe-sha256", required=True)
        if name == "ingest-numeric":
            sub.add_argument("--reference", type=Path, required=True)
            sub.add_argument("--candidate", type=Path, required=True)
        if name == "ingest-performance":
            sub.add_argument("--performance", type=Path, required=True)
        if name == "summarize-selection":
            sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "execution-init":
            result = execution_initialize(args.output, args.max_collection_attempts, args.max_abba_attempts)
        elif args.command == "execution-inspect":
            result = execution_inspect(args.ledger, args.expect_head)
        elif args.command == "execution-register-catalog":
            result = execution_register_catalog(args.ledger, args.catalog, args.expect_catalog_sha256, args.expect_head)
        elif args.command in ("execution-reserve-collection", "execution-reserve-abba"):
            stage, key = ("COLLECTION", args.collection_attempt_key) if args.command == "execution-reserve-collection" else ("ABBA", args.pair_key)
            result = execution_reserve(args.ledger, args.catalog_sha256, stage, key, args.expect_head)
        elif args.command == "execution-fail":
            result = execution_fail(args.ledger, args.reservation_id, args.reason, args.expect_head)
        elif args.command == "execution-ingest-collection":
            result = execution_ingest_collection(args.ledger, args.reservation_id, args.collection, args.expect_head)
        elif args.command == "execution-compare":
            result = execution_compare(args.ledger, args.catalog_sha256, args.comparison_key, args.expect_head)
        elif args.command == "execution-ingest-abba":
            result = execution_ingest_abba(args.ledger, args.reservation_id, args.performance, args.expect_head)
        elif args.command == "execution-summary":
            result = execution_summary(args.ledger, args.catalog_sha256, args.output, args.expect_head)
        elif args.command == "init":
            result = initialize(args.search_plan, args.workload, args.baselines, args.output, args.sgf_root)
        elif args.command == "status":
            result = inspect(args.ledger, args.expect_head)
        elif args.command == "summarize-selection":
            result = summarize_selection(args.ledger, args.output, args.expect_head)
        else:
            result = ingest(args.ledger, "NUMERIC" if args.command == "ingest-numeric" else "PERFORMANCE", args.recipe_sha256,
                            reference=getattr(args, "reference", None), candidate=getattr(args, "candidate", None),
                            performance=getattr(args, "performance", None), expect_head=args.expect_head)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"quantization_selection_ledger: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
