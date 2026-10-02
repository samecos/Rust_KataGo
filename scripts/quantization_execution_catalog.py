#!/usr/bin/env python3
"""CPU-only preregistration of finite unified seeds and required legacy controls.

This catalog is a plan, not an attempt ledger or an observed runtime identity.
It cannot authorize GPU execution, certify a baseline, rank candidates, reset a
consumed budget, confirm a finalist, read holdout outputs or publish a result.
Future execution must consult a durable cross-catalog attempt ledger first.
"""
import argparse
import hashlib
import math
from pathlib import Path
import sys

import legacy_quantization_execution as legacy
import quantization_selection_ledger as ledger

SCHEMA = "rustgo-quantization-execution-catalog-v1"
SPEC_SCHEMA = "rustgo-planned-quantization-execution-spec-v1"
require, compact, digest = ledger.require, ledger.compact, ledger.digest
read_json, file_record, write_new = ledger.read_json, ledger.file_record, ledger.write_new


def execution_spec_id(semantic):
    return "rustgo-planned-execution-v1:" + digest(compact(semantic))


def fixed_unified_config(batch, recipe_sha):
    # The existing v1 collector defaults, with its generated path replaced by
    # content identity. No environment/config/tactic search is performed here.
    return dict(rules="chinese", komi="7.5", nnBackend="cudaquantbackend", nnMaxBatchSize=batch,
                numNNServerThreadsPerModel=1, numSearchThreads=1, maxVisits=1, ponderingEnabled="false",
                nnCacheSizePowerOfTwo=-1, nnMutexPoolSizePowerOfTwo=8,
                cudaQuantPlan=dict(recipe_sha256=recipe_sha))


def pair_key(stage, workload_id, reference_id, left, right):
    # ABBA is one unordered comparison. Control/candidate direction is retained
    # separately for interpretation, and cannot create another budget slot.
    identity = dict(schema="rustgo-semantic-experiment-key-v2", stage=stage, workload_id=workload_id,
                    execution_spec_ids=sorted([left, right]))
    if stage == "NUMERIC_COMPARE":
        identity["reference_execution_spec_id"] = reference_id
    else:
        require(stage == "ABBA", "pair stage must be ABBA or NUMERIC_COMPARE")
    return "rustgo-planned-pair-v2:" + digest(compact(identity))


def collection_attempt_key(workload_id, execution_id):
    return "rustgo-planned-collection-v2:" + digest(compact(dict(schema="rustgo-semantic-experiment-key-v2",
                    stage="NUMERIC_COLLECTION", workload_id=workload_id, execution_spec_id=execution_id)))


def numeric_registration(workload_id, reference_id, execution_id):
    attempt = collection_attempt_key(workload_id, execution_id)
    return dict(pair_key=attempt, collection_attempt_key=attempt,
                comparison_key=pair_key("NUMERIC_COMPARE", workload_id, reference_id, reference_id, execution_id),
                workload_id=workload_id, budget_unit="UNIQUE_SPEC_COLLECTION",
                reference_execution_spec_id=reference_id, candidate_execution_spec_id=execution_id,
                is_reference_self_check=execution_id == reference_id, maximum_attempts=1)


def ordered_request_semantics(records):
    """Hash precisely the ordered Position/EvalParameters protobuf messages.

    Protocol compiles the existing schema in this process, without spawning a
    tool or starting a server. It is also used by the actual collector. Corpus
    names, split labels, source paths, JSON whitespace and report hashes are
    provenance, not runtime inputs. No output protobuf is read here.
    """
    from google.protobuf.json_format import ParseDict
    from worker_protocol_tools import Protocol

    protocol = Protocol(ledger.ROOT / "crates/kata_worker/proto/worker.proto")
    result, seen = hashlib.sha256(), set()
    try:
        for row in records:
            position = ParseDict(row["position"], protocol.pb.Position(), ignore_unknown_fields=False)
            parameters = ParseDict(row["parameters"], protocol.pb.EvalParameters(), ignore_unknown_fields=False)
            raw_parts = [message.SerializeToString(deterministic=True) for message in (position, parameters)]
            framed = b"".join(len(raw).to_bytes(8, "little") + raw for raw in raw_parts)
            semantic_hash = hashlib.sha256(framed).digest()
            require(semantic_hash not in seen, "duplicate actual Position/EvalParameters despite different corpus metadata")
            seen.add(semantic_hash)
            result.update(len(framed).to_bytes(8, "little") + framed)
    finally:
        protocol.close()
    return dict(schema="rustgo-ordered-position-evalparameters-v1", encoding="deterministic-protobuf-length-framed",
                count=len(records), sha256=result.hexdigest())


def stage_workloads(work, model_sha, ordered):
    # Normalize equivalent JSON numbers to the actual CLI float and wire lease.
    timeout = float(work["task_timeout"])
    common = dict(schema="rustgo-semantic-stage-workload-v2", model_sha256=model_sha,
                  ordered_request_semantics=ordered, concurrency=work["concurrency"], capacity=work["capacity"],
                  task_timeout_seconds=timeout, lease_ms=math.ceil(timeout * 1000), window_mode="continuous")
    numeric = dict(**common, stage="NUMERIC_COLLECTION", complete_corpus_passes=1)
    abba = dict(**common, stage="ABBA", warmup_requests=work["warmup"], complete_corpus_cycles=work["cycles"],
                order=["baseline", "candidate", "candidate", "baseline"])
    return numeric, abba


def unique_records(records):
    result = {}
    for item in records:
        require(item["path"] not in result or ledger.same(item, result[item["path"]]), "conflicting source identity")
        result[item["path"]] = item
    return [result[path] for path in sorted(result)]


def build_catalog(search_plan, workload_path, unified_binary, batches, legacy_specs, *, reference_batch, scratch_parent, sgf_root=None):
    search_plan, workload_path, unified_binary = (Path(p).resolve() for p in (search_plan, workload_path, unified_binary))
    require(type(batches) in (list, tuple) and 1 <= len(batches) <= 64
            and all(type(batch) is int and 1 <= batch <= 64 for batch in batches), "explicit batches must be integers in 1..64")
    batches = sorted(set(batches))
    require(type(reference_batch) is int and reference_batch in batches, "reference batch must be an explicitly registered unified batch")
    require(type(legacy_specs) in (list, tuple) and 1 <= len(legacy_specs) <= 64, "explicit finite legacy controls are required")
    spec_paths = sorted({str(Path(path).resolve()) for path in legacy_specs})
    if sgf_root is not None:
        sgf_root = Path(sgf_root).resolve()
        require(sgf_root.is_dir(), "explicit SGF root must exist")
    work = read_json(workload_path)
    require(work.get("split") in ("calibration", "selection"), "holdout outputs are forbidden; workload must select calibration or selection")
    plan, sources = ledger.validate_search(search_plan, Path(scratch_parent), sgf_root)
    ledger.validate_workload(work, plan)
    candidates = plan["candidates"]
    require(1 <= len(candidates) <= 3 and candidates[0]["name"] == "fp16", "only existing v1 FP16/INT8 seeds are supported")
    manifest = read_json(plan["source_paths"]["model_manifest"])
    template = read_json(plan["source_paths"]["fp16_template"])
    model_sha = plan["identity"]["source_sha256"]["model"]
    ledger.planner.validate_model_inputs(manifest, template, model_sha)
    shapes_sha = digest(compact(sorted(manifest["projections"], key=lambda row: row["id"])))
    binary_record = file_record(unified_binary)
    corpus_path = Path(plan["source_paths"]["corpus_manifest"])
    corpus, _, blobs, requests, _, _ = ledger.planner.validate_corpus(corpus_path, sgf_root)
    count = len(requests[ledger.planner.SPLITS.index(work["split"])])
    require(0 < count <= 100000, "selected complete split requires 1..100000 requests")
    require(work["warmup"] + count * work["cycles"] <= 1000000, "per-arm request budget exceeds one million")
    require(work["smoke"] or count * work["cycles"] >= 1024, "formal ABBA needs >=1024 complete-corpus requests")
    ordered = ordered_request_semantics(requests[ledger.planner.SPLITS.index(work["split"])])
    numeric_workload, abba_workload = stage_workloads(work, model_sha, ordered)
    numeric_workload_id, abba_workload_id = digest(compact(numeric_workload)), digest(compact(abba_workload))
    # Retain workload_id as the application performance workload. Numeric
    # collection accounting has its own identity independent of ABBA settings.
    workload_id = abba_workload_id
    rows, reference_id = {}, None

    def add(kind, execution, labels, bindings):
        semantic = dict(schema=SPEC_SCHEMA, kind=kind, model_sha256=model_sha,
                        graph_sha256=manifest["graph_sha256"], projection_shapes_sha256=shapes_sha, execution=execution)
        identity = execution_spec_id(semantic)
        row = rows.setdefault(identity, dict(execution_spec_id=identity, kind=kind, semantic=semantic, roles=[], labels=[], bindings=[]))
        row["labels"] = sorted(set(row["labels"]) | set(labels))
        row["bindings"] = unique_records([*row["bindings"], *bindings])
        return identity

    for candidate in candidates:
        recipe_path = ledger.safe_file(search_plan.parent, candidate["recipe_file"])
        recipe = read_json(recipe_path)
        require(digest(ledger.planner.compact(ledger.planner.canonical_recipe(recipe,
                    {p["id"]: p["precision"] for p in recipe["projections"]}))) == candidate["recipe_sha256"], "candidate recipe semantic SHA differs")
        require(all(p["precision"] in ("fp16", "int8") for p in recipe["projections"]), "only v1 FP16/INT8 seeds are allowed")
        for batch in batches:
            execution = dict(backend="cudaquantbackend", binary_sha256=binary_record["sha256"], batch=batch,
                        configuration=fixed_unified_config(batch, candidate["recipe_sha256"]), environment={},
                        precision_recipe=ledger.planner.canonical_recipe(recipe, {p["id"]: p["precision"] for p in recipe["projections"]}),
                        recipe_sha256=candidate["recipe_sha256"])
            identity = add("unified_v1_recipe", execution, candidate["aliases"], [binary_record, file_record(recipe_path)])
            rows[identity]["roles"] = ["UNIFIED_CANDIDATE"]
            if candidate["name"] == "fp16" and batch == reference_batch:
                require(all(p["precision"] == "fp16" for p in recipe["projections"]) and reference_id is None,
                        "canonical reference must be unique complete unified FP16")
                reference_id = identity
                rows[identity]["roles"].append("CANONICAL_FP16_REFERENCE")
    require(reference_id is not None, "canonical unified FP16 reference is absent")
    control_backends = set()
    for path in spec_paths:
        loaded = legacy.load_spec(path)
        require(loaded["spec"]["model"]["sha256"] == model_sha, "legacy control targets a different model")
        control_backends.add(loaded["backend"])
        bindings = [file_record(path), *[file_record(item["path"]) for item in loaded["files"]]]
        identity = add("legacy_explicit_spec", loaded["semantic"],
                    [loaded["spec"]["label"]] if loaded["spec"].get("label") else [], bindings)
        rows[identity]["roles"] = ["REQUIRED_LEGACY_CONTROL"]
        sources.extend(bindings)
    require(control_backends == {"cudabackend", "cudaint8backend"}, "incomplete control set: explicit legacy FP16 and INT8 are both required")
    controls = sorted(identity for identity, row in rows.items() if row["kind"] == "legacy_explicit_spec")
    unified = sorted(identity for identity, row in rows.items() if row["kind"] == "unified_v1_recipe")
    numeric = [numeric_registration(numeric_workload_id, reference_id, identity)
               for identity in [reference_id, *sorted(set(rows) - {reference_id})]]
    performance = {}
    for candidate in unified:
        for control in [reference_id, *controls]:
            if candidate == control:
                continue
            key = pair_key("ABBA", abba_workload_id, reference_id, control, candidate)
            require(key not in performance, "duplicate unordered performance pair")
            performance[key] = dict(pair_key=key, workload_id=abba_workload_id, baseline_execution_spec_id=control, candidate_execution_spec_id=candidate,
                    reference_execution_spec_id=reference_id, order=["baseline", "candidate", "candidate", "baseline"], maximum_attempts=1,
                    requires_numeric_pass_for=sorted({reference_id, control, candidate}))
    require(len(numeric) <= work["max_numeric_evaluations"], "numeric budget cannot cover the entire unique execution set")
    require(len(performance) <= work["max_performance_evaluations"], "performance budget cannot cover every required control pair")
    sources.extend([file_record(workload_path), binary_record, file_record(Path(__file__)), *ledger.source_versions()])
    sources = unique_records(sources)
    for record in sources:
        ledger.verify_record(record)
    semantics = dict(schema=SCHEMA, search_id=plan["search_id"], workload_id=workload_id,
                    experiment_key_schema="rustgo-semantic-experiment-key-v2", numeric_workload_id=numeric_workload_id,
                    abba_workload_id=abba_workload_id, decision_settings=dict(metric=work["metric"], smoke=work["smoke"], split=work["split"],
                         minimum_improvement=work["minimum_improvement"], maximum_relative_spread=work["maximum_relative_spread"]),
                    execution_spec_ids=sorted(rows), canonical_reference_execution_spec_id=reference_id,
                    required_control_execution_spec_ids=controls, numeric_pair_keys=[row["pair_key"] for row in numeric],
                    performance_pair_keys=sorted(performance),
                    budget=dict(numeric_attempts=len(numeric), maximum_numeric_attempts=work["max_numeric_evaluations"],
                                abba_attempts=len(performance), maximum_abba_attempts=work["max_performance_evaluations"]))
    return dict(schema=SCHEMA, catalog_id=digest(compact(semantics)), identity=semantics,
                inputs=dict(search_plan=str(search_plan), workload=str(workload_path), unified_binary=str(unified_binary),
                            batches=batches, reference_batch=reference_batch, legacy_specs=spec_paths, sgf_root=str(sgf_root) if sgf_root else None),
                source_records=sources, workload=work, workload_id=workload_id,
                numeric_workload=numeric_workload, numeric_workload_id=numeric_workload_id,
                abba_workload=abba_workload, abba_workload_id=abba_workload_id, selected_request_count=count,
                corpus_status=plan["corpus_status"], execution_specs=[rows[key] for key in sorted(rows)],
                numeric_pairs=numeric, performance_pairs=[performance[key] for key in sorted(performance)],
                status="PLANNED_UNEXECUTED", gpu_execution_authorized=False, subprocesses_started=0, holdout_outputs_read=0,
                actual_execution_observed=False, driver_api_version=None, gpu_uuid=None, deployment_adopted=False,
                publish_allowed=False, production_certified=False, baseline_optimality="NOT_ESTABLISHED",
                independent_confirmation="NOT_REGISTERED_OR_AUTHORIZED", attempts_consumed=None, remaining_budget=None,
                limitations=["planning only; not a ledger and does not prove unused or remaining attempt budget",
                             "future execution must reject previously attempted semantic pair keys across new catalog directories",
                             "numeric pair_key aliases collection_attempt_key; comparison_key never authorizes another collection",
                             "all controls require new numerical qualification and observed execution identity on the same workload",
                             "canonical FP16 is a numerical reference, not a declaration of fastest or best certified baseline",
                             "source paths must remain present and unchanged; retain catalog SHA independently",
                             "no holdout evaluation, confirmation, deployment, original certification reuse or v1 ledger admission"])


def create_catalog(search_plan, workload, unified_binary, batches, legacy_specs, output, *, reference_batch, sgf_root=None):
    output = Path(output).resolve()
    require(not output.exists() and output.parent.is_dir(), "catalog output must be a NEW directory with an existing parent")
    contract = build_catalog(search_plan, workload, unified_binary, batches, legacy_specs,
                             reference_batch=reference_batch, scratch_parent=output.parent, sgf_root=sgf_root)
    output.mkdir()
    write_new(output / "catalog.json", contract)
    return dict(status="PLANNED_UNEXECUTED", catalog=file_record(output / "catalog.json"), catalog_id=contract["catalog_id"],
                unique_execution_specs=len(contract["execution_specs"]), budget=contract["identity"]["budget"],
                gpu_execution_authorized=False, attempts_consumed=None, remaining_budget=None, publish_allowed=False)


def validate_catalog(path, expect_sha=None):
    path = Path(path).resolve()
    require(path.name == "catalog.json" and not path.is_symlink(), "verification requires catalog.json")
    require(expect_sha is None or file_record(path)["sha256"] == expect_sha, "catalog SHA differs from expected anchor")
    contract = read_json(path)
    require(contract.get("schema") == SCHEMA, "unsupported catalog schema")
    for record in contract["source_records"]:
        ledger.verify_record(record)
    inputs = contract["inputs"]
    rebuilt = build_catalog(inputs["search_plan"], inputs["workload"], inputs["unified_binary"], inputs["batches"], inputs["legacy_specs"],
                             reference_batch=inputs["reference_batch"], scratch_parent=path.parent,
                             sgf_root=inputs["sgf_root"])
    require(ledger.same(contract, rebuilt), "catalog differs from independently rebuilt sources, roles, pairs or budget")
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    for name in ("search-plan", "workload", "unified-binary", "output"):
        prepare.add_argument("--" + name, type=Path, required=True)
    prepare.add_argument("--batches", type=int, nargs="+", required=True)
    prepare.add_argument("--reference-batch", type=int, required=True)
    prepare.add_argument("--legacy-spec", type=Path, action="append", required=True)
    prepare.add_argument("--sgf-root", type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--catalog", type=Path, required=True)
    verify.add_argument("--expect-sha256")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = create_catalog(args.search_plan, args.workload, args.unified_binary, args.batches, args.legacy_spec,
                                    args.output, reference_batch=args.reference_batch, sgf_root=args.sgf_root)
        else:
            contract = validate_catalog(args.catalog, args.expect_sha256)
            result = dict(status="CPU_VERIFIED_UNEXECUTED", catalog_id=contract["catalog_id"], gpu_execution_authorized=False,
                          attempts_consumed=None, remaining_budget=None, publish_allowed=False)
        print(ledger.planner.pretty(result).decode(), end="")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
