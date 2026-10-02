#!/usr/bin/env python3
"""Compare two frozen collect_quantization_corpus directories entirely on CPU.

The reference must contain an explicit, complete, all-FP16 precision recipe whose
canonical identity agrees with the actually loaded profile. Verify corpus files,
request indices/protobufs, output metadata/protobuf hashes, and full semantic
coverage before comparing. Metrics use protobuf values, never rounded JSON floats.

The maximum white-win-probability difference must be <= 0.06. Nonfinite outputs
and policy legality changes are hard failures. This experiment never certifies
FP32 accuracy or playing strength. NOT_READY or non-holdout data is diagnostic;
even a ready holdout pass is only an experimental holdout result.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from collect_quantization_corpus import check_sha, load_corpus, load_mxfp8_inspection, read_jsonl, sha256, validate_collection_window_mode
from validate_quantized_backend import PROFILE_LINE, save_json, strict_json


ROOT = Path(__file__).resolve().parents[1]
WIN_LIMIT = 0.06
META_KEYS = ("name", "game_id", "split", "ply", "phase", "semantic_position_sha256", "board_state_sha256")
RECIPE_KEYS = ("schema", "version", "quantization_semantics_version", "model_sha256", "graph_sha256", "projections")
PROJECTION_KEYS = ("id", "expected_n", "expected_k", "precision")


def bound_artifacts(directory, artifacts, required):
    seen, raw_files = set(), {}
    for artifact in artifacts:
        name = artifact["file"]
        if not isinstance(name, str) or Path(name).name != name or name in seen:
            raise ValueError("duplicate or nonlocal artifact name")
        seen.add(name)
        check_sha(artifact["sha256"], f"artifact {name}")
        raw = (directory / name).read_bytes()
        if sha256(raw) != artifact["sha256"] or ("bytes" in artifact and len(raw) != artifact["bytes"]):
            raise ValueError(f"artifact hash/size mismatch: {directory / name}")
        raw_files[name] = raw
    if not required <= seen:
        raise ValueError(f"missing bound artifacts: {sorted(required - seen)}")
    return raw_files


def frozen_recipe(directory, report, full_reference=None):
    if report["backend"] != "cudaquantbackend" or not report.get("actual_profile"):
        if full_reference is None:
            raise ValueError("reference requires an actual unified profile and frozen explicit all-FP16 recipe")
        return None
    if full_reference is not None and not report.get("recipe_file_sha256"):
        # The candidate may have used the unified backend's implicit FP16 base.
        # Accept only the already verified complete FP16 canonical identity.
        canonical_hash = sha256(json.dumps(full_reference, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))
        if ((directory / "recipe.json").exists() or re.search(r"(?m)^cudaQuantPlan\s*=", report["config_text"])
                or report["actual_profile"]["recipe_sha256"] != canonical_hash
                or report["actual_profile"]["graph_sha256"] != full_reference["graph_sha256"]):
            raise ValueError("candidate without a frozen recipe does not match the verified implicit FP16 base")
        return full_reference
    raw = (directory / "recipe.json").read_bytes()
    if sha256(raw) != report.get("recipe_file_sha256"):
        raise ValueError("frozen recipe file hash mismatch or no explicitly frozen recipe")
    recipe = strict_json(raw)
    if isinstance(recipe, dict) and (recipe.get("version") == 2 or recipe.get("quantization_semantics_version") == 2
            or (isinstance(recipe.get("projections"), list)
                and any(isinstance(p, dict) and p.get("precision") == "mxfp8" for p in recipe["projections"]))):
        if full_reference is None:
            raise ValueError("reference must be the verified complete v1 all-FP16 recipe, never an MXFP8 recipe")
        return load_mxfp8_inspection(directory, report, raw, full_reference=full_reference)
    if report.get("recipe_inspection"):
        raise ValueError("MXFP8 inspection cannot be attached to a legacy v1 recipe")
    if (set(recipe) != set(RECIPE_KEYS) or recipe["schema"] != "rustgo-precision-recipe"
            or type(recipe["version"]) is not int or recipe["version"] != 1
            or type(recipe["quantization_semantics_version"]) is not int or recipe["quantization_semantics_version"] != 1
            or recipe["model_sha256"] != report["model_sha256"]
            or recipe["graph_sha256"] != report["actual_profile"]["graph_sha256"]):
        raise ValueError("unsupported or mismatched frozen precision recipe")
    projections = {}
    for p in recipe["projections"]:
        if (set(p) != set(PROJECTION_KEYS) or not isinstance(p["id"], str) or p["id"] in projections
                or re.fullmatch(r"trunk\.block\d+\.pair\d+\.(?:attention\.(?:qkv|out)|ffn\.(?:dual|down))", p["id"]) is None
                or any(type(p[key]) is not int or p[key] <= 0 for key in ("expected_n", "expected_k"))
                or p["precision"] not in ("fp16", "int8")):
            raise ValueError("invalid/duplicate projection in frozen precision recipe")
        projections[p["id"]] = {key: p[key] for key in PROJECTION_KEYS}
    if full_reference is None:
        if not projections or any(p["precision"] != "fp16" for p in projections.values()):
            raise ValueError("reference frozen recipe must explicitly contain only FP16 projections")
        resolved = projections
    else:
        if recipe["graph_sha256"] != full_reference["graph_sha256"]:
            raise ValueError("candidate/reference graph identity differs")
        resolved = {p["id"]: dict(p) for p in full_reference["projections"]}
        for key, p in projections.items():
            if key not in resolved or any(p[k] != resolved[key][k] for k in ("expected_n", "expected_k")):
                raise ValueError("candidate projection ID/shape differs from complete reference recipe")
            resolved[key] = p
    # Rust hashes serialized PrecisionRecipe in struct field order, with ALL
    # projections in BTreeMap ID order. No floating scalars occur in this schema.
    canonical = {key: recipe[key] for key in RECIPE_KEYS[:-1]}
    canonical["projections"] = [resolved[key] for key in sorted(resolved)]
    canonical_hash = sha256(json.dumps(canonical, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    if canonical_hash != report["actual_profile"]["recipe_sha256"]:
        raise ValueError("canonical frozen recipe identity mismatch (reference must include the complete projection list)")
    return canonical


def load_collection(directory, protocol, *, full_reference=None, max_requests=100000):
    from google.protobuf.json_format import ParseDict
    from compare_worker_outputs import validate_result

    report_raw = (directory / "report.json").read_bytes()
    report = strict_json(report_raw)
    if (report.get("schema") != "rustgo-quantization-collection-v1" or report.get("status") != "COLLECTED_UNVERIFIED"
            or report.get("drained") is not True or report.get("backend") not in ("cudabackend", "cudaint8backend", "cudaquantbackend")
            or not 1 <= report["expected_requests"] <= max_requests
            or report["completed_requests"] != report["expected_requests"]):
        raise ValueError("collection is incomplete or has an unsupported schema/backend/status")
    for key in ("model_sha256", "binary_sha256", "schema_sha256", "requests_sha256", "corpus_manifest_sha256",
                "corpus_request_set_sha256", "source_request_file_sha256", "ordered_output_hashes_sha256"):
        check_sha(report[key], key)
    if report["schema_sha256"] != sha256((ROOT / "crates/kata_worker/proto/worker.proto").read_bytes()):
        raise ValueError("collection protobuf schema differs from the available parser")
    files = bound_artifacts(directory, report["artifacts"], {"outputs.jsonl", "requests-index.jsonl", "worker.cfg"})
    validate_collection_window_mode(directory, report)
    legacy_observation = None
    if "legacy_execution" in report:
        from legacy_quantization_execution import validate_collection
        legacy_observation = validate_collection(directory, report)
    if report.get("recipe_inspection") and not {"recipe.json", "worker.log"} <= files.keys():
        raise ValueError("MXFP8 collection must bind its frozen recipe and actual Worker log as artifacts")
    if report.get("recipe_inspection") and report["backend"] != "cudaquantbackend":
        raise ValueError("MXFP8 recipe inspection requires the actual unified backend")
    # Collector Path.read_text uses universal newlines, while file SHA binds
    # the original Windows CRLF bytes. Check both representations explicitly.
    config_text = files["worker.cfg"].decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    if config_text != report["config_text"] or sha256(files["worker.cfg"]) != report["config_sha256"]:
        raise ValueError("collection config identity mismatch")
    source_files = bound_artifacts(directory / "source", report["source_artifacts"],
                                   {"manifest.json", "games.jsonl", f"{report['split']}.requests.jsonl"})
    if (sha256(source_files["manifest.json"]) != report["corpus_manifest_sha256"]
            or sha256(source_files[f"{report['split']}.requests.jsonl"]) != report["source_request_file_sha256"]):
        raise ValueError("collection source identities disagree with artifacts")
    records, games, corpus, _ = load_corpus(directory / "source" / f"{report['split']}.requests.jsonl",
                                           directory / "source/manifest.json", max_requests)
    if (len(records) != report["expected_requests"] or corpus["request_set_sha256"] != report["corpus_request_set_sha256"]
            or corpus["status"] != report["corpus_status"] or records[0]["split"] != report["split"]
            or len({r["game_id"] for r in records}) != report["unique_games"]):
        raise ValueError("collection source counts/split/readiness identity mismatch")
    heartbeat = report["final_heartbeat"]
    if (int(heartbeat["completed_requests"]) != len(records) or int(heartbeat["nn_rows"]) != len(records)
            or int(heartbeat["failed_requests"]) != 0 or int(heartbeat["in_flight"]) != 0
            or int(heartbeat["nn_batches"]) <= 0):
        raise ValueError("collection final heartbeat does not confirm complete uncached evaluation")
    hello = ParseDict(report["hello"], protocol.pb.WorkerHello(), ignore_unknown_fields=False)
    if (hello.model_sha256 != report["model_sha256"] or hello.protocol_version != 1 or hello.input_profile != "katago-eval-v1"
            or hello.max_in_flight != report["capacity"] or not hello.supports_ownership or hello.model_version <= 0
            or re.search(rf"(?:^|;\s*)backend={report['backend']}(?:;|$)", hello.backend_info) is None):
        raise ValueError("actual Worker Hello disagrees with collection identity")
    if report["backend"] == "cudaquantbackend":
        profile = report["actual_profile"]
        for key in ("model_sha256", "graph_sha256", "recipe_sha256"):
            check_sha(profile[key], f"actual_profile.{key}")
        actual_id = profile["inference_profile_id"]
        if (profile["model_sha256"] != report["model_sha256"] or re.fullmatch(r"rustgo-quant-v1:[0-9a-f]{64}", actual_id) is None
                or re.findall(r"(?:^|;\s*)inference-profile=([^;\s]+)", hello.backend_info) != [actual_id]
                or re.findall(r"(?m)^cudaQuantExpectedProfile=(\S+)\s*$", report["config_text"]) != [actual_id]):
            raise ValueError("actual inference profile does not match Worker/config")
        logged = set(PROFILE_LINE.findall((directory / "worker.log").read_text(encoding="utf-8", errors="replace")))
        if logged != {(profile["model_sha256"], profile["graph_sha256"], profile["recipe_sha256"], actual_id)}:
            raise ValueError("Worker log does not bind the actual model/graph/recipe/profile")
    recipe = frozen_recipe(directory, report, full_reference)
    index = list(read_jsonl(files["requests-index.jsonl"], "requests-index.jsonl"))
    if len(index) != len(records):
        raise ValueError("incomplete request index")
    requests, by_semantic, request_digest = {}, {}, hashlib.sha256()
    for task_id, (record, entry) in enumerate(zip(records, index, strict=True), 1):
        if any(entry[key] != record[key] for key in META_KEYS) or entry["task_id"] != task_id:
            raise ValueError("request index changed order, semantic identity or corpus metadata")
        if entry["source_sgfs"] != games[record["game_id"]]["sources"]:
            raise ValueError("request index source SGF provenance mismatch")
        raw = (directory / "requests" / f"{task_id:06d}.pb").read_bytes()
        request = protocol.pb.EvalRequest.FromString(raw)
        if sha256(raw) != entry["request_sha256"] or request.SerializeToString(deterministic=True) != raw:
            raise ValueError("request protobuf hash/canonical serialization mismatch")
        indexed = ParseDict(entry["request"], protocol.pb.EvalRequest(), ignore_unknown_fields=False)
        if indexed.SerializeToString(deterministic=True) != raw:
            raise ValueError("request JSON index differs from exact protobuf")
        position = ParseDict(record["position"], protocol.pb.Position(), ignore_unknown_fields=False)
        parameters = ParseDict(record["parameters"], protocol.pb.EvalParameters(), ignore_unknown_fields=False)
        if (request.position.SerializeToString(deterministic=True) != position.SerializeToString(deterministic=True)
                or request.parameters.SerializeToString(deterministic=True) != parameters.SerializeToString(deterministic=True)
                or request.model_sha256 != report["model_sha256"] or request.task_id != task_id or request.generation != 1
                or request.session_id != f"quant-corpus-{report['source_request_file_sha256'][:24]}"
                or not 0 < request.lease_ms <= 3600000 or not request.parameters.skip_cache):
            raise ValueError("request protobuf changed corpus semantics or envelope identity")
        without_hash = protocol.pb.EvalRequest()
        without_hash.CopyFrom(request)
        without_hash.ClearField("input_hash")
        if request.input_hash.hex() != entry["input_hash"] or request.input_hash.hex() != sha256(without_hash.SerializeToString(deterministic=True)):
            raise ValueError("request input_hash contract mismatch")
        request_digest.update(len(raw).to_bytes(8, "little") + raw)
        requests[task_id] = (record, request)
    if request_digest.hexdigest() != report["requests_sha256"]:
        raise ValueError("complete ordered request protobuf set hash mismatch")
    output_hashes = {}
    for entry in read_jsonl(files["outputs.jsonl"], "outputs.jsonl"):
        task_id = int(entry["result"]["task_id"])
        if task_id not in requests or task_id in output_hashes:
            raise ValueError("unknown or duplicate output task ID")
        record, request = requests[task_id]
        if any(entry[key] != record[key] for key in META_KEYS):
            raise ValueError("output metadata changed semantic request identity")
        raw = (directory / "results" / f"{task_id:06d}.pb").read_bytes()
        result = protocol.pb.EvalResult.FromString(raw)
        if sha256(raw) != entry["result_sha256"] or result.SerializeToString(deterministic=True) != raw:
            raise ValueError("result protobuf hash/canonical serialization mismatch")
        indexed = ParseDict(entry["result"], protocol.pb.EvalResult(), ignore_unknown_fields=False)
        if indexed.SerializeToString(deterministic=True) != raw:
            raise ValueError("result JSON metadata/float values do not round-trip to saved protobuf")
        validate_result(result, request, hello)
        output_raw = result.output.SerializeToString(deterministic=True)
        if sha256(output_raw) != entry["output_sha256"]:
            raise ValueError("NNOutput hash differs from saved protobuf")
        semantic = record["semantic_position_sha256"]
        if semantic in by_semantic:
            raise ValueError("duplicate semantic position in output set")
        by_semantic[semantic] = dict(record=record, request=request, output=result.output, raw=output_raw)
        output_hashes[task_id] = entry["output_sha256"]
    if set(output_hashes) != set(requests):
        raise ValueError("missing output task IDs")
    ordered = hashlib.sha256()
    for task_id in sorted(output_hashes):
        ordered.update(task_id.to_bytes(8, "little") + bytes.fromhex(output_hashes[task_id]))
    if ordered.hexdigest() != report["ordered_output_hashes_sha256"]:
        raise ValueError("complete ordered output set hash mismatch")
    return dict(report=report, report_sha256=sha256(report_raw), recipe=recipe, positions=by_semantic,
                legacy_observation=legacy_observation)


def stats(values):
    ordered = sorted(values)
    return dict(max=max(ordered), mean=math.fsum(ordered) / len(ordered),
                p95=ordered[math.ceil(0.95 * len(ordered)) - 1], count=len(ordered))


def compare(reference, candidate):
    a, b = reference["report"], candidate["report"]
    for key in ("model_sha256", "schema_sha256", "corpus_manifest_sha256", "corpus_request_set_sha256",
                "source_request_file_sha256", "requests_sha256", "split", "expected_requests"):
        if a[key] != b[key]:
            raise ValueError(f"candidate/reference collection identity differs: {key}")
    left, right = reference["positions"], candidate["positions"]
    if left.keys() != right.keys():
        raise ValueError("candidate/reference semantic request coverage differs")
    win, score, kl, intersections, top_ranks = [], [], [], [], []
    policy_sum = policy_sq = owner_sum = owner_sq = 0.0
    policy_count = owner_count = 0
    policy_max = owner_max = 0.0
    top1_matches = exact = 0
    hard_failures, cases = [], []
    for semantic in sorted(left):
        ref, got = left[semantic], right[semantic]
        if ref["record"] != got["record"] or ref["request"].SerializeToString(deterministic=True) != got["request"].SerializeToString(deterministic=True):
            raise ValueError("semantic SHA collision or confused complete request")
        x, y = ref["output"], got["output"]
        p, q = list(x.policy), list(y.policy)
        legal = {i for i, value in enumerate(p) if value >= 0}
        other_legal = {i for i, value in enumerate(q) if value >= 0}
        legal_equal = legal == other_legal
        if not legal_equal:
            hard_failures.append(dict(semantic_position_sha256=semantic, name=ref["record"]["name"],
                                      reason="policy legality set changed", reference_only=sorted(legal - other_legal),
                                      candidate_only=sorted(other_legal - legal)))
        ranks_a = sorted(legal, key=lambda i: (-p[i], i))
        ranks_b = sorted(other_legal, key=lambda i: (-q[i], i))
        if not ranks_a or not ranks_b:
            raise ValueError("empty legal policy support")
        top1 = ranks_a[0] == ranks_b[0]
        top1_matches += top1
        k = min(5, len(ranks_a), len(ranks_b))
        intersection = len(set(ranks_a[:k]) & set(ranks_b[:k])) / k
        intersections.append(intersection)
        rank = ranks_b.index(ranks_a[0]) + 1 if ranks_a[0] in other_legal else None
        if rank is not None:
            top_ranks.append(rank)
        divergence = None
        if legal_equal:
            divergence = max(0.0, math.fsum(p[i] * math.log(p[i] / max(q[i], 1e-30)) for i in legal if p[i] > 0))
            kl.append(divergence)
            errors = [abs(p[i] - q[i]) for i in legal]
            policy_sum += math.fsum(errors)
            policy_sq += math.fsum(error * error for error in errors)
            policy_count += len(errors)
            policy_max = max(policy_max, max(errors))
        owner_errors = [abs(u - v) for u, v in zip(x.ownership, y.ownership, strict=True)]
        owner_sum += math.fsum(owner_errors)
        owner_sq += math.fsum(error * error for error in owner_errors)
        owner_count += len(owner_errors)
        owner_max = max(owner_max, max(owner_errors))
        win_error, score_error = abs(x.white_win_prob - y.white_win_prob), abs(x.white_score_mean - y.white_score_mean)
        win.append(win_error)
        score.append(score_error)
        is_exact = ref["raw"] == got["raw"]
        exact += is_exact
        cases.append(dict(semantic_position_sha256=semantic, name=ref["record"]["name"], game_id=ref["record"]["game_id"],
                          ply=ref["record"]["ply"], phase=ref["record"]["phase"], exact_output_bytes=is_exact,
                          reference_white_win_prob=x.white_win_prob, candidate_white_win_prob=y.white_win_prob,
                          white_win_probability_abs_error=win_error, white_score_mean_abs_error=score_error,
                          policy_legal_set_equal=legal_equal, policy_kl=divergence, policy_top1_equal=top1,
                          policy_top5_intersection_fraction=intersection, candidate_rank_of_reference_top1=rank,
                          ownership_max_abs_error=max(owner_errors)))
    win_stats = stats(win)
    passed = not hard_failures and win_stats["max"] <= WIN_LIMIT
    holdout = a["split"] == "holdout" and a["corpus_status"] == "READY" and b["corpus_status"] == "READY"
    status = ("EXPERIMENTAL_HOLDOUT_" if holdout else "DIAGNOSTIC_") + ("PASSED" if passed else "FAILED")
    return dict(status=status, cases=cases, positions=len(cases), unique_games=a["unique_games"],
                exact_output_bytes=dict(matches=exact, total=len(cases), all_exact=exact == len(cases)),
                experimental_gate=dict(result="PASS" if passed else "FAIL", metric="maximum absolute white_win_prob difference",
                                       limit=WIN_LIMIT, measured=win_stats["max"], all_win_statistics_within_limit=all(win_stats[k] <= WIN_LIMIT for k in ("max", "mean", "p95"))),
                win_probability_abs_error=win_stats, score_mean_abs_error=stats(score),
                policy=dict(legality_matches=len(cases) - len(hard_failures), top1_matches=top1_matches,
                            top1_fraction=top1_matches / len(cases), kl=stats(kl) if kl else None, kl_candidate_floor=1e-30,
                            top5_intersection_fraction=stats(intersections), candidate_rank_of_reference_top1=stats(top_ranks) if top_ranks else None,
                            max_abs_error=policy_max, mean_abs_error=policy_sum / policy_count if policy_count else None,
                            rmse=math.sqrt(policy_sq / policy_count) if policy_count else None,
                            ranking_ties="probability descending, vertex ascending; top-k includes pass if ranked"),
                ownership=dict(max_abs_error=owner_max, mean_abs_error=owner_sum / owner_count,
                               rmse=math.sqrt(owner_sq / owner_count), points=owner_count),
                hard_gates=dict(nonfinite_outputs="PASS", policy_legality="PASS" if not hard_failures else "FAIL", failures=hard_failures),
                percentile_method="nearest rank: sorted[ceil(0.95*n)-1]", split=a["split"], corpus_status=a["corpus_status"],
                independent_holdout_eligible=holdout, production_certified=False, original_fp32_gate="NOT_EVALUATED",
                playing_strength="NOT_MEASURED", performance="NOT_MEASURED",
                independence_note="Game-level split is checked against the bound corpus manifest; this tool does not prove that a human or selector never inspected holdout outputs.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-requests", type=int, default=100000)
    args = parser.parse_args()
    args.reference, args.candidate, args.output = (p.resolve() for p in (args.reference, args.candidate, args.output))
    if args.output.exists() or args.max_requests < 1:
        parser.error("output must be a new directory and max-requests positive")
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(schema="rustgo-quantization-corpus-comparison-v1", status="DIAGNOSTIC_FAILED",
                  reference=str(args.reference), candidate=str(args.candidate), production_certified=False,
                  original_fp32_gate="NOT_EVALUATED", playing_strength="NOT_MEASURED")
    protocol = None
    try:
        from worker_protocol_tools import Protocol
        protocol = Protocol(ROOT / "crates/kata_worker/proto/worker.proto")
        reference = load_collection(args.reference, protocol, max_requests=args.max_requests)
        candidate = load_collection(args.candidate, protocol, full_reference=reference["recipe"], max_requests=args.max_requests)
        result.update(compare(reference, candidate))
        result["identities"] = {name: dict(report_sha256=data["report_sha256"],
                                           model_sha256=data["report"]["model_sha256"],
                                           requests_sha256=data["report"]["requests_sha256"],
                                           binary_sha256=data["report"]["binary_sha256"],
                                           actual_profile=data["report"].get("actual_profile"),
                                           recipe_inspection=data["report"].get("recipe_inspection"),
                                           batch=data["report"]["batch"], capacity=data["report"]["capacity"],
                                           request_window=data["report"]["request_window"], batching=data["report"]["batching"])
                                for name, data in (("reference", reference), ("candidate", candidate))}
        cases = result.pop("cases")
        save_json(args.output / "cases.json", cases)
        result["cases_artifact"] = dict(file="cases.json", sha256=sha256((args.output / "cases.json").read_bytes()))
        print(f"{result['status']}: n={result['positions']} win max/mean/P95="
              f"{result['win_probability_abs_error']['max']:.6g}/{result['win_probability_abs_error']['mean']:.6g}/"
              f"{result['win_probability_abs_error']['p95']:.6g}; exact={result['exact_output_bytes']['matches']}/{result['positions']}")
        return 0 if result["experimental_gate"]["result"] == "PASS" else 1
    except Exception as error:
        result.update(status="DIAGNOSTIC_FAILED", collection_validation="FAIL", error=str(error))
        print(f"DIAGNOSTIC_FAILED: {error}")
        return 1
    finally:
        if protocol is not None:
            protocol.close()
        save_json(args.output / "report.json", result)


if __name__ == "__main__":
    raise SystemExit(main())
