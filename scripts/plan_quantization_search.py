#!/usr/bin/env python3
"""Create CPU-only, bounded research seeds; never run or certify a candidate.

Requires quant-inspect's complete manifest/FP16 template, original model bytes,
and a corpus directory from sgf_quantization_corpus. Original SGF hashes and all
corpus input artifacts are checked. No holdout outputs, model outputs, GPU APIs,
subprocesses or downloads are used. Incomplete corpora produce BLOCKED_CORPUS.

Recipe hashes reproduce Rust PrecisionRecipe's canonical struct field order and
sorted, complete projection list. Python cannot reconstruct the graph digest
from the projection-only manifest: the Rust loader must still check it against
the parsed source model before any later inference.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

# These helpers only validate JSON and semantic prefixes. Importing this module
# does not load protobuf/gRPC or invoke its collection entry point.
import collect_quantization_corpus as corpus_io


SCHEMA = "rustgo-quantization-search-plan"
VERSION = 1
DEFAULT_SEED = "rustgo-quant-search-v1"
SPLITS = ("calibration", "selection", "holdout")
SUFFIXES = ("attention.qkv", "attention.out", "ffn.dual", "ffn.down")
RECIPE_KEYS = {"schema", "version", "quantization_semantics_version", "model_sha256", "graph_sha256", "projections"}
CORPUS_KEYS = {"artifacts", "duplicate_records", "leakage_policy", "legacy_counts", "legacy_fixture_scope",
               "options", "production_certified", "readiness", "rejected_sources", "request_format",
               "request_set_sha256", "rule_scope", "sampling", "schema", "source_policy", "split_policy",
               "status", "unique_games"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def object_fields(value, fields, label):
    require(type(value) is dict and set(value) == set(fields), f"{label}: missing or unknown fields")


def positive_integer(value, label, maximum=1_000_000):
    require(type(value) is int and 0 < value <= maximum, f"{label}: invalid positive integer")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compact(value, *, sorted_keys=False):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      sort_keys=sorted_keys, allow_nan=False).encode("utf-8")


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON field {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite JSON constant {value}")

    def finite_float(value):
        result = float(value)
        require(math.isfinite(result), f"nonfinite JSON number {value}")
        return result

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid, parse_float=finite_float)


def validate_model_inputs(manifest, template, model_sha):
    object_fields(manifest, {"schema", "model_sha256", "graph_sha256", "projections"}, "model manifest")
    object_fields(template, RECIPE_KEYS, "FP16 template")
    require(manifest["schema"] == "rustgo-layer-graph-v1", "unsupported model manifest schema")
    require(template["schema"] == "rustgo-precision-recipe" and type(template["version"]) is int
            and template["version"] == 1 and type(template["quantization_semantics_version"]) is int
            and template["quantization_semantics_version"] == 1, "unsupported precision recipe version")
    for obj in (manifest, template):
        for key in ("model_sha256", "graph_sha256"):
            corpus_io.check_sha(obj[key], key)
    require(manifest["model_sha256"] == template["model_sha256"] == model_sha, "source model SHA256 mismatch")
    require(manifest["graph_sha256"] == template["graph_sha256"], "manifest/template graph SHA256 mismatch")
    require(type(manifest["projections"]) is list and manifest["projections"], "empty/malformed projections")
    projections, groups = {}, {}
    for projection in manifest["projections"]:
        object_fields(projection, {"id", "layer_index", "n", "k"}, "manifest projection")
        name = projection["id"]
        require(type(name) is str and name not in projections, "duplicate/malformed projection ID")
        match = re.fullmatch(r"trunk\.block([0-9]{2,})\.pair([0-9]{2,})\.(attention\.(?:qkv|out)|ffn\.(?:dual|down))", name)
        require(match is not None, f"unsupported projection ID {name}")
        block, pair, kind = int(match[1]), int(match[2]), match[3]
        require(name == f"trunk.block{block:02}.pair{pair:02}.{kind}", "noncanonical structural ID")
        for key in ("layer_index", "n", "k"):
            positive_integer(projection[key], f"{name}.{key}")
        projections[name] = projection
        groups.setdefault((block, pair), {})[kind] = projection
    blocks = sorted({block for block, _ in groups})
    require(blocks == list(range(len(blocks))), "block IDs must be contiguous from zero")
    next_attention, common_mid = 3, None
    for block in blocks:
        pairs = sorted(pair for b, pair in groups if b == block)
        require(pairs == list(range(len(pairs))), "pair IDs must be contiguous from zero")
        for pair in pairs:
            group = groups[block, pair]
            require(set(group) == set(SUFFIXES), "every pair requires all four projections")
            qkv, out, dual, down = (group[kind] for kind in SUFFIXES)
            mid, hidden = out["n"], dual["n"]
            require(mid % 32 == 0 and (common_mid is None or common_mid == mid), "inconsistent/unsupported trunk mid width")
            common_mid = mid
            require(hidden % 8 == 0 and hidden <= 3 * mid, "unsupported FFN hidden width")
            require((qkv["n"], qkv["k"], out["k"], dual["k"], down["n"], down["k"])
                    == (3 * mid, mid, mid, mid, mid, hidden), "inconsistent projection shapes")
            require(qkv["layer_index"] == out["layer_index"] == next_attention
                    and dual["layer_index"] == down["layer_index"] == next_attention + 2,
                    "projection layer indices differ from the canonical nested graph structure")
            next_attention += 4
        next_attention += 4  # block gates/up/down/RMS before the next attention
    require(type(template["projections"]) is list, "malformed template projections")
    seen = set()
    for entry in template["projections"]:
        object_fields(entry, {"id", "expected_n", "expected_k", "precision"}, "template projection")
        name = entry["id"]
        require(type(name) is str and name in projections and name not in seen, "unknown/duplicate template projection")
        seen.add(name)
        require(entry["precision"] == "fp16", "input must be a complete all-FP16 template")
        for key in ("expected_n", "expected_k"):
            positive_integer(entry[key], key)
        require((entry["expected_n"], entry["expected_k"]) == (projections[name]["n"], projections[name]["k"]),
                f"template shape mismatch for {name}")
    require(seen == set(projections), "incomplete FP16 template: implicit defaults are forbidden")
    return projections


def canonical_recipe(template, precisions=None):
    precisions = precisions or {}
    # Match Rust's Serialize field order, not alphabetically sorted JSON keys.
    return dict(schema=template["schema"], version=template["version"],
                quantization_semantics_version=template["quantization_semantics_version"],
                model_sha256=template["model_sha256"], graph_sha256=template["graph_sha256"],
                projections=[dict(id=p["id"], expected_n=p["expected_n"], expected_k=p["expected_k"],
                                  precision=precisions.get(p["id"], "fp16"))
                             for p in sorted(template["projections"], key=lambda p: p["id"])])


def seed_candidates(template, projections):
    result, by_hash = [], {}
    for label, threshold in (("fp16", None), ("all-ffn-int8", 0), ("wide-ffn-384-int8", 384)):
        choices = {}
        if threshold is not None:
            for name, projection in projections.items():
                if name.endswith(".ffn.dual") and projection["n"] >= threshold:
                    choices[name] = choices[name.removesuffix("dual") + "down"] = "int8"
        recipe = canonical_recipe(template, choices)
        recipe_sha = digest(compact(recipe))
        if recipe_sha in by_hash:
            by_hash[recipe_sha]["aliases"].append(label)
            continue
        candidate = dict(name=label, aliases=[label], recipe_sha256=recipe_sha,
                         recipe=recipe, int8_projection_count=len(choices),
                         int8_ffn_group_count=len(choices) // 2)
        result.append(candidate)
        by_hash[recipe_sha] = candidate
    return result


def local_path(root, name):
    require(type(name) is str and name and "\\" not in name and not Path(name).is_absolute(), "invalid source relative path")
    require(all(part not in ("", ".", "..") for part in name.split("/")) and ":" not in name, "unsafe source relative path")
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), f"missing/nonlocal source {name}")
    return path


def validate_corpus(path, sgf_root=None):
    raw = path.read_bytes()
    manifest = strict_json(raw)
    object_fields(manifest, CORPUS_KEYS, "corpus manifest")
    require(manifest["schema"] == corpus_io.SCHEMA, "unsupported corpus schema")
    require(manifest["status"] in ("READY", "NOT_READY") and manifest["production_certified"] is False,
            "invalid corpus readiness/certification state")
    object_fields(manifest["options"], {"input", "output", "seed", "games_per_split", "positions_per_game", "assume_rules", "assume_komi"}, "corpus options")
    require(type(manifest["options"]["seed"]) is str and manifest["options"]["seed"], "missing corpus split seed")
    quotas = manifest["options"]["games_per_split"]
    require(type(quotas) is list and len(quotas) == 3, "invalid split quotas")
    for count in quotas:
        positive_integer(count, "split quota")
    artifacts, blobs = {}, {}
    expected = {"games.jsonl", "rejections.jsonl", "duplicates.jsonl"} | {
        f"{split}.{suffix}" for split in SPLITS for suffix in ("requests.jsonl", "worker-fixture.json")}
    require(type(manifest["artifacts"]) is list, "invalid corpus artifact table")
    for artifact in manifest["artifacts"]:
        object_fields(artifact, {"file", "bytes", "sha256"}, "corpus artifact")
        name = artifact["file"]
        require(type(name) is str and name in expected and name not in artifacts, "duplicate/unknown corpus input artifact")
        corpus_io.check_sha(artifact["sha256"], name)
        require(type(artifact["bytes"]) is int and artifact["bytes"] >= 0, "invalid corpus artifact length")
        blob = local_path(path.parent, name).read_bytes()
        require(len(blob) == artifact["bytes"] and digest(blob) == artifact["sha256"], f"corpus source hash/size mismatch: {name}")
        artifacts[name], blobs[name] = artifact, blob
    require(set(artifacts) == expected, "incomplete corpus input artifact table")
    # Also reject duplicate JSON fields in auxiliary input evidence.
    for name, blob in blobs.items():
        if name.endswith(".worker-fixture.json"):
            strict_json(blob)
    rejections = list(corpus_io.read_jsonl(blobs["rejections.jsonl"], "rejections"))
    duplicates = list(corpus_io.read_jsonl(blobs["duplicates.jsonl"], "duplicates"))
    require(type(manifest["rejected_sources"]) is int and manifest["rejected_sources"] == len(rejections)
            and type(manifest["duplicate_records"]) is int and manifest["duplicate_records"] == len(duplicates), "corpus evidence counts mismatch")
    root = (sgf_root or Path(manifest["options"]["input"])).resolve()
    require(root.is_dir(), "SGF source directory missing; use --sgf-root for relocated sources")
    games, source_hashes = {}, {}
    for game in corpus_io.read_jsonl(blobs["games.jsonl"], "games"):
        corpus_io.check_sha(game.get("game_id"), "game_id")
        require(game["game_id"] not in games and type(game.get("sources")) is list and game["sources"], "duplicate game or missing source provenance")
        require(game.get("split") in (*SPLITS, None), "invalid game split")
        plies = game.get("retained_plies")
        require(type(plies) is list and all(type(p) is int and 8 <= p <= 10000 for p in plies)
                and len(set(plies)) == len(plies), "invalid/duplicate retained plies")
        for source in game["sources"]:
            object_fields(source, {"path", "sha256"}, "SGF source")
            corpus_io.check_sha(source["sha256"], "source SGF SHA256")
            source_path = local_path(root, source["path"])
            require(file_digest(source_path) == source["sha256"], f"source SGF hash mismatch: {source['path']}")
            require(source["path"] not in source_hashes, "duplicate SGF source assigned to multiple games")
            source_hashes[source["path"]] = source["sha256"]
        games[game["game_id"]] = game
    require(type(manifest["unique_games"]) is int and len(games) == manifest["unique_games"], "unique game count mismatch")
    records, counts = [], []
    seen = {key: set() for key in ("name", "semantic_position_sha256", "board_state_sha256")}
    for split in SPLITS:
        rows = list(corpus_io.read_jsonl(blobs[f"{split}.requests.jsonl"], split))
        for record in rows:
            require(record.get("game_id") in games, "request source game missing")
            corpus_io.validate_record(record, games[record["game_id"]])
            require(record["split"] == split, "request file/split mismatch")
            for key in seen:
                require(record[key] not in seen[key], f"duplicate/cross-split leakage: {key}")
                seen[key].add(record[key])
        expected_rows = {(g["game_id"], ply) for g in games.values() if g["split"] == split for ply in g["retained_plies"]}
        require({(r["game_id"], r["ply"]) for r in rows} == expected_rows and len(rows) == len(expected_rows), "incomplete full split requests")
        records.append(rows)
        counts.append(dict(split=split, assigned_games=sum(g["split"] == split for g in games.values()),
                           games_with_retained_positions=len({r["game_id"] for r in rows}), distinct_semantic_positions=len(rows)))
    corpus_io.check_sha(manifest["request_set_sha256"], "request_set_sha256")
    require(digest(compact(records)) == manifest["request_set_sha256"], "full corpus request-set hash mismatch")
    readiness = manifest["readiness"]
    require(type(readiness) is dict and readiness.get("status") == manifest["status"]
            and readiness.get("symmetry_multiplicity_counted") is False
            and type(readiness.get("holdout_minimum_games")) is int and readiness["holdout_minimum_games"] == 128
            and type(readiness.get("holdout_minimum_distinct_positions")) is int and readiness["holdout_minimum_distinct_positions"] == 4096,
            "unsupported corpus readiness contract")
    reported = readiness.get("splits")
    require(type(reported) is list and len(reported) == 3, "incomplete readiness splits")
    for actual, report, quota in zip(counts, reported, quotas):
        require(all(report.get(k) == v and type(report.get(k)) is type(v) for k, v in actual.items())
                and report.get("requested_games") == quota, "readiness count differs from verified requests")
    ready = all(c["games_with_retained_positions"] >= q for c, q in zip(counts, quotas))
    ready &= counts[2]["games_with_retained_positions"] >= 128 and counts[2]["distinct_semantic_positions"] >= 4096
    require(manifest["status"] == ("READY" if ready else "NOT_READY"), "corpus readiness status contradicts verified counts")
    return manifest, raw, blobs, records, counts, source_hashes


def write_new(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
    return dict(file=path.name, bytes=len(raw), sha256=digest(raw))


def pretty(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def run(args):
    require(not args.output.exists(), "output directory already exists")
    require(type(args.seed) is str and 0 < len(args.seed) <= 256, "seed must contain 1..256 characters")
    require(1 <= args.max_candidates <= 8 and 1 <= args.max_finalists <= min(2, args.max_candidates), "budget requires <=8 candidates and <=2 finalists")
    manifest_raw, template_raw = args.model_manifest.read_bytes(), args.fp16_template.read_bytes()
    manifest, template = strict_json(manifest_raw), strict_json(template_raw)
    model_sha = file_digest(args.model)
    projections = validate_model_inputs(manifest, template, model_sha)
    candidates = seed_candidates(template, projections)
    require(len(candidates) <= args.max_candidates, "candidate budget is smaller than the unique initial seeds")
    corpus, corpus_raw, blobs, records, counts, sgf_hashes = validate_corpus(args.corpus_manifest.resolve(), args.sgf_root)
    # FP16 remains the control; deterministic order never ranks predicted speed.
    candidates[1:] = sorted(candidates[1:], key=lambda c: digest(compact([SCHEMA, VERSION, args.seed, c["recipe_sha256"]])))
    source_hashes = dict(model=model_sha, model_manifest=digest(manifest_raw), fp16_template=digest(template_raw),
                         corpus_manifest=digest(corpus_raw), games=digest(blobs["games.jsonl"]),
                         planner=file_digest(Path(__file__)), corpus_validation_helper=file_digest(Path(corpus_io.__file__)),
                         json_validation_helper=file_digest(Path(corpus_io.strict_json.__code__.co_filename)))
    contract = dict(schema=SCHEMA, version=VERSION, seed=args.seed, corpus_split_seed=corpus["options"]["seed"],
                    source_sha256=source_hashes, graph_sha256=manifest["graph_sha256"], source_sgf_sha256=sgf_hashes,
                    calibration_requests_sha256=digest(blobs["calibration.requests.jsonl"]),
                    selection_requests_sha256=digest(blobs["selection.requests.jsonl"]),
                    holdout_inputs_sha256=digest(blobs["holdout.requests.jsonl"]),
                    request_set_sha256=corpus["request_set_sha256"],
                    candidate_recipe_sha256=[c["recipe_sha256"] for c in candidates],
                    max_candidates=args.max_candidates, max_finalists=args.max_finalists)
    search_id = digest(compact(contract, sorted_keys=True))
    # Reserve only after all CPU validation; never replace an existing result.
    args.output.mkdir()
    artifacts = []
    for name, raw in (("model-manifest.json", manifest_raw), ("fp16-template.json", template_raw),
                      ("corpus-manifest.json", corpus_raw), ("games.jsonl", blobs["games.jsonl"]),
                      ("calibration.requests.jsonl", blobs["calibration.requests.jsonl"]),
                      ("selection.requests.jsonl", blobs["selection.requests.jsonl"])):
        artifacts.append(write_new(args.output / name, raw))
    candidate_rows = []
    for candidate in candidates:
        recipe = candidate.pop("recipe")
        name = candidate["name"] + ".recipe.json"
        artifact = write_new(args.output / name, pretty(recipe))
        artifacts.append(artifact)
        candidate_rows.append(dict(**candidate, recipe_file=name, recipe_file_sha256=artifact["sha256"],
                                   status="RESEARCH_SEED_UNEVALUATED", calibration="NOT_RUN", selection="NOT_RUN",
                                   performance="NOT_MEASURED", holdout="NOT_OPENED", selected=False))
    ledger = dict(schema=SCHEMA, version=VERSION, search_id=search_id,
                  status="BLOCKED_CORPUS" if corpus["status"] != "READY" else "PLANNED_UNVERIFIED",
                  publish_allowed=False, production_certified=False, identity=contract,
                  source_paths=dict(model=str(args.model.resolve()), model_manifest=str(args.model_manifest.resolve()),
                                    fp16_template=str(args.fp16_template.resolve()), corpus_manifest=str(args.corpus_manifest.resolve())),
                  corpus_status=corpus["status"], corpus_counts=counts, corpus_readiness=corpus["readiness"],
                  selection=dict(policy="complete frozen selection split; no subsampling", requests_file="selection.requests.jsonl",
                                 requests_sha256=contract["selection_requests_sha256"], request_count=len(records[1]),
                                 semantic_position_sha256=[r["semantic_position_sha256"] for r in records[1]]),
                  budget=dict(max_candidates=args.max_candidates, initial_unique_candidates=len(candidates),
                              remaining_candidate_slots=args.max_candidates-len(candidates), max_finalists=args.max_finalists,
                              finalists=[], max_holdout_candidates=1, holdout_outputs_read=0),
                  candidates=candidate_rows,
                  future_profile_groups=dict(status="UNSELECTED", candidates_generated=0,
                                             requires="measured complete RMSNorm/FFN group cost plus selection evidence; no MAC-based speed claims"),
                  validation_scope="source hashes, complete projection schema, structural IDs/shapes and full corpus inputs verified; graph digest is linked to quant-inspect, not recomputed in Python; Rust recipe loader must revalidate before inference",
                  calibration_scope="existing dynamic absmax INT8 only; no fitted scales, activations, model outputs or labels consumed",
                  publication_requirements=["qualified independent corpus", "selection accuracy and complete-cost measurements",
                                            "frozen final recipe and execution identity", "one unopened holdout evaluation",
                                            "fresh-process reload verification; no READY marker is emitted by this planner"],
                  artifacts=artifacts)
    write_new(args.output / "search-plan.json", pretty(ledger))
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Original source model; hashed only")
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--fp16-template", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--sgf-root", type=Path, help="Relocated original SGFs; defaults to corpus options.input")
    parser.add_argument("--output", type=Path, required=True, help="NEW directory with an existing parent")
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--max-finalists", type=int, default=2)
    args = parser.parse_args()
    try:
        ledger = run(args)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"plan_quantization_search: {error}", file=sys.stderr)
        return 1
    print(json.dumps(dict(status=ledger["status"], search_id=ledger["search_id"],
                          candidates=len(ledger["candidates"]), publish_allowed=False), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
