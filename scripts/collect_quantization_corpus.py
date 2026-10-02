#!/usr/bin/env python3
"""Collect complete SGF corpus requests with one isolated real local NN Worker.

Reads the full *.requests.jsonl exported by sgf_quantization_corpus, verifies its
manifest and game provenance, and preserves Position/EvalParameters verbatim.
This never parses SGFs again or expands symmetries. Duplicate/missing records,
changed artifact hashes, protocol identity mismatches and nonfinite outputs fail.

Writes streaming outputs.jsonl and exact protobuf results for reuse by a precision
selector. Outputs are postprocessed white-perspective NNOutput values, including
policy and ownership, not raw logits or activation calibration statistics. No
FP32 accuracy certificate or playing-strength claim is produced. The only server
is a fresh temporary 127.0.0.1 gRPC harness; only this script's child is drained.
Optional continuous refill buffers complete result protobufs under a payload
limit and defers validation/serialization/disk IO until Drain; chunked is default.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from validate_quantized_backend import PROFILE_LINE, config_path, discover_profile, parse_batch_trace, run_logged, save_json, strict_json


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rustgo-quantization-corpus-v1"
RECORD_KEYS = {"name", "game_id", "split", "ply", "phase", "semantic_position_sha256",
               "board_state_sha256", "position", "parameters"}
POSITION_KEYS = {"board_size", "rules", "komi", "initial_player", "next_player", "initial_stones", "moves"}
PARAMETER_BOOLS = {"include_ownership", "skip_cache", "conservative_pass", "enable_passing_hacks",
                   "always_compute_pass_alive", "exclude_territory_adjacent_to_atari", "avoid_mytdagger_hack",
                   "allow_terminal_search_history", "force_non_terminal"}
PARAMETER_RANGES = {"policy_temperature": (0.001, 100.0), "policy_optimism": (0.0, 1.0),
                    "draw_equivalent_wins_for_white": (0.0, 1.0), "playout_doubling_advantage": (-100.0, 100.0)}
# Deliberately a small Worker-only grammar. Unknown keys could introduce another
# model, include, runtime file or indexed override that this collector cannot bind.
IMPORTED_CONFIG_KEYS = {
    "rules", "komi", "nnBackend", "nnMaxBatchSize", "numNNServerThreadsPerModel",
    "nnCacheSizePowerOfTwo", "nnMutexPoolSizePowerOfTwo", "numSearchThreads", "maxVisits",
    "ponderingEnabled", "cudaTacticPlan", "cudaInt8Scope", "cudaInt8MinFfnWidth",
    "deviceToUse", "gpuToUse", "cudabackendDeviceToUse", "cudabackendGpuToUse",
    "cudaint8backendDeviceToUse", "cudaint8backendGpuToUse", "useFP16", "useNHWC",
    "cudabackendUseFP16", "cudaint8backendUseFP16", "nnRandomize", "nnRandSeed",
    "nnForcedSymmetry", "cudaDisableWarmup",
}

RECIPE_KEYS = ("schema", "version", "quantization_semantics_version", "model_sha256", "graph_sha256", "projections")
PROJECTION_KEYS = ("id", "expected_n", "expected_k", "precision")
IDENTITY_FILES = ("model-manifest.json", "fp16-recipe.json", "source-recipe.json", "recipe-identity.json")
IDENTITY_SCOPE = ("CPU export from the recorded binary and original model; independently reconstructed ordered envelope "
                  "and checked source digest layout/shape, then matched actual Worker recipe identity. "
                  "Python does not independently parse or rehash original FP32 weight tensors.")
# Field order is the Rust Mxfp8Semantics struct order, not alphabetic JSON order.
# Future numerical semantics must use a new version instead of changing these.
MXFP8_SEMANTICS = dict(
    schema="rustgo-mxfp8-semantics-v1", revision=1,
    quantization_version="mxfp8-e4m3-rne-ue8m0-ceil-fp32-block32-half-boundary-v1",
    value_format="e4m3fn-rne-satfinite", block_size=32,
    weight_source="lowered-model-f32-tensor-before-any-half-conversion",
    activation_source="existing-f16-storage-boundary", scale_format="ue8m0",
    scale_rounding="fp32-rne(absmax/448)-then-positive-infinity-satfinite",
    value_rounding="fp32-rne(x*pow2(127-scale-byte))-then-e4m3-rne-satfinite",
    zero_block="q=positive-zero;scale=one;nonzero-block-preserves-negative-zero",
    padding="K-to32-q=positive-zero;scale-outer-to128-inner-to4-unused-bytes=0",
    scale_layout="cublaslt-128x4-swizzled-v1",
    gemm_layout="TN:row-major-X-times-row-major-W-transpose;alpha-f32-one",
    compute="fp32-accumulation", fast_accum=False,
    outputs="ffn-dual,attention-qkv=f16-beta0;ffn-down,attention-out=f32-beta1",
    algorithm_selection="first-of32-heuristics-with-AlgorithmCheck-TensorCore-E4M3-FP32-no-timing-v1",
    invalid_output="sticky-nan-inf-nonfinite-output;discard-entire-forward",
)


def ordered_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def same_json(left, right):
    # Unlike Python equality, distinguish booleans, integers and floating values.
    return (json.dumps(left, sort_keys=True, ensure_ascii=False, allow_nan=False)
            == json.dumps(right, sort_keys=True, ensure_ascii=False, allow_nan=False))


def projection_map(recipe, model_sha, graph_sha, versions, allowed):
    if (not isinstance(recipe, dict) or set(recipe) != set(RECIPE_KEYS)
            or recipe["schema"] != "rustgo-precision-recipe"
            or any(type(recipe[k]) is not int for k in ("version", "quantization_semantics_version"))
            or (recipe["version"], recipe["quantization_semantics_version"]) != versions
            or recipe["model_sha256"] != model_sha or recipe["graph_sha256"] != graph_sha
            or not isinstance(recipe["projections"], list)):
        raise ValueError("unsupported or mismatched exported precision recipe")
    result = {}
    for p in recipe["projections"]:
        if (not isinstance(p, dict) or set(p) != set(PROJECTION_KEYS) or not isinstance(p["id"], str)
                or p["id"] in result
                or re.fullmatch(r"trunk\.block\d+\.pair\d+\.(?:attention\.(?:qkv|out)|ffn\.(?:dual|down))", p["id"]) is None
                or any(type(p[k]) is not int or not 0 < p[k] <= (1 << 63) - 1 for k in ("expected_n", "expected_k"))
                or not isinstance(p["precision"], str) or p["precision"] not in allowed):
            raise ValueError("invalid or duplicate exported projection")
        result[p["id"]] = {key: p[key] for key in PROJECTION_KEYS}
    return result


def verify_mxfp8_identity(raw_files, source_raw, model_sha, *, full_reference=None, actual_profile=None):
    """Reconstruct v2 hashing independently; source tensor digests come from Rust.

    This verifies their exact membership/layout/model binding, not their tensor
    preimages. The actual loader independently hashes those original tensors;
    its recipe hash must match the reconstructed full envelope before acceptance.
    """
    if set(raw_files) != set(IDENTITY_FILES) or raw_files["source-recipe.json"] != source_raw:
        raise ValueError("incomplete identity export or changed original recipe bytes")
    source, manifest, fp16, identity = [strict_json(raw_files[key]) for key in
        ("source-recipe.json", "model-manifest.json", "fp16-recipe.json", "recipe-identity.json")]
    wrapper_keys = {"schema", "version", "model_sha256", "graph_sha256", "source_recipe_file",
                    "source_recipe_file_sha256", "recipe_sha256", "canonical_encoding", "canonical_envelope"}
    if (not isinstance(identity, dict) or set(identity) != wrapper_keys
            or identity["schema"] != "rustgo-recipe-identity-export" or type(identity["version"]) is not int
            or identity["version"] != 1 or identity["model_sha256"] != model_sha
            or identity["source_recipe_file"] != "source-recipe.json"
            or identity["source_recipe_file_sha256"] != sha256(source_raw)
            or identity["canonical_encoding"] != "serde-json-compact-ordered-v1"):
        raise ValueError("identity export wrapper/source model or recipe binding mismatch")
    check_sha(model_sha, "identity model SHA")
    graph_sha = identity["graph_sha256"]
    check_sha(graph_sha, "identity graph SHA")
    check_sha(identity["recipe_sha256"], "identity recipe SHA")
    if (not isinstance(manifest, dict) or set(manifest) != {"schema", "model_sha256", "graph_sha256", "projections"}
            or manifest["schema"] != "rustgo-layer-graph-v1" or manifest["model_sha256"] != model_sha
            or manifest["graph_sha256"] != graph_sha or not isinstance(manifest["projections"], list)):
        raise ValueError("model manifest identity/schema mismatch")
    shapes = {}
    for p in manifest["projections"]:
        if (not isinstance(p, dict) or set(p) != {"id", "layer_index", "n", "k"}
                or not isinstance(p["id"], str) or p["id"] in shapes
                or type(p["layer_index"]) is not int or p["layer_index"] < 0
                or any(type(p[k]) is not int or p[k] <= 0 for k in ("n", "k"))):
            raise ValueError("invalid or duplicate model manifest projection")
        shapes[p["id"]] = (p["n"], p["k"])
    base = projection_map(fp16, model_sha, graph_sha, (1, 1), {"fp16"})
    if not base or {key: (p["expected_n"], p["expected_k"]) for key, p in base.items()} != shapes:
        raise ValueError("FP16 template must cover the complete model manifest exactly")
    if full_reference is not None:
        reference = projection_map(full_reference, model_sha, graph_sha, (1, 1), {"fp16"})
        if reference != base:
            raise ValueError("exported full FP16 manifest differs from the independently verified FP16 reference")
    selected = projection_map(source, model_sha, graph_sha, (2, 2), {"fp16", "int8", "mxfp8"})
    if not any(p["precision"] == "mxfp8" for p in selected.values()):
        raise ValueError("v2 requires an explicitly selected MXFP8 projection")
    resolved = {key: dict(p) for key, p in base.items()}
    for key, p in selected.items():
        if key not in shapes or (p["expected_n"], p["expected_k"]) != shapes[key]:
            raise ValueError("source recipe projection ID/shape differs from model manifest")
        resolved[key] = p
    canonical = {key: source[key] for key in RECIPE_KEYS[:-1]}
    canonical["projections"] = [resolved[key] for key in sorted(resolved)]
    envelope = identity["canonical_envelope"]
    if (not isinstance(envelope, dict)
            or set(envelope) != {"schema", "recipe", "mxfp8_semantics", "source_f32_projections"}
            or envelope["schema"] != "rustgo-resolved-precision-recipe-v2"
            or not same_json(envelope["recipe"], canonical)
            or not same_json(envelope["mxfp8_semantics"], MXFP8_SEMANTICS)
            or not isinstance(envelope["source_f32_projections"], list)):
        raise ValueError("v2 canonical envelope, default FP16 completion or fixed MXFP8 semantics mismatch")
    selected_mx = [p for p in canonical["projections"] if p["precision"] == "mxfp8"]
    sources = envelope["source_f32_projections"]
    if len(sources) != len(selected_mx):
        raise ValueError("MXFP8 source digest coverage mismatch")
    ordered_sources = []
    for entry, p in zip(sources, selected_mx, strict=True):
        dual = p["id"].endswith(".ffn.dual")
        expected = dict(id=p["id"], gemm_n=p["expected_n"] * (2 if dual else 1),
                        logical_k=p["expected_k"], source_stride=p["expected_k"],
                        packing="row-major-gate-then-up-v1" if dual else "row-major-v1")
        if (not isinstance(entry, dict) or set(entry) != {*expected, "source_f32_sha256"}
                or not same_json({key: entry[key] for key in expected}, expected)):
            raise ValueError("MXFP8 source ID/order/shape/packing mismatch")
        check_sha(entry["source_f32_sha256"], "original F32 projection digest")
        expected["source_f32_sha256"] = entry["source_f32_sha256"]
        ordered_sources.append(expected)
    rebuilt = dict(schema="rustgo-resolved-precision-recipe-v2", recipe=canonical,
                   mxfp8_semantics=dict(MXFP8_SEMANTICS), source_f32_projections=ordered_sources)
    canonical_sha = sha256(ordered_json(rebuilt))
    if canonical_sha != identity["recipe_sha256"]:
        raise ValueError("independently reconstructed ordered v2 envelope SHA mismatch")
    if actual_profile is not None and any(actual_profile[key] != value for key, value in
        (("model_sha256", model_sha), ("graph_sha256", graph_sha), ("recipe_sha256", canonical_sha))):
        raise ValueError("CPU identity export differs from the actual loaded Worker recipe")
    return canonical, identity


def load_mxfp8_inspection(directory, report, source_raw, *, full_reference=None):
    inspection = report.get("recipe_inspection")
    if (not isinstance(inspection, dict) or inspection.get("schema") != "rustgo-collection-recipe-inspection-v1"
            or inspection.get("directory") != "recipe-inspection"
            or inspection.get("binary_sha256") != report["binary_sha256"]
            or inspection.get("model_sha256") != report["model_sha256"]
            or inspection.get("source_recipe_file_sha256") != sha256(source_raw)
            or inspection.get("validation_scope") != IDENTITY_SCOPE):
        raise ValueError("MXFP8 requires an artifact-bound CPU export from the same binary/model/recipe")
    raw_files = {}
    for item in inspection["artifacts"]:
        name = item["file"]
        if name not in IDENTITY_FILES or name in raw_files:
            raise ValueError("unexpected or duplicate identity export artifact")
        raw = (directory / "recipe-inspection" / name).read_bytes()
        check_sha(item["sha256"], "CPU export artifact")
        if (sha256(raw) != item["sha256"] or type(item["bytes"]) is not int or len(raw) != item["bytes"]):
            raise ValueError("identity export artifact hash/size mismatch")
        raw_files[name] = raw
    canonical, identity = verify_mxfp8_identity(raw_files, source_raw, report["model_sha256"],
        full_reference=full_reference, actual_profile=report.get("actual_profile"))
    if inspection.get("recipe_sha256") != identity["recipe_sha256"]:
        raise ValueError("inspection report disagrees with the checked canonical recipe")
    command = inspection.get("command")
    if (not isinstance(command, list) or len(command) != 8
            or command[:4] != [report["binary"], "quant-inspect", "--model", report["model"]]
            or command[4] != "--recipe" or command[6] != "--output"
            or Path(command[5]).name != "recipe.json" or Path(command[7]).name != "recipe-inspection"
            or Path(command[5]).parent != Path(command[7]).parent):
        raise ValueError("CPU inspection command does not bind the actual binary/model/frozen recipe")
    return canonical


def inspect_mxfp8_recipe(args, report, recipe_copy, environment):
    """Only v2 needs the new CLI; frozen v1 binaries remain reproducible."""
    from compare_worker_outputs import file_hash

    source_raw = recipe_copy.read_bytes()
    recipe = strict_json(source_raw)
    if not isinstance(recipe, dict) or not isinstance(recipe.get("projections"), list):
        raise ValueError("precision recipe must be an object containing projections")
    needs_export = (recipe.get("version") == 2 or recipe.get("quantization_semantics_version") == 2
                    or any(isinstance(p, dict) and p.get("precision") == "mxfp8" for p in recipe["projections"]))
    if not needs_export:
        return
    if file_hash(args.binary) != report["binary_sha256"] or file_hash(args.model) != report["model_sha256"]:
        raise ValueError("binary/model changed before CPU recipe inspection")
    inspection = args.output / "recipe-inspection"
    command = [str(args.binary), "quant-inspect", "--model", str(args.model),
               "--recipe", str(recipe_copy), "--output", str(inspection)]
    # No fallback on unsupported --recipe, invalid semantics or source mismatch.
    run_logged(command, args.output / "recipe-inspection-command", environment, args.startup_timeout)
    raw_files = {name: (inspection / name).read_bytes() for name in IDENTITY_FILES}
    _, identity = verify_mxfp8_identity(raw_files, source_raw, report["model_sha256"])
    report["recipe_inspection"] = dict(
        schema="rustgo-collection-recipe-inspection-v1", directory="recipe-inspection", command=command,
        binary_sha256=report["binary_sha256"], model_sha256=report["model_sha256"],
        source_recipe_file_sha256=sha256(source_raw), recipe_sha256=identity["recipe_sha256"],
        artifacts=[dict(file=name, sha256=sha256(raw), bytes=len(raw)) for name, raw in raw_files.items()],
        validation_scope=IDENTITY_SCOPE,
    )
    if (file_hash(args.binary) != report["binary_sha256"] or file_hash(args.model) != report["model_sha256"]
            or recipe_copy.read_bytes() != source_raw):
        raise ValueError("binary/model/frozen recipe changed during CPU recipe inspection")
    load_mxfp8_inspection(args.output, report, source_raw)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def check_sha(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label}: expected lowercase SHA256")


def json_line(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"


def read_jsonl(raw, label):
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{label}:{line_number}: blank JSONL row")
        try:
            record = strict_json(line)
            if not isinstance(record, dict):
                raise ValueError("record must be an object")
        except (ValueError, TypeError) as error:
            raise ValueError(f"{label}:{line_number}: {error}") from error
        yield record


def validate_record(record, game):
    if set(record) != RECORD_KEYS or set(record["position"]) != POSITION_KEYS:
        raise ValueError("corpus request fields do not match the supported complete request schema")
    for key in ("game_id", "semantic_position_sha256", "board_state_sha256"):
        check_sha(record[key], key)
    if not isinstance(record["name"], str) or not record["name"] or not isinstance(record["phase"], str):
        raise ValueError("request name/phase must be strings with a nonempty name")
    if record["split"] not in ("calibration", "selection", "holdout") or record["split"] != game["split"]:
        raise ValueError("request split differs from its source game")
    position, parameters = record["position"], record["parameters"]
    if position["board_size"] != 19 or position["rules"] != "chinese":
        raise ValueError("current Worker corpus requires 19x19 Chinese rules")
    komi = position["komi"]
    if (type(komi) not in (int, float) or not math.isfinite(komi) or not -150 <= komi <= 150
            or komi * 2 != round(komi * 2)):
        raise ValueError("komi must be a finite half-integer in [-150,150]")
    if type(record["ply"]) is not int or record["ply"] != len(position["moves"]) or not 0 <= record["ply"] <= 10000:
        raise ValueError("ply must equal the complete history length, at most 10000")
    if record["ply"] not in game["retained_plies"]:
        raise ValueError("request ply is not listed among its game's retained plies")
    if any(type(position[key]) is not int or position[key] not in (1, 2) for key in ("initial_player", "next_player")):
        raise ValueError("invalid initial/next player")
    occupied = set()
    for stone in position["initial_stones"]:
        if (set(stone) != {"color", "vertex"} or type(stone["color"]) is not int or stone["color"] not in (1, 2)
                or type(stone["vertex"]) is not int or not 0 <= stone["vertex"] < 361 or stone["vertex"] in occupied):
            raise ValueError("invalid or duplicate initial stone")
        occupied.add(stone["vertex"])
    player = position["initial_player"]
    for move in position["moves"]:
        if (set(move) != {"color", "vertex"} or type(move["color"]) is not int or move["color"] != player
                or type(move["vertex"]) is not int or not -1 <= move["vertex"] < 361):
            raise ValueError("invalid move or out-of-turn complete history")
        player = 3 - player
    if player != position["next_player"]:
        raise ValueError("next_player differs from complete replay order")
    semantics = game["semantics"]
    if (semantics["board_size"] != 19 or semantics["komi_half_points"] != komi * 2
            or semantics["initial_player"] != position["initial_player"]
            or semantics["initial_stones"] != [[s["color"], s["vertex"]] for s in position["initial_stones"]]
            or semantics["moves"][:record["ply"]] != [[m["color"], m["vertex"]] for m in position["moves"]]):
        raise ValueError("request position is not the recorded source game's exact history prefix")
    if set(parameters) != PARAMETER_BOOLS | set(PARAMETER_RANGES) | {"symmetry", "max_history"}:
        raise ValueError("request parameters are incomplete or contain unsupported fields")
    if any(type(parameters[key]) is not bool for key in PARAMETER_BOOLS):
        raise ValueError("request boolean parameters must be booleans")
    for key, (low, high) in PARAMETER_RANGES.items():
        value = parameters[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"invalid request parameter {key}")
    if (type(parameters["symmetry"]) is not int or not 0 <= parameters["symmetry"] <= 7
            or type(parameters["max_history"]) is not int or not 0 <= parameters["max_history"] <= 10000):
        raise ValueError("invalid symmetry or history limit")
    if not parameters["include_ownership"] or not parameters["skip_cache"]:
        raise ValueError("corpus collection requires explicit ownership and skipped cache; inputs are never silently changed")
    if parameters["always_compute_pass_alive"] or parameters["exclude_territory_adjacent_to_atari"]:
        raise ValueError("current Rust Worker supports only false/false history modes")


def load_corpus(request_path, manifest_path, max_requests):
    manifest_raw = manifest_path.read_bytes()
    manifest = strict_json(manifest_raw)
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"expected corpus schema {SCHEMA}")
    check_sha(manifest.get("request_set_sha256"), "corpus request_set_sha256")
    artifacts = {}
    for artifact in manifest["artifacts"]:
        name = artifact["file"]
        if not isinstance(name, str) or Path(name).name != name or name in artifacts:
            raise ValueError("duplicate or nonlocal corpus artifact name")
        check_sha(artifact["sha256"], f"artifact {name}")
        artifacts[name] = artifact
    if request_path.parent != manifest_path.parent or not request_path.name.endswith(".requests.jsonl"):
        raise ValueError("requests must be a complete *.requests.jsonl beside its corpus manifest")

    def verified(name):
        if name not in artifacts:
            raise ValueError(f"corpus manifest does not bind {name}")
        raw = (manifest_path.parent / name).read_bytes()
        if sha256(raw) != artifacts[name]["sha256"] or len(raw) != artifacts[name]["bytes"]:
            raise ValueError(f"corpus artifact hash/size changed: {name}")
        return raw

    requests_raw, games_raw = verified(request_path.name), verified("games.jsonl")
    games = {}
    for game in read_jsonl(games_raw, "games.jsonl"):
        check_sha(game["game_id"], "game_id")
        if game["game_id"] in games or not game["sources"]:
            raise ValueError("duplicate game identity or missing provenance")
        for source in game["sources"]:
            check_sha(source["sha256"], "source SGF SHA")
            if not isinstance(source["path"], str) or not source["path"]:
                raise ValueError("missing source SGF path")
        games[game["game_id"]] = game
    records, unique = [], {key: set() for key in ("name", "semantic_position_sha256", "board_state_sha256")}
    for record in read_jsonl(requests_raw, request_path.name):
        if len(records) >= max_requests:
            raise ValueError(f"corpus exceeds --max-requests={max_requests}; no subset is silently collected")
        game = games.get(record.get("game_id"))
        if game is None:
            raise ValueError("request refers to a missing source game")
        validate_record(record, game)
        for key, seen in unique.items():
            if record[key] in seen:
                raise ValueError(f"duplicate corpus {key}: {record[key]}")
            seen.add(record[key])
        records.append(record)
    if not records or len({r["split"] for r in records}) != 1:
        raise ValueError("request file must contain one nonempty split")
    split = records[0]["split"]
    if request_path.name != f"{split}.requests.jsonl":
        raise ValueError("request filename disagrees with its split")
    expected = {(game["game_id"], ply) for game in games.values() if game["split"] == split for ply in game["retained_plies"]}
    observed = {(record["game_id"], record["ply"]) for record in records}
    if observed != expected or len(observed) != len(records):
        raise ValueError("request file has duplicate/missing game-ply records compared with games.jsonl")
    return records, games, manifest, {"manifest.json": manifest_raw, "games.jsonl": games_raw,
                                     request_path.name: requests_raw}


def make_requests(protocol, records, model_sha, session, lease_ms):
    from google.protobuf.json_format import ParseDict
    requests, semantic_hashes = [], set()
    digest = hashlib.sha256()
    for index, record in enumerate(records, 1):
        position = ParseDict(record["position"], protocol.pb.Position(), ignore_unknown_fields=False)
        parameters = ParseDict(record["parameters"], protocol.pb.EvalParameters(), ignore_unknown_fields=False)
        # This hash excludes task/model/session to detect repeated wire semantics,
        # while the actual input_hash below binds the complete request envelope.
        semantic = hashlib.sha256()
        for message in (position, parameters):
            raw = message.SerializeToString(deterministic=True)
            semantic.update(len(raw).to_bytes(8, "little") + raw)
        if semantic.hexdigest() in semantic_hashes:
            raise ValueError("duplicate protobuf Position/EvalParameters despite different corpus identities")
        semantic_hashes.add(semantic.hexdigest())
        request = protocol.pb.EvalRequest(task_id=index, generation=1, session_id=session,
                                         model_sha256=model_sha, position=position, parameters=parameters, lease_ms=lease_ms)
        request.input_hash = hashlib.sha256(request.SerializeToString(deterministic=True)).digest()
        raw = request.SerializeToString(deterministic=True)
        digest.update(len(raw).to_bytes(8, "little") + raw)
        requests.append(request)
    return requests, digest.hexdigest()


def continuous_capture():
    return dict(results=[], failure_message=None, submitted=0, heartbeat_count=0,
                payload_bytes=0, max_pending=0, pending_task_ids=[])


def drive_continuous(peer, prepared, window, capacity, task_timeout, buffer_limit_bytes,
                     capture, *, clock=time.monotonic):
    """Pure peer driver: refill before result accounting; no serialization or IO.

    prepared contains (ServerMessage, five-field result identity). Results stay
    as protobuf objects until the caller has drained/stopped its own Worker.
    ByteSize after refill bounds total protobuf payload, not total process RSS.
    A rejected oversized final message can add at most one transport-bounded
    message to the retained evidence. No request outside prepared is ever sent.
    """
    if (type(window) is not int or type(capacity) is not int or not 1 <= window <= min(capacity, 256)
            or not 1 <= capacity <= 4096 or not prepared
            or not math.isfinite(task_timeout) or task_timeout <= 0
            or type(buffer_limit_bytes) is not int or buffer_limit_bytes <= 0):
        raise ValueError("invalid continuous window/capacity/timeout/buffer limit")
    identities = [entry[1][0] for entry in prepared]
    if len(set(identities)) != len(identities):
        raise ValueError("prepared corpus contains duplicate task IDs")
    if capture["results"] or capture["submitted"]:
        raise ValueError("continuous capture must be fresh")
    pending, next_index = {}, 0

    def dispatch(index):
        message, identity = prepared[index]
        pending[identity[0]] = (identity, clock())
        peer.outgoing.put(message)
        capture["submitted"] += 1
        capture["max_pending"] = max(capture["max_pending"], len(pending))

    try:
        while next_index < min(window, len(prepared)):
            dispatch(next_index)
            next_index += 1
        while pending:
            # A heartbeat or another result never renews a lost request's lease.
            deadline = min(sent for _, sent in pending.values()) + task_timeout
            remaining = deadline - clock()
            if remaining <= 0:
                raise TimeoutError(f"continuous request deadline expired; pending {sorted(pending)}")
            message = peer.receive(remaining)
            capture["failure_message"] = message
            if clock() > deadline:
                raise TimeoutError(f"continuous request deadline expired; pending {sorted(pending)}")
            if message.HasField("heartbeat"):
                capture["heartbeat_count"] += 1
                if message.heartbeat.failed_requests:
                    raise AssertionError("Worker heartbeat reported failed requests")
                capture["failure_message"] = None
                continue
            if not message.HasField("result"):
                raise AssertionError("unexpected Worker message during continuous collection")
            result = message.result
            active = pending.get(result.task_id)
            if active is None:
                raise AssertionError(f"duplicate/stale/unsubmitted result {result.task_id}")
            actual = (result.task_id, result.generation, result.session_id, result.input_hash, result.model_sha256)
            if actual != active[0]:
                raise AssertionError(f"continuous result identity mismatch: {result.task_id}")
            if result.error_code or not result.HasField("output"):
                raise AssertionError(f"evaluation failed: {result.task_id}: {result.error_code}: {result.error_message}")
            del pending[result.task_id]
            # Match the benchmark's refill boundary. Only small identity/error
            # checks precede it; detailed output validation and all disk IO wait.
            if next_index < len(prepared):
                dispatch(next_index)
                next_index += 1
            capture["results"].append(result)
            capture["payload_bytes"] += result.ByteSize()
            capture["failure_message"] = None
            if capture["payload_bytes"] > buffer_limit_bytes:
                raise ValueError("continuous protobuf result buffer exceeds --max-buffered-result-mib")
        if len(capture["results"]) != len(prepared) or capture["submitted"] != len(prepared):
            raise AssertionError("continuous driver did not collect the complete corpus")
        return capture
    finally:
        capture["pending_task_ids"] = sorted(pending)


def save_continuous_evidence(capture, directory, report):
    """Call only after Drain or termination; retain raw evidence even on failure."""
    artifacts = []
    for result in capture["results"]:
        raw = result.SerializeToString(deterministic=True)
        name = f"{result.task_id:06d}.pb"
        path = directory / "results" / name
        if path.exists():
            if path.read_bytes() != raw:
                raise ValueError("refuse to replace conflicting retained continuous result")
        else:
            with path.open("xb") as stream:
                stream.write(raw)
        artifacts.append(dict(task_id=result.task_id, file=f"results/{name}", sha256=sha256(raw), bytes=len(raw)))
    failure = capture["failure_message"]
    if failure is not None:
        raw = failure.SerializeToString(deterministic=True)
        path = directory / "continuous-failure-message.pb"
        if path.exists():
            if path.read_bytes() != raw:
                raise ValueError("refuse to replace conflicting continuous failure message")
        else:
            with path.open("xb") as stream:
                stream.write(raw)
        artifacts.append(dict(file=path.name, sha256=sha256(raw), bytes=len(raw)))
    summary = dict(schema="rustgo-continuous-corpus-buffer-v1", submitted=capture["submitted"],
                   received_basic_identity_checked=len(capture["results"]), heartbeat_count=capture["heartbeat_count"],
                   protobuf_payload_bytes=capture["payload_bytes"], max_pending=capture["max_pending"],
                   pending_task_ids=capture["pending_task_ids"], artifacts=artifacts,
                   validation="Raw evidence retention only; numerical validation and final collection status are separate")
    save_json(directory / "continuous-buffer.json", summary)
    report["continuous_buffer"] = {key: value for key, value in summary.items() if key != "artifacts"}


def validate_collection_window_mode(directory, report):
    """Validate recorded refill mode and its complete saved buffer evidence.

    This checks the collector's evidence consistency, not historical scheduler
    timing or equality of physical batches in separate processes.
    """
    mode = report.get("window_mode", "chunked")
    if mode not in ("chunked", "continuous"):
        raise ValueError("unsupported collection window_mode")
    if mode == "chunked":
        if any(key in report for key in ("collection_buffer_contract", "continuous_buffer", "continuous_numerically_validated_results")):
            raise ValueError("chunked collection cannot claim continuous buffer evidence")
        return mode
    count, window = report["expected_requests"], report["request_window"]
    contract = report.get("collection_buffer_contract", {})
    limit = contract.get("max_protobuf_payload_bytes")
    if (contract.get("schema") != "rustgo-continuous-corpus-buffer-contract-v1"
            or type(count) is not int or count <= 0 or type(window) is not int or not 1 <= window <= min(report["capacity"], 256)
            or type(contract.get("max_results")) is not int or contract["max_results"] != count or type(limit) is not int
            or not 1024 * 1024 <= limit <= 4096 * 1024 * 1024 or limit % (1024 * 1024)
            or type(report.get("continuous_numerically_validated_results")) is not int
            or report["continuous_numerically_validated_results"] != count):
        raise ValueError("continuous collection lacks a complete bounded-buffer contract/validation count")
    bound = [a for a in report.get("artifacts", []) if a.get("file") == "continuous-buffer.json"]
    raw = (directory / "continuous-buffer.json").read_bytes()
    if len(bound) != 1 or sha256(raw) != bound[0].get("sha256"):
        raise ValueError("continuous buffer evidence is not bound to the collection report")
    summary = strict_json(raw)
    if (summary.get("schema") != "rustgo-continuous-corpus-buffer-v1"
            or not same_json(report.get("continuous_buffer"), {k: v for k, v in summary.items() if k != "artifacts"})
            or any(type(summary.get(k)) is not int for k in ("submitted", "received_basic_identity_checked", "heartbeat_count", "protobuf_payload_bytes", "max_pending"))
            or summary["submitted"] != count or summary["received_basic_identity_checked"] != count
            or summary["heartbeat_count"] < 0 or summary.get("pending_task_ids") != []
            or summary["max_pending"] != min(window, count)
            or not 0 < summary["protobuf_payload_bytes"] <= limit):
        raise ValueError("continuous buffer counts, pending tasks or declared payload bounds disagree")
    seen, total = set(), 0
    for artifact in summary["artifacts"]:
        task_id = artifact.get("task_id")
        if (type(task_id) is not int or not 1 <= task_id <= count or task_id in seen
                or artifact.get("file") != f"results/{task_id:06d}.pb"
                or type(artifact.get("bytes")) is not int or artifact["bytes"] <= 0):
            raise ValueError("continuous result evidence contains missing/duplicate/unknown tasks")
        result_raw = (directory / artifact["file"]).read_bytes()
        if sha256(result_raw) != artifact.get("sha256") or len(result_raw) != artifact["bytes"]:
            raise ValueError("continuous result artifact hash/size mismatch")
        total += len(result_raw)
        seen.add(task_id)
    if seen != set(range(1, count + 1)) or total != summary["protobuf_payload_bytes"]:
        raise ValueError("continuous result evidence does not cover all requests or payload bytes")
    return mode


def load_worker_config(path, backend, batch):
    """CPU-only import; preserve settings and snapshot every supported runtime file.

    Relative execution-file paths follow the Worker's ROOT working directory,
    exactly as normal Rust config loading does. This does not validate a CUDA
    plan's device/build certificate; the actual Rust loader must still do so.
    """
    if backend not in ("cudabackend", "cudaint8backend"):
        raise ValueError("--config currently supports only legacy CUDA/INT8 backends")
    path = Path(path).resolve()
    raw = path.read_bytes()
    lines, values, indices = raw.decode("utf-8").splitlines(), {}, {}
    for index, line in enumerate(lines):
        body = line.partition("#")[0].strip()
        if not body:
            continue
        if body.startswith("@") or re.match(r"(?i)include(?:\s|=|$)", body):
            raise ValueError(f"{path}:{index + 1}: config includes/directives are unsupported")
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_-]*)\s*=\s*(.+)", body)
        if not match:
            raise ValueError(f"{path}:{index + 1}: expected simple key=value")
        key, value = match[1], match[2].strip()
        if key in values:
            raise ValueError(f"{path}:{index + 1}: duplicate config key {key}")
        if key not in IMPORTED_CONFIG_KEYS:
            raise ValueError(f"{path}:{index + 1}: unsupported or indexed config key {key}")
        if value.startswith('"'):
            if len(value) < 2 or not value.endswith('"') or any(c in value[1:-1] for c in '\\"'):
                raise ValueError(f"{path}:{index + 1}: quoted values cannot contain escapes or embedded quotes")
            value = value[1:-1]
        elif '"' in value or "'" in value:
            raise ValueError(f"{path}:{index + 1}: ambiguous quoted value")
        if not value or "\x00" in value:
            raise ValueError(f"{path}:{index + 1}: empty or invalid config value")
        values[key], indices[key] = value, index
    required = {"nnBackend", "nnMaxBatchSize", "numNNServerThreadsPerModel",
                "nnCacheSizePowerOfTwo", "nnMutexPoolSizePowerOfTwo"}
    if missing := required - values.keys():
        raise ValueError(f"imported config requires explicit keys: {sorted(missing)}")
    if values["nnBackend"] != backend:
        raise ValueError("imported nnBackend must exactly match --backend")
    integers = {}
    for key, low, high in (("nnMaxBatchSize", 1, 64), ("numNNServerThreadsPerModel", 1, 1024),
                           ("nnCacheSizePowerOfTwo", -1, 48), ("nnMutexPoolSizePowerOfTwo", -1, 24)):
        if re.fullmatch(r"-?\d+", values[key]) is None or not low <= int(values[key]) <= high:
            raise ValueError(f"imported {key} must be an integer in [{low},{high}]")
        integers[key] = int(values[key])
    if integers["nnMaxBatchSize"] != batch:
        raise ValueError("imported nnMaxBatchSize must exactly match --batch")
    runtime_files = []
    if "cudaTacticPlan" in values:
        source = Path(values["cudaTacticPlan"])
        source = (source if source.is_absolute() else ROOT / source).resolve()
        if not source.is_file():
            raise ValueError(f"cudaTacticPlan file not found (relative paths use Worker cwd {ROOT}): {source}")
        runtime_files.append(dict(config_key="cudaTacticPlan", source_path=str(source),
                                  file="runtime-cudaTacticPlan.json", raw=source.read_bytes()))
    return dict(source_path=str(path), raw=raw, lines=lines, values=values, indices=indices,
                integers=integers, runtime_files=runtime_files)


def freeze_worker_config(imported, output):
    """Write immutable input snapshots and change only execution-file paths."""
    lines, artifacts = list(imported["lines"]), []
    source_file = output / "source-worker.cfg"
    with source_file.open("xb") as stream:
        stream.write(imported["raw"])
    source = dict(file=source_file.name, path=imported["source_path"],
                  sha256=sha256(imported["raw"]), bytes=len(imported["raw"]))
    for item in imported["runtime_files"]:
        destination = output / item["file"]
        with destination.open("xb") as stream:
            stream.write(item["raw"])
        lines[imported["indices"][item["config_key"]]] = f"{item['config_key']} = {config_path(destination)}"
        artifacts.append(dict(config_key=item["config_key"], file=item["file"],
                              sha256=sha256(item["raw"]), bytes=len(item["raw"]),
                              source_path=item["source_path"], source_sha256=sha256(item["raw"])))
    return "\n".join(lines) + "\n", source, artifacts


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("binary", "model", "requests", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="Default: manifest.json beside --requests")
    parser.add_argument("--backend", choices=("cudabackend", "cudaint8backend", "cudaquantbackend"), default="cudabackend")
    parser.add_argument("--config", type=Path, help="Existing simple legacy CUDA/INT8 Worker config; preserve explicit batch/NN threads/cache and freeze cudaTacticPlan")
    parser.add_argument("--execution-spec", type=Path,
                        help="Explicit legacy execution specification; requires --config, freezes source identity and verifies actual fingerprints/launch evidence")
    parser.add_argument("--recipe", type=Path, help="Unified backend recipe; omitted means its FP16 base recipe")
    parser.add_argument("--mxfp8-scale-clear-fusion", choices=("0", "1"),
                        help="Explicit MXFP8 experiment tactic; recorded and bound to the actual runtime profile. Default: backend default (off)")
    parser.add_argument("--int8-scope", choices=("ffn", "transformer"), help="Generated INT8 config only (default: ffn); incompatible with --config")
    parser.add_argument("--batch", type=int, default=8, help="NN physical batch capacity, 1..64; independent of Worker request capacity")
    parser.add_argument("--capacity", type=int, default=32, help="Worker protocol max_in_flight hard limit, 1..4096; independent of NN batch size")
    parser.add_argument("--window", type=int, default=32, help="Outstanding request window, 1..256; refill policy follows --window-mode and must not exceed --capacity")
    parser.add_argument("--window-mode", choices=("chunked", "continuous"), default="chunked",
                        help="chunked preserves the existing fixed-window collector; continuous refills each completion before output bookkeeping")
    parser.add_argument("--max-buffered-result-mib", type=int, default=256,
                        help="Continuous mode only: maximum retained protobuf result payload MiB (1..4096), not a process RSS limit")
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--task-timeout", type=float, default=180)
    parser.add_argument("--max-requests", type=int, default=100000, help="Reject larger input files instead of truncating")
    parser.add_argument("--trace-batches", action="store_true", help="Diagnostic hashes/batches; never performance evidence")
    args = parser.parse_args()
    for key in ("binary", "model", "requests", "output"):
        setattr(args, key, getattr(args, key).resolve())
    args.manifest = (args.manifest or args.requests.parent / "manifest.json").resolve()
    if args.recipe:
        args.recipe = args.recipe.resolve()
    if args.config:
        args.config = args.config.resolve()
        if args.backend == "cudaquantbackend" or args.recipe or args.int8_scope is not None:
            parser.error("--config supports legacy CUDA/INT8 only and cannot be combined with --recipe or --int8-scope")
    if args.execution_spec:
        args.execution_spec = args.execution_spec.resolve()
        if (not args.config or args.backend == "cudaquantbackend" or args.recipe
                or args.mxfp8_scale_clear_fusion is not None or args.trace_batches):
            parser.error("--execution-spec requires a legacy --config and forbids unified recipes/tactics and batch tracing")
    if args.recipe and args.backend != "cudaquantbackend":
        parser.error("--recipe requires --backend cudaquantbackend")
    if args.mxfp8_scale_clear_fusion is not None and (args.backend != "cudaquantbackend" or not args.recipe):
        parser.error("--mxfp8-scale-clear-fusion requires --backend cudaquantbackend and an explicit MXFP8 --recipe")
    if not 1 <= args.batch <= 64 or not 1 <= args.capacity <= 4096 or not 1 <= args.window <= 256 or args.max_requests < 1:
        parser.error("batch/capacity/window/max-requests out of range")
    if args.window > args.capacity:
        parser.error("--window must not exceed --capacity (Worker protocol max_in_flight hard limit)")
    if not 1 <= args.max_buffered_result_mib <= 4096:
        parser.error("--max-buffered-result-mib must be in 1..4096")
    if (not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0
            or not math.isfinite(args.task_timeout) or not 0 < args.task_timeout <= 3600):
        parser.error("startup timeout must be finite positive; task timeout must be in (0,3600]")
    if args.execution_spec and args.startup_timeout > 3600:
        parser.error("legacy execution fingerprint timeout must be <=3600 seconds")
    for path in [args.binary, args.model, args.requests, args.manifest,
                 *([args.recipe] if args.recipe else []), *([args.config] if args.config else []),
                 *([args.execution_spec] if args.execution_spec else [])]:
        if not path.is_file():
            parser.error(f"file not found: {path}")
    if args.output.exists():
        parser.error(f"refuse existing output directory: {args.output}")
    try:
        args.imported_config = load_worker_config(args.config, args.backend, args.batch) if args.config else None
        args.loaded_execution_spec = None
        if args.execution_spec:
            import legacy_quantization_execution as legacy_execution
            args.loaded_execution_spec = legacy_execution.load_spec(
                args.execution_spec, binary=args.binary, model=args.model, config=args.config,
                backend=args.backend, batch=args.batch)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.int8_scope = args.int8_scope or "ffn"
    return args


def main():
    args = parse_args()
    records, games, corpus, sources = load_corpus(args.requests, args.manifest, args.max_requests)
    from compare_worker_outputs import file_hash, message_dict, validate_result
    from tune_runtime import clean_environment
    from worker_protocol_tools import Protocol, WorkerHarness

    model_sha = file_hash(args.model)
    args.output.mkdir(parents=True, exist_ok=False)
    source_dir = args.output / "source"
    source_dir.mkdir()
    for filename, raw in sources.items():
        (source_dir / filename).write_bytes(raw)
    report = dict(schema="rustgo-quantization-collection-v1", status="RUNNING", model_sha256=model_sha,
                  model=str(args.model), binary=str(args.binary), binary_sha256=file_hash(args.binary),
                  backend=args.backend, batch=args.batch, capacity=args.capacity, request_window=args.window,
                  window_mode=args.window_mode,
                  split=records[0]["split"], expected_requests=len(records), completed_requests=0,
                  unique_games=len({record["game_id"] for record in records}),
                  corpus_manifest_sha256=sha256(sources["manifest.json"]),
                  corpus_request_set_sha256=corpus["request_set_sha256"],
                  corpus_status=corpus["status"], source_request_file_sha256=sha256(sources[args.requests.name]),
                  source_artifacts=[dict(file=name, sha256=sha256(raw), bytes=len(raw)) for name, raw in sources.items()],
                  source_validation="manifest-bound game/request artifacts verified; original SGFs are not reread",
                  server_scope="fresh isolated 127.0.0.1 harness; no production endpoint",
                  production_certified=False, fp32_accuracy="NOT_MEASURED_ON_THIS_CORPUS",
                  playing_strength="NOT_MEASURED", performance="NOT_MEASURED", batch_trace_enabled=args.trace_batches,
                  config_mode="imported" if args.imported_config else "generated",
                  nn_server_threads=args.imported_config["integers"]["numNNServerThreadsPerModel"] if args.imported_config else 1,
                  nn_cache_size_power_of_two=args.imported_config["integers"]["nnCacheSizePowerOfTwo"] if args.imported_config else -1,
                  runtime_artifacts=[],
                  cache_policy="every request explicitly sets skip_cache=true; configured cache allocation is recorded, never overridden",
                  output_scope="complete postprocessed white-perspective NNOutput; no raw logits/activation statistics")
    if args.window_mode == "continuous":
        report["collection_buffer_contract"] = dict(
            schema="rustgo-continuous-corpus-buffer-contract-v1",
            max_results=len(records), max_protobuf_payload_bytes=args.max_buffered_result_mib * 1024 * 1024,
            accounting="ByteSize after immediate refill; protobuf payload only, not allocator/native object overhead or process RSS",
            overflow="fail and retain at most one additional transport-bounded received message",
            validation_and_serialization="after successful Worker Drain; on failure after stopping only this collector's Worker",
            equivalence="Continuous refill policy matches the benchmark, but bookkeeping and timing differ; no identical physical-batch or exact timing claim",
        )
    protocol = harness = process = None
    buffered = None
    buffered_evidence_saved = False
    try:
        protocol = Protocol(ROOT / "crates/kata_worker/proto/worker.proto")
        report["schema_sha256"] = file_hash(ROOT / "crates/kata_worker/proto/worker.proto")
        # Session depends only on corpus selection, never precision/model choice.
        session = f"quant-corpus-{report['source_request_file_sha256'][:24]}"
        requests, request_sha = make_requests(protocol, records, model_sha, session, math.ceil(args.task_timeout * 1000))
        report.update(requests_sha256=request_sha, input_hash_contract="SHA256 deterministic EvalRequest with empty input_hash")
        request_dir, result_dir = args.output / "requests", args.output / "results"
        request_dir.mkdir()
        result_dir.mkdir()
        with (args.output / "requests-index.jsonl").open("x", encoding="utf-8") as index:
            for record, request in zip(records, requests, strict=True):
                raw = request.SerializeToString(deterministic=True)
                (request_dir / f"{request.task_id:06d}.pb").write_bytes(raw)
                index.write(json_line(dict(name=record["name"], game_id=record["game_id"], split=record["split"],
                                           ply=record["ply"], phase=record["phase"],
                                           semantic_position_sha256=record["semantic_position_sha256"],
                                           board_state_sha256=record["board_state_sha256"], task_id=request.task_id,
                                           input_hash=request.input_hash.hex(), request_sha256=sha256(raw),
                                           source_sgfs=games[record["game_id"]]["sources"], request=message_dict(request))))
        environment = clean_environment()
        if args.loaded_execution_spec is not None:
            environment.update(args.loaded_execution_spec["environment"])
        if args.mxfp8_scale_clear_fusion is not None:
            environment["KATAGO_CUDA_MXFP8_SCALE_CLEAR_FUSION"] = args.mxfp8_scale_clear_fusion
        if args.trace_batches:
            environment["KATAGO_CUDA_BATCH_TRACE"] = "1"
        report["controlled_environment"] = {key: value for key, value in environment.items() if key.startswith("KATAGO_")}
        cfg = args.output / "worker.cfg"
        if args.imported_config:
            config, report["source_config"], report["runtime_artifacts"] = freeze_worker_config(args.imported_config, args.output)
        else:
            config = (f"rules=chinese\nkomi=7.5\nnnBackend={args.backend}\nnnMaxBatchSize={args.batch}\n"
                      "numNNServerThreadsPerModel=1\nnumSearchThreads=1\nmaxVisits=1\nponderingEnabled=false\n"
                      "nnCacheSizePowerOfTwo=-1\nnnMutexPoolSizePowerOfTwo=8\n")
        # Config komi is only an initialization default. Each request carries its
        # original explicit komi; Engine.prepare replays that complete Position.
        if args.backend == "cudaint8backend" and not args.imported_config:
            config += f"cudaInt8Scope={args.int8_scope}\ncudaInt8MinFfnWidth=0\n"
        if args.recipe:
            recipe_copy = args.output / "recipe.json"
            recipe_copy.write_bytes(args.recipe.read_bytes())
            report["recipe_source"] = str(args.recipe)
            report["recipe_file_sha256"] = file_hash(recipe_copy)
            report["runtime_artifacts"].append(dict(config_key="cudaQuantPlan", file=recipe_copy.name,
                sha256=report["recipe_file_sha256"], bytes=recipe_copy.stat().st_size,
                source_path=str(args.recipe), source_sha256=report["recipe_file_sha256"]))
            config += f"cudaQuantPlan={config_path(recipe_copy)}\n"
            inspect_mxfp8_recipe(args, report, recipe_copy, environment)
            if args.mxfp8_scale_clear_fusion is not None and not report.get("recipe_inspection"):
                raise ValueError("explicit MXFP8 scale-clear experiment requires an MXFP8 recipe validated by CPU quant-inspect")
        cfg.write_text(config, encoding="utf-8")
        discovery = discover_profile(args, "profile", cfg, model_sha, environment) if args.backend == "cudaquantbackend" else None
        report.update(actual_profile=discovery, config_sha256=file_hash(cfg), config_text=cfg.read_text(encoding="utf-8"))
        if args.loaded_execution_spec is not None:
            import legacy_quantization_execution as legacy_execution
            report["legacy_execution"] = dict(spec=legacy_execution.freeze_spec(args.loaded_execution_spec, args.output))
            save_json(args.output / "report.json", report)
            report["legacy_execution"]["before"] = legacy_execution.probe_fingerprint(
                args.loaded_execution_spec, args.output, "before", environment=environment,
                timeout=args.startup_timeout)
        if report.get("recipe_inspection"):
            load_mxfp8_inspection(args.output, report, (args.output / "recipe.json").read_bytes())
            if file_hash(args.binary) != report["binary_sha256"] or file_hash(args.model) != model_sha:
                raise ValueError("binary/model changed between CPU inspection and GTP profile discovery")
        report["runtime_validation"] = "actual Rust loader validates backend/plan at startup; snapshots do not certify optimality or numerical correctness"
        harness = WorkerHarness(protocol)
        command = [str(args.binary), "nnworker", "--server", f"127.0.0.1:{harness.port}",
                   "--worker-id", "quantization-corpus-local", "--capacity", str(args.capacity), "--once",
                   "--model", str(args.model), "--model-sha256", model_sha, "--config", str(cfg)]
        report["command"] = command
        save_json(args.output / "report.json", report)
        completed = set()
        output_hashes = {}
        prepared = None
        if args.window_mode == "continuous":
            buffered = continuous_capture()
            # Construct messages before Worker launch; RPC serialization still
            # occurs at the gRPC boundary, just as in the benchmark driver.
            prepared = [(protocol.pb.ServerMessage(evaluate=request),
                         (request.task_id, request.generation, request.session_id,
                          request.input_hash, request.model_sha256)) for request in requests]
        with (args.output / "worker.log").open("xb") as log, \
                (args.output / "outputs.jsonl").open("x", encoding="utf-8") as outputs:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            peer = harness.accept(process, args.startup_timeout)
            hello = peer.hello
            if (hello.model_sha256 != model_sha or hello.protocol_version != 1 or hello.input_profile != "katago-eval-v1"
                    or hello.max_in_flight != args.capacity or hello.max_board_size != 19 or hello.model_version <= 0
                    or not hello.supports_ownership or not hello.supports_friendly_pass_search
                    or re.search(rf"(?:^|;\s*)backend={args.backend}(?:;|$)", hello.backend_info) is None):
                raise AssertionError("worker Hello does not match the configured real model/backend/capacity")
            if discovery and re.findall(r"(?:^|;\s*)inference-profile=([^;\s]+)", hello.backend_info) != [discovery["inference_profile_id"]]:
                raise AssertionError("Worker actual profile differs from GTP discovery")
            if args.backend == "cudaint8backend":
                values = args.imported_config["values"] if args.imported_config else {}
                scope = values.get("cudaInt8Scope", "ffn") if args.imported_config else args.int8_scope
                width = values.get("cudaInt8MinFfnWidth", "0")
                if (re.findall(r"(?:^|;\s*)int8-scope=([^;\s]+)", hello.backend_info) != [scope]
                        or re.findall(r"(?:^|;\s*)int8-min-ffn-width=([^;\s]+)", hello.backend_info) != [width]):
                    raise AssertionError("Worker actual INT8 scope/minimum width differs from preserved config")
            report["hello"] = message_dict(hello)
            save_json(args.output / "report.json", report)
            if buffered is not None:
                drive_continuous(peer, prepared, args.window, args.capacity, args.task_timeout,
                                 args.max_buffered_result_mib * 1024 * 1024, buffered)
                completed.update(result.task_id for result in buffered["results"])
                report["completed_requests"] = len(completed)
            chunk_starts = range(0, len(requests), args.window) if buffered is None else ()
            for start in chunk_starts:
                pending = {request.task_id: (records[index], request)
                           for index, request in enumerate(requests[start:start + args.window], start)}
                for _, request in pending.values():
                    peer.outgoing.put(protocol.pb.ServerMessage(evaluate=request))
                deadline = time.monotonic() + args.task_timeout
                while pending:
                    message = peer.receive(max(0.001, deadline - time.monotonic()))
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"request window timed out; missing task IDs {sorted(pending)}")
                    if message.HasField("heartbeat"):
                        continue
                    if not message.HasField("result") or message.result.task_id not in pending or message.result.task_id in completed:
                        raise AssertionError("unexpected, duplicate, stale, or wrong-window result")
                    result = message.result
                    record, request = pending.pop(result.task_id)
                    raw = result.SerializeToString(deterministic=True)
                    (result_dir / f"{result.task_id:06d}.pb").write_bytes(raw)
                    try:
                        validate_result(result, request, hello)
                    except Exception:
                        save_json(args.output / "invalid-result.json", dict(name=record["name"], result=message_dict(result)))
                        raise
                    output_raw = result.output.SerializeToString(deterministic=True)
                    outputs.write(json_line(dict(name=record["name"], game_id=record["game_id"], split=record["split"],
                                                 ply=record["ply"], phase=record["phase"],
                                                 semantic_position_sha256=record["semantic_position_sha256"],
                                                 board_state_sha256=record["board_state_sha256"],
                                                 result=message_dict(result), result_sha256=sha256(raw),
                                                 output_sha256=sha256(output_raw))))
                    outputs.flush()
                    output_hashes[result.task_id] = sha256(output_raw)
                    completed.add(result.task_id)
                    report["completed_requests"] = len(completed)
                save_json(args.output / "report.json", report)
                print(f"corpus: {len(completed)}/{len(requests)} {records[0]['split']}", flush=True)
            if completed != {request.task_id for request in requests}:
                raise AssertionError("missing output requests after all windows")
            heartbeat = peer.settled(len(requests))
            if heartbeat.completed_requests != len(requests) or heartbeat.nn_rows != len(requests) or heartbeat.nn_batches == 0:
                raise AssertionError("final heartbeat counts disagree with complete uncached corpus")
            report["final_heartbeat"] = message_dict(heartbeat)
            report["batching"] = dict(nn_rows=heartbeat.nn_rows, nn_batches=heartbeat.nn_batches,
                                      mean_rows_per_batch=heartbeat.nn_rows / heartbeat.nn_batches)
            peer.outgoing.put(protocol.pb.ServerMessage(drain=protocol.pb.Drain(reason="complete corpus collected")))
            if process.wait(timeout=30) != 0:
                raise AssertionError("worker did not exit successfully after Drain")
            report["drained"] = True
            if buffered is not None:
                # All inference has stopped. Preserve every raw result before
                # detailed validation, including later results if an earlier
                # one fails a numerical gate.
                save_continuous_evidence(buffered, args.output, report)
                buffered_evidence_saved = True
                by_id = {request.task_id: (record, request) for record, request in zip(records, requests, strict=True)}
                report["continuous_numerically_validated_results"] = 0
                for result in buffered["results"]:
                    record, request = by_id[result.task_id]
                    raw = (result_dir / f"{result.task_id:06d}.pb").read_bytes()
                    try:
                        validate_result(result, request, hello)
                    except Exception:
                        save_json(args.output / "invalid-result.json", dict(name=record["name"], result=message_dict(result)))
                        raise
                    output_raw = result.output.SerializeToString(deterministic=True)
                    outputs.write(json_line(dict(name=record["name"], game_id=record["game_id"], split=record["split"],
                                                 ply=record["ply"], phase=record["phase"],
                                                 semantic_position_sha256=record["semantic_position_sha256"],
                                                 board_state_sha256=record["board_state_sha256"],
                                                 result=message_dict(result), result_sha256=sha256(raw),
                                                 output_sha256=sha256(output_raw))))
                    output_hashes[result.task_id] = sha256(output_raw)
                    report["continuous_numerically_validated_results"] += 1
                outputs.flush()
                print(f"corpus: {len(completed)}/{len(requests)} {records[0]['split']} continuous (validated after Drain)", flush=True)
        if report.get("recipe_inspection"):
            if (file_hash(args.binary) != report["binary_sha256"] or file_hash(args.model) != model_sha
                    or file_hash(args.output / "recipe.json") != report["recipe_file_sha256"]):
                raise ValueError("binary/model/frozen recipe changed during MXFP8 collection")
            load_mxfp8_inspection(args.output, report, (args.output / "recipe.json").read_bytes())
            profile = report["actual_profile"]
            logged = set(PROFILE_LINE.findall((args.output / "worker.log").read_text(encoding="utf-8", errors="replace")))
            if logged != {(profile["model_sha256"], profile["graph_sha256"], profile["recipe_sha256"], profile["inference_profile_id"])}:
                raise ValueError("actual Worker loader identity differs from the checked CPU export/GTP profile")
            if args.mxfp8_scale_clear_fusion is not None:
                observed = set(re.findall(
                    r"\[cuda-tactic\] name=mxfp8_scale_clear_fusion requested=([01]) launch=(fused|separate) effective=([01])",
                    (args.output / "worker.log").read_text(encoding="utf-8", errors="replace")))
                value = args.mxfp8_scale_clear_fusion
                expected = (value, "fused" if value == "1" else "separate", value)
                if observed != {expected}:
                    raise ValueError(f"MXFP8 scale-clear actual launch marker differs: {observed}; expected {expected}")
                report["mxfp8_scale_clear_execution"] = dict(requested=value, launch=expected[1], effective=value)
        if args.trace_batches:
            trace = parse_batch_trace((args.output / "worker.log").read_text(encoding="utf-8", errors="replace"),
                                      len(requests), args.batch)
            save_json(args.output / "batch-trace.json", trace)
            report["batching"]["physical_batch_distribution"] = trace["physical_batch_distribution"]
        if args.loaded_execution_spec is not None:
            report["legacy_execution"]["after"] = legacy_execution.probe_fingerprint(
                args.loaded_execution_spec, args.output, "after", environment=environment,
                timeout=args.startup_timeout)
        digest = hashlib.sha256()
        for task_id in sorted(output_hashes):
            digest.update(task_id.to_bytes(8, "little") + bytes.fromhex(output_hashes[task_id]))
        report.update(status="COLLECTED_UNVERIFIED", ordered_output_hashes_sha256=digest.hexdigest(),
                      outputs_order="completion order; join by task_id/name/semantic_position_sha256, never line index",
                      artifacts=[dict(file=name, sha256=file_hash(args.output / name))
                                 for name in ("outputs.jsonl", "requests-index.jsonl", "worker.cfg",
                                              *(["source-worker.cfg"] if args.imported_config else []),
                                              *(["worker.log"] if report.get("recipe_inspection") or report.get("legacy_execution") else []),
                                              *(["legacy-execution-spec.json"] if report.get("legacy_execution") else []),
                                              *(["continuous-buffer.json"] if buffered is not None else []),
                                              *(item["file"] for item in report["runtime_artifacts"]))])
        validate_collection_window_mode(args.output, report)
        if args.loaded_execution_spec is not None:
            observed = legacy_execution.validate_collection(args.output, report)
            report["legacy_execution"]["observed_execution_id"] = observed["observed_execution_id"]
        return 0
    except Exception as error:
        report.update(status="FAILED", error=str(error))
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if buffered is not None and not buffered_evidence_saved:
            try:
                save_continuous_evidence(buffered, args.output, report)
            except Exception as evidence_error:
                # Never replace the original failure or pretend partial evidence
                # is a completed/validated collection.
                report["continuous_evidence_error"] = str(evidence_error)
        if harness is not None:
            harness.close()
        if protocol is not None:
            protocol.close()
        save_json(args.output / "report.json", report)


if __name__ == "__main__":
    raise SystemExit(main())
