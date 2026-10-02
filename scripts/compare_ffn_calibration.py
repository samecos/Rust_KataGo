#!/usr/bin/env python3
"""Compare bound single-FFN calibration captures. CPU only; never runs a model.

The 6 percentage-point observation is against the same model's FP16 recipe,
not the original FP32 certificate, combined-recipe accuracy, or performance.
"""
import argparse
import ast
import hashlib
import json
import math
import ntpath
import os
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
HEADS = (("policy", 2172), ("value", 3), ("misc", 10), ("moremisc", 8), ("ownership", 361))
SCALARS = ("white_win_prob", "white_loss_prob", "white_no_result_prob", "white_score_mean",
           "white_score_mean_sq", "white_lead", "var_time_left", "shortterm_winloss_error", "shortterm_score_error")
BITS_STAGE = "shared-output-before-protobuf-default-elision"
SCOPE = "Same-model FP16 calibration observation only; not FP32 certification, combined-recipe accuracy, playing strength or performance."


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def integer(value, minimum=0, maximum=2**64-1):
    require(type(value) is int and minimum <= value <= maximum, "invalid bounded integer")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def decode_json(raw):
    return json.loads(raw, object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def canonical(value, sorted_keys=False):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=sorted_keys).encode()


def resolved_path(path):
    path = Path(path).resolve(strict=True)
    if os.name == "nt":
        value = str(path)
        original = path
        if value[:8].upper() == "\\\\?\\UNC\\":
            value = "\\\\"+value[8:]
        elif value.startswith("\\\\?\\"):
            require(len(value) >= 7 and value[4].isascii() and value[4].isalpha()
                    and value[5:7] == ":\\", "unsupported extended path namespace")
            value = value[4:]
        path = Path(ntpath.normcase(ntpath.normpath(value)))
        require(os.path.samefile(original, path), "path spelling normalization changed the real file")
    return path


class Sources:
    """Only follows explicitly bound files; never searches directories."""
    def __init__(self):
        self.items = {}

    def add(self, source):
        require(isinstance(source, dict) and set(source) == {"path", "bytes", "sha256"}, "invalid Source schema")
        path = Path(source["path"])
        require(path.is_absolute() and valid_sha(source["sha256"]), "invalid Source identity")
        integer(source["bytes"])
        path = resolved_path(path)
        key = os.path.normcase(str(path))
        normalized = dict(path=str(path), bytes=source["bytes"], sha256=source["sha256"])
        require(key not in self.items or self.items[key] == normalized, "conflicting Source binding")
        require(path.is_file() and path.stat().st_size == source["bytes"], "Source size changed")
        with path.open("rb") as stream:
            require(hashlib.file_digest(stream, "sha256").hexdigest() == source["sha256"], "Source SHA changed")
        self.items[key] = normalized
        return path

    def read(self, source, cap=64*1024*1024):
        require(source["bytes"] <= cap, "Source exceeds read cap")
        path = self.add(source)
        raw = path.read_bytes()
        require(len(raw) == source["bytes"] and sha(raw) == source["sha256"], "Source changed while reading")
        return raw

    def json(self, source):
        return decode_json(self.read(source))

    def anchor(self, path, expected_sha):
        path = Path(path)
        require(path.is_absolute() and valid_sha(expected_sha), "absolute result path and SHA required")
        return dict(path=str(resolved_path(path)), bytes=path.stat().st_size, sha256=expected_sha)

    def relative(self, root, binding):
        name = binding["path"]
        require(isinstance(name, str) and name and not any(c in name for c in "\\:\0")
                and all(p not in ("", ".", "..") for p in name.split("/")), "nonlocal binding")
        root = resolved_path(root)
        path = resolved_path(root / name)
        require(path.is_relative_to(root), "binding escapes corpus")
        return dict(binding, path=str(path))

    def recheck(self):
        for source in list(self.items.values()):
            self.add(source)


def load_metrics(sources):
    namespace = {"math": math}
    selected = []
    for filename, names in [("validate_int8_backend.py", {"quantization_metrics"}),
                            ("compare_worker_outputs.py", {"compare_outputs", "SCALAR_GATES", "ARRAY_GATES"})]:
        path = ROOT / "scripts" / filename
        raw = path.read_bytes()
        source = dict(path=str(path.resolve()), bytes=len(raw), sha256=sha(raw))
        sources.add(source)
        nodes = []
        found = set()
        for node in ast.parse(raw, filename=str(path)).body:
            name = node.name if isinstance(node, ast.FunctionDef) else (
                node.targets[0].id if isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) else None)
            if name in names:
                require(name not in found, "duplicate metric definition")
                if isinstance(node, ast.FunctionDef):
                    require(not node.decorator_list and not node.args.defaults and not node.args.kw_defaults,
                            "metric definition acquired executable decorators/defaults")
                else:
                    ast.literal_eval(node.value)
                nodes.append(node)
                found.add(name)
        require(found == names, "metric definition missing")
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        selected.append(dict(source=source, definitions=sorted(names)))
    return namespace["quantization_metrics"], namespace["compare_outputs"], selected


def varint(raw, position):
    start, value = position, 0
    for shift in range(0, 70, 7):
        require(position < len(raw), "truncated protobuf varint")
        byte = raw[position]
        position += 1
        value |= (byte & 127) << shift
        if byte < 128:
            require(value < 2**64 and (position-start == 1 or byte != 0), "noncanonical protobuf varint")
            return value, position
    raise ValueError("overflowed protobuf varint")


def decode_pb(raw):
    """Read the exact canonical prost NNOutput subset, not a general RPC client."""
    bits = dict(policy_f32=[], ownership_f32=[], scalars_f64=[0]*9, has_shortterm_error=False)
    position, previous = 0, 0
    while position < len(raw):
        tag, position = varint(raw, position)
        field, wire = tag >> 3, tag & 7
        require(previous < field <= 12, "unknown/duplicate/unordered NNOutput field")
        previous = field
        if field in (1, 11):
            require(wire == 2, "repeated f32 must use canonical packed encoding")
            length, position = varint(raw, position)
            require(length > 0 and length % 4 == 0 and position+length <= len(raw), "invalid packed f32")
            bits["policy_f32" if field == 1 else "ownership_f32"] = list(struct.unpack("<"+"I"*(length//4), raw[position:position+length]))
            position += length
        elif 2 <= field <= 10:
            require(wire == 1 and position+8 <= len(raw), "invalid double wire field")
            value = struct.unpack_from("<Q", raw, position)[0]
            require(value not in (0, 1 << 63), "prost default scalar must be elided")
            bits["scalars_f64"][field-2] = value
            position += 8
        else:
            require(wire == 0, "invalid bool wire field")
            value, position = varint(raw, position)
            require(value == 1, "noncanonical bool")
            bits["has_shortterm_error"] = True
    return bits


def output_from_bits(bits):
    require(set(bits) == {"policy_f32", "ownership_f32", "scalars_f64", "has_shortterm_error"}, "typed schema mismatch")
    require(len(bits["policy_f32"]) == 362 and len(bits["ownership_f32"]) == 361
            and len(bits["scalars_f64"]) == 9 and type(bits["has_shortterm_error"]) is bool,
            "typed output shape/capability mismatch")
    require(bits["has_shortterm_error"], "native v17 shortterm capability missing")
    arrays = {key: [struct.unpack("<f", struct.pack("<I", integer(v, maximum=2**32-1)))[0] for v in bits[name]]
              for key, name in (("policy", "policy_f32"), ("ownership", "ownership_f32"))}
    scalars = dict(zip(SCALARS, (struct.unpack("<d", struct.pack("<Q", integer(v)))[0] for v in bits["scalars_f64"]), strict=True))
    require(all(math.isfinite(v) for values in arrays.values() for v in values)
            and all(math.isfinite(v) for v in scalars.values()), "nonfinite PB/typed output")
    policy = arrays["policy"]
    require(all(v == -1.0 or 0.0 <= v <= 1.0001 for v in policy)
            and abs(sum(max(v, 0.0) for v in policy)-1.0) <= 1e-4, "invalid policy or normalization")
    require(all(-1.0001 <= v <= 1.0001 for v in arrays["ownership"]), "invalid ownership")
    require(all(0.0 <= scalars[k] <= 1.0 for k in SCALARS[:3])
            and abs(sum(scalars[k] for k in SCALARS[:3])-1.0) <= 1e-5, "invalid value probabilities")
    return dict(**arrays, **scalars, has_shortterm_error=True)


def validate_bits(row, pb):
    require(row["bits_stage"] == BITS_STAGE, "unknown typed bits stage")
    pre, wire = row["bits"], row["wire_bits"]
    output_from_bits(pre)
    output = output_from_bits(wire)
    require(decode_pb(pb) == wire, "actual protobuf differs from wire_bits")
    normalized = dict(pre, scalars_f64=[0 if x == 1 << 63 else x for x in pre["scalars_f64"]])
    require(normalized == wire, "prewire differs beyond protobuf scalar default elision")
    return output


def slice_bytes(blob, descriptor, offset):
    require(set(descriptor) == {"offset", "bytes", "sha256"}, "invalid slice schema")
    size = integer(descriptor["bytes"], 1)
    require(integer(descriptor["offset"]) == offset and offset+size <= len(blob), "slice gap/overlap/out of bounds")
    raw = blob[offset:offset+size]
    require(valid_sha(descriptor["sha256"]) and sha(raw) == descriptor["sha256"], "slice SHA mismatch")
    return raw, offset+size


def recipe_identity(recipe):
    require(set(recipe) == {"schema", "version", "quantization_semantics_version", "model_sha256", "graph_sha256", "projections"}
            and recipe["schema"] == "rustgo-precision-recipe" and recipe["version"] == 1
            and recipe["quantization_semantics_version"] == 1, "unsupported recipe")
    entries = recipe["projections"]
    require(entries and len({p["id"] for p in entries}) == len(entries), "empty/duplicate recipe projection")
    result = {key: recipe[key] for key in ("schema", "version", "quantization_semantics_version", "model_sha256", "graph_sha256")}
    result["projections"] = []
    for p in sorted(entries, key=lambda p: p["id"]):
        require(set(p) == {"id", "expected_n", "expected_k", "precision"} and p["precision"] in ("fp16", "int8"), "invalid projection")
        integer(p["expected_n"], 1); integer(p["expected_k"], 1)
        result["projections"].append({key: p[key] for key in ("id", "expected_n", "expected_k", "precision")})
    return sha(canonical(result))


class Capture:
    def __init__(self, sources, source, reference):
        self.sources, self.source = sources, source
        self.result = result = sources.json(source)
        require(result["status"] == "CAPTURE_COMPLETE_METRICS_PENDING" and result["sources_unchanged"] is True
                and result["gpu_model_uploads"] == 1, "capture is not complete")
        self.intent = intent = sources.json(result["intent"])
        require(intent["schema"] == "rustgo-ffn-calibration-attempt-v1" and intent["mode"] == "collect"
                and intent["planned_uploads"] == 1 and intent["retry_allowed"] is False, "invalid capture intent")
        require(result["sources"] == intent["sources"] and len(result["sources"]) == 6, "source inventory mismatch")
        for item in result["sources"]:
            sources.add(item)
        self.plan = plan = sources.json(result["sources"][0])
        require(plan == intent["plan"] and plan["schema"] == "rustgo-ffn-calibration-collection-plan-v1"
                and plan["mode"] in ("complete-calibration", "bounded-diagnostic"), "plan differs from intent")
        require(result["sources"][2:] == [plan[k] for k in ("model", "recipe", "proposal", "provenance")], "plan source inventory differs")
        self.recipe = recipe = sources.json(plan["recipe"])
        require(recipe_identity(recipe) == plan["resolved_recipe_sha256"] == result["recipe_sha256"], "resolved recipe SHA mismatch")
        for key in ("model_sha256", "graph_sha256"):
            require(recipe[key] == result[key] and valid_sha(result[key]), "model/graph identity mismatch")
        require(plan["model"]["sha256"] == result["model_sha256"] and plan["graph_sha256"] == result["graph_sha256"], "plan model/graph differs")
        require(result["output_contract_sha256"] == intent["output_contract_sha256"] and valid_sha(result["output_contract_sha256"]), "output contract differs")
        group_ids = plan["group_ids"]
        require(len(group_ids) == (0 if reference else 1), "reference must be FP16; candidate must target one FFN")
        actual = {p["id"] for p in recipe["projections"] if p["precision"] == "int8"}
        expected = set() if reference else {group_ids[0]+".dual", group_ids[0]+".down"}
        require(actual == expected, "wrong reference/ablation precision")
        proposal = sources.json(plan["proposal"])
        declared = proposal.pop("proposal_sha256")
        require(declared == plan["proposal_object_sha256"] == sha(canonical(proposal, True)), "proposal object SHA differs")
        require(proposal["model_sha256"] == result["model_sha256"] and proposal["graph_sha256"] == result["graph_sha256"], "proposal model/graph differs")
        matched = [p for p in proposal["recipes"] if p["recipe_sha256"] == result["recipe_sha256"]]
        require(len(matched) == 1 and matched[0]["recipe"] == recipe and matched[0]["group_ids"] == group_ids
                and matched[0]["role"] == ("FP16_REFERENCE" if reference else "SINGLE_FFN_INT8_ABLATION"), "proposal recipe binding differs")
        load = sources.json(result["load_attempt"])
        require(load == dict(model=plan["model"], recipe=plan["recipe"], ordinal=0), "load attempt differs")
        self.runtime = runtime = sources.json(result["runtime"])
        native = runtime["native"]
        require(runtime["schema"] == "rustgo-ffn-calibration-runtime-v1" and runtime["model_uploads"] == 1
                and all(runtime[k] == 0 for k in ("implicit_warmup", "graph_capture", "graph_replay", "automatic_retries"))
                and runtime["raw_heads_prefilled_nan_each_forward"] is True, "runtime execution scope differs")
        for key in ("model_sha256", "graph_sha256"):
            require(native[key] == result[key], "runtime model/graph differs")
        require(native["resolved_recipe_sha256"] == result["recipe_sha256"]
                and native["source_recipe_sha256"] == plan["recipe"]["sha256"], "runtime recipe differs")
        sources.add(runtime["executable"])
        require(runtime["executable"]["sha256"] == result["sources"][1]["sha256"], "runtime executable differs")
        for library in runtime["libraries"]:
            sources.add(library)
        declared_projections = {p["id"]: p for p in recipe["projections"]}
        require(len(native["projections"]) == len(declared_projections), "runtime projection count differs")
        seen = set()
        for entry in native["projections"]:
            p = entry["projection"]; key = p["id"]
            require(key not in seen and key in declared_projections, "runtime projection missing/duplicate")
            seen.add(key); expected_projection = declared_projections[key]
            require((p["n"], p["k"], entry["precision"]) == (expected_projection["expected_n"], expected_projection["expected_k"], expected_projection["precision"]), "runtime precision/shape differs")
        indices = plan["numeric_input_indices"]
        require(indices and len(indices) <= 2044 and len(set(indices)) == len(indices), "invalid numeric indices")
        for index in indices: integer(index, maximum=4096)
        require(result["numeric_inputs"] == len(indices) and result["numeric_rows"] == plan["expected_numeric_rows"], "numeric count differs")
        if plan["mode"] == "complete-calibration":
            require(indices == list(range(2044)) and result["numeric_rows"] == 2044, "complete requires exact 2044 B1")
        self.calls = intent["calls"]
        require(len(self.calls) == result["consumed_forwards"] == plan["expected_forwards"], "forward count differs")
        require([c["input_index"] for c in self.calls if c["phase"] == "numeric"] == indices, "numeric call order differs")
        journal = sources.read(result["journal"])
        require(journal.endswith(b"\n"), "journal lacks terminal newline")
        events = [decode_json(line) for line in journal.splitlines()]
        require(len(events) == len(self.calls), "journal consumption differs")
        for ordinal, (event, call) in enumerate(zip(events, self.calls, strict=True)):
            require(event["ordinal"] == ordinal and event["event"]["expected"] == call, "journal expected call differs")
            observed = event["event"]["runtime"]
            require(all(observed[k] == call[k] for k in ("phase", "iteration", "physical_batch", "tensor_sha256"))
                    and observed["runtime_attempt"] == ordinal+1
                    and all(observed[k] == result[k] for k in ("model_sha256", "graph_sha256", "recipe_sha256")), "journal observed call differs")
        for cost in result["cost"]: sources.add(cost)
        self.provenance = sources.json(plan["provenance"])
        require(self.provenance["source_split"] == "calibration" and self.provenance["model_sha256"] == result["model_sha256"], "wrong input provenance")
        self.corpus_root = resolved_path(plan["provenance"]["path"]).parent
        self.prepared = sources.json(sources.relative(self.corpus_root, self.provenance["prepared_manifest"]))
        self.proofs = sources.json(sources.relative(self.corpus_root, self.provenance["request_provenance"]))
        require(self.prepared["split"] == "calibration" and self.prepared["model_sha256"] == result["model_sha256"]
                and len(self.prepared["requests"]) == len(self.proofs) == self.provenance["encoded_count"], "prepared corpus identity/count differs")
        if plan["mode"] == "complete-calibration":
            require(self.provenance["coverage_mode"] == "complete-calibration" and self.provenance["encoded_count"] == 2044, "incomplete input corpus")

    def check_input(self, descriptor):
        path = resolved_path(descriptor["manifest_path"])
        require(path.is_relative_to(self.corpus_root), "input manifest escapes corpus")
        binding = next((b for b in self.provenance["input_manifests"]
                        if resolved_path(self.corpus_root/b["path"]) == path), None)
        require(binding is not None and binding["sha256"] == descriptor["manifest_sha256"], "input manifest is not bound in provenance")
        manifest = self.sources.json(self.sources.relative(self.corpus_root, binding))
        require(manifest["schema"] == "rustgo-encoded-group-cost-input-v1" and manifest["purpose"] == "calibration", "input manifest scope differs")
        matched = [item for item in manifest["inputs"] if item["id"] == descriptor["id"]]
        require(len(matched) == 1 and matched[0]["physical_batch"] == descriptor["physical_batch"], "input missing/duplicated/wrong batch in manifest")
        item = matched[0]
        spatial = self.sources.read(self.sources.relative(path.parent, item["spatial"]), 8*22*361*4)
        global_values = self.sources.read(self.sources.relative(path.parent, item["global"]), 8*19*4)
        batch = descriptor["physical_batch"]
        require(len(spatial) == batch*22*361*4 and len(global_values) == batch*19*4
                and sha(b"rustgo-encoded-group-cost-tensors-v1\0"+struct.pack("<Q", batch)+spatial+global_values) == descriptor["tensor_sha256"], "input tensor domain/shape differs")
        for index, identity in enumerate(descriptor["rows"]):
            logical = integer(identity["logical_row"], maximum=len(self.proofs)-1)
            entry, proof = self.prepared["requests"][logical], self.proofs[logical]
            require(entry["pb"] == proof["pb"] and entry["record"] == proof["record"], "input PB proof differs")
            for key in ("source_line", "task_id", "input_hash_hex", "wire_semantic_sha256", "source_line_sha256"):
                require(entry[key] == proof[key] == identity[key], "row source/request identity differs: "+key)
            for key in ("name", "game_id", "phase", "ply", "semantic_position_sha256", "board_state_sha256"):
                require(entry["record"][key] == identity[key], "row semantic record differs: "+key)
            require(entry["record"]["split"] == "calibration" and entry["pb"]["sha256"] == identity["pb_sha256"], "row PB SHA/split differs")
            self.sources.add(self.sources.relative(self.corpus_root, entry["pb"]))
            s = spatial[index*22*361*4:(index+1)*22*361*4]
            g = global_values[index*19*4:(index+1)*19*4]
            require(sha(s) == proof["post_spatial_sha256"] and sha(g) == proof["global_sha256"]
                    and sha(b"rustgo-worker-v7-nchw19-row-f32le-v1\0"+s+g) == identity["row_feature_sha256"] == proof["post_feature_sha256"], "row feature/physical slice differs")

    def rows(self):
        numeric_calls = [c for c in self.calls if c["phase"] == "numeric"]
        count = row_count = 0
        names, identities, placements = set(), set(), set()
        for chunk_source in self.result["chunks"]:
            chunk = self.sources.json(chunk_source)
            require(chunk["schema"] == "rustgo-ffn-calibration-output-chunk-v1"
                    and 1 <= len(chunk["inputs"]) <= self.plan["chunk_inputs"] <= 128, "invalid chunk schema/size")
            raw, pb = self.sources.read(chunk["raw"], 128*8*2554*4), self.sources.read(chunk["pb"], 128*8*20000)
            raw_offset = pb_offset = 0
            for entry in chunk["inputs"]:
                require(count < len(numeric_calls), "extra numeric input")
                call = numeric_calls[count]; descriptor = entry["input"]
                batch = integer(descriptor["physical_batch"], 1, 8)
                require(entry["input_index"] == call["input_index"] and batch == call["physical_batch"]
                        and entry["tensor_sha256"] == descriptor["tensor_sha256"] == call["tensor_sha256"]
                        and call["iteration"] == 0, "numeric placement/tensor differs")
                if self.plan["mode"] == "complete-calibration":
                    require(batch == descriptor["target_batch"] == 1, "complete numeric input is not B1")
                require(len(entry["rows"]) == len(descriptor["rows"]) == batch and len(entry["heads"]) == 5, "physical row/head count differs")
                self.check_input(descriptor)
                head_bytes = []
                for (name, width), head in zip(HEADS, entry["heads"], strict=True):
                    require(head["head"] == name and head["row_width"] == width and head["slice"]["bytes"] == batch*width*4, "raw head shape differs")
                    data, raw_offset = slice_bytes(raw, head["slice"], raw_offset)
                    require(all(math.isfinite(v[0]) for v in struct.iter_unpack("<f", data)), "nonfinite complete raw head")
                    head_bytes.append(data)
                for physical_row, row in enumerate(entry["rows"]):
                    identity = row["identity"]
                    require(row["row"] == physical_row and identity == descriptor["rows"][physical_row], "row identity/placement differs")
                    placement = (entry["input_index"], descriptor["target_batch"], batch, physical_row)
                    require(placement not in placements, "duplicate numeric placement")
                    placements.add(placement)
                    if self.plan["mode"] == "complete-calibration":
                        require(identity["name"] not in names and identity["logical_row"] not in identities, "duplicate numeric request")
                    names.add(identity["name"]); identities.add(identity["logical_row"])
                    require(identity["source_line"] == identity["task_id"] == identity["logical_row"]+1
                            and all(valid_sha(identity[k]) for k in ("pb_sha256", "row_feature_sha256", "wire_semantic_sha256", "input_hash_hex", "source_line_sha256", "semantic_position_sha256", "board_state_sha256", "game_id")), "invalid row identity")
                    if self.plan["mode"] == "complete-calibration":
                        require(identity["logical_row"] == row_count, "complete request order/coverage differs")
                    require(identity["phase"] in ("early", "middle", "late") or isinstance(identity["phase"], str) and bool(identity["phase"]), "missing phase")
                    per_head = [data[physical_row*width*4:(physical_row+1)*width*4] for (_, width), data in zip(HEADS, head_bytes, strict=True)]
                    require(sha(b"".join(per_head)) == row["raw_row_sha256"], "raw row SHA differs")
                    row_pb, pb_offset = slice_bytes(pb, row["pb"], pb_offset)
                    output = validate_bits(row, row_pb)
                    yield dict(identity=identity, placement=(entry["input_index"], batch, physical_row, descriptor["target_batch"], entry["tensor_sha256"]),
                               descriptor=descriptor, output=output, raw=per_head)
                    row_count += 1
                count += 1
            require(raw_offset == len(raw) and pb_offset == len(pb), "unreferenced chunk bytes")
        require(count == self.result["numeric_inputs"] and row_count == self.result["numeric_rows"] and row_count > 0, "missing numeric inputs/rows")


def compare(reference_path, reference_sha, candidate_path, candidate_sha):
    sources = Sources()
    quantization_metrics, compare_outputs, metric_sources = load_metrics(sources)
    # Bind this adapter and the wire declaration too, rather than source text names alone.
    for path in (Path(__file__), ROOT / "crates/kata_worker/proto/worker.proto"):
        raw = path.read_bytes(); sources.add(dict(path=str(path.resolve()), bytes=len(raw), sha256=sha(raw)))
    ref = Capture(sources, sources.anchor(reference_path, reference_sha), True)
    got = Capture(sources, sources.anchor(candidate_path, candidate_sha), False)
    for key in ("model_sha256", "graph_sha256", "output_contract_sha256", "numeric_inputs", "numeric_rows"):
        require(ref.result[key] == got.result[key], "cross-capture identity/count differs: "+key)
    for key in ("mode", "provenance", "numeric_input_indices", "expected_numeric_rows"):
        require(ref.plan[key] == got.plan[key], "cross-capture input plan differs: "+key)
    require(ref.intent["compiled_sources"] == got.intent["compiled_sources"], "collector/postprocess source versions differ")
    for key in ("device", "driver_version", "backend_build", "effective_tactics", "libraries"):
        require(ref.runtime[key] == got.runtime[key], "runtime identity differs: "+key)
    shape = lambda r: [(p["id"], p["expected_n"], p["expected_k"]) for p in sorted(r["projections"], key=lambda p: p["id"])]
    require(shape(ref.recipe) == shape(got.recipe), "projection inventory differs")
    left, right, groups = {"results": []}, {"results": []}, {"game": {}, "phase": {}, "target_batch": {}}
    unique_requests = set()
    raw_metrics = {name: dict(count=0, max_abs=0.0, sum_squared=0.0, bitwise_equal=0) for name, _ in HEADS}
    for a, b in zip(ref.rows(), got.rows(), strict=True):
        require(a["identity"] == b["identity"] and a["placement"] == b["placement"] and a["descriptor"] == b["descriptor"], "cross-capture request/placement differs")
        require(all((x < 0) == (y < 0) for x, y in zip(a["output"]["policy"], b["output"]["policy"], strict=True)), "policy legality differs")
        unique_requests.add(a["identity"]["logical_row"])
        for side, value in ((left, a), (right, b)):
            name = value["identity"]["name"]
            if ref.plan["mode"] == "bounded-diagnostic":
                name += "@placement:"+":".join(str(x) for x in value["placement"][:4])
            side["results"].append(dict(name=name, result=dict(output=value["output"])))
        index = len(left["results"])-1
        for field, identity_key in (("game", "game_id"), ("phase", "phase")):
            groups[field].setdefault(a["identity"][identity_key], []).append(index)
        groups["target_batch"].setdefault(str(a["descriptor"]["target_batch"]), []).append(index)
        for (name, _), x, y in zip(HEADS, a["raw"], b["raw"], strict=True):
            metric = raw_metrics[name]
            for (p,), (q,), (pbits,), (qbits,) in zip(struct.iter_unpack("<f", x), struct.iter_unpack("<f", y), struct.iter_unpack("<I", x), struct.iter_unpack("<I", y), strict=True):
                difference = abs(p-q); metric["count"] += 1
                metric["max_abs"] = max(metric["max_abs"], difference)
                metric["sum_squared"] += difference*difference
                metric["bitwise_equal"] += pbits == qbits
    metrics = quantization_metrics(left, right)
    diagnostic = compare_outputs(left, right)
    diagnostic["scope"] = "Existing threshold diagnostic against same-model FP16; its PASS/FAIL is not the original FP32 certificate."
    by_group = {}
    for field, values in groups.items():
        by_group[field] = {}
        for identity, indices in sorted(values.items()):
            r = {"results": [left["results"][i] for i in indices]}
            c = {"results": [right["results"][i] for i in indices]}
            accuracy = quantization_metrics(r, c)
            thresholds = compare_outputs(r, c)
            by_group[field][identity] = dict(accuracy=accuracy, fields=thresholds["fields"],
                win_probability_max_abs_pp=accuracy["win_probability_max_abs"]*100,
                win_probability_mean_abs_pp=accuracy["win_probability_mean_abs"]*100,
                policy_top1_mismatches_near_ties=thresholds["policy_top1_mismatches_near_ties"])
    for metric in raw_metrics.values():
        metric["rmse"] = math.sqrt(metric.pop("sum_squared")/metric["count"])
        metric["bitwise_equal_ratio"] = metric["bitwise_equal"]/metric["count"]
    complete = ref.plan["mode"] == "complete-calibration"
    report = dict(schema="rustgo-single-ffn-calibration-comparison-v1", status="COMPARED", scope=SCOPE,
                  reference=ref.source, candidate=got.source, model_sha256=ref.result["model_sha256"],
                  graph_sha256=ref.result["graph_sha256"], output_contract_sha256=ref.result["output_contract_sha256"],
                  reference_recipe_sha256=ref.result["recipe_sha256"], candidate_recipe_sha256=got.result["recipe_sha256"],
                  group_ids=got.plan["group_ids"], coverage_mode=ref.plan["mode"],
                  request_weighting="one weight per original request" if complete else "one weight per physical placement observation; cross-packing repeated requests are not independent semantic samples",
                  unique_request_count=len(unique_requests), placement_observations=len(left["results"]),
                  repeated_request_observations=len(left["results"])-len(unique_requests),
                  accuracy=metrics, existing_threshold_diagnostic=diagnostic, raw_heads=raw_metrics, by_group=by_group,
                  win_probability_max_abs_pp=metrics["win_probability_max_abs"]*100,
                  win_probability_mean_abs_pp=metrics["win_probability_mean_abs"]*100,
                  calibration_6pp_observation=("PASS" if metrics["win_probability_max_abs"] <= 0.06 else "FAIL") if complete else "NOT_FULL_CALIBRATION",
                  production_certified=False, performance_measured=False, metric_definitions=metric_sources,
                  protobuf_bits="Metrics use actual wire-decoded values; pre-wire typed bits separately checked with only double default-zero elision allowed.")
    sources.recheck()
    report["bound_sources"] = list(sources.items.values())
    report["sources_unchanged"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "candidate"):
        parser.add_argument("--"+name, type=Path, required=True)
        parser.add_argument("--"+name+"-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.is_absolute() and not args.output.exists(), "fresh absolute output directory required")
    args.output.mkdir()
    try:
        report = compare(args.reference, args.reference_sha256, args.candidate, args.candidate_sha256)
        pending = args.output / "report.json.pending"
        with pending.open("xb") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2).encode()+b"\n")
            stream.flush(); os.fsync(stream.fileno())
        os.link(pending, args.output / "report.json")
        print(json.dumps(dict(status=report["status"], cases=report["accuracy"]["cases"], calibration_6pp_observation=report["calibration_6pp_observation"])))
    except BaseException as error:
        with (args.output / "FAILED.json").open("x", encoding="utf-8") as stream:
            json.dump(dict(status="FAIL_NO_RETRY", error=repr(error)), stream, ensure_ascii=False, indent=2)
            stream.flush(); os.fsync(stream.fileno())
        raise


if __name__ == "__main__":
    main()
