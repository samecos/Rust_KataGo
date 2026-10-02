"""Read the frozen complete Worker input provenance; no model or inference calls.

This is an outer-driver metadata/tensor admission helper, not a replacement for
the collector's strict Rust PB replay and production feature re-encoding.
Only explicitly bound files are followed. Tensor bytes are released per input.
"""
from pathlib import Path
import hashlib
import math
import struct

from compare_ffn_calibration import Sources, decode_json, integer, require, resolved_path, sha, valid_sha

TOTAL = 2044
SPATIAL_BYTES = 22 * 361 * 4
GLOBAL_BYTES = 19 * 4
ROW_DOMAIN = b"rustgo-worker-v7-nchw19-row-f32le-v1\0"
TENSOR_DOMAIN = b"rustgo-encoded-group-cost-tensors-v1\0"
COST_RANK_DOMAIN = b"rustgo-ffn-calibration-cost-sample-v1\0"
PINNED_SOURCES = {
    "corpus_manifest": "c7b80205d26d5dd7c7cf38dceb8b88bcc8b3b452b764f7a83113d156fc88c842",
    "calibration_requests": "c3d3ee4b4f75fadc613e343beaf60c60d95879bb89e6d516cfa5b4278650238f",
    "games_metadata": "c0c9de9d8ed7dc8e99f9dda57ebb974e99c49c7bb99601f93235be671acc0768",
    "collector": "840fb9fb15e60f400b080e453ba9e9912f87339944ae7e9b233328f0f02133c4",
    "protocol_tools": "c01e02eb4a755074372ac7a8c0481502c5f5cc62a48f40ce91c0397a8d051b1a",
    "validator": "2dd88c396411e85d233c9723fafc9e8513beb376f42569bb5dbdd4fcd3de9f7f",
    "proto": "8a43d78331e64040f44e2c0fe49456d6d75c9ea88b11b111e03473fb48741e5e",
}
IDENTITY_FIELDS = ("source_line", "task_id", "input_hash_hex", "wire_semantic_sha256", "source_line_sha256")
RECORD_FIELDS = ("name", "game_id", "phase", "ply", "semantic_position_sha256", "board_state_sha256")


def _fields(value, names):
    require(type(value) is dict and set(value) == set(names.split()), "unexpected object fields")


def _lines(raw):
    require(raw and raw.endswith(b"\n"), "JSONL requires final LF")
    result = [line.removesuffix(b"\r") for line in raw[:-1].split(b"\n")]
    require(all(result), "empty JSONL line")
    return result


def _bound(sources, root, binding, cap=64*1024*1024):
    _fields(binding, "path bytes sha256")
    integer(binding["bytes"], 0, cap)
    source = sources.relative(root, binding)
    return source, sources.read(source, cap)


def _provenance(p):
    _fields(p, "schema status source_split coverage_mode source_count encoded_count packing packings feature_contract model_sha256 prepared_manifest prepared_manifest_sha256 source_copies request_provenance rows input_manifests encoder_executable_sha256 compiled_sources model_loaded gpu_called outputs_read row_feature_domain group_cost_domain")
    require(p["schema"] == "rustgo-worker-calibration-input-provenance-v1"
            and p["status"] == "CPU_INPUT_ENCODING_ONLY_NOT_MODEL_VALIDATION"
            and p["source_split"] == "calibration" and p["coverage_mode"] == "complete-calibration"
            and type(p["source_count"]) is int and p["source_count"] == TOTAL
            and type(p["encoded_count"]) is int and p["encoded_count"] == TOTAL
            and p["packing"] == "source-order-exact-tail-no-padding"
            and p["packings"] == [1, 3, 8] and all(type(b) is int for b in p["packings"]), "complete three-packing corpus required")
    require(valid_sha(p["model_sha256"]) and all(p[k] is False for k in ("model_loaded", "gpu_called", "outputs_read")), "encoder identity/scope differs")
    require(p["feature_contract"] == {"inputs_version":7,"spatial":[22,19,19],"global":[19],"layout":"NCHW","dtype":"f32le",
        "spatial_symmetry_applied":True,"global_transform":"identity",
        "model_contract_validation":"NOT_LOADED; caller must verify v7/22/19/no-SGF-metadata before inference"}, "feature contract differs")
    require(p["row_feature_domain"] == "rustgo-worker-v7-nchw19-row-f32le-v1\\0 + spatial-row-f32LE + global-row-f32LE"
            and p["group_cost_domain"] == "rustgo-encoded-group-cost-tensors-v1\\0 + u64LE(B) + all-spatial-f32LE + all-global-f32LE", "hash domain differs")
    require(p["prepared_manifest"]["path"] == "prepared-requests.json"
            and p["prepared_manifest_sha256"] == p["prepared_manifest"]["sha256"]
            and p["request_provenance"]["path"] == "request-provenance.json" and p["rows"]["path"] == "rows.jsonl", "metadata identity differs")
    require(valid_sha(p["encoder_executable_sha256"]) and p["compiled_sources"]
            and all(valid_sha(s) for s in p["compiled_sources"].values()), "invalid encoder source claims")


def _row_map(actual, descriptor, row, manifest, spatial, global_values):
    expected = descriptor["rows"][row]
    value = dict(target_batch=descriptor["target_batch"], physical_batch=descriptor["physical_batch"], row=row,
        logical_row=expected["logical_row"], source_line=expected["source_line"], task_id=expected["task_id"],
        manifest=manifest, input_id=descriptor["id"], input_tensor_sha256=descriptor["tensor_sha256"],
        spatial_file=spatial["path"], global_file=global_values["path"], spatial_sha256=spatial["sha256"], global_sha256=global_values["sha256"],
        spatial_offset_bytes=row*SPATIAL_BYTES, spatial_bytes=SPATIAL_BYTES,
        global_offset_bytes=row*GLOBAL_BYTES, global_bytes=GLOBAL_BYTES,
        row_feature_sha256=expected["row_feature_sha256"], padding=False)
    require(actual == value and actual.get("padding") is False, "physical row/offset/identity differs")
    for key in ("target_batch", "physical_batch", "row", "logical_row", "source_line", "task_id", "spatial_offset_bytes", "spatial_bytes", "global_offset_bytes", "global_bytes"):
        integer(actual[key])


def _tensor_rows(descriptor, spatial, global_values, proofs):
    batch = descriptor["physical_batch"]
    require(len(spatial) == batch*SPATIAL_BYTES and len(global_values) == batch*GLOBAL_BYTES, "tensor shape differs")
    require(all(math.isfinite(x[0]) for raw in (spatial, global_values) for x in struct.iter_unpack("<f", raw)), "nonfinite tensor")
    require(sha(TENSOR_DOMAIN+struct.pack("<Q",batch)+spatial+global_values) == descriptor["tensor_sha256"], "tensor SHA domain differs")
    for row, identity in enumerate(descriptor["rows"]):
        proof = proofs[identity["logical_row"]]
        s = spatial[row*SPATIAL_BYTES:(row+1)*SPATIAL_BYTES]
        g = global_values[row*GLOBAL_BYTES:(row+1)*GLOBAL_BYTES]
        require(sha(s) == proof["post_spatial_sha256"] and sha(g) == proof["global_sha256"]
                and sha(ROW_DOMAIN+s+g) == proof["post_feature_sha256"] == identity["row_feature_sha256"], "row tensor/proof differs")


def load_inputs(source: dict, sources: Sources) -> list[dict]:
    """Validate the complete explicit input tree and return 2,982 metadata rows.

    `source` is an already expected absolute provenance Source, never a freshly
    rebound hash. Caller binds its model to provenance.model_sha256 separately.
    The same Sources instance must be retained for prelaunch/terminal rechecks.
    """
    p = sources.json(source); _provenance(p)
    root = resolved_path(source["path"]).parent
    _, raw = _bound(sources, root, p["prepared_manifest"]); prepared = decode_json(raw)
    _fields(prepared, "schema mode split source_total selected_count packings model_sha256 session_id generation lease_ms ordered_pb_sha256 sources requests producer")
    require(prepared["schema"] == "rustgo-calibration-pb-input-v1" and prepared["mode"] == p["coverage_mode"]
            and prepared["split"] == "calibration" and prepared["source_total"] == prepared["selected_count"] == TOTAL
            and prepared["packings"] == p["packings"] and prepared["model_sha256"] == p["model_sha256"]
            and prepared["sources"] == p["source_copies"] and len(prepared["requests"]) == TOTAL, "prepared identity/coverage differs")
    integer(prepared["source_total"],TOTAL,TOTAL); integer(prepared["selected_count"],TOTAL,TOTAL)
    require(prepared["generation"] == 1 and type(prepared["generation"]) is int
            and isinstance(prepared["session_id"],str) and 1 <= len(prepared["session_id"]) <= 128
            and valid_sha(prepared["ordered_pb_sha256"]), "invalid prepared envelope")
    integer(prepared["lease_ms"],1,3600000)
    roles = set(); original = None
    require(len(prepared["sources"]) == 8, "source inventory differs")
    for origin in prepared["sources"]:
        _fields(origin,"role original_path file")
        role = origin["role"]
        require(role not in roles and origin["original_path"] and origin["file"]["path"].startswith("sources/"), "duplicate/invalid source role")
        roles.add(role)
        expected = prepared["producer"]["script_sha256"] if role == "preparer" else PINNED_SOURCES.get(role)
        require(valid_sha(expected) and origin["file"]["sha256"] == expected, "pinned source changed")
        _, raw = _bound(sources,root,origin["file"])
        if role == "calibration_requests": original = _lines(raw)
    require(roles == set(PINNED_SOURCES)|{"preparer"} and original is not None and len(original) == TOTAL, "missing calibration source")
    _, raw = _bound(sources,root,p["request_provenance"]); proofs = decode_json(raw)
    require(type(proofs) is list and len(proofs) == TOTAL, "proof count differs")
    identities=[]; unique={k:set() for k in ("name","semantic_position_sha256","board_state_sha256","pb_sha256")}; ordered=hashlib.sha256()
    for i,(entry,proof,line) in enumerate(zip(prepared["requests"],proofs,original,strict=True)):
        _fields(entry,"source_line source_line_sha256 record task_id pb input_hash_hex wire_semantic_sha256")
        _fields(proof,"source_line source_line_sha256 record task_id pb input_hash_hex wire_semantic_sha256 prepared pre_spatial_sha256 post_spatial_sha256 global_sha256 pre_feature_sha256 post_feature_sha256")
        require(entry["record"] == proof["record"] == decode_json(line) and entry["record"]["split"] == "calibration"
                and sha(line) == entry["source_line_sha256"] and entry["pb"] == proof["pb"]
                and entry["pb"]["path"] == f"requests/{i+1:04}.pb", "prepared/proof/original JSON join differs")
        require(all(entry[k] == proof[k] for k in IDENTITY_FIELDS)
                and integer(entry["source_line"],1,TOTAL) == integer(entry["task_id"],1,TOTAL) == i+1, "source row order differs")
        pb, raw = _bound(sources,root,entry["pb"],512*1024)
        ordered.update(struct.pack("<Q",len(raw))); ordered.update(raw)
        identity = dict(logical_row=i,**{k:entry[k] for k in IDENTITY_FIELDS},**{k:entry["record"][k] for k in RECORD_FIELDS},
            pb_sha256=entry["pb"]["sha256"],row_feature_sha256=proof["post_feature_sha256"],pb=pb)
        for key in ("pb_sha256","row_feature_sha256","wire_semantic_sha256","input_hash_hex","source_line_sha256","game_id","semantic_position_sha256","board_state_sha256"):
            require(valid_sha(identity[key]),"invalid row SHA")
        for key in ("pre_spatial_sha256","post_spatial_sha256","global_sha256","pre_feature_sha256","post_feature_sha256"):
            require(valid_sha(proof[key]),"invalid encoded proof SHA")
        require(all(isinstance(identity[k],str) and identity[k] for k in ("name","phase")),"missing semantic label")
        integer(identity["ply"],0,10000)
        for key,seen in unique.items():
            require(identity[key] not in seen,"duplicate semantic/PB source row"); seen.add(identity[key])
        identities.append(identity)
    require(ordered.hexdigest() == prepared["ordered_pb_sha256"], "ordered PB inventory differs")
    _, raw = _bound(sources,root,p["rows"]); maps=[decode_json(line) for line in _lines(raw)]
    require(len(maps) == TOTAL*3,"row packing coverage differs")
    inputs=[]; shard_index=0; row_cursor=0
    for target in (1,3,8):
        ranges=[(first,min(target,TOTAL-first)) for first in range(0,TOTAL,target)]
        for shard,offset in enumerate(range(0,len(ranges),128)):
            directory=f"b{target}-shard-{shard:03}"; manifest_name=directory+"/inputs.json"
            require(shard_index < len(p["input_manifests"]),"missing input shard")
            binding=p["input_manifests"][shard_index]; shard_index+=1
            require(binding["path"] == manifest_name,"duplicate/misordered shard")
            manifest,raw=_bound(sources,root,binding); data=decode_json(raw); chunk=ranges[offset:offset+128]
            _fields(data,"schema purpose provenance inputs")
            require(data["schema"] == "rustgo-encoded-group-cost-input-v1" and data["purpose"] == "calibration"
                    and len(data["inputs"]) == len(chunk)
                    and data["provenance"] == f"CPU_WORKER_V7_INPUTS_ONLY; mode=complete-calibration; prepared_sha256={p['prepared_manifest_sha256']}; see external provenance.json and rows.jsonl; no model/accuracy/performance claim", "shard scope/count differs")
            for item,(first,batch) in zip(data["inputs"],chunk,strict=True):
                _fields(item,"id physical_batch spatial global")
                identity=f"calibration-b{target}-row-{first:04}"
                require(item["id"] == identity and integer(item["physical_batch"],1,8) == batch
                        and item["spatial"]["path"] == identity+".spatial.f32le"
                        and item["global"]["path"] == identity+".global.f32le", "input identity or physical tail differs")
                spatial_binding=dict(item["spatial"],path=directory+"/"+item["spatial"]["path"])
                global_binding=dict(item["global"],path=directory+"/"+item["global"]["path"])
                spatial,s=_bound(sources,root,spatial_binding,8*SPATIAL_BYTES)
                global_values,g=_bound(sources,root,global_binding,8*GLOBAL_BYTES)
                label=sha(TENSOR_DOMAIN+struct.pack("<Q",batch)+s+g)
                descriptor=dict(index=len(inputs),id=identity,target_batch=target,physical_batch=batch,tensor_sha256=label,
                    manifest=manifest,spatial=spatial,rows=identities[first:first+batch])
                descriptor["global"]=global_values
                for row in range(batch):
                    _row_map(maps[row_cursor],descriptor,row,manifest_name,spatial_binding,global_binding);row_cursor+=1
                _tensor_rows(descriptor,s,g,proofs)
                inputs.append(descriptor)
    require(shard_index == len(p["input_manifests"]) and row_cursor == len(maps),"extra input shard or row")
    return inputs


def choose_cost_inputs(inputs: list[dict]) -> list[dict]:
    """Rank complete physical inputs by domain/B/tensor SHA, never by outputs.

    Rank order is execution order; the first selected input receives exactly the
    one W=3 warmup for that B. The two models use the same tensor identities.
    """
    require(isinstance(inputs,list) and inputs,"missing input directory")
    require([x["index"] for x in inputs] == list(range(len(inputs))),"input indices/order differ")
    require(len({x["id"] for x in inputs}) == len(inputs),"duplicate input identity")
    result=[]
    for batch in (1,3,8):
        pool=[]
        for item in inputs:
            integer(item["index"]);integer(item["target_batch"],1,8);integer(item["physical_batch"],1,8)
            require(valid_sha(item["tensor_sha256"]),"invalid tensor SHA")
            if item["target_batch"] != batch or item["physical_batch"] != batch:continue
            rank=sha(COST_RANK_DOMAIN+struct.pack("<Q",batch)+bytes.fromhex(item["tensor_sha256"]))
            pool.append((rank,item["id"],item["index"]))
        require(len(pool) >= 8,"not enough complete physical cost inputs")
        result.append(dict(physical_batch=batch,input_indices=[x[2] for x in sorted(pool)[:8]],warmup=3,measurements=4))
    return result
