#!/usr/bin/env python3
"""Read fixed-policy complete FFN event records; never executes a model or GPU.

API: summarize_capture(capture_path, expected_result_sha) -> JSON-compatible report.
Costs describe the explicitly frozen inputs and physical batches. Neither finite
diagnostics nor complete-calibration captures establish end-to-end speed gains.
"""
import argparse
import json
import math
import os
from pathlib import Path
import struct

import compare_ffn_calibration as io

require, integer = io.require, io.integer
POLICY = {"KATAGO_CUDA_SPLITK":"0", "KATAGO_CUDA_INT8_GEMM_TUNE":"0",
          "KATAGO_CUDA_CUBLASLT_RANK":"heuristic", "KATAGO_CUDA_FFN_COMPACT_R1":"0",
          "KATAGO_CUDA_CUBLASLT":"1", "KATAGO_CUDA_DUALFFN":"0", "KATAGO_CUDA_FUSION":"none",
          "KATAGO_CUDA_GEMM_LAYOUT":"tn", "KATAGO_CUDA_INT8_RMS_FUSION":"1", "KATAGO_CUDA_NOGRAPH":"1",
          "KATAGO_CUDA_NOPIPELINE":"1", "KATAGO_CUDA_PADBATCH":"0", "KATAGO_CUDA_BATCH_TRACE":"1"}
SCOPE = ("Complete RMS+FFN+adjacent Gate CUDA-event observations for fixed inputs and policy. "
         "Group costs are not summed into network latency. No ABBA, end-to-end speedup, "
         "accuracy, production certification or search-leaf distribution claim.")


def f32_value(entry):
    bits = integer(entry["elapsed_ms_bits"], maximum=2**32-1)
    value = struct.unpack("<f", struct.pack("<I", bits))[0]
    declared = entry["elapsed_ms"]
    require(type(declared) in (int, float) and math.isfinite(declared) and math.isfinite(value)
            and value >= 0.0, "invalid/nonfinite/negative cost")
    try:
        actual = struct.unpack("<I", struct.pack("<f", declared))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("cost number cannot represent f32") from error
    require(actual == bits, "cost number differs from original f32 bits")
    return value


def statistics(values):
    require(values and all(math.isfinite(x) and x >= 0.0 for x in values), "empty/invalid cost distribution")
    ordered = sorted(values)
    def percentile(q):
        position = (len(ordered)-1)*q
        low = int(position); high = min(low+1, len(ordered)-1)
        return ordered[low]+(ordered[high]-ordered[low])*(position-low)
    mean = math.fsum(values)/len(values)
    return dict(count=len(values), mean_ms=mean, min_ms=ordered[0], max_ms=ordered[-1],
                median_ms=percentile(0.5), p05_ms=percentile(0.05), p95_ms=percentile(0.95),
                population_stddev_ms=math.sqrt(math.fsum((x-mean)**2 for x in values)/len(values)),
                percentile_method="linear interpolation at q*(n-1); descriptive, not confidence intervals")


def projection(n, k, precision):
    padded = (k+15)//16*16
    return dict(n=n, k=k, kp=padded, input_stride=padded if precision == "fp16" else k,
                output_stride=n, precision=precision)


def group_descriptor(group, expected, previous_end=0):
    require(set(group) == {"id", "rms_layer", "ffn_layer", "end_layer", "mid", "hidden", "dual", "down"}, "unknown group descriptor field")
    require(group["id"] == expected["id"] and group["ffn_layer"] == expected["layer_index"], "group identity/layer differs")
    ffn = integer(group["ffn_layer"], 1)
    require(integer(group["rms_layer"]) == ffn-1 and group["rms_layer"] >= previous_end
            and group["end_layer"] in (ffn+1, ffn+2), "overlapping/invalid whole FFN boundary")
    mid, hidden = integer(group["mid"], 1), integer(group["hidden"], 1)
    require(mid in (384,512) and hidden % 8 == 0 and (mid, hidden) == (expected["mid"], expected["hidden"]), "wrong FFN dimensions")
    require(group["dual"] == projection(2*hidden, mid, expected["dual_precision"])
            and group["down"] == projection(mid, hidden, expected["down_precision"]), "precision/padding/stride differs")
    require(expected["dual_precision"] == expected["down_precision"] in ("fp16", "int8"), "only whole FFN ablations are supported")


def launch(roles, kind, layout, dimensions):
    return dict(roles=roles, kind=kind, layout=layout, dimensions=dimensions)


def validate_route(route, group, batch):
    """Actual fixed-13-key dispatches, including legal padded FP16 fallbacks."""
    require(set(route) == {"count", "launches", "overflow"} and route["overflow"] is False
            and len(route["launches"]) == 16, "invalid route storage/overflow")
    count = integer(route["count"], 1, 16)
    require(all(x is not None for x in route["launches"][:count])
            and all(x is None for x in route["launches"][count:]), "route holes/hidden launches")
    observed = route["launches"][:count]
    c,h = group["mid"],group["hidden"]; hp=(h+15)//16*16; rows=batch*361
    gate = group["end_layer"] == group["ffn_layer"]+2
    expected = []
    if group["dual"]["precision"] == "fp16":
        expected.append([launch(1,"rms_norm_f32_w4" if c == 384 else "rms_norm_f32_w4_generic","f32-to-half-row-major",[c]*5)])
        expected.append([launch(2,"cublaslt_f16out","tn",[2*h,c,c,c,2*h]),
                         launch(2,"hgemm_t64_f16out" if rows < 1024 else "hgemm_v2_f16out","tn-row-major-half",[2*h,c,c,c,2*h])])
        expected.append([launch(4,"swiglu_dual" if h == hp else "swiglu_dual_padded",
                                "half-row-major" if h == hp else "half-row-major-zero-padding",[h,2*h,hp,2*h,hp])])
        down = [launch(8,"hgemm_t64" if rows < 1024 else "hgemm_v2","tn-half-input-f32-residual",[c,h,hp,hp,c])]
        if h == hp:
            down.append(launch(8,"cublaslt_f32_residual","tn",[c,h,h,h,c]))
        expected.append(down)
    else:
        expected = [[launch(1,f"int8_rms_quantize{c}_warp4","f32-register-half-to-i8-row-major",[2*h,c,c,c,c])],
                    [launch(2,"cublaslt_i8_i32","tn-row-major-i8-to-i32",[2*h,c,c,c,2*h])],
                    [launch(4,"int8_dequant_swiglu_quantize_"+("warp4" if h <= 512 and rows >= 1024 else "cta256"),
                            "i32-register-half-to-i8-row-major-zero-padding",[h,2*h,hp,2*h,hp])],
                    [launch(8,"cublaslt_i8_i32","tn-row-major-i8-to-i32",[c,h,hp,hp,c])],
                    [launch(8,"int8_dequant_fp32_residual","i32-to-f32-row-major",[c,h,hp,c,c])]]
    if gate: expected.append([launch(16,"gate_silu_f32","f32-row-major",[c]*5)])
    require(len(observed) == len(expected) and all(item in choices for item, choices in zip(observed,expected,strict=True)),
            "route roles/kind/layout/dimensions differ from fixed strategy")


def expected_groups(runtime, recipe):
    native = runtime["native"]
    ids = native["ffn_group_ids"]
    require(len(ids) in (33,45) and len(set(ids)) == len(ids), "requires complete 33/45 native FFN groups")
    projections = {p["projection"]["id"]:p for p in native["projections"]}
    require(len(projections) == len(native["projections"]), "duplicate native projection")
    actual_ids = [entry["projection"]["id"][:-5] for entry in native["projections"] if entry["projection"]["id"].endswith(".ffn.dual")]
    require(actual_ids == ids, "native FFN group inventory/order differs")
    recipe_entries = {p["id"]:p for p in recipe["projections"]}
    groups = []
    for identity in ids:
        dual,down = projections[identity+".dual"], projections[identity+".down"]
        d,p = dual["projection"], down["projection"]
        # Native manifest describes one logical dual projection (N=H); the
        # materialized group descriptor/dispatch concatenates both (N=2H).
        require(d["layer_index"] == p["layer_index"] and (d["n"],d["k"]) == (p["k"],p["n"]), "projection group dimensions/layer differ")
        require(dual["precision"] == recipe_entries[identity+".dual"]["precision"]
                and down["precision"] == recipe_entries[identity+".down"]["precision"], "recipe group precision differs")
        groups.append(dict(id=identity,layer_index=d["layer_index"],mid=d["k"],hidden=p["k"],dual_precision=dual["precision"],down_precision=down["precision"]))
    require([g["layer_index"] for g in groups] == sorted(g["layer_index"] for g in groups), "FFN group execution order differs")
    return groups


def expected_binding(capture, binding):
    runtime,result = capture.runtime,capture.result
    require(binding["schema"] == "rustgo-group-cost" and binding["version"] == 1
            and binding["mode"] == "DIRECT_DIAGNOSTIC", "wrong cost binding schema/mode")
    for key in ("model_sha256","graph_sha256","recipe_sha256"):
        require(binding[key] == result[key], "cost model/graph/recipe binding differs")
    require(binding["executable_sha256"] == runtime["executable"]["sha256"]
            and binding["backend_build"] == runtime["backend_build"]
            and binding["driver_version"] == runtime["driver_version"], "cost executable/build/driver differs")
    for key,target in (("gpu_name","gpu_name"),("gpu_uuid_hex","uuid_hex"),("compute_capability","compute_capability"),("sm_count","sm_count")):
        require(binding[key] == runtime["device"][target], "cost GPU identity differs")
    require(runtime["effective_tactics"] == POLICY, "unsupported collector policy")
    tactics=binding["tactics"]
    require(all(key in tactics and tactics[key] == value for key,value in POLICY.items() if key != "KATAGO_CUDA_BATCH_TRACE")
            and all(value == POLICY.get(key) for key,value in tactics.items()), "cost tactic binding differs")


def head_hashes(heads,batch):
    require(len(heads) == len(io.HEADS), "missing cost output head")
    for item,(name,width) in zip(heads,io.HEADS,strict=True):
        require(set(item) == {"bytes","head","sha256"} and item["head"] == name
                and item["bytes"] == batch*width*4 and io.valid_sha(item["sha256"]), "wrong cost head shape/hash")


def sample(capture,value,inventory,batch,label,slot,sequence,phase,baseline=None):
    require(value["physical_batch"] == batch and value["activation_rows"] == batch*361
            and value["input_label_sha256"] == label and value["workspace_slot"] == slot
            and value["sequence"] == sequence and value["phase"] == phase, "cost sample batch/input/slot/sequence/phase differs")
    require(type(value["setup_observed"]) is bool and (phase != "measure" or not value["setup_observed"]), "setup during measured cost")
    expected_binding(capture,value["binding"])
    require(len(value["groups"]) == len(inventory), "incomplete FFN group coverage")
    previous=0; output=[]
    for index,(entry,expected) in enumerate(zip(value["groups"],inventory,strict=True)):
        group=entry["group"]; group_descriptor(group,expected,previous);previous=group["end_layer"]
        validate_route(entry["route"],group,batch)
        if baseline is not None:
            require(group == baseline["groups"][index]["group"] and entry["route"] == baseline["groups"][index]["route"]
                    and value["binding"] == baseline["binding"], "cost descriptor/route/binding changed after warmup")
        output.append((f32_value(entry),entry["elapsed_ms_bits"]))
    return output


def input_identity(capture,descriptor,index,batch):
    require(descriptor["target_batch"] == descriptor["physical_batch"] == batch and len(descriptor["rows"]) == batch,
            "tail/other packing cannot stand in for complete cost batch")
    count=integer(capture.provenance["encoded_count"],1,2044);remaining=integer(index)
    first=None
    for target in capture.provenance["packings"]:
        integer(target,1,8); require(target in (1,3,8), "unsupported input packing")
        length=(count+target-1)//target
        if remaining < length:
            require(target == batch, "cost index targets another packing")
            first=remaining*target;break
        remaining-=length
    require(first is not None and first+batch <= count
            and [row["logical_row"] for row in descriptor["rows"]] == list(range(first,first+batch))
            and descriptor["id"] == f"calibration-b{batch}-row-{first:04}", "cost index/member order differs")
    capture.check_input(descriptor)


def summarize_costs(capture):
    """Core validator, also exercised by bounded CPU fixtures."""
    plan,result,runtime=capture.plan,capture.result,capture.runtime
    inventory=expected_groups(runtime,capture.recipe)
    require(runtime["effective_tactics"] == POLICY, "unsupported collector policy")
    costs=plan["cost"]
    require(1 <= len(costs) <= 3 and len({c["physical_batch"] for c in costs}) == len(costs), "missing/duplicate cost batches")
    require(len(result["cost"]) == sum(1+len(c["input_indices"]) for c in costs), "missing/extra cost artifacts")
    calls=capture.calls
    journal=[io.decode_json(line) for line in capture.sources.read(result["journal"]).splitlines()]
    require(len(journal) == len(calls), "cost journal length differs")
    cursor=0; file_index=0; collector_slots=set();group_slots=set();numeric_slots={};batches=[]
    def consume(expected,collector_slot,group_slot,first_warm=False):
        nonlocal cursor
        require(cursor < len(calls) and calls[cursor] == expected, "cost call differs from fixed W/S/M order")
        event=journal[cursor]
        require(event["ordinal"] == cursor and event["event"]["expected"] == expected, "cost journal expected call differs")
        observed=event["event"]["runtime"]
        require(observed["workspace_kind"] == "cost" and observed["collector_slot"] == collector_slot
                and observed["group_workspace_slot"] == (None if first_warm else group_slot)
                and observed["stream_handle"] == runtime["stream_handle"], "cost journal workspace/stream differs")
        require(observed["runtime_attempt"] == cursor+1 and all(observed[key] == expected[key] for key in ("phase","iteration","physical_batch","tensor_sha256"))
                and all(observed[key] == result[key] for key in ("model_sha256","graph_sha256","recipe_sha256")), "cost journal observed identity differs")
        cursor+=1
    def call(index,descriptor,phase,iteration):
        return dict(input_index=index,phase=phase,iteration=iteration,physical_batch=descriptor["physical_batch"],tensor_sha256=descriptor["tensor_sha256"])
    def read_artifact(expected_name):
        nonlocal file_index
        source=result["cost"][file_index];file_index+=1
        require(io.resolved_path(source["path"]).name == expected_name, "cost artifact order/name differs")
        return source,capture.sources.json(source)
    for index in plan["numeric_input_indices"]:
        require(cursor < len(calls), "numeric journal prefix missing")
        current=calls[cursor];event=journal[cursor]["event"]["runtime"]
        require(current["phase"] == "numeric" and current["iteration"] == 0 and current["input_index"] == index
                and event["workspace_kind"] == "numeric" and event["group_workspace_slot"] is None
                and event["stream_handle"] == runtime["stream_handle"], "numeric prefix interferes with cost slots")
        batch=current["physical_batch"];slot=integer(event["collector_slot"],1)
        require(numeric_slots.get(batch,slot) == slot and (batch in numeric_slots or slot not in collector_slots), "numeric workspace reused by another batch")
        numeric_slots[batch]=slot;collector_slots.add(slot);cursor+=1
    for cost in costs:
        batch=integer(cost["physical_batch"],1,8);require(batch in (1,3,8),"unsupported cost batch")
        warmup=integer(cost["warmup"],1,32);measurements=integer(cost["measurements"],1,64)
        indices=cost["input_indices"]
        require(1 <= len(indices) <= 128 and len(set(indices)) == len(indices), "invalid/duplicate cost samples")
        warm_source,warm=read_artifact(f"cost-b{batch}-warm.json")
        descriptor=warm["input"];input_identity(capture,descriptor,indices[0],batch)
        wr=warm["runtime"];collector=integer(wr["collector_slot"],1);slot=integer(wr["group_workspace_slot"],1)
        require(collector not in collector_slots and slot not in group_slots,"cost workspace reused across batch/kind")
        collector_slots.add(collector);group_slots.add(slot)
        require(wr["warmup"] == warmup and len(wr["samples"]) == warmup and wr["ready_to_measure"] is True
                and wr["physical_batch"] == batch and wr["input_tensor_sha256"] == descriptor["tensor_sha256"], "fixed warmup metadata differs")
        warm_records=[];sequence=0;last_warm=None
        for iteration,wrapped in enumerate(wr["samples"]):
            consume(call(indices[0],descriptor,"cost-warmup",iteration),collector,slot,first_warm=iteration==0)
            sequence+=1;current=wrapped["sample"]
            values=sample(capture,current,inventory,batch,descriptor["tensor_sha256"],slot,sequence,"warmup")
            head_hashes(wrapped["raw_output_heads"],batch)
            if last_warm is not None:
                require(current["binding"] == last_warm["binding"] and [g["group"] for g in current["groups"]] == [g["group"] for g in last_warm["groups"]], "warmup group/binding changed")
            last_warm=current
            warm_records.append(dict(sequence=sequence,setup_observed=current["setup_observed"],elapsed_ms_bits=[bits for _,bits in values]))
        require(last_warm["setup_observed"] is False,"last fixed warmup is not setup-free")
        input_reports=[];aggregate=[[] for _ in inventory];means=[[] for _ in inventory]
        for index in indices:
            source,record=read_artifact(f"cost-b{batch}-input{index:04}.json")
            descriptor=record["input"];input_identity(capture,descriptor,index,batch)
            if index == indices[0]:require(descriptor == warm["input"],"first measured input differs from frozen warm tensor")
            require(record["input_index"] == index and record["tensor_sha256"] == descriptor["tensor_sha256"], "measured input binding differs")
            measured=record["runtime"]
            require(measured["physical_batch"] == batch and measured["collector_slot"] == collector and measured["group_workspace_slot"] == slot
                    and measured["warmup_once_for_batch"] == warmup and measured["measurements"] == measurements
                    and len(measured["samples"]) == measurements and measured["forward_attempts_this_input"] == measurements+2
                    and measured["input_tensor_sha256"] == descriptor["tensor_sha256"]
                    and measured["all_five_heads_bitwise_equal"] is True,"measured session/count metadata differs")
            before=measured["ordinary_before_heads"];head_hashes(before,batch);head_hashes(measured["ordinary_after_heads"],batch)
            require(before == measured["ordinary_after_heads"],"ordinary before/after output hashes differ")
            consume(call(index,descriptor,"cost-before",0),collector,slot)
            per_group=[[] for _ in inventory];per_bits=[[] for _ in inventory]
            for iteration,wrapped in enumerate(measured["samples"]):
                consume(call(index,descriptor,"cost-measure",iteration),collector,slot)
                sequence+=1
                values=sample(capture,wrapped["sample"],inventory,batch,descriptor["tensor_sha256"],slot,sequence,"measure",last_warm)
                head_hashes(wrapped["raw_output_heads"],batch)
                require(wrapped["bitwise_matches_before"] is True and wrapped["raw_output_heads"] == before,"measured output hashes differ from ordinary")
                for group_index,(value,bits) in enumerate(values):per_group[group_index].append(value);per_bits[group_index].append(bits)
            consume(call(index,descriptor,"cost-after",0),collector,slot)
            reports=[]
            for group_index,expected in enumerate(inventory):
                stats=statistics(per_group[group_index]);aggregate[group_index].extend(per_group[group_index]);means[group_index].append(stats["mean_ms"])
                reports.append(dict(group_id=expected["id"],elapsed_ms_bits=per_bits[group_index],elapsed_ms=per_group[group_index],statistics=stats))
            input_reports.append(dict(input_index=index,input=descriptor,source=source,groups=reports))
        summaries=[dict(group=last_warm["groups"][i]["group"],route=last_warm["groups"][i]["route"],
                        pooled_measurements=statistics(aggregate[i]),equal_input_mean_ms=math.fsum(means[i])/len(means[i]),
                        input_means_distribution=statistics(means[i])) for i in range(len(inventory))]
        batches.append(dict(physical_batch=batch,activation_rows=batch*361,collector_slot=collector,group_workspace_slot=slot,
            warmup=warmup,measurements_per_input=measurements,sample_inputs=len(indices),planned_forwards=warmup+len(indices)*(measurements+2),
            warmup_source=warm_source,warmup_records=warm_records,binding=last_warm["binding"],inputs=input_reports,groups=summaries))
    require(cursor == len(calls) == result["consumed_forwards"] == plan["expected_forwards"] and file_index == len(result["cost"]), "unconsumed/extra cost calls/artifacts")
    return dict(schema="rustgo-ffn-calibration-cost-summary-v1",status="SUMMARIZED_COST_OBSERVATIONS",scope=SCOPE,
                coverage_mode=plan["mode"],numeric_accuracy_checked=False,numeric_chunks_revalidated=False,
                model_sha256=result["model_sha256"],graph_sha256=result["graph_sha256"],recipe_sha256=result["recipe_sha256"],
                output_contract_sha256=result["output_contract_sha256"],group_ids=plan["group_ids"],complete_ffn_groups=len(inventory),
                weighting="equal frozen input weight within each physical batch; fixed equal M per input; warmup excluded",
                capture=capture.source,batches=batches,end_to_end_performance_certified=False,
                raw_output_bit_gate="Recorded full-head SHA equality and producer bit-gate flags checked; cost files contain hashes, not independent raw output blobs.")


def summarize_capture(capture_path, expected_result_sha):
    sources=io.Sources();anchor=sources.anchor(capture_path,expected_result_sha)
    result=sources.json(anchor);intent=sources.json(result["intent"])
    capture=io.Capture(sources,anchor,reference=not intent["plan"]["group_ids"])
    for path in (Path(__file__),Path(io.__file__)):
        raw=path.read_bytes();sources.add(dict(path=str(path.resolve()),bytes=len(raw),sha256=io.sha(raw)))
    report=summarize_costs(capture)
    sources.recheck();report["bound_sources"]=list(sources.items.values());report["sources_unchanged"]=True
    return report


def write_report(directory, report):
    directory=Path(directory)
    require(directory.is_absolute() and not directory.exists(),"fresh absolute report directory required")
    directory.mkdir()
    raw=json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2).encode()+b"\n"
    pending=directory/"report.json.pending"
    with pending.open("xb") as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    os.link(pending,directory/"report.json")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture",type=Path,required=True)
    parser.add_argument("--capture-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    require(args.output.is_absolute() and not args.output.exists(),"fresh absolute report directory required")
    try:
        report=summarize_capture(args.capture,args.capture_sha256)
        write_report(args.output,report)
        print(json.dumps(dict(status=report["status"],groups=report["complete_ffn_groups"],physical_batches=[b["physical_batch"] for b in report["batches"]])))
    except BaseException as error:
        args.output.mkdir(exist_ok=True)
        with (args.output/"FAILED.json").open("x",encoding="utf-8") as stream:
            json.dump(dict(status="FAIL_NO_RETRY",error=repr(error)),stream,ensure_ascii=False,indent=2)
            stream.flush();os.fsync(stream.fileno())
        raise


if __name__ == "__main__":main()
