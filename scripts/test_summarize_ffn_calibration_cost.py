"""CPU-only synthetic group events. No model, CUDA, or real capture reads.

Core fixtures intentionally stub input feature validation (covered by the existing
comparator); their artifact Sources and final source-change checks are real.
"""
import copy
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

import summarize_ffn_calibration_cost as m


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def group(index=0, hidden=472, precision="fp16", mid=384, gate=True):
    layer = 2+index*4
    return dict(id=f"block{index}.ffn", rms_layer=layer-1, ffn_layer=layer,
        end_layer=layer+1+int(gate), mid=mid, hidden=hidden,
        dual=dict(n=2*hidden,k=mid,kp=mid,input_stride=mid,output_stride=2*hidden,precision=precision),
        down=dict(n=mid,k=hidden,kp=(hidden+15)//16*16,
                  input_stride=(hidden+15)//16*16 if precision == "fp16" else hidden,
                  output_stride=mid,precision=precision))


def route(g, batch):
    c,h=g["mid"],g["hidden"];hp=(h+15)//16*16
    def item(role,kind,layout,dims):return dict(roles=role,kind=kind,layout=layout,dimensions=dims)
    if g["dual"]["precision"] == "fp16":
        entries=[item(1,"rms_norm_f32_w4" if c == 384 else "rms_norm_f32_w4_generic","f32-to-half-row-major",[c]*5),
            item(2,"cublaslt_f16out","tn",[2*h,c,c,c,2*h]),
            item(4,"swiglu_dual" if h == hp else "swiglu_dual_padded",
                 "half-row-major" if h == hp else "half-row-major-zero-padding",[h,2*h,hp,2*h,hp]),
            item(8,"cublaslt_f32_residual" if h == hp else ("hgemm_t64" if batch == 1 else "hgemm_v2"),
                 "tn" if h == hp else "tn-half-input-f32-residual",[c,h,hp,hp,c])]
    else:
        entries=[item(1,f"int8_rms_quantize{c}_warp4","f32-register-half-to-i8-row-major",[2*h,c,c,c,c]),
            item(2,"cublaslt_i8_i32","tn-row-major-i8-to-i32",[2*h,c,c,c,2*h]),
            item(4,"int8_dequant_swiglu_quantize_"+("warp4" if h <= 512 and batch >= 3 else "cta256"),
                 "i32-register-half-to-i8-row-major-zero-padding",[h,2*h,hp,2*h,hp]),
            item(8,"cublaslt_i8_i32","tn-row-major-i8-to-i32",[c,h,hp,hp,c]),
            item(8,"int8_dequant_fp32_residual","i32-to-f32-row-major",[c,h,hp,c,c])]
    if g["end_layer"] == g["ffn_layer"]+2:entries.append(item(16,"gate_silu_f32","f32-row-major",[c]*5))
    return dict(count=len(entries),overflow=False,launches=entries+[None]*(16-len(entries)))


class Fixture:
    def __init__(self, root, count=33, batches=(1,3,8), sample_count=2, warmup=2, measurements=2, mode="bounded-diagnostic"):
        self.root=Path(root);self.objects={};self.sources=m.io.Sources()
        self.groups=[group(i,hidden=472 if i%2 else 512,precision="int8" if i == 0 else "fp16",
                           mid=384 if count == 33 else 512,gate=i%2 == 0) for i in range(count)]
        self.runtime=dict(executable={"sha256":"e"*64},backend_build={"synthetic":True},driver_version=13030,
            device=dict(gpu_name="SYNTHETIC",uuid_hex="a"*32,compute_capability="12.0",sm_count=70),
            effective_tactics=copy.deepcopy(m.POLICY),stream_handle=123,
            native=dict(ffn_group_ids=[g["id"] for g in self.groups],projections=[]))
        self.recipe={"projections":[]}
        for g in self.groups:
            for role in ("dual","down"):
                p=g[role];identity=g["id"]+"."+role
                self.runtime["native"]["projections"].append(dict(projection=dict(id=identity,n=g["hidden"] if role == "dual" else p["n"],k=p["k"],layer_index=g["ffn_layer"]),precision=p["precision"]))
                self.recipe["projections"].append(dict(id=identity,precision=p["precision"]))
        self.result=dict(model_sha256="a"*64,graph_sha256="b"*64,recipe_sha256="c"*64,output_contract_sha256="d"*64,cost=[])
        self.binding=dict(schema="rustgo-group-cost",version=1,mode="DIRECT_DIAGNOSTIC",
            **{k:self.result[k] for k in ("model_sha256","graph_sha256","recipe_sha256")},
            executable_sha256="e"*64,backend_build=self.runtime["backend_build"],driver_version=13030,
            gpu_name="SYNTHETIC",gpu_uuid_hex="a"*32,compute_capability="12.0",sm_count=70,
            tactics={k:v for k,v in m.POLICY.items() if k != "KATAGO_CUDA_BATCH_TRACE"})
        self.binding["tactics"]["KATAGO_CUDA_ATTN"]=None
        self.plan=dict(mode=mode,group_ids=[self.groups[0]["id"]],cost=[],numeric_input_indices=[0])
        self.calls=[];self.events=[];self.input_checks=[]
        self.add_call(0,self.descriptor(1,0),"numeric",0,1,None,"numeric")
        offsets={1:0,3:2044,8:2726}
        for position,b in enumerate(batches):
            collector=position+2;slot=position+1;indices=list(range(offsets[b],offsets[b]+sample_count))
            self.plan["cost"].append(dict(physical_batch=b,warmup=warmup,measurements=measurements,input_indices=indices))
            d=self.descriptor(b,0);warm=[];sequence=0
            for iteration in range(warmup):
                self.add_call(indices[0],d,"cost-warmup",iteration,collector,None if iteration == 0 else slot)
                sequence+=1;warm.append(dict(sample=self.sample(d,slot,sequence,"warmup",setup=iteration==0),raw_output_heads=self.heads(b)))
            self.objects[f"cost-b{b}-warm.json"]=dict(input=d,runtime=dict(physical_batch=b,collector_slot=collector,group_workspace_slot=slot,
                warmup=warmup,input_tensor_sha256=d["tensor_sha256"],samples=warm,ready_to_measure=True))
            for input_number,index in enumerate(indices):
                d=self.descriptor(b,input_number*b);samples=[]
                self.add_call(index,d,"cost-before",0,collector,slot)
                for iteration in range(measurements):
                    self.add_call(index,d,"cost-measure",iteration,collector,slot);sequence+=1
                    sample=self.sample(d,slot,sequence,"measure",cost=1+2*input_number+2*iteration)
                    samples.append(dict(sample=sample,raw_output_heads=self.heads(b),bitwise_matches_before=True))
                self.add_call(index,d,"cost-after",0,collector,slot)
                self.objects[f"cost-b{b}-input{index:04}.json"]=dict(input=d,input_index=index,tensor_sha256=d["tensor_sha256"],runtime=dict(
                    physical_batch=b,collector_slot=collector,group_workspace_slot=slot,warmup_once_for_batch=warmup,
                    input_tensor_sha256=d["tensor_sha256"],measurements=measurements,forward_attempts_this_input=measurements+2,all_five_heads_bitwise_equal=True,
                    ordinary_before_heads=self.heads(b),ordinary_after_heads=self.heads(b),samples=samples))
        self.result["consumed_forwards"]=self.plan["expected_forwards"]=len(self.calls)

    def descriptor(self,b,first):
        return dict(id=f"calibration-b{b}-row-{first:04}",physical_batch=b,target_batch=b,
            tensor_sha256=m.io.sha(f"input-{b}-{first}".encode()),rows=[dict(logical_row=i) for i in range(first,first+b)])

    def heads(self,b):return [dict(head=n,bytes=b*w*4,sha256="f"*64) for n,w in m.io.HEADS]

    def sample(self,d,slot,sequence,phase,setup=False,cost=1.0):
        b=d["physical_batch"]
        return dict(binding=copy.deepcopy(self.binding),physical_batch=b,activation_rows=b*361,workspace_slot=slot,
            sequence=sequence,input_label_sha256=d["tensor_sha256"],phase=phase,setup_observed=setup,
            groups=[dict(group=copy.deepcopy(g),route=route(g,b),elapsed_ms=cost,elapsed_ms_bits=bits(cost)) for g in self.groups])

    def add_call(self,index,d,phase,iteration,collector,slot,kind="cost"):
        call=dict(input_index=index,phase=phase,iteration=iteration,physical_batch=d["physical_batch"],tensor_sha256=d["tensor_sha256"])
        self.calls.append(call)
        observed=dict(workspace_kind=kind,collector_slot=collector,group_workspace_slot=slot,stream_handle=123,runtime_attempt=len(self.calls),
            **{k:call[k] for k in ("phase","iteration","physical_batch","tensor_sha256")},
            **{k:self.result[k] for k in ("model_sha256","graph_sha256","recipe_sha256")})
        self.events.append(dict(ordinal=len(self.calls)-1,event=dict(expected=copy.deepcopy(call),runtime=observed)))

    def write(self,name,raw):
        path=self.root/name;path.write_bytes(raw)
        return dict(path=str(path.resolve()),bytes=len(raw),sha256=m.io.sha(raw))

    def capture(self):
        self.result["cost"]=[self.write(name,json.dumps(value).encode()) for name,value in self.objects.items()]
        self.result["journal"]=self.write("journal.jsonl",b"\n".join(json.dumps(e).encode() for e in self.events))
        return SimpleNamespace(plan=self.plan,result=self.result,runtime=self.runtime,recipe=self.recipe,calls=self.calls,sources=self.sources,
            provenance=dict(encoded_count=2044,packings=[1,3,8]),check_input=self.input_checks.append,source={"synthetic_core_fixture":True})


class Costs(unittest.TestCase):
    def setUp(self):self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
    def tearDown(self):self.temporary.cleanup()
    def fixture(self,**kwargs):return Fixture(self.root,**kwargs)
    def measured(self,f):return f.objects["cost-b1-input0000.json"]["runtime"]
    def reject(self,fixture):
        with self.assertRaises((ValueError,KeyError,TypeError)):m.summarize_costs(fixture.capture())

    def test_full_33_group_three_batch_summary_preserves_bits_and_weights(self):
        f=self.fixture();report=m.summarize_costs(f.capture())
        self.assertEqual(report["complete_ffn_groups"],33);self.assertEqual(len(f.input_checks),9)
        for b in report["batches"]:
            self.assertEqual(b["planned_forwards"],10)
            g=b["groups"][0];self.assertEqual(g["equal_input_mean_ms"],3)
            self.assertEqual(g["pooled_measurements"]["count"],4)
            self.assertEqual(b["inputs"][0]["groups"][0]["elapsed_ms_bits"],[bits(1),bits(3)])
        self.assertFalse(report["end_to_end_performance_certified"])

    def test_45_groups_complete_mode_keeps_separate_numeric_scope(self):
        f=self.fixture(count=45,mode="complete-calibration",batches=(1,))
        report=m.summarize_costs(f.capture());self.assertEqual(report["complete_ffn_groups"],45)
        self.assertFalse(report["numeric_accuracy_checked"])

    def test_full_cost_schedule_s8_w3_m4_all_batches(self):
        f=self.fixture(sample_count=8,warmup=3,measurements=4)
        report=m.summarize_costs(f.capture())
        self.assertEqual(len(f.calls),154)  # one synthetic numeric plus 3*(3+8*(4+2))
        self.assertEqual(len(f.input_checks),27)
        self.assertEqual([b["group_workspace_slot"] for b in report["batches"]],[1,2,3])
        for b in report["batches"]:
            self.assertEqual(b["planned_forwards"],51)
            self.assertEqual(b["sample_inputs"],8)
            self.assertEqual(b["groups"][0]["pooled_measurements"]["count"],32)
            self.assertEqual(b["groups"][0]["equal_input_mean_ms"],11)
        last=f.objects["cost-b8-input2733.json"]["runtime"]["samples"][-1]["sample"]
        self.assertEqual(last["sequence"],35)  # ordinary before/after never advance GroupCost
        last["sequence"]=7
        # Rebind this deliberate invalid fixture; the semantic gate must reject it.
        f.sources=m.io.Sources()
        self.reject(f)

    def test_f32_bits_and_nonfinite(self):
        self.assertEqual(m.f32_value(dict(elapsed_ms=0.1,elapsed_ms_bits=bits(0.1))),struct.unpack("<f",struct.pack("<f",0.1))[0])
        for number,value in [(1,bits(2)),(-1,bits(-1)),(float("nan"),0x7fc00000),(float("inf"),0x7f800000),(1,True)]:
            with self.subTest(number=number,value=value),self.assertRaises(ValueError):m.f32_value(dict(elapsed_ms=number,elapsed_ms_bits=value))

    def test_distribution_definition(self):
        stats=m.statistics([1,3,5]);self.assertEqual(stats["mean_ms"],3);self.assertAlmostEqual(stats["p05_ms"],1.2)
        self.assertAlmostEqual(stats["population_stddev_ms"],(8/3)**0.5)
        with self.assertRaises(ValueError):m.statistics([])

    def test_padded_fp16_uses_handwritten_down_not_lt(self):
        for b,kind in ((1,"hgemm_t64"),(3,"hgemm_v2"),(8,"hgemm_v2")):
            g=group();r=route(g,b);self.assertEqual(r["launches"][3]["kind"],kind);m.validate_route(r,g,b)
            r["launches"][3]["kind"]="cublaslt_f32_residual";r["launches"][3]["layout"]="tn"
            with self.assertRaises(ValueError):m.validate_route(r,g,b)

    def test_unpadded_fp16_allows_lt_and_valid_handwritten_fallback(self):
        for b in (1,3,8):
            g=group(hidden=512);r=route(g,b);m.validate_route(r,g,b)
            r["launches"][1].update(kind="hgemm_t64_f16out" if b == 1 else "hgemm_v2_f16out",layout="tn-row-major-half")
            r["launches"][3].update(kind="hgemm_t64" if b == 1 else "hgemm_v2",layout="tn-half-input-f32-residual")
            m.validate_route(r,g,b)

    def test_int8_padding_and_warp_boundary(self):
        for h in (240,512,520,1152):
            for b in (1,3,8):
                g=group(hidden=h,precision="int8");r=route(g,b);m.validate_route(r,g,b)
                self.assertEqual(g["down"]["input_stride"],h)
                expected="warp4" if h <= 512 and b >= 3 else "cta256"
                self.assertTrue(r["launches"][2]["kind"].endswith(expected))
                r["launches"][0]["kind"]="int8_rms_quantize384_cta256"
                with self.assertRaises(ValueError):m.validate_route(r,g,b)

    def test_route_storage_gate_layout_and_roles_reject(self):
        g=group();base=route(g,1)
        for edit in (lambda r:r.update(overflow=True),lambda r:r["launches"].__setitem__(0,None),
                     lambda r:r["launches"][0].update(roles=3),lambda r:r["launches"][0].update(layout="unknown"),
                     lambda r:r.update(count=4)):
            r=copy.deepcopy(base);edit(r)
            with self.assertRaises(ValueError):m.validate_route(r,g,1)
        r=copy.deepcopy(base);r["launches"][15]={}
        with self.assertRaises(ValueError):m.validate_route(r,g,1)

    def test_missing_reordered_and_wrong_precision_group(self):
        for change in (lambda groups:groups.pop(),lambda groups:groups.reverse(),lambda groups:groups[0]["group"]["down"].update(precision="fp16")):
            f=self.fixture();change(self.measured(f)["samples"][0]["sample"]["groups"]);self.reject(f)

    def test_native_inventory_and_dual_dimensions_reject(self):
        for change in (lambda native:native["ffn_group_ids"].pop(),lambda native:native["projections"][0]["projection"].update(n=1024)):
            f=self.fixture();change(f.runtime["native"]);self.reject(f)

    def test_measure_setup_and_last_warm_setup_reject(self):
        f=self.fixture();self.measured(f)["samples"][0]["sample"]["setup_observed"]=True;self.reject(f)
        f=self.fixture();f.objects["cost-b1-warm.json"]["runtime"]["samples"][-1]["sample"]["setup_observed"]=True;self.reject(f)

    def test_sequence_continues_across_inputs_and_slot_bound(self):
        for field,value in (("sequence",3),("workspace_slot",9),("input_label_sha256","0"*64),("activation_rows",1)):
            f=self.fixture();f.objects["cost-b1-input0001.json"]["runtime"]["samples"][0]["sample"][field]=value;self.reject(f)

    def test_journal_phase_stream_slot_and_ordinal_reject(self):
        for field,value in (("workspace_kind","numeric"),("stream_handle",999),("group_workspace_slot",1),("runtime_attempt",0)):
            f=self.fixture();f.events[1]["event"]["runtime"][field]=value;self.reject(f)
        f=self.fixture();f.calls[4]["phase"]="cost-after";self.reject(f)

    def test_fixed_counts_and_extra_calls_reject(self):
        for field,value in (("measurements",3),("warmup_once_for_batch",3),("forward_attempts_this_input",5)):
            f=self.fixture();self.measured(f)[field]=value;self.reject(f)
        f=self.fixture();self.measured(f)["samples"].pop();self.reject(f)
        f=self.fixture();f.plan["cost"][0]["input_indices"].append(0);self.reject(f)
        f=self.fixture();f.calls.append(copy.deepcopy(f.calls[-1]));self.reject(f)

    def test_binding_gpu_recipe_policy_changes_reject(self):
        for field,value in (("gpu_uuid_hex","f"*32),("recipe_sha256","f"*64),("driver_version",1)):
            f=self.fixture();self.measured(f)["samples"][0]["sample"]["binding"][field]=value;self.reject(f)
        f=self.fixture();self.measured(f)["samples"][0]["sample"]["binding"]["tactics"]["KATAGO_CUDA_DUALFFN"]="1";self.reject(f)

    def test_tail_wrong_packing_or_input_order_reject(self):
        for field,value in (("target_batch",3),("id","calibration-b1-row-0001")):
            f=self.fixture();f.objects["cost-b1-warm.json"]["input"][field]=value;self.reject(f)
        f=self.fixture();f.objects["cost-b3-input2044.json"]["input"]["rows"].reverse();self.reject(f)

    def test_measured_route_drift_and_raw_heads_reject(self):
        f=self.fixture();g=self.measured(f)["samples"][0]["sample"]["groups"][2]
        g["route"]["launches"][3].update(kind="hgemm_t64",layout="tn-half-input-f32-residual");self.reject(f)
        f=self.fixture();self.measured(f)["ordinary_after_heads"][0]["sha256"]="0"*64;self.reject(f)
        f=self.fixture();self.measured(f)["samples"][0]["bitwise_matches_before"]=False;self.reject(f)

    def test_artifact_source_hash_change_and_final_recheck(self):
        f=self.fixture();capture=f.capture();path=Path(capture.result["cost"][0]["path"]);original=path.read_bytes()
        path.write_bytes(original+b" ")
        with self.assertRaises(ValueError):m.summarize_costs(capture)
        path.write_bytes(original);m.summarize_costs(capture);path.write_bytes(original+b" ")
        with self.assertRaises(ValueError):capture.sources.recheck()

    def test_wrong_root_sha_refused_before_capture_load(self):
        path=self.root/"result.json";path.write_text("{}",encoding="utf-8")
        with self.assertRaises(ValueError):m.summarize_capture(path,"0"*64)

    def test_output_cannot_overwrite(self):
        directory=self.root/"report";m.write_report(directory,{"status":"SYNTHETIC"})
        first=(directory/"report.json").read_bytes()
        with self.assertRaises(ValueError):m.write_report(directory,{"status":"CHANGED"})
        self.assertEqual((directory/"report.json").read_bytes(),first)


if __name__ == "__main__":unittest.main(verbosity=2)
