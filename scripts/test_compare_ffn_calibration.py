"""CPU-only synthetic captures; never reads a real model or runs CUDA."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

_spec = importlib.util.spec_from_file_location("metrics_adapter", Path(__file__).with_name("compare_ffn_calibration.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def varint(value):
    out = bytearray()
    while value >= 128:
        out.append((value & 127) | 128); value >>= 7
    out.append(value)
    return bytes(out)


def fbits(value, size):
    return struct.unpack("<I" if size == 4 else "<Q", struct.pack("<f" if size == 4 else "<d", value))[0]


def encode_pb(bits):
    raw = bytearray()
    for field in range(1, 13):
        if field in (1, 11):
            values = bits["policy_f32" if field == 1 else "ownership_f32"]
            if values:
                data = struct.pack("<"+"I"*len(values), *values)
                raw += varint((field << 3) | 2)+varint(len(data))+data
        elif 2 <= field <= 10:
            value = bits["scalars_f64"][field-2]
            if value not in (0, 1 << 63):
                raw += varint((field << 3) | 1)+struct.pack("<Q", value)
        elif bits["has_shortterm_error"]:
            raw += varint(field << 3)+b"\x01"
    return bytes(raw)


def output_bits(candidate=False):
    policy = [0.0]*362
    policy[0], policy[1], policy[2] = (0.59 if candidate else 0.6), -1.0, (0.41 if candidate else 0.4)
    return dict(policy_f32=[fbits(x, 4) for x in policy], ownership_f32=[fbits(0.125, 4)]*361,
                scalars_f64=[fbits(x, 8) for x in ((0.56 if candidate else 0.55), (0.44 if candidate else 0.45), 0.0, 1.0+2**-52, 3.0, -0.0, 0.0, 0.0, 0.0)],
                has_shortterm_error=True)


class Fixture:
    def __init__(self, root, diagnostic_repeat=False):
        self.root = Path(root)
        self.model = self.write("model.bin", b"synthetic model bytes, not a parser fixture")
        self.exe = self.write("collector.exe", b"synthetic executable identity only")
        self.dll = self.write("library.dll", b"synthetic DLL identity only")
        self.graph = "a"*64; self.contract = "b"*64
        self.recipes = []
        for candidate in (False, True):
            self.recipes.append(dict(schema="rustgo-precision-recipe", version=1, quantization_semantics_version=1,
                model_sha256=self.model["sha256"], graph_sha256=self.graph, projections=[dict(id="trunk.ffn."+suffix,
                expected_n=4, expected_k=4, precision="int8" if candidate else "fp16") for suffix in ("dual", "down")]))
        self.recipe_shas = [m.recipe_identity(r) for r in self.recipes]
        proposal = dict(schema="rustgo-single-ffn-calibration-recipe-proposal-v1", model_sha256=self.model["sha256"], graph_sha256=self.graph,
            recipes=[dict(role="SINGLE_FFN_INT8_ABLATION" if i else "FP16_REFERENCE", group_ids=["trunk.ffn"] if i else [],
                          recipe_sha256=self.recipe_shas[i], recipe=recipe) for i, recipe in enumerate(self.recipes)])
        self.proposal_object = m.sha(m.canonical(proposal, True)); proposal["proposal_sha256"] = self.proposal_object
        self.proposal = self.write("proposal.json", proposal)
        self.identities, requests, proofs = [], [], []
        self.spatial_rows, self.global_rows = [], []
        for index in range(3):
            s = struct.pack("<"+"f"*(22*361), *([1.0]*361+[index/4.0]*(21*361)))
            g = struct.pack("<"+"f"*19, *([index/4.0]*19))
            self.spatial_rows.append(s); self.global_rows.append(g)
            pb = self.write(f"corpus/requests/{index}.pb", f"synthetic request {index}".encode())
            local_pb = dict(pb, path=f"requests/{index}.pb")
            record = dict(name=f"request-{index}", game_id="c"*64, phase="early" if index == 0 else "late", ply=index,
                          semantic_position_sha256=m.sha(f"semantic-{index}".encode()), board_state_sha256=m.sha(f"board-{index}".encode()), split="calibration")
            common = dict(source_line=index+1, task_id=index+1, input_hash_hex=m.sha(f"input-{index}".encode()),
                          wire_semantic_sha256=m.sha(f"wire-{index}".encode()), source_line_sha256=m.sha(f"source-{index}".encode()))
            feature = m.sha(b"rustgo-worker-v7-nchw19-row-f32le-v1\0"+s+g)
            self.identities.append(dict(logical_row=index, pb_sha256=pb["sha256"], row_feature_sha256=feature,
                                       **common, **{k:v for k,v in record.items() if k != "split"}))
            requests.append(dict(pb=local_pb, record=record, **common))
            proofs.append(dict(pb=local_pb, record=record, post_feature_sha256=feature, post_spatial_sha256=m.sha(s), global_sha256=m.sha(g), **common))
        prepared = self.write("corpus/prepared.json", dict(split="calibration", model_sha256=self.model["sha256"], requests=requests))
        proof = self.write("corpus/proof.json", proofs)
        self.descriptors = []
        manifests = []
        selections = [(1,[0]), (3,[0,1,2])] if diagnostic_repeat else [(1,[0]), (1,[1])]
        for index,(target, members) in enumerate(selections):
            s = b"".join(self.spatial_rows[i] for i in members); g = b"".join(self.global_rows[i] for i in members)
            spatial = self.write(f"corpus/input-{index}/spatial.f32le", s)
            global_values = self.write(f"corpus/input-{index}/global.f32le", g)
            manifest = self.write(f"corpus/input-{index}/inputs.json", dict(schema="rustgo-encoded-group-cost-input-v1", purpose="calibration",
                inputs=[dict(id=f"input-{index}", physical_batch=len(members), spatial=dict(spatial,path="spatial.f32le"), **{"global":dict(global_values,path="global.f32le")})]))
            manifests.append(dict(manifest,path=f"input-{index}/inputs.json"))
            self.descriptors.append(dict(id=f"input-{index}", manifest_path=manifest["path"], manifest_sha256=manifest["sha256"], target_batch=target,
                physical_batch=len(members), tensor_sha256=m.sha(b"rustgo-encoded-group-cost-tensors-v1\0"+struct.pack("<Q",len(members))+s+g), rows=[self.identities[i] for i in members]))
        self.provenance = self.write("corpus/provenance.json", dict(source_split="calibration", model_sha256=self.model["sha256"],
            coverage_mode="prefix-diagnostic", encoded_count=3, prepared_manifest=dict(prepared,path="prepared.json"),
            request_provenance=dict(proof,path="proof.json"), input_manifests=manifests))
        self.source = [self.capture(False), self.capture(True)]

    def write(self, name, value):
        path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True)
        raw = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode()+b"\n"
        path.write_bytes(raw)
        return dict(path=str(path.resolve()), bytes=len(raw), sha256=m.sha(raw))

    def capture(self, candidate, mutate=None):
        index = int(candidate); name = "candidate" if candidate else "reference"
        recipe = self.recipes[index]
        recipe_source = self.write(name+"/recipe.json", recipe)
        plan = dict(schema="rustgo-ffn-calibration-collection-plan-v1", mode="bounded-diagnostic", model=self.model, graph_sha256=self.graph,
                    recipe=recipe_source, proposal=self.proposal, proposal_object_sha256=self.proposal_object,
                    resolved_recipe_sha256=self.recipe_shas[index], group_ids=["trunk.ffn"] if candidate else [],
                    provenance=self.provenance, numeric_input_indices=list(range(len(self.descriptors))), expected_forwards=len(self.descriptors),
                    expected_numeric_rows=sum(d["physical_batch"] for d in self.descriptors), chunk_inputs=128, cost=[])
        calls = [dict(input_index=i, phase="numeric", iteration=0, physical_batch=d["physical_batch"], tensor_sha256=d["tensor_sha256"]) for i,d in enumerate(self.descriptors)]
        plan_source = self.write(name+"/plan.json", plan)
        sources = [plan_source, self.exe, self.model, recipe_source, self.proposal, self.provenance]
        intent = dict(schema="rustgo-ffn-calibration-attempt-v1", mode="collect", planned_uploads=1, retry_allowed=False,
                      sources=sources, plan=plan, compiled_sources={"adapter":"d"*64}, calls=calls, output_contract_sha256=self.contract)
        runtime = dict(schema="rustgo-ffn-calibration-runtime-v1", model_uploads=1, implicit_warmup=0, graph_capture=0, graph_replay=0,
            automatic_retries=0, raw_heads_prefilled_nan_each_forward=True, executable=self.exe, libraries=[self.dll],
            device={"gpu_name":"SYNTHETIC"}, driver_version=0, backend_build="SYNTHETIC", effective_tactics={"SYNTHETIC":"1"},
            native=dict(model_sha256=self.model["sha256"], graph_sha256=self.graph, resolved_recipe_sha256=self.recipe_shas[index],
                source_recipe_sha256=recipe_source["sha256"], projections=[dict(projection=dict(id=p["id"],n=p["expected_n"],k=p["expected_k"]), precision=p["precision"]) for p in recipe["projections"]]))
        raw_blob=bytearray(); pb_blob=bytearray(); entries=[]
        for i,d in enumerate(self.descriptors):
            heads=[]; row_raw=[bytearray() for _ in d["rows"]]
            for head_name,width in m.HEADS:
                values=[0.0]*(width*d["physical_batch"])
                if candidate and head_name == "value": values[0]=0.125
                data=struct.pack("<"+"f"*len(values), *values)
                heads.append(dict(head=head_name,row_width=width,slice=dict(offset=len(raw_blob),bytes=len(data),sha256=m.sha(data))))
                raw_blob+=data
                for r in range(d["physical_batch"]): row_raw[r]+=data[r*width*4:(r+1)*width*4]
            rows=[]
            for r,identity in enumerate(d["rows"]):
                bits=output_bits(candidate); pb=encode_pb(bits)
                rows.append(dict(row=r,identity=identity,raw_row_sha256=m.sha(row_raw[r]),
                    pb=dict(offset=len(pb_blob),bytes=len(pb),sha256=m.sha(pb)),bits_stage=m.BITS_STAGE,bits=bits,wire_bits=m.decode_pb(pb)))
                pb_blob+=pb
            entries.append(dict(input_index=i,input=d,tensor_sha256=d["tensor_sha256"],heads=heads,rows=rows))
        if mutate: mutate(plan,intent,runtime,entries,raw_blob,pb_blob)
        # Mutations are intentionally rebound to exercise admission rather than only outer SHA checks.
        plan_source=self.write(name+"/plan.json",plan);sources[0]=plan_source
        raw_source=self.write(name+"/raw.f32le",bytes(raw_blob));pb_source=self.write(name+"/pb.bin",bytes(pb_blob))
        chunk=self.write(name+"/chunk.json",dict(schema="rustgo-ffn-calibration-output-chunk-v1",raw=raw_source,pb=pb_source,inputs=entries))
        journal=[]
        for ordinal,call in enumerate(calls):
            event=dict(**{k:v for k,v in call.items() if k != "input_index"},runtime_attempt=ordinal+1,
                       model_sha256=self.model["sha256"],graph_sha256=self.graph,recipe_sha256=self.recipe_shas[index])
            journal.append(m.canonical(dict(ordinal=ordinal,event=dict(expected=call,runtime=event)))+b"\n")
        result=dict(status="CAPTURE_COMPLETE_METRICS_PENDING",sources_unchanged=True,gpu_model_uploads=1,
            model_sha256=self.model["sha256"],graph_sha256=self.graph,recipe_sha256=self.recipe_shas[index],output_contract_sha256=self.contract,
            sources=sources,intent=self.write(name+"/intent.json",intent),load_attempt=self.write(name+"/load.json",dict(model=self.model,recipe=recipe_source,ordinal=0)),
            runtime=self.write(name+"/runtime.json",runtime),journal=self.write(name+"/journal.jsonl",b"".join(journal)),cost=[],chunks=[chunk],
            consumed_forwards=len(calls),numeric_inputs=len(self.descriptors),numeric_rows=plan["expected_numeric_rows"])
        return self.write(name+"/result.json",result)

    def compare(self):
        return m.compare(self.source[0]["path"],self.source[0]["sha256"],self.source[1]["path"],self.source[1]["sha256"])


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="rustgo-calibration-metrics-")
        self.addCleanup(self.temp.cleanup)
        self.fixture=Fixture(self.temp.name)

    def test_real_pure_functions_and_synthetic_bound_capture(self):
        report=self.fixture.compare()
        self.assertEqual(report["accuracy"]["cases"],2)
        self.assertAlmostEqual(report["win_probability_max_abs_pp"],1.0)
        self.assertGreater(report["accuracy"]["policy_kl_mean"],0.0)
        self.assertEqual(report["raw_heads"]["policy"]["bitwise_equal_ratio"],1.0)
        self.assertEqual(report["raw_heads"]["value"]["max_abs"],0.125)
        self.assertEqual(report["calibration_6pp_observation"],"NOT_FULL_CALIBRATION")
        self.assertNotIn("diagnostic_within_6pp",report)
        self.assertFalse(report["production_certified"])
        self.assertEqual(report["by_group"]["phase"]["early"]["accuracy"]["cases"],1)
        self.assertAlmostEqual(report["by_group"]["phase"]["early"]["win_probability_max_abs_pp"],1.0)

    def test_wire_minus_zero_elision_only_and_f64_epsilon_survives(self):
        bits=output_bits(); raw=encode_pb(bits); wire=m.decode_pb(raw)
        self.assertEqual(bits["scalars_f64"][5],1<<63)
        self.assertEqual(wire["scalars_f64"][5],0)
        self.assertEqual(wire["scalars_f64"][3],fbits(1+2**-52,8))
        row=dict(bits_stage=m.BITS_STAGE,bits=bits,wire_bits=wire)
        m.validate_bits(row,raw)
        row["bits"]["scalars_f64"][3]=fbits(1.0,8)
        with self.assertRaises(ValueError): m.validate_bits(row,raw)

    def test_canonical_wire_rejects_unknown_duplicate_truncated_and_default_scalar(self):
        raw=encode_pb(output_bits())
        for malformed in (raw+b"\x60\x01",raw+b"\x68\x01",raw[:-1],raw+b"\x11"+b"\0"*8,b"\x8a\x00"):
            with self.subTest(malformed=malformed[-10:]),self.assertRaises(ValueError): m.decode_pb(malformed)

    def test_nonfinite_typed_and_probability_or_ownership_fail(self):
        for key,index,value in (("scalars_f64",0,fbits(float("nan"),8)),("policy_f32",0,fbits(0.2,4)),("ownership_f32",0,fbits(2,4))):
            bits=output_bits();bits[key][index]=value
            with self.assertRaises(ValueError): m.output_from_bits(bits)

    def test_missing_input_is_not_partial_success(self):
        self.fixture.source[1]=self.fixture.capture(True,lambda p,i,r,e,raw,pb:e.pop())
        with self.assertRaises(ValueError): self.fixture.compare()

    def test_wrong_placement_and_rebound_raw_row_sha_rejected(self):
        def mutate(p,i,r,e,raw,pb):e[0]["rows"][0]["row"]=1
        self.fixture.source[1]=self.fixture.capture(True,mutate)
        with self.assertRaises(ValueError): self.fixture.compare()

    def test_input_feature_binding_cannot_follow_wrong_row(self):
        def mutate(p,i,r,e,raw,pb):
            e[0]["input"]=copy.deepcopy(e[0]["input"])
            e[0]["input"]["rows"][0]["row_feature_sha256"]="e"*64
            e[0]["rows"][0]["identity"]=e[0]["input"]["rows"][0]
        self.fixture.source[1]=self.fixture.capture(True,mutate)
        with self.assertRaises(ValueError):self.fixture.compare()

    def test_reference_must_be_all_fp16(self):
        with self.assertRaises(ValueError):m.compare(self.fixture.source[1]["path"],self.fixture.source[1]["sha256"],self.fixture.source[0]["path"],self.fixture.source[0]["sha256"])

    def test_complete_cannot_be_two_rows_relabelled(self):
        self.fixture.source[1]=self.fixture.capture(True,lambda p,i,r,e,raw,pb:p.update(mode="complete-calibration"))
        with self.assertRaises(ValueError):self.fixture.compare()

    def test_missing_or_overlapping_raw_slice_rejected(self):
        self.fixture.source[1]=self.fixture.capture(True,lambda p,i,r,e,raw,pb:e[0]["heads"][1]["slice"].update(offset=0))
        with self.assertRaises(ValueError):self.fixture.compare()

    def test_nonfinite_unused_raw_channel_rejected(self):
        def mutate(p,i,r,e,raw,pb):
            raw[1000*4:1000*4+4]=struct.pack("<f",float("nan"))
            head=e[0]["heads"][0];head["slice"]["sha256"]=m.sha(raw[:2172*4])
        self.fixture.source[1]=self.fixture.capture(True,mutate)
        with self.assertRaises(ValueError):self.fixture.compare()

    def test_changed_bound_source_fails_without_rebinding(self):
        Path(self.fixture.model["path"]).write_bytes(b"changed")
        with self.assertRaises(ValueError):self.fixture.compare()

    def test_source_recheck_rejects_late_same_size_mutation(self):
        sources=m.Sources();sources.add(self.fixture.dll)
        path=Path(self.fixture.dll["path"]);raw=path.read_bytes();path.write_bytes(bytes([raw[0]^1])+raw[1:])
        with self.assertRaises(ValueError):sources.recheck()

    def test_diagnostic_cross_packing_reports_placements_not_semantic_gate(self):
        with tempfile.TemporaryDirectory(prefix="rustgo-calibration-metrics-repeat-") as directory:
            report=Fixture(directory,diagnostic_repeat=True).compare()
        self.assertEqual(report["placement_observations"],4)
        self.assertEqual(report["unique_request_count"],3)
        self.assertEqual(report["repeated_request_observations"],1)
        self.assertEqual(report["by_group"]["target_batch"]["1"]["accuracy"]["cases"],1)
        self.assertEqual(report["by_group"]["target_batch"]["3"]["accuracy"]["cases"],3)
        self.assertEqual(report["calibration_6pp_observation"],"NOT_FULL_CALIBRATION")

    @unittest.skipUnless(os.name == "nt", "actual Windows extended DOS path")
    def test_rust_extended_dos_manifest_and_python_normal_provenance_are_same_file(self):
        for descriptor in self.fixture.descriptors:
            descriptor["manifest_path"] = "\\\\?\\"+descriptor["manifest_path"]
        self.fixture.source = [self.fixture.capture(False), self.fixture.capture(True)]
        self.assertEqual(self.fixture.compare()["accuracy"]["cases"], 2)
        self.assertEqual(m.resolved_path(self.fixture.model["path"]), m.resolved_path("\\\\?\\"+self.fixture.model["path"]))

    def test_existing_near_tie_is_annotation_not_relaxed_top1(self):
        sources=m.Sources();quant,compare,_=m.load_metrics(sources)
        a=m.output_from_bits(output_bits());b=copy.deepcopy(a)
        a["policy"]=[0.0]*362;b["policy"]=[0.0]*362
        a["policy"][0],a["policy"][1]=0.5001,0.4999
        b["policy"][0],b["policy"][1]=0.4999,0.5001
        left=dict(results=[dict(name="tie",result=dict(output=a))]);right=dict(results=[dict(name="tie",result=dict(output=b))])
        self.assertEqual(quant(left,right)["top1_matches"],0)
        actual=compare(left,right)
        self.assertEqual(actual["policy_top1_mismatches_near_ties"],1)
        self.assertEqual(actual["result"],"FAIL")

    def test_negative_policy_legality_difference_rejected_before_metrics(self):
        def mutate(p,i,r,e,raw,pb):
            bits=e[0]["rows"][0]["bits"];bits["policy_f32"][1]=fbits(0.0,4)
            original=e[0]["rows"][0]["pb"];replacement=encode_pb(bits)
            self.assertEqual(len(replacement),original["bytes"])
            pb[:len(replacement)]=replacement
            original["sha256"]=m.sha(replacement);e[0]["rows"][0]["wire_bits"]=m.decode_pb(replacement)
        self.fixture.source[1]=self.fixture.capture(True,mutate)
        with self.assertRaisesRegex(ValueError,"policy legality differs"):self.fixture.compare()


if __name__ == "__main__":
    unittest.main(verbosity=2)
