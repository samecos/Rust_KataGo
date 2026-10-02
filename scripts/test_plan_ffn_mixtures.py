"""Bounded CPU test definitions; synthetic metadata, no model/Worker/GPU."""
import copy
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import plan_ffn_mixtures as p


def template(count=3,model='a'):
    entries=[]
    for n in range(count):
        stem=f'trunk.block00.pair{n:02}'
        for suffix,width,k in (('attention.qkv',1152,384),('attention.out',384,384),
                               ('ffn.dual',136,384),('ffn.down',384,136)):
            entries.append(dict(id=stem+'.'+suffix,expected_n=width,expected_k=k,precision='fp16'))
    return dict(schema='rustgo-precision-recipe',version=1,quantization_semantics_version=1,
                model_sha256=model*64,graph_sha256='c'*64,projections=entries)


def report(win=0.01):
    return dict(accuracy=dict(win_probability_max_abs=win,win_probability_mean_abs=win/2,
        policy_kl_mean=0.02,policy_kl_max=0.1,score_mean_max_abs=1.0,score_mean_mean_abs=0.2,top1_matches=2000),
        win_probability_max_abs_pp=win*100,win_probability_mean_abs_pp=win*50,
        calibration_6pp_observation='PASS' if win <= 0.06 else 'FAIL')


def models(risks=(1.0,2.0,4.0),savings=(1.0,1.0,1.0)):
    result={}
    for family,identity in (('b11','a'),('b15','b')):
        recipe=template(len(risks),identity);groups=p.recipe_groups(recipe,len(risks));evidence=[]
        for group,risk,saving in zip(groups,risks,savings,strict=True):
            evidence.append(dict(group_id=group,sensitivity=dict(win_max_abs_pp=risk,policy_kl_mean=0.1,score_mean_max_abs=1.0),
                numeric_eligible=True,capture={'synthetic':group},costs={b:dict(cost_eligible=True,saved_target_group_ms=saving) for b in p.BATCHES}))
        result[family]=dict(recipe=recipe,groups=groups,evidence=evidence,reference={'synthetic':'reference'})
    return result


def manifest_fixture():
    recipe=p.compose_recipe(template(1),[])
    entry=dict(role='FP16_REFERENCE',group_ids=[],recipe=recipe,recipe_sha256=p.recipe_sha(recipe),
        recipe_object_sha256=p.object_sha(recipe),recipe_source=dict(path='D:/模型/配方.json',bytes=100,sha256='d'*64),
        aliases=[dict(physical_batch=1,risk_proxy_budget_pp=1.5,risk_proxy_used_pp=0.0)],
        reuse_capture={'not_part_of_admission':True},status='REUSE_OBSERVED_FP16_REFERENCE')
    return dict(family='b11',model_sha256=recipe['model_sha256'],graph_sha256=recipe['graph_sha256'],
        output_contract_sha256='e'*64,reference_recipe_sha256=p.recipe_sha(recipe),recipes=[entry],
        groups=[dict(cost_ms=0.0000000123,sensitivity=0.12345678901234567)])


def cost_pair(candidate_means=None):
    group='trunk.block00.pair00.ffn';candidate_means=candidate_means or [0.5]*8
    pair=[]
    for precision,means,identity in (('fp16',[1.0]*8,'d'),('int8',candidate_means,'e')):
        binding=dict(schema='rustgo-group-cost',version=1,mode='DIRECT_DIAGNOSTIC',
            model_sha256='a'*64,graph_sha256='b'*64,recipe_sha256=identity*64,executable_sha256='f'*64,
            backend_build={'synthetic':1},gpu_name='fixture',gpu_uuid_hex='00',compute_capability='12.0',
            sm_count=1,driver_version=1,tactics={'KATAGO_CUDA_DUALFFN':'0'})
        descriptor=dict(id=group,rms_layer=4,ffn_layer=5,end_layer=7,mid=384,hidden=136,
            dual=p.costs.projection(272,384,precision),down=p.costs.projection(384,136,precision))
        inputs=[]
        for n,mean in enumerate(means):
            bits=struct.unpack('<I',struct.pack('<f',mean))[0]
            values=[mean]*4
            inputs.append(dict(input_index=n,input=dict(index=n,tensor_sha256=f'{n:064x}',rows=[{'pb_sha256':f'{n+10:064x}'}]),
                source={'synthetic':str(n)},groups=[dict(group_id=group,elapsed_ms_bits=[bits]*4,elapsed_ms=values,
                    statistics=p.costs.statistics(values))]))
        batches=[dict(physical_batch=b,warmup=3,sample_inputs=8,measurements_per_input=4,planned_forwards=51,
            binding=copy.deepcopy(binding),inputs=copy.deepcopy(inputs),
            groups=[dict(group=copy.deepcopy(descriptor),route={'synthetic':precision},equal_input_mean_ms=math.fsum(means)/8)]) for b in p.BATCHES]
        pair.append(dict(model_sha256='a'*64,graph_sha256='b'*64,recipe_sha256=identity*64,batches=batches))
    return pair,group


class MixtureBoundaries(unittest.TestCase):
    def test_reference_complete_real_attention_names_and_non16_hidden(self):
        recipe=template();groups=p.recipe_groups(recipe,3)
        self.assertEqual(len(groups),3)
        bad=copy.deepcopy(recipe);bad['projections'][0]['id']=bad['projections'][0]['id'].replace('.attention.','.attn.')
        with self.assertRaises(ValueError):p.recipe_groups(bad,3)
        bad=copy.deepcopy(recipe);bad['projections'].pop(0)
        with self.assertRaises(ValueError):p.recipe_groups(bad,3)

    def test_recipe_composition_preserves_all_shapes_attention_and_input(self):
        recipe=template();before=copy.deepcopy(recipe);group=p.recipe_groups(recipe,3)[1]
        got=p.compose_recipe(recipe,[group])
        self.assertEqual(recipe,before)
        self.assertEqual({v['id'] for v in got['projections'] if v['precision']=='int8'},{group+'.dual',group+'.down'})
        self.assertEqual([(v['id'],v['expected_n'],v['expected_k']) for v in got['projections']],
                         sorted((v['id'],v['expected_n'],v['expected_k']) for v in recipe['projections']))
        with self.assertRaises(ValueError):p.compose_recipe(recipe,[group,group])

    def test_ordered_recipe_sha_is_not_sorted_object_hash(self):
        recipe=template(1);got=p.compose_recipe(recipe,[])
        expected=dict(schema='rustgo-precision-recipe',version=1,quantization_semantics_version=1,
            model_sha256='a'*64,graph_sha256='c'*64,projections=[dict(id=v['id'],expected_n=v['expected_n'],
                expected_k=v['expected_k'],precision='fp16') for v in sorted(recipe['projections'],key=lambda v:v['id'])])
        self.assertEqual(p.recipe_sha(got),p.io.sha(p.io.canonical(expected)))
        self.assertNotEqual(p.recipe_sha(got),p.object_sha(got))
        reverse=dict(reversed(list(recipe.items())));reverse['projections']=list(reversed(recipe['projections']))
        self.assertEqual(p.recipe_sha(reverse),p.recipe_sha(got))

    def test_probability_units_threshold_and_nonfinite_rejected(self):
        observed,eligible=p.sensitivity(report(0.06));self.assertTrue(eligible);self.assertEqual(observed['win_max_abs_pp'],6.0)
        self.assertFalse(p.sensitivity(report(0.060001))[1])
        bad=report();bad['win_probability_max_abs_pp']=0.01
        with self.assertRaises(ValueError):p.sensitivity(bad)
        bad=report();bad['calibration_6pp_observation']='FAIL'
        with self.assertRaises(ValueError):p.sensitivity(bad)
        for value in (float('nan'),float('inf'),-0.1,True):
            bad=report();bad['accuracy']['policy_kl_mean']=value
            with self.assertRaises(ValueError):p.sensitivity(bad)

    def test_cost_six_of_eight_and_mean_both_required(self):
        pair,g=cost_pair([0.5]*6+[1.5]*2);obs=p.target_cost_pair(*pair,g,1)
        self.assertTrue(obs['cost_eligible']);self.assertEqual(obs['input_wins'],6)
        pair,g=cost_pair([0.5]*5+[1.25]*3)
        self.assertFalse(p.target_cost_pair(*pair,g,1)['cost_eligible'])
        pair,g=cost_pair([0.5]*6+[4.0]*2)
        self.assertFalse(p.target_cost_pair(*pair,g,1)['cost_eligible'])
        pair,g=cost_pair([1.0]*8)
        self.assertFalse(p.target_cost_pair(*pair,g,1)['cost_eligible'])

    def test_cost_allows_precision_stride_change_but_rejects_identity_change(self):
        pair,g=cost_pair();p.target_cost_pair(*pair,g,8)
        self.assertEqual(pair[0]['batches'][2]['groups'][0]['group']['down']['input_stride'],144)
        self.assertEqual(pair[1]['batches'][2]['groups'][0]['group']['down']['input_stride'],136)
        for mutate in (lambda b:b['binding'].update(driver_version=2),
                       lambda b:b['inputs'][0]['input'].update(tensor_sha256='0'*63+'1'),
                       lambda b:b['groups'][0]['group'].update(end_layer=6),
                       lambda b:b['groups'][0]['group']['down'].update(input_stride=144),
                       lambda b:b['inputs'][0]['groups'][0]['elapsed_ms_bits'].__setitem__(0,0)):
            bad=copy.deepcopy(pair);mutate(bad[1]['batches'][2])
            with self.assertRaises(ValueError):p.target_cost_pair(*bad,g,8)

    def test_budget_greedy_skips_oversized_first_item_and_continues(self):
        data=models((2.0,1.0),(100.0,1.0));_,aliases=p.nominate(data)
        first=aliases[0]
        self.assertEqual(first['selected_groups'],[data['b11']['groups'][1]])
        self.assertEqual([d['selected'] for d in first['decisions']],[False,True])
        self.assertEqual(first['risk_proxy_used_pp'],1.0)

    def test_fixed_budget_equality_and_zero_floor_do_not_charge_artificial_risk(self):
        data=models((0.0,1.5),(1.0,1.0));_,aliases=p.nominate(data)
        self.assertEqual(aliases[0]['selected_groups'],data['b11']['groups'])
        self.assertEqual(aliases[0]['risk_proxy_used_pp'],1.5)
        self.assertEqual(aliases[0]['decisions'][0]['rank_score'],100.0)

    def test_deterministic_tie_break_and_cross_batch_recipe_dedup(self):
        data=models((1.0,1.0),(1.0,1.0));data['b11']['evidence'].reverse()
        candidates,aliases=p.nominate(data)
        self.assertEqual(aliases[0]['selected_groups'],[data['b11']['groups'][0]])
        self.assertEqual(len(aliases),18);self.assertEqual(len(candidates),4)
        for c in candidates:self.assertIn(len(c['aliases']),(3,6))
        self.assertNotEqual(candidates[0]['recipe_sha256'],candidates[2]['recipe_sha256'])

    def test_empty_reuses_reference_single_reuses_capture_never_authorizes_calls(self):
        data=models((1.0,),(1.0,))
        for e in data['b11']['evidence']:e['numeric_eligible']=False
        candidates,aliases=p.nominate(data)
        self.assertTrue(all(a['role']=='FP16_REFERENCE' for a in aliases[:9]))
        self.assertEqual(len(candidates),1)
        candidate=candidates[0]
        self.assertEqual(candidate['role'],'SINGLE_FFN_INT8_ABLATION')
        self.assertEqual(candidate['reuse_capture'],data['b15']['evidence'][0]['capture'])
        self.assertEqual(candidate['new_gpu_calls_authorized'],0)
        self.assertTrue(candidate['whole_graph_accuracy_unverified'] and candidate['performance_unverified'])

    def test_cost_eligibility_is_per_physical_batch(self):
        data=models((1.0,),(1.0,));data['b11']['evidence'][0]['costs'][3]['cost_eligible']=False
        _,aliases=p.nominate(data)
        self.assertEqual([a['role']=='FP16_REFERENCE' for a in aliases[:9]],[False]*3+[True]*3+[False]*3)

    def test_complete_index_rejects_missing_wrong_order_and_certified_claim(self):
        names=[f'{f}-{n:03}' for f,count in p.FAMILIES.items() for n in range(1,count+1)]
        index=dict(schema='rustgo-full-ffn-calibration-comparison-index-v1',status='COMPLETE_CALIBRATION_COMPARISONS',
            sources_unchanged=True,reference_captures=2,candidate_comparisons=78,requests_per_comparison=2044,
            adopted=False,production_certified=False,model_forwards=0,gpu_calls=0,worker_calls=0,
            comparisons=[dict(name=n,ordinal=i) for i,n in enumerate(names)],
            all_case_cost_summaries=[dict(name=f'{f}-{n:03}') for f,count in p.FAMILIES.items() for n in range(count+1)])
        p.check_index(index)
        for change in (dict(candidate_comparisons=77),dict(production_certified=True),dict(status='RUNNING')):
            with self.assertRaises(ValueError):p.check_index(dict(index,**change))
        bad=copy.deepcopy(index);bad['comparisons'].reverse()
        with self.assertRaises(ValueError):p.check_index(bad)

    def test_source_membership_rejects_rebound_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'source.json';path.write_text('{}',encoding='utf-8');source=p.record(path)
            authority=p.directory([source]);sources=p.io.Sources()
            self.assertEqual(p.read_bound(source,authority,sources),{})
            with self.assertRaises(ValueError):p.read_bound(dict(source,sha256='0'*64),authority,sources)
            path.write_text('{"changed":true}',encoding='utf-8')
            with self.assertRaises(ValueError):p.read_bound(source,authority,sources)

    def test_metadata_failure_leaves_first_failure_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);index=root/'index.json';index.write_text('{}',encoding='utf-8');source=p.record(index)
            output=root/'attempt'
            with patch.object(p,'load_evidence',side_effect=ValueError('first metadata failure')),self.assertRaises(ValueError):
                p.plan(index,source['sha256'],output)
            failure=p.io.decode_json((output/'FAILED.json').read_bytes())
            self.assertEqual(failure['error']['message'],'first metadata failure')
            self.assertFalse((output/'proposals.json').exists())
            before=(output/'FAILED.json').read_bytes()
            with self.assertRaises(ValueError):p.plan(index,source['sha256'],output)
            self.assertEqual((output/'FAILED.json').read_bytes(),before)

    def test_cached_serializer_cannot_bind_a_changed_helper_as_current_source(self):
        p.recipe_function()  # Populate the production cache before another plan().
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);index=root/'index.json';index.write_text('{}',encoding='utf-8');source=p.record(index)
            changed=root/'helper.py';changed.write_text('# different helper',encoding='utf-8')
            with patch.object(p,'RECIPE_HELPER',changed),patch.object(p,'load_evidence') as read,self.assertRaises(ValueError):
                p.plan(index,source['sha256'],root/'attempt')
            read.assert_not_called()

    def test_admission_is_exact_no_float_projection_not_full_float_manifest(self):
        manifest=manifest_fixture();admission=p.admission_projection(manifest)
        self.assertEqual(set(admission),{'schema','recipes',*p.ADMISSION_HEADER})
        self.assertEqual(set(admission['recipes'][0]),set(p.ADMISSION_RECIPE))
        original=p.io.sha(p.admission_bytes(admission));whole=p.object_sha(manifest)
        manifest['groups'][0]['cost_ms']=0.1
        manifest['recipes'][0]['aliases'][0]['risk_proxy_budget_pp']=3.0
        self.assertEqual(p.io.sha(p.admission_bytes(p.admission_projection(manifest))),original)
        self.assertNotEqual(p.object_sha(manifest),whole)
        manifest['recipes'][0]['recipe_source']['sha256']='f'*64
        self.assertNotEqual(p.io.sha(p.admission_bytes(p.admission_projection(manifest))),original)
        self.assertEqual(admission['recipes'][0]['recipe_source']['sha256'],'d'*64)

    def test_admission_rejects_nested_floats_nonstring_keys_and_out_of_u64(self):
        for value in (0.0,-0.0,1e-20,float('nan'),float('inf'),-1,2**64,(),b'bytes'):
            with self.subTest(value=value),self.assertRaises(ValueError):p.admission_bytes({'nested':[{'value':value}]})
        for key in (0,True,None):
            with self.subTest(key=key),self.assertRaises(ValueError):p.admission_bytes({'nested':{key:'value'}})
        self.assertEqual(p.admission_bytes([None,True,False,0,2**64-1]),b'[null,true,false,0,18446744073709551615]')

    def test_admission_compact_utf8_chinese_paths_and_json_escaping(self):
        value={'z':1,'a':'D:\\模型\\配方.json'}
        expected='{"a":"D:\\\\模型\\\\配方.json","z":1}'.encode('utf-8')
        self.assertEqual(p.admission_bytes(value),expected)
        self.assertNotIn(b'\\u',p.admission_bytes(value))
        self.assertEqual(p.admission_bytes({'b':'quote"\n','a':True}),b'{"a":true,"b":"quote\\"\\n"}')

    def test_admission_preserves_boolean_kind_but_rejects_boolean_numeric_fields(self):
        self.assertNotEqual(p.admission_bytes({'value':True}),p.admission_bytes({'value':1}))
        manifest=manifest_fixture();manifest['recipes'][0]['recipe_source']['bytes']=True
        with self.assertRaises(ValueError):p.admission_projection(manifest)
        manifest=manifest_fixture();manifest['recipes'][0]['recipe']['projections'][0]['expected_n']=True
        with self.assertRaises(ValueError):p.admission_projection(manifest)


if __name__ == '__main__':unittest.main()
