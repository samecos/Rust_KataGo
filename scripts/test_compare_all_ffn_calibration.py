"""CPU-only boundary definitions; no real capture/model/GPU is needed."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import compare_all_ffn_calibration as d


class Boundaries(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.registration=self.source('registration.json')
        self.capture=self.source('capture.json')

    def source(self,name):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('{}',encoding='utf-8');return d.record(path)

    def aggregate(self):
        return dict(status='FULL_CAPTURE_METRICS_PENDING',sources_unchanged=True,error=None,source_error=None,
            retry_allowed=False,registration=self.registration,verified_cases=80,reserved_cases=80,
            receipts=[dict(name=f'b11-{n:03}' if n<34 else f'b15-{n-34:03}') for n in range(80)],
            verified_forwards=175760,observed_forward_attempts=175760,verified_physical_rows=212480)

    def pair(self):
        ref=dict(name='b11-000',family='b11',recipe_index=0,role='FP16_REFERENCE',group_ids=[],
            capture=self.capture,cost_summary=self.source('reference-cost.json'),model_sha256='a'*64,
            graph_sha256='b'*64,recipe_sha256='c'*64,output_contract_sha256='d'*64)
        candidate=dict(ref,name='b11-001',recipe_index=1,role='SINGLE_FFN_INT8_ABLATION',group_ids=['g.ffn'],
            capture=self.source('candidate.json'),cost_summary=self.source('candidate-cost.json'),recipe_sha256='e'*64)
        return dict(reference=ref,candidate=candidate)

    def report(self,pair):
        a,b=pair['reference'],pair['candidate']
        return dict(schema='rustgo-single-ffn-calibration-comparison-v1',status='COMPARED',sources_unchanged=True,
            coverage_mode='complete-calibration',unique_request_count=2044,placement_observations=2044,
            accuracy=dict(cases=2044),repeated_request_observations=0,reference=a['capture'],candidate=b['capture'],
            group_ids=b['group_ids'],model_sha256=a['model_sha256'],graph_sha256=a['graph_sha256'],
            output_contract_sha256=a['output_contract_sha256'],reference_recipe_sha256=a['recipe_sha256'],
            candidate_recipe_sha256=b['recipe_sha256'],calibration_6pp_observation='PASS',production_certified=False,
            performance_measured=False,bound_sources=[a['capture'],b['capture']])

    def test_full_aggregate_counts_and_first_failure_are_strict(self):
        value=self.aggregate();d._aggregate(value,self.registration)
        for key,changed in [('status','FAIL_STOPPED_NO_RETRY'),('verified_cases',79),('reserved_cases',81),
            ('observed_forward_attempts',175761),('verified_physical_rows',212481),('sources_unchanged',False),
            ('error',{'failed':'case'})]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                d._aggregate(dict(value,**{key:changed}),self.registration)

    def test_aggregate_rejects_duplicate_receipts_and_other_registration(self):
        value=self.aggregate();value['receipts'][-1]=value['receipts'][0]
        with self.assertRaises(ValueError):d._aggregate(value,self.registration)
        value=self.aggregate();value['registration']=self.capture
        with self.assertRaises(ValueError):d._aggregate(value,self.registration)

    def test_source_membership_cannot_rebind_changed_or_unlisted_output(self):
        inventory={d._key(self.capture):self.capture}
        d._member(self.capture,inventory)
        with self.assertRaises(ValueError):d._member(dict(self.capture,sha256='f'*64),inventory)
        with self.assertRaises(ValueError):d._member(self.registration,inventory)

    def test_numerical_threshold_failure_is_retained_as_observation(self):
        pair=self.pair();report=self.report(pair)
        authority={d._key(s):s for s in report['bound_sources']}
        d.check_report(report,pair,authority)
        report['calibration_6pp_observation']='FAIL'
        d.check_report(report,pair,authority)
        self.assertEqual(report['calibration_6pp_observation'],'FAIL')

    def test_report_wrong_pair_incomplete_or_certification_claim_rejected(self):
        pair=self.pair();report=self.report(pair)
        authority={d._key(s):s for s in report['bound_sources']}
        for key,changed in [('candidate',pair['reference']['capture']),('unique_request_count',2043),
            ('coverage_mode','bounded-diagnostic'),('production_certified',True),('performance_measured',True),
            ('group_ids',['wrong.ffn']),('candidate_recipe_sha256','f'*64)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                d.check_report(dict(report,**{key:changed}),pair,authority)

    def test_cost_summary_links_exact_capture_and_fixed_samples(self):
        pair=self.pair();case=pair['candidate'];indices=list(range(8))
        plan=dict(model={'sha256':'a'*64},graph_sha256='b'*64,resolved_recipe_sha256='e'*64,
            cost=[dict(physical_batch=b,input_indices=indices) for b in (1,3,8)])
        groups=[f'g{i}.ffn' for i in range(33)]
        report=dict(schema='rustgo-ffn-calibration-cost-summary-v1',status='SUMMARIZED_COST_OBSERVATIONS',
            sources_unchanged=True,coverage_mode='complete-calibration',end_to_end_performance_certified=False,
            numeric_accuracy_checked=False,numeric_chunks_revalidated=False,capture=case['capture'],group_ids=case['group_ids'],
            model_sha256='a'*64,graph_sha256='b'*64,recipe_sha256='e'*64,output_contract_sha256='d'*64,complete_ffn_groups=33,
            batches=[dict(physical_batch=b,warmup=3,measurements_per_input=4,sample_inputs=8,planned_forwards=51,
                groups=[dict(group={'id':g}) for g in groups],inputs=[dict(input_index=i,groups=[dict(group_id=g) for g in groups]) for i in indices]) for b in (1,3,8)])
        reg={'contracts':{'b11':'d'*64}};d._cost(report,case['capture'],case,plan,reg)
        for field,value in [('warmup',4),('sample_inputs',7),('measurements_per_input',2)]:
            changed=copy.deepcopy(report);changed['batches'][0][field]=value
            with self.assertRaises(ValueError):d._cost(changed,case['capture'],case,plan,reg)
        changed=dict(report,capture=pair['reference']['capture'])
        with self.assertRaises(ValueError):d._cost(changed,case['capture'],case,plan,reg)

    def test_publish_refuses_overwrite_and_keeps_first_bytes(self):
        path=self.root/'result.json';first=d.put(path,{'state':'first'})
        with self.assertRaises(FileExistsError):d.put(path,{'state':'second'})
        self.assertEqual(d.record(path),first)
        self.assertTrue(path.with_name(path.name+'.pending').exists())

    def test_failed_cpu_child_preserves_consumption_and_stops_before_next(self):
        pair=self.pair();second=copy.deepcopy(pair);second['candidate']['name']='b11-002'
        output=self.root/'new-cpu';observed=[]
        def fail(command,folder,name,*args):
            observed.append(name)
            self.assertIn('compare_ffn_calibration.py',command[2])
            self.assertNotIn('collect',command)
            self.assertTrue((folder/(name+'-consumed.json')).exists())
            self.assertFalse((folder/'b11-002-consumed.json').exists())
            raise ValueError('synthetic CPU failure')
        with patch.object(d,'admit',return_value=({},[pair,second],[])),patch.object(d,'run_child',side_effect=fail):
            with self.assertRaisesRegex(ValueError,'synthetic CPU failure'):
                d.compare_all(Path(self.registration['path']),self.registration['sha256'],Path(self.capture['path']),self.capture['sha256'],output)
        self.assertEqual(observed,['b11-001'])
        failure=json.loads((output/'FAILED.json').read_bytes())
        self.assertEqual(failure['status'],'FAILED_CPU_COMPARISON_STOPPED')
        self.assertEqual(failure['completed_comparisons'],[])
        self.assertFalse((output/'index.json').exists())
        self.assertEqual(failure['model_forwards'],0)
        with self.assertRaises(ValueError):
            d.compare_all(Path(self.registration['path']),self.registration['sha256'],Path(self.capture['path']),self.capture['sha256'],output)


if __name__=='__main__':unittest.main(verbosity=2)
