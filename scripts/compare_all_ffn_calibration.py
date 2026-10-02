#!/usr/bin/env python3
"""Compare 78 frozen single-FFN candidates on the already captured 2,044 rows.

No collector, model, GPU or Worker is launched. Existing reference captures and
cost summaries are reused read-only. A numerical threshold FAIL is retained as
an observation; missing/changed artifacts or failed CPU children stop the run.
This file is a separate CPU revision, outside the GPU run's source inventory.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

import compare_ffn_calibration as io
# These frozen helpers validate metadata and supervise the one supplied child.
# execute(), child_environment() and all inference entry points are never called.
from run_ffn_calibration import record, same_source, validate_registration, run_child

ROOT = Path(__file__).resolve().parents[1]
COMPARATOR = ROOT/'scripts/compare_ffn_calibration.py'
PER_PAIR_TIMEOUT = 900
TOTAL_TIMEOUT = 43200
COUNTS = {'b11':34, 'b15':46}
require = io.require


def put(path, value):
    """Write+sync pending, then publish without replacement; retain pending."""
    path = Path(path)
    pending = path.with_name(path.name+'.pending')
    with pending.open('xb') as f:
        f.write(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2).encode()+b'\n')
        f.flush();os.fsync(f.fileno())
    os.link(pending,path)
    return record(path)


def _key(source):
    return os.path.normcase(str(io.resolved_path(source['path'])))


def _member(source, inventory):
    key = _key(source)
    require(key in inventory and same_source(source,inventory[key]), 'Source not bound by capture/registration: '+str(source['path']))


def _aggregate(value, registration_source):
    require(value['status']=='FULL_CAPTURE_METRICS_PENDING' and value['sources_unchanged'] is True
            and value['error'] is None and value['source_error'] is None and value['retry_allowed'] is False,
            'full capture is incomplete or failed')
    require(same_source(value['registration'],registration_source), 'aggregate belongs to another registration')
    require(value['verified_cases']==value['reserved_cases']==len(value['receipts'])==80
            and value['verified_forwards']==value['observed_forward_attempts']==175760
            and value['verified_physical_rows']==212480, 'aggregate full counts differ')
    require(len({r['name'] for r in value['receipts']})==80, 'duplicate capture receipt')


def _capture(value, source, case, plan, reg):
    require(value['status']=='CAPTURE_COMPLETE_METRICS_PENDING' and value['sources_unchanged'] is True
            and value['numeric_inputs']==value['numeric_rows']==2044
            and value['consumed_forwards']==2197 and value['gpu_model_uploads']==1, 'case incomplete')
    for key,expected in [('model_sha256',plan['model']['sha256']),('graph_sha256',plan['graph_sha256']),
                         ('recipe_sha256',plan['resolved_recipe_sha256']),('output_contract_sha256',reg['contracts'][case['family']])]:
        require(value[key]==expected, 'capture identity differs: '+key)
    expected=[case['plan'],reg['executable'],plan['model'],plan['recipe'],plan['proposal'],plan['provenance']]
    require(len(value['sources'])==len(expected)
            and all(same_source(a,b) for a,b in zip(value['sources'],expected,strict=True)), 'capture source identity differs')
    require(io.resolved_path(source['path'])==io.resolved_path(Path(reg['execution_output'])/case['name']/'result.json'), 'capture is outside frozen case directory')


def _cost(value, capture, case, plan, reg):
    require(value['schema']=='rustgo-ffn-calibration-cost-summary-v1'
            and value['status']=='SUMMARIZED_COST_OBSERVATIONS' and value['sources_unchanged'] is True
            and value['coverage_mode']=='complete-calibration'
            and value['end_to_end_performance_certified'] is False
            and value['numeric_accuracy_checked'] is False and value['numeric_chunks_revalidated'] is False,
            'cost summary scope/status differs')
    require(same_source(value['capture'],capture) and value['group_ids']==case['group_ids'], 'cost summary capture/group differs')
    for key,expected in [('model_sha256',plan['model']['sha256']),('graph_sha256',plan['graph_sha256']),
                         ('recipe_sha256',plan['resolved_recipe_sha256']),('output_contract_sha256',reg['contracts'][case['family']])]:
        require(value[key]==expected,'cost summary identity differs: '+key)
    require(value['complete_ffn_groups']==COUNTS[case['family']]-1 and len(value['batches'])==3,'cost group/batch directory incomplete')
    for batch,expected in zip(value['batches'],plan['cost'],strict=True):
        require(batch['physical_batch']==expected['physical_batch'] and batch['warmup']==3
                and batch['measurements_per_input']==4 and batch['sample_inputs']==8
                and batch['planned_forwards']==51 and len(batch['groups'])==value['complete_ffn_groups']
                and [x['input_index'] for x in batch['inputs']]==expected['input_indices'], 'cost frozen sampling differs')
        ids=[g['group']['id'] for g in batch['groups']]
        require(len(set(ids))==len(ids),'duplicate cost group')
        require(all([g['group_id'] for g in x['groups']]==ids for x in batch['inputs']), 'cost input group directory differs')


def admit(registration_source, aggregate_source, sources):
    """Read only completed Sources; never discover/rebind a result by filename."""
    reg=sources.json(registration_source)
    cases=validate_registration(reg,sources)
    require(len(cases)==80 and [c['name'] for c,_,_ in cases]==[f'{f}-{n:03}' for f,count in COUNTS.items() for n in range(count)], 'full role/order directory differs')
    require(io.resolved_path(aggregate_source['path'])==io.resolved_path(Path(reg['execution_output'])/'result.json'), 'aggregate path differs from frozen output')
    aggregate=sources.json(aggregate_source);_aggregate(aggregate,registration_source)
    # The existing GPU aggregate is the authority for every output Source,
    # especially cost summaries, which are not embedded in each receipt.
    output_directory={}
    for source in aggregate['output_sources']:
        key=_key(source)
        require(key not in output_directory,'duplicate aggregate output Source')
        require(io.resolved_path(source['path']).is_relative_to(io.resolved_path(reg['execution_output']))
                or key in sources.items,'unbound external output Source')
        sources.add(source);output_directory[key]=source
    authority=dict(sources.items)
    source_inventory=sources.json(reg['source_inventory'])
    original={_key(s):s for s in source_inventory}
    # Reuse exactly the comparator/metric definitions pinned before GPU capture.
    for name in ('compare_ffn_calibration.py','validate_int8_backend.py','compare_worker_outputs.py','run_ffn_calibration.py','summarize_ffn_calibration_cost.py'):
        current=record(ROOT/'scripts'/name);_member(current,original);sources.add(current)
    captures={};pairings=[];cost_index=[]
    for (case,plan,_),receipt in zip(cases,aggregate['receipts'],strict=True):
        require(receipt['name']==case['name'] and receipt['forwards']==2197 and receipt['physical_rows']==2656,'receipt order/count differs')
        expected_role='FP16_REFERENCE' if case['recipe_index']==0 else 'SINGLE_FFN_INT8_ABLATION'
        require(case['role']==expected_role and len(case['group_ids'])==(0 if case['recipe_index']==0 else 1),'case role differs')
        _member(receipt['result'],output_directory)
        captured=sources.json(receipt['result']);_capture(captured,receipt['result'],case,plan,reg)
        exit_record=sources.json(receipt['exit'])
        require(exit_record['actual_exit_code']==0 and exit_record['error'] is None and exit_record['cleanup_error'] is None,'capture child failed')
        expected_command=[reg['executable']['path'],'--mode','collect','--plan',case['plan']['path'],
            '--plan-sha256',case['plan']['sha256'],'--output',str(Path(reg['execution_output'])/case['name'])]
        require(exit_record['command']==expected_command,'capture child command differs')
        for key in ('stdout','stderr'):sources.add(exit_record[key])
        cost_path=Path(reg['execution_output'])/(case['name']+'-cost-summary.json')
        key=os.path.normcase(str(io.resolved_path(cost_path)))
        require(key in output_directory,'missing bound case cost summary')
        cost_source=output_directory[key];cost=sources.json(cost_source)
        _cost(cost,receipt['result'],case,plan,reg)
        for s in cost['bound_sources']:
            _member(s,authority);sources.add(s)
        item=dict(name=case['name'],family=case['family'],recipe_index=case['recipe_index'],role=case['role'],
            group_ids=case['group_ids'],plan=case['plan'],capture=receipt['result'],cost_summary=cost_source,
            model_sha256=plan['model']['sha256'],graph_sha256=plan['graph_sha256'],
            recipe_sha256=plan['resolved_recipe_sha256'],output_contract_sha256=reg['contracts'][case['family']])
        cost_index.append(item)
        if case['recipe_index']==0:captures[case['family']]=item
        else:
            require(case['family'] in captures,'candidate precedes reference')
            pairings.append(dict(reference=captures[case['family']],candidate=item))
    require(len(captures)==2 and len(pairings)==78,'reference/candidate coverage differs')
    return reg,pairings,cost_index


def check_report(report,pair,authority):
    reference,candidate=pair['reference'],pair['candidate']
    require(report['schema']=='rustgo-single-ffn-calibration-comparison-v1' and report['status']=='COMPARED'
            and report['sources_unchanged'] is True and report['coverage_mode']=='complete-calibration'
            and report['unique_request_count']==report['placement_observations']==report['accuracy']['cases']==2044
            and report['repeated_request_observations']==0,'CPU report incomplete')
    require(same_source(report['reference'],reference['capture']) and same_source(report['candidate'],candidate['capture'])
            and report['group_ids']==candidate['group_ids'],'CPU report pair/group differs')
    for key in ('model_sha256','graph_sha256','output_contract_sha256'):
        require(report[key]==reference[key]==candidate[key],'CPU report identity differs: '+key)
    require(report['reference_recipe_sha256']==reference['recipe_sha256']
            and report['candidate_recipe_sha256']==candidate['recipe_sha256']
            and report['calibration_6pp_observation'] in ('PASS','FAIL')
            and report['production_certified'] is False and report['performance_measured'] is False,'CPU report scope differs')
    for source in report['bound_sources']:_member(source,authority)


def compare_all(registration_path,registration_sha,capture_path,capture_sha,output):
    output=Path(output)
    require(output.is_absolute() and not output.exists(),'fresh absolute CPU output directory required')
    output.mkdir(exist_ok=False)
    sources=io.Sources();outputs=io.Sources();receipts=[];error=None;pairings=[];costs=[]
    started=time.perf_counter();registration=None;aggregate=None
    try:
        # These sources are separate from, and never appended to, the GPU inventory.
        for path in (Path(__file__),Path(sys.executable)):
            sources.add(record(path))
        registration=sources.anchor(registration_path,registration_sha)
        aggregate=sources.anchor(capture_path,capture_sha)
        reg,pairings,costs=admit(registration,aggregate,sources)
        sources.recheck()
        before=list(sources.items.values());authority=dict(sources.items)
        outputs.add(put(output/'source-before.json',before))
        outputs.add(put(output/'intent.json',dict(schema='rustgo-full-ffn-cpu-comparison-attempt-v1',
            registration=registration,capture=aggregate,pairings=pairings,comparisons=78,
            per_pair_timeout_seconds=PER_PAIR_TIMEOUT,total_timeout_seconds=TOTAL_TIMEOUT,
            comparator=record(COMPARATOR),python=record(sys.executable),driver=record(__file__),
            model_forwards=0,gpu_calls=0,worker_calls=0,retries=0,
            numerical_threshold_failures_are_observations=True)))
        for ordinal,pair in enumerate(pairings):
            candidate,reference=pair['candidate'],pair['reference'];name=candidate['name']
            remaining=TOTAL_TIMEOUT-(time.perf_counter()-started)
            require(remaining>0,'CPU total deadline exhausted')
            destination=output/name
            command=[sys.executable,'-B',str(COMPARATOR),'--reference',reference['capture']['path'],
                '--reference-sha256',reference['capture']['sha256'],'--candidate',candidate['capture']['path'],
                '--candidate-sha256',candidate['capture']['sha256'],'--output',str(destination)]
            for source in (reference['capture'],candidate['capture'],reference['cost_summary'],candidate['cost_summary']):sources.add(source)
            outputs.add(put(output/(name+'-consumed.json'),dict(ordinal=ordinal,pair=pair,command=command,phase='CPU_ONLY_NO_GPU_RETRY')))
            run_child(command,output,name,os.environ.copy(),min(PER_PAIR_TIMEOUT,remaining))
            for suffix in ('-launched.json','-exit.json','.stdout.log','.stderr.log'):outputs.add(record(output/(name+suffix)))
            source=record(destination/'report.json');report=outputs.json(source)
            check_report(report,pair,authority)
            for bound in report['bound_sources']:sources.add(bound)
            entry=dict(ordinal=ordinal,name=name,family=candidate['family'],group_ids=candidate['group_ids'],
                reference=reference['capture'],candidate=candidate['capture'],report=source,
                reference_cost_summary=reference['cost_summary'],candidate_cost_summary=candidate['cost_summary'],
                calibration_6pp_observation=report['calibration_6pp_observation'])
            outputs.add(put(output/(name+'-verified.json'),entry));receipts.append(entry)
            print(json.dumps(dict(candidate=name,status='COMPARED',completed=len(receipts),
                calibration_6pp_observation=entry['calibration_6pp_observation'])),flush=True)
        require(len(receipts)==78,'missing comparisons')
        sources.recheck();outputs.recheck()
        outputs.add(put(output/'source-after.json',list(sources.items.values())))
        require(before==list(sources.items.values()),'CPU input source inventory changed')
        index=put(output/'index.json',dict(schema='rustgo-full-ffn-calibration-comparison-index-v1',
            status='COMPLETE_CALIBRATION_COMPARISONS',registration=registration,capture=aggregate,
            reference_captures=2,candidate_comparisons=78,requests_per_comparison=2044,comparisons=receipts,
            all_case_cost_summaries=costs,sources_unchanged=True,bound_sources=list(sources.items.values()),
            output_sources=list(outputs.items.values()),model_forwards=0,gpu_calls=0,worker_calls=0,
            adopted=False,production_certified=False,
            scope='Single-FFN calibration observations and links to fixed-input same-policy costs; no mixture/old-control-matrix performance acceptance.'))
        return index
    except BaseException as failure:
        error=dict(type=type(failure).__name__,message=str(failure),traceback=traceback.format_exc())
        put(output/'FAILED.json',dict(status='FAILED_CPU_COMPARISON_STOPPED',error=error,registration=registration,
            capture=aggregate,completed_comparisons=receipts,model_forwards=0,gpu_calls=0,worker_calls=0,
            retry_scope='New separately frozen CPU revision may reuse captures read-only; GPU recollection is forbidden.'))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('registration','capture'):
        parser.add_argument('--'+key,required=True,type=Path)
        parser.add_argument('--'+key+'-sha256',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result=compare_all(args.registration,args.registration_sha256,args.capture,args.capture_sha256,args.output)
    # Successful publication is terminal; an unavailable display pipe is not a
    # reason to recollect/recompare an already committed set of observations.
    try:print(json.dumps(result),flush=True)
    except OSError:pass


if __name__=='__main__':main()
