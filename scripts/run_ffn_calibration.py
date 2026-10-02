#!/usr/bin/env python3
"""Run a frozen, finite full-calibration registration once. Never retries a case.

This driver collects observations, not an accepted precision/execution plan.
It owns only the direct collector child it launches; no Worker is contacted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from compare_ffn_calibration import Sources, canonical, decode_json, require, resolved_path, sha

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'rustgo-full-ffn-calibration-registration-v1'
COUNTS = {'b11': 34, 'b15': 46}
TACTICS = {
    'KATAGO_CUDA_SPLITK': '0', 'KATAGO_CUDA_INT8_GEMM_TUNE': '0',
    'KATAGO_CUDA_CUBLASLT_RANK': 'heuristic', 'KATAGO_CUDA_FFN_COMPACT_R1': '0',
    'KATAGO_CUDA_CUBLASLT': '1', 'KATAGO_CUDA_DUALFFN': '0',
    'KATAGO_CUDA_FUSION': 'none', 'KATAGO_CUDA_GEMM_LAYOUT': 'tn',
    'KATAGO_CUDA_INT8_RMS_FUSION': '1', 'KATAGO_CUDA_NOGRAPH': '1',
    'KATAGO_CUDA_NOPIPELINE': '1', 'KATAGO_CUDA_PADBATCH': '0',
    'KATAGO_CUDA_BATCH_TRACE': '1',
}


def record(path):
    path = resolved_path(path)
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)


def put(path, value):
    """Publish without replacing an existing file, including a prior failed attempt."""
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    with pending.open('xb') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode() + b'\n')
        f.flush()
        os.fsync(f.fileno())
    os.link(pending, path)
    pending.unlink()
    return record(path)


def same_source(left, right):
    return (resolved_path(left['path']) == resolved_path(right['path'])
            and left['sha256'] == right['sha256'] and left['bytes'] == right['bytes'])


def expected_calls(plan, inputs):
    require(plan['mode'] == 'complete-calibration', 'full run requires complete-calibration')
    exact = [n for n, i in enumerate(inputs) if i['target_batch'] == i['physical_batch'] == 1]
    require(exact == list(range(2044)) and plan['numeric_input_indices'] == exact,
            'full numeric coverage/order differs')
    require(plan['expected_numeric_rows'] == 2044 and plan['chunk_inputs'] == 64,
            'numeric row/chunk budget differs')
    calls = []

    def add(index, phase, iteration):
        require(type(index) is int and 0 <= index < len(inputs), 'input index outside corpus')
        item = inputs[index]
        calls.append(dict(input_index=index, phase=phase, iteration=iteration,
                          physical_batch=item['physical_batch'], tensor_sha256=item['tensor_sha256']))

    for index in exact:
        add(index, 'numeric', 0)
    require([c['physical_batch'] for c in plan['cost']] == [1, 3, 8], 'cost batches differ')
    for cost in plan['cost']:
        b, indices = cost['physical_batch'], cost['input_indices']
        require(cost['warmup'] == 3 and cost['measurements'] == 4
                and len(indices) == len(set(indices)) == 8, 'cost W/M/S differs')
        for index in indices:
            require(type(index) is int and 0 <= index < len(inputs), 'cost input outside corpus')
            require(inputs[index]['target_batch'] == inputs[index]['physical_batch'] == b,
                    'tail or other packing substituted for full cost batch')
        for n in range(3):
            add(indices[0], 'cost-warmup', n)
        for index in indices:
            add(index, 'cost-before', 0)
            for n in range(4):
                add(index, 'cost-measure', n)
            add(index, 'cost-after', 0)
    require(len(calls) == plan['expected_forwards'] == 2197, 'forward budget differs')
    require(sum(c['physical_batch'] for c in calls) == 2656, 'physical row budget differs')
    return calls


def check_recipe(plan, recipe, proposed, admitted):
    require(proposed['recipe'] == recipe and proposed['recipe_sha256'] == plan['resolved_recipe_sha256']
            and proposed['group_ids'] == plan['group_ids'], 'proposal recipe/group differs')
    require(admitted['status'] == 'CPU_ACTUAL_GRAPH_ADMISSION_PASS'
            and admitted['canonical_recipe_bytes_sha256'] == admitted['resolved_recipe_sha256']
            == plan['resolved_recipe_sha256'] and admitted['group_ids'] == plan['group_ids']
            and admitted['source_object_sha256'] == sha(canonical(recipe, sorted_keys=True)),
            'recipe not identical to actual CPU admission')
    groups = plan['group_ids']
    require(len(groups) <= 1, 'not a single FFN or reference')
    role = 'SINGLE_FFN_INT8_ABLATION' if groups else 'FP16_REFERENCE'
    require(proposed['role'] == admitted['role'] == role, 'wrong recipe role')
    projections = recipe['projections']
    require(len({p['id'] for p in projections}) == len(projections) == admitted['projection_count'],
            'projection duplicate/missing')
    require(all(p['precision'] in ('fp16', 'int8') for p in projections), 'unsupported precision')
    require({p['id'] for p in projections if p['precision'] == 'int8'}
            == {g + suffix for g in groups for suffix in ('.dual', '.down')},
            'recipe does not quantize exactly target dual/down')
    return role


def validate_registration(registration, sources):
    from ffn_calibration_inputs import choose_cost_inputs
    reg = registration
    require(reg['schema'] == SCHEMA and reg['scope'] == 'FULL_CALIBRATION_CAPTURE_ONLY'
            and reg['no_retries'] is True, 'registration scope/schema differs')
    require(reg['tactics'] == TACTICS, 'fixed tactics differ')
    require(reg['limits'] == dict(processes=80, uploads=80, forwards=175760,
            physical_rows=212480, numeric_rows=163520, per_case_timeout_seconds=1800,
            total_timeout_seconds=43200, per_case_output_bytes=256*1024**2,
            total_output_bytes=20*1024**3, minimum_free_bytes=25*1024**3), 'limits differ')
    require(Path(reg['execution_output']).is_absolute() and Path(reg['consume_file']).is_absolute(),
            'execution paths must be absolute')
    require(Path(reg['execution_output']).parent.resolve() == Path(reg['consume_file']).parent.resolve(),
            'execution output/consume guard not in same frozen directory')
    for s in sources.json(reg['source_inventory']):
        sources.add(s)
    sources.add(reg['executable'])
    require(same_source(reg['driver'], record(__file__)), 'driver differs from prepared source')
    admission = sources.json(reg['admission'])
    require(admission['status'] == 'PASS' and admission['admitted_recipes'] == 80
            and admission['negative_checks'] == 14 and admission['model_parses'] == 2
            and admission['graph_lowers'] == 2, 'incomplete actual admission')
    require([m['family'] for m in admission['models']] == ['b11', 'b15'], 'admitted models differ')
    require([m['family'] for m in reg['models']] == ['b11', 'b15'], 'registered models differ')
    require(len(reg['cases']) == 80 and len({c['name'] for c in reg['cases']}) == 80,
            'missing/duplicate cases')
    loaded, cases = {}, []
    for model, actual in zip(reg['models'], admission['models']):
        family = model['family']
        proposal = sources.json(actual['proposal'])
        require(len(proposal['recipes']) == len(actual['admitted_recipes']) == COUNTS[family],
                'model recipe set incomplete')
        declared = dict(proposal)
        digest = declared.pop('proposal_sha256')
        require(sha(canonical(declared, sorted_keys=True)) == digest, 'proposal object SHA differs')
        index = sources.json(model['inputs'])
        require(index['model_sha256'] == actual['model']['sha256']
                and index['provenance'] == model['provenance'], 'input index model/provenance differs')
        inputs = index['inputs']
        require(len(inputs) == 2982, 'input packing coverage differs')
        selected = choose_cost_inputs(inputs)
        require(model['cost'] == selected, 'cost samples differ from fixed input-only selection')
        sources.add(model['provenance'])
        loaded[family] = inputs
        for n, (proposed, admitted) in enumerate(zip(proposal['recipes'], actual['admitted_recipes'])):
            case = reg['cases'][len(cases)]
            require(case['name'] == f'{family}-{n:03}' and case['family'] == family
                    and case['recipe_index'] == admitted['index'] == n, 'case sequence/admission index differs')
            plan = sources.json(case['plan'])
            require(plan['schema'] == 'rustgo-ffn-calibration-collection-plan-v1'
                    and plan['binary'] is True and plan['compressed'] is True, 'plan schema/format differs')
            require(same_source(plan['model'], actual['model'])
                    and plan['graph_sha256'] == actual['graph_sha256']
                    and same_source(plan['proposal'], actual['proposal'])
                    and plan['proposal_object_sha256'] == digest
                    and same_source(plan['provenance'], model['provenance']), 'plan identity differs')
            role = check_recipe(plan, sources.json(plan['recipe']), proposed, admitted)
            require(case['role'] == role and case['group_ids'] == plan['group_ids']
                    and plan['cost'] == selected, 'case role/group/sampling differs')
            calls = expected_calls(plan, inputs)
            require(case['forwards'] == 2197 and case['physical_rows'] == 2656
                    and case['numeric_rows'] == 2044 and case['uploads'] == 1, 'case budget differs')
            cases.append((case, plan, calls))
    require(len(cases) == reg['limits']['processes'] and
            sum(len(calls) for _, _, calls in cases) == reg['limits']['forwards'], 'global budget differs')
    return cases


def child_environment():
    env = {k: v for k, v in os.environ.items()
           if not k.upper().startswith(('KATAGO_CUDA_', 'RUSTGO_OUTPUT_'))}
    env.update(TACTICS)
    cuda = Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3')
    env['PATH'] = str(cuda/'bin/x64') + os.pathsep + str(cuda/'bin') + os.pathsep + env.get('PATH', '')
    return env


def tree_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def run_child(command, output, name, env, timeout, monitor=lambda: None):
    """Always retain a receipt; terminate only the Popen child owned here."""
    child, error, code, cleanup_error = None, None, None, None
    start = time.perf_counter()
    outpath, errpath = output/(name+'.stdout.log'), output/(name+'.stderr.log')
    try:
        with outpath.open('xb') as out, errpath.open('xb') as err:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err)
            put(output/(name+'-launched.json'), dict(pid=child.pid, command=command))
            deadline, next_monitor = start + timeout, start
            while True:
                now = time.perf_counter()
                require(now < deadline, 'child deadline exhausted; no retry')
                if now >= next_monitor:
                    monitor()
                    next_monitor = now + 2
                try:
                    code = child.wait(timeout=min(.25, deadline-now))
                    break
                except subprocess.TimeoutExpired:
                    pass
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    finally:
        if child is not None:
            try:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=30)
                code = child.returncode
            except BaseException as exc:
                cleanup_error = dict(type=type(exc).__name__, message=str(exc))
                code = child.returncode
    receipt = dict(command=command, pid=child.pid if child is not None else None,
                   actual_exit_code=code, seconds=time.perf_counter()-start, error=error,
                   cleanup_error=cleanup_error,
                   stdout=record(outpath) if outpath.exists() else None,
                   stderr=record(errpath) if errpath.exists() else None)
    put(output/(name+'-exit.json'), receipt)
    require(error is None and cleanup_error is None and code == 0, 'child failed; remaining cases unconsumed')
    return receipt


def verify_capture(path, case, plan, calls, reg, output_sources):
    result_source = record(path/'result.json')
    value = output_sources.json(result_source)
    require(value['status'] == 'CAPTURE_COMPLETE_METRICS_PENDING' and value['sources_unchanged'] is True
            and value['consumed_forwards'] == len(calls) and value['numeric_inputs'] == value['numeric_rows'] == 2044
            and value['gpu_model_uploads'] == 1, 'capture status/count differs')
    require(value['model_sha256'] == plan['model']['sha256'] and value['graph_sha256'] == plan['graph_sha256']
            and value['recipe_sha256'] == plan['resolved_recipe_sha256'], 'capture model/recipe differs')
    require(len(value['sources']) == 6, 'capture source count differs')
    for actual, expected in zip(value['sources'], [case['plan'], reg['executable'], plan['model'],
                                                plan['recipe'], plan['proposal'], plan['provenance']]):
        require(same_source(actual, expected), 'capture input Source differs')
    intent = output_sources.json(value['intent'])
    require(intent['plan'] == plan and intent['calls'] == calls and intent['mode'] == 'collect'
            and intent['retry_allowed'] is False and intent['planned_uploads'] == 1, 'capture intent differs')
    runtime = output_sources.json(value['runtime'])
    for key, expected in reg['runtime_identity'].items():
        require(runtime[key] == expected, 'runtime identity changed: '+key)
    require(runtime['executable']['sha256'] == reg['executable']['sha256']
            and runtime['native']['resolved_recipe_sha256'] == plan['resolved_recipe_sha256']
            and runtime['native']['source_recipe_sha256'] == plan['recipe']['sha256'], 'runtime recipe/exe differs')
    require(value['output_contract_sha256'] == reg['contracts'][case['family']]
            == intent['output_contract_sha256'], 'output contract differs')
    for key in ('implicit_warmup', 'graph_capture', 'graph_replay', 'automatic_retries'):
        require(runtime[key] == 0, 'extra runtime activity')
    require(runtime['model_uploads'] == 1 and runtime['raw_heads_prefilled_nan_each_forward'] is True,
            'runtime upload/head initialization differs')
    load = output_sources.json(value['load_attempt'])
    require(load == dict(model=plan['model'], recipe=plan['recipe'], ordinal=0), 'load attempt differs')
    journal = [decode_json(line) for line in output_sources.read(value['journal']).splitlines()]
    require(len(journal) == len(calls), 'journal attempt count differs')
    for ordinal, (event, call) in enumerate(zip(journal, calls)):
        observed = event['event']['runtime']
        require(event['ordinal'] == ordinal and event['event']['expected'] == call
                and observed['runtime_attempt'] == ordinal+1
                and all(observed[k] == call[k] for k in ('phase', 'iteration', 'physical_batch', 'tensor_sha256')),
                'journal runtime diverged from schedule')
    # The Rust collector checks raw finite/PB semantics. Here bind every output blob
    # and exact chunk placement; the CPU metric and cost readers perform semantics.
    require(len(value['chunks']) == 32 and len(value['cost']) == 27, 'chunk/cost file count differs')
    numeric = []
    for source in value['chunks']:
        chunk = output_sources.json(source)
        for blob in (chunk['raw'], chunk['pb']):
            output_sources.add(blob)
        numeric.extend(i['input_index'] for i in chunk['inputs'])
    require(numeric == plan['numeric_input_indices'], 'chunk coverage/order differs')
    for source in value['cost']:
        output_sources.add(source)
    # Stop before the next GPU case if any group, binding, route, setup or
    # before/measure/after bitgate record is malformed. No new forward is run.
    from summarize_ffn_calibration_cost import summarize_capture
    cost_report = summarize_capture(Path(result_source['path']), result_source['sha256'])
    require(cost_report['status'] == 'SUMMARIZED_COST_OBSERVATIONS' and cost_report['sources_unchanged'] is True,
            'cost record validation incomplete')
    for source in cost_report['bound_sources']:
        output_sources.add(source)
    output_sources.add(put(path.parent/(case['name']+'-cost-summary.json'), cost_report))
    return result_source


def execute(registration_path, expected_sha):
    sources = Sources()
    registration_source = sources.anchor(registration_path, expected_sha)
    reg = sources.json(registration_source)
    cases = validate_registration(reg, sources)
    output, guard = Path(reg['execution_output']), Path(reg['consume_file'])
    require(not output.exists() and not guard.exists() and not guard.with_name(guard.name+'.pending').exists(),
            'registration already consumed or output exists; no retry')
    require(shutil.disk_usage(output.parent).free >= reg['limits']['minimum_free_bytes'], 'insufficient free disk')
    # A single immutable guard fixes this registration to one output. Never remove it.
    put(guard, dict(registration=registration_source, output=str(output), status='CONSUMED_NO_RETRY'))
    output.mkdir(exist_ok=False)
    receipts, error, output_sources = [], None, Sources()
    started = time.perf_counter()
    before = list(sources.items.values())
    try:
        put(output/'source-before.json', before)
        put(output/'intent.json', dict(registration=registration_source, limits=reg['limits'],
                                      tactics=TACTICS, production_worker_untouched=True))
        for ordinal, (case, plan, calls) in enumerate(cases):
            require(time.perf_counter()-started < reg['limits']['total_timeout_seconds'], 'total deadline exhausted')
            sources.add(case['plan'])
            destination = output/case['name']
            command = [reg['executable']['path'], '--mode', 'collect', '--plan', case['plan']['path'],
                       '--plan-sha256', case['plan']['sha256'], '--output', str(destination)]
            # Even a failed launch or interrupted launch consumes this case.
            put(output/(case['name']+'-consumed.json'), dict(ordinal=ordinal, case=case, command=command,
                                                           reserved_before_launch=True))

            def monitor():
                require(tree_bytes(destination) <= reg['limits']['per_case_output_bytes'], 'case storage limit')
                require(tree_bytes(output) <= reg['limits']['total_output_bytes'], 'total storage limit')
                require(shutil.disk_usage(output).free >= 1024**3, 'remaining disk below safety reserve')

            timeout = min(reg['limits']['per_case_timeout_seconds'],
                          reg['limits']['total_timeout_seconds']-(time.perf_counter()-started))
            receipt = run_child(command, output, case['name'], child_environment(), timeout, monitor)
            monitor()
            result_source = verify_capture(destination, case, plan, calls, reg, output_sources)
            receipts.append(dict(name=case['name'], exit=record(output/(case['name']+'-exit.json')),
                                 result=result_source, forwards=len(calls), physical_rows=case['physical_rows']))
            put(output/(case['name']+'-verified.json'), receipts[-1])
            print(json.dumps(dict(case=case['name'], status='CAPTURE_VERIFIED', completed=len(receipts),
                                  actual_exit_code=receipt['actual_exit_code'], seconds=receipt['seconds'])), flush=True)
        sources.recheck()
        output_sources.recheck()
    except BaseException as failure:
        error = dict(type=type(failure).__name__, error=str(failure), traceback=traceback.format_exc())
        put(output/'first-failure.json', error)
    after, source_error = [], None
    try:
        after = [record(s['path']) for s in before]
        require(after == before, 'source inventory changed')
    except BaseException as failure:
        source_error = str(failure)
    put(output/'source-after.json', after)
    passed = error is None and source_error is None and len(receipts) == 80
    value = dict(status='FULL_CAPTURE_METRICS_PENDING' if passed else 'FAIL_STOPPED_NO_RETRY',
                 registration=registration_source, receipts=receipts, error=error, source_error=source_error,
                 sources_unchanged=after == before, verified_cases=len(receipts),
                 reserved_cases=len(list(output.glob('*-consumed.json'))),
                 verified_forwards=sum(r['forwards'] for r in receipts),
                 observed_forward_attempts=sum(len(p.read_bytes().splitlines()) for p in output.glob('*/forward-attempts.jsonl')),
                 verified_physical_rows=sum(r['physical_rows'] for r in receipts),
                 output_sources=list(output_sources.items.values()), retry_allowed=False,
                 scope='Full single-FFN calibration capture; metrics/cost aggregation and mixture validation remain pending.')
    result = put(output/'result.json', value)
    print(json.dumps(result), flush=True)
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', required=True, type=Path)
    parser.add_argument('--registration-sha256', required=True)
    args = parser.parse_args()
    return 0 if execute(args.registration, args.registration_sha256) else 1


if __name__ == '__main__':
    raise SystemExit(main())
