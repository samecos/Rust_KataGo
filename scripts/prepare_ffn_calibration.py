#!/usr/bin/env python3
"""Freeze the current B11/B15 full single-FFN calibration run. CPU metadata only."""
import argparse
from pathlib import Path

from compare_ffn_calibration import Sources, require, canonical, sha
from run_ffn_calibration import ROOT, SCHEMA, TACTICS, COUNTS, record, put, check_recipe, expected_calls
from ffn_calibration_inputs import load_inputs, choose_cost_inputs

BASE = ROOT/'target/unified-quant-calibration-collector-r1'
PREREQUISITES = {
    'cpu': (BASE/'cpu-r2/result.json', 'aa5a7eb77fd806808ec8f8fa5bcb685919e39c793582a3a2ad970de96d69a39f'),
    'cuda': (BASE/'cuda-first/result.json', '44f126ccc5930624aab280ec566df76799bad47085e6eb9c3393b1ab50853ed1'),
    'preflight': (BASE/'preflight-first/result.json', 'ce25e59ad9bc1a8da997320319616fd8ac378d540a748a4f02fed1e5de7884e1'),
    'admission': (ROOT/'target/unified-quant-calibration-recipes-check-r2/result.json',
                  'd4e1d35f3ebdb4b6819e4af63e1dfca93cb5984d3cfe66043619edfe53c36bfd'),
    'finite': (BASE/'finite-gpu-first/result.json', '5a5d6613a4a81fb10b8fe8a519af32ab5a4fcf3b9b99740bda7aad25686bb66b'),
    'audit': (ROOT/'target/unified-quant-calibration-collector-terminal-audit-r1/r2/report.json',
              'e62ea1754166eb21462d958c54cc4ff1d9ff3145373bf9a63927ee5bc050d3b3'),
    'metrics': (BASE/'real-metrics-first/result.json', '8ae420e8aa8761965d39ac8aa597ca831f09234f916919978f088d68ebc7f764'),
}


def prepare(output, checks_path, checks_sha):
    require(output.is_absolute() and not output.exists(), 'fresh absolute prepared directory required')
    output.mkdir(parents=False, exist_ok=False)
    sources, bound, evidence = Sources(), {}, {}
    for name, (path, expected) in PREREQUISITES.items():
        bound[name] = sources.anchor(path, expected)
        evidence[name] = sources.json(bound[name])
    checks_source = sources.anchor(checks_path, checks_sha)
    checks = sources.json(checks_source)
    require(checks['status'] == 'PASS_FULL_DRIVER_CPU_CHECKS' and checks['actual_exit_code'] == 0,
            'full driver CPU checks not complete')
    for source in checks['sources']:
        sources.add(source)
    require(evidence['cpu']['status'] == evidence['cuda']['status'] == evidence['admission']['status'] == 'PASS',
            'prerequisite failed')
    require(evidence['audit']['status'] == 'PASS_INDEPENDENT_FINITE_COLLECTOR_AUDIT'
            and evidence['audit']['counts']['forwards'] == 104, 'finite audit not proven')
    admission_source = evidence['admission']['admission_report']
    admission = sources.json(admission_source)
    require(admission['status'] == 'PASS' and admission['admitted_recipes'] == 80
            and admission['negative_checks'] == 14, 'actual recipe admission incomplete')
    for inventory_path in (BASE/'cuda-first/source-after.json',
                           ROOT/'target/unified-quant-calibration-recipes-check-r2/source-after.json'):
        inventory = sources.json(record(inventory_path))
        for source in inventory:
            sources.add(source)
    executable = evidence['cuda']['receipts'][0]['artifacts'][0]['executable']
    sources.add(executable)
    for name in ('run_ffn_calibration.py', 'prepare_ffn_calibration.py', 'ffn_calibration_inputs.py',
                 'compare_ffn_calibration.py', 'summarize_ffn_calibration_cost.py',
                 'validate_int8_backend.py', 'compare_worker_outputs.py'):
        sources.add(record(ROOT/'scripts'/name))
    cases, models, contracts, pairings = [], [], {}, []
    runtime_identity = None
    for actual in admission['models']:
        family = actual['family']
        require(family in COUNTS and len(actual['admitted_recipes']) == COUNTS[family], 'model recipe count differs')
        base_source = record(BASE/'preflight-plans'/f'{family}-plan.json')
        base = sources.json(base_source)
        proposal = sources.json(base['proposal'])
        require(proposal['proposal_sha256'] == base['proposal_object_sha256'], 'proposal object identity differs')
        proposed_body = dict(proposal)
        proposed_body.pop('proposal_sha256')
        require(sha(canonical(proposed_body, sorted_keys=True)) == base['proposal_object_sha256'], 'proposal hash differs')
        inputs = load_inputs(base['provenance'], sources)
        cost = choose_cost_inputs(inputs)
        input_source = put(output/f'{family}-inputs.json', dict(schema='rustgo-full-ffn-input-index-v1',
            model_sha256=base['model']['sha256'], provenance=base['provenance'], inputs=inputs))
        sources.add(input_source)
        models.append(dict(family=family, inputs=input_source, provenance=base['provenance'], cost=cost))
        pairings.append([(i['id'], i['tensor_sha256']) for i in inputs])
        reference_source = record(BASE/'finite-gpu-first'/f'{family}-reference/result.json')
        reference = sources.json(reference_source)
        contracts[family] = reference['output_contract_sha256']
        runtime = sources.json(reference['runtime'])
        observed = {k: runtime[k] for k in ('device', 'driver_version', 'backend_build', 'effective_tactics',
                                           'libraries', 'library_order', 'library_resolution')}
        if runtime_identity is None:
            runtime_identity = observed
        require(runtime_identity == observed and observed['effective_tactics'] == TACTICS, 'reference runtime differs')
        for library in runtime['libraries']:
            sources.add(library)
        require(len(proposal['recipes']) == COUNTS[family], 'proposal count differs')
        for n, (candidate, admitted) in enumerate(zip(proposal['recipes'], actual['admitted_recipes'])):
            name = f'{family}-{n:03}'
            recipe = put(output/f'{name}-recipe.json', candidate['recipe'])
            plan = dict(base, mode='complete-calibration', recipe=recipe,
                        resolved_recipe_sha256=candidate['recipe_sha256'], group_ids=candidate['group_ids'],
                        numeric_input_indices=list(range(2044)), cost=cost, expected_forwards=2197,
                        expected_numeric_rows=2044, chunk_inputs=64)
            role = check_recipe(plan, candidate['recipe'], candidate, admitted)
            expected_calls(plan, inputs)
            plan_source = put(output/f'{name}-plan.json', plan)
            sources.add(recipe)
            sources.add(plan_source)
            cases.append(dict(name=name, family=family, recipe_index=n, role=role, group_ids=candidate['group_ids'],
                              plan=plan_source, forwards=2197, physical_rows=2656, numeric_rows=2044, uploads=1))
    require(len(cases) == 80 and pairings[0] == pairings[1], 'full case/input pairing differs')
    require([m['cost'] for m in models][0] == [m['cost'] for m in models][1], 'models did not select identical inputs')
    sources.recheck()
    inventory_source = put(output/'source-inventory.json', list(sources.items.values()))
    registration = dict(schema=SCHEMA, scope='FULL_CALIBRATION_CAPTURE_ONLY', no_retries=True,
        driver=record(ROOT/'scripts/run_ffn_calibration.py'), executable=executable,
        checks=checks_source, prerequisites=bound, admission=admission_source,
        source_inventory=inventory_source, models=models, cases=cases, contracts=contracts,
        runtime_identity=runtime_identity, tactics=TACTICS, execution_output=str(output/'capture'),
        consume_file=str(output/'execution-consumed.json'),
        limits=dict(processes=80, uploads=80, forwards=175760, physical_rows=212480, numeric_rows=163520,
                    per_case_timeout_seconds=1800, total_timeout_seconds=43200,
                    per_case_output_bytes=256*1024**2, total_output_bytes=20*1024**3,
                    minimum_free_bytes=25*1024**3),
        cost_sampling='SHA256(domain + u64LE(B) + tensor_SHA_bytes), first 8 full inputs per packing; no output-informed selection',
        cost_scope='24 fixed tensors, not full-corpus/search-leaf cost distribution; same-policy FP16 control, DUALFFN=0',
        production_worker_pid_do_not_stop=78408)
    result = put(output/'registration.json', registration)
    print(__import__('json').dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--checks', required=True, type=Path)
    parser.add_argument('--checks-sha256', required=True)
    args = parser.parse_args()
    prepare(args.output, args.checks, args.checks_sha256)


if __name__ == '__main__':
    main()
