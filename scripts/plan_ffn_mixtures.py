#!/usr/bin/env python3
"""Nominate bounded FP16/INT8 mixtures from completed calibration metadata.

Pure CPU planning only: no subprocess, model, Worker, ledger or GPU entry point.
The fixed v1 strategy is a heuristic, not an accuracy bound or speedup estimate.
It reads the complete comparison index, reports, cost summaries and recipes; it
does not reread numeric blobs or reinterpret their recorded producer guarantees.
"""
import argparse
import ast
import copy
from functools import lru_cache
import math
import os
from pathlib import Path
import sys
import traceback

import compare_ffn_calibration as io
import compare_all_ffn_calibration as comparisons
import summarize_ffn_calibration_cost as costs
from run_ffn_calibration import record, same_source, TACTICS

require = io.require
FAMILIES = {'b11': 33, 'b15': 45}
RECIPE_HELPER = Path(__file__).resolve().with_name('plan_quantization_search.py')
RECIPE_HELPER_SHA = 'b574518ea68fcc3996bba57c1d56cbd5adc4af5bf26daa77057c625809fefd86'
ADMISSION_SCHEMA = 'rustgo-ffn-mixture-admission-v1'
ADMISSION_HEADER = ('family','model_sha256','graph_sha256','output_contract_sha256','reference_recipe_sha256')
ADMISSION_RECIPE = ('role','group_ids','recipe','recipe_source','recipe_sha256','recipe_object_sha256')
BATCHES = (1, 3, 8)
BUDGETS_PP = (1.5, 3.0, 6.0)
STRATEGY = dict(version=1, physical_batches=list(BATCHES), risk_proxy_budgets_pp=list(BUDGETS_PP),
    cost_sample_inputs=8, measurements_per_input=4, minimum_input_wins=6,
    rank_denominator_floor_pp=0.01, max_aliases=18, max_unique_nonempty_recipes=18,
    sorting='saved_target_group_ms/max(single_group_win_max_abs_pp,0.01) descending; '
            'policy_kl_mean, score_mean_max_abs, stable_group_id ascending',
    selection='Budget-greedy sets; skip an over-budget group and continue scanning.',
    risk_proxy='Sum of isolated single-group maximum win-probability differences in percentage points; '
               'NOT a bound on a mixture, NOT an additive-error assumption.',
    numeric_scope='2044 original calibration requests at physical B1; no B3/B8 numerical validation is inferred.',
    cost_scope='Paired target whole-FFN groups on the same frozen eight inputs per physical batch; '
               'four event measurements per input; positive equal-input mean saving and at least six input wins. '
               'These hash-ranked cost tensors do not establish equivalence with the complete numerical distribution.')
SCOPE = ('Proposals only, not globally optimal. Whole-graph calibration and the complete old control matrix '
         'remain required. Fixed DUALFFN=0/NOGRAPH=1 costs do not represent the strongest old FP16 baseline. '
         'Future ledger checks must reject already-consumed identities or forbidden retries; this planner '
         'does not register experiments or authorize new GPU calls.')


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'invalid '+label)
    return float(value)


def object_sha(value):
    """Planning object identity, deliberately NOT a Rust resolved-recipe SHA."""
    return io.sha(io.canonical(value, sorted_keys=True))


def admission_bytes(value):
    """Cross-language envelope: sorted string keys, UTF-8, no floats or signed integers.

    JSON booleans remain booleans, not Python's integer subclass. The actual
    recipe/Source fields impose their own narrower types. No ASCII escaping is
    forced, including Chinese paths; ordinary JSON escaping is unchanged.
    """
    def check(node):
        kind=type(node)
        if node is None or kind in (str,bool):return
        if kind is int:
            require(0 <= node <= 2**64-1, 'admission integer outside u64')
        elif kind is list:
            for child in node:check(child)
        elif kind is dict:
            require(all(type(key) is str for key in node), 'admission object keys must be strings')
            for child in node.values():check(child)
        else:
            raise ValueError('admission permits no floats or non-JSON values')
    check(value)
    raw=io.canonical(value,sorted_keys=True)
    require(len(raw) <= 4*1024*1024, 'admission envelope exceeds bound')
    return raw


def admission_projection(manifest):
    """Exact no-float projection also checked independently by the v2 consumer.

    Costs, aliases, risk proxies, historical captures and statuses remain in the
    full manifest, protected by its file Source and Python proposal-object SHA.
    They are deliberately not part of Rust's independent admission digest.
    """
    projected=dict(schema=ADMISSION_SCHEMA,**{key:copy.deepcopy(manifest[key]) for key in ADMISSION_HEADER})
    projected['recipes']=[{key:copy.deepcopy(recipe[key]) for key in ADMISSION_RECIPE} for recipe in manifest['recipes']]
    require(projected['family'] in FAMILIES and 1 <= len(projected['recipes']) <= 10,
            'admission family/recipe generation bound differs')
    for recipe in projected['recipes']:
        source=recipe['recipe_source']
        require(set(source) == {'path','bytes','sha256'} and type(source['path']) is str
                and io.valid_sha(source['sha256']), 'invalid admission recipe Source')
        io.integer(source['bytes'])
        for entry in recipe['recipe']['projections']:
            io.integer(entry['expected_n'],1);io.integer(entry['expected_k'],1)
        io.integer(recipe['recipe']['version'],1);io.integer(recipe['recipe']['quantization_semantics_version'],1)
    admission_bytes(projected)
    return projected


@lru_cache(maxsize=1)
def recipe_function():
    """Use the exact existing pure serializer without importing collection code."""
    raw=RECIPE_HELPER.read_bytes()
    require(len(raw) < 1024*1024 and io.sha(raw) == RECIPE_HELPER_SHA, 'pinned recipe serializer changed')
    nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name == 'canonical_recipe']
    require(len(nodes) == 1 and not nodes[0].decorator_list, 'recipe serializer definition differs')
    namespace={}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(RECIPE_HELPER),'exec'),namespace)
    return namespace['canonical_recipe']


def recipe_sha(recipe):
    choices={p['id']:p['precision'] for p in recipe['projections']}
    return io.sha(io.canonical(recipe_function()(recipe,choices)))


def directory(records):
    result = {}
    for source in records:
        require(isinstance(source, dict) and set(source) == {'path','bytes','sha256'}
                and Path(source['path']).is_absolute() and io.valid_sha(source['sha256']), 'invalid indexed Source')
        io.integer(source['bytes'])
        key = comparisons._key(source)
        require(key not in result, 'duplicate indexed Source')
        result[key] = source
    return result


def read_bound(source, authority, sources):
    comparisons._member(source, authority)
    return sources.json(source)


def check_index(index):
    require(index['schema'] == 'rustgo-full-ffn-calibration-comparison-index-v1'
            and index['status'] == 'COMPLETE_CALIBRATION_COMPARISONS'
            and index['sources_unchanged'] is True and index['reference_captures'] == 2
            and index['candidate_comparisons'] == 78 and index['requests_per_comparison'] == 2044
            and index['adopted'] is False and index['production_certified'] is False
            and index['model_forwards'] == index['gpu_calls'] == index['worker_calls'] == 0,
            'incomplete or differently scoped comparison index')
    names = [f'{family}-{n:03}' for family,count in FAMILIES.items() for n in range(1,count+1)]
    require(len(index['comparisons']) == 78 and [r['name'] for r in index['comparisons']] == names
            and [r['ordinal'] for r in index['comparisons']] == list(range(78)), 'comparison order/coverage differs')
    names = [f'{family}-{n:03}' for family,count in FAMILIES.items() for n in range(count+1)]
    require(len(index['all_case_cost_summaries']) == 80
            and [r['name'] for r in index['all_case_cost_summaries']] == names, 'cost order/coverage differs')


def recipe_groups(recipe, count):
    require(set(recipe) == {'schema','version','quantization_semantics_version','model_sha256','graph_sha256','projections'}
            and recipe['schema'] == 'rustgo-precision-recipe'
            and recipe['version'] == recipe['quantization_semantics_version'] == 1,
            'requires complete v1 FP16/INT8 recipe')
    require(io.valid_sha(recipe['model_sha256']) and io.valid_sha(recipe['graph_sha256']), 'invalid recipe identity')
    entries = recipe['projections']
    require(entries and len({p['id'] for p in entries}) == len(entries), 'duplicate/empty projections')
    for p in entries:
        require(set(p) == {'id','expected_n','expected_k','precision'} and isinstance(p['id'],str)
                and p['precision'] == 'fp16', 'reference projection is not explicit FP16')
        io.integer(p['expected_n'],1);io.integer(p['expected_k'],1)
        require(p['id'].endswith(('.ffn.dual','.ffn.down','.attention.qkv','.attention.out')), 'unknown projection role')
    groups = sorted(p['id'][:-5] for p in entries if p['id'].endswith('.ffn.dual'))
    require(len(groups) == count and {p['id'][:-5] for p in entries if p['id'].endswith('.ffn.down')} == set(groups),
            'incomplete FFN group inventory')
    by_id={p['id']:p for p in entries}
    expected={identity for g in groups for identity in
              (g+'.dual',g+'.down',g[:-4]+'.attention.qkv',g[:-4]+'.attention.out')}
    require(set(by_id) == expected, 'reference omits/duplicates an attention or FFN projection')
    for g in groups:
        dual,down=by_id[g+'.dual'],by_id[g+'.down']
        qkv,out=by_id[g[:-4]+'.attention.qkv'],by_id[g[:-4]+'.attention.out']
        h,c=dual['expected_n'],dual['expected_k']
        require((down['expected_n'],down['expected_k'],qkv['expected_n'],qkv['expected_k'],out['expected_n'],out['expected_k'])
                == (c,h,3*c,c,c,c), 'reference projection dimensions disagree')
    return groups


def compose_recipe(reference, selected):
    wanted = {g+s for g in selected for s in ('.dual','.down')}
    require(len(wanted) == 2*len(selected) and wanted <= {p['id'] for p in reference['projections']}, 'invalid selected groups')
    # Explicit full inventory, preserving all admitted dimensions and identities.
    return recipe_function()(reference,{identity:'int8' for identity in wanted})


def sensitivity(report):
    accuracy = report['accuracy']
    fields = ('win_probability_max_abs','win_probability_mean_abs','policy_kl_mean','policy_kl_max',
              'score_mean_max_abs','score_mean_mean_abs')
    observed = {k:finite(accuracy[k],k) for k in fields}
    require(observed['win_probability_max_abs'] <= 1, 'invalid probability difference')
    observed['top1_matches'] = io.integer(accuracy['top1_matches'],0,2044)
    observed['win_max_abs_pp'] = observed['win_probability_max_abs']*100.0
    require(report['win_probability_max_abs_pp'] == observed['win_max_abs_pp']
            and report['win_probability_mean_abs_pp'] == observed['win_probability_mean_abs']*100,
            'reported probability/percentage-point units differ')
    eligible = observed['win_probability_max_abs'] <= 0.06
    require(report['calibration_6pp_observation'] == ('PASS' if eligible else 'FAIL'), '6pp flag differs from metric')
    return observed, eligible


def target_cost_pair(reference, candidate, group_id, batch):
    require([b['physical_batch'] for b in reference['batches']] == list(BATCHES)
            and [b['physical_batch'] for b in candidate['batches']] == list(BATCHES), 'cost batch order differs')
    ref, got = (next(b for b in cost['batches'] if b['physical_batch'] == batch) for cost in (reference,candidate))
    require({k:v for k,v in ref['binding'].items() if k != 'recipe_sha256'}
            == {k:v for k,v in got['binding'].items() if k != 'recipe_sha256'},
            'cost device/build/driver/tactics or model/graph identity differs')
    for b,summary in ((ref,reference),(got,candidate)):
        require(b['binding']['schema'] == 'rustgo-group-cost' and b['binding']['version'] == 1
                and b['binding']['mode'] == 'DIRECT_DIAGNOSTIC'
                and all(b['binding'][k] == summary[k] for k in ('model_sha256','graph_sha256','recipe_sha256')),
                'cost binding does not describe its own summary')
    for b in (ref,got):
        require(b['warmup'] == 3 and b['sample_inputs'] == len(b['inputs']) == 8
                and b['measurements_per_input'] == 4 and b['planned_forwards'] == 51,
                'fixed S8/W3/M4 sampling differs')
        require(len({i['input_index'] for i in b['inputs']}) == 8, 'duplicate cost input')
    def one(groups, key):
        require(len({g[key]['id'] if key == 'group' else g[key] for g in groups}) == len(groups), 'duplicate cost group')
        rows = [g for g in groups if (g[key]['id'] if key == 'group' else g[key]) == group_id]
        require(len(rows) == 1, 'target cost group missing')
        return rows[0]
    rg,cg = one(ref['groups'],'group'),one(got['groups'],'group')
    for k in ('id','rms_layer','ffn_layer','end_layer','mid','hidden'):
        require(rg['group'][k] == cg['group'][k], 'target group boundaries/dimensions differ')
    c,h = rg['group']['mid'],rg['group']['hidden']
    for g,precision in ((rg,'fp16'),(cg,'int8')):
        require(g['group']['dual'] == costs.projection(2*h,c,precision)
                and g['group']['down'] == costs.projection(c,h,precision), 'target group precision/stride differs')
    pairs=[];ref_means=[];got_means=[]
    for a,b in zip(ref['inputs'],got['inputs'],strict=True):
        require(a['input_index'] == b['input_index'] and a['input'] == b['input'], 'cost tensors/rows/order differ')
        means=[]
        for row in (a,b):
            g=one(row['groups'],'group_id')
            require(len(g['elapsed_ms_bits']) == len(g['elapsed_ms']) == 4, 'cost measurement count differs')
            values=[costs.f32_value(dict(elapsed_ms_bits=bits,elapsed_ms=value))
                    for bits,value in zip(g['elapsed_ms_bits'],g['elapsed_ms'],strict=True)]
            require(g['statistics'] == costs.statistics(values), 'cost input statistics differ from f32 observations')
            means.append(g['statistics']['mean_ms'])
        ref_means.append(means[0]);got_means.append(means[1])
        pairs.append(dict(input_index=a['input_index'],input=a['input'],reference_source=a['source'],candidate_source=b['source'],
            reference_mean_ms=means[0],candidate_mean_ms=means[1],saved_ms=means[0]-means[1]))
    ref_mean,got_mean=math.fsum(ref_means)/8,math.fsum(got_means)/8
    require(rg['equal_input_mean_ms'] == ref_mean and cg['equal_input_mean_ms'] == got_mean, 'cost mean weighting differs')
    wins=sum(p['saved_ms'] > 0 for p in pairs);saving=ref_mean-got_mean
    return dict(physical_batch=batch,group_id=group_id,reference_group=rg,candidate_group=cg,inputs=pairs,
        reference_mean_ms=ref_mean,candidate_mean_ms=got_mean,saved_target_group_ms=saving,input_wins=wins,
        cost_eligible=saving > 0 and wins >= 6)


def load_evidence(index_source, sources):
    """Consume completed metadata only; inherited blob SHA bindings are not revalidated here."""
    index=sources.json(index_source);check_index(index)
    inherited=directory(index['bound_sources']);generated=directory(index['output_sources'])
    authority=dict(inherited)
    for key,source in generated.items():
        require(key not in authority or same_source(source,authority[key]), 'input/output Source conflict')
        authority[key]=source
    for module in (io,comparisons,costs):
        source=record(module.__file__);comparisons._member(source,authority);sources.add(source)
    source=record(Path(__file__).with_name('run_ffn_calibration.py'))
    comparisons._member(source,authority);sources.add(source)
    registration=read_bound(index['registration'],authority,sources)
    require(registration['schema'] == 'rustgo-full-ffn-calibration-registration-v1'
            and registration['tactics'] == TACTICS and registration['no_retries'] is True
            and len(registration['cases']) == 80, 'registration strategy/coverage differs')
    aggregate=read_bound(index['capture'],authority,sources);comparisons._aggregate(aggregate,index['registration'])
    reports={r['name']:r for r in index['comparisons']};models={};summaries={}
    for case,item,receipt in zip(registration['cases'],index['all_case_cost_summaries'],aggregate['receipts'],strict=True):
        require(case['name'] == item['name'] == receipt['name'] and case['family'] in FAMILIES
                and case['name'] == f"{case['family']}-{case['recipe_index']:03}"
                and case['family'] == item['family'] and case['recipe_index'] == item['recipe_index']
                and case['role'] == item['role'] and case['group_ids'] == item['group_ids'], 'case directory differs')
        require(same_source(case['plan'],item['plan']) and same_source(receipt['result'],item['capture']), 'case sources differ')
        expected_role='FP16_REFERENCE' if item['recipe_index'] == 0 else 'SINGLE_FFN_INT8_ABLATION'
        require(item['role'] == expected_role and len(item['group_ids']) == (0 if item['recipe_index'] == 0 else 1), 'wrong case role')
        plan=read_bound(item['plan'],authority,sources)
        for key,expected in [('model_sha256',plan['model']['sha256']),('graph_sha256',plan['graph_sha256']),
                             ('recipe_sha256',plan['resolved_recipe_sha256']),('output_contract_sha256',registration['contracts'][case['family']])]:
            require(item[key] == expected, 'case identity differs: '+key)
        require(plan['group_ids'] == case['group_ids'], 'plan target differs')
        recipe=read_bound(plan['recipe'],authority,sources)
        require(recipe['model_sha256'] == item['model_sha256'] and recipe['graph_sha256'] == item['graph_sha256'], 'recipe model/graph differs')
        require(recipe_sha(recipe) == item['recipe_sha256'], 'canonical recipe SHA differs from admitted identity')
        summary=read_bound(item['cost_summary'],authority,sources)
        comparisons._cost(summary,item['capture'],case,plan,registration)
        for source in summary['bound_sources']:comparisons._member(source,authority)
        summaries[item['name']]=summary
        family=item['family']
        if item['recipe_index'] == 0:
            groups=recipe_groups(recipe,FAMILIES[family])
            models[family]=dict(reference=item,reference_recipe_source=plan['recipe'],
                recipe=compose_recipe(recipe,[]),groups=groups,evidence=[])
            continue
        model=models[family];group=item['group_ids'][0]
        require(group in model['groups'] and compose_recipe(model['recipe'],[group]) == recipe,
                'single-FFN recipe changes unrelated precision/dimensions or omits projections')
        row=reports[item['name']];reference=model['reference']
        require(row['family'] == family and row['group_ids'] == [group]
                and same_source(row['reference'],reference['capture']) and same_source(row['candidate'],item['capture'])
                and same_source(row['reference_cost_summary'],reference['cost_summary'])
                and same_source(row['candidate_cost_summary'],item['cost_summary']), 'comparison receipt differs')
        report=read_bound(row['report'],generated,sources)
        comparisons.check_report(report,dict(reference=reference,candidate=item),authority)
        observed,numeric_ok=sensitivity(report)
        require(row['calibration_6pp_observation'] == report['calibration_6pp_observation'], 'receipt threshold differs')
        evidence=dict(group_id=group,source_case=item['name'],capture=item['capture'],
            report=row['report'],recipe_source=plan['recipe'],recipe_sha256=item['recipe_sha256'],
            reference_cost_summary=reference['cost_summary'],candidate_cost_summary=item['cost_summary'],
            sensitivity=observed,numeric_eligible=numeric_ok,costs={})
        for b in BATCHES:
            evidence['costs'][b]=target_cost_pair(summaries[reference['name']],summary,group,b)
        model['evidence'].append(evidence)
    require(list(models) == list(FAMILIES), 'model coverage differs')
    for family,model in models.items():
        require(len(model['evidence']) == FAMILIES[family]
                and sorted(e['group_id'] for e in model['evidence']) == model['groups'], 'single-group evidence incomplete/duplicate')
    return models,index


def nominate(models):
    """Deterministic bounded heuristic over already validated evidence; no I/O."""
    nominations=[];unique={}
    for family in FAMILIES:
        model=models[family]
        for batch in BATCHES:
            def rank(e):
                m=e['sensitivity'];saving=e['costs'][batch]['saved_target_group_ms']
                return (-saving/max(m['win_max_abs_pp'],0.01),m['policy_kl_mean'],m['score_mean_max_abs'],e['group_id'])
            eligible=sorted((e for e in model['evidence'] if e['numeric_eligible'] and e['costs'][batch]['cost_eligible']),key=rank)
            for budget in BUDGETS_PP:
                selected=[];decisions=[];risks=[]
                for e in eligible:
                    risk=e['sensitivity']['win_max_abs_pp']
                    take=math.fsum(risks+[risk]) <= budget
                    decisions.append(dict(group_id=e['group_id'],selected=take,reason='selected' if take else 'risk_proxy_budget',
                        rank_score=-rank(e)[0],risk_proxy_pp=risk,saved_target_group_ms=e['costs'][batch]['saved_target_group_ms']))
                    if take:selected.append(e['group_id']);risks.append(risk)
                selected.sort();recipe=compose_recipe(model['recipe'],selected);identity=recipe_sha(recipe)
                alias=dict(family=family,physical_batch=batch,risk_proxy_budget_pp=budget,selected_groups=selected,
                    risk_proxy_used_pp=math.fsum(risks),decisions=decisions,
                    excluded_groups=[dict(group_id=e['group_id'],numeric_eligible=e['numeric_eligible'],
                        cost_eligible=e['costs'][batch]['cost_eligible']) for e in model['evidence'] if e not in eligible])
                if not selected:
                    alias.update(role='FP16_REFERENCE',reference=model['reference'],recipe_sha256=identity)
                else:
                    key=(recipe['model_sha256'],identity)
                    if key not in unique:
                        selected_evidence=[e for e in model['evidence'] if e['group_id'] in selected]
                        unique[key]=dict(id=family+'-'+identity[:16],family=family,recipe=recipe,
                            recipe_sha256=identity,recipe_object_sha256=object_sha(recipe),
                            role='SINGLE_FFN_INT8_ABLATION' if len(selected) == 1 else 'FFN_INT8_MIXTURE',
                            group_ids=selected,aliases=[],evidence=[e for e in model['evidence'] if e['group_id'] in selected],
                            status='REUSE_OBSERVED_SINGLE_FFN' if len(selected) == 1 else 'PROPOSED_UNEXECUTED',
                            precision_pair='int8/int8',reuse_capture=selected_evidence[0]['capture'] if len(selected) == 1 else None,
                            predicted_group_savings_only=True,whole_graph_accuracy_unverified=True,performance_unverified=True,
                            new_gpu_calls_authorized=0,rust_recipe_resolution='REQUIRED_BEFORE_EXECUTION')
                    require(unique[key]['recipe'] == recipe, 'recipe object hash collision')
                    alias.update(role=unique[key]['role'],candidate_id=unique[key]['id'],recipe_sha256=identity)
                    unique[key]['aliases'].append(dict(physical_batch=batch,risk_proxy_budget_pp=budget,
                        risk_proxy_used_pp=alias['risk_proxy_used_pp']))
                nominations.append(alias)
    require(len(nominations) == 18 and len(unique) <= 18, 'proposal generation exceeded fixed budget')
    return list(unique.values()),nominations


def plan(index_path, expected_sha, output):
    output=Path(output)
    require(output.is_absolute() and not output.exists(), 'fresh absolute planning output required')
    output.mkdir(exist_ok=False);sources=io.Sources();outputs=io.Sources();anchor=None
    try:
        for path in (Path(__file__),Path(sys.executable)):sources.add(record(path))
        # Check on every plan() call, even if this process cached the pure AST.
        helper=record(RECIPE_HELPER)
        require(helper['sha256'] == RECIPE_HELPER_SHA, 'pinned recipe serializer changed')
        sources.add(helper)
        anchor=sources.anchor(index_path,expected_sha)
        outputs.add(comparisons.put(output/'intent.json',dict(schema='rustgo-ffn-mixture-planning-attempt-v1',
            comparison_index=anchor,strategy=STRATEGY,scope=SCOPE,model_forwards=0,gpu_calls=0,ledger_writes=0)))
        models,index=load_evidence(anchor,sources);sources.recheck();before=list(sources.items.values())
        outputs.add(comparisons.put(output/'source-before.json',before))
        candidates,aliases=nominate(models)
        manifests=[]
        for family,model in models.items():
            reference=model['reference']
            entry=dict(role='FP16_REFERENCE',group_ids=[],recipe=model['recipe'],recipe_sha256=reference['recipe_sha256'],
                recipe_object_sha256=object_sha(model['recipe']),recipe_source=model['reference_recipe_source'],
                status='REUSE_OBSERVED_FP16_REFERENCE',precision_pair='fp16/fp16',reuse_capture=reference['capture'],aliases=[])
            recipes=[entry]
            for candidate in (c for c in candidates if c['family'] == family):
                candidate['recipe_source']=comparisons.put(output/(candidate['id']+'.recipe.json'),candidate['recipe'])
                outputs.add(candidate['recipe_source']);recipes.append(candidate)
            manifest=dict(schema='rustgo-ffn-mixture-calibration-recipe-proposal-v1',status='PROPOSED_UNVERIFIED_MIXTURES',
                family=family,model_sha256=reference['model_sha256'],graph_sha256=reference['graph_sha256'],
                output_contract_sha256=reference['output_contract_sha256'],reference_recipe_sha256=reference['recipe_sha256'],
                comparison_index=anchor,groups=model['evidence'],recipes=recipes,
                strategy=STRATEGY,
                proposal_limit=dict(max_nonempty_recipes=9,actual_nonempty_recipes=len(recipes)-1,new_gpu_calls_authorized=0),
                predicted_group_savings_only=True,whole_graph_accuracy_unverified=True,performance_unverified=True,
                scope=SCOPE,source_registry_touched=False)
            manifest['admission']=admission_projection(manifest)
            manifest['admission_sha256']=io.sha(admission_bytes(manifest['admission']))
            # Full object identity includes measured floats and is Python-only.
            # Rust v2 independently hashes admission, never this full object.
            manifest['proposal_sha256']=object_sha(manifest)
            source=comparisons.put(output/(family+'-proposals.json'),manifest);outputs.add(source)
            manifests.append(dict(family=family,proposal=source,proposal_sha256=manifest['proposal_sha256'],
                admission_sha256=manifest['admission_sha256']))
        sources.recheck();outputs.recheck()
        require(before == list(sources.items.values()), 'planning input source inventory changed')
        outputs.add(comparisons.put(output/'source-after.json',before))
        return comparisons.put(output/'proposals.json',dict(schema='rustgo-ffn-mixture-proposals-v1',
            status='PROPOSED_UNVERIFIED_MIXTURES',comparison_index=anchor,registration=index['registration'],
            capture=index['capture'],strategy=STRATEGY,scope=SCOPE,models=manifests,nominations=aliases,
            fixed_tactics=TACTICS,unique_nonempty_recipes=len(candidates),aliases=18,
            model_forwards=0,gpu_calls=0,worker_calls=0,ledger_writes=0,adopted=False,production_certified=False,
            predicted_group_savings_only=True,whole_graph_accuracy_unverified=True,performance_unverified=True,
            admission_encoding='UTF-8 compact JSON, recursively sorted string object keys, no forced ASCII escaping; '
                'only string, u64 integer, boolean, null, array and object; floats forbidden. '
                'v2 proposal_object_sha256 binds admission_sha256; full manifest remains bound by its file Source.',
            sources_unchanged=True,verified_metadata_sources=before,output_sources=list(outputs.items.values()),
            inherited_evidence_scope='Upstream full-capture and comparison producer claims remain bound by index SHA; '
                'this planner rechecks opened metadata, not raw tensors, PB blobs, model bytes or full transitive sources.',
            followup=['Actual Rust parser/graph admission of every new full recipe.',
                'Ledger identity check, including inherited consumed/negative attempts; no retry authorization.',
                'Separately frozen finite whole-graph calibration budget.',
                'Complete old FP16/INT8 control matrix, actual execution/shape identity, and ABBA performance gates.']))
    except BaseException as failure:
        comparisons.put(output/'FAILED.json',dict(status='FAILED_CPU_PLANNING_STOPPED',comparison_index=anchor,
            error=dict(type=type(failure).__name__,message=str(failure),traceback=traceback.format_exc()),
            gpu_calls=0,ledger_writes=0,no_retry_or_recollection_authorized=True))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',required=True,type=Path)
    parser.add_argument('--index-sha256',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();source=plan(args.index,args.index_sha256,args.output)
    try:print(io.canonical(source).decode(),flush=True)
    except OSError:pass


if __name__ == '__main__':main()
