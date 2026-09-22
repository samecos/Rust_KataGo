#!/usr/bin/env python3
"""Fit a small output-bias correction on a separate SGF, then reject or validate it.

This is an offline post-hoc calibration experiment, not activation calibration,
QAT, an Elo test, or a change to the inference engine. Parameters are fitted only
on the supplied SGF. The existing C++ FP32 fixture is used only after fitting.
"""
import argparse
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import autotune_reference as reference_io
from compare_worker_outputs import build_requests, collect_worker, compare_outputs, file_hash, write_json
from tune_runtime import ROOT, clean_environment
from validate_int8_backend import quantization_metrics
from worker_protocol_tools import Protocol


def canonical(moves):
    variants = []
    for swap in (False, True):
        for fx in (False, True):
            for fy in (False, True):
                transformed = []
                for color, vertex in moves:
                    if vertex == -1:
                        transformed.append((color, vertex))
                        continue
                    x, y = vertex % 19, vertex // 19
                    if swap:
                        x, y = y, x
                    transformed.append((color, (18 - y if fy else y) * 19 + (18 - x if fx else x)))
                variants.append(tuple(transformed))
    return min(variants)


def logit(p):
    p = max(1e-8, min(1 - 1e-8, p))
    return math.log(p / (1 - p))


def margin(output):
    total = output['white_win_prob'] + output['white_loss_prob']
    return logit(output['white_win_prob'] / total) if total else 0.0


def affine(xs, ys, weights, bias_limit):
    # Ridge towards the identity; bounded slope prevents tail extrapolation.
    aa = sum(w*x*x for x, w in zip(xs, weights)) + 1.0
    ab = sum(w*x for x, w in zip(xs, weights))
    bb = sum(weights) + 1.0
    ay = sum(w*x*(y-x) for x, y, w in zip(xs, ys, weights))
    by = sum(w*(y-x) for x, y, w in zip(xs, ys, weights))
    det = aa*bb - ab*ab
    return [max(.98, min(1.02, 1 + (ay*bb-by*ab)/det)),
            max(-bias_limit, min(bias_limit, (by*aa-ay*ab)/det))]


def fit(teacher, candidate, signs):
    a = [r['result']['output'] for r in teacher['results']]
    b = [r['result']['output'] for r in candidate['results']]
    s = [signs[r['name']] for r in teacher['results']]
    weights = [max(.01, x['white_win_prob'] * x['white_loss_prob']) for x in a]
    value = affine([t*margin(x) for t, x in zip(s, b)], [t*margin(x) for t, x in zip(s, a)], weights, .05)
    score = affine([t*x['white_score_mean'] for t, x in zip(s, b)],
                   [t*x['white_score_mean'] for t, x in zip(s, a)], [1.0]*len(a), .25)
    # Convex cross entropy in inverse temperature. A positive temperature
    # preserves policy ordering; it cannot recover an already changed top-1.
    def gradient(beta):
        total = 0.0
        for x, y in zip(a, b):
            pairs = [(p, math.log(max(q, 1e-30))) for p, q in zip(x['policy'], y['policy']) if p >= 0]
            peak = max(v for _, v in pairs)
            exp = [math.exp(beta*(v-peak)) for _, v in pairs]
            total += sum(e*v for e, (_, v) in zip(exp, pairs))/sum(exp) - sum(p*v for p, v in pairs)
        return total
    lo, hi = .98, 1.02
    for _ in range(48):
        mid = (lo + hi)/2
        if gradient(mid) > 0:
            hi = mid
        else:
            lo = mid
    return dict(value_margin_affine=value, score_affine=score, policy_inverse_temperature=(lo+hi)/2)


def apply(report, params, signs):
    result = copy.deepcopy(report)
    for row in result['results']:
        o = row['result']['output']; sign = signs[row['name']]
        a, b = params['value_margin_affine']
        value = sign * (a * sign * margin(o) + b)
        win = 1 / (1 + math.exp(-max(-80, min(80, value))))
        decisive = 1 - o['white_no_result_prob']
        o['white_win_prob'], o['white_loss_prob'] = decisive*win, decisive*(1-win)
        a, b = params['score_affine']; old = o['white_score_mean']
        new = a*old + sign*b
        variance = max(0.0, o['white_score_mean_sq'] - old*old)
        o['white_score_mean'], o['white_score_mean_sq'] = new, a*a*variance + new*new
        beta = params['policy_inverse_temperature']
        policy = [math.pow(max(p, 1e-30), beta) if p >= 0 else -1.0 for p in o['policy']]
        norm = sum(p for p in policy if p >= 0)
        o['policy'] = [p/norm if p >= 0 else p for p in policy]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--fixture', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--accuracy-reports', nargs='+', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    fixture = json.loads(args.fixture.read_text(encoding='utf-8'))
    if {len(p['moves']) % 2 for p in fixture['positions']} != {0, 1}:
        ap.error('calibration must include both players to move')
    heldout = json.loads((ROOT/'scripts/fixtures/worker_positions.json').read_text(encoding='utf-8'))
    train_keys = {canonical(p['moves']) for p in fixture['positions']}
    if fixture['source_sgf_sha256'] == heldout['source_sgf_sha256'] or train_keys & {canonical(p['moves']) for p in heldout['positions']}:
        ap.error('calibration and validation histories overlap (including board symmetries)')
    for a in fixture['positions']:
        for b in heldout['positions']:
            length = min(len(a['moves']), len(b['moves']))
            if length >= 8 and canonical(a['moves'][:length]) == canonical(b['moves'][:length]):
                ap.error('calibration and validation share a game prefix of at least 8 plies')
    binary_sha, model_sha = file_hash(args.binary), file_hash(args.model)
    profiles = []
    for path in args.accuracy_reports:
        r = json.loads(path.read_text(encoding='utf-8'))
        if r['status'] != 'COLLECTED' or r['binary_sha256'] != binary_sha or r['model_sha256'] != model_sha:
            ap.error('accuracy report must match the exact binary and model')
        profiles.append((path, r))
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = Protocol(ROOT/'crates/kata_worker/proto/worker.proto')
    report = dict(status='RUNNING', production_certified=False, calibration_kind='offline-output-bias-trial',
        teacher='same-build FP16; independently checked against C++ FP32 on validation fixtures',
        model_sha256=model_sha, binary_sha256=binary_sha, calibration_fixture_sha256=file_hash(args.fixture),
        source_sgf_sha256=fixture['source_sgf_sha256'], calibration_histories=len(fixture['positions']),
        calibration_games=1, heldout_history_overlap=0, profiles=[])
    try:
        requests = build_requests(protocol.pb, fixture, model_sha, 0)
        fingerprints = reference_io.fingerprints(requests, model_sha)
        signs = {name: (1 if req.position.next_player == protocol.pb.WHITE else -1) for name, req in requests}
        held_requests = build_requests(protocol.pb, heldout, model_sha, 0)
        held_signs = {name: (1 if req.position.next_player == protocol.pb.WHITE else -1) for name, req in held_requests}
        gold = reference_io.validate_reference(reference_io.read_reference(args.reference), reference_io.fingerprints(held_requests, model_sha), held_requests, protocol)
        options = SimpleNamespace(output=args.output, model=args.model, request_window=32, startup_timeout=180, task_timeout=180)
        base = 'rules=chinese\nkomi=7.5\nnnMaxBatchSize=8\nnumNNServerThreadsPerModel=1\nnnCacheSizePowerOfTwo=10\nnnMutexPoolSizePowerOfTwo=8\n'
        environment = clean_environment()
        cfg = args.output/'fp16.cfg'; cfg.write_text(base+'nnBackend=cudabackend\n', encoding='utf-8')
        teacher = collect_worker('fp16', args.binary, cfg, options, protocol, requests, fingerprints, environment)
        for path, accuracy in profiles:
            width = accuracy.get('min_ffn_width', 0)
            label = f'ffn-min{width}'
            cfg = args.output/f'{label}.cfg'
            cfg.write_text(base+f'nnBackend=cudaint8backend\ncudaInt8Scope=ffn\ncudaInt8MinFfnWidth={width}\n', encoding='utf-8')
            candidate = collect_worker(label, args.binary, cfg, options, protocol, requests, fingerprints, environment)
            if f'int8-min-ffn-width={width};' not in candidate['hello']['backend_info']:
                raise AssertionError('calibration worker precision identity mismatch')
            params = fit(teacher, candidate, signs)
            entry = dict(min_ffn_width=width, parameters=params, training_before=quantization_metrics(teacher, candidate),
                         training_after=quantization_metrics(teacher, apply(candidate, params, signs)), validation=[])
            for window in (1, 32):
                original = json.loads((path.parent/f'ffn-w{window}/outputs.json').read_text(encoding='utf-8'))
                corrected = apply(original, params, held_signs)
                write_json(args.output/f'{label}-corrected-w{window}.json', corrected)
                gate = compare_outputs(gold, corrected)
                entry['validation'].append(dict(window=window, before=quantization_metrics(gold, original),
                    after=quantization_metrics(gold, corrected), original_fp32_gate=gate))
            entry['precision_restored'] = all(v['original_fp32_gate']['result'] == 'PASS' for v in entry['validation'])
            report['profiles'].append(entry)
            write_json(args.output/'report.json', report)
            print(json.dumps(dict(min_ffn_width=width, parameters=params, precision_restored=entry['precision_restored'])), flush=True)
        report['status'] = 'COLLECTED'
    except Exception as error:
        report.update(status='FAIL', error=str(error))
        raise
    finally:
        protocol.close()
        write_json(args.output/'report.json', report)


if __name__ == '__main__':
    main()
