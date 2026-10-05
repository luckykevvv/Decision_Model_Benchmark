"""Validated paired metrics; operating thresholds use calibration data only."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

class ValidationError(ValueError):
    pass

def require(condition, message):
    if not condition:
        raise ValidationError(message)

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]

def unique(rows, field):
    mapping = {row[field]: row for row in rows}
    require(len(mapping) == len(rows), f'Duplicate {field}')
    return mapping

def auc(positive, negative):
    if len(positive) == 0 or len(negative) == 0:
        return None
    positive, negative = np.asarray(positive), np.sort(negative)
    ranks = (np.searchsorted(negative, positive, 'left') + np.searchsorted(negative, positive, 'right')) / 2
    return float(ranks.sum() / (len(positive) * len(negative)))

def wilson(successes, n):
    if not n:
        return None
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return [center-radius, center+radius]

def rate(values):
    values = np.asarray(values, dtype=bool)
    n, successes = len(values), int(values.sum())
    return {'n': n, 'successes': successes, 'value': float(values.mean()) if n else None, 'wilson95': wilson(successes, n)}

def entropy(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    if len(probabilities) <= 1:
        return 0.0
    # Rounded API probabilities are normalized before computing entropy.
    probabilities = probabilities / probabilities.sum()
    positive = probabilities[probabilities > 0]
    return float(-(positive * np.log(positive)).sum() / np.log(len(probabilities)))

def select_threshold(present_scores, absent_scores, limit=.05):
    present_scores, absent_scores = np.asarray(present_scores), np.asarray(absent_scores)
    require(len(present_scores) > 0 and len(absent_scores) > 0, 'Calibration pairs required')
    candidates = np.unique(np.concatenate([present_scores, absent_scores]))
    best = None
    for threshold in candidates:
        false_rate = float((present_scores > threshold).mean())
        detection = float((absent_scores > threshold).mean())
        if false_rate <= limit:
            objective = (detection, -false_rate, float(threshold))
            if best is None or objective > best[0]:
                best = (objective, float(threshold))
    require(best is not None, 'No calibration threshold satisfies constraint')
    return best[1]

def validate(run_dir):
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    for filename, expected in manifest['file_sha256'].items():
        require(hashlib.sha256((run_dir / filename).read_bytes()).hexdigest() == expected, f'Changed frozen file: {filename}')
    cases = unique(read_jsonl(run_dir / 'cases.jsonl'), 'case_id')
    requests = unique(read_jsonl(run_dir / 'requests.jsonl'), 'request_id')
    predictions = unique(read_jsonl(run_dir / 'predictions.jsonl'), 'request_id')
    require(set(predictions) == set(requests), 'Incomplete or extraneous predictions')
    seen_text = set()
    for case in cases.values():
        require(case['split'] in ['calibration', 'test'], 'Unknown split')
        text_identity = (case['task'], ' '.join(case['text'].casefold().split()))
        require(text_identity not in seen_text, 'Duplicate text or split leakage')
        seen_text.add(text_identity)
    full, pairs, cells = {}, defaultdict(dict), {}
    for request_id, request in requests.items():
        require(request['case_id'] in cases, 'Unknown case')
        case = cases[request['case_id']]
        for field in ['task', 'split']:
            require(request[field] == case[field], f'Mismatched {field}')
        require(request['gold_label'] == case['label'], 'Mismatched gold label')
        require(request['replicate'] == 0, 'Multiple candidate replicates are not supported in v1')
        require(request['name_mode'] in ['natural', 'neutral'], 'Unknown option name mode')
        require(type(request['with_none']) is bool, 'with_none must be boolean')
        require(request['presence'] in ['full', 'present', 'absent'], 'Unknown presence')
        criteria = request['criteria']
        require(bool(criteria) and all(isinstance(description, str) and description for description in criteria.values()), 'Invalid criteria')
        ordinary = set(criteria) - {'none'}
        require(len(ordinary) == request['candidate_size'], 'Wrong candidate count')
        require(('none' in criteria) == request['with_none'], 'Wrong none membership')
        expected_gold = case['label'] if request['name_mode'] == 'natural' else f'option_{case["label_id"]:03d}'
        if request['presence'] == 'absent':
            require(request['gold_option'] is None and expected_gold not in ordinary, 'Absent request offers the reference class')
        else:
            require(request['gold_option'] == expected_gold and expected_gold in ordinary, 'Wrong reference option')
        cell = (case['case_id'], request['name_mode'], request['presence'], request['candidate_size'], request['with_none'])
        require(cell not in cells, 'Duplicated experimental cell')
        cells[cell] = request
        if request['presence'] == 'full':
            require(request['pair_id'] is None, 'Full request has pair_id')
            full[(case['case_id'], request['name_mode'], request['with_none'])] = request
        else:
            require(bool(request['pair_id']), 'Missing pair_id')
            require(request['presence'] not in pairs[request['pair_id']], 'Duplicated pair member')
            pairs[request['pair_id']][request['presence']] = request
        prediction = predictions[request_id]
        probabilities = prediction['probabilities']
        require(set(probabilities) == set(criteria), 'Wrong probability keys')
        require(all(type(value) in [float, int] and math.isfinite(value) and 0 <= value <= 1 for value in probabilities.values()), 'Invalid probability values')
        decimals=manifest.get('reported_probability_decimals')
        require(decimals is None or type(decimals)==int and 1<=decimals<=8, 'Invalid declared probability precision')
        tolerance=.001 if decimals is None else .5*10**(-decimals)*len(probabilities)+1e-9
        require(sum(probabilities.values())>0 and abs(sum(probabilities.values()) - 1) <= tolerance, 'Probability sum differs from one beyond declared rounding tolerance')
        require(prediction['choice'] in criteria, 'Invalid choice')
    for case_id, mode, with_none in list(full):
        reference = full[(case_id, mode, with_none)]
        require((case_id, mode, not with_none) in full, 'Missing full-set condition')
        for other in [r for r in requests.values() if r['case_id'] == case_id and r['name_mode'] == mode]:
            for key, description in other['criteria'].items():
                if key != 'none':
                    require(key in reference['criteria'] and description == reference['criteria'][key], 'Changed class description or unknown class')
    for pair in pairs.values():
        require(set(pair) == {'present', 'absent'}, 'Incomplete matched pair')
        present, absent = pair['present'], pair['absent']
        for field in ['case_id', 'task', 'split', 'candidate_size', 'name_mode', 'with_none', 'replicate', 'instruction']:
            require(present[field] == absent[field], f'Pair differs on {field}')
        p, a = set(present['criteria']) - {'none'}, set(absent['criteria']) - {'none'}
        require(p-a == {present['gold_option']} and len(a-p) == 1, 'Pair must substitute exactly one ordinary option')
        for key in set(present['criteria']) & set(absent['criteria']):
            require(present['criteria'][key] == absent['criteria'][key], 'Pair changes retained description')
    for case in cases.values():
        for mode in ['natural', 'neutral']:
            for with_none in [False, True]:
                require((case['case_id'], mode, with_none) in full, 'Missing baseline for case')
    return manifest, cases, requests, predictions, full

def bootstrap(detections, false_rejections, scores, replicates, seed):
    n = len(detections)
    if not n:
        return {'delta': None, 'auc': {name: None for name in scores}}
    point = {'delta': float((detections-false_rejections).mean()), 'auc': {name: auc(absent, present) for name, (present, absent) in scores.items()}}
    samples = {'delta': [], **{name: [] for name in scores}}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        indices = rng.integers(0, n, size=n)
        samples['delta'].append(float((detections[indices]-false_rejections[indices]).mean()))
        for name, (present, absent) in scores.items():
            samples[name].append(auc(absent[indices], present[indices]))
    def ci(values):
        return [float(x) for x in np.percentile(values, [2.5, 97.5])] if values else None
    return {'delta': {'value': point['delta'], 'bootstrap95': ci(samples['delta'])},
            'auc': {name: {'value': value, 'bootstrap95': ci(samples[name])} for name, value in point['auc'].items()}}

def analyze(run_dir, bootstrap_replicates=2000, seed=20261001):
    run_dir = Path(run_dir)
    manifest, cases, requests, predictions, full = validate(run_dir)
    cells = {(r['case_id'], r['name_mode'], r['candidate_size'], r['presence'], r['with_none']): r for r in requests.values()}
    groups = sorted({(r['task'], r['candidate_size'], r['name_mode']) for r in requests.values() if r['presence'] == 'present'})
    baseline = {}
    for task, mode in sorted({(c['task'], mode) for c in cases.values() for mode in ['natural', 'neutral']}):
        group_cases = [c for c in cases.values() if c['task'] == task and c['split'] == 'test']
        baseline[f'{task}|{mode}'] = {}
        for with_none in [False, True]:
            values = [predictions[full[(c['case_id'], mode, with_none)]['request_id']]['choice'] == full[(c['case_id'], mode, with_none)]['gold_option'] for c in group_cases]
            baseline[f'{task}|{mode}']['with_none' if with_none else 'without_none'] = rate(values)
    summaries = {}
    for task, k, mode in groups:
        arrays = {}
        for split in ['calibration', 'test']:
            group_cases = sorted([c for c in cases.values() if c['task'] == task and c['split'] == split], key=lambda c: c['case_id'])
            detection, false_rejection, correctness, baseline_correct = [], [], [], []
            score_lists = {name: ([], []) for name in ['negative_max', 'entropy', 'none_probability']}
            present_confidence = []
            for case in group_cases:
                responses = {}
                for presence in ['present', 'absent']:
                    for with_none in [False, True]:
                        key = (case['case_id'], mode, k, presence, with_none)
                        require(key in cells, f'Missing experimental condition: {key}')
                        r = cells[key]
                        responses[(presence, with_none)] = (r, predictions[r['request_id']])
                detection.append(responses[('absent', True)][1]['choice'] == 'none')
                false_rejection.append(responses[('present', True)][1]['choice'] == 'none')
                r, p = responses[('present', False)]
                correctness.append(p['choice'] == r['gold_option'])
                present_confidence.append(max(p['probabilities'].values()))
                original = full[(case['case_id'], mode, False)]
                baseline_correct.append(predictions[original['request_id']]['choice'] == original['gold_option'])
                for index, presence in enumerate(['present', 'absent']):
                    plain = responses[(presence, False)][1]['probabilities']
                    score_lists['negative_max'][index].append(-max(plain.values()))
                    score_lists['entropy'][index].append(entropy(list(plain.values())))
                    score_lists['none_probability'][index].append(responses[(presence, True)][1]['probabilities']['none'])
            arrays[split] = {'d': np.asarray(detection, dtype=float), 'f': np.asarray(false_rejection, dtype=float),
                'correct': correctness, 'baseline_correct': np.asarray(baseline_correct, dtype=bool),
                'confidence': np.asarray(present_confidence), 'scores': {name: (np.asarray(p), np.asarray(a)) for name, (p, a) in score_lists.items()}}
        test, cal = arrays['test'], arrays['calibration']
        calibrated = {}
        for name, (present, absent) in cal['scores'].items():
            threshold = select_threshold(present, absent)
            p_test, a_test = test['scores'][name]
            calibrated[name] = {'threshold': threshold, 'calibration_d': rate(absent > threshold), 'calibration_f': rate(present > threshold),
                                'test_d': rate(a_test > threshold), 'test_f': rate(p_test > threshold)}
        correct = test['baseline_correct']
        conditional_scores = {name: (p[correct], a[correct]) for name, (p, a) in test['scores'].items()}
        summary = {'n_test': len(test['d']), 'n_calibration': len(cal['d']), 'candidate_size': k,
            'present_accuracy': rate(test['correct']), 'detection': rate(test['d']), 'false_rejection': rate(test['f']),
            'paired': bootstrap(test['d'], test['f'], test['scores'], bootstrap_replicates, seed),
            'baseline_correct_subset': {'n': int(correct.sum()), 'detection': rate(test['d'][correct]), 'false_rejection': rate(test['f'][correct]),
                'auc': {name: auc(a, p) for name, (p, a) in conditional_scores.items()}}, 'calibrated_thresholds': calibrated}
        curve = []
        error = ~np.asarray(test['correct'], dtype=bool)
        for threshold in sorted(set(test['confidence']), reverse=True):
            accepted = test['confidence'] >= threshold
            curve.append({'threshold': float(threshold), 'coverage': float(accepted.mean()), 'error_risk': float(error[accepted].mean()), 'n_accepted': int(accepted.sum())})
        summary['diagnostic_risk_coverage'] = curve
        summaries[f'{task}|k{k}|{mode}'] = summary
    return {'protocol': manifest.get('protocol'), 'profile': manifest.get('profile'), 'n_cases': len(cases), 'n_predictions': len(predictions),
        'bootstrap_replicates': bootstrap_replicates, 'bootstrap_seed': seed, 'baseline': baseline, 'groups': summaries,
        'notes': ['Calibration thresholds maximize empirical detection subject to present false rejection <= 0.05; ties prefer lower false rejection then higher threshold.',
                  'Reject iff score > threshold. This empirical calibration constraint has no population risk guarantee.',
                  'Bootstrap resamples paired items within task without class stratification. Intervals are marginal, not simultaneous.',
                  'Risk-coverage curves are test diagnostics, not test-tuned policy recommendations.',
                  'Gold-label deletion simulates absence under reference labels, not naturally novel categories.',
                  ('Jev native API rounded scores are retained, with sum tolerance implied by declared precision; entropy renormalizes them.' if manifest.get('model',{}).get('provider')=='TypeSafe' else 'Inference uses Laya package temperature handling, including clamping of the 11+ options bucket.') ]}

def render_report(metrics):
    lines = ['# Systematic candidate-coverage evaluation', '', f"Cases: {metrics['n_cases']}; predictions: {metrics['n_predictions']}.", '',
             '## Complete-set test accuracy', '', '| Task and option names | Without none | With none | Test n |', '|---|---:|---:|---:|']
    for key, baseline in metrics['baseline'].items():
        lines.append(f"| {key.replace('|', ' / ')} | {baseline['without_none']['value']:.3f} | {baseline['with_none']['value']:.3f} | {baseline['without_none']['n']} |")
    lines += ['', '## Paired candidate-set test results', '', '| Group | n | Detection | False rejection | D-F | AUROC max | AUROC entropy | AUROC none |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key, group in metrics['groups'].items():
        aucs = group['paired']['auc']
        lines.append(f"| {key.replace('|', ' / ')} | {group['n_test']} | {group['detection']['value']:.3f} | {group['false_rejection']['value']:.3f} | {group['paired']['delta']['value']:.3f} | {aucs['negative_max']['value']:.3f} | {aucs['entropy']['value']:.3f} | {aucs['none_probability']['value']:.3f} |")
    lines += ['', '## Calibration-only thresholds on the held-out test set', '', '| Group | Score | Threshold | Test D | Test F |', '|---|---|---:|---:|---:|']
    for key, group in metrics['groups'].items():
        for score, result in group['calibrated_thresholds'].items():
            lines.append(f"| {key.replace('|', ' / ')} | {score} | {result['threshold']:.5f} | {result['test_d']['value']:.3f} | {result['test_f']['value']:.3f} |")
    lines += ['', '## Interpretation boundaries', ''] + ['- ' + note for note in metrics['notes']]
    lines += ['', 'All Wilson intervals, paired bootstrap intervals, baseline-correct subsets and risk-coverage points are in metrics.json.']
    return '\n'.join(lines) + '\n'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--bootstrap-replicates', type=int, default=2000)
    args = parser.parse_args()
    metrics = analyze(args.run_dir, args.bootstrap_replicates)
    root = Path(args.run_dir)
    for name, text in [('metrics.json', json.dumps(metrics, indent=2, allow_nan=False) + '\n'), ('report.md', render_report(metrics))]:
        temporary = root / (name + '.tmp')
        temporary.write_text(text, encoding='utf-8')
        temporary.replace(root / name)
    print(f'Validated {metrics["n_predictions"]} predictions; wrote metrics.json and report.md')
