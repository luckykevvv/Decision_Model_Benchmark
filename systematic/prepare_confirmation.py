"""Freeze v2 core/robustness input matrices before observing model outputs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from systematic.specs import ROOT, TASKS, key_for, NONE_DESCRIPTION

ALIGNED_NONE = {
    'ag_news': 'None of the listed topics correctly describes the main topic of this news article',
    'dbpedia': 'None of the listed categories correctly describes the main entity in this text',
    'emotion': 'None of the listed emotions best describes the emotion expressed in this text',
    'trec': 'None of the listed answer types matches the type of answer this question asks for',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def rng_for(*parts):
    return random.Random(int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:16], 16))


def build_requests(cases, profile):
    requests = []
    for case in cases:
        task, gold = case['task'], case['label_id']
        all_labels = list(range(len(TASKS[task]['labels'])))
        wrong = [label for label in all_labels if label != gold]
        orders = [0] if profile == 'core' else [0, 1]
        members = [0] if profile == 'core' else list(range(5))
        counts = sorted({2, min(4, len(all_labels)-1), len(all_labels)-1})
        conditions = []
        for order in orders:
            # Full controls only vary order, never duplicate member seeds.
            ordered = all_labels[:]
            rng_for(20261003, case['case_id'], 'full-order', order).shuffle(ordered)
            conditions.append(('full', len(all_labels), ordered, None, 0, order))
        for k in counts:
            fixed_absent = rng_for(20261003, case['case_id'], 'controlled-base', k).sample(wrong, k)
            fixed_slot = rng_for(20261003, case['case_id'], 'controlled-slot', k).randrange(k)
            retained = {label for i, label in enumerate(fixed_absent) if i != fixed_slot}
            alternatives = [label for label in wrong if label not in retained]
            rng_for(20261003, case['case_id'], 'controlled-alternatives', k).shuffle(alternatives)
            for member in members:
                absent = rng_for(20261003, case['case_id'], 'member', k, member).sample(wrong, k)
                if profile == 'robustness-controlled':
                    absent = fixed_absent[:]
                    absent[fixed_slot] = alternatives[member % len(alternatives)]
                for order in orders:
                    ordered = absent[:]
                    rng = rng_for(20261003, case['case_id'], 'pair-order', k, member, order)
                    if profile == 'robustness-controlled':
                        permutation = list(range(k))
                        rng_for(20261003, case['case_id'], 'controlled-order', k, order).shuffle(permutation)
                        present = absent[:]
                        present[fixed_slot] = gold
                        ordered = [absent[i] for i in permutation]
                        present = [present[i] for i in permutation]
                    else:
                        rng.shuffle(ordered)
                        present = ordered[:]
                        present[rng.randrange(k)] = gold
                    pair = f"{case['case_id']}:k{k}:m{member}:o{order}"
                    conditions += [('present', k, present, pair, member, order), ('absent', k, ordered, pair, member, order)]
        for mode in ('natural', 'neutral'):
            for presence, k, ordinary, basepair, member, order in conditions:
                for with_none in (False, True):
                    wording_modes = ['aligned']
                    if profile != 'core' and with_none and member == 0 and order == 0:
                        wording_modes.append('legacy')
                    for none_mode in wording_modes:
                        criteria = {key_for(label, task, mode): TASKS[task]['labels'][label][1] for label in ordinary}
                        if with_none:
                            description = ALIGNED_NONE[task] if none_mode == 'aligned' else NONE_DESCRIPTION
                            # Same NONE slot across present/absent and name/wording variants.
                            items = list(criteria.items())
                            slot = rng_for(20261003, case['case_id'], 'none-slot', k, 0 if profile == 'robustness-controlled' else member, order, presence == 'full').randrange(k+1)
                            items.insert(slot, ('none', description))
                            criteria = dict(items)
                        request = {'case_id': case['case_id'], 'task': task, 'split': case['split'],
                                   'presence': presence, 'candidate_size': k, 'name_mode': mode,
                                   'with_none': with_none, 'replicate': member*2+order,
                                   'member_seed': member, 'order_seed': order, 'none_mode': none_mode,
                                   'pair_id': f'{basepair}:{mode}:{int(with_none)}:{none_mode}' if basepair else None,
                                   'instruction': TASKS[task]['instruction'], 'criteria': criteria,
                                   'gold_option': key_for(gold, task, mode) if presence != 'absent' else None,
                                   'gold_label': case['label']}
                        payload = {**request, 'criteria': list(criteria.items())}
                        request['request_id'] = hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()
                        requests.append(request)
    return requests


def audit(cases, requests):
    assert len({r['request_id'] for r in requests}) == len(requests)
    case_map = {c['case_id']: c for c in cases}
    pairs = defaultdict(dict)
    names = defaultdict(dict)
    for r in requests:
        assert r['task'] == case_map[r['case_id']]['task'] and r['split'] == case_map[r['case_id']]['split']
        ordinary = [key for key in r['criteria'] if key != 'none']
        assert len(ordinary) == r['candidate_size'] and ('none' in r['criteria']) == r['with_none']
        assert (r['gold_option'] in ordinary) if r['presence'] != 'absent' else r['gold_option'] is None
        if r['pair_id']:
            assert r['presence'] not in pairs[r['pair_id']]
            pairs[r['pair_id']][r['presence']] = r
        signature = (r['case_id'], r['presence'], r['candidate_size'], r['member_seed'], r['order_seed'], r['with_none'], r['none_mode'])
        names[signature][r['name_mode']] = r
    for pair in pairs.values():
        assert set(pair) == {'present', 'absent'}
        p, a = pair['present'], pair['absent']
        assert list(p['criteria'].values()).count(p['criteria'][p['gold_option']]) >= 1
        assert set(p['criteria']) - set(a['criteria']) == {p['gold_option']}
        assert len(set(a['criteria']) - set(p['criteria'])) == 1
        pi, ai = list(p['criteria']), list(a['criteria'])
        assert len(pi) == len(ai)
        assert sum(x != y for x, y in zip(pi, ai)) == 1
        for key in set(p['criteria']) & set(a['criteria']):
            assert pi.index(key) == ai.index(key) and p['criteria'][key] == a['criteria'][key]
    for variants in names.values():
        assert set(variants) == {'natural', 'neutral'}
        assert list(variants['natural']['criteria'].values()) == list(variants['neutral']['criteria'].values())
    return {'n_cases': len(cases), 'n_requests': len(requests), 'n_pairs': len(pairs),
            'paired_slot_checks_passed': True, 'matched_naming_checks_passed': True}


def prepare(profile, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Refusing to overwrite frozen confirmation inputs')
    index = ROOT / 'outputs/paper1-confirmation-index-20261003'
    cases = [json.loads(line) for line in (index / 'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    if profile != 'core':
        plan = json.loads((ROOT / 'outputs/paper1-confirmation-analysis-plan-20261003/subsets.json').read_text(encoding='utf-8'))
        selected = {key for subset in plan['robustness_subset'].values() for key in subset['case_ids']}
        cases = [case for case in cases if case['case_id'] in selected]
    requests = build_requests(cases, profile)
    checks = audit(cases, requests)
    if profile == 'core':
        assert checks['n_requests'] == 39168
    output.mkdir(parents=True)
    write_rows(output / 'cases.jsonl', cases)
    write_rows(output / 'requests.jsonl', requests)
    membership = defaultdict(set)
    for r in requests:
        if r['presence'] == 'absent' and r['name_mode'] == 'natural' and not r['with_none']:
            membership[f"{r['case_id']}|k{r['candidate_size']}"] .add(tuple(sorted(r['criteria'])))
    manifest = {'protocol': 'paper1-confirmation-v2', 'profile': 'robustness' if profile != 'core' else profile,
                'design_id': profile, 'seed': 20261003,
                'stage': 'reference-label confirmation',
                'n_cases': len(cases), 'n_requests': len(requests), 'task_specs': TASKS,
                'aligned_none': ALIGNED_NONE, 'human_review_status': 'not_completed',
                'index_sha256': digest(index / 'cases.jsonl'),
                'protocol_sha256': digest(ROOT / 'systematic/CONFIRMATION_PROTOCOL.md'),
                'code_sha256': digest(Path(__file__)),
                'model': {'repo': 'convaiinnovations/laya', 'revision': '55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851',
                          'device': 'cuda', 'threads': 4, 'max_len': 512, 'head_max_len': 256},
                'file_sha256': {name: digest(output / name) for name in ('cases.jsonl', 'requests.jsonl')},
                'counts': dict(Counter(f"{c['task']}/{c['split']}/{c['label']}" for c in cases)),
                'unique_absent_membership_per_case_k': {k: len(v) for k, v in membership.items()}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    (output / 'plan_audit.json').write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=['core', 'robustness', 'robustness-controlled'], required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    prepare(args.profile, args.output_dir)
