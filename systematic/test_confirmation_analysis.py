"""Known-value tests for systematic.confirmation_analysis.

The fixtures are small synthetic ``paper1-confirmation-v2`` run directories with
deterministic probabilities, so every asserted partition, threshold and
aggregation value below is derived by hand from the fixture design rather than
from the analyzer under test.
"""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

try:  # python -m systematic.test_confirmation_analysis
    from systematic.confirmation_analysis import (
        ValidationError, analyze, evaluate_policy, run, select_threshold_weighted, validate_run)
except ImportError:  # pragma: no cover - direct script execution from systematic/
    from confirmation_analysis import (  # type: ignore
        ValidationError, analyze, evaluate_policy, run, select_threshold_weighted, validate_run)


LABELS = {'alpha': ['A0', 'A1', 'A2'], 'beta': ['B0', 'B1', 'B2']}


def option_key(task, name_mode, label_id):
    return LABELS[task][label_id] if name_mode == 'natural' else f'option_{label_id:03d}'


def option_key(task, name_mode, label_id):
    return LABELS[task][label_id] if name_mode == 'natural' else f'option_{label_id:03d}'


def build_cases(n_cal, n_test):
    cases = []
    for task, labels in LABELS.items():
        for split, count in (('calibration', n_cal), ('test', n_test)):
            for label_id, label in enumerate(labels):
                for index in range(count):
                    case_id = f'{task}:{split}:{label_id}:{index}'
                    cases.append({'case_id': case_id, 'task': task, 'split': split, 'label': label,
                                  'label_id': label_id, 'text': f'{task} {split} {label} text {index}',
                                  'row_idx': index, 'source_split': 'train' if split == 'calibration' else 'test',
                                  'normalized_text_sha256': f'hash-{case_id}'})
    return cases


def criteria_for(task, name_mode, indices):
    return {option_key(task, name_mode, index): f'description {LABELS[task][index]}' for index in indices}


def make_prediction(request, discrete=False, legacy=False):
    criteria = request['criteria']
    ordinary = [key for key in criteria if key != 'none']
    gold = request['gold_option']
    presence = request['presence']
    if discrete:
        choice = gold if presence != 'absent' else ('none' if request['with_none'] else ordinary[0])
        return {'request_id': request['request_id'], 'choice': choice, 'probabilities': None}
    if presence == 'full':
        if request['with_none']:
            probabilities = {key: (0.7 if key == gold else 0.1) for key in ordinary}
            probabilities['none'] = 0.1
        else:
            probabilities = {key: (0.8 if key == gold else 0.1) for key in ordinary}
        choice = gold
    elif presence == 'present':
        if request['with_none']:
            if legacy:
                probabilities = {key: (0.70 if key == gold else 0.20 / (len(ordinary) - 1)) for key in ordinary}
                probabilities['none'] = 0.10
            else:
                probabilities = {key: (0.75 if key == gold else 0.20 / (len(ordinary) - 1)) for key in ordinary}
                probabilities['none'] = 0.05
        else:
            probabilities = {key: (0.8 if key == gold else 0.2 / (len(ordinary) - 1)) for key in ordinary}
        choice = gold
    else:  # absent
        if request['with_none']:
            if legacy:
                probabilities = {key: 0.20 / len(ordinary) for key in ordinary}
                probabilities['none'] = 0.80
            else:
                probabilities = {key: 0.10 / len(ordinary) for key in ordinary}
                probabilities['none'] = 0.90
            choice = 'none'
        else:
            probabilities = {key: 0.5 for key in ordinary}
            choice = ordinary[0]
    return {'request_id': request['request_id'], 'choice': choice, 'probabilities': probabilities}


def build_run(root, n_cal=2, n_test=2, member_seeds=(0,), order_seeds=(0,), with_legacy=False, discrete=False):
    cases = build_cases(n_cal, n_test)
    case_by_id = {case['case_id']: case for case in cases}
    requests, predictions = [], []
    for case in cases:
        task, mode = case['task'], None
        for name_mode in ('natural', 'neutral'):
            full_indices = list(range(3))
            others = [index for index in range(3) if index != case['label_id']]
            present_indices = [case['label_id'], others[0]]
            absent_indices = [others[1], others[0]]
            for member_seed in member_seeds:
                for order_seed in order_seeds:
                    variant = (member_seed, order_seed)
                    definitions = [
                        ('full', 3, full_indices, option_key(task, name_mode, case['label_id']), None),
                        ('present', 2, present_indices, option_key(task, name_mode, case['label_id']), variant),
                        ('absent', 2, absent_indices, None, variant),
                    ]
                    for presence, candidate_size, indices, gold_option, pair_variant in definitions:
                        for with_none in (False, True):
                            none_modes = ['aligned']
                            if with_none and with_legacy and variant == (0, 0):
                                none_modes.append('legacy')
                            for none_mode in none_modes:
                                pair_id = None
                                if pair_variant is not None:
                                    pair_id = (f"{case['case_id']}|{name_mode}|{member_seed}|{order_seed}|"
                                               f"{int(with_none)}|{none_mode}")
                                request_id = (f"{case['case_id']}|{name_mode}|{presence}|{candidate_size}|"
                                              f"{with_none}|{member_seed}|{order_seed}|{none_mode}")
                                criteria = criteria_for(task, name_mode, indices)
                                if with_none:
                                    criteria = dict(criteria)
                                    criteria['none'] = 'None of the options'
                                request = {
                                    'request_id': request_id, 'case_id': case['case_id'], 'task': task,
                                    'split': case['split'], 'presence': presence,
                                    'candidate_size': candidate_size, 'name_mode': name_mode,
                                    'with_none': with_none, 'replicate': 0, 'pair_id': pair_id,
                                    'instruction': 'Select the best option.', 'criteria': criteria,
                                    'gold_option': gold_option, 'gold_label': case['label'],
                                    'member_seed': member_seed, 'order_seed': order_seed, 'none_mode': none_mode}
                                requests.append(request)
                                predictions.append(make_prediction(request, discrete=discrete,
                                                                   legacy=(none_mode == 'legacy')))
    return cases, requests, predictions


def save(root, cases, requests, predictions, profile='core'):
    for name, rows in (('cases.jsonl', cases), ('requests.jsonl', requests), ('predictions.jsonl', predictions)):
        (root / name).write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in ('cases.jsonl', 'requests.jsonl')}
    (root / 'manifest.json').write_text(json.dumps({
        'protocol': 'paper1-confirmation-v2', 'profile': profile, 'model': {'name': 'toy'},
        'file_sha256': hashes}), encoding='utf-8')


def write_plan(root, cases, per_task=80):
    equal = {}
    for task in LABELS:
        ids = sorted(case['case_id'] for case in cases if case['task'] == task and case['split'] == 'calibration')
        equal[task] = {'case_ids': ids[:per_task], 'n_texts': min(per_task, len(ids)),
                       'per_class': {}}
    path = root / 'subsets.json'
    path.write_text(json.dumps({'equal_budget_calibration': equal, 'equal_budget_seed': 20261007}), encoding='utf-8')
    return path


class ConfirmationAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'run'
        self.root.mkdir()
        self.cases, self.requests, self.predictions = build_run(self.root)
        save(self.root, self.cases, self.requests, self.predictions)

    def tearDown(self):
        self.temp.cleanup()

    def reload(self, cases=None, requests=None, predictions=None):
        save(self.root, cases or self.cases, requests or self.requests, predictions or self.predictions)

    def find(self, requests, **fields):
        matches = [request for request in requests
                   if all(request[key] == value for key, value in fields.items())]
        self.assertEqual(len(matches), 1, f'expected one request for {fields}, got {len(matches)}')
        return matches[0]

    def prediction_for(self, predictions, request):
        return next(prediction for prediction in predictions if prediction['request_id'] == request['request_id'])

    def test_native_partitions_known_values(self):
        group = analyze(self.root, bootstrap_replicates=5, joint=False)['groups']['alpha|k2|natural|aligned']
        present = group['native']['test']['present']
        absent = group['native']['test']['absent']
        self.assertEqual(group['n_text']['test'], 6)
        self.assertEqual(group['n_variants']['test'], 6)
        self.assertEqual(present['correct']['text_mean'], 1.0)
        self.assertEqual(present['wrong']['text_mean'], 0.0)
        self.assertEqual(present['rejected']['text_mean'], 0.0)
        self.assertEqual(absent['rejected']['text_mean'], 1.0)
        self.assertEqual(absent['accepted']['text_mean'], 0.0)

        requests = copy.deepcopy(self.requests)
        predictions = copy.deepcopy(self.predictions)
        present_request = self.find(requests, case_id='alpha:test:0:0', task='alpha', name_mode='natural',
                                    presence='present', with_none=True, candidate_size=2,
                                    member_seed=0, order_seed=0)
        absent_request = self.find(requests, case_id='alpha:test:1:0', task='alpha', name_mode='natural',
                                   presence='absent', with_none=True, candidate_size=2,
                                   member_seed=0, order_seed=0)
        self.prediction_for(predictions, present_request)['choice'] = 'none'  # false rejection
        absent_choice = next(key for key in absent_request['criteria'] if key != 'none')
        self.prediction_for(predictions, absent_request)['choice'] = absent_choice  # false acceptance
        self.reload(requests=requests, predictions=predictions)
        group = analyze(self.root, bootstrap_replicates=5, joint=False)['groups']['alpha|k2|natural|aligned']
        present = group['native']['test']['present']
        absent = group['native']['test']['absent']
        self.assertAlmostEqual(present['rejected']['text_mean'], 1 / 6)
        self.assertAlmostEqual(present['correct']['text_mean'], 5 / 6)
        self.assertAlmostEqual(absent['accepted']['text_mean'], 1 / 6)
        self.assertAlmostEqual(absent['rejected']['text_mean'], 5 / 6)
        # The paired delta on test is (rejected - false_rejected) averaged per text.
        self.assertAlmostEqual(group['native']['test']['paired_delta']['text_mean'], 4 / 6)

    def test_strict_threshold_ties_and_false_rejection_budget(self):
        # score == threshold is accepted, so the maximum present score is a legal zero-F threshold.
        self.assertEqual(select_threshold_weighted({'t': [0.5]}, {'t': [0.5]}), 0.5)
        policy = evaluate_policy({'t': [0.5]}, {'t': [0.5]}, {'t': [True]}, 0.5, 0, 0)
        self.assertEqual(policy['false_rejection']['text_mean'], 0.0)
        self.assertEqual(policy['detection']['text_mean'], 0.0)
        # Present score above the F budget excludes a lower candidate; the budget picks the smaller loss.
        threshold = select_threshold_weighted({'t': [0.4, 0.5]}, {'t': [0.5, 0.9]})
        self.assertEqual(threshold, 0.5)
        policy = evaluate_policy({'t': [0.4, 0.5]}, {'t': [0.5, 0.9]}, {'t': [True, True]}, threshold, 0, 0)
        self.assertEqual(policy['false_rejection']['text_mean'], 0.0)
        self.assertEqual(policy['detection']['text_mean'], 0.5)
        # Text-equal weighting: text A has three variants, text B one; A must not dominate.
        present = {'A': [0.6, 0.6, 0.6], 'B': [0.0]}
        absent = {'A': [0.9, 0.9, 0.9], 'B': [0.1]}
        threshold = select_threshold_weighted(present, absent)
        policy = evaluate_policy(present, absent, {k: [True] * len(v) for k, v in present.items()}, threshold, 0, 0)
        self.assertEqual(threshold, 0.6)
        self.assertAlmostEqual(policy['detection']['text_mean'], 0.5)
        self.assertAlmostEqual(policy['detection']['pooled'], 0.75)

    def test_no_target_leakage(self):
        original = analyze(self.root, bootstrap_replicates=5, joint=False)
        requests = copy.deepcopy(self.requests)
        predictions = copy.deepcopy(self.predictions)
        for request in requests:
            if request['split'] != 'test':
                continue
            prediction = self.prediction_for(predictions, request)
            if prediction.get('probabilities') is not None:
                keys = list(request['criteria'])
                prediction['probabilities'] = {key: 1 / len(keys) for key in keys}
        self.reload(requests=requests, predictions=predictions)
        changed = analyze(self.root, bootstrap_replicates=5, joint=False)
        for key, group in original['groups'].items():
            for score, entry in group['scores'].items():
                self.assertEqual(entry['threshold'], changed['groups'][key]['scores'][score]['threshold'])

    def test_discrete_only_track(self):
        cases, requests, predictions = build_run(self.root, discrete=True)
        save(self.root, cases, requests, predictions)
        metrics = analyze(self.root, bootstrap_replicates=5, joint=False)
        group = metrics['groups']['alpha|k2|natural|aligned']
        self.assertFalse(group['has_probabilities'])
        for score in ('negative_max', 'normalized_entropy', 'none_probability'):
            self.assertIsNone(group['scores'][score]['threshold'])
            self.assertIn('discrete-only', group['scores'][score]['undefined_reason'])
        self.assertEqual(group['native']['test']['present']['correct']['text_mean'], 1.0)
        self.assertEqual(group['native']['test']['absent']['rejected']['text_mean'], 1.0)

    def test_failures_are_not_wrong_or_zero(self):
        requests = copy.deepcopy(self.requests)
        predictions = copy.deepcopy(self.predictions)
        present_request = self.find(requests, case_id='alpha:test:0:0', task='alpha', name_mode='natural',
                                    presence='present', with_none=True, candidate_size=2,
                                    member_seed=0, order_seed=0)
        absent_request = self.find(requests, case_id='alpha:test:1:0', task='alpha', name_mode='natural',
                                   presence='absent', with_none=True, candidate_size=2,
                                   member_seed=0, order_seed=0)
        for request, status in ((present_request, 'interface_failure'), (absent_request, 'truncation_failure')):
            prediction = self.prediction_for(predictions, request)
            prediction.clear()
            prediction.update({'request_id': request['request_id'], 'status': status,
                               'choice': None, 'probabilities': None, 'error': 'synthetic'})
        self.reload(requests=requests, predictions=predictions)
        group = analyze(self.root, bootstrap_replicates=5, joint=False)['groups']['alpha|k2|natural|aligned']
        self.assertAlmostEqual(group['native']['test']['present']['interface_failure']['text_mean'], 1 / 6)
        self.assertAlmostEqual(group['native']['test']['present']['wrong']['text_mean'], 0.0)
        self.assertAlmostEqual(group['native']['test']['absent']['interface_failure']['text_mean'], 1 / 6)
        self.assertEqual(group['native']['test']['failures_by_status']['interface_failure'], 1)
        self.assertEqual(group['native']['test']['failures_by_status']['truncation_failure'], 1)
        # A failed present/absent with-NONE call can never enter a none_probability score family.
        self.assertEqual(group['scores']['none_probability']['eligible']['test']['n_text'], 4)
        # The plain calls of those same variants are unaffected and remain eligible.
        self.assertEqual(group['scores']['negative_max']['eligible']['test']['n_text'], 6)

    def test_family_aggregation_and_bootstrap_determinism(self):
        cases, requests, predictions = build_run(self.root, n_cal=2, n_test=2, member_seeds=(0, 1),
                                                order_seeds=(0,), with_legacy=True)
        # Give one text a single variant so text-equal and pooled fractions differ.
        target = 'alpha:test:0:0'
        drop = {request['request_id'] for request in requests
                if request['case_id'] == target and request['member_seed'] == 1}
        requests = [request for request in requests if request['request_id'] not in drop]
        predictions = [prediction for prediction in predictions if prediction['request_id'] not in drop]
        # Keep the surviving variant of the target text but make it wrong: text-equal and
        # pooled fractions then disagree because the other five texts have two variants.
        target_present = next(request for request in requests
                              if request['case_id'] == target and request['name_mode'] == 'natural'
                              and request['presence'] == 'present' and request['with_none'] is True)
        target_prediction = next(prediction for prediction in predictions
                                 if prediction['request_id'] == target_present['request_id'])
        target_prediction['choice'] = next(key for key in target_present['criteria']
                                           if key not in ('none', target_present['gold_option']))
        save(self.root, cases, requests, predictions, profile='robustness')
        metrics = analyze(self.root, bootstrap_replicates=25, joint=False)
        group = metrics['groups']['alpha|k2|natural|aligned']
        self.assertEqual(group['n_text']['test'], 6)
        self.assertEqual(group['n_variants']['test'], 11)
        present = group['native']['test']['present']['correct']
        self.assertAlmostEqual(present['pooled'], 10 / 11)
        self.assertAlmostEqual(present['text_mean'], 5 / 6)
        repeated = analyze(self.root, bootstrap_replicates=25, joint=False)
        self.assertEqual(present['bootstrap95'],
                         repeated['groups']['alpha|k2|natural|aligned']['native']['test']['present']['correct']['bootstrap95'])
        # Legacy wording is a separate group and never leaks into the aligned group.
        self.assertIn('alpha|k2|natural|legacy', metrics['groups'])
        self.assertEqual(metrics['groups']['alpha|k2|natural|legacy']['scores']['none_probability']['eligible']['test']['n_text'], 6)

    def test_malformed_and_missing_responses(self):
        # Missing prediction.
        save(self.root, self.cases, self.requests, self.predictions[:-1])
        with self.assertRaises(ValidationError):
            validate_run(self.root)
        save(self.root, self.cases, self.requests, self.predictions)
        # Extraneous prediction.
        extra = dict(self.predictions[0], request_id='invented')
        save(self.root, self.cases, self.requests, self.predictions + [extra])
        with self.assertRaises(ValidationError):
            validate_run(self.root)
        save(self.root, self.cases, self.requests, self.predictions)
        # status ok must carry an offered choice.
        predictions = copy.deepcopy(self.predictions)
        self.prediction_for(predictions, self.requests[0])['choice'] = None
        save(self.root, self.cases, self.requests, predictions)
        with self.assertRaises(ValidationError):
            validate_run(self.root)
        save(self.root, self.cases, self.requests, self.predictions)
        # Failure rows must not carry a choice.
        predictions = copy.deepcopy(self.predictions)
        prediction = self.prediction_for(predictions, self.requests[0])
        prediction.update({'status': 'interface_failure', 'choice': self.requests[0]['gold_option'],
                           'probabilities': None})
        save(self.root, self.cases, self.requests, predictions)
        with self.assertRaises(ValidationError):
            validate_run(self.root)
        save(self.root, self.cases, self.requests, self.predictions)
        # Probabilities must match criteria and sum within tolerance.
        predictions = copy.deepcopy(self.predictions)
        prediction = next(prediction for prediction in predictions if prediction['probabilities'] is not None)
        prediction['probabilities'][next(iter(prediction['probabilities']))] = 0.99
        save(self.root, self.cases, self.requests, predictions)
        with self.assertRaises(ValidationError):
            validate_run(self.root)
        save(self.root, self.cases, self.requests, self.predictions)
        # member_seed outside the frozen 0..4 range.
        requests = copy.deepcopy(self.requests)
        requests[0]['member_seed'] = 7
        save(self.root, self.cases, requests, self.predictions)
        with self.assertRaises(ValidationError):
            validate_run(self.root)
        save(self.root, self.cases, self.requests, self.predictions)
        # Legacy NONE wording is not allowed on plain calls.
        requests = copy.deepcopy(self.requests)
        plain = self.find(requests, case_id='alpha:test:0:0', task='alpha', name_mode='natural',
                          presence='present', with_none=False, candidate_size=2,
                          member_seed=0, order_seed=0)
        plain['none_mode'] = 'legacy'
        save(self.root, self.cases, requests, self.predictions)
        with self.assertRaises(ValidationError):
            validate_run(self.root)
        save(self.root, self.cases, self.requests, self.predictions)
        # A pair must replace the reference option in the same slot.
        requests = copy.deepcopy(self.requests)
        absent = self.find(requests, case_id='alpha:test:1:0', task='alpha', name_mode='natural',
                           presence='absent', with_none=True, candidate_size=2,
                           member_seed=0, order_seed=0)
        absent['criteria'] = {key: value for key, value in reversed(list(absent['criteria'].items()))}
        save(self.root, self.cases, requests, self.predictions)
        with self.assertRaises(ValidationError):
            validate_run(self.root)
        save(self.root, self.cases, self.requests, self.predictions)
        # A failure row needs error metadata beyond the null choice/probabilities.
        predictions = copy.deepcopy(self.predictions)
        predictions[0] = {'request_id': self.requests[0]['request_id'], 'status': 'interface_failure',
                          'choice': None, 'probabilities': None}
        save(self.root, self.cases, self.requests, predictions)
        with self.assertRaises(ValidationError):
            validate_run(self.root)
        save(self.root, self.cases, self.requests, self.predictions)
        # A core-profile run may not carry robustness member seeds.
        requests = copy.deepcopy(self.requests)
        requests[0]['member_seed'] = 1
        save(self.root, self.cases, requests, self.predictions, profile='core')
        with self.assertRaises(ValidationError):
            validate_run(self.root)

    def test_eighty_subset_calibration_and_transfer(self):
        cases, requests, predictions = build_run(self.root, n_cal=30, n_test=2)
        save(self.root, cases, requests, predictions)
        plan = write_plan(self.root.parent, cases, per_task=80)
        metrics = analyze(self.root, bootstrap_replicates=5, subset_plan=plan, joint=False)
        transfer = metrics['transfer']
        self.assertTrue(transfer['available'])
        cross = [row for row in transfer['rows'] if row['transfer_type'] == 'cross_task']
        self.assertTrue(cross)
        for row in cross:
            subset = row['source_calibration_subset']
            self.assertEqual(subset['n_text'], 80)
            self.assertEqual(row['target_test']['n_text'], 6)
        none_rows = [row for row in cross if row['score'] == 'none_probability']
        self.assertTrue(none_rows)
        # A different frozen subset changes the source threshold registry, never the target denominator.
        small = write_plan(self.root.parent, cases, per_task=10)
        changed = analyze(self.root, bootstrap_replicates=5, subset_plan=small, joint=False)
        changed_cross = [row for row in changed['transfer']['rows'] if row['transfer_type'] == 'cross_task']
        for row in changed_cross:
            self.assertEqual(row['source_calibration_subset']['n_text'], 10)
            self.assertEqual(row['target_test']['n_text'], 6)

    def test_cli_run_writes_new_directory_and_refuses_overwrite(self):
        output = Path(self.temp.name) / 'out'
        metrics = run(self.root, output, bootstrap_replicates=5, joint=False)
        self.assertTrue((output / 'metrics.json').is_file())
        self.assertTrue((output / 'report.md').is_file())
        self.assertEqual(metrics['protocol'], 'paper1-confirmation-v2')
        self.assertEqual(metrics['validation']['complete'], True)
        with self.assertRaises(ValidationError):
            run(self.root, output, bootstrap_replicates=5, joint=False)
        # The frozen run must not be written into.
        with self.assertRaises(ValidationError):
            run(self.root, self.root / 'nested', bootstrap_replicates=5, joint=False)


if __name__ == '__main__':
    unittest.main()
