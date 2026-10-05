import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from analyze import analyze, auc, select_threshold, validate, ValidationError, wilson

def fixture(root):
    cases, requests, predictions = [], [], []
    labels = ['Alpha', 'Beta', 'Gamma']
    for split in ['calibration', 'test']:
        for gold, label in enumerate(labels):
            case_id = f'{split}-{gold}'
            cases.append({'case_id': case_id, 'task': 'toy', 'split': split, 'label': label,
                          'label_id': gold, 'text': f'{split} unique text {gold}', 'row_idx': gold,
                          'source_split': 'train' if split == 'calibration' else 'test'})
            for mode in ['natural', 'neutral']:
                key = lambda index: labels[index] if mode == 'natural' else f'option_{index:03d}'
                other = [index for index in range(3) if index != gold]
                for presence, indices in [('full', [0, 1, 2]), ('present', [gold, other[0]]), ('absent', other)]:
                    for with_none in [False, True]:
                        criteria = {key(index): labels[index] + ' description' for index in indices}
                        if with_none:
                            criteria['none'] = 'None of the options'
                        request_id = f'{case_id}-{mode}-{presence}-{with_none}'
                        target = 'none' if presence == 'absent' and with_none else key(indices[0]) if presence == 'absent' else key(gold)
                        probability = .6 if presence == 'absent' and not with_none else .8
                        probabilities = {option: probability if option == target else (1-probability)/(len(criteria)-1) for option in criteria}
                        requests.append({'request_id': request_id, 'case_id': case_id, 'task': 'toy', 'split': split,
                            'presence': presence, 'candidate_size': len(indices), 'name_mode': mode, 'with_none': with_none,
                            'replicate': 0, 'pair_id': None if presence == 'full' else f'{case_id}-{mode}-{with_none}',
                            'instruction': 'Select category', 'criteria': criteria, 'gold_option': key(gold) if presence != 'absent' else None,
                            'gold_label': label})
                        predictions.append({'request_id': request_id, 'choice': target, 'probabilities': probabilities})
    save(root, cases, requests, predictions)
    return cases, requests, predictions

def save(root, cases, requests, predictions):
    for name, rows in [('cases.jsonl', cases), ('requests.jsonl', requests), ('predictions.jsonl', predictions)]:
        (root / name).write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ['cases.jsonl', 'requests.jsonl']}
    (root / 'manifest.json').write_text(json.dumps({'file_sha256': hashes}), encoding='utf-8')

class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cases, self.requests, self.predictions = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_known_auc_ties(self):
        self.assertEqual(auc([.5, .9], [.5, .8]), .625)
        self.assertEqual(auc([1, 1], [0, 0]), 1)
        self.assertEqual(auc([.5], [.5]), .5)
        self.assertIsNone(auc([], [1]))

    def test_wilson_zero_and_all(self):
        self.assertAlmostEqual(wilson(0, 10)[1], .2775327998628892)
        self.assertAlmostEqual(wilson(10, 10)[0], .7224672001371107)

    def test_threshold_must_maximize_detection(self):
        self.assertEqual(select_threshold([.1, .2], [.8, .9]), .2)

    def test_perfect_paired_and_undefined_subset(self):
        result = analyze(self.root, bootstrap_replicates=20)
        group = result['groups']['toy|k2|natural']
        self.assertEqual(group['detection']['value'], 1)
        self.assertEqual(group['false_rejection']['value'], 0)
        self.assertEqual(group['paired']['delta']['bootstrap95'], [1, 1])
        self.assertEqual(group['baseline_correct_subset']['n'], 3)
        for request, prediction in zip(self.requests, self.predictions):
            if request['presence'] == 'full' and request['split'] == 'test' and not request['with_none']:
                prediction['choice'] = next(key for key in request['criteria'] if key != request['gold_option'])
        save(self.root, self.cases, self.requests, self.predictions)
        result = analyze(self.root, bootstrap_replicates=20)
        for group in result['groups'].values():
            self.assertEqual(group['baseline_correct_subset']['n'], 0)
            self.assertIsNone(group['baseline_correct_subset']['auc']['none_probability'])

    def test_threshold_test_data_invariance(self):
        original = analyze(self.root, bootstrap_replicates=10)
        for request, prediction in zip(self.requests, self.predictions):
            if request['split'] == 'test':
                prediction['probabilities'] = {key: 1/len(request['criteria']) for key in request['criteria']}
        save(self.root, self.cases, self.requests, self.predictions)
        changed = analyze(self.root, bootstrap_replicates=10)
        for key in original['groups']:
            for score in original['groups'][key]['calibrated_thresholds']:
                self.assertEqual(original['groups'][key]['calibrated_thresholds'][score]['threshold'], changed['groups'][key]['calibrated_thresholds'][score]['threshold'])

    def test_missing_and_duplicate_predictions(self):
        for predictions in [self.predictions[:-1], self.predictions + [self.predictions[0]]]:
            save(self.root, self.cases, self.requests, predictions)
            with self.assertRaises(ValidationError):
                validate(self.root)

    def test_bad_probability_values(self):
        for value in [-.1, 1.1, float('nan')]:
            predictions = copy.deepcopy(self.predictions)
            first = predictions[0]['probabilities']
            first[next(iter(first))] = value
            save(self.root, self.cases, self.requests, predictions)
            with self.assertRaises(ValidationError):
                validate(self.root)

    def test_wrong_probability_key(self):
        first = self.predictions[0]['probabilities']
        first['invented'] = first.pop(next(iter(first)))
        save(self.root, self.cases, self.requests, self.predictions)
        with self.assertRaises(ValidationError):
            validate(self.root)

    def test_pair_and_gold_corruption(self):
        requests = copy.deepcopy(self.requests)
        absent = next(r for r in requests if r['presence'] == 'absent' and not r['with_none'])
        absent['pair_id'] = 'unmatched'
        save(self.root, self.cases, requests, self.predictions)
        with self.assertRaises(ValidationError):
            validate(self.root)
        requests = copy.deepcopy(self.requests)
        requests[0]['gold_option'] = 'Beta'
        save(self.root, self.cases, requests, self.predictions)
        with self.assertRaises(ValidationError):
            validate(self.root)

    def test_duplicate_text_and_frozen_hash(self):
        self.cases[-1]['text'] = self.cases[0]['text']
        save(self.root, self.cases, self.requests, self.predictions)
        with self.assertRaises(ValidationError):
            validate(self.root)
        fixture(self.root)
        with (self.root / 'requests.jsonl').open('a', encoding='utf-8') as stream:
            stream.write('\n')
        with self.assertRaises(ValidationError):
            validate(self.root)

if __name__ == '__main__':
    unittest.main()
