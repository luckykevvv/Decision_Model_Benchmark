"""Paper 1 confirmation scoring for the ``paper1-confirmation-v2`` run schema.

This module scores independently prepared public-data confirmation runs that
have a ``core`` and/or a ``robustness`` profile.  It never mutates the frozen
run directory and never fits a threshold, score, source task or prompt on target
test outcomes.  ``metrics.json`` and ``report.md`` are written to a brand new
``--output-dir``.

The Wilson-interval, AUC, normalized-entropy and strict ``score > threshold``
calibration formulas are reused from :mod:`systematic.analyze`; the confirmation
version adds text-family weighting, member/order variants, the aligned/legacy
NONE wording condition, failure and discrete-only tracks and staged-profile
reporting.  The v1 ``validate`` function is deliberately not used because it
assumes a single member/order replicate and mandatory probabilities.

Schema summary (see the request/response and protocol sections of ``README.md``):

* run dir has ``cases.jsonl``, ``requests.jsonl``, ``manifest.json``,
  ``predictions.jsonl``;
* ``manifest['protocol'] == 'paper1-confirmation-v2'`` and
  ``manifest['file_sha256']`` includes the frozen ``cases.jsonl`` /
  ``requests.jsonl``;
* requests keep the v1 keys plus ``member_seed`` (0..4), ``order_seed`` (0..1)
  and ``none_mode`` (``aligned``/``legacy``);
* predictions are native choices with an optional probability map; rows with
  ``status`` in ``interface_failure``/``truncation_failure`` have null choice
  and probabilities; ``status`` may be omitted meaning ``ok``.

Entry point::

    python -m systematic.confirmation_analysis \
        --run-dir RUN --output-dir NEW_OUT \
        --bootstrap-replicates 2000 \
        --subset-plan outputs/paper1-confirmation-analysis-plan-20261003/subsets.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROTOCOL = 'paper1-confirmation-v2'
SCORE_NAMES = ('negative_max', 'normalized_entropy', 'none_probability')
NATIVE_PRESENT = ('correct', 'wrong', 'rejected', 'interface_failure')
NATIVE_ABSENT = ('rejected', 'accepted', 'interface_failure')
ALLOWED_STATUS = ('ok', 'interface_failure', 'truncation_failure')
ALLOWED_NONE_MODE = ('aligned', 'legacy')
DEFAULT_LIMIT = .05
DEFAULT_SEED = 20261005
SOURCE_CODE = 'systematic/confirmation_analysis.py'


class ValidationError(ValueError):
    """Raised when the frozen run cannot be validated or is incomplete."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


# ---------------------------------------------------------------------------
# Small deterministic helpers (formulas reused from systematic/analyze.py)
# ---------------------------------------------------------------------------


def sha256_path(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def unique_by(rows, field):
    mapping = {row[field]: row for row in rows}
    require(len(mapping) == len(rows), f'Duplicate {field}')
    return mapping


def stable_seed(base, *parts):
    """Deterministic per-key bootstrap seed; keeps a single declared base seed."""
    digest = hashlib.sha256('|'.join(str(part) for part in parts).encode('utf-8')).digest()
    return (int(base) + int.from_bytes(digest[:4], 'big')) % (2 ** 31 - 1)


def wilson(successes, n):
    """Wilson score interval, reused from systematic/analyze.py."""
    if not n:
        return None
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [center - radius, center + radius]


def auc(positive, negative):
    """Rank AUC, ties at 0.5, reused from systematic/analyze.py."""
    if len(positive) == 0 or len(negative) == 0:
        return None
    positive, negative = np.asarray(positive, dtype=float), np.sort(np.asarray(negative, dtype=float))
    ranks = (np.searchsorted(negative, positive, 'left') + np.searchsorted(negative, positive, 'right')) / 2
    return float(ranks.sum() / (len(positive) * len(negative)))


def entropy(probabilities):
    """Normalized Shannon entropy; rounded scores are renormalized first."""
    probabilities = np.asarray(probabilities, dtype=float)
    if len(probabilities) <= 1:
        return 0.0
    probabilities = probabilities / probabilities.sum()
    positive = probabilities[probabilities > 0]
    return float(-(positive * np.log(positive)).sum() / np.log(len(probabilities)))


def percentiles(samples):
    if not samples:
        return None
    return [float(x) for x in np.percentile(samples, [2.5, 97.5])]


def bootstrap_ids(text_ids, replicates, seed, statistic):
    """Cluster bootstrap over text ids; the statistic receives resampled ids."""
    ids = sorted(text_ids)
    if not ids or replicates <= 0:
        return None
    rng = np.random.default_rng(seed)
    n = len(ids)
    samples = []
    for _ in range(replicates):
        draw = [ids[index] for index in rng.integers(0, n, size=n)]
        value = statistic(draw)
        if value is not None and math.isfinite(value):
            samples.append(float(value))
    return percentiles(samples)


def family_stats(values_by_text, replicates, seed):
    """Per-text mean, equal-weight across texts, with a text-cluster bootstrap."""
    values = {text: np.asarray(v, dtype=float) for text, v in values_by_text.items() if len(v)}
    if not values:
        return {'n_text': 0, 'n_variants': 0, 'text_mean': None, 'pooled': None, 'bootstrap95': None}
    text_means = {text: float(v.mean()) for text, v in values.items()}
    values_concat = np.concatenate(list(values.values()))
    bootstrap = bootstrap_ids(text_means, replicates, seed,
                              lambda ids: float(np.mean([text_means[text] for text in ids])))
    return {'n_text': len(values), 'n_variants': len(values_concat),
            'text_mean': float(np.mean(list(text_means.values()))),
            'pooled': float(values_concat.mean()), 'bootstrap95': bootstrap}


def rate_family(per_text_bool, replicates, seed):
    """Rate with equal text weight over variants, pooled fraction and Wilson CI."""
    values = {text: np.asarray(v, dtype=bool) for text, v in per_text_bool.items() if len(v)}
    if not values:
        return {'n_text': 0, 'n_variants': 0, 'successes': 0, 'text_mean': None, 'pooled': None,
                'wilson95': None, 'bootstrap95': None}
    text_means = {text: float(v.mean()) for text, v in values.items()}
    successes = int(sum(int(v.sum()) for v in values.values()))
    n_variants = int(sum(len(v) for v in values.values()))
    bootstrap = bootstrap_ids(text_means, replicates, seed,
                              lambda ids: float(np.mean([text_means[text] for text in ids])))
    return {'n_text': len(values), 'n_variants': n_variants, 'successes': successes,
            'text_mean': float(np.mean(list(text_means.values()))),
            'pooled': successes / n_variants,
            'wilson95': wilson(successes, n_variants) if all(len(v) == 1 for v in values.values()) else None,
            'bootstrap95': bootstrap}


def select_threshold_weighted(present_by_text, absent_by_text, limit=DEFAULT_LIMIT):
    """Maximize calibration detection with text-equal weight and F <= limit.

    Strict rejection is ``score > threshold``.  Ties prefer lower F, then a
    higher threshold (same rule as ``systematic.analyze.select_threshold``).
    """
    present = {text: np.asarray(v, dtype=float) for text, v in present_by_text.items() if len(v)}
    absent = {text: np.asarray(v, dtype=float) for text, v in absent_by_text.items() if len(v)}
    require(present and absent, 'Calibration score families required')
    candidates = np.unique(np.concatenate([np.concatenate(list(present.values())),
                                           np.concatenate(list(absent.values()))]))
    # Broadcast all threshold comparisons, retaining one equal weight per text.
    # This is the same estimator as the scalar loop, including unequal families.
    def family_rates(mapping):
        width = max(len(v) for v in mapping.values())
        matrix = np.full((len(mapping), width), np.nan)
        lengths = np.array([len(v) for v in mapping.values()], dtype=float)
        for i, values in enumerate(mapping.values()):
            matrix[i, :len(values)] = values
        hits = (matrix[None, :, :] > candidates[:, None, None]).sum(axis=2)
        return (hits / lengths[None, :]).mean(axis=1)
    false_rates, detections = family_rates(present), family_rates(absent)
    best = None
    for threshold, false_rate, detection in zip(candidates, false_rates, detections):
        false_rate, detection = float(false_rate), float(detection)
        if false_rate <= limit:
            objective = (detection, -false_rate, float(threshold))
            if best is None or objective > best[0]:
                best = (objective, float(threshold))
    require(best is not None, 'No calibration threshold satisfies the false-rejection constraint')
    return best[1]


def evaluate_policy(present_by_text, absent_by_text, present_correct_by_text, threshold, replicates, seed):
    """Evaluate a frozen threshold; D/F and the accepted/rejected decomposition."""
    texts = sorted(set(present_by_text) & set(absent_by_text))
    detection, false_rejection = {}, {}
    correct_accepted, wrong_accepted, acceptance, absent_false_acceptance = {}, {}, {}, {}
    for text in texts:
        present = np.asarray(present_by_text[text], dtype=float)
        absent = np.asarray(absent_by_text[text], dtype=float)
        correct = np.asarray(present_correct_by_text[text], dtype=bool)
        rejected = present > threshold
        accepted = ~rejected
        detection[text] = list(absent > threshold)
        false_rejection[text] = list(rejected)
        correct_accepted[text] = list(accepted & correct)
        wrong_accepted[text] = list(accepted & ~correct)
        acceptance[text] = list(accepted)
        absent_false_acceptance[text] = list(absent <= threshold)
    return {'threshold': threshold, 'n_text': len(texts),
            'n_variants': int(sum(len(present_by_text[t]) for t in texts)),
            'detection': rate_family(detection, replicates, stable_seed(seed, 'policy', 'detection', threshold)),
            'false_rejection': rate_family(false_rejection, replicates, stable_seed(seed, 'policy', 'fr', threshold)),
            'present_correct_accepted': rate_family(correct_accepted, replicates, stable_seed(seed, 'policy', 'correct', threshold)),
            'present_wrong_accepted': rate_family(wrong_accepted, replicates, stable_seed(seed, 'policy', 'wrong', threshold)),
            'present_acceptance': rate_family(acceptance, replicates, stable_seed(seed, 'policy', 'accept', threshold)),
            'absent_false_acceptance': rate_family(absent_false_acceptance, replicates, stable_seed(seed, 'policy', 'afa', threshold))}


def auc_stats(present_by_text, absent_by_text, replicates, seed):
    """Pooled rank AUC plus a per-text paired comparison, both text-clustered."""
    present = {t: np.asarray(v, dtype=float) for t, v in present_by_text.items() if len(v)}
    absent = {t: np.asarray(v, dtype=float) for t, v in absent_by_text.items() if len(v)}
    texts = sorted(set(present) & set(absent))
    if not texts:
        return {'n_text': 0, 'n_variants': 0, 'pooled': None, 'paired_text_mean': None, 'bootstrap95': None}

    def pooled(ids):
        p = np.concatenate([present[t] for t in ids])
        a = np.concatenate([absent[t] for t in ids])
        return auc(a, p)

    paired = []
    for text in texts:
        p, a = present[text], absent[text]
        paired.append(float(np.where(a > p, 1.0, np.where(a == p, .5, 0.0)).mean()))
    return {'n_text': len(texts), 'n_variants': int(sum(len(present[t]) for t in texts)),
            'pooled': pooled(texts), 'paired_text_mean': float(np.mean(paired)),
            'bootstrap95': bootstrap_ids(texts, replicates, stable_seed(seed, 'auc'), pooled)}


def reliability_curve(pairs, bins=10):
    """Fixed equal-width reliability bins on [0, 1]; empty bins have no estimate."""
    buckets = [{'lower': i / bins, 'upper': (i + 1) / bins, 'n': 0, 'pred_sum': 0.0, 'obs_sum': 0.0}
               for i in range(bins)]
    for predicted, observed in pairs:
        index = min(int(predicted * bins), bins - 1)
        bucket = buckets[index]
        bucket['n'] += 1
        bucket['pred_sum'] += predicted
        bucket['obs_sum'] += observed
    return [{'lower': b['lower'], 'upper': b['upper'], 'n': b['n'],
             'mean_predicted': (b['pred_sum'] / b['n']) if b['n'] else None,
             'observed': (b['obs_sum'] / b['n']) if b['n'] else None} for b in buckets]


def brier_stats(pairs, replicates, seed):
    """Brier score over (predicted, observed) pairs with text-equal weighting."""
    if not pairs:
        return {'n_pairs': 0, 'n_text': 0, 'prevalence': None, 'brier': None,
                'brier_text_mean': None, 'brier_bootstrap95': None, 'reliability': reliability_curve([])}
    by_text = defaultdict(list)
    for text, predicted, observed in pairs:
        by_text[text].append((predicted - observed) ** 2)
    text_means = {t: float(np.mean(v)) for t, v in by_text.items()}
    bootstrap = bootstrap_ids(text_means, replicates, seed,
                              lambda ids: float(np.mean([text_means[t] for t in ids])))
    return {'n_pairs': len(pairs), 'n_text': len(by_text),
            'prevalence': float(np.mean([observed for _, _, observed in pairs])),
            'brier': float(np.mean([(p - o) ** 2 for _, p, o in pairs])),
            'brier_text_mean': float(np.mean(list(text_means.values()))),
            'brier_bootstrap95': bootstrap,
            'reliability': reliability_curve([(p, o) for _, p, o in pairs])}


# ---------------------------------------------------------------------------
# Prediction accessors
# ---------------------------------------------------------------------------


def status_of(prediction):
    return prediction.get('status', 'ok')


def probabilities_of(prediction):
    return prediction.get('probabilities')


def plain_scores(request, prediction):
    """negative_max and normalized entropy from a plain (no-NONE) call."""
    if status_of(prediction) != 'ok':
        return None
    probabilities = probabilities_of(prediction)
    if probabilities is None:
        return None
    values = [float(probabilities[key]) for key in request['criteria']]
    return {'negative_max': -max(values), 'normalized_entropy': entropy(values)}


def none_probability(request, prediction):
    if status_of(prediction) != 'ok':
        return None
    probabilities = probabilities_of(prediction)
    if probabilities is None or 'none' not in probabilities:
        return None
    return float(probabilities['none'])


def argmax_ordinary(request, prediction):
    """Highest ordinary probability; ties resolve to the first offered option."""
    probabilities = probabilities_of(prediction)
    require(probabilities is not None, 'argmax requires probabilities')
    ordinary = [key for key in request['criteria'] if key != 'none']
    best = ordinary[0]
    for key in ordinary[1:]:
        if probabilities[key] > probabilities[best]:
            best = key
    return best


def present_category(request, prediction):
    if status_of(prediction) != 'ok':
        return 'interface_failure'
    choice = prediction.get('choice')
    if choice == 'none':
        return 'rejected'
    if choice == request['gold_option']:
        return 'correct'
    return 'wrong'


def absent_category(request, prediction):
    if status_of(prediction) != 'ok':
        return 'interface_failure'
    return 'rejected' if prediction.get('choice') == 'none' else 'accepted'


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def cell_index_key(request):
    return (request['case_id'], request['name_mode'], request['presence'], request['candidate_size'],
            request['with_none'], request['member_seed'], request['order_seed'], request['none_mode'])


def validate_run(run_dir, allow_partial=False):
    """Validate the frozen schema; reject malformed or incomplete runs."""
    run_dir = Path(run_dir).resolve()
    require(run_dir.is_dir(), f'Run directory missing: {run_dir}')
    manifest_path = run_dir / 'manifest.json'
    require(manifest_path.is_file(), 'manifest.json missing')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    require(manifest.get('protocol') == PROTOCOL,
            f"Unexpected manifest protocol: {manifest.get('protocol')!r}")
    file_hashes = manifest.get('file_sha256') or {}
    require('cases.jsonl' in file_hashes and 'requests.jsonl' in file_hashes,
            'manifest file_sha256 must include cases.jsonl and requests.jsonl')
    for filename, expected in file_hashes.items():
        path = run_dir / filename
        require(path.is_file(), f'Declared frozen file missing: {filename}')
        require(sha256_path(path) == expected, f'Changed frozen file: {filename}')
    decimals = manifest.get('reported_probability_decimals')
    require(decimals is None or (type(decimals) is int and 1 <= decimals <= 8),
            'Invalid reported_probability_decimals')

    cases = unique_by(read_jsonl(run_dir / 'cases.jsonl'), 'case_id')
    requests = unique_by(read_jsonl(run_dir / 'requests.jsonl'), 'request_id')
    predictions = unique_by(read_jsonl(run_dir / 'predictions.jsonl'), 'request_id')
    require(requests, 'Run contains no requests')
    require(set(predictions) == set(requests), 'Incomplete or extraneous predictions')

    seen_text = set()
    text_index = manifest.get('case_representation') == 'text-index-v1'
    for case in cases.values():
        require(case.get('split') in ('calibration', 'test'), f"Unknown split for {case.get('case_id')}")
        if text_index:
            require('text' not in case, 'Text-index runs must not silently mix in source text')
            for field in ('text_sha256', 'normalized_text_sha256'):
                value = case.get(field)
                require(isinstance(value, str) and len(value) == 64 and
                        all(c in '0123456789abcdef' for c in value), f'Invalid {field}')
            identity = (case['task'], case['normalized_text_sha256'])
        else:
            require(isinstance(case.get('text'), str) and case['text'].strip(), 'Empty case text')
            identity = (case['task'], ' '.join(case['text'].casefold().split()))
        require(identity not in seen_text, 'Duplicate text or split leakage')
        seen_text.add(identity)

    cells = {}
    index = {}
    pairs = defaultdict(dict)
    full_baselines = defaultdict(set)
    for request_id, request in requests.items():
        request_id = str(request_id)
        require(request.get('case_id') in cases, f'Unknown case: {request.get("case_id")}')
        case = cases[request['case_id']]
        for field in ('task', 'split'):
            require(request.get(field) == case[field], f'Mismatched {field} for {request_id}')
        require(request.get('gold_label') == case['label'], f'Mismatched gold label for {request_id}')
        require(request.get('name_mode') in ('natural', 'neutral'), f'Unknown option name mode: {request_id}')
        require(type(request.get('with_none')) is bool, f'with_none must be boolean: {request_id}')
        presence = request.get('presence')
        require(presence in ('full', 'present', 'absent'), f'Unknown presence: {request_id}')
        candidate_size = request.get('candidate_size')
        require(type(candidate_size) is int and candidate_size >= 1, f'Invalid candidate_size: {request_id}')
        member_seed = request.get('member_seed')
        order_seed = request.get('order_seed')
        none_mode = request.get('none_mode')
        require(type(member_seed) is int and 0 <= member_seed <= 4,
                f'member_seed must be an integer in 0..4: {request_id}')
        require(type(order_seed) is int and order_seed in (0, 1),
                f'order_seed must be an integer in 0..1: {request_id}')
        require(none_mode in ALLOWED_NONE_MODE, f'Unknown none_mode: {request_id}')
        require(type(request.get('replicate', 0)) is int and 0 <= request.get('replicate', 0) <= 9,
                f'Invalid replicate metadata: {request_id}')
        if not request['with_none']:
            require(none_mode == 'aligned', f'Plain calls must use aligned none_mode: {request_id}')
        if none_mode == 'legacy':
            require(request['with_none'] and member_seed == 0 and order_seed == 0,
                    f'Legacy NONE wording only on with-NONE calls at member/order seed 0: {request_id}')
        criteria = request.get('criteria')
        require(isinstance(criteria, dict) and criteria, f'Invalid criteria: {request_id}')
        require(all(isinstance(key, str) and isinstance(value, str) and value
                    for key, value in criteria.items()), f'Invalid criteria entries: {request_id}')
        ordinary = [key for key in criteria if key != 'none']
        require(len(ordinary) == candidate_size, f'Wrong candidate count: {request_id}')
        require(('none' in criteria) == request['with_none'], f'Wrong none membership: {request_id}')
        expected_gold = case['label'] if request['name_mode'] == 'natural' else f'option_{case["label_id"]:03d}'
        if presence == 'absent':
            require(request.get('gold_option') is None and expected_gold not in ordinary,
                    f'Absent request offers the reference class: {request_id}')
        else:
            require(request.get('gold_option') == expected_gold and expected_gold in ordinary,
                    f'Wrong reference option: {request_id}')
        if presence == 'full':
            require(request.get('pair_id') is None, f'Full request has pair_id: {request_id}')
            full_baselines[(request['case_id'], request['name_mode'])].add(request['with_none'])
        else:
            require(isinstance(request.get('pair_id'), str) and request['pair_id'],
                    f'Missing pair_id: {request_id}')
            require(presence not in pairs[request['pair_id']], f'Duplicated pair member: {request_id}')
            pairs[request['pair_id']][presence] = request
        cell = cell_index_key(request)
        require(cell not in cells, f'Duplicated experimental cell: {cell}')
        cells[cell] = request
        index[cell] = request

        prediction = predictions[request_id]
        status = status_of(prediction)
        require(status in ALLOWED_STATUS, f'Unknown prediction status {status!r}: {request_id}')
        if status == 'ok':
            require(prediction.get('choice') in criteria, f'Prediction ok must have an offered choice: {request_id}')
            probabilities = probabilities_of(prediction)
            if probabilities is not None:
                require(isinstance(probabilities, dict), f'Invalid probabilities: {request_id}')
                require(set(probabilities) == set(criteria), f'Wrong probability keys: {request_id}')
                require(all(not isinstance(value, bool) and isinstance(value, (int, float))
                            and math.isfinite(value) and 0 <= value <= 1
                            for value in probabilities.values()),
                        f'Invalid probability values: {request_id}')
                total = float(sum(probabilities.values()))
                tolerance = .001 if decimals is None else .5 * 10 ** (-decimals) * len(probabilities) + 1e-9
                require(total > 0 and abs(total - 1) <= tolerance,
                        f'Probability sum differs from one beyond declared rounding tolerance: {request_id}')
        else:
            require(prediction.get('choice') is None, f'Failure row must have null choice: {request_id}')
            require(prediction.get('probabilities') is None, f'Failure row must have null probabilities: {request_id}')
            metadata = set(prediction) - {'request_id', 'status', 'choice', 'probabilities'}
            require(metadata, f'Failure row is missing error metadata: {request_id}')

    profile = manifest.get('profile')
    if profile == 'core':
        for request_id, request in requests.items():
            require(request['member_seed'] == 0 and request['order_seed'] == 0,
                    f'Core profile must use member/order seed 0: {request_id}')

    for pair_id, members in pairs.items():
        require(set(members) == {'present', 'absent'}, f'Incomplete matched pair: {pair_id}')
        present, absent = members['present'], members['absent']
        for field in ('case_id', 'task', 'split', 'candidate_size', 'name_mode', 'with_none',
                      'member_seed', 'order_seed', 'none_mode'):
            require(present[field] == absent[field], f'Pair differs on {field}: {pair_id}')
        present_order = [key for key in present['criteria'] if key != 'none']
        absent_order = [key for key in absent['criteria'] if key != 'none']
        require(len(present_order) == len(absent_order), f'Pair changes candidate count: {pair_id}')
        differing = [i for i in range(len(present_order)) if present_order[i] != absent_order[i]]
        require(len(differing) == 1, f'Pair must substitute exactly one ordinary option at the same slot: {pair_id}')
        slot = differing[0]
        require(present_order[slot] == present['gold_option'],
                f'Pair substitution must replace the reference option: {pair_id}')
        require(absent_order[slot] not in present['criteria'],
                f'Pair replacement is not a new option: {pair_id}')
        for key in set(present_order) & set(absent_order):
            require(present['criteria'][key] == absent['criteria'][key],
                    f'Pair changes retained description: {pair_id}')

    if not allow_partial:
        require(set(case_id for case_id, _ in full_baselines) == set(cases),
                'Incomplete run: every case must have at least one full-set baseline')
        case_ids_with_requests = {request['case_id'] for request in requests.values()}
        missing_cases = sorted(set(cases) - case_ids_with_requests)
        require(not missing_cases, f'Incomplete run: cases without requests: {missing_cases[:5]}')
        missing_baselines = []
        for (case_id, name_mode), settings in full_baselines.items():
            if settings != {False, True}:
                missing_baselines.append((case_id, name_mode, sorted(settings)))
        require(not missing_baselines,
                f'Incomplete run: missing full-set baselines (with/without NONE): {missing_baselines[:5]}')

    return manifest, cases, requests, predictions, index


# ---------------------------------------------------------------------------
# Family extraction
# ---------------------------------------------------------------------------


def group_variants(group_requests, none_mode, seeds=None):
    """Return case_id -> [(member_seed, order_seed, present_request, absent_request)]."""
    record = defaultdict(lambda: defaultdict(dict))
    for request in group_requests:
        if request['presence'] not in ('present', 'absent') or not request['with_none']:
            continue
        if request['none_mode'] != none_mode:
            continue
        if seeds is not None and (request['member_seed'], request['order_seed']) not in seeds:
            continue
        record[request['case_id']][(request['member_seed'], request['order_seed'])][request['presence']] = request
    variants = {}
    for case_id, by_variant in record.items():
        rows = []
        for member_seed, order_seed in sorted(by_variant):
            slot = by_variant[(member_seed, order_seed)]
            if 'present' in slot and 'absent' in slot:
                rows.append((member_seed, order_seed, slot['present'], slot['absent']))
        variants[case_id] = rows
    return variants


def extract_families(variants, cases, index, predictions, candidate_size, name_mode):
    """Per-split native partitions and complete-score families.

    A score family contributes only when both the present and the absent call of
    the same variant have usable scores (status ``ok`` and probabilities for a
    continuous row).  Ineligible/failed variants are counted, never imputed.
    """
    output = {}
    for split in ('calibration', 'test'):
        data = {
            'native_present': defaultdict(list), 'native_absent': defaultdict(list),
            'scores': {score: {'present': defaultdict(list), 'absent': defaultdict(list),
                               'present_correct': defaultdict(list)} for score in SCORE_NAMES},
            'missing': {score: 0 for score in SCORE_NAMES},
            'total_variants': 0, 'failures': defaultdict(int),
        }
        for case_id, rows in sorted(variants.items()):
            case = cases[case_id]
            if case['split'] != split:
                continue
            for member_seed, order_seed, present_request, absent_request in rows:
                data['total_variants'] += 1
                present_prediction = predictions[present_request['request_id']]
                absent_prediction = predictions[absent_request['request_id']]
                data['native_present'][case_id].append(present_category(present_request, present_prediction))
                data['native_absent'][case_id].append(absent_category(absent_request, absent_prediction))
                for prediction in (present_prediction, absent_prediction):
                    if status_of(prediction) != 'ok':
                        data['failures'][status_of(prediction)] += 1

                plain_present = index.get((case_id, name_mode, 'present', candidate_size, False,
                                           member_seed, order_seed, 'aligned'))
                plain_absent = index.get((case_id, name_mode, 'absent', candidate_size, False,
                                          member_seed, order_seed, 'aligned'))
                for score in SCORE_NAMES:
                    if score == 'none_probability':
                        present_value = none_probability(present_request, present_prediction)
                        absent_value = none_probability(absent_request, absent_prediction)
                        correct = None
                    else:
                        present_value = absent_value = None
                        correct = None
                        if plain_present is not None and plain_absent is not None:
                            present_plain_prediction = predictions[plain_present['request_id']]
                            absent_plain_prediction = predictions[plain_absent['request_id']]
                            present_scores = plain_scores(plain_present, present_plain_prediction)
                            absent_scores = plain_scores(plain_absent, absent_plain_prediction)
                            if present_scores is not None and absent_scores is not None:
                                present_value = present_scores[score]
                                absent_value = absent_scores[score]
                    if present_value is not None and score == 'none_probability':
                        correct = argmax_ordinary(present_request, present_prediction) == present_request['gold_option']
                    elif present_value is not None and plain_present is not None:
                        correct = argmax_ordinary(plain_present, predictions[plain_present['request_id']]) == plain_present['gold_option']
                    if present_value is None or absent_value is None:
                        data['missing'][score] += 1
                    else:
                        data['scores'][score]['present'][case_id].append(float(present_value))
                        data['scores'][score]['absent'][case_id].append(float(absent_value))
                        data['scores'][score]['present_correct'][case_id].append(bool(correct))
        output[split] = data
    for split, data in output.items():
        for score in SCORE_NAMES:
            require(set(data['scores'][score]['present']) == set(data['scores'][score]['absent']),
                    f'Paired score-family correspondence broken for {score} in {split}')
    return output


def coverage_stats(variants, cases, predictions, split, replicates, seed, bins=10):
    """Coverage Brier/reliability for raw p(NONE) on balanced present/absent pairs."""
    pairs = []
    for case_id, rows in sorted(variants.items()):
        if cases[case_id]['split'] != split:
            continue
        for _, _, present_request, absent_request in rows:
            present_value = none_probability(present_request, predictions[present_request['request_id']])
            absent_value = none_probability(absent_request, predictions[absent_request['request_id']])
            if present_value is not None and absent_value is not None:
                pairs.extend(((case_id, present_value, 0), (case_id, absent_value, 1)))
    result = brier_stats(pairs, replicates, stable_seed(seed, 'coverage', split))
    result['reliability'] = reliability_curve([(p, o) for _, p, o in pairs], bins)
    result['note'] = 'Balanced constructed mixture: each variant contributes one present and one absent call.'
    return result


def conditional_ordinary_stats(variants, cases, index, predictions, candidate_size, name_mode, split, with_none=False):
    """Conditional ordinary choice calibration on present plain aligned calls."""
    pairs = []
    zero_mass = 0
    defined = 0
    decisions = []
    for case_id, rows in sorted(variants.items()):
        if cases[case_id]['split'] != split:
            continue
        for member_seed, order_seed, _, _ in rows:
            request = index.get((case_id, name_mode, 'present', candidate_size, with_none,
                                 member_seed, order_seed, 'aligned'))
            if request is None:
                continue
            prediction = predictions[request['request_id']]
            if status_of(prediction) != 'ok' or probabilities_of(prediction) is None:
                continue
            probabilities = probabilities_of(prediction)
            ordinary = [key for key in request['criteria'] if key != 'none']
            total = float(sum(probabilities[key] for key in ordinary))
            best = ordinary[0]
            for key in ordinary[1:]:
                if probabilities[key] > probabilities[best]:
                    best = key
            decisions.append(best == request['gold_option'])
            if total <= 0:
                zero_mass += 1
                continue
            normalized = {key: probabilities[key] / total for key in ordinary}
            best = ordinary[0]
            for key in ordinary[1:]:
                if normalized[key] > normalized[best]:
                    best = key
            pairs.append((case_id, float(normalized[best]), 1.0 if best == request['gold_option'] else 0.0))
            defined += 1
    result = brier_stats(pairs, 0, 0)
    result['defined'] = defined
    result['zero_ordinary_mass'] = zero_mass
    result['undefined_fraction'] = (zero_mass / (zero_mass + defined)) if (zero_mass + defined) else None
    result['decision_accuracy'] = (float(np.mean(decisions)) if decisions else None)
    result['decision_n'] = len(decisions)
    result['note'] = ('Predicted probability is the renormalized ordinary mass of the offered-order argmax; '
                      'all-zero ordinary mass is undefined for calibration but stays in the decision denominator.')
    return result


# ---------------------------------------------------------------------------
# Group summary
# ---------------------------------------------------------------------------


def summarize_group(key, group_requests, cases, index, predictions, replicates, limit, seed):
    task, candidate_size, name_mode, none_mode = key
    group_key = f'{task}|k{candidate_size}|{name_mode}|{none_mode}'
    variants = group_variants(group_requests, none_mode)
    families = extract_families(variants, cases, index, predictions, candidate_size, name_mode)
    has_probabilities = any(probabilities_of(predictions[request['request_id']]) is not None
                            for request in group_requests)
    member_sets = {frozenset(key for key in present['criteria'] if key != 'none')
                   for rows in variants.values() for (_, _, present, _) in rows}
    failure_metadata = defaultdict(int)
    for request in group_requests:
        prediction = predictions[request['request_id']]
        if status_of(prediction) != 'ok':
            for field in set(prediction) - {'request_id', 'status', 'choice', 'probabilities'}:
                failure_metadata[field] += 1
    summary = {'group': group_key, 'task': task, 'candidate_size': candidate_size,
               'name_mode': name_mode, 'none_mode': none_mode,
               'member_seeds': sorted({request['member_seed'] for request in group_requests}),
               'order_seeds': sorted({request['order_seed'] for request in group_requests}),
               'multiple_variants': len({(request['member_seed'], request['order_seed'])
                                         for request in group_requests}) > 1,
               'n_unique_member_sets': len(member_sets),
               'failure_metadata_fields': dict(sorted(failure_metadata.items())),
               'has_probabilities': has_probabilities, 'n_text': {}, 'n_variants': {}, 'native': {},
               'scores': {}, 'coverage': {}, 'conditional_ordinary': {}, 'conditional_ordinary_with_none': {}}
    for split in ('calibration', 'test'):
        data = families[split]
        text_ids = [case_id for case_id, rows in variants.items() if cases[case_id]['split'] == split and rows]
        summary['n_text'][split] = len(text_ids)
        summary['n_variants'][split] = data['total_variants']
        present_rates = {category: rate_family({text: [value == category for value in values]
                                                for text, values in data['native_present'].items()},
                                               replicates, stable_seed(seed, group_key, split, 'present', category))
                         for category in NATIVE_PRESENT}
        absent_rates = {category: rate_family({text: [value == category for value in values]
                                               for text, values in data['native_absent'].items()},
                                              replicates, stable_seed(seed, group_key, split, 'absent', category))
                        for category in NATIVE_ABSENT}
        absent_reject = {text: [value == 'rejected' for value in values]
                         for text, values in data['native_absent'].items()}
        present_reject = {text: [value == 'rejected' for value in values]
                          for text, values in data['native_present'].items()}
        delta = {text: float(np.mean(absent_reject[text])) - float(np.mean(present_reject[text]))
                 for text in absent_reject if text in present_reject}
        summary['native'][split] = {
            'present': present_rates, 'absent': absent_rates,
            'paired_delta': {'n_text': len(delta), 'text_mean': float(np.mean(list(delta.values()))) if delta else None,
                             'bootstrap95': bootstrap_ids(delta, replicates,
                                                          stable_seed(seed, group_key, split, 'delta'),
                                                          lambda ids: float(np.mean([delta[t] for t in ids]))),
                             'note': 'Detection minus false rejection, text-cluster bootstrap.'},
            'failures_by_status': {status: int(count) for status, count in sorted(data['failures'].items())},
        }
        summary['coverage'][split] = coverage_stats(variants, cases, predictions, split, replicates,
                                                    stable_seed(seed, group_key, split, 'coverage'))
        summary['conditional_ordinary'][split] = conditional_ordinary_stats(
            variants, cases, index, predictions, candidate_size, name_mode, split) if none_mode == 'aligned' else None
        summary['conditional_ordinary_with_none'][split] = conditional_ordinary_stats(
            variants, cases, index, predictions, candidate_size, name_mode, split, True) if none_mode == 'aligned' else None

    for score in SCORE_NAMES:
        entry = {'eligible': {}, 'missing_families': {},
                 'calibration': None, 'test': None, 'auc': {}, 'undefined_reason': None}
        present_maps = {}
        for split in ('calibration', 'test'):
            data = families[split]['scores'][score]
            present = {text: values for text, values in data['present'].items() if values}
            absent = {text: values for text, values in data['absent'].items() if values}
            correct = {text: values for text, values in data['present_correct'].items() if values}
            present_maps[split] = (present, absent, correct)
            entry['eligible'][split] = {'n_text': len(set(present) & set(absent)),
                                        'n_variants': int(sum(len(present[t]) for t in set(present) & set(absent)))}
            entry['missing_families'][split] = int(families[split]['missing'][score])
            entry['auc'][split] = auc_stats(present, absent, replicates,
                                            stable_seed(seed, group_key, split, score, 'auc'))
        cal_present, cal_absent, cal_correct = present_maps['calibration']
        test_present, test_absent, test_correct = present_maps['test']
        if cal_present and cal_absent:
            threshold = select_threshold_weighted(cal_present, cal_absent, limit)
            entry['calibration'] = evaluate_policy(cal_present, cal_absent, cal_correct, threshold, replicates,
                                                   stable_seed(seed, group_key, score, 'cal'))
            if test_present and test_absent:
                entry['test'] = evaluate_policy(test_present, test_absent, test_correct, threshold, replicates,
                                                stable_seed(seed, group_key, score, 'test'))
            entry['threshold'] = threshold
        else:
            entry['threshold'] = None
            entry['undefined_reason'] = ('no complete score families in calibration'
                                         if has_probabilities else 'discrete-only track: no probabilities')
        summary['scores'][score] = entry
    return summary


def build_controls(requests, predictions, replicates, seed):
    """Plain accuracy and full-set controls, text-equal across variants."""
    buckets = defaultdict(lambda: defaultdict(list))
    for request in requests.values():
        if request['presence'] == 'present' and not request['with_none']:
            label = (request['task'], request['name_mode'], 'plain_present', 'aligned')
        elif request['presence'] == 'full':
            label = (request['task'], request['name_mode'],
                     'full_with_none' if request['with_none'] else 'full_plain', request['none_mode'])
        else:
            continue
        buckets[label][request['case_id']].append(present_category(request, predictions[request['request_id']]))
    output = {}
    for label, by_case in sorted(buckets.items()):
        key = '|'.join(label)
        entry = {'n_text': len(by_case), 'n_variants': int(sum(len(v) for v in by_case.values()))}
        for category in ('correct', 'wrong', 'interface_failure'):
            entry[category] = rate_family({text: [value == category for value in values]
                                           for text, values in by_case.items()},
                                          replicates, stable_seed(seed, key, category))
        output[key] = entry
    return output


# ---------------------------------------------------------------------------
# Primary and sensitivity transfer
# ---------------------------------------------------------------------------


def load_subset_plan(path):
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    plan = json.loads(path.read_text(encoding='utf-8'))
    require(isinstance(plan.get('equal_budget_calibration'), dict) and plan['equal_budget_calibration'],
            'Subset plan missing equal_budget_calibration')
    return plan


def subset_ids_for(plan, task):
    entry = plan['equal_budget_calibration'].get(task)
    require(entry is not None, f'Subset plan has no equal-budget calibration for {task}')
    return set(entry['case_ids'])


def restrict(by_text, case_ids):
    return {text: values for text, values in by_text.items() if text in case_ids and values}


def fit_threshold(families, key, score, case_ids, limit):
    data = families[key]['calibration']['scores'][score]
    present = restrict(data['present'], case_ids)
    absent = restrict(data['absent'], case_ids)
    if not present or not absent:
        return None
    return select_threshold_weighted(present, absent, limit)


def eval_test(families, key, score, threshold, replicates, seed):
    data = families[key]['test']['scores'][score]
    present = {t: v for t, v in data['present'].items() if v}
    absent = {t: v for t, v in data['absent'].items() if v}
    correct = {t: v for t, v in data['present_correct'].items() if v}
    if not present or not absent:
        return None
    return evaluate_policy(present, absent, correct, threshold, replicates, seed)


def eval_calibration_subset(families, key, score, case_ids, threshold, replicates, seed):
    data = families[key]['calibration']['scores'][score]
    present = restrict(data['present'], case_ids)
    absent = restrict(data['absent'], case_ids)
    correct = restrict(data['present_correct'], case_ids)
    if not present or not absent:
        return None
    return evaluate_policy(present, absent, correct, threshold, replicates, seed)


def joint_bootstrap(families, source_key, target_key, score, case_ids, limit, replicates, seed):
    """Supplementary draw that refits the source threshold on resampled texts."""
    source = families[source_key]['calibration']['scores'][score]
    source_present = restrict(source['present'], case_ids)
    source_absent = restrict(source['absent'], case_ids)
    target = families[target_key]['test']['scores'][score]
    target_present = {t: v for t, v in target['present'].items() if v}
    target_absent = {t: v for t, v in target['absent'].items() if v}
    if not source_present or not source_absent or not target_present or not target_absent:
        return None
    source_texts = sorted(set(source_present) & set(source_absent))
    target_texts = sorted(set(target_present) & set(target_absent))
    rng = np.random.default_rng(seed)
    detections, false_rejections = [], []
    for _ in range(replicates):
        draw = [source_texts[i] for i in rng.integers(0, len(source_texts), size=len(source_texts))]
        try:
            # Distinct draw keys retain bootstrap multiplicity; keys by text
            # would collapse repeated draws and fit on unique texts instead.
            threshold = select_threshold_weighted({i: source_present[t] for i, t in enumerate(draw)},
                                                  {i: source_absent[t] for i, t in enumerate(draw)}, limit)
        except ValidationError:
            continue
        draw_target = [target_texts[i] for i in rng.integers(0, len(target_texts), size=len(target_texts))]
        detections.append(float(np.mean([float(np.mean(np.asarray(target_absent[t]) > threshold))
                                         for t in draw_target])))
        false_rejections.append(float(np.mean([float(np.mean(np.asarray(target_present[t]) > threshold))
                                               for t in draw_target])))
    return {'detection_bootstrap95': percentiles(detections),
            'false_rejection_bootstrap95': percentiles(false_rejections),
            'replicates_used': len(detections),
            'note': 'Source threshold refit on resampled calibration texts; target test texts resampled jointly.'}


def build_transfer(groups, cases, index, predictions, plan, replicates, seed, limit,
                   full_matrix=False, joint=False):
    if plan is None:
        return {'available': False, 'note': 'No subset plan supplied; primary transfer not computed.'}
    missing = {task: len(set(subset_ids_for(plan, task)) - set(cases))
               for task in sorted({key[0] for key in groups})}
    if any(missing.values()):
        return {'available': False, 'missing_frozen_calibration_texts': missing,
                'note': 'This profile does not contain the complete frozen equal-budget calibration subsets; transfer is computed on core only.'}
    core_keys = [key for key in sorted(groups) if key[3] == 'aligned']
    require(core_keys, 'No aligned groups available for transfer')
    families = {}
    for key in core_keys:
        variants = group_variants(groups[key], 'aligned', seeds={(0, 0)})
        families[key] = extract_families(variants, cases, index, predictions, key[1], key[2])
    for key in core_keys:
        ids = subset_ids_for(plan, key[0])
        require(ids, f'Empty equal-budget subset for {key[0]}')
    subset_note = 'Thresholds fit on the frozen 80-text equal-budget calibration subset per task.'

    rows = []
    primary_pairs = [(source, target) for source in core_keys if source[1] == 2 and source[2] == 'natural'
                     for target in core_keys if target[1] == 2 and target[2] == 'natural' and target[0] != source[0]]

    count_name_pairs = []
    for task in sorted({key[0] for key in core_keys}):
        natural_two = (task, 2, 'natural', 'aligned')
        if natural_two in families:
            for target in core_keys:
                if target[0] == task and target[2] == 'natural' and target[1] != 2:
                    count_name_pairs.append(('count', natural_two, target))
        for source in core_keys:
            if source[0] == task and source[2] == 'natural':
                target = (task, source[1], 'neutral', 'aligned')
                if target in families:
                    count_name_pairs.append(('name', source, target))

    def emit(kind, source, target, score, threshold, source_cal, target_test, local, joint_result=None):
        rows.append({
            'transfer_type': kind,
            'source': f'{source[0]}|k{source[1]}|{source[2]}|{source[3]}',
            'target': f'{target[0]}|k{target[1]}|{target[2]}|{target[3]}',
            'source_task': source[0], 'target_task': target[0],
            'score': score, 'threshold': threshold,
            'source_calibration_subset': source_cal, 'target_test': target_test,
            'local_reference': local, 'joint_source_uncertainty': joint_result,
            'supplemental': 'trec' in (source[0], target[0]),
        })

    for source, target in primary_pairs:
        source_ids = subset_ids_for(plan, source[0])
        target_ids = subset_ids_for(plan, target[0])
        for score in SCORE_NAMES:
            threshold = fit_threshold(families, source, score, source_ids, limit)
            if threshold is None:
                continue
            source_cal = eval_calibration_subset(families, source, score, source_ids, threshold, replicates,
                                                 stable_seed(seed, 'transfer', source, score, 'src'))
            target_test = eval_test(families, target, score, threshold, replicates,
                                    stable_seed(seed, 'transfer', target, score, 'tgt'))
            local_threshold = fit_threshold(families, target, score, target_ids, limit)
            local = None
            if local_threshold is not None:
                local = {'threshold': local_threshold,
                         'target_test': eval_test(families, target, score, local_threshold, replicates,
                                                  stable_seed(seed, 'transfer', target, score, 'local'))}
            joint_result = None
            if joint and (source, target) in primary_pairs:
                joint_result = joint_bootstrap(families, source, target, score, source_ids, limit,
                                               replicates, stable_seed(seed, 'transfer', source, target, score, 'joint'))
            emit('cross_task', source, target, score, threshold, source_cal, target_test, local, joint_result)

    for kind, source, target in count_name_pairs:
        source_ids = subset_ids_for(plan, source[0])
        target_ids = subset_ids_for(plan, target[0])
        for score in SCORE_NAMES:
            threshold = fit_threshold(families, source, score, source_ids, limit)
            if threshold is None:
                continue
            source_cal = eval_calibration_subset(families, source, score, source_ids, threshold, replicates,
                                                 stable_seed(seed, 'transfer', source, score, kind, 'src'))
            target_test = eval_test(families, target, score, threshold, replicates,
                                    stable_seed(seed, 'transfer', target, score, kind, 'tgt'))
            local_threshold = fit_threshold(families, target, score, target_ids, limit)
            local = None
            if local_threshold is not None:
                local = {'threshold': local_threshold,
                         'target_test': eval_test(families, target, score, local_threshold, replicates,
                                                  stable_seed(seed, 'transfer', target, score, kind, 'local'))}
            emit(kind, source, target, score, threshold, source_cal, target_test, local)

    if full_matrix:
        for source in core_keys:
            for target in core_keys:
                if source == target:
                    continue
                if (source, target) in primary_pairs:
                    continue
                source_ids = subset_ids_for(plan, source[0])
                target_ids = subset_ids_for(plan, target[0])
                for score in SCORE_NAMES:
                    threshold = fit_threshold(families, source, score, source_ids, limit)
                    if threshold is None:
                        continue
                    source_cal = eval_calibration_subset(families, source, score, source_ids, threshold,
                                                         replicates, stable_seed(seed, 'transfer', source, score, 'fm', 'src'))
                    target_test = eval_test(families, target, score, threshold, replicates,
                                            stable_seed(seed, 'transfer', target, score, 'fm', 'tgt'))
                    local_threshold = fit_threshold(families, target, score, target_ids, limit)
                    local = None
                    if local_threshold is not None:
                        local = {'threshold': local_threshold,
                                 'target_test': eval_test(families, target, score, local_threshold, replicates,
                                                          stable_seed(seed, 'transfer', target, score, 'fm', 'local'))}
                    emit('full_matrix', source, target, score, threshold, source_cal, target_test, local)

    return {'available': True, 'note': subset_note, 'rows': rows,
            'count_name_note': ('Count/name transfers reuse each task\'s frozen 80-text calibration subset; '
                                'target texts are reused across configurations and rows are correlated.'),
            'trEC_supplemental_note': 'Rows whose source or target task is TREC are supplemental support-shift diagnostics.',
            'no_target_selection_note': ('All ordered source/target, score and threshold configurations are reported; '
                                         'no score or source was selected using target test outcomes.')}


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


def analyze(run_dir, bootstrap_replicates=2000, seed=DEFAULT_SEED, limit=DEFAULT_LIMIT,
            subset_plan=None, allow_partial=False, full_matrix=False, joint=True):
    manifest, cases, requests, predictions, index = validate_run(run_dir, allow_partial=allow_partial)
    groups = defaultdict(list)
    for request in requests.values():
        if request['presence'] == 'full':
            continue
        none_mode = request['none_mode'] if request['with_none'] else 'aligned'
        groups[(request['task'], request['candidate_size'], request['name_mode'], none_mode)].append(request)

    summaries = {}
    for key in sorted(groups):
        summaries[f'{key[0]}|k{key[1]}|{key[2]}|{key[3]}'] = summarize_group(
            key, groups[key], cases, index, predictions, bootstrap_replicates, limit, seed)

    controls = build_controls(requests, predictions, bootstrap_replicates, seed)
    plan = load_subset_plan(subset_plan)
    transfer = build_transfer(groups, cases, index, predictions, plan, bootstrap_replicates, seed, limit,
                              full_matrix=full_matrix, joint=joint)

    source_code_path = Path(__file__).resolve()
    provenance = {
        'source_run_dir': str(Path(run_dir).resolve()),
        'source_sha256': {name: sha256_path(Path(run_dir) / name)
                          for name in ('manifest.json', 'cases.jsonl', 'requests.jsonl', 'predictions.jsonl')},
        'subset_plan': None,
        'analysis_code': {'path': SOURCE_CODE, 'sha256': sha256_path(source_code_path)},
        'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
    }
    if plan is not None:
        subset_path = Path(subset_plan).resolve()
        provenance['subset_plan'] = {'path': str(subset_path), 'sha256': sha256_path(subset_path),
                                     'index_cases_sha256': plan.get('index_cases_sha256'),
                                     'equal_budget_seed': plan.get('equal_budget_seed'),
                                     'robustness_seed': plan.get('robustness_seed')}

    return {
        'protocol': PROTOCOL, 'manifest_protocol': manifest.get('protocol'),
        'manifest_status': manifest.get('status'), 'model': manifest.get('model'),
        'profile': manifest.get('profile'),
        'n_cases': len(cases), 'n_requests': len(requests), 'n_predictions': len(predictions),
        'bootstrap_replicates': bootstrap_replicates, 'bootstrap_seed': seed,
        'calibration_false_rejection_limit': limit,
        'allow_partial': allow_partial,
        'validation': {'complete': True, 'n_groups': len(summaries),
                       'n_cases_with_requests': len({r['case_id'] for r in requests.values()})},
        'provenance': provenance,
        'groups': summaries,
        'controls': controls,
        'transfer': transfer,
        'notes': [
            'Independent public-data confirmation scoring; the v1 validator is not reused because it assumes one replicate and mandatory probabilities.',
            'Native partitions use the with-NONE call; failures are a separate category and are never counted as wrong or as a zero score.',
            'Rate families average per text over variants first, then over texts equally; bootstrap resamples text families (2,000 draws, seed 20261005).',
            'Intervals are marginal descriptive intervals, not p-values, universal rankings, risk guarantees or simultaneous tests.',
            'Thresholds maximize empirical calibration detection subject to present false rejection <= 5% with strict score > threshold.',
            'Calibration thresholds use complete-score families only; discrete-only models keep native decisions and receive no invented scores.',
            'Coverage Brier/reliability uses raw p(NONE) on the balanced constructed mixture; zero ordinary mass is reported separately.',
            'The primary transfer never selects a score or source using target test outcomes; target-local recalibration is a separate label-consuming reference.',
            'Confirmation outputs stay separate from v1 and discovery artifacts.',
        ],
    }


def _fmt(value, digits=3):
    if value is None:
        return 'n/a'
    if isinstance(value, str):
        return value
    return f'{value:.{digits}f}'


def render_report(metrics):
    lines = ['# Paper 1 confirmation scoring report', '',
             f"Protocol: `{metrics['protocol']}`", '',
             f"Cases: {metrics['n_cases']}; requests: {metrics['n_requests']}; predictions: {metrics['n_predictions']}.",
             f"Bootstrap replicates: {metrics['bootstrap_replicates']}; seed: {metrics['bootstrap_seed']}; "
             f"calibration F limit: {metrics['calibration_false_rejection_limit']}.", '']
    lines += ['## Native decision partitions', '',
              'Per-text equal weight over variants; `n_text`/`n_variants` and pooled fractions are in metrics.json.', '',
              '| Group | Split | n text | n var | P correct | P wrong | P rejected | P fail | A rejected | A accepted | A fail |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key, group in sorted(metrics['groups'].items()):
        for split in ('calibration', 'test'):
            if not group['n_variants'].get(split):
                continue
            present = group['native'][split]['present']
            absent = group['native'][split]['absent']
            lines.append('| ' + ' | '.join([
                key.replace('|', ' / '), split, str(group['n_text'][split]), str(group['n_variants'][split]),
                _fmt(present['correct']['text_mean']), _fmt(present['wrong']['text_mean']),
                _fmt(present['rejected']['text_mean']), _fmt(present['interface_failure']['text_mean']),
                _fmt(absent['rejected']['text_mean']), _fmt(absent['accepted']['text_mean']),
                _fmt(absent['interface_failure']['text_mean'])]) + ' |')
    lines += ['', '## Score ranking and frozen-threshold calibration', '',
              '| Group | Split | Score | Eligible texts | AUC | Threshold (cal) | Test D | Test F |',
              '|---|---|---|---:|---:|---:|---:|---:|']
    for key, group in sorted(metrics['groups'].items()):
        for split in ('calibration', 'test'):
            for score in SCORE_NAMES:
                entry = group['scores'][score]
                auc_value = entry['auc'][split]['pooled']
                threshold = entry.get('threshold')
                test = entry.get('test')
                lines.append('| ' + ' | '.join([
                    key.replace('|', ' / '), split, score, str(entry['eligible'][split]['n_text']), _fmt(auc_value),
                    _fmt(threshold, 5), _fmt(test['detection']['text_mean'] if test else None),
                    _fmt(test['false_rejection']['text_mean'] if test else None)]) + ' |')
    lines += ['', '## Coverage and conditional ordinary calibration', '',
              '| Group | Split | Coverage n | Brier | Prevalence | Ordinary defined | Zero ordinary mass | Decision accuracy |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for key, group in sorted(metrics['groups'].items()):
        for split in ('calibration', 'test'):
            coverage = group['coverage'][split]
            ordinary = group['conditional_ordinary'][split]
            lines.append('| ' + ' | '.join([
                key.replace('|', ' / '), split, str(coverage['n_pairs']), _fmt(coverage['brier']),
                _fmt(coverage['prevalence']), str(ordinary['defined'] if ordinary else 'n/a'),
                str(ordinary['zero_ordinary_mass'] if ordinary else 'n/a'),
                _fmt(ordinary['decision_accuracy'] if ordinary else None)]) + ' |')
    lines += ['', '## Plain and full-set controls', '',
              '| Control | n text | n var | Correct | Wrong | Interface failure |', '|---|---:|---:|---:|---:|---:|']
    for key, entry in sorted(metrics['controls'].items()):
        lines.append('| ' + ' | '.join([
            key.replace('|', ' / '), str(entry['n_text']), str(entry['n_variants']), _fmt(entry['correct']['text_mean']),
            _fmt(entry['wrong']['text_mean']), _fmt(entry['interface_failure']['text_mean'])]) + ' |')
    transfer = metrics['transfer']
    lines += ['', '## Policy transfer', '']
    if not transfer.get('available'):
        lines += [f"- Not computed: {transfer.get('note')}"]
    else:
        lines += [f"- {transfer['note']}", '', '| Type | Source | Target | Score | Threshold | Test D | Test F | Local D | Local F | Joint D CI |',
                  '|---|---|---|---:|---:|---:|---:|---:|---:|---|']
        for row in transfer['rows']:
            target = row['target_test']
            local = row['local_reference']['target_test'] if row['local_reference'] else None
            joint = row.get('joint_source_uncertainty')
            lines.append('| ' + ' | '.join([
                row['transfer_type'], row['source'].replace('|', ' / '), row['target'].replace('|', ' / '), row['score'], _fmt(row['threshold'], 5),
                _fmt(target['detection']['text_mean'] if target else None),
                _fmt(target['false_rejection']['text_mean'] if target else None),
                _fmt(local['detection']['text_mean'] if local else None),
                _fmt(local['false_rejection']['text_mean'] if local else None),
                str(joint['detection_bootstrap95']) if joint else 'n/a']) + ' |')
    lines += ['', '## Interpretation boundaries', ''] + ['- ' + note for note in metrics['notes']]
    lines += ['', 'Full partitions, Wilson intervals, bootstrap intervals, eligibility counts, reliability bins, '
              'thresholds, AUCs, transfer rows and provenance hashes are in `metrics.json`.', '']
    return '\n'.join(lines)


def run(run_dir, output_dir, bootstrap_replicates=2000, seed=DEFAULT_SEED, limit=DEFAULT_LIMIT,
        subset_plan=None, allow_partial=False, full_matrix=False, joint=True):
    source = Path(run_dir).resolve()
    output = Path(output_dir).resolve()
    require(not output.exists(), f'Refusing to overwrite existing output directory: {output}')
    require(source != output and source not in output.parents,
            'Output directory must be outside the frozen run directory')
    metrics = analyze(source, bootstrap_replicates=bootstrap_replicates, seed=seed, limit=limit,
                      subset_plan=subset_plan, allow_partial=allow_partial, full_matrix=full_matrix, joint=joint)
    output.mkdir(parents=True)
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (output / 'report.md').write_text(render_report(metrics), encoding='utf-8')
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--bootstrap-replicates', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--limit', type=float, default=DEFAULT_LIMIT)
    parser.add_argument('--subset-plan',
                        default='outputs/paper1-confirmation-analysis-plan-20261003/subsets.json')
    parser.add_argument('--allow-partial', action='store_true',
                        help='Permit scoring a run whose case/request matrix is intentionally partial.')
    parser.add_argument('--full-matrix', action='store_true',
                        help='Add the exploratory all-ordered-group transfer matrix.')
    parser.add_argument('--no-joint-bootstrap', action='store_true',
                        help='Skip the supplementary source-threshold-refit bootstrap.')
    args = parser.parse_args(argv)
    metrics = run(args.run_dir, args.output_dir, bootstrap_replicates=args.bootstrap_replicates,
                  seed=args.seed, limit=args.limit, subset_plan=args.subset_plan,
                  allow_partial=args.allow_partial, full_matrix=args.full_matrix,
                  joint=not args.no_joint_bootstrap)
    print(f"Validated {metrics['n_predictions']} predictions across {metrics['validation']['n_groups']} groups; "
          f"wrote metrics.json and report.md")


if __name__ == '__main__':
    main()
