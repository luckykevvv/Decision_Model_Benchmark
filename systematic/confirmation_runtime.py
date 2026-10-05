"""Shared frozen-input validation, native-response resume and failure accounting."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_run(source, output, model, decimals=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not (output / 'manifest.json').exists():
        manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
        assert manifest['protocol'] == 'paper1-confirmation-v2'
        for name in ('cases.jsonl', 'requests.jsonl'):
            shutil.copyfile(source / name, output / name)
        manifest['source_model'] = manifest['model']
        manifest['model'] = model
        if decimals is not None:
            manifest['reported_probability_decimals'] = decimals
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['model'] == model, 'Changed model on resume'
    for name, expected in manifest['file_sha256'].items():
        assert digest(output / name) == expected, f'Changed frozen inputs: {name}'
        assert digest(source / name) == expected, f'Changed source inputs: {name}'
    cases = {c['case_id']: c for c in rows(output / 'cases.jsonl')}
    assert all(isinstance(c.get('text'), str) and c['text'].strip() for c in cases.values()), 'Inference requires reconstructed text; prepare inputs with --with-text.'
    request_rows = rows(output / 'requests.jsonl')
    requests = {r['request_id']: r for r in request_rows}
    assert len(requests) == len(request_rows) == manifest['n_requests']
    existing = {}
    if (output / 'predictions.jsonl').exists():
        for prediction in rows(output / 'predictions.jsonl'):
            identity = prediction['request_id']
            assert identity not in existing and identity in requests
            validate(prediction, requests[identity], decimals)
            existing[identity] = prediction
    return manifest, cases, requests, existing


def validate(prediction, request, decimals=None):
    status = prediction.get('status', 'ok')
    if status != 'ok':
        assert status in ('interface_failure', 'truncation_failure')
        assert prediction.get('choice') is None and prediction.get('probabilities') is None
        return
    assert prediction['choice'] in request['criteria'], 'Invalid choice'
    probabilities = prediction.get('probabilities')
    if probabilities is None:
        return  # Explicit discrete-only model track.
    assert set(probabilities) == set(request['criteria']), 'Changed response keys'
    assert all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in probabilities.values())
    tolerance = .001 if decimals is None else .5 * 10 ** (-decimals) * len(probabilities) + 1e-9
    assert sum(probabilities.values()) > 0 and abs(sum(probabilities.values())-1) <= tolerance


def save_runtime(output, data):
    output = Path(output)
    path = output / 'runtime.json'
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        for name in ('model', 'code_sha256', 'settings'):
            assert previous[name] == data[name], f'Changed {name} on resume'
    else:
        data['created_at_utc'] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(data, indent=2, default=str) + '\n', encoding='utf-8')


def preflight_cases(cases):
    selected = {}
    for case in cases.values():
        if case['split'] == 'calibration':
            selected.setdefault(case['task'], case['case_id'])
    return set(selected.values())


def failure(request, exc, status='interface_failure'):
    return {'request_id': request['request_id'], 'status': status, 'choice': None,
            'probabilities': None, 'error_type': type(exc).__name__,
            'error': str(exc)[:300]}
