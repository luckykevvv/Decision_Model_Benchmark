"""Immutable released artifacts and explicit text-index run materialization."""
import gzip
import hashlib
import json
from pathlib import Path

RESOURCES = Path(__file__).resolve().parent / 'resources'
PROFILES = ('core', 'controlled', 'intent')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def read_gzip(path):
    return gzip.decompress(Path(path).read_bytes())

def rows(content):
    return [json.loads(line) for line in content.decode('utf-8').splitlines() if line.strip()]

def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()

def jsonl_bytes(values):
    return ''.join(json.dumps(value, ensure_ascii=False)+'\n' for value in values).encode()

def new_output(path):
    path = Path(path).resolve()
    if path.exists():
        raise ValueError(f'Refusing to overwrite {path}; choose a new output directory.')
    return path

def materialize(profile, output, model=None, *, with_text=False, cache_dir=None, offline=False, quick=False):
    """Build a derived analysis run, or byte-identical original input files."""
    if profile not in PROFILES:
        raise ValueError(f'Unknown profile: {profile}')
    output = new_output(output)
    source = RESOURCES/'frozen'/profile
    manifest_path = source/'input_manifest.json' if model is None else source/'models'/model/'manifest.json'
    manifest = read_json(manifest_path)
    cases = rows(read_gzip(source/'case_index.jsonl.gz'))
    requests = read_gzip(source/'requests.jsonl.gz')
    original_hashes = dict(manifest['file_sha256'])
    if hashlib.sha256(requests).hexdigest() != original_hashes['requests.jsonl']:
        raise ValueError('Released requests differ from their original fingerprint.')
    if with_text:
        from .datasets import reconstruct
        case_bytes = reconstruct(profile, cases, cache_dir=cache_dir, offline=offline)
        if hashlib.sha256(case_bytes).hexdigest() != original_hashes['cases.jsonl']:
            raise ValueError('Reconstructed case bytes differ from the original frozen cases.')
        representation = 'original-text'
    else:
        case_bytes = jsonl_bytes(cases)
        representation = 'text-index-v1'
    prediction_bytes = None
    if model is not None:
        prediction_bytes = read_gzip(source/'models'/model/'predictions.jsonl.gz')
        release = read_json(RESOURCES/'release.json')
        expected = release['original_artifact_sha256'][f'{profile}/{model}/predictions.jsonl']
        if hashlib.sha256(prediction_bytes).hexdigest() != expected:
            raise ValueError('Released predictions differ from their original fingerprint.')
    if quick:
        if profile != 'core' or with_text:
            raise ValueError('Quick is the recorded AG News core slice.')
        cases = [case for case in cases if case['task']=='ag_news']
        ids = {case['case_id'] for case in cases}
        selected = [request for request in rows(requests) if request['case_id'] in ids]
        request_ids = {request['request_id'] for request in selected}
        case_bytes, requests = jsonl_bytes(cases), jsonl_bytes(selected)
        if prediction_bytes is not None:
            prediction_bytes = jsonl_bytes([p for p in rows(prediction_bytes) if p['request_id'] in request_ids])
        manifest['n_cases'], manifest['n_requests'] = len(cases), len(selected)
        manifest['release_slice'] = 'ag_news only; quick integration check, not the complete published benchmark'
    manifest['case_representation'] = representation
    manifest['original_input_sha256'] = original_hashes
    manifest['file_sha256'] = {'cases.jsonl': hashlib.sha256(case_bytes).hexdigest(),
                               'requests.jsonl': hashlib.sha256(requests).hexdigest()}
    manifest['benchmark_release'] = '0.2.0'
    output.mkdir(parents=True)
    (output/'cases.jsonl').write_bytes(case_bytes)
    (output/'requests.jsonl').write_bytes(requests)
    (output/'manifest.json').write_bytes(json_bytes(manifest))
    if prediction_bytes is not None:
        (output/'predictions.jsonl').write_bytes(prediction_bytes)
    return output
