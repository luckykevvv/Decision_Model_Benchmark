"""Download pinned source files and freeze cases and request interventions."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import urllib.request

if __package__:
    from .specs import ROOT, TASKS, SEED, PROFILES, NONE_DESCRIPTION, key_for
else:
    from specs import ROOT, TASKS, SEED, PROFILES, NONE_DESCRIPTION, key_for

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def text_key(text):
    return hashlib.sha256(' '.join(text.casefold().split()).encode()).hexdigest()

def dump_jsonl(path, rows):
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')

def get_source(task_name, split):
    spec = TASKS[task_name]
    filename = ('train_5500.label' if split == 'train' else 'TREC_10.label') if spec['format'] == 'trec_raw' else f"{spec['prefix']}/{split}-00000-of-00001.parquet"
    local = ROOT / 'cache/datasets' / task_name / spec['revision'] / Path(filename).name
    local.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{filename}"
    if not local.exists():
        temporary = local.with_suffix(local.suffix + '.download')
        print(f'Downloading {task_name}/{split}', flush=True)
        request = urllib.request.Request(url, headers={'User-Agent': 'laya-systematic-evaluation/1.0'})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        temporary.replace(local)
    return local, {'url': url, 'sha256': digest(local), 'repo': spec['repo'], 'revision': spec['revision']}

def load_rows(task, path):
    if TASKS[task]['format'] == 'trec_raw':
        mapping = {label: index for index, (label, _) in enumerate(TASKS[task]['labels'])}
        rows = []
        for line in path.read_text(encoding='latin-1').splitlines():
            category, text = line.split(' ', 1)
            rows.append({'label': mapping[category.split(':')[0]], 'text': text.strip()})
        return rows
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    labels = table.column('label').to_pylist()
    if task == 'dbpedia':
        texts = [f'{title}\n{content}' for title, content in zip(table.column('title').to_pylist(), table.column('content').to_pylist(), strict=True)]
    else:
        texts = table.column('text').to_pylist()
    return [{'label': label, 'text': text} for label, text in zip(labels, texts, strict=True)]

def prepare(profile, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ['cases.jsonl', 'requests.jsonl', 'manifest.json', 'predictions.jsonl']:
        if (output / name).exists():
            raise SystemExit(f'Refusing to overwrite a frozen run: {output / name}')
    calibration_n, test_n = PROFILES[profile]
    pilot_cases = [json.loads(line) for line in (ROOT / 'outputs/laya-missing-option-pilot/pilot_cases.jsonl').read_text(encoding='utf-8').splitlines()]
    prior = defaultdict(set)
    for case in pilot_cases:
        prior[case['task']].add(text_key(case['state']))
    cases, sources, counts = [], {}, {}
    for task_index, task in enumerate(TASKS):
        available = {}
        for source_split in ['train', 'test']:
            path, provenance = get_source(task, source_split)
            rows = load_rows(task, path)
            sources[f'{task}/{source_split}'] = {**provenance, 'total_rows': len(rows)}
            available[source_split] = rows
        seen = set(prior[task])
        rng = random.Random(SEED + task_index)
        task_counts = {}
        # Test selected first; calibration excludes every selected test text and pilot text.
        for source_split, split, requested in [('test', 'test', test_n), ('train', 'calibration', calibration_n)]:
            buckets = defaultdict(list)
            within = set()
            for row_idx, row in enumerate(available[source_split]):
                key = text_key(row['text'])
                if key in seen or key in within:
                    continue
                within.add(key)
                buckets[int(row['label'])].append((row_idx, row))
            for label_id, (label, _) in enumerate(TASKS[task]['labels']):
                pool = buckets[label_id]
                if not pool:
                    raise RuntimeError(f'No examples of {task}/{split}/{label}')
                selected = rng.sample(pool, min(requested, len(pool)))
                task_counts[f'{split}/{label}'] = {'requested': requested, 'selected': len(selected), 'eligible': len(pool)}
                for row_idx, row in selected:
                    seen.add(text_key(row['text']))
                    cases.append({'case_id': f'{task}:{split}:{row_idx}', 'task': task, 'split': split,
                                  'label': label, 'label_id': label_id, 'text': row['text'],
                                  'row_idx': row_idx, 'source_split': source_split})
        counts[task] = task_counts
        print(f'Frozen {task}: {sum(x["selected"] for x in task_counts.values())} cases', flush=True)
    cases.sort(key=lambda c: (c['task'], c['split'], c['row_idx']))
    requests = []
    for case in cases:
        task, gold = case['task'], case['label_id']
        classes = list(range(len(TASKS[task]['labels'])))
        rng = random.Random(int(hashlib.sha256((str(SEED) + case['case_id']).encode()).hexdigest()[:16], 16))
        complete = classes[:]
        other = [label for label in classes if label != gold]
        conditions = [('full', len(classes), complete, None)] + [
            condition for k in sorted({2, min(4, len(classes)-1), len(classes)-1})
            for condition in build_pair(case, k, other, rng)]
        for mode in ['natural', 'neutral']:
            for presence, k, ordered, basepair in conditions:
                for with_none in [False, True]:
                    criteria = {key_for(label, task, mode): TASKS[task]['labels'][label][1] for label in ordered}
                    if with_none:
                        criteria['none'] = NONE_DESCRIPTION
                    request = {'case_id': case['case_id'], 'task': task, 'split': case['split'],
                        'presence': presence, 'candidate_size': k, 'name_mode': mode, 'with_none': with_none,
                        'replicate': 0, 'pair_id': f'{basepair}:{mode}:{int(with_none)}' if basepair else None,
                        'instruction': TASKS[task]['instruction'], 'criteria': criteria,
                        'gold_option': key_for(gold, task, mode) if presence != 'absent' else None,
                        'gold_label': case['label']}
                    key_payload = {**request, 'criteria': list(criteria.items())}
                    request['request_id'] = hashlib.sha256(json.dumps(key_payload, ensure_ascii=False).encode()).hexdigest()
                    requests.append(request)
    dump_jsonl(output / 'cases.jsonl', cases)
    dump_jsonl(output / 'requests.jsonl', requests)
    manifest = {'protocol': 'laya-systematic-v1', 'profile': profile, 'seed': SEED,
        'calibration_per_class_requested': calibration_n, 'test_per_class_requested': test_n,
        'counts': counts, 'sources': sources, 'task_specs': TASKS,
        'model': {'repo': 'convaiinnovations/laya', 'revision': '55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851',
                  'device': 'cpu', 'threads': 4, 'max_len': 512, 'head_max_len': 256},
        'file_sha256': {name: digest(output / name) for name in ['cases.jsonl', 'requests.jsonl']},
        'pilot_excluded_by_normalized_text': True, 'n_cases': len(cases), 'n_requests': len(requests)}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Prepared {len(cases)} cases, {len(requests)} requests in {output}', flush=True)

def build_pair(case, k, other, rng):
    absent = rng.sample(other, k)
    rng.shuffle(absent)
    present = absent[:]
    present[rng.randrange(k)] = case['label_id']
    pair = f'{case["case_id"]}:k{k}:r0'
    return [('present', k, present, pair), ('absent', k, absent, pair)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=PROFILES, default='v1')
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    prepare(args.profile, args.run_dir)
