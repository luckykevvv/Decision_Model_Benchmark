import argparse
from collections import defaultdict
import importlib.metadata
import json
from pathlib import Path
import time

from systematic.confirmation_runtime import ROOT, digest, failure, preflight_cases, prepare_run, save_runtime, validate
from systematic.model_runtime import load_agent, REVISION


def run(args):
    device = getattr(args, 'device', 'cuda')
    model = {'repo': 'convaiinnovations/laya', 'revision': REVISION, 'device': device, 'threads': 1,
             'max_len': 512, 'head_max_len': 256, 'laya_version': '0.3.21'}
    output = Path(args.output_dir)
    manifest, cases, requests, existing = prepare_run(args.inputs, output, model)
    pending = [r for r in requests.values() if r['request_id'] not in existing]
    if args.preflight:
        selected = preflight_cases(cases)
        pending = [r for r in pending if r['case_id'] in selected]
    if not pending:
        print(f'No pending responses ({len(existing)}/{len(requests)})')
        return
    agent, snapshot = load_agent(device)
    import torch
    torch.set_num_threads(1)
    assert importlib.metadata.version('laya') == '0.3.21'
    save_runtime(output, {'model': model, 'settings': {'batch_size': args.batch_size},
        'code_sha256': {name: digest(ROOT / 'systematic' / name) for name in ('run_confirmation_laya.py', 'confirmation_runtime.py', 'model_runtime.py')},
        'package_versions': {name: importlib.metadata.version(name) for name in ('torch','transformers','laya','numpy')},
        'snapshot': snapshot, 'effective_temperature': agent.temperature,
        'effective_temperature_by_options': agent.temperature_by_options})
    groups = defaultdict(list)
    for r in pending:
        groups[json.dumps([r['instruction'], list(r['criteria'].items())])].append(r)
    added, start = 0, time.perf_counter()
    with (output / 'predictions.jsonl').open('a', encoding='utf-8') as stream:
        for gi, group in enumerate(groups.values()):
            r0 = group[0]
            question = {'topic': {'type': 'choice', 'instructions': r0['instruction'], 'criteria': r0['criteria']}}
            for offset in range(0, len(group), args.batch_size):
                batch = group[offset:offset+args.batch_size]
                t0 = time.perf_counter()
                try:
                    answers = agent.predict_batch([cases[r['case_id']]['text'] for r in batch], question,
                                                 batch_size=args.batch_size, max_len=512, head_max_len=256)
                    assert len(answers) == len(batch)
                    saved = []
                    for request, answer in zip(batch, answers):
                        a = answer['answers']['topic']
                        p = {'request_id': request['request_id'], 'status':'ok', 'choice':a['choice'],
                             'probabilities':a['probabilities'], 'batch_seconds':time.perf_counter()-t0,
                             'batch_n':len(batch), 'usage':answer.get('usage')}
                        validate(p, request)
                        saved.append(p)
                except Exception as exc:
                    saved = [failure(r, exc) for r in batch]
                for prediction in saved:
                    stream.write(json.dumps(prediction) + '\n')
                    added += 1
                stream.flush()
            if gi % 100 == 0 or gi+1 == len(groups):
                print(f'Laya saved {len(existing)+added}/{len(requests)}; group {gi+1}/{len(groups)}; {time.perf_counter()-start:.1f}s', flush=True)
    print(f'Finished Laya {len(existing)+added}/{len(requests)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--preflight', action='store_true')
    run(parser.parse_args())
