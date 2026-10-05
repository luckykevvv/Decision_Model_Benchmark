import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time

from systematic.confirmation_runtime import ROOT, digest, failure, preflight_cases, prepare_run, save_runtime, validate
from systematic.jev_api import Client, MODEL, ENDPOINT, USD_PER_MILLION_INPUT


def run(args):
    output = Path(args.output_dir)
    model = {'provider':'TypeSafe', 'id':MODEL, 'endpoint':ENDPOINT}
    manifest, cases, requests, existing = prepare_run(args.inputs, output, model, decimals=2)
    selected = preflight_cases(cases) if args.preflight else set(cases)
    pending = [r for r in requests.values() if r['request_id'] not in existing and r['case_id'] in selected]
    if not pending:
        print(f'No pending responses ({len(existing)}/{len(requests)})')
        return
    client = Client(output, args.budget_usd)
    save_runtime(output, {'model':model, 'settings':{'workers':args.workers, 'chunk_questions':64, 'budget_usd':args.budget_usd},
        'code_sha256':{name:digest(ROOT/'systematic'/name) for name in ('run_confirmation_jev.py','confirmation_runtime.py','jev_api.py')},
        'available_models':client.models(), 'price_usd_per_million_input':USD_PER_MILLION_INPUT,
        'documentation':'https://docs.typesafe.ai/models', 'score_precision':'two decimal raw scores; entropy normalizes'})
    jobs, groups = [], defaultdict(list)
    for r in pending:
        groups[r['case_id']].append(r)
    for case, group in groups.items():
        for offset in range(0,len(group),64):
            jobs.append((case,group[offset:offset+64]))
    def infer(case, batch):
        questions = {r['request_id']:{'type':'choice','instructions':r['instruction'],'criteria':r['criteria']} for r in batch}
        try:
            native, call_id, seconds = client.evaluate(cases[case]['text'],questions)
            result = [{'request_id':rid,'status':'ok','choice':a['choice'],'probabilities':a['probabilities'],
                       'confidence':a['confidence'],'model':native['model'],'api_call_id':call_id,
                       'batch_seconds':seconds,'batch_n':len(batch)} for rid,a in native['answers'].items()]
            for p in result:
                validate(p,requests[p['request_id']],2)
            return result
        except Exception as exc:
            # Client sanitizes its exceptions and retains native call ledger.
            return [failure(r,exc) for r in batch]
    completed, added, start = 0,0,time.perf_counter()
    with (output/'predictions.jsonl').open('a',encoding='utf-8') as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(infer,*job) for job in jobs]
        for future in as_completed(futures):
            for p in future.result():
                stream.write(json.dumps(p,ensure_ascii=False)+'\n')
                added+=1
            stream.flush();completed+=1
            if completed%50==0 or completed==len(jobs):
                print(f'Jev saved {len(existing)+added}/{len(requests)}; calls {completed}/{len(jobs)}; {time.perf_counter()-start:.1f}s; ledger USD {client.used*USD_PER_MILLION_INPUT/1e6:.4f}',flush=True)
    print(f'Finished Jev {len(existing)+added}/{len(requests)}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--budget-usd',type=float,default=1)
    run(parser.parse_args())
