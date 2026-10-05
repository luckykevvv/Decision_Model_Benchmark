import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

from systematic.confirmation_runtime import rows,digest,validate
from systematic.analyze import entropy,select_threshold


def rate(values,seed=20261005):
    if not values:return {'n_text':0,'value':None,'bootstrap95':None}
    array=np.asarray(values,dtype=float);rng=np.random.default_rng(seed)
    means=array[rng.integers(0,len(array),size=(2000,len(array)))].mean(axis=1)
    return {'n_text':len(values),'value':float(array.mean()),'bootstrap95':np.percentile(means,[2.5,97.5]).tolist()}


def analyze(run_dir):
    root=Path(run_dir);manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['file_sha256'].items():assert digest(root/name)==expected
    cases={c['case_id']:c for c in rows(root/'cases.jsonl')}
    request_rows=rows(root/'requests.jsonl');requests={r['request_id']:r for r in request_rows}
    prediction_rows=rows(root/'predictions.jsonl');predictions={p['request_id']:p for p in prediction_rows}
    assert len(predictions)==len(prediction_rows)==len(requests)==len(request_rows)==manifest['n_requests']
    assert set(predictions)==set(requests)
    cells={}
    groups=defaultdict(list)
    for r in requests.values():
        validate(predictions[r['request_id']],r,manifest.get('reported_probability_decimals'))
        assert r['global_oos']==cases[r['case_id']]['global_oos']
        assert r['reference_covered']==(cases[r['case_id']]['label'] in r['criteria'])
        cells[(r['case_id'],r['track'],r['presence'],r['candidate_size'],r['with_none'])]=r
        if r['with_none']:groups[(r['track'],r['candidate_size'],r['split'])].append(r)
    summaries={}
    for (track,k,split),group in sorted(groups.items()):
        covered=[r for r in group if r['reference_covered']]
        missing=[r for r in group if not r['reference_covered'] and not r['global_oos']]
        oos=[r for r in group if r['global_oos']]
        def flags(selected,category):
            values=[]
            for r in selected:
                p=predictions[r['request_id']];status=p.get('status','ok')
                if category=='failure':flag=status!='ok'
                elif category=='reject':flag=status=='ok' and p['choice']=='none'
                elif category=='correct':flag=status=='ok' and p['choice']==r['gold_option']
                elif category=='wrong':flag=status=='ok' and p['choice']!='none' and p['choice']!=r['gold_option']
                else:flag=status=='ok' and p['choice']!='none'
                values.append(flag)
            return rate(values)
        inscope=[r for r in group if not r['global_oos']]
        summaries[f'{track}|k{k}|{split}']={'n_queries':len({r['case_id'] for r in group}),
            'n_covered':len(covered),'n_missing_inscope':len(missing),'n_global_oos':len(oos),
            'retrieval_coverage':rate([r['reference_covered'] for r in inscope]) if track=='natural_retrieval' else None,
            'covered':{name:flags(covered,name) for name in ('correct','wrong','reject','failure')},
            'missing_inscope':{name:flags(missing,name) for name in ('reject','accept','failure')},
            'global_oos':{name:flags(oos,name) for name in ('reject','accept','failure')}}
    score_policies={}
    for track,k in sorted({(r['track'],r['candidate_size']) for r in requests.values()}):
        scores={split:defaultdict(list) for split in ('calibration','test')}
        for r in requests.values():
            if r['track']!=track or r['candidate_size']!=k or not r['with_none']:continue
            pp=predictions[r['request_id']]
            plain_r=cells[(r['case_id'],track,r['presence'],k,False)];plain=predictions[plain_r['request_id']]
            if pp.get('status','ok')!='ok' or plain.get('status','ok')!='ok' or pp.get('probabilities') is None or plain.get('probabilities') is None:continue
            score_values={'none_probability':pp['probabilities']['none'],
                          'negative_max':-max(plain['probabilities'].values()),
                          'normalized_entropy':entropy(list(plain['probabilities'].values()))}
            for score,value in score_values.items():scores[r['split']][score].append((r,value))
        for score,cal in scores['calibration'].items():
            p=[s for r,s in cal if r['reference_covered']];a=[s for r,s in cal if not r['reference_covered']]
            if not p or not a:continue
            threshold=select_threshold(p,a)
            test=scores['test'][score]
            score_policies[f'{track}|k{k}|{score}']={'threshold':threshold,'n_cal_covered':len(p),'n_cal_missing':len(a),
                'test_f':rate([s>threshold for r,s in test if r['reference_covered']]),
                'test_missing_inscope_d':rate([s>threshold for r,s in test if not r['reference_covered'] and not r['global_oos']]),
                'test_global_oos_d':rate([s>threshold for r,s in test if r['global_oos']]),
                'note':'Calibration-only threshold; missing pool uses the recorded calibration mixture, including public OOS in natural track.'}
    return {'model':manifest['model'],'n_cases':len(cases),'n_predictions':len(predictions),
            'source':manifest['source'],'retriever':manifest['retriever'],'summaries':summaries,'score_policies':score_policies,
            'sha256':{name:digest(root/name) for name in ('manifest.json','cases.jsonl','requests.jsonl','predictions.jsonl')},
            'notes':['Public dataset reference labels define the evaluation endpoints.',
                     'Artificial omission, naturally missed in-scope candidates and global OOS are separate populations.',
                     'No gold repair in natural retrieval; selected query texts excluded from training retrieval corpus.',
                     'Rates are conditional on each stated subset; no pooling into a deployment OOS prevalence estimate.',
                     '2000 text bootstrap draws, marginal. Synthetic present/absent copies are matched by query; per-track D/F intervals are marginal rates.',
                     'No invented probabilities for the discrete Qwen track.']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();output=Path(args.output);assert not output.exists()
    result=analyze(args.run_dir);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');print('Validated extension',result['n_predictions'],'responses')
