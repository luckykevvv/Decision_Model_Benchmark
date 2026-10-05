"""Paired name/order/NONE controls on fixed-input robustness families."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from systematic.confirmation_analysis import validate_run,status_of


def summarize(values,seed):
    ids=sorted(values)
    data=np.array([np.mean(values[key],axis=0) for key in ids])
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(ids),size=(2000,len(ids)))
    means=data[draws].mean(axis=1)
    return {'n_text':len(ids),'n_matched_variants':sum(len(v) for v in values.values()),
            'delta_d':float(data[:,0].mean()),'delta_f':float(data[:,1].mean()),
            'delta_d_bootstrap95':np.percentile(means[:,0],[2.5,97.5]).tolist(),
            'delta_f_bootstrap95':np.percentile(means[:,1],[2.5,97.5]).tolist()}


def analyze(run_dir):
    manifest,cases,requests,predictions,index=validate_run(run_dir)
    assert manifest.get('design_id')=='robustness-controlled','Pure control requires fixed-slot design'
    cells={}
    for r in requests.values():
        if r['with_none'] and r['presence'] in ('present','absent') and r['split']=='test':
            cells[(r['case_id'],r['candidate_size'],r['name_mode'],r['member_seed'],r['order_seed'],r['none_mode'],r['presence'])]=r
    groups=defaultdict(lambda:defaultdict(list))
    excluded=defaultdict(int)
    for case in cases.values():
        if case['split']!='test':continue
        counts=sorted({key[1] for key in cells if key[0]==case['case_id']})
        for k in counts:
            comparisons=[('name',('natural',m,o,'aligned'),('neutral',m,o,'aligned')) for m in range(5) for o in range(2)]
            comparisons += [('order',('natural',m,0,'aligned'),('natural',m,1,'aligned')) for m in range(5)]
            comparisons += [('none_wording',('natural',0,0,'legacy'),('natural',0,0,'aligned'))]
            for factor,left,right in comparisons:
                rs=[]
                for config in (left,right):
                    rs.append([cells[(case['case_id'],k,*config,presence)] for presence in ('absent','present')])
                if factor=='order':
                    assert all(set(rs[0][i]['criteria'])==set(rs[1][i]['criteria']) for i in (0,1))
                if factor=='none_wording':
                    assert all(list(rs[0][i]['criteria'])==list(rs[1][i]['criteria']) for i in (0,1))
                group=f"{case['task']}|k{k}|{factor}"
                responses=[[predictions[r['request_id']] for r in pair] for pair in rs]
                if any(status_of(p)!='ok' for pair in responses for p in pair):
                    excluded[group]+=1;continue
                delta=[int(responses[1][i]['choice']=='none')-int(responses[0][i]['choice']=='none') for i in (0,1)]
                groups[group][case['case_id']].append(delta)
    results={key:summarize(values,20261005+int(hashlib.sha256(key.encode()).hexdigest()[:6],16)) for key,values in groups.items()}
    return {'model':manifest['model'],'profile':manifest.get('design_id'),'groups':results,
            'excluded_interface_comparisons':dict(excluded),
            'directions':{'name':'neutral minus natural','order':'order1 minus order0','none_wording':'aligned minus legacy'},
            'bootstrap_replicates':2000,'notes':['Matched indicators averaged within each text before text-cluster bootstrap. Intervals marginal.',
                                              'Member intervention changes one absent distractor at a fixed slot; present set is invariant. It is not five independent text samples.',
                                              'No paired effect implies general causal attribution beyond these specific frozen inputs.']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output)
    assert not out.exists(),'Refusing to overwrite control analysis'
    result=analyze(Path(args.run_dir));out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('Completed',len(result['groups']),'paired control groups')
