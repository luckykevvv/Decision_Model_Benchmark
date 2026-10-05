"""One entry point for released inputs, adapters, scoring and displays."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from .artifacts import RESOURCES, PROFILES, materialize, new_output, read_json, json_bytes, sha

def verify_resources():
    checks=read_json(RESOURCES/'SHA256.json')
    for name, expected in checks.items():
        if sha(RESOURCES/name)!=expected:
            raise ValueError(f'Released artifact checksum mismatch: {name}')
    return {'release':'0.2.0','files_verified':len(checks),'status':'passed'}

def numeric_points(value, path=()):
    """Compare deterministic counts/points, independently of bootstrap budget."""
    if any(any(token in part.lower() for token in ('bootstrap','wilson','interval','replicate','sha256','provenance')) for part in path):
        return {}
    if isinstance(value, dict):
        result={}
        for key, child in value.items():result.update(numeric_points(child,path+(str(key),)))
        return result
    if isinstance(value,list):
        result={}
        for index,child in enumerate(value):result.update(numeric_points(child,path+(str(index),)))
        return result
    if type(value) in (int,float) and math.isfinite(value):return {path:float(value)}
    return {}

def compare_points(actual, reference):
    expected=numeric_points(reference)
    observed=numeric_points(actual)
    shared=set(expected)&set(observed)
    differences=[('/'.join(path),expected[path],observed[path]) for path in shared
                 if not math.isclose(expected[path],observed[path],rel_tol=1e-11,abs_tol=1e-12)]
    if differences:raise ValueError(f'Point estimates differ from released references: {differences[:8]}')
    if not shared:raise ValueError('No reference point estimates compared')
    return {'numeric_points_compared':len(shared),'status':'passed'}

def score_run(profile, run_dir, output_dir, *, replicates=2000, joint=False, subset_plan=None):
    output=new_output(output_dir)
    if profile=='intent':
        from systematic.analyze_intent_extension import analyze
        metrics=analyze(run_dir)
        output.mkdir(parents=True)
        (output/'metrics.json').write_bytes(json_bytes(metrics))
    else:
        from systematic.confirmation_analysis import run
        metrics=run(run_dir,output,bootstrap_replicates=replicates,joint=joint,
                    subset_plan=subset_plan or RESOURCES/'subsets.json')
        if profile=='controlled':
            from systematic.confirmation_controls import analyze
            (output/'controls.json').write_bytes(json_bytes(analyze(run_dir)))
    return metrics

def reproduce(args):
    verify_resources()
    output=new_output(args.output_dir)
    profiles=PROFILES if args.profile=='all' else ('core',) if args.profile=='quick' else (args.profile,)
    release=read_json(RESOURCES/'release.json')
    selected=[]
    for profile in profiles:
        choices=release['profiles'][profile]['models']
        models=['jev'] if args.profile=='quick' else choices if args.model=='all' else [args.model]
        if any(model not in choices for model in models):raise ValueError(f'Model not released in {profile}: {models}')
        selected.extend((profile,model) for model in models)
    output.mkdir(parents=True)
    checks=[]
    for profile,model in selected:
        print(f'Scoring {profile}/{model} from recorded predictions...',flush=True)
        run=materialize(profile,output/'runs'/profile/model,model,quick=args.profile=='quick')
        metrics=score_run(profile,run,output/'results'/f'{profile}-{model}',replicates=args.bootstrap_replicates,
                          joint=args.joint_bootstrap)
        if args.profile=='quick':
            reference=read_json(RESOURCES/'reference'/f'core-{model}/metrics.json')
            check=compare_points(metrics['groups'],{k:v for k,v in reference['groups'].items() if k.startswith('ag_news|')})
        else:
            reference_path=RESOURCES/'reference'/f'intent-{model}.json' if profile=='intent' else RESOURCES/'reference'/f'{profile}-{model}/metrics.json'
            check=compare_points(metrics,read_json(reference_path))
        checks.append({'profile':profile,'model':model,'n_predictions':metrics['n_predictions'],**check})
    report={'release':'0.2.0','profile':args.profile,'classification_bootstrap_replicates':args.bootstrap_replicates,
            'intent_bootstrap_replicates':2000,'supplementary_source_refit':args.joint_bootstrap,
            'verification':checks,'new_inference':False,
            'note':'Quick/reduced-bootstrap runs check integration and points; published classification intervals use 2000 draws.'}
    (output/'reproduction_checks.json').write_bytes(json_bytes(report))
    print(json.dumps(report,indent=2))

def run_model(args):
    source=Path(args.inputs).resolve()
    manifest=read_json(source/'manifest.json')
    if manifest.get('case_representation')=='text-index-v1':
        raise ValueError('Inference requires text: use prepare --with-text first.')
    cache=Path(args.cache_dir).resolve()
    os.environ['DECISION_BENCHMARK_CACHE']=str(cache)
    if args.model=='jev':
        if args.budget_usd is None or not math.isfinite(args.budget_usd) or args.budget_usd<=0:
            raise ValueError('Jev inference requires an explicit positive --budget-usd.')
        from systematic.run_confirmation_jev import run
        run(SimpleNamespace(inputs=source,output_dir=args.output_dir,workers=args.workers,
                            budget_usd=args.budget_usd,preflight=args.preflight))
    elif args.model=='laya':
        from systematic.run_confirmation_laya import run
        run(SimpleNamespace(inputs=source,output_dir=args.output_dir,batch_size=args.batch_size or 8,
                            device=args.device or 'cpu',preflight=args.preflight))
    elif args.model=='qwen':
        if args.device=='cpu':raise ValueError('The pinned Qwen BF16 adapter requires CUDA.')
        from systematic.run_confirmation_llm import run
        run(SimpleNamespace(inputs=source,output_dir=args.output_dir,batch_size=args.batch_size or 32,preflight=args.preflight))
    else:
        if manifest.get('profile')!='intent-extension':raise ValueError('BM25 scope gate applies to CLINC150 inputs.')
        from systematic.intent_scope_baseline import run
        run(source,args.output_dir,cache/'datasets')

def displays(args):
    output=new_output(args.output_dir)
    output.mkdir(parents=True)
    results=Path(args.metrics_dir).resolve()
    calls=[('systematic.plot_confirmation_protocol',['--output-dir',str(output/'figures')]),
           ('systematic.plot_confirmation_results',['--results',str(results),'--output-dir',str(output/'figures')]),
           ('systematic.build_confirmation_tables',['--results-dir',str(results),'--output-dir',str(output)]),
           ('systematic.build_intent_table',['--results-dir',str(results),'--output-dir',str(output)])]
    for module,arguments in calls:subprocess.run([sys.executable,'-m',module,*arguments],check=True)

def parser():
    p=argparse.ArgumentParser(description='Decision Model Benchmark: prepare, run, score and reproduce.')
    p.add_argument('--version',action='version',version='0.2.0')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('verify',help='Verify bundled artifact checksums without downloads.')
    sub.add_parser('list',help='List frozen profiles, model coverage and response counts.')
    prep=sub.add_parser('prepare',help='Materialize exact frozen inputs; optionally reconstruct source texts.')
    prep.add_argument('--profile',choices=PROFILES,required=True)
    prep.add_argument('--output-dir',required=True)
    prep.add_argument('--with-text',action='store_true')
    prep.add_argument('--cache-dir',default='cache/datasets')
    prep.add_argument('--offline',action='store_true')
    rep=sub.add_parser('reproduce',help='Score frozen predictions without GPUs, API keys or source downloads.')
    rep.add_argument('--profile',choices=('quick',*PROFILES,'all'),default='quick')
    rep.add_argument('--model',choices=('all','laya','jev','qwen','bm25'),default='all')
    rep.add_argument('--output-dir',required=True)
    rep.add_argument('--bootstrap-replicates',type=int,default=2000)
    rep.add_argument('--joint-bootstrap',action='store_true')
    score=sub.add_parser('score',help='Score adapter outputs in the shared frozen-run schema.')
    score.add_argument('--profile',choices=PROFILES,required=True)
    score.add_argument('--run-dir',required=True)
    score.add_argument('--output-dir',required=True)
    score.add_argument('--subset-plan',help='Explicit calibration subsets for a custom transfer experiment.')
    score.add_argument('--bootstrap-replicates',type=int,default=2000)
    score.add_argument('--joint-bootstrap',action='store_true')
    run=sub.add_parser('run',help='Run a model adapter on prepared text inputs; resume validated predictions.')
    run.add_argument('--model',choices=('laya','jev','qwen','bm25'),required=True)
    run.add_argument('--inputs',required=True)
    run.add_argument('--output-dir',required=True)
    run.add_argument('--cache-dir',default='cache')
    run.add_argument('--device',choices=('cpu','cuda'))
    run.add_argument('--batch-size',type=int)
    run.add_argument('--workers',type=int,default=4)
    run.add_argument('--budget-usd',type=float)
    run.add_argument('--preflight',action='store_true')
    down=sub.add_parser('download-model',help='Download the pinned Laya or Qwen checkpoint without inference.')
    down.add_argument('--model',choices=('laya','qwen'),required=True)
    down.add_argument('--cache-dir',default='cache')
    fig=sub.add_parser('displays',help='Rebuild four confirmation figures and six tables from complete reference metrics.')
    fig.add_argument('--metrics-dir',default=str(RESOURCES/'reference'))
    fig.add_argument('--output-dir',required=True)
    return p

def main(argv=None):
    p=parser();args=p.parse_args(argv)
    try:
        if getattr(args,'bootstrap_replicates',1)<1:raise ValueError('Bootstrap replicate count must be positive.')
        if getattr(args,'batch_size',None) is not None and args.batch_size<1:raise ValueError('Batch size must be positive.')
        if getattr(args,'workers',1)<1:raise ValueError('Worker count must be positive.')
        if args.command=='verify':print(json.dumps(verify_resources(),indent=2))
        elif args.command=='list':print(json.dumps(read_json(RESOURCES/'release.json')['profiles'],indent=2))
        elif args.command=='prepare':
            verify_resources()
            print(materialize(args.profile,args.output_dir,with_text=args.with_text,cache_dir=args.cache_dir,offline=args.offline))
        elif args.command=='reproduce':reproduce(args)
        elif args.command=='score':score_run(args.profile,args.run_dir,args.output_dir,replicates=args.bootstrap_replicates,
                                           joint=args.joint_bootstrap,subset_plan=args.subset_plan)
        elif args.command=='run':run_model(args)
        elif args.command=='displays':displays(args)
        else:
            from huggingface_hub import snapshot_download
            from systematic.run_confirmation_llm import REPO,REVISION
            repo,revision=('convaiinnovations/laya','55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851') if args.model=='laya' else (REPO,REVISION)
            print(snapshot_download(repo,revision=revision,cache_dir=str(Path(args.cache_dir)/'huggingface')))
    except (ValueError,FileNotFoundError,RuntimeError) as error:p.exit(2,f'error: {error}\n')

if __name__=='__main__':main()
