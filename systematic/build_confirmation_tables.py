"""Populate manuscript tables only from completed validated confirmation runs."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
from decision_benchmark.artifacts import RESOURCES
RESULTS=RESOURCES/'reference'
LATEX=ROOT/'runs/displays'
TASKS=[('ag_news','AG News',3),('dbpedia','DBpedia',13),('emotion','Emotion',5),('trec','TREC*',5)]
MODELS=[('laya','Laya'),('jev','Jev'),('qwen','Qwen7B')]


def pct(value):
    return '--' if value is None else f'{100*value:.1f}'


def cell(rate):
    ci=rate['bootstrap95']
    point=pct(rate['text_mean'])
    return point if ci is None else f'{point} [{pct(ci[0])}, {pct(ci[1])}]'


def table_file(name,lines):
    (LATEX/'tables'/name).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def build():
    records=[]
    for key,label in MODELS:
        path=RESULTS/f'core-{key}/metrics.json'
        if path.exists():
            data=json.loads(path.read_text(encoding='utf-8'))
            assert data['validation']['complete'] and data['bootstrap_replicates']==2000
            records.append((key,label,data))
    assert records
    lines=[r'\begin{longtable}{llrrrrrr}',
      r'\caption{Reference-label confirmation with natural names at the largest incomplete count. Rates are percentages; covered correct/wrong/rejected/failure and missing rejection/acceptance/failure partitions retain all attempted test texts. Fail $P/A$ denotes covered/missing failure percentages. *TREC has no ABBR test support and is supplemental. Marginal 95\% text-bootstrap intervals are retained in source data.}\label{tab:confirmation_main}\\',
      r'\toprule Task & Model & Test $n$ & Correct & Wrong & $F$ & $D$ & Fail $P/A$\\\midrule\endfirsthead',
      r'\toprule Task & Model & Test $n$ & Correct & Wrong & $F$ & $D$ & Fail $P/A$\\\midrule\endhead']
    for task,title,k in TASKS:
        for key,label,data in records:
            group=data['groups'][f'{task}|k{k}|natural|aligned']
            native=group['native']['test'];p,a=native['present'],native['absent']
            lines.append(' & '.join([title,label,str(group['n_text']['test']),pct(p['correct']['text_mean']),pct(p['wrong']['text_mean']),
                                    pct(p['rejected']['text_mean']),pct(a['rejected']['text_mean']),
                                    f"{pct(p['interface_failure']['text_mean'])}/{pct(a['interface_failure']['text_mean'])}"])+r'\\')
    lines += [r'\bottomrule',r'\end{longtable}']
    table_file('confirmation_main.tex',lines)
    lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Recorded confirmation interfaces and configurations. Qwen7B denotes Qwen2.5-7B-Instruct. Model/checkpoint revisions and software versions are retained in runtime manifests; hosted internal compute is not inferred.}\label{tab:confirmation_models}',
           r'\begin{tabular}{p{0.13\linewidth}p{0.31\linewidth}p{0.40\linewidth}}',r'\toprule Model & Frozen model/interface & Recorded configuration\\\midrule']
    for key,label,data in records:
        if key=='laya':
            desc=r'Laya 0.3.21, pinned revision';settings=r'CUDA, 1 CPU thread; 512 sequence / 256 head tokens; batch 8. Native temperature handling.'
        elif key=='jev':
            desc=r'Jev 1.13.0, hosted Choice';settings=r'Four API workers; up to 64 questions per call; native choices and rounded two-decimal scores.'
        else:
            desc=r'Qwen2.5-7B-Instruct, pinned revision';settings=r'BF16, greedy; batch 32; input cap 2,048 / output cap 32 tokens; exact-key parser; discrete track.'
        lines.append(f'{label} & {desc} & {settings}'+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}'];table_file('confirmation_models.tex',lines)
    lines=[r'\begin{longtable}{llrrr}',r'\caption{Coverage-confidence diagnostics on balanced present/absent pairs, using natural names and the largest incomplete count. Brier scores use raw $p(\texttt{none})$; AUROC measures ranking. Laya and Jev provide probability outputs.}\label{tab:confirmation_calibration}\\',
           r'\toprule Task & Model & Brier & AUROC & Local calibrated $D/F$ (\%)\\\midrule\endfirsthead',
           r'\toprule Task & Model & Brier & AUROC & Local calibrated $D/F$ (\%)\\\midrule\endhead']
    for task,title,k in TASKS:
        for key,label,data in records:
            group=data['groups'][f'{task}|k{k}|natural|aligned'];coverage=group['coverage']['test'];score=group['scores']['none_probability']
            if coverage['brier'] is None:continue
            policy=score['test']
            lines.append(f"{title} & {label} & {coverage['brier_text_mean']:.3f} & {score['auc']['test']['pooled']:.3f} & {pct(policy['detection']['text_mean'])}/{pct(policy['false_rejection']['text_mean'])}"+r'\\')
    lines += [r'\bottomrule',r'\end{longtable}'];table_file('confirmation_calibration.tex',lines)
    lines=[r'\begin{longtable}{lllp{0.27\linewidth}p{0.27\linewidth}}',r'\caption{All none-score task transfers at $k=2$ and natural names, with exactly 80 calibration pairs per source/local reference. Cells show target $D/F$ percentages; local references consume target calibration labels. Every ordered task pair is retained. Frozen-threshold and refitted-source uncertainty are recorded separately in source data. *TREC comparisons are supplemental.}\label{tab:confirmation_transfer}\\',
           r'\toprule Model & Source & Target & Frozen source $D/F$ & Local reference $D/F$\\\midrule\endfirsthead',
           r'\toprule Model & Source & Target & Frozen source $D/F$ & Local reference $D/F$\\\midrule\endhead']
    names={task:title for task,title,k in TASKS}
    for key,label,data in records:
        for row in data['transfer'].get('rows',[]):
            if row['transfer_type']!='cross_task' or row['score']!='none_probability':continue
            target,local=row['target_test'],row['local_reference']['target_test']
            assert row['source_calibration_subset']['n_text']==80
            lines.append(' & '.join([label,names[row['source_task']],names[row['target_task']],
              f"{pct(target['detection']['text_mean'])}/{pct(target['false_rejection']['text_mean'])}",
              f"{pct(local['detection']['text_mean'])}/{pct(local['false_rejection']['text_mean'])}"])+r'\\')
    lines += [r'\bottomrule',r'\end{longtable}'];table_file('confirmation_transfer.tex',lines)
    lines=[r'\begin{longtable}{lllrp{0.20\linewidth}p{0.20\linewidth}}',r'\caption{Matched interface controls at the largest incomplete count, using natural names except for the name contrast. Changes in $D/F$ are percentage points with marginal paired-text 95\% bootstrap intervals (2,000 draws). Each task starts with 20 test texts; both-side validity determines eligible $n$. Excluded comparisons remain in source data, while native failure denominators retain all attempts. Order holds members fixed; wording holds keys/order fixed. Variants are not independent texts.}\label{tab:confirmation_controls}\\',
           r'\toprule Task & Model & Control & Text $n$ & $\Delta D$ [95\%] & $\Delta F$ [95\%]\\\midrule\endfirsthead',
           r'\toprule Task & Model & Control & Text $n$ & $\Delta D$ [95\%] & $\Delta F$ [95\%]\\\midrule\endhead']
    factor_names={'name':'ID minus natural','order':'Order 1 minus 0','none_wording':'Aligned minus old'}
    for key,label,data in records:
        path=RESULTS/f'controls-{key}.json'
        if not path.exists():continue
        controls=json.loads(path.read_text(encoding='utf-8'))
        for task,title,k in TASKS:
            for factor in factor_names:
                row=controls['groups'][f'{task}|k{k}|{factor}']
                lines.append(' & '.join([title,label,factor_names[factor],str(row['n_text']),
                  f"{pct(row['delta_d'])} [{pct(row['delta_d_bootstrap95'][0])}, {pct(row['delta_d_bootstrap95'][1])}]",
                  f"{pct(row['delta_f'])} [{pct(row['delta_f_bootstrap95'][0])}, {pct(row['delta_f_bootstrap95'][1])}]"])+r'\\')
    lines += [r'\bottomrule',r'\end{longtable}'];table_file('confirmation_controls.tex',lines)
    status={'completed_core_models':[label for key,label,data in records],
            'completed_core_responses':sum(data['n_predictions'] for key,label,data in records),
            'unique_confirmation_texts':1456,'test_texts':856,
            'public_oos_run_completed':all((RESULTS/f'intent-{key}.json').exists() for key in ('laya','jev','qwen')),
            'plot_scoreless_models_in_discrete_track_only':True}
    (LATEX/'manuscript_status.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(status,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results-dir',default=str(RESULTS));p.add_argument('--output-dir',required=True)
    args=p.parse_args();RESULTS=Path(args.results_dir);LATEX=Path(args.output_dir)
    (LATEX/'tables').mkdir(parents=True,exist_ok=True);build()
