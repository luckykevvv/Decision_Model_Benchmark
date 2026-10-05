"""F2-F4 contract: native partitions, confidence quality and frozen transfer.

Only complete observed runs are drawn; no mock rows or missing-model zeroes.
F2 decomposes outcomes and actual policy operations. F3 stratifies confidence
validity by public task. F4 keeps detection loss and false rejection separate.
Core source n is text n, intervals marginal 2000-text-bootstrap diagnostics.
Python exports editable PDF/SVG and 600-dpi previews at 180 mm width.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch,Rectangle

ROOT=Path(__file__).resolve().parents[1]
from tools.figure_layout_check import require_matplotlib_panel_alignment

TASKS=[('ag_news','AG News',3),('dbpedia','DBpedia',13),('emotion','Emotion',5),('trec','TREC*',5)]
MODELS=[('laya','Laya','#3676A5'),('jev','Jev','#BF6F34'),('qwen','Qwen7B','#73767D')]
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,
 'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})


def load(results):
    records=[]
    for key,label,color in MODELS:
        path=results/f'core-{key}/metrics.json'
        if path.exists():
            metrics=json.loads(path.read_text(encoding='utf-8'))
            assert metrics['validation']['complete'] and not metrics.get('allow_partial')
            assert metrics['bootstrap_replicates']==2000
            records.append((key,label,color,metrics,path))
    assert records,'No completed core metrics'
    return records


def export(fig,out,name,source):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(out/f'{name}.alignment.json'),
                                      tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(out/f'{name}.pdf')
    fig.savefig(out/f'{name}.svg')
    fig.savefig(out/f'{name}.png',dpi=600)
    fig.savefig(out/f'{name}.preview.png',dpi=150)
    (out/f'{name}.source.json').write_text(json.dumps(source,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    plt.close(fig)


def rate(row):
    return row['text_mean']


def interval(row):
    return row['bootstrap95']


def plot_partitions(records,out,provenance):
    rows=[]
    for task,title,k in TASKS:
        for key,label,color,data,path in records:
            group=data['groups'][f'{task}|k{k}|natural|aligned']
            rows.append({'task':task,'title':title,'model':label,'k':k,'color':color,'group':group})
    fig,axes=plt.subplots(1,3,figsize=(180/25.4,125/25.4))
    fig.subplots_adjust(left=.21,right=.99,bottom=.27,top=.91,wspace=.15)
    y=np.arange(len(rows))
    colors={'correct':'#6C9A86','wrong':'#D49C68','rejected':'#5B82A5','interface_failure':'#343B43','accepted':'#D49C68'}
    for j,(ax,side,title,categories) in enumerate([
        (axes[0],'present','Present outcomes',('correct','wrong','rejected','interface_failure')),
        (axes[1],'absent','Absent outcomes',('rejected','accepted','interface_failure'))]):
        left=np.zeros(len(rows))
        for category in categories:
            values=np.array([rate(row['group']['native']['test'][side][category]) for row in rows])
            ax.barh(y,values,left=left,height=.66,color=colors[category],edgecolor='white',linewidth=.25)
            left+=values
        assert np.allclose(left,1)
        ax.set_xlim(0,1);ax.set_ylim(len(rows)-.5,-.5);ax.set_xticks([0,.5,1]);ax.set_xlabel('Fraction of test texts')
        ax.set_yticks(y);ax.set_title(title,pad=8)
        ax.set_yticklabels([f"{r['title']} / {r['model']}" for r in rows] if j==0 else ['']*len(rows))
        ax.tick_params(axis='y',length=0)
        ax.text(0,1.025,chr(97+j),transform=ax.transAxes,fontweight='bold',fontsize=9,va='bottom')
    ax=axes[2]
    for i,row in enumerate(rows):
        native=row['group']['native']['test']
        threshold=row['group']['scores']['none_probability'].get('test')
        for offset,metric,color in [(-.17,'detection','#3676A5'),(.17,'false_rejection','#6F757C')]:
            value=native['absent']['rejected'] if metric=='detection' else native['present']['rejected']
            ci=interval(value);point=rate(value)
            if ci:
                ax.plot(ci,[i+offset,i+offset],color=color,lw=.8)
            ax.scatter([point],[i+offset],color=color,s=15,marker='o',zorder=3)
            if threshold:
                ax.scatter([rate(threshold[metric])],[i+offset],color=color,s=17,marker='D',edgecolor='white',lw=.3,zorder=4)
    ax.set_xlim(-.03,1.03);ax.set_ylim(len(rows)-.5,-.5);ax.set_xticks([0,.5,1]);ax.set_yticks(y);ax.set_yticklabels(['']*len(rows));ax.tick_params(axis='y',length=0)
    ax.set_title('Native and calibrated D/F',pad=8);ax.set_xlabel('Rate');ax.text(0,1.025,'c',transform=ax.transAxes,fontweight='bold',fontsize=9,va='bottom')
    handles=[Patch(color=colors['correct'],label='Correct'),Patch(color=colors['wrong'],label='Wrong / false accept'),
             Patch(color=colors['rejected'],label='Reject'),Patch(color=colors['interface_failure'],label='Failure'),
             Line2D([],[],marker='o',ls='',color='#555555',label='Native'),Line2D([],[],marker='D',ls='',color='#555555',label='Calibrated')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.59,.025),ncol=3,fontsize=7,columnspacing=1.1)
    fig.text(.21,.14,'c: blue = detection; gray = false rejection. *TREC has five supported test classes.',fontsize=7)
    export(fig,out,'confirmation_comparison',{'provenance':provenance,'n_definition':'independent test text; core single variant',
          'native_uncertainty':'marginal text-bootstrap95, 2000 draws; conditional on observed configuration',
          'calibrated_policy':'same-task full calibration subset, no test tuning; unavailable for discrete Qwen',
          'rows':[{k:v for k,v in row.items() if k!='color'} for row in rows]})


def score_records(records):
    return [record for record in records if record[3]['groups']['ag_news|k3|natural|aligned']['coverage']['test']['brier'] is not None]


def plot_reliability(records,out,provenance):
    records=score_records(records)
    if not records:return
    fig,axes=plt.subplots(2,2,figsize=(180/25.4,125/25.4))
    fig.subplots_adjust(left=.10,right=.98,bottom=.16,top=.90,wspace=.28,hspace=.45)
    source=[]
    for i,(ax,(task,title,k)) in enumerate(zip(axes.flat,TASKS)):
        ax.plot([0,1],[0,1],ls='--',color='#B4BAC0',lw=.8)
        ax.set_xlim(-.04,1.04);ax.set_ylim(-.04,1.04);ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
        ax.set_xlabel('Mean reported p(NONE)');ax.set_ylabel('Observed absence fraction')
        ax.set_title(f'{title} (k={k})',pad=7);ax.text(0,1.06,chr(97+i),transform=ax.transAxes,fontsize=9,fontweight='bold',va='bottom')
        for key,label,color,data,path in records:
            coverage=data['groups'][f'{task}|k{k}|natural|aligned']['coverage']['test']
            bins=[b for b in coverage['reliability'] if b['n']]
            ax.plot([b['mean_predicted'] for b in bins],[b['observed'] for b in bins],marker='o',ms=3,lw=1,color=color,label=label)
            source.append({'model':label,'task':task,'k':k,'coverage':coverage})
    fig.legend([Line2D([],[],color=r[2],marker='o',ms=3) for r in records],[r[1] for r in records],
               loc='lower center',bbox_to_anchor=(.5,.035),ncol=len(records))
    fig.text(.10,.005,'Balanced synthetic mixture; empty bins omitted, all bin counts retained in source data.',fontsize=7)
    export(fig,out,'confirmation_calibration',{'provenance':provenance,'uncertainty':'reliability points are descriptive; Brier text-bootstrap95 in source data',
           'mixture_prevalence':.5,'bin_rule':'10 fixed equal-width bins; no plot for empty bin','rows':source})


def plot_transfer(records,out,provenance):
    records=score_records(records)
    if not records:return
    fig,axes=plt.subplots(len(records),2,squeeze=False,figsize=(180/25.4,(64*len(records)+25)/25.4))
    fig.subplots_adjust(left=.14,right=.98,bottom=.30,top=.88,wspace=.35,hspace=.60)
    source=[];short=['AG','DB','Emo','TREC*']
    for i,(key,label,color,data,path) in enumerate(records):
        transfers=[r for r in data['transfer']['rows'] if r['transfer_type']=='cross_task' and r['score']=='none_probability']
        assert len(transfers)==12
        matrices={metric:np.full((4,4),np.nan) for metric in ('detection','false_rejection')}
        tids=[t[0] for t in TASKS]
        for row in transfers:
            s,t=tids.index(row['source_task']),tids.index(row['target_task'])
            assert row['source_calibration_subset']['n_text']==80
            for metric in matrices:
                matrices[metric][s,t]=rate(row['target_test'][metric])
                matrices[metric][t,t]=rate(row['local_reference']['target_test'][metric])
        for j,(metric,title,cmap) in enumerate([('detection','Detection D','Blues'),('false_rejection','False rejection F','Oranges')]):
            ax=axes[i,j];array=matrices[metric]
            assert np.isfinite(array).all()
            ax.imshow(array,vmin=0,vmax=1,cmap=cmap,aspect='auto')
            ax.set_xticks(range(4),short);ax.set_yticks(range(4),short);ax.set_xlabel('Target task');ax.set_ylabel('Source task')
            ax.set_title(f'{label}: {title}',pad=7);ax.text(0,1.07,chr(97+i*2+j),transform=ax.transAxes,fontsize=9,fontweight='bold',va='bottom')
            for y in range(4):
                for x in range(4):
                    ax.text(x,y,f'{100*array[y,x]:.1f}',ha='center',va='center',fontsize=7,color='white' if array[y,x]>.62 else '#27313C')
                    if x==y:
                        ax.add_patch(Rectangle((x-.47,y-.47),.94,.94,fill=False,edgecolor='#444444',lw=.8,ls='--'))
        source.append({'model':label,'matrices':{k:v.tolist() for k,v in matrices.items()},'transfer_rows':transfers})
    fig.text(.14,.075,'Values: percent, common scale 0-100. Dashed diagonals use target-local recalibration.',fontsize=7)
    fig.text(.14,.035,'Off-diagonal: source-only 80-pair calibration. *TREC transfers are supplemental.',fontsize=7)
    export(fig,out,'confirmation_transfer',{'provenance':provenance,'anchor':'natural names, k=2, m0/o0, p(NONE)',
          'diagonal':'80-pair target-local reference, separately label-consuming','off_diagonal':'no target calibration labels',
          'all_ordered_task_pairs_retained':True,'rates_conditional_on_frozen_threshold':True,'rows':source})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',default='outputs/paper1-confirmation-results-20261003')
    p.add_argument('--output-dir',default='outputs/laya-missing-option-pilot/latex/figures')
    args=p.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    records=load(Path(args.results).resolve())
    provenance={r[1]:{'metric_path':str(r[4].relative_to(Path(args.results).resolve())),'sha256':hashlib.sha256(r[4].read_bytes()).hexdigest()} for r in records}
    plot_partitions(records,out,provenance);plot_reliability(records,out,provenance);plot_transfer(records,out,provenance)
    print('Exported complete-run-only F2-F4 for',', '.join(r[1] for r in records))
