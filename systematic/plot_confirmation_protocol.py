"""F1 contract: a frozen paired protocol separates coverage, choice and policy.

Method schematic, no empirical performance claims or mock numerical data.
Actual frozen input counts/hashes supply provenance. Single drawing axes.
Python vector exports, 180x85 mm, editable text >=7pt, no journal-specific claim.
"""
import argparse
import json
from pathlib import Path
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]
from tools.figure_layout_check import require_matplotlib_panel_alignment
from decision_benchmark.artifacts import RESOURCES
p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True)
args=p.parse_args();OUT=Path(args.output_dir);OUT.mkdir(parents=True,exist_ok=True)
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                     'font.size':7,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(180/25.4,85/25.4))
fig.subplots_adjust(left=.015,right=.985,bottom=.025,top=.98)
ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')


def box(x,y,w,h,title,body,color):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.005,rounding_size=0.012',
                               facecolor=color,edgecolor='#718096',lw=.7))
    ax.text(x,y+h/2-.055,title,ha='center',va='top',fontweight='bold',fontsize=7.5)
    ax.text(x,y-.02,body,ha='center',va='center',fontsize=7,linespacing=1.4)


box(.12,.67,.21,.39,'Public data','600 calibration texts\n856 test texts\nKnown-text exclusion\nTREC supplemental','#F0F3F6')
box(.375,.67,.21,.39,'Paired candidates','Matched ordinary k\nPresent / absent pairs\nNames, members, order\nTarget-aligned NONE','#EAF0F6')
box(.63,.67,.21,.39,'Shared model input','Laya / Jev / Qwen7B\nNative decisions\nScores when available\nFailures retained','#EAF0F6')
box(.88,.67,.21,.39,'Distinct outcomes','Correct / wrong choice\nFalse rejection\nDetection / acceptance\nInterface failure','#F0F3F6')
for x1,x2 in ((.233,.262),(.488,.517),(.743,.767)):
    ax.annotate('',xy=(x2,.67),xytext=(x1,.67),arrowprops={'arrowstyle':'->','color':'#526779','lw':1})
box(.34,.235,.36,.28,'Source calibration only','Frozen 80-text transfer subsets\nSelect threshold at empirical F <= 5%\nNo target-test policy selection','#F3EEE4')
box(.77,.235,.36,.28,'Frozen target evaluation','Detection and false rejection\nCalibration and robustness checks\nTarget recalibration labeled separately','#E9F1ED')
ax.annotate('',xy=(.34,.388),xytext=(.34,.461),arrowprops={'arrowstyle':'->','color':'#526779','lw':1})
ax.annotate('',xy=(.588,.235),xytext=(.532,.235),arrowprops={'arrowstyle':'->','color':'#526779','lw':1})
ax.annotate('',xy=(.82,.388),xytext=(.82,.461),arrowprops={'arrowstyle':'->','color':'#526779','lw':1})
ax.text(.5,.025,'Public reference labels; artificial omission, retrieval misses and global OOS are separate.',ha='center',fontsize=7,color='#495663')
fig.canvas.draw()
require_matplotlib_panel_alignment(fig,json_out=str(OUT/'confirmation_protocol.alignment.json'),strict=True,tolerance_pt=1.5)
fig.savefig(OUT/'confirmation_protocol.pdf')
fig.savefig(OUT/'confirmation_protocol.svg')
fig.savefig(OUT/'confirmation_protocol.png',dpi=600)
source={profile:json.loads((RESOURCES/'frozen'/profile/'input_manifest.json').read_text(encoding='utf-8'))
        for profile in ('core','controlled')}
(OUT/'confirmation_protocol.source.json').write_text(json.dumps({'figure_kind':'method schematic, no performance values',
                        'profiles':{k:{'n_cases':v['n_cases'],'n_requests':v['n_requests'],'hashes':v['file_sha256']} for k,v in source.items()}},indent=2)+'\n',encoding='utf-8')
print('Exported actual protocol schematic PDF/SVG/PNG with frozen input provenance')
