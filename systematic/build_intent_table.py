import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
from decision_benchmark.artifacts import RESOURCES
p=argparse.ArgumentParser();p.add_argument('--results-dir',default=str(RESOURCES/'reference'));p.add_argument('--output-dir',required=True)
args=p.parse_args();RESULTS=Path(args.results_dir);OUT=Path(args.output_dir);(OUT/'tables').mkdir(parents=True,exist_ok=True)
models=[('laya','Laya'),('jev','Jev'),('qwen','Qwen7B'),('bm25','BM25 scope gate*')]


def value(row):
    return '--' if row['value'] is None else f"{100*row['value']:.1f}"


lines=[r'\begin{longtable}{llrrrrr}',r'\caption{T6: public CLINC150 extension, native with-NONE decisions. Artificial omission uses 300 matched in-scope test queries at $k=5$. Natural retrieval uses 300 in-scope and 100 global-OOS test queries; retrieved sets are not repaired. Coverage is the retriever\textquotesingle s reference-label hit rate on in-scope queries. $D_{\mathrm{miss}}$ and $F_{\mathrm{cov}}$ have the conditional missing/covered denominators; $D_{\mathrm{OOS}}$ uses the 100 global-OOS queries. Dashes mark populations that do not apply. Marginal 2,000-text-bootstrap intervals and failures are retained in source artifacts. *The lexical reference consumes 14,700 labeled training utterances and supports only the fixed 150-intent ontology; it is not a generic open-class model.}\label{tab:confirmation_intent}\\',
 r'\toprule Track / $k$ & Model & Coverage & $n_{\mathrm{cov}}/n_{\mathrm{miss}}$ & $D_{\mathrm{miss}}$ & $F_{\mathrm{cov}}$ & $D_{\mathrm{OOS}}$\\\midrule\endfirsthead',
 r'\toprule Track / $k$ & Model & Coverage & $n_{\mathrm{cov}}/n_{\mathrm{miss}}$ & $D_{\mathrm{miss}}$ & $F_{\mathrm{cov}}$ & $D_{\mathrm{OOS}}$\\\midrule\endhead']
sources={}
for track,k in [('artificial',5),('natural_retrieval',2),('natural_retrieval',5),('natural_retrieval',10)]:
    for key,label in models:
        path=RESULTS/f'intent-{key}.json'
        if not path.exists():continue
        data=json.loads(path.read_text(encoding='utf-8'));assert data['n_predictions']==6900
        row=data['summaries'][f'{track}|k{k}|test']
        coverage=value(row['retrieval_coverage']) if row['retrieval_coverage'] else '--'
        track_label=('Artificial' if track=='artificial' else 'Retrieved')+f' / {k}'
        lines.append(' & '.join([track_label,label,coverage,f"{row['n_covered']}/{row['n_missing_inscope']}",
                     value(row['missing_inscope']['reject']),value(row['covered']['reject']),value(row['global_oos']['reject'])])+r'\\')
        sources[key]=data
lines += [r'\bottomrule',r'\end{longtable}']
(OUT/'tables/confirmation_intent.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(OUT/'intent_table_source.json').write_text(json.dumps(sources,indent=2)+'\n',encoding='utf-8')
print('Built T6 for',', '.join(sources))
