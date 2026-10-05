"""Closed-scope labeled-corpus reference; not a generic open-class model."""
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import re

from systematic.confirmation_runtime import ROOT,prepare_run,digest


def norm(text):return ' '.join(text.casefold().split())
def tokens(text):return re.findall(r'[a-z0-9]+',text.casefold())


def run(source=None, output=None, cache=None):
    source=Path(source) if source is not None else ROOT/'outputs/paper1-intent-extension-20261003/inputs'
    original=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    cache=Path(cache) if cache is not None else ROOT/'cache/datasets'
    data_file=cache/'clinc150'/original['source']['revision']/'data_full.json'
    assert digest(data_file)==original['source']['file_sha256']
    raw=json.loads(data_file.read_text(encoding='utf-8'))
    model={'provider':'closed-scope-labeled-corpus-reference','id':'BM25-scope-gate',
           'supported_training_intents':150,'labeled_corpus_utterances':14700,
           'generic_open_class_support':False,'native_probabilities':False}
    output=Path(output) if output is not None else ROOT/'outputs/paper1-intent-extension-20261003/bm25'
    manifest,cases,requests,existing=prepare_run(source,output,model)
    assert not existing,'Reference baseline should run once'
    known_queries={norm(c['text']) for c in cases.values()}
    documents=[(text,label) for text,label in raw['train'] if norm(text) not in known_queries]
    assert len(documents)==14700
    labels=sorted({label for text,label in documents});assert len(labels)==150
    postings=defaultdict(list);lengths=[]
    for index,(text,label) in enumerate(documents):
        counts=Counter(tokens(text));lengths.append(sum(counts.values()))
        for word,tf in counts.items():postings[word].append((index,tf))
    avg=sum(lengths)/len(lengths)
    by_case={}
    for identity,case in cases.items():
        document_scores=defaultdict(float)
        for word in set(tokens(case['text'])):
            entries=postings.get(word,[])
            if not entries:continue
            idf=math.log(1+(len(documents)-len(entries)+.5)/(len(entries)+.5))
            for index,tf in entries:
                document_scores[index]+=idf*tf*2.2/(tf+1.2*(.25+.75*lengths[index]/avg))
        scores={label:0. for label in labels}
        for index,value in document_scores.items():scores[documents[index][1]]=max(scores[documents[index][1]],value)
        ranked=sorted(labels,key=lambda label:(-scores[label],label));by_case[identity]=(scores,ranked)
    with (output/'predictions.jsonl').open('w',encoding='utf-8') as stream:
        for r in requests.values():
            scores,ranked=by_case[r['case_id']];ordinary=[key for key in r['criteria'] if key!='none']
            assert all(key in labels for key in ordinary),'Known taxonomy only'
            if r['track']=='natural_retrieval':assert ordinary==ranked[:r['candidate_size']]
            winner=ranked[0]
            if r['with_none'] and winner not in ordinary:choice='none'
            else:choice=max(ordinary,key=lambda key:scores[key])
            stream.write(json.dumps({'request_id':r['request_id'],'status':'ok','choice':choice,'probabilities':None,
                                     'full_scope_prediction':winner,'max_corpus_similarity':scores[winner]})+'\n')
    (output/'runtime.json').write_text(json.dumps({'model':model,'method':'BM25 maximum document score per known intent; reject if full-scope winner not offered and NONE exists',
                           'limitations':'Uses labeled training corpus and fixed intent identifiers; not an open-type comparator or calibrated OOS detector.',
                           'query_corpus_overlap':0,'code_sha256':digest(Path(__file__))},indent=2)+'\n',encoding='utf-8')
    print('Completed fixed-scope reference baseline',len(requests),'requests')


if __name__=='__main__':run()
