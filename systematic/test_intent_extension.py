import json
from pathlib import Path
import tempfile
import unittest

from systematic.analyze_intent_extension import analyze
from systematic.confirmation_runtime import digest


class IntentMetricTests(unittest.TestCase):
    def test_conditional_populations_and_no_test_threshold_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cases=[];requests=[];predictions=[]
            for split in ('calibration','test'):
                for kind,label,covered in [('covered','x',True),('miss','y',False),('oos','oos',False)]:
                    identity=split+'-'+kind
                    cases.append({'case_id':identity,'split':split,'label':label,'global_oos':kind=='oos'})
                    for with_none in (False,True):
                        criteria={'x':'x topic','z':'z topic'}
                        if with_none:criteria['none']='No offered intent fits'
                        rid=identity+str(with_none)
                        requests.append({'request_id':rid,'case_id':identity,'track':'natural_retrieval','presence':'retrieved',
                                         'split':split,'candidate_size':2,'with_none':with_none,'criteria':criteria,
                                         'reference_covered':covered,'global_oos':kind=='oos','gold_option':'x' if covered else None})
                        pnone=.1 if covered else .8
                        probabilities={'x':1-pnone-.1,'z':.1,'none':pnone} if with_none else {'x':.8 if covered else .6,'z':.2 if covered else .4}
                        predictions.append({'request_id':rid,'status':'ok','choice':'none' if with_none and not covered else 'x','probabilities':probabilities})
            def save():
                for name,rows in [('cases.jsonl',cases),('requests.jsonl',requests),('predictions.jsonl',predictions)]:
                    (root/name).write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
                (root/'manifest.json').write_text(json.dumps({'n_requests':len(requests),'model':{'id':'toy'},'source':{},'retriever':{},
                     'file_sha256':{name:digest(root/name) for name in ('cases.jsonl','requests.jsonl')}}))
            save();original=analyze(root)
            test=original['summaries']['natural_retrieval|k2|test']
            self.assertEqual(test['n_queries'],3)
            self.assertEqual((test['n_covered'],test['n_missing_inscope'],test['n_global_oos']),(1,1,1))
            self.assertEqual(test['retrieval_coverage']['value'],.5)
            self.assertEqual(test['covered']['reject']['value'],0)
            self.assertEqual(test['missing_inscope']['reject']['value'],1)
            self.assertEqual(test['global_oos']['reject']['value'],1)
            for r,p in zip(requests,predictions):
                if r['split']=='test' and r['with_none']:
                    p['probabilities']={'x':.49,'z':.5,'none':.01}
            save();changed=analyze(root)
            for key,value in original['score_policies'].items():
                self.assertEqual(value['threshold'],changed['score_policies'][key]['threshold'])


if __name__=='__main__':unittest.main()
