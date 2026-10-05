import unittest
from unittest.mock import patch
import numpy as np

import systematic.confirmation_analysis as scoring


class CoordinatorReviewTests(unittest.TestCase):
    def test_bootstrap_keeps_repeated_source_texts(self):
        s=('source',2,'natural','aligned');t=('target',2,'natural','aligned')
        data={'present':{str(i):[i/20] for i in range(10)},'absent':{str(i):[.9] for i in range(10)}}
        families={s:{'calibration':{'scores':{'none_probability':data}}},
                  t:{'test':{'scores':{'none_probability':data}}}}
        observed=[]
        original=scoring.select_threshold_weighted
        def checked(p,a,limit):
            observed.append((len(p),len(a)))
            return original(p,a,limit)
        with patch.object(scoring,'select_threshold_weighted',side_effect=checked):
            result=scoring.joint_bootstrap(families,s,t,'none_probability',list(data['present']),.05,20,13)
        self.assertEqual(result['replicates_used'],20)
        self.assertEqual(observed,[(10,10)]*20)

    def test_no_variant_pseudoreplication_wilson(self):
        result=scoring.rate_family({'a':[True,False],'b':[False,False]},20,13)
        self.assertIsNone(result['wilson95'])
        self.assertAlmostEqual(result['text_mean'],.25)
        self.assertEqual(result['n_text'],2)

    def test_balanced_coverage_excludes_failed_pair(self):
        p={'request_id':'p'};a={'request_id':'a'}
        variants={'c':[(0,0,p,a)]};cases={'c':{'split':'test'}}
        predictions={'p':{'status':'ok','probabilities':{'x':.8,'none':.2}},
                     'a':{'status':'interface_failure','probabilities':None}}
        result=scoring.coverage_stats(variants,cases,predictions,'test',10,13)
        self.assertEqual(result['n_pairs'],0)
        self.assertIsNone(result['prevalence'])

    def test_vectorized_threshold_matches_scalar_weighting(self):
        rng=np.random.default_rng(3)
        for _ in range(15):
            p={str(i):rng.choice(np.arange(11)/10,size=i%3+1) for i in range(6)}
            a={key:rng.choice(np.arange(11)/10,size=len(value)) for key,value in p.items()}
            grid=np.unique(np.concatenate([*p.values(),*a.values()]))
            feasible=[]
            for threshold in grid:
                f=float(np.mean([float(np.mean(v>threshold)) for v in p.values()]))
                d=float(np.mean([float(np.mean(v>threshold)) for v in a.values()]))
                if f<=.05:
                    feasible.append((d,-f,float(threshold)))
            self.assertEqual(scoring.select_threshold_weighted(p,a),max(feasible)[2])


if __name__=='__main__':
    unittest.main()
