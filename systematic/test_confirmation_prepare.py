import unittest
from systematic.prepare_confirmation import audit,build_requests
from systematic.run_confirmation_llm import parse_choice


class PreparationTests(unittest.TestCase):
    def test_controlled_order_preserves_members_and_member_swaps_preserve_slots(self):
        case={'case_id':'trec:calibration:0','task':'trec','split':'calibration','label':'NUM','label_id':5,'text':'How many?'}
        requests=build_requests([case],'robustness-controlled')
        audit([case],requests)
        def chosen(k,presence,m,o):
            return next(r for r in requests if r['candidate_size']==k and r['presence']==presence and r['name_mode']=='natural' and r['with_none'] and r['none_mode']=='aligned' and r['member_seed']==m and r['order_seed']==o)
        for k in (2,4,5):
            for m in range(5):
                for presence in ('present','absent'):
                    self.assertEqual(set(chosen(k,presence,m,0)['criteria']),set(chosen(k,presence,m,1)['criteria']))
            for o in (0,1):
                base=chosen(k,'absent',0,o)
                for m in range(1,5):
                    other=chosen(k,'absent',m,o)
                    a,b=list(base['criteria']),list(other['criteria'])
                    self.assertLessEqual(sum(x!=y for x,y in zip(a,b)),1)
                    for key in set(a)&set(b):
                        self.assertEqual(a.index(key),b.index(key))
                    self.assertEqual(chosen(k,'present',0,o)['criteria'],chosen(k,'present',m,o)['criteria'])

    def test_determinism_paired_position_and_wording(self):
        cases=[{'case_id':'trec:calibration:0','task':'trec','split':'calibration','label':'NUM','label_id':5,'text':'How many?'}]
        core=build_requests(cases,'core')
        self.assertEqual(core,build_requests(cases,'core'))
        self.assertEqual(audit(cases,core)['n_requests'],28)
        robust=build_requests(cases,'robustness')
        self.assertEqual(audit(cases,robust)['n_requests'],262)
        for request in robust:
            if request['none_mode']=='legacy':
                self.assertTrue(request['with_none'])
                self.assertEqual(request['member_seed'],0)
                self.assertEqual(request['order_seed'],0)
        bad=[dict(r) for r in core]
        present=next(r for r in bad if r['presence']=='present' and r['with_none'])
        present['criteria']=dict(reversed(list(present['criteria'].items())))
        with self.assertRaises(AssertionError):
            audit(cases,bad)

    def test_exact_llm_output_not_repaired(self):
        self.assertEqual(parse_choice('"NUM"',{'NUM':'n','none':'x'}),'NUM')
        self.assertEqual(parse_choice(' none\n',{'NUM':'n','none':'x'}),'none')
        for value in ('The answer is NUM','{"choice":"NUM"}','num','unknown'):
            with self.assertRaises(ValueError):
                parse_choice(value,{'NUM':'n','none':'x'})


if __name__=='__main__':
    unittest.main()
