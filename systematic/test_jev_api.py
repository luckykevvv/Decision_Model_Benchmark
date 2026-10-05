import unittest
from jev_api import validate_answer
class ContractTests(unittest.TestCase):
    def test_native_confidence_is_distinct(self):
        validate_answer({'type':'choice','choice':'x','probabilities':{'x':.8,'y':.2},'confidence':.278},{'x':'X','y':'Y'})
    def test_preserves_labels_and_rejects_bad_contracts(self):
        criteria={'option_000':'first','none':'absent'}
        valid={'type':'choice','choice':'none','probabilities':{'none':.7,'option_000':.3},'confidence':.118}
        validate_answer(valid,criteria)
        for bad in [{**valid,'choice':'None'},{**valid,'probabilities':{'none':.7,'option_000':.31}},
                    {**valid,'confidence':float('nan')}]:
            with self.assertRaises(ValueError):validate_answer(bad,criteria)
    def test_rounded_tie_is_valid(self):
        validate_answer({'type':'choice','choice':'x','probabilities':{'x':.3333,'y':.3334,'z':.3333},'confidence':0},dict.fromkeys(['x','y','z'],'rubric'))
    def test_declared_two_decimal_rounding(self):
        answer={'type':'choice','choice':'x','probabilities':{'x':.68,'y':.27,'z':.04},'confidence':.52}
        with self.assertRaises(ValueError):validate_answer(answer,dict.fromkeys(['x','y','z'],'rubric'))
        validate_answer(answer,dict.fromkeys(['x','y','z'],'rubric'),rounding_decimals=2)
        with self.assertRaises(ValueError):
            validate_answer({**answer,'probabilities':{'x':.68,'y':.27,'z':0}},dict.fromkeys(['x','y','z'],'rubric'),rounding_decimals=2)
    def test_provider_choice_preserved_for_argmax_audit(self):
        validate_answer({'type':'choice','choice':'b','probabilities':{'a':.37,'b':.36,'c':.26,'d':.01},'confidence':.01},dict.fromkeys(['a','b','c','d'],'rubric'),rounding_decimals=2)
if __name__=='__main__':unittest.main()
