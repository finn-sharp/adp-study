"""
================================================================================
test_association_expansion.py - FP-growth/Eclat 전수 oracle·Apriori 동등성·규칙 수학·한도 검증.
================================================================================
설명:
    - FP-growth/Eclat 전수 oracle·Apriori 동등성·규칙 수학·한도 검증.
분석 가정:
    - 테스트의 고정 입력·합성 자료와 명시한 허용오차를 사용합니다. 시험 데이터의 성능 보장이 아닙니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-27
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""
import ast
from dataclasses import fields
from itertools import combinations
import json
from pathlib import Path
import unittest
import numpy as np
from src.Association.apriori import run_apriori
from src.Association.fp_growth import run_fp_growth,FPGrowthProfile,FPGrowthOutputDTO
from src.Association.eclat import run_eclat,EclatProfile,EclatOutputDTO
from src.Recipes import load_profile


def oracle(transactions,support,length):
    items=sorted(set().union(*map(set,transactions)));counts={}
    for size in range(1,min(length,len(items))+1):
        for pattern in combinations(items,size):
            count=sum(set(pattern)<=set(t) for t in transactions)
            if count/len(transactions)>=support:counts[pattern]=count
    return counts


class AssociationExpansionTest(unittest.TestCase):
    def test_randomized_complete_counts_and_apriori_equivalence(self):
        rng=np.random.RandomState(42)
        for _ in range(50):
            transactions=[[str(i) for i in range(7) if rng.rand()<.45] for j in range(23)]
            support=float(rng.choice([.07,.2,.3,.5,1.]));length=int(rng.randint(1,5))
            expected=oracle(transactions,support,length)
            options={'min_support':support,'max_length':length,'min_confidence':0.,'min_lift':0.}
            reference=run_apriori(transactions,overrides=options)
            for runner in (run_fp_growth,run_eclat):
                result=runner(transactions,overrides=options)
                got={row.items:row.count for row in result.itemsets.itertuples()}
                self.assertEqual(got,expected)
                self.assertEqual(result.rules.to_dict('records'),reference.rules.to_dict('records'))

    def test_duplicate_empty_permutation_threshold_and_rule_math(self):
        transactions=[['a','a','b'],['a','b'],['a'],[]]
        for runner in (run_fp_growth,run_eclat):
            result=runner(transactions,overrides={'min_support':.5,'min_lift':0.})
            self.assertEqual(result.n_transactions,4)
            self.assertEqual(dict(zip(result.itemsets['items'],result.itemsets['count'])),{('a',):3,('b',):2,('a','b'):2})
            rule=result.rules[(result.rules.antecedents==('a',))].iloc[0]
            self.assertAlmostEqual(rule.confidence,2/3);self.assertAlmostEqual(rule.lift,4/3)
            self.assertAlmostEqual(rule.leverage,.125);self.assertAlmostEqual(rule.conviction,1.5)
            reverse=runner(list(reversed(transactions)),overrides={'min_support':.5,'min_lift':0.})
            self.assertEqual(result.to_json(),reverse.to_json())
            result=runner([['a']]*7+[[]]*93,overrides={'min_support':.07})
            self.assertEqual(result.itemsets.iloc[0]['count'],7)
            self.assertEqual(len(runner([[],[]]).itemsets),0)

    def test_limits_and_invalid_inputs_do_not_return_partial_results(self):
        for runner in (run_fp_growth,run_eclat):
            for config in ({'min_support':0},{'min_confidence':-.1},{'min_lift':-1},{'max_length':True}):
                with self.assertRaises(ValueError):runner([['a']],overrides=config)
            for value in ([],[None],[[1]],[[' ']]):
                with self.assertRaises(ValueError):runner(value)
            for config in ({'max_itemsets':1},{'max_rules':1},{'max_rule_candidates':1}):
                with self.assertRaises(ValueError):runner([['a','b','c']],overrides=config)
        with self.assertRaises(ValueError):run_fp_growth([['a','b']],overrides={'max_nodes':1})
        with self.assertRaises(ValueError):run_eclat([['a','b','c']],overrides={'max_intersections':1})
        with self.assertRaises(ValueError):run_eclat([['a']],load_profile('eclat'),overrides={})

    def test_full_profile_dto_function_and_section_contract(self):
        for key,P,O in [('fp-growth',FPGrowthProfile,FPGrowthOutputDTO),('eclat',EclatProfile,EclatOutputDTO)]:
            module=__import__(P.__module__,fromlist=['x']);source=Path(module.__file__).read_text()
            labels=['1. Imports','2. Input Profile','3. Output DTO','4. Input Validation','5. Individual Model Runners','6. Public Runner','7. Example Execution']
            self.assertEqual([source.index(label) for label in labels],sorted(source.index(label) for label in labels))
            self.assertIsInstance(load_profile(key),P)
            for cls in (P,O):
                for field in fields(cls):self.assertIn(field.name+' (',cls.__doc__)
            for node in ast.walk(ast.parse(source)):
                if isinstance(node,ast.FunctionDef):
                    for label in ('Args:','Returns:','Raises:'):self.assertIn(label,ast.get_docstring(node))


if __name__=='__main__':unittest.main()
