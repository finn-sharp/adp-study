"""
================================================================================
test_anova_modules.py - 통계 API 참조 결과, 설계 제약, DTO 및 전체 필드 명세를 검증합니다.
================================================================================
설명:
    - 통계 API 참조 결과, 설계 제약, DTO 및 전체 필드 명세를 검증합니다.
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
import importlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway

from src.stats.tests.parametric.Average.Multivariate.OneWay import OneWayANOVAProfile, OneWayANOVAOutputDTO, OneWayANOVAResult, run_oneway_anova
from src.stats.tests.parametric.Average.Multivariate.TwoWay import TwoWayANOVAProfile, TwoWayANOVAOutputDTO, TwoWayANOVAResult, run_twoway_anova
from src.stats.tests.parametric.Average.Multivariate.MANOVA import MANOVAProfile, run_manova
from src.stats.tests.parametric.Average.Multivariate.PostHoc import PostHocProfile, run_posthoc


class ANOVAModulesTest(unittest.TestCase):
    def setUp(self):
        self.groups = [np.array([1.,2.,3.,4.]),np.array([4.,5.,7.,9.,10.]),np.array([8.,9.,10.,12.])]
        self.y = np.array([1.,2.,3.,4.,3.,4.,5.,7.,5.,6.,7.,9.,10.,11.,12.,14.])
        self.f1 = ['A']*8+['B']*8
        self.f2 = (['Low']*4+['High']*4)*2

    def test_classic_and_welch_reference(self):
        for equal_var in (True,False):
            with self.subTest(equal_var=equal_var):
                result=run_oneway_anova(self.groups,OneWayANOVAProfile(equal_var=equal_var))
                reference=anova_oneway(self.groups,use_var='equal' if equal_var else 'unequal')
                self.assertAlmostEqual(result.statistic,reference.statistic)
                self.assertAlmostEqual(result.p_value,reference.pvalue)
                np.testing.assert_allclose([result.df_between,result.df_within],reference.df)
                self.assertEqual(result.group_ns,[4,5,4])
                self.assertEqual(result.reject_h0,result.p_value<0.05)
                self.assertIn('anova_table',json.loads(result.to_json()))
        result=run_oneway_anova(self.groups)
        self.assertAlmostEqual(result.p_value,stats.f_oneway(*self.groups).pvalue)
        self.assertIs(OneWayANOVAResult,OneWayANOVAOutputDTO)
        self.assertIs(TwoWayANOVAResult,TwoWayANOVAOutputDTO)

    def test_two_way_types_and_interaction(self):
        data=pd.DataFrame({'y':self.y,'A':self.f1,'B':self.f2})
        for interaction in (True,False):
            for typ in (1,2,3):
                with self.subTest(interaction=interaction,typ=typ):
                    formula='y ~ C(A, Sum)'+(' * ' if interaction else ' + ')+'C(B, Sum)'
                    reference=anova_lm(ols(formula,data).fit(),typ=typ)
                    result=run_twoway_anova(self.y,self.f1,self.f2,TwoWayANOVAProfile(interaction,typ),
                                            factor1_name='요인 1',factor2_name='두 번째 요인')
                    self.assertAlmostEqual(result.f_factor1,reference.loc['C(A, Sum)','F'])
                    self.assertAlmostEqual(result.p_factor2,reference.loc['C(B, Sum)','PR(>F)'])
                    if interaction:
                        self.assertAlmostEqual(result.p_interaction,reference.loc['C(A, Sum):C(B, Sum)','PR(>F)'])
                    else:
                        self.assertIsNone(result.p_interaction)
                    self.assertIn('C(요인 1, Sum)',result.anova_table.index)
                    self.assertIn('anova_table',json.loads(result.to_json()))

    def test_manova_all_criteria_reference(self):
        rng=np.random.RandomState(12)
        Y=np.vstack([rng.normal(size=(20,2))+shift for shift in (0,0.5,1.0)])
        labels=['A']*20+['B']*20+['C']*20
        data=pd.DataFrame({'y1':Y[:,0],'y2':Y[:,1],'group':labels})
        reference=MANOVA.from_formula('y1 + y2 ~ C(group)',data).mv_test().results['C(group)']['stat']
        criteria={'pillai':"Pillai's trace",'wilks':"Wilks' lambda",'hotelling':'Hotelling-Lawley trace','roy':"Roy's greatest root"}
        for criterion,row in criteria.items():
            with self.subTest(criterion=criterion):
                result=run_manova(pd.DataFrame(Y,columns=['응답 1','응답 2']),labels,MANOVAProfile(criterion=criterion))
                self.assertAlmostEqual(result.statistic,reference.loc[row,'Value'])
                self.assertAlmostEqual(result.p_value,reference.loc[row,'Pr > F'])
                self.assertAlmostEqual(result.f_value,reference.loc[row,'F Value'])
                self.assertEqual(result.group_ns,[20,20,20])
                self.assertEqual(result.response_names,['응답 1','응답 2'])
                self.assertEqual(result.test_table.shape,(4,6))
                self.assertEqual(result.reject_h0,result.p_value<0.05)
                self.assertIn('test_table',json.loads(result.to_json()))

    def test_tukey_reference_and_pair_direction(self):
        result=run_posthoc(self.groups,group_names=['Z','A','B'])
        labels=np.concatenate([np.repeat(i,len(g)) for i,g in enumerate(self.groups)])
        reference=pairwise_tukeyhsd(np.concatenate(self.groups),labels)
        np.testing.assert_allclose(result.comparisons.p_adjusted,reference.pvalues)
        np.testing.assert_allclose(result.comparisons[['ci_lower','ci_upper']],reference.confint)
        self.assertEqual(result.comparisons.group1.tolist(),['Z','Z','A'])
        self.assertAlmostEqual(result.comparisons.mean_diff.iloc[0],self.groups[1].mean()-self.groups[0].mean())
        self.assertEqual(result.num_comparisons,3)
        self.assertTrue(result.comparisons.p_raw.isna().all())
        json.loads(result.to_json())

    def test_pairwise_t_corrections_and_confidence_intervals(self):
        pairs=[(0,1),(0,2),(1,2)]
        for equal_var in (True,False):
            for method in ('holm','bonferroni'):
                with self.subTest(equal_var=equal_var,method=method):
                    result=run_posthoc(self.groups,PostHocProfile(method,equal_var))
                    raw=[stats.ttest_ind(self.groups[i],self.groups[j],equal_var=equal_var).pvalue for i,j in pairs]
                    reject,corrected,_,_=multipletests(raw,method=method)
                    np.testing.assert_allclose(result.comparisons.p_raw,raw)
                    np.testing.assert_allclose(result.comparisons.p_adjusted,corrected)
                    np.testing.assert_array_equal(result.comparisons.reject_h0,reject)
                    if method=='holm':
                        self.assertTrue(result.comparisons.ci_lower.isna().all())
                    else:
                        np.testing.assert_array_equal(result.comparisons.reject_h0,
                            (result.comparisons.ci_lower>0)|(result.comparisons.ci_upper<0))
                    self.assertIn('comparisons',json.loads(result.to_json()))

    def test_invalid_inputs_and_degenerate_designs(self):
        for alpha in (0,1,float('nan'),True):
            with self.subTest(alpha=alpha),self.assertRaises(ValueError):
                run_oneway_anova(self.groups,OneWayANOVAProfile(alpha=alpha))
        for groups in ([[1],[2,3]],[[1,2],[3,np.nan]],[[1,1],[2,2]]):
            with self.assertRaises(ValueError):
                run_oneway_anova(groups)
        with self.assertRaises(ValueError):
            run_oneway_anova([[1,1],[2,3]],OneWayANOVAProfile(equal_var=False))
        with self.assertRaises(TypeError):
            run_posthoc(self.groups,object())
        with self.assertRaises(ValueError):
            run_posthoc(self.groups,PostHocProfile(equal_var=False))
        with self.assertRaises(ValueError):
            run_posthoc(self.groups,group_names=['same']*3)
        with self.assertRaises(ValueError):
            run_twoway_anova(self.y[:12],self.f1[:12],self.f2[:12])
        with self.assertRaises(ValueError):
            run_twoway_anova(self.y,self.f1,self.f2,TwoWayANOVAProfile(ss_type=True))
        with self.assertRaises(ValueError):
            run_twoway_anova(self.y,self.f1,[None]+self.f2[1:])
        Y=np.column_stack([self.y,self.y*2])
        with self.assertRaises(ValueError):
            run_manova(Y,self.f1)
        with self.assertRaises(ValueError):
            run_manova(Y[:,0],self.f1)
        with self.assertRaises(ValueError):
            run_manova(Y,self.f1,MANOVAProfile(criterion='bad'))

    def test_uniform_complete_specification(self):
        modules=[('OneWay','OneWayANOVA'),('TwoWay','TwoWayANOVA'),('MANOVA','MANOVA'),('PostHoc','PostHoc')]
        for filename,prefix in modules:
            with self.subTest(filename=filename):
                canonical={'OneWay':'src.stats.tests.parametric.Average.anova.one_way','TwoWay':'src.stats.tests.parametric.Average.anova.two_way','PostHoc':'src.stats.tests.parametric.Average.anova.posthoc','MANOVA':'src.stats.tests.parametric.Multivariate.manova'}
                module=importlib.import_module(canonical[filename])
                source=Path(module.__file__).read_text(encoding='utf-8')
                sections=['1. Imports','2. Input Profile','3. Output DTO','4. Input Validation',
                          '5. Individual Model Runners','6. Public Runner','7. Example Execution']
                positions=[source.index(s) for s in sections]
                self.assertEqual(positions,sorted(positions))
                tree=ast.parse(source)
                for label in ('설명:','분석 가정:','작성 정보:','변경 이력 (Revision History):'):
                    self.assertIn(label,ast.get_docstring(tree))
                for name in ('__author__','__email__','__version__','__status__'):
                    self.assertTrue(getattr(module,name))
                for cls in (getattr(module,prefix+'Profile'),getattr(module,prefix+'OutputDTO')):
                    for field in fields(cls):
                        self.assertIn(field.name+' (',cls.__doc__)
                for node in tree.body:
                    if isinstance(node,ast.FunctionDef):
                        for label in ('Args:','Returns:','Raises:'):
                            self.assertIn(label,ast.get_docstring(node))


if __name__=='__main__':
    unittest.main()
