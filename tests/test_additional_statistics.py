"""검정 참조값·정확 검정·방향성·도수 조건과 모듈 명세 검증."""
import ast
from dataclasses import fields
import importlib
import json
from pathlib import Path
import unittest
import numpy as np
from scipy import stats
from src.Recipes import load_profile
from src.stats.tests.nonParametric.chi_square import run_chi_square
from src.stats.tests.nonParametric.independence import run_independence
from src.stats.tests.nonParametric.mcnemar import run_mcnemar
from src.stats.tests.parametric.Variance.f_test import run_f_test
from src.stats.tests.diagnostics.normality import run_normality
from src.stats.tests.diagnostics.homogeneity import run_homogeneity

KEYS = ['chi-square','independence','mcnemar','f-test','normality','homogeneity',
        'tf-classification','tf-regression','tf-rnn','tf-time-series']

class AdditionalStatisticsTest(unittest.TestCase):
    def test_goodness_expected_and_ddof(self):
        obs=[12,25,19,24];exp=[20,20,20,20]
        r=run_chi_square(obs,exp,overrides={'ddof':1})
        ref=stats.chisquare(obs,exp,ddof=1)
        self.assertAlmostEqual(r.statistic,ref.statistic);self.assertAlmostEqual(r.p_value,ref.pvalue)
        self.assertEqual(r.details['df'],2)
        self.assertTrue(run_chi_square([1,2,1]).warnings)
        for expected in ([1,2,3,4],[0,20,30,30]):
            with self.assertRaises(ValueError):run_chi_square(obs,expected)
        with self.assertRaises(ValueError):run_chi_square([1.5,3])
        with self.assertRaises(ValueError):run_chi_square(obs,overrides={'ddof':3})

    def test_independence_yates_and_cramers_v(self):
        table=[[30,12],[8,40]]
        for correction in (False,True):
            r=run_independence(table,overrides={'correction':correction})
            ref=stats.chi2_contingency(table,correction=correction)
            self.assertAlmostEqual(r.statistic,ref[0]);self.assertAlmostEqual(r.p_value,ref[1])
            np.testing.assert_allclose(r.details['expected'],ref[3])
            self.assertAlmostEqual(r.details['cramers_v'],np.sqrt(stats.chi2_contingency(table,correction=False)[0]/90))
        multi=run_independence([[12,14],[20,11],[7,10]])
        self.assertEqual(multi.details['df'],2)
        self.assertAlmostEqual(multi.p_value,stats.chi2_contingency([[12,14],[20,11],[7,10]])[1])
        with self.assertRaises(ValueError):run_independence([[0,0],[2,4]])

    def test_mcnemar_exact_and_approximate(self):
        from statsmodels.stats.contingency_tables import mcnemar
        table=[[10,9],[2,30]]
        for exact in (False,True):
            for correction in (False,True):
                r=run_mcnemar(table,overrides={'exact':exact,'correction':correction})
                ref=mcnemar(table,exact=exact,correction=correction)
                self.assertAlmostEqual(r.statistic,ref.statistic);self.assertAlmostEqual(r.p_value,ref.pvalue)
        self.assertEqual(run_mcnemar([[5,0],[0,4]]).p_value,1)
        with self.assertRaises(ValueError):run_mcnemar([[5,0],[0,4]],overrides={'exact':False})
        with self.assertRaises(ValueError):run_mcnemar(np.ones((3,3)))

    def test_f_direction_and_variance_ratio(self):
        x=np.array([1,2,3,9,10]);y=np.array([2,3,4,5,7,9])
        s=np.var(x,ddof=1)/np.var(y,ddof=1)/2
        for alt in ('less','greater','two-sided'):
            r=run_f_test(x,y,overrides={'ratio_0':2,'alternative':alt})
            expected=stats.f.cdf(s,4,5) if alt=='less' else stats.f.sf(s,4,5) if alt=='greater' else 2*min(stats.f.cdf(s,4,5),stats.f.sf(s,4,5))
            self.assertAlmostEqual(r.p_value,expected)
        self.assertAlmostEqual(run_f_test(x,y).p_value,run_f_test(y,x).p_value)
        with self.assertRaises(ValueError):run_f_test([1,1],[2,3])

    def test_normality_and_variance_diagnostics(self):
        x=np.random.RandomState(9).normal(size=50);y=x*2+3
        for name,fun in [('shapiro',stats.shapiro),('dagostino',stats.normaltest),('jarque_bera',stats.jarque_bera)]:
            r=run_normality(x,overrides={'method':name});ref=fun(x)
            self.assertAlmostEqual(r.p_value,ref.pvalue)
            self.assertEqual(r.reject_h0,r.p_value<r.alpha)
            json.loads(r.to_json())
        for name in ('levene','bartlett','fligner'):
            r=run_homogeneity(x,y,overrides={'method':name})
            self.assertAlmostEqual(r.p_value,getattr(stats,name)(x,y).pvalue)
        with self.assertRaises(ValueError):run_normality([1,2,3],overrides={'method':'dagostino'})
        with self.assertRaises(ValueError):run_normality([3]*20)
        with self.assertRaises(ValueError):run_homogeneity([1,1],[2,3])
        with self.assertRaises(ValueError):run_f_test(x,y,overrides={'alpha':True})

    def test_profiles_dtos_and_seven_sections(self):
        from src.Recipes.loader import _REGISTRY
        for key in KEYS:
            mod,cls,default,tasks=_REGISTRY[key]
            module=importlib.import_module(mod)
            tree=ast.parse(Path(module.__file__).read_text())
            source=Path(module.__file__).read_text()
            sections=['1. Imports','2. Input Profile','3. Output DTO','4. Input Validation','5. Individual Model Runners','6. Public Runner','7. Example Execution']
            positions=[source.index(s) for s in sections];self.assertEqual(positions,sorted(positions))
            for task in tasks:
                p=load_profile(key,task=task)
                for f in fields(p):self.assertIn(f.name+' (',p.__doc__)
            dto=getattr(module,cls.replace('Profile','OutputDTO'))
            for f in fields(dto):self.assertIn(f.name+' (',dto.__doc__)
            for node in ast.walk(tree):
                if isinstance(node,ast.FunctionDef):
                    for label in ('Args:','Returns:','Raises:'):self.assertIn(label,ast.get_docstring(node))

if __name__=='__main__':unittest.main()
