"""
================================================================================
test_probability_distributions.py - 16종 분포의 모수화·확률/밀도·끝점·모멘트·재현·엄격 JSON 검증.
================================================================================
설명:
    - 16종 분포의 모수화·확률/밀도·끝점·모멘트·재현·엄격 JSON 검증.
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
import importlib
import json
import unittest
import numpy as np
from scipy import stats
from src.Recipes.loader import _REGISTRY
from src.stats.distributions.normal import run_normal,NormalProfile
from src.stats.distributions.hypergeometric import run_hypergeometric
from src.stats.distributions.geometric import run_geometric
from src.stats.distributions.negative_binomial import run_negative_binomial
from src.stats.distributions.cauchy import run_cauchy
from src.stats.distributions.student_t import run_student_t
from src.stats.distributions.f import run_f
from src.stats.distributions.exponential import run_exponential
from src.stats.distributions.weibull import run_weibull
from src.stats.distributions.lognormal import run_lognormal

class ProbabilityDistributionsTest(unittest.TestCase):
    def test_all_families_reference_and_runtime_contract(self):
        references={'bernoulli':stats.bernoulli(.5),'binomial':stats.binom(10,.5),'normal':stats.norm(),
                    'hypergeometric':stats.hypergeom(50,10,5),'geometric':stats.geom(.5),
                    'negative_binomial':stats.nbinom(4,.5),'poisson':stats.poisson(3),
                    'exponential':stats.expon(scale=4),'weibull':stats.weibull_min(1.5,scale=1000),
                    'chi_square':stats.chi2(5),'gamma':stats.gamma(2,scale=3),'beta':stats.beta(2,3),
                    'student_t':stats.t(10),'cauchy':stats.cauchy(),'lognormal':stats.lognorm(1,scale=1),'f':stats.f(5,10)}
        for name,ref in references.items():
            key='distribution-'+name.replace('_','-');mod=_REGISTRY[key][0]
            run=getattr(importlib.import_module(mod),'run_'+name)
            r=run(x=[0,1,2],q=[.25,.5,.75],n_samples=5,interval=(0,2))
            np.testing.assert_allclose(r.cdf,ref.cdf([0,1,2]))
            np.testing.assert_allclose(r.survival,ref.sf([0,1,2]))
            np.testing.assert_allclose(r.quantiles,ref.ppf([.25,.5,.75]))
            np.testing.assert_allclose(r.inverse_survival,ref.isf([.25,.5,.75]))
            if r.distribution_kind=='discrete':np.testing.assert_allclose(r.pmf,ref.pmf([0,1,2]));self.assertIsNone(r.pdf)
            else:np.testing.assert_allclose(r.pdf,ref.pdf([0,1,2]));self.assertIsNone(r.pmf)
            np.testing.assert_allclose(r.samples,run(n_samples=5).samples)
            payload=json.loads(r.to_json());self.assertNotIn('distribution',payload)
            self.assertIs(r.to_dict(include_runtime=True)['distribution'],r.distribution)
            if name!='cauchy':self.assertAlmostEqual(r.mean,ref.mean());self.assertAlmostEqual(r.variance,ref.var())

    def test_discrete_closed_interval_inclusive_and_empty(self):
        r=run_hypergeometric(x=[2],interval=(2,2))
        self.assertAlmostEqual(r.interval_probability,r.pmf[0])
        r=run_geometric(interval=(1.2,1.8));self.assertEqual(r.interval_probability,0)
        self.assertAlmostEqual(run_geometric(interval=(1,2)).interval_probability,.75)
        r=run_geometric(interval=(-np.inf,np.inf));self.assertEqual(r.interval_probability,1)

    def test_pdf_counting_examples(self):
        self.assertAlmostEqual(run_geometric(x=4,overrides={'p':.2}).pmf[0],.2*.8**3)
        failures=run_geometric(x=3,overrides={'p':.2,'counting':'failures'})
        self.assertAlmostEqual(failures.pmf[0],.2*.8**3)
        total=run_negative_binomial(x=10,overrides={'r':4,'p':.3,'counting':'trials'})
        self.assertAlmostEqual(total.pmf[0],stats.nbinom.pmf(6,4,.3));self.assertAlmostEqual(total.mean,4/.3)
        self.assertAlmostEqual(run_negative_binomial(overrides={'r':4,'p':.3}).mean,4*.7/.3)
        self.assertAlmostEqual(run_exponential(x=2).cdf[0],1-np.exp(-.25*2))
        r=run_weibull(overrides={'shape':1.5,'scale':1000})
        from scipy.special import gamma
        self.assertAlmostEqual(r.mean,1000*gamma(1+1/1.5))
        r=run_lognormal(overrides={'mu':2.,'sigma':.4})
        self.assertAlmostEqual(r.mean,np.exp(2+.4**2/2))

    def test_continuous_tail_and_density_are_not_probability(self):
        r=run_normal(x=0,interval=(0,0));self.assertGreater(r.pdf[0],0);self.assertEqual(r.interval_probability,0)
        r=run_normal(interval=(8,9));self.assertGreater(r.interval_probability,0)
        self.assertAlmostEqual(r.interval_probability,stats.norm.sf(8)-stats.norm.sf(9),delta=1e-25)
        self.assertEqual(run_normal(interval=(-np.inf,np.inf)).interval_probability,1)

    def test_undefined_moments_and_degenerate_cases(self):
        r=run_cauchy();self.assertIsNone(r.mean);self.assertIsNone(r.variance)
        r=run_student_t(overrides={'df':1});self.assertFalse(r.moment_defined['mean'])
        r=run_student_t(overrides={'df':2});self.assertTrue(r.moment_defined['mean']);self.assertFalse(r.moment_defined['variance'])
        r=run_f(overrides={'df_den':2});self.assertIsNone(r.mean)
        r=run_hypergeometric(overrides={'N':1,'K':1,'n':1});self.assertEqual(r.mean,1);self.assertEqual(r.variance,0)
        self.assertEqual(run_geometric(overrides={'p':1}).variance,0)

    def test_all_32_pdf_examples_with_explicit_corrections(self):
        from examples.statistics_pdf_examples import run_pdf_examples
        records=run_pdf_examples()
        self.assertEqual(len(records),32);self.assertEqual(len({r['id'] for r in records}),32)
        json.dumps(records,allow_nan=False)
        self.assertEqual(next(r for r in records if r['id']=='3a')['continuous_point_probability'],0)
        self.assertTrue(next(r for r in records if r['id']=='12a')['assumptions'])

    def test_input_errors_recipe_and_profile_precedence(self):
        for kwargs in ({'overrides':{'sigma':0}},{'q':1.1},{'x':np.nan},{'interval':(2,1)},{'n_samples':True}):
            with self.assertRaises(ValueError):run_normal(**kwargs)
        with self.assertRaises(ValueError):run_hypergeometric(overrides={'N':5,'K':6})
        with self.assertRaises(ValueError):run_geometric(overrides={'p':0})
        with self.assertRaises(ValueError):run_normal(NormalProfile(),overrides={'mu':2})
        r=run_normal(overrides={'mu':5,'sigma':2});self.assertEqual(r.mean,5);self.assertEqual(r.variance,4)

if __name__=='__main__':unittest.main()
