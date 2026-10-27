"""
================================================================================
test_statistics_complete.py - PDF 기법 전수 API·독립 참조값·정확 도수표·설계 경계와 호환성 검증.
================================================================================
설명:
    - PDF 기법 전수 API·독립 참조값·정확 도수표·설계 경계와 호환성 검증.
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
from collections import Counter
from dataclasses import fields
import importlib
from itertools import permutations
import json
from pathlib import Path
import unittest
import numpy as np
from scipy import stats
from src.Recipes import load_profile
from src.Recipes.loader import _REGISTRY
from src.stats.tests.parametric.Average.one_sample import run_one_sample_mean
from src.stats.tests.parametric.Average.two_sample import run_two_sample_mean
from src.stats.tests.parametric.Average.paired import run_paired_t
from src.stats.tests.parametric.Variance.one_sample import run_one_sample_variance
from src.stats.tests.parametric.Proportion.one_sample import run_one_sample_proportion
from src.stats.tests.parametric.Proportion.two_sample import run_two_sample_proportion
from src.stats.tests.nonParametric.Location.sign import run_sign
from src.stats.tests.nonParametric.Location.wilcoxon import run_wilcoxon
from src.stats.tests.nonParametric.Location.mann_whitney import run_mann_whitney
from src.stats.tests.nonParametric.Location.kruskal_wallis import run_kruskal_wallis
from src.stats.tests.nonParametric.Location.mood_median import run_mood_median
from src.stats.tests.nonParametric.Location.friedman import run_friedman
from src.stats.tests.nonParametric.Sequence.runs import run_runs
from src.stats.tests.parametric.Average.anova.repeated_measures import run_repeated_measures_anova
from src.stats.tests.diagnostics.anderson_darling import run_anderson_darling
from src.stats.tests.diagnostics.kolmogorov_smirnov import run_kolmogorov_smirnov
from src.stats.tests.parametric.Multivariate.hotelling import run_hotelling
from src.stats.tests.diagnostics.box_m import run_box_m
from src.stats.tests.Association.odds_ratio import run_odds_ratio
from src.stats.tests.Association.risk_ratio import run_risk_ratio
from src.stats.tests.Association.cmh import run_cmh
from src.stats.tests.nonParametric.Categorical.fisher_exact import run_fisher_exact
from src.stats.tests.nonParametric.Categorical.bowker import run_bowker
from src.stats.tests.nonParametric.Categorical.cochran_q import run_cochran_q
from src.stats.tests.nonParametric.Categorical.homogeneity import run_chi_square_homogeneity
from src.stats.tests.TimeSeries.adf import run_adf

NEW_KEYS=['one-sample-mean','two-sample-mean','paired-t','one-sample-variance','one-sample-proportion','two-sample-proportion','sign','wilcoxon','runs','mann-whitney','kruskal-wallis','mood-median','friedman','repeated-measures-anova','anderson-darling','kolmogorov-smirnov','hotelling','box-m','odds-ratio','risk-ratio','fisher-exact','cmh','bowker','cochran-q','adf','chi-square-homogeneity']
DIST_KEYS=[k for k in _REGISTRY if k.startswith('distribution-')]

class StatisticsCompleteTest(unittest.TestCase):
    def setUp(self):
        self.x=np.array([1.,2.,4.,5.,8.,9.]);self.y=np.array([2.,3.,4.,6.,9.,10.])
        self.rng=np.random.RandomState(8)
        self.repeated=np.array([[1,3,5],[2,1,6],[3,4,2],[4,2,7],[5,6,8]],dtype=float)

    def test_means_three_alternatives_and_intervals(self):
        from statsmodels.stats.weightstats import DescrStatsW, CompareMeans
        for alt in ('two-sided','less','greater'):
            a=run_one_sample_mean(self.x,overrides={'mu_0':2.,'alternative':alt})
            self.assertAlmostEqual(a.p_value,stats.ttest_1samp(self.x,2,alternative=alt).pvalue)
            expected=DescrStatsW(self.x).tconfint_mean(alpha=.05,alternative='larger' if alt=='greater' else 'smaller' if alt=='less' else alt)
            np.testing.assert_allclose(a.confidence_interval,expected)
            p=run_paired_t(self.x,self.y,overrides={'alternative':alt,'diff_0':.2})
            self.assertAlmostEqual(p.p_value,stats.ttest_1samp(self.x-self.y,.2,alternative=alt).pvalue)
            self.assertAlmostEqual(p.mean_difference,np.mean(self.x-self.y))
            for equal in (True,False):
                r=run_two_sample_mean(self.x,self.y,overrides={'equal_var':equal,'alternative':alt,'diff_0':.4})
                self.assertAlmostEqual(r.p_value,stats.ttest_ind(self.x-.4,self.y,equal_var=equal,alternative=alt).pvalue)
                expected=CompareMeans(DescrStatsW(self.x),DescrStatsW(self.y)).tconfint_diff(alpha=.05,alternative='larger' if alt=='greater' else 'smaller' if alt=='less' else alt,usevar='pooled' if equal else 'unequal')
                np.testing.assert_allclose(r.confidence_interval,expected)
        z=run_one_sample_mean(self.x,overrides={'known_var':True,'sigma':2.})
        self.assertAlmostEqual(z.p_value,2*stats.norm.sf(abs(self.x.mean()/(2/np.sqrt(len(self.x))))))
        z=run_two_sample_mean(self.x,self.y,overrides={'known_var':True,'sigma1':2.,'sigma2':3.})
        self.assertTrue(np.isinf(z.df))

    def test_variance_reference_and_direction(self):
        df=len(self.x)-1;v=self.x.var(ddof=1)
        for alt in ('two-sided','less','greater'):
            r=run_one_sample_variance(self.x,overrides={'variance_0':2.,'alternative':alt})
            stat=df*v/2
            pv=stats.chi2.cdf(stat,df) if alt=='less' else stats.chi2.sf(stat,df) if alt=='greater' else 2*min(stats.chi2.cdf(stat,df),stats.chi2.sf(stat,df))
            self.assertAlmostEqual(r.p_value,pv)
            json.loads(r.to_json())
        with self.assertRaises(ValueError):run_one_sample_variance([1,1,1])

    def test_proportions_exact_score_and_difference(self):
        from statsmodels.stats.proportion import proportions_ztest, proportion_confint
        for alt in ('two-sided','less','greater'):
            r=run_one_sample_proportion(3,12,overrides={'p_0':.4,'alternative':alt})
            self.assertAlmostEqual(r.p_value,stats.binomtest(3,12,.4,alternative=alt).pvalue)
            a=run_one_sample_proportion(3,12,overrides={'p_0':.4,'alternative':alt,'method':'normal_score'})
            score=(.25-.4)/np.sqrt(.4*.6/12)
            self.assertAlmostEqual(a.statistic,score)
            b=run_two_sample_proportion(3,12,10,20,overrides={'alternative':alt})
            ref=proportions_ztest([3,10],[12,20],alternative='smaller' if alt=='less' else 'larger' if alt=='greater' else alt)
            self.assertAlmostEqual(b.p_value,ref[1])
            self.assertLessEqual(b.difference_interval[0],b.difference)
            self.assertGreaterEqual(b.difference_interval[1],b.difference)
        self.assertEqual(run_one_sample_proportion(0,10).confidence_intervals[0][0],0)
        with self.assertRaises(ValueError):run_one_sample_proportion(True,10)
        with self.assertRaises(ValueError):run_two_sample_proportion(0,10,0,10)

    def test_sign_wilcoxon_and_missing_pairs(self):
        from src.stats.tests.nonParametric.Location.sign import SignProfile
        r=run_sign([1,2,0,-1,4]);self.assertEqual((r.positive_count,r.negative_count,r.zero_count),(3,1,1))
        self.assertAlmostEqual(r.p_value,stats.binomtest(3,4,.5).pvalue)
        for alt in ('two-sided','less','greater'):
            r=run_wilcoxon([1,2,-3,4,5],overrides={'alternative':alt,'method':'exact'})
            self.assertAlmostEqual(r.p_value,stats.wilcoxon([1,2,-3,4,5],alternative=alt).pvalue)
        for method in ('wilcox','pratt','zsplit'):
            r=run_wilcoxon([0,1,-1,2,3],overrides={'zero_method':method})
            self.assertEqual(r.method_used,'asymptotic');self.assertEqual(r.zero_count,1)
        r=run_wilcoxon([0,0]);self.assertEqual(r.p_value,1)
        with self.assertRaises(ValueError):run_wilcoxon([1,1,2],overrides={'method':'exact'})
        x=np.r_[self.x,np.nan];y=np.r_[self.y,10]
        r=run_paired_t(x,y,overrides={'nan_policy':'omit'})
        self.assertEqual(r.n_effective,len(self.x))
        self.assertAlmostEqual(r.p_value,run_paired_t(self.x,self.y).p_value)
        with self.assertRaises(ValueError):run_sign(self.x,self.y)

    def test_independent_rank_and_median_references(self):
        for method in ('auto','asymptotic'):
            r=run_mann_whitney(self.x,self.y,overrides={'method':method})
            self.assertAlmostEqual(r.p_value,stats.mannwhitneyu(self.x,self.y,method='asymptotic').pvalue)
            self.assertEqual(r.u1+r.u2,len(self.x)*len(self.y))
        with self.assertRaises(ValueError):run_mann_whitney(self.x,self.y,overrides={'method':'exact'})
        groups=[self.x,self.y,self.x+4]
        self.assertAlmostEqual(run_kruskal_wallis(*groups).p_value,stats.kruskal(*groups).pvalue)
        for ties in ('below','above','ignore'):
            r=run_mood_median(*groups,overrides={'ties':ties});ref=stats.median_test(*groups,ties=ties)
            self.assertAlmostEqual(r.p_value,ref[1]);np.testing.assert_array_equal(r.table,ref[3])

    def test_friedman_and_repeated_anova_reference(self):
        from statsmodels.stats.anova import AnovaRM
        import pandas as pd
        r=run_friedman(self.repeated);ref=stats.friedmanchisquare(*self.repeated.T)
        self.assertAlmostEqual(r.p_value,ref.pvalue)
        n,k=self.repeated.shape
        frame=pd.DataFrame({'subject':np.repeat(np.arange(n),k),'condition':np.tile(np.arange(k),n),'value':self.repeated.ravel()})
        ref=AnovaRM(frame,'value','subject',within=['condition']).fit().anova_table.iloc[0]
        r=run_repeated_measures_anova(self.repeated)
        self.assertAlmostEqual(r.statistic,ref['F Value']);self.assertAlmostEqual(r.p_value,ref['Pr > F'])
        gg=run_repeated_measures_anova(self.repeated,overrides={'correction':'greenhouse_geisser'})
        self.assertAlmostEqual(gg.df_num,r.df_num*gg.epsilon)
        self.assertTrue(1/(k-1)<=gg.epsilon<=1)
        np.testing.assert_allclose(run_repeated_measures_anova(self.repeated+np.arange(n)[:,None]).statistic,r.statistic)
        # A large common offset must not make a valid residual variance degenerate.
        np.testing.assert_allclose(run_repeated_measures_anova(self.repeated+1e9).statistic,r.statistic,rtol=1e-6)
        with self.assertRaises(ValueError):run_friedman(np.ones((5,3)))

    def test_runs_manual_sequence(self):
        r=run_runs([0,0,1,1,0,1,0,1],overrides={'correction':False})
        self.assertEqual(r.runs,6);self.assertEqual(r.group_counts,[4,4])
        self.assertAlmostEqual(r.expected_runs,5)
        self.assertAlmostEqual(r.variance_runs,12/7)
        self.assertAlmostEqual(r.statistic,1/np.sqrt(12/7))
        with self.assertRaises(ValueError):run_runs([1,1,1])

    def test_distribution_diagnostics_and_unit_root(self):
        from statsmodels.stats.diagnostic import lilliefors
        from statsmodels.tsa.stattools import adfuller
        x=self.rng.normal(size=80)
        for family in ('expon','logistic','gumbel_l','gumbel_r'):
            sample=getattr(stats,family).rvs(size=80,random_state=np.random.RandomState(42))
            diagnostic=run_anderson_darling(sample,overrides={'distribution':family})
            self.assertIsNone(diagnostic.p_value)
        r=run_anderson_darling(x);ref=stats.anderson(x)
        self.assertIsNone(r.p_value);self.assertEqual(r.reject_h0,ref.statistic>ref.critical_values[2])
        with self.assertRaises(ValueError):run_anderson_darling(x,overrides={'alpha':.03})
        r=run_kolmogorov_smirnov(x);self.assertAlmostEqual(r.p_value,stats.kstest(x,'norm').pvalue)
        r=run_kolmogorov_smirnov(x,overrides={'estimated_parameters':True})
        self.assertAlmostEqual(r.p_value,lilliefors(x)[1])
        r=run_kolmogorov_smirnov(self.rng.exponential(size=80),overrides={'distribution':'expon','estimated_parameters':True})
        self.assertEqual(r.distribution_parameters['loc'],0)
        with self.assertRaises(ValueError):run_kolmogorov_smirnov(x,overrides={'estimated_parameters':True,'parameters':{'loc':0}})
        for auto in ('AIC',None):
            r=run_adf(x,overrides={'max_lag':2,'autolag':auto});ref=adfuller(x,maxlag=2,autolag=auto)
            self.assertAlmostEqual(r.p_value,ref[1]);self.assertEqual(r.used_lag,ref[2])
        with self.assertRaises(ValueError):run_adf(x,overrides={'nan_policy':'omit'})

    def test_hotelling_box_covariance_references_and_invariance(self):
        from statsmodels.stats.multivariate import test_mvmean_2indep, test_cov_oneway
        x=self.rng.normal(size=(20,3));y=self.rng.normal(size=(25,3))+.4
        r=run_hotelling(x,y);ref=test_mvmean_2indep(x,y)
        self.assertAlmostEqual(r.f_value,ref.statistic);self.assertAlmostEqual(r.p_value,ref.pvalue)
        cov=[np.cov(x,rowvar=False),np.cov(y,rowvar=False)]
        b=run_box_m(x,y);ref=test_cov_oneway(cov,[20,25])
        self.assertAlmostEqual(b.statistic,ref.statistic_chi2);self.assertAlmostEqual(b.p_value,ref.pvalue_chi2)
        A=np.array([[2,1,0],[0,3,1],[0,0,2]])
        self.assertAlmostEqual(run_hotelling(x@A,y@A).t_squared,r.t_squared)
        self.assertAlmostEqual(run_box_m(x@A,y@A).m_value,b.m_value)
        with self.assertRaises(ValueError):run_hotelling(np.ones((10,2)),np.ones((12,2)))
        # Individual singular sample covariances are supported when pooled covariance is SPD.
        small=run_hotelling([[0,0],[1,0]],[[0,1],[0,2]])
        self.assertTrue(np.isfinite(small.p_value))

    def test_or_rr_cmh_and_zero_policy(self):
        from statsmodels.stats.contingency_tables import Table2x2, StratifiedTable
        t=np.array([[12,28],[8,42]])
        oracle=Table2x2(t,shift_zeros=False)
        for run,name in ((run_odds_ratio,'oddsratio'),(run_risk_ratio,'riskratio')):
            r=run(t);self.assertAlmostEqual(r.estimate,getattr(oracle,name))
            np.testing.assert_allclose(r.confidence_interval,getattr(oracle,name+'_confint')())
            with self.assertRaises(ValueError):run([[0,10],[2,8]])
            corrected=run([[0,10],[2,8]],overrides={'zero_correction':'haldane'})
            self.assertTrue(corrected.details['zero_correction_applied'])
        tables=np.array([[[11,6],[9,5]],[[8,3],[7,5]],[[7,5],[4,6]]])
        r=run_cmh(tables);oracle=StratifiedTable(tables.transpose(1,2,0),shift_zeros=False)
        self.assertAlmostEqual(r.p_value,oracle.test_null_odds().pvalue)
        self.assertAlmostEqual(r.pooled_odds_ratio,oracle.oddsratio_pooled)

    def test_fisher_rxc_microstate_enumeration_and_monte_carlo(self):
        table=np.array([[2,0,1],[0,3,1]])
        # Independent oracle: enumerate 210 equally likely assignments of column labels to seven slots.
        sequences=set(permutations([0,0,1,1,1,2,2]));counts=Counter()
        for seq in sequences:
            t=np.array([np.bincount(seq[:3],minlength=3),np.bincount(seq[3:],minlength=3)])
            counts[tuple(t.ravel())]+=1
        observed=counts[tuple(table.ravel())]
        expected=sum(v for v in counts.values() if v<=observed)/len(sequences)
        r=run_fisher_exact(table)
        self.assertAlmostEqual(r.p_value,expected);self.assertEqual(r.n_tables,len(counts))
        self.assertAlmostEqual(r.table_probability,observed/len(sequences))
        mc=run_fisher_exact(table,overrides={'method':'monte_carlo','n_resamples':8000})
        self.assertLess(abs(mc.p_value-expected),.03)
        self.assertAlmostEqual(mc.p_value,run_fisher_exact(table,overrides={'method':'monte_carlo','n_resamples':8000}).p_value)
        with self.assertRaises(ValueError):run_fisher_exact(table,overrides={'max_tables':1})
        for alt in ('two-sided','less','greater'):
            r=run_fisher_exact([[2,8],[7,3]],overrides={'alternative':alt})
            self.assertAlmostEqual(r.p_value,stats.fisher_exact([[2,8],[7,3]],alternative=alt)[1])
        enum=run_fisher_exact([[2,8],[7,3]],overrides={'method':'enumeration'})
        self.assertAlmostEqual(enum.p_value,stats.fisher_exact([[2,8],[7,3]])[1])

    def test_categorical_bowker_cochran_homogeneity(self):
        from statsmodels.stats.contingency_tables import SquareTable,cochrans_q
        t=np.array([[12,4,2],[1,15,3],[5,2,10]])
        r=run_bowker(t);ref=SquareTable(t,shift_zeros=False).symmetry()
        self.assertAlmostEqual(r.p_value,ref.pvalue);self.assertEqual(r.df,ref.df)
        binary=np.array([[1,0,1],[1,1,0],[0,1,0],[1,0,0],[0,1,1]])
        r=run_cochran_q(binary);self.assertAlmostEqual(r.p_value,cochrans_q(binary,return_object=True).pvalue)
        r=run_chi_square_homogeneity([[30,20],[10,40],[20,30]])
        self.assertAlmostEqual(r.p_value,stats.chi2_contingency([[30,20],[10,40],[20,30]])[1])
        self.assertIn('범주분포',r.null_hypothesis)
        with self.assertRaises(ValueError):run_cochran_q([[0,2],[1,0]])

    def test_canonical_compatibility_imports(self):
        pairs=[('parametric.Average.Multivariate.OneWay','parametric.Average.anova.one_way','run_oneway_anova'),('parametric.Average.Multivariate.MANOVA','parametric.Multivariate.manova','MANOVAProfile'),('nonParametric.chi_square','nonParametric.Categorical.chi_square','ChiSquareOutputDTO'),('nonParametric.independence','nonParametric.Categorical.independence','run_independence'),('nonParametric.mcnemar','nonParametric.Categorical.mcnemar','McNemarProfile')]
        for old,new,name in pairs:
            self.assertIs(getattr(importlib.import_module('src.stats.tests.'+old),name),getattr(importlib.import_module('src.stats.tests.'+new),name))
        from src.stats.tests.parametric.Average.Univariate import TestProfile,_run_t_test
        old=_run_t_test(self.x,TestProfile(2),self.x.mean(),self.x.std(ddof=1))
        self.assertAlmostEqual(old.p_value,run_one_sample_mean(self.x,overrides={'mu_0':2}).p_value)
        json.loads(old.to_json())

    def test_complete_contract_and_recipe_fields(self):
        self.assertEqual(len(DIST_KEYS),16);self.assertEqual(len(NEW_KEYS),26)
        for key in NEW_KEYS+DIST_KEYS:
            module,profile_name,_,_=_REGISTRY[key];m=importlib.import_module(module)
            source=Path(m.__file__).read_text();tree=ast.parse(source)
            self.assertEqual([source.index(s) for s in ['1. Imports','2. Input Profile','3. Output DTO','4. Input Validation','5. Individual Model Runners','6. Public Runner','7. Example Execution']],sorted(source.index(s) for s in ['1. Imports','2. Input Profile','3. Output DTO','4. Input Validation','5. Individual Model Runners','6. Public Runner','7. Example Execution']))
            for C in (type(load_profile(key)),getattr(m,profile_name.replace('Profile','OutputDTO'))):
                for f in fields(C):self.assertIn(f.name+' (',C.__doc__)
            for node in ast.walk(tree):
                if isinstance(node,ast.FunctionDef):
                    for label in ('Args:','Returns:','Raises:'):self.assertIn(label,ast.get_docstring(node))
