"""ANOVA/MANOVA/사후검정의 기준 수치·다중성·특이 입력 검증."""
import unittest
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from src.stats.anova import ANOVAProfile, run_anova
from src.stats.manova import MANOVAProfile, run_manova
from src.stats.posthoc import PostHocProfile, run_posthoc


class MultivariateTest(unittest.TestCase):
    def setUp(self):
        rng=np.random.RandomState(7)
        self.samples=[rng.normal(0,1,16),rng.normal(1,2,20),rng.normal(2,1,18)]
        self.y=np.concatenate(self.samples);self.labels=np.repeat(['A','B','C'],[16,20,18])
        self.frame=pd.DataFrame({'response':self.y,'group':self.labels})
        self.frame['second']=.4*self.y+rng.normal(0,1,len(self.y))

    def test_anova_classic_reference(self):
        r=run_anova(self.frame,ANOVAProfile('response',['group']))
        f,p=stats.f_oneway(*self.samples)
        self.assertAlmostEqual(r.statistic,f,places=10);self.assertAlmostEqual(r.p_value,p,places=10)

    def test_anova_welch_reference(self):
        from statsmodels.stats.oneway import anova_oneway
        r=run_anova(self.frame,ANOVAProfile('response',['group'],equal_var=False))
        ref=anova_oneway(self.samples,use_var='unequal')
        self.assertAlmostEqual(r.statistic,ref.statistic,places=10)
        self.assertAlmostEqual(r.p_value,ref.pvalue,places=10)

    def test_two_way_reference_safe_names(self):
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm
        rng=np.random.RandomState(19)
        d=pd.DataFrame({'y':rng.normal(size=48),'a':np.repeat(['A','B'],24),'b':np.tile(np.repeat(['L','H'],12),2)})
        r=run_anova(d.rename(columns={'y':'Y space','a':'A space','b':'B space'}),ANOVAProfile('Y space',['A space','B space'],method='two_way',ss_type=3))
        ref=anova_lm(ols('y ~ C(a, Sum)*C(b, Sum)',d).fit(),typ=3)
        self.assertAlmostEqual(r.f_factor1,ref.loc['C(a, Sum)','F'],places=9)
        self.assertAlmostEqual(r.f_interaction,ref.loc['C(a, Sum):C(b, Sum)','F'],places=9)

    def test_repeated_wrapper(self):
        rng=np.random.RandomState(3)
        d=pd.DataFrame({'id':np.repeat(range(10),3),'visit':np.tile(['A','B','C'],10),'y':rng.normal(size=30)})
        r=run_anova(d,ANOVAProfile('y',['visit'],method='repeated',subject='id'))
        self.assertIn('reject_h0',r.anova_table.columns)
        with self.assertRaises(ValueError):run_anova(d,ANOVAProfile('y',['visit'],method='repeated'))

    def test_manova_reference(self):
        r=run_manova(self.frame,MANOVAProfile(['response','second'],['group']))
        ref=MANOVA.from_formula('response + second ~ C(group, Sum)',self.frame).mv_test()
        np.testing.assert_allclose(r.tables['group']['Pr > F'].values,ref.results['C(group, Sum)']['stat']['Pr > F'].values.astype(float),rtol=1e-8)
        self.assertEqual(len(r.tables['group']),4)
        self.assertNotIn('Intercept',r.tables)

    def test_manova_two_factor_interaction(self):
        rng=np.random.RandomState(8)
        d=pd.DataFrame({'y1':rng.normal(size=60),'y2':rng.normal(size=60),'a':np.repeat(['A','B'],30),'b':np.tile(np.repeat(['L','H'],15),2)})
        r=run_manova(d,MANOVAProfile(['y1','y2'],['a','b'],interaction=True))
        self.assertEqual(set(r.tables),{'a','b','a:b'})

    def test_manova_rejects_singular_and_missing(self):
        d=self.frame.copy();d['second']=2*d.response
        with self.assertRaises(ValueError):run_manova(d,MANOVAProfile(['response','second'],['group']))
        d=self.frame.copy();d.loc[0,'response']=np.nan
        with self.assertRaises(ValueError):run_manova(d,MANOVAProfile(['response','second'],['group']))
        with self.assertRaises(ValueError):run_manova(self.frame,MANOVAProfile(['response'],['group']))

    def test_tukey_reference_and_sign(self):
        r=run_posthoc(self.y,self.labels)
        ref=pairwise_tukeyhsd(self.y,self.labels)
        np.testing.assert_allclose(r.table.p_adjusted,ref.pvalues)
        np.testing.assert_allclose(r.table[['ci_lower','ci_upper']],ref.confint)
        self.assertAlmostEqual(r.table.iloc[0].mean_diff,np.mean(self.samples[1])-np.mean(self.samples[0]))

    def test_games_howell_two_group_matches_welch(self):
        y=np.concatenate(self.samples[:2]);labels=self.labels[:len(y)]
        r=run_posthoc(y,labels,PostHocProfile(method='games_howell',equal_var=False))
        t,p=stats.ttest_ind(*self.samples[:2],equal_var=False)
        self.assertAlmostEqual(r.table.iloc[0].p_adjusted,p,places=6)
        self.assertAlmostEqual(r.table.iloc[0].statistic,abs(t)*np.sqrt(2),places=9)
        row=r.table.iloc[0]
        se=np.sqrt(np.var(self.samples[0],ddof=1)/len(self.samples[0])+np.var(self.samples[1],ddof=1)/len(self.samples[1]))
        half=stats.t.ppf(.975,row['df'])*se
        self.assertAlmostEqual(row.ci_lower,row.mean_diff-half,places=5)

    def test_pairwise_holm_bonferroni_reference(self):
        raw=[stats.ttest_ind(self.samples[a],self.samples[b],equal_var=False).pvalue for a,b in [(0,1),(0,2),(1,2)]]
        for method in ('holm','bonferroni'):
            r=run_posthoc(self.y,self.labels,PostHocProfile(method='pairwise_t',equal_var=False,correction=method))
            np.testing.assert_allclose(r.table.p_adjusted,multipletests(raw,method=method)[1])
            if method=='holm':self.assertTrue(r.table.ci_lower.isna().all())
            else:self.assertTrue(np.isfinite(r.table.ci_lower).all())

    def test_posthoc_rejects_invalid_assumptions(self):
        for p in (PostHocProfile(method='games_howell'),PostHocProfile(method='tukey',equal_var=False),PostHocProfile(alpha=0)):
            with self.assertRaises(ValueError):run_posthoc(self.y,self.labels,p)
        with self.assertRaises(ValueError):run_posthoc([1,1,2,2],['A','A','B','B'])

    def test_anova_configuration_validation(self):
        for p in (ANOVAProfile('response',[]),ANOVAProfile('response',['group'],method='two_way'),ANOVAProfile('response',['group'],alpha=2)):
            with self.assertRaises(ValueError):run_anova(self.frame,p)

if __name__=='__main__':unittest.main()
