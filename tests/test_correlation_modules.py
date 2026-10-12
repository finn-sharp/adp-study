"""상관계수 참조값·행렬 다중검정·범주 부호·변환·명세 검증."""
import ast,importlib,json,inspect
from dataclasses import fields
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from src.Recipes import load_profile
from src.stats.correlation.numeric.pearson import run_pearson,run_pearson_matrix
from src.stats.correlation.numeric.spearman import run_spearman,run_spearman_matrix
from src.stats.correlation.numeric.kendall import run_kendall,run_kendall_matrix
from src.stats.correlation.categorical.cramers_v import run_cramers_v
from src.stats.correlation.categorical.phi import run_phi
from src.stats.correlation.matrix_transform import run_correlation_matrix_transform
from src.stats.correlation._common import adjusted_pvalues


class CorrelationModulesTest(unittest.TestCase):
    def setUp(self):
        self.x=np.array([1.,3.,2.,5.,8.,9.,12.,6.]);self.y=np.array([2.,6.,3.,7.,8.,12.,10.,9.])

    def test_numeric_reference_directions_and_confidence_interval(self):
        for runner,reference in [(run_pearson,stats.pearsonr),(run_spearman,stats.spearmanr),(run_kendall,stats.kendalltau)]:
            result=runner(self.x,self.y);r,p=reference(self.x,self.y)
            self.assertAlmostEqual(result.coefficient,r);self.assertAlmostEqual(result.p_value,p)
            for alternative in ('less','greater'):
                native='alternative' in inspect.signature(reference).parameters
                settings={'alternative':alternative}
                if runner is run_kendall and not native:
                    # Exact discrete tails cannot use the complement of two-sided p.
                    with self.assertRaises(ValueError):runner(self.x,self.y,overrides=settings)
                    settings['method']='asymptotic'
                    r,p=reference(self.x,self.y,method='asymptotic')
                result=runner(self.x,self.y,overrides=settings)
                expected=reference(self.x,self.y,alternative=alternative)[1] if native else p/2 if (r>0)==(alternative=='greater') else 1-p/2
                self.assertAlmostEqual(result.p_value,expected)
            json.loads(result.to_json())
            with self.assertRaises(ValueError):runner(np.ones(8),self.y)
        result=run_pearson(self.x,self.y)
        z=np.arctanh(result.coefficient);width=stats.norm.ppf(.975)/np.sqrt(len(self.x)-3)
        np.testing.assert_allclose(result.confidence_interval,np.tanh([z-width,z+width]))
        self.assertIsNone(run_pearson([1,2,3],[2,4,3]).confidence_interval)

    def test_pairwise_missing_counts_and_multiple_comparison_adjustments(self):
        X=pd.DataFrame({'a':self.x,'b':self.y,'c':self.x[::-1]});X.loc[0,'b']=np.nan;X.loc[1,'c']=np.nan
        for runner in (run_pearson_matrix,run_spearman_matrix,run_kendall_matrix):
            result=runner(X,overrides={'nan_policy':'omit'})
            self.assertEqual(result.n_matrix.loc['b','c'],6);self.assertEqual(result.n_matrix.loc['a','b'],7)
            np.testing.assert_allclose(result.correlation_matrix,result.correlation_matrix.T)
            self.assertTrue(np.isnan(np.diag(result.p_value_matrix)).all())
            self.assertFalse(np.diag(result.significance_matrix).any())
            p=[result.p_value_matrix.iloc[i,j] for i,j in [(0,1),(0,2),(1,2)]]
            q=multipletests(p,method='holm')[1]
            np.testing.assert_allclose(q,[result.adjusted_p_value_matrix.iloc[i,j] for i,j in [(0,1),(0,2),(1,2)]])
        for method in ('bonferroni','holm','fdr_bh'):
            values=[.01,.05,.002,.4];np.testing.assert_allclose(adjusted_pvalues(values,method),multipletests(values,method=method)[1])
        undefined=run_pearson_matrix(np.column_stack([self.x,np.ones(8)]))
        self.assertTrue(np.isnan(undefined.correlation_matrix.iloc[0,1]));self.assertTrue(undefined.warnings)

    def test_spearman_permutation_reproducible_and_kendall_ties(self):
        kwargs={'p_value_method':'permutation','n_resamples':999,'random_state':12}
        first=run_spearman(self.x,self.y,overrides=kwargs);second=run_spearman(self.x,self.y,overrides=kwargs)
        self.assertEqual(first.to_json(),second.to_json());self.assertGreater(first.p_value,0)
        with self.assertRaises(ValueError):run_kendall([1,1,2],[3,2,1],overrides={'method':'exact'})
        with self.assertRaises(ValueError):run_pearson(pd.Series(self.x,index=range(8)),pd.Series(self.y,index=range(1,9)))

    def test_categorical_cramers_bias_and_phi_orientation(self):
        x=np.array(['a']*30+['b']*30+['c']*30);y=np.array(['x']*20+['y']*10+['x']*10+['y']*20+['x']*15+['y']*15)
        result=run_cramers_v(x,y,overrides={'bias_correction':False});table=pd.crosstab(x,y).to_numpy();chi,p,df,expected=stats.chi2_contingency(table,correction=False)
        self.assertAlmostEqual(result.coefficient,np.sqrt(chi/len(x)/min(table.shape[0]-1,table.shape[1]-1)))
        self.assertAlmostEqual(result.p_value,p)
        self.assertLessEqual(run_cramers_v(x,y).coefficient,result.coefficient)
        binary_x=np.array([0]*20+[1]*20);binary_y=np.array([0]*15+[1]*5+[0]*5+[1]*15)
        phi=run_phi(binary_x,binary_y);self.assertAlmostEqual(phi.coefficient,np.corrcoef(binary_x,binary_y)[0,1])
        inverted=run_phi(binary_x,binary_y,overrides={'positive_x':0});self.assertAlmostEqual(inverted.coefficient,-phi.coefficient)
        exact=run_phi(binary_x,binary_y,overrides={'p_value_method':'fisher'});self.assertAlmostEqual(exact.p_value,stats.fisher_exact(phi.observed.to_numpy())[1])
        with self.assertRaises(ValueError):run_phi(['a','a','b','b'],[0,1,0,1])
        with self.assertRaises(ValueError):run_cramers_v(['a']*5,[0,1,1,0,1])

    def test_matrix_tidy_triangle_distance_missing_and_label_validation(self):
        matrix=pd.DataFrame([[1,.8,-.4],[.8,1,.2],[-.4,.2,1]],index=list('abc'),columns=list('abc'))
        tidy=run_correlation_matrix_transform(matrix);self.assertEqual(len(tidy.pairs),3);self.assertEqual(tidy.pairs.iloc[0].abs_coefficient,.8)
        upper=run_correlation_matrix_transform(matrix,overrides={'mode':'upper'});self.assertTrue(np.isnan(upper.transformed_matrix.iloc[1,0]))
        signed=run_correlation_matrix_transform(matrix,overrides={'mode':'distance'});self.assertAlmostEqual(signed.transformed_matrix.loc['a','c'],np.sqrt(2.8));self.assertTrue((np.diag(signed.transformed_matrix)==0).all())
        absolute=run_correlation_matrix_transform(matrix,overrides={'mode':'distance','distance_kind':'absolute'});self.assertAlmostEqual(absolute.transformed_matrix.loc['a','c'],np.sqrt(1.2))
        incomplete=matrix.copy();incomplete.loc['a','b']=incomplete.loc['b','a']=np.nan
        with self.assertRaises(ValueError):run_correlation_matrix_transform(incomplete)
        result=run_correlation_matrix_transform(incomplete,overrides={'allow_missing':True});self.assertEqual(len(result.pairs),2);self.assertTrue(result.warnings)
        with self.assertRaises(ValueError):run_correlation_matrix_transform(matrix.iloc[::-1])
        with self.assertRaises(ValueError):run_correlation_matrix_transform(matrix,overrides={'max_p_value':.05})
        json.loads(signed.to_json())

    def test_profile_dto_and_function_contracts(self):
        cases=[('correlation-pearson','numeric.pearson','Pearson'),('correlation-spearman','numeric.spearman','Spearman'),('correlation-kendall','numeric.kendall','Kendall'),('correlation-cramers-v','categorical.cramers_v','CramersV'),('correlation-phi','categorical.phi','Phi'),('correlation-matrix-transform','matrix_transform','CorrelationMatrixTransform')]
        for key,path,name in cases:
            module=importlib.import_module('src.stats.correlation.'+path);source=Path(module.__file__).read_text()
            self.assertIsInstance(load_profile(key),getattr(module,name+'Profile'))
            for cls in (getattr(module,name+'Profile'),getattr(module,name+'OutputDTO')):
                for field in fields(cls):self.assertIn(field.name+' (',cls.__doc__)
            for node in ast.walk(ast.parse(source)):
                if isinstance(node,ast.FunctionDef):
                    for label in ('Args:','Returns:','Raises:'):self.assertIn(label,ast.get_docstring(node))


if __name__=='__main__':unittest.main()
