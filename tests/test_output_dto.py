"""전체 공개 분석 출력 계약 및 과학 데이터 직렬화/호환성 회귀 검사."""
import ast
import importlib
import inspect
import json
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints
import unittest
import numpy as np
import pandas as pd
from scipy import sparse
from src.output import OutputDTO
from src.stats.analysis import run_descriptive
from src.MachineLearning.Linear.least_squares import run_least_squares, LeastSquaresOutputDTO
from src.MachineLearning.common import SupervisedResult, SupervisedOutputDTO


@dataclass
class PayloadDTO(OutputDTO):
    values: Any
    model: Any = None


class OutputDTOTests(unittest.TestCase):
    def test_all_analysis_entrypoints_declare_concrete_dto(self):
        count=0
        extra={'resample_training_data','distance_matrix','make_linkage_matrix','analyze_english','explain_text'}
        for path in Path('src').rglob('*.py'):
            tree=ast.parse(path.read_text())
            functions=[n.name for n in tree.body if isinstance(n,ast.FunctionDef) and (n.name.startswith('run_') or n.name in extra)]
            if not functions:continue
            module=importlib.import_module('.'.join(path.with_suffix('').parts))
            for name in functions:
                with self.subTest(path=str(path),function=name):
                    dtype=get_type_hints(getattr(module,name)).get('return')
                    self.assertTrue(inspect.isclass(dtype) and issubclass(dtype,OutputDTO))
                    self.assertTrue(is_dataclass(dtype))
                    count+=1
        self.assertGreaterEqual(count,45)

    def test_result_alias_and_model_specific_return(self):
        self.assertIs(SupervisedResult,SupervisedOutputDTO)
        X=np.arange(40.).reshape(20,2);y=X[:,0]*2+3
        result=run_least_squares(X,y)
        self.assertIsInstance(result,LeastSquaresOutputDTO)
        self.assertIsInstance(result,SupervisedResult)
        self.assertNotIn('model',result.to_dict())
        self.assertIs(result.to_dict(include_runtime=True)['model'],result.model)
        payload=json.loads(result.to_json())
        self.assertEqual(len(payload['predictions']),len(result.y_test))
        self.assertEqual(payload['feature_names'],result.feature_names)

    def test_fitted_encoder_and_transformer_are_runtime_only(self):
        from src.Association.apriori import run_association
        from src.stats.regression import RegressionProfile, run_regression_inference
        association = run_association([['milk', 'bread'], ['milk'], ['bread']])
        x = np.linspace(-2, 2, 30).reshape(-1, 1)
        y = 1 + x[:, 0] ** 2 + np.random.RandomState(42).normal(0, 0.1, 30)
        regression = run_regression_inference(x, y, RegressionProfile(polynomial_degree=2))
        for result, field in [(association, 'encoder'), (regression, 'transformer')]:
            with self.subTest(field=field):
                self.assertIsNotNone(getattr(result, field))
                self.assertIs(result.to_dict(include_runtime=True)[field], getattr(result, field))
                payload = json.loads(result.to_json())
                self.assertNotIn(field, payload)
                self.assertTrue(payload)

    def test_descriptive_attribute_key_and_missing_key(self):
        result=run_descriptive([1,2,3,4,100])
        self.assertEqual(result.mean,result['mean'])
        self.assertEqual(result.get('absent',12),12)
        self.assertIn('sample_std',dict(result))
        with self.assertRaises(KeyError):result['absent']
        self.assertEqual(json.loads(result.to_json())['outlier_indices'],[4])

    def test_scientific_json_keeps_labels_and_sparse_coordinates(self):
        index=pd.date_range('2020-01-01',periods=2)
        frame=pd.DataFrame({'score':[1.,np.nan]},index=index)
        matrix=sparse.csr_matrix(([4.],([0],[2])),shape=(2,3))
        dto=PayloadDTO({'frame':frame,'sparse':matrix,'series':pd.Series([1.,np.inf],index=['a','b']),
                        'items':frozenset(['B','A']),'np_bool':np.bool_(True),'array':np.array([1.,np.nan,np.inf])},object())
        payload=json.loads(dto.to_json())['values']
        self.assertEqual(payload['frame']['columns'],['score'])
        self.assertEqual(payload['frame']['index'][0],index[0].isoformat())
        self.assertEqual(payload['frame']['data'],[[1.],[None]])
        self.assertEqual(payload['sparse'],{'format':'coo','shape':[2,3],'row':[0],'col':[2],'data':[4.]})
        self.assertEqual(payload['series']['data'],[1.,None])
        self.assertEqual(payload['items'],['A','B'])
        self.assertEqual(payload['array'],[1.,None,None])
        self.assertIs(payload['np_bool'],True)

    def test_json_collision_and_runtime_guard(self):
        with self.assertRaises(ValueError):PayloadDTO({1:'number','1':'string'}).to_json()
        with self.assertRaises(ValueError):PayloadDTO(1).to_dict(json_safe=True,include_runtime=True)
        with self.assertRaises(TypeError):PayloadDTO(object()).to_json()

    def test_nested_anova_output_serializes_and_legacy_access_works(self):
        from src.stats import ANOVAProfile,run_anova
        frame=pd.DataFrame({'y':[1.,2,4,6,7,8],'g':['a']*3+['b']*3})
        dto=run_anova(frame,ANOVAProfile('y',['g']))
        self.assertAlmostEqual(dto.statistic,dto.analysis.statistic)
        data=json.loads(dto.to_json())
        self.assertEqual(data['method'],'one_way')
        self.assertIn('statistic',data['analysis'])
        self.assertIn('columns',data['anova_table'])

    def test_matrix_dto_preserves_numpy_compatibility(self):
        from src.MachineLearning.Clustering.clustering import distance_matrix,make_linkage_matrix
        X=np.array([[0.,0],[1,2],[4,5]])
        distances=distance_matrix(X)
        self.assertIsInstance(distances,OutputDTO)
        np.testing.assert_allclose(np.diag(distances),0)
        self.assertEqual(json.loads(distances.to_json())['metric'],'euclidean')
        self.assertEqual(make_linkage_matrix(X).shape,(2,4))

    def test_pairplot_dto_preserves_figure_and_metadata(self):
        from src.Visualization.plots import run_pairplot,PairPlotOutputDTO
        import matplotlib.pyplot as plt
        dto=run_pairplot(pd.DataFrame({'a':[1,2,3],'b':[4,5,6]}))
        try:
            self.assertIsInstance(dto,PairPlotOutputDTO)
            self.assertIs(dto.fig,dto.grid.fig)
            self.assertEqual(json.loads(dto.to_json()),{'feature_names':['a','b'],'hue':None})
        finally:plt.close(dto.figure)

if __name__=='__main__':unittest.main()
