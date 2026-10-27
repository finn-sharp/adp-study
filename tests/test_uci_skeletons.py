"""
================================================================================
test_uci_skeletons.py - 네트워크 없는 스켈레톤 계약·누수·CV·시계열/장바구니 변환 검증.
================================================================================
설명:
    - 네트워크 없는 스켈레톤 계약·누수·CV·시계열/장바구니 변환 검증.
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
import contextlib
from dataclasses import fields
import importlib
from io import BytesIO, StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile
import numpy as np
import pandas as pd
from src.datasets import uci as _datasets
from src.datasets import adult as _adult
from src.MachineLearning.skeleton._common import fit_evaluate
from src.MachineLearning.skeleton import linear_regression, pca, mds
from src.MachineLearning.base.feature import FeatureSpec
from src.MachineLearning.base.preprocessor import BasePreprocessor


class UCISkeletonTest(unittest.TestCase):
    def test_feature_schema_and_preprocessor_configuration(self):
        with self.assertRaises(ValueError):FeatureSpec('numeric','age','passthrough')
        with self.assertRaises(ValueError):FeatureSpec('numeric',['a','a'],'passthrough')
        feature=FeatureSpec('numeric',['a'],'passthrough')
        processor=BasePreprocessor([feature])
        with self.assertRaises(ValueError):processor.add_feature(feature)
        self.assertEqual(processor.describe()[0]['transformer'],'passthrough')
        self.assertIs(processor.fit(pd.DataFrame({'a':[1.,2.]})),processor)
        processor.remove_feature('numeric')
        with self.assertRaises(ValueError):processor.get_preprocessor()

    def test_all_modules_import_without_network_or_training_and_follow_contract(self):
        domains = {'MachineLearning':25, 'DeepLearning':8, 'Association':3, 'Networks':1, 'Preprocessing':19, 'stats':10, 'TimeSeries':3}
        with patch.object(_datasets, 'urlopen', side_effect=AssertionError('Import must be offline')), patch.object(_adult, 'urlopen', side_effect=AssertionError('Import must be offline')):
            for domain,count in domains.items():
                package=importlib.import_module('src.'+domain+'.skeleton')
                root=Path(package.__file__).parent
                paths=[p for p in root.glob('*.py') if not p.name.startswith('_') and p.stem != 'distributions']
                self.assertEqual(len(paths),count)
                for path in paths:
                    module = importlib.import_module('src.'+domain+'.skeleton.' + path.stem)
                    for name in ('load_data', 'load_recipe', 'build_pipeline', 'main'):
                        self.assertTrue(callable(getattr(module, name)))
                    tree = ast.parse(path.read_text())
                    for name in ('__author__', '__email__', '__version__', '__status__'):
                        self.assertIsInstance(getattr(module, name), str)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            for label in ('Args:', 'Returns:', 'Raises:'):
                                self.assertIn(label, ast.get_docstring(node))
                for path in root.glob('*.py'):
                    source=path.read_text()
                    for forbidden in ('matplotlib', 'seaborn', 'plotly', 'plt.', '.plot(', 'savefig('):
                        self.assertNotIn(forbidden, source)

    def test_domain_separation_and_statistics_compatibility(self):
        root=Path(linear_regression.__file__).parent
        self.assertFalse(any(p.stem.startswith('tf_') for p in root.glob('*.py')))
        self.assertFalse(any((root/(name+'.py')).exists() for name in ('apriori','fp_growth','eclat','sna')))
        from src.stats.skeleton.distributions import run_pdf_examples
        from examples.statistics_pdf_examples import run_pdf_examples as old
        self.assertIs(old,run_pdf_examples)
        self.assertEqual(len(run_pdf_examples()),32)

    def test_preprocessing_and_grid_search_use_training_rows_only(self):
        rng = np.random.RandomState(5)
        frame = pd.DataFrame(rng.normal(size=(60, 3)), columns=['a','b','c'])
        frame['target'] = frame.a * 2 + frame.b - frame.c
        pipeline = linear_regression.build_pipeline(['a','b','c'], overrides={'scaling':'standard', 'param_grid':{'fit_intercept':[True,False]}})
        result = fit_evaluate(frame, pipeline, 'OLS')
        scaler = result.model.pipeline.named_steps['preprocessor'].named_transformers_['numeric']
        np.testing.assert_allclose(scaler.mean_, frame.iloc[result.train_indices][['a','b','c']].mean())
        self.assertFalse(set(result.train_indices) & set(result.test_indices))
        self.assertLess(result.metrics['rmse'], 1e-10)
        self.assertTrue(result.details['best_params'])
        np.testing.assert_allclose(result.predict(frame.iloc[result.test_indices].drop(columns='target')), result.predictions)
        self.assertNotIn('model', result.to_dict())

    def test_libras_frame_order_and_time_series_dates(self):
        frame = pd.DataFrame([list(range(90))+[1], list(range(90,180))+[2]])
        with patch.object(_datasets, 'dataset_bytes', return_value=frame.to_csv(index=False,header=False).encode()):
            X,y = _datasets.load_libras()
        self.assertEqual(X.shape,(2,45,2))
        np.testing.assert_array_equal(X[0,1],[2,3])
        bike = b'dteday,cnt\n2011-01-02,2\n2011-01-01,1\n'
        with patch.object(_datasets, 'dataset_bytes', return_value=bike):
            data = _datasets.load_bike()
        np.testing.assert_array_equal(data.cnt,[1,2])
        with patch.object(_datasets, 'dataset_bytes', return_value=b'dteday,cnt\n2011-01-01,1\n2011-01-03,2\n'):
            with self.assertRaises(ValueError):_datasets.load_bike()

    def test_retail_cancellation_product_codes_and_complete_invoice_groups(self):
        data = pd.DataFrame([
            ('1','12345',1,2,'2011-01-01'),('2','34567',1,2,'2011-01-02'),
            ('1','23456',1,2,'2011-01-03'),('2','23456',1,2,'2011-01-04'),
            ('C1','99999',1,2,'2011-01-01'),('1','POST',1,2,'2011-01-01'),
            ('1','88888',-1,2,'2011-01-01'),('3','45678',1,2,'2011-01-05')],
            columns=['InvoiceNo','StockCode','Quantity','UnitPrice','InvoiceDate'])
        with patch.object(_datasets,'dataset_bytes',return_value=b'x'),patch.object(pd,'read_excel',return_value=data):
            baskets=_datasets.load_baskets(max_transactions=2,max_items=10)
        self.assertEqual(baskets,[{'12345','23456'},{'34567','23456'}])

    def test_transform_support_and_profile_conflicts(self):
        X=pd.DataFrame(np.random.RandomState(2).normal(size=(20,4)),columns=list('abcd'))
        pipeline=pca.build_pipeline(X.columns)
        pipeline.pipeline.fit(X)
        self.assertEqual(pipeline.pipeline.transform(X.iloc[:2]).shape,(2,2))
        pipeline=mds.build_pipeline(X.columns,overrides={'max_iter':20,'n_init':1})
        pipeline.pipeline.fit(X)
        with self.assertRaises(ValueError):pipeline.pipeline.transform(X.iloc[:2])
        with self.assertRaises(ValueError):pca.build_pipeline(X.columns,profile=pca.load_recipe(),overrides={})

    def test_adult_official_split_missing_and_target_normalization(self):
        row='39, ?, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, '
        with tempfile.TemporaryDirectory() as directory:
            with ZipFile(Path(directory)/'adult.zip','w') as archive:
                archive.writestr('adult.data',row+'<=50K\n')
                archive.writestr('adult.test','| official test file\n'+row+'>50K.\n')
            with patch.object(_adult,'urlopen',side_effect=AssertionError('Cache must be reused')):
                train,test=_adult.load_adult(directory)
            self.assertEqual(train.income.tolist(),['<=50K'])
            self.assertEqual(test.income.tolist(),['>50K'])
            self.assertTrue(pd.isna(train.workclass.iloc[0]))
            self.assertEqual(train.columns.tolist(),_adult.ADULT_COLUMNS)
            from src.Preprocessing.skeleton.feature_pipeline import load_data,build_pipeline
            X,new,y,new_y=load_data(directory)
            self.assertNotIn('income',X.columns)
            self.assertEqual(y.tolist(),['<=50K'])
            features,transformer=build_pipeline()
            self.assertFalse(hasattr(transformer,'transformers_'))
            self.assertEqual([name for name,_ in features[0].transformer.steps],['impute','scale'])
        with self.assertRaises(ValueError):_adult.load_adult(max_rows=0)


if __name__=='__main__':unittest.main()
