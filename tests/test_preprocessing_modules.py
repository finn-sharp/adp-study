"""
================================================================================
test_preprocessing_modules.py - 학습 전용 전처리·미지 범주·FeatureSpec/CV·RoPE 수학 검증.
================================================================================
설명:
    - 학습 전용 전처리·미지 범주·FeatureSpec/CV·RoPE 수학 검증.
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
import importlib,json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PowerTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV,KFold,cross_validate
from src.Recipes import load_profile
from src.Preprocessing.numeric.scaling import run_numeric_scaling,build_numeric_scaling_transformer
from src.Preprocessing.numeric.imputation import run_numeric_imputation,build_numeric_imputation_transformer
from src.Preprocessing.numeric.power import run_numeric_power,build_numeric_power_transformer
from src.Preprocessing.categorical.encoding import run_categorical_encoding,build_categorical_encoding_transformer
from src.Preprocessing.categorical.imputation import run_categorical_imputation
from src.Preprocessing.positional.rope import run_rope,build_rope_transformer
from src.MachineLearning.base.feature import FeatureSpec
from src.MachineLearning.base.preprocessor import BasePreprocessor
from src.Preprocessing.feature_pipeline import make_feature_spec,build_feature_preprocessor,run_feature_preprocessing


class PreprocessingModulesTest(unittest.TestCase):
    def test_scaling_training_only_inverse_and_all_methods(self):
        X=pd.DataFrame({'a':[1.,3.,5.],'b':[2.,4.,8.]});new=X+100
        r=run_numeric_scaling(X,new)
        ref=StandardScaler().fit(X)
        np.testing.assert_allclose(r.preprocessed_feature,ref.transform(X))
        np.testing.assert_allclose(r.new_preprocessed_feature,ref.transform(new))
        np.testing.assert_allclose(r.transformer.estimator_.mean_,X.mean())
        np.testing.assert_allclose(r.transformer.inverse_transform(r.preprocessed_feature),X)
        for method in ('none','standard','minmax','robust','maxabs','normalize'):
            output=run_numeric_scaling(X,new,overrides={'method':method});self.assertEqual(output.preprocessed_feature.shape,X.shape)
            json.loads(output.to_json())
        with self.assertRaises(ValueError):r.transformer.transform(new[['b','a']])
        self.assertNotIn('transformer',r.to_dict())

    def test_imputation_nullable_indicator_unknown_missing_and_empty_columns(self):
        X=pd.DataFrame({'a':pd.Series([1,None,5],dtype='Int64'),'b':[2.,4.,6.]})
        new=pd.DataFrame({'a':[None,100.],'b':[None,8.]})
        r=run_numeric_imputation(X,new,overrides={'add_indicator':True})
        self.assertEqual(r.preprocessed_feature.shape,(3,3));np.testing.assert_allclose(r.new_preprocessed_feature[:,0],[3,100])
        np.testing.assert_allclose(r.new_preprocessed_feature[:,1],[4,8])
        self.assertIn('missingindicator_a',r.output_features)
        for strategy in ('mean','median','most_frequent','constant'):
            self.assertEqual(run_numeric_imputation(X,overrides={'strategy':strategy}).preprocessed_feature.shape,(3,2))
        with self.assertRaises(ValueError):run_numeric_imputation([[np.nan],[np.nan]])
        self.assertEqual(run_numeric_imputation([[np.nan]],overrides={'strategy':'constant'}).preprocessed_feature.shape,(1,1))

    def test_power_domain_training_state_and_factory_validation(self):
        X=np.array([[1.,2.],[2.,4.],[4.,9.],[8.,10.]])
        for method in ('yeo-johnson','box-cox'):
            r=run_numeric_power(X,X+.5,overrides={'method':method});reference=PowerTransformer(method=method).fit(X)
            np.testing.assert_allclose(r.preprocessed_feature,reference.transform(X))
            np.testing.assert_allclose(r.new_preprocessed_feature,reference.transform(X+.5))
        with self.assertRaises(ValueError):build_numeric_power_transformer(overrides={'method':'box-cox'}).fit([[0.],[1.]])
        with self.assertRaises(ValueError):build_numeric_power_transformer().fit([[1.],[1.]])

    def test_categorical_imputation_encoders_unknowns_and_collisions(self):
        X=pd.DataFrame({'city':['Seoul','Busan','Seoul'],'grade':['low','high','low']})
        new=pd.DataFrame({'city':['Tokyo'],'grade':['middle']})
        one=run_categorical_encoding(X,new);np.testing.assert_array_equal(one.new_preprocessed_feature,0)
        self.assertEqual(len(one.output_features),4)
        with self.assertRaises(ValueError):run_categorical_encoding(X,new,overrides={'handle_unknown':'error'})
        ordinal=run_categorical_encoding(X,new,overrides={'method':'ordinal','category_orders':[['Busan','Seoul'],['low','high']]})
        np.testing.assert_array_equal(ordinal.preprocessed_feature[:,1],[0,1,0]);np.testing.assert_array_equal(ordinal.new_preprocessed_feature,[[-1,-1]])
        frequency=run_categorical_encoding(X,new,overrides={'method':'frequency'})
        np.testing.assert_allclose(frequency.preprocessed_feature[:,0],[2/3,1/3,2/3]);np.testing.assert_array_equal(frequency.new_preprocessed_feature,0)
        source=pd.DataFrame({'city':['Seoul',None,'Seoul']})
        r=run_categorical_imputation(source,pd.DataFrame({'city':[None]}));self.assertEqual(r.new_preprocessed_feature[0,0],'Seoul')
        self.assertEqual(run_categorical_imputation([[None]],overrides={'strategy':'constant'}).preprocessed_feature[0,0],'(missing)')
        with self.assertRaises(ValueError):run_categorical_encoding(np.array([[1],['1']],dtype=object))
        sparse=run_categorical_encoding(X,overrides={'sparse_output':True});json.loads(sparse.to_json())

    def test_feature_spec_clone_and_cv_keep_preprocessing_in_folds(self):
        X=pd.DataFrame({'a':[1.,2.,4.,8.,16.,32.],'b':[2.,5.,3.,9.,12.,25.]});y=np.arange(6.)
        transform=Pipeline([('impute',build_numeric_imputation_transformer()),('scale',build_numeric_scaling_transformer())])
        pre=BasePreprocessor([FeatureSpec('numeric',['a','b'],transform)]).get_preprocessor()
        model=Pipeline([('preprocess',pre),('model',Ridge())]);search=GridSearchCV(model,{'model__alpha':[.1,1.]},cv=3).fit(X,y)
        np.testing.assert_allclose(search.best_estimator_.named_steps['preprocess'].named_transformers_['numeric'].named_steps['scale'].estimator_.mean_,X.mean())
        clone(build_categorical_encoding_transformer()).fit(pd.DataFrame({'c':['a','b']}))
        cv=KFold(n_splits=3)
        fitted=cross_validate(model,X,y,cv=cv,return_estimator=True)['estimator']
        for estimator,(train,test) in zip(fitted,cv.split(X)):
            scaler=estimator.named_steps['preprocess'].named_transformers_['numeric'].named_steps['scale'].estimator_
            np.testing.assert_allclose(scaler.mean_,X.iloc[train].mean())
            self.assertEqual(scaler.n_samples_seen_,len(train))

    def test_explicit_feature_steps_mixed_groups_target_and_overlap_guards(self):
        X=pd.DataFrame({'amount':[1.,3.,np.nan],'category':['a',None,'b'],'unused':[5,6,7]})
        numeric=make_feature_spec('numeric',['amount'],[('impute',build_numeric_imputation_transformer()),('scale',build_numeric_scaling_transformer())])
        from src.Preprocessing.categorical.imputation import build_categorical_imputation_transformer
        category=make_feature_spec('category',['category'],[('impute',build_categorical_imputation_transformer()),('encode',build_categorical_encoding_transformer())])
        result=run_feature_preprocessing(X,[numeric,category],X.copy())
        self.assertEqual(result.preprocessed_feature.shape,(3,3));self.assertNotIn('remainder__unused',result.output_features)
        self.assertEqual([step['name'] for step in result.feature_specs[0]['steps']],['impute','scale'])
        clone(build_feature_preprocessor([numeric,category])).fit(X)
        profile=load_profile('feature-preprocessing',overrides={'target_columns':['income']})
        self.assertIsInstance(profile.target_columns,list)
        build_feature_preprocessor([numeric,category],profile)
        with self.assertRaises(ValueError):build_feature_preprocessor([numeric],overrides={'target_columns':['amount']})
        with self.assertRaises(ValueError):build_feature_preprocessor([numeric],overrides={'target_columns':['income'],'remainder':'passthrough'})
        duplicate=make_feature_spec('other',['amount'],[('scale',build_numeric_scaling_transformer())])
        with self.assertRaises(ValueError):build_feature_preprocessor([numeric,duplicate])
        rope=FeatureSpec('position',['amount'],build_rope_transformer())
        with self.assertRaises(ValueError):build_feature_preprocessor([rope])
        with self.assertRaises(ValueError):make_feature_spec('invalid','amount',[('scale',build_numeric_scaling_transformer())])

    def test_rope_formula_inverse_norm_partial_dims_relative_position_and_clone(self):
        rng=np.random.RandomState(3);X=rng.normal(size=(2,5,8)).astype(np.float32);positions=np.arange(5)+2
        for layout in ('interleaved','split_half'):
            forward=run_rope(X,positions,overrides={'rotary_dim':6,'layout':layout})
            inverse=run_rope(forward.preprocessed_feature,positions,overrides={'rotary_dim':6,'layout':layout,'inverse':True})
            np.testing.assert_allclose(inverse.preprocessed_feature,X,atol=2e-6)
            np.testing.assert_allclose(np.linalg.norm(forward.preprocessed_feature,axis=-1),np.linalg.norm(X,axis=-1),rtol=1e-6)
            np.testing.assert_array_equal(forward.preprocessed_feature[...,6:],X[...,6:])
        vector=np.array([[1.,0.]]);r=run_rope(vector,np.array([1.]));np.testing.assert_allclose(r.preprocessed_feature,[[np.cos(1),np.sin(1)]])
        q=rng.normal(size=(1,8));k=rng.normal(size=(1,8));rq=run_rope(q,[2]).preprocessed_feature[0];rk=run_rope(k,[5]).preprocessed_feature[0];relative=run_rope(k,[3]).preprocessed_feature[0]
        self.assertAlmostEqual(float(rq@rk),float(q[0]@relative))
        clone(build_rope_transformer()).fit_transform(X)
        with self.assertRaises(ValueError):run_rope(np.ones((3,5)))
        with self.assertRaises(ValueError):run_rope(X,[-1,0,1,2,3])

    def test_full_metadata_profile_dto_sections_and_function_contract(self):
        cases=[('numeric-scaling','src.Preprocessing.numeric.scaling','NumericScaling'),('numeric-imputation','src.Preprocessing.numeric.imputation','NumericImputation'),('numeric-power','src.Preprocessing.numeric.power','NumericPower'),('categorical-imputation','src.Preprocessing.categorical.imputation','CategoricalImputation'),('categorical-encoding','src.Preprocessing.categorical.encoding','CategoricalEncoding'),('rope','src.Preprocessing.positional.rope','RoPE')]
        cases.append(('feature-preprocessing','src.Preprocessing.feature_pipeline','FeaturePreprocessing'))
        for key,path,name in cases:
            module=importlib.import_module(path);source=Path(module.__file__).read_text()
            self.assertIsInstance(load_profile(key),getattr(module,name+'Profile'))
            for index,section in enumerate(('Imports','Input Profile','Output DTO','Input Validation','Individual Model Runners','Public Runner','Example Execution'),1):
                self.assertIn(str(index)+'. '+section,source)
            for cls in (getattr(module,name+'Profile'),getattr(module,name+'OutputDTO')):
                for field in fields(cls):self.assertIn(field.name+' (',cls.__doc__)
            for node in ast.walk(ast.parse(source)):
                if isinstance(node,ast.FunctionDef):
                    for label in ('Args:','Returns:','Raises:'):self.assertIn(label,ast.get_docstring(node))


if __name__=='__main__':unittest.main()
