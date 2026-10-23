"""feature_design UCI 스켈레톤.

원 자료 → JSON/Profile → 학습 전용 파이프라인 → 수치/진단/사용자 Axes.
작성자: 김재현(Finn), 2026-10-12. 변경 이력: v1.0.0 최초 추가.
라이선스: Proprietary; UCI 원본은 CC BY 4.0. import에서 다운로드·학습·그림 생성 없음.
"""
__author__='김재현(Finn)'
__email__='penguin.klg@gmail.com'
__version__='1.0.0'
__status__='Development'
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.Recipes import load_profile
from src.datasets.uci import load_tabular
from src.Preprocessing.numeric.imputation import build_numeric_imputation_transformer
from src.Preprocessing.numeric.scaling import build_numeric_scaling_transformer
from src.Preprocessing.feature_pipeline import make_feature_spec,build_feature_preprocessor
from src.Preprocessing.regression.polynomial import run_polynomial
from src.Preprocessing.regression.vif_selection import run_vif_selection
from src.Preprocessing.regression.target_transform import run_target_transform
from src.Preprocessing.regression.influence_filter import run_influence_filter

def load_data(cache_dir=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        cache_dir: 전용 명세의 cache_dir 입력.
    Returns:
        tuple: 명시 분리 train/test Feature·train 타깃.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    data=load_tabular('regression',cache_dir=cache_dir);X=data.iloc[:,:4];train,test=train_test_split(np.arange(len(data)),test_size=.3,random_state=1234)
    return X.iloc[train],X.iloc[test],data.target.iloc[train].to_numpy()

def load_recipe(overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        overrides: 전용 명세의 overrides 입력.
    Returns:
        dict: 각각 전용 Profile.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    names={'polynomial':'regression-polynomial','vif':'regression-vif-selection','target':'regression-target-transform','influence':'regression-influence-filter'};updates=overrides or {}
    if set(updates)-set(names):raise ValueError('전처리 pane 이름 오류')
    return {name:load_profile(key,overrides=updates.get(name)) for name,key in names.items()}

def build_pipeline(feature_columns):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        feature_columns: 전용 명세의 feature_columns 입력.
    Returns:
        ColumnTransformer: 미적합 Feature 전처리.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    feature=make_feature_spec('numeric',list(feature_columns),[('impute',build_numeric_imputation_transformer()),('scale',build_numeric_scaling_transformer())])
    return build_feature_preprocessor([feature])

def main(cache_dir=None, overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        cache_dir: 전용 명세의 cache_dir 입력.
        overrides: 전용 명세의 overrides 입력.
    Returns:
        dict: 학습 전용 생성/선택/타깃 변환·명시 관측 처리. 최종 model은 별도 fit.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    train,test,y=load_data(cache_dir);profiles=load_recipe(overrides);processor=build_pipeline(train.columns).fit(train);names=processor.get_feature_names_out().tolist();A=pd.DataFrame(processor.transform(train),index=train.index,columns=names);B=pd.DataFrame(processor.transform(test),index=test.index,columns=names)
    poly=run_polynomial(A,B,profiles['polynomial']);expanded=pd.DataFrame(poly.preprocessed_feature,index=train.index,columns=poly.output_features);future=pd.DataFrame(poly.new_preprocessed_feature,index=test.index,columns=poly.output_features)
    vif=run_vif_selection(expanded,future,profiles['vif']);target=run_target_transform(y,profile=profiles['target']);design=pd.DataFrame(vif.preprocessed_feature,index=train.index,columns=vif.output_features);influence=run_influence_filter(design,target.transformed_target,profiles['influence'])
    return {'polynomial':poly,'vif':vif,'target':target,'influence':influence}

if __name__=='__main__':
    result=main();print(result['vif'].selection_history);print(result['influence'].review_table.head())
