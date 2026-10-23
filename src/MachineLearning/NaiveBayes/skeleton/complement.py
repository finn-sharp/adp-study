"""complement UCI 스켈레톤.

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
from src.Recipes import load_profile
from src.datasets.uci import load_tabular,load_bike
from src.datasets.adult import load_adult
from src.datasets.optdigits import load_optdigits
from src.MachineLearning.NaiveBayes.complement import run_complement,build_complement_classifier

def load_data(cache_dir=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        cache_dir: 전용 명세의 cache_dir 입력.
    Returns:
        tuple: 분포 정의역 원 Feature·원 타깃·선택 공식 보류 위치.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    train,y_train,test,y_test=load_optdigits(cache_dir);train=train.reshape(len(train),-1);test=test.reshape(len(test),-1);X=np.vstack([train,test])
    return pd.DataFrame(X,columns=['block_'+str(i) for i in range(64)]),np.r_[y_train,y_test],np.arange(len(train),len(train)+len(test))

def load_recipe(overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        overrides: 전용 명세의 overrides 입력.
    Returns:
        object: 분포별 전용 Profile.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    settings={};settings.update(overrides or {})
    return load_profile('naive-bayes-complement',overrides=settings)

def build_pipeline(profile=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        profile: 전용 명세의 profile 입력.
    Returns:
        callable: 원 자료를 train/CV 분포 모형에 연결하는 미실행 runner.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    from functools import partial
    return partial(run_complement,profile=load_recipe() if profile is None else profile)

def main(cache_dir=None, overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        cache_dir: 전용 명세의 cache_dir 입력.
        overrides: 전용 명세의 overrides 입력.
    Returns:
        OutputDTO: 보류 지표·원 라벨 posterior·학습 전용 분포 파라미터.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    X,y,test=load_data(cache_dir);profile=load_recipe(overrides)
    pipeline=build_pipeline(profile);result=pipeline(X,y,test_indices=test)
    print(result.test_metrics)
    return result

if __name__=='__main__':main()
