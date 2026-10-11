"""
================================================================================
extended_workflow.py - 회귀·ARIMA/SARIMA·수치형/범주형 리샘플링 사용 예제
================================================================================
설명: 저장소 루트에서 python -m examples.extended_workflow 실행.
작성 정보: 작성일 2026-10-11, 버전 1.0.0
변경 이력: 2026-10-11 v1.0.0 외부 자료 없이 실행 가능한 호출 예제 추가
================================================================================
"""
__version__ = "1.0.0"
__status__ = "Development"
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.MachineLearning.Regression import (run_multiple_regression, run_polynomial_regression,
    run_nonlinear_regression, PolynomialRegressionProfile, NonlinearRegressionProfile)
from src.TimeSeries import run_arima, run_sarima, ARIMAProfile, SARIMAProfile
from src.Preprocessing import SamplingProfile, resample_training_data, build_sampling_pipeline


def main():
    """모듈별 학습·평가와 안전한 리샘플링 호출을 실행합니다.

    Args:
        없음. 고정 시드 합성 데이터 사용.
    Returns:
        None: 보류 지표 및 클래스 분포를 출력.
    Raises:
        ImportError: requirements-review.txt 선택 의존성 미설치.
    """
    rng=np.random.RandomState(18)
    X=rng.normal(size=(90,3)); y=4+X.dot([2,-3,.5])
    print('multiple:',run_multiple_regression(X,y).test_metrics)
    y_poly=1+X[:,0]**2+2*X[:,0]*X[:,1]
    print('polynomial:',run_polynomial_regression(X,y_poly,PolynomialRegressionProfile(degree=2)).test_metrics)
    x=np.linspace(-1,1,90).reshape(-1,1)
    curve=lambda z,a,b,c:a*np.exp(b*z[0])+c
    fit=run_nonlinear_regression(x,curve(x.T,2.,.7,.3),curve,NonlinearRegressionProfile(initial_params=(1.,.5,0.)))
    print('nonlinear parameters:',fit.model.named_steps['model'].parameters_)
    series=np.sin(np.arange(72)*2*np.pi/6)+rng.normal(0,.1,72)
    print('ARIMA:',run_arima(series,ARIMAProfile(order=(1,0,0),holdout=6)).metrics)
    print('SARIMA:',run_sarima(series,SARIMAProfile(order=(0,0,0),seasonal_order=(1,0,0,6),holdout=6)).metrics)
    X,y=make_classification(n_samples=180,n_features=4,n_redundant=0,weights=[.75,.25],random_state=18)
    train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=.25,stratify=y,random_state=18)
    sampled=resample_training_data(train_X,train_y,SamplingProfile(method='smote',neighbors=3))
    print('numeric counts:',sampled.counts_before,sampled.counts_after)
    pipeline=build_sampling_pipeline(LogisticRegression(max_iter=500),SamplingProfile(method='smote',neighbors=3),before_sampling=StandardScaler())
    search=GridSearchCV(pipeline,{'model__C':[.1,1.]},cv=3,error_score='raise').fit(train_X,train_y)
    print('sampling holdout accuracy:',search.score(test_X,test_y))
    # 전체 열이 범주형인 예제. 부호 구분 경계 0은 데이터에서 학습하지 않습니다.
    categorical=np.where(train_X>0,'positive','nonpositive')
    result=resample_training_data(categorical,train_y,SamplingProfile(method='smoten',data_kind='categorical',neighbors=3))
    print('categorical counts:',result.counts_before,result.counts_after)
    cat_pipe=build_sampling_pipeline(LogisticRegression(max_iter=500),SamplingProfile(method='smoten',data_kind='categorical',neighbors=3),after_sampling=OneHotEncoder(handle_unknown='ignore'))
    cat_pipe.fit(categorical,train_y)
    print('categorical holdout accuracy:',cat_pipe.score(np.where(test_X>0,'positive','nonpositive'),test_y))

if __name__=='__main__':main()
