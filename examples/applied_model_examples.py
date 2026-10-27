"""
================================================================================
applied_model_examples.py - 응용 모의에 필요한 새8개 모듈의 오프라인 실제 실행·DTO 직렬화.
================================================================================
설명:
    - 응용 모의에 필요한 새8개 모듈의 오프라인 실제 실행·DTO 직렬화.
분석 가정:
    - 입력의 형상·타입·자료 순서를 호출자가 확인합니다. 계산과 파일 쓰기의 조건은 각 공개 함수의 계약을 따릅니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-27
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from src.stats.regression.information_selection import run_information_selection
from src.MachineLearning.LinearModels.weighted_regression import run_weighted_regression
from src.MachineLearning.GeneralizedLinearModels.logistic import run_logistic
from src.MachineLearning.GeneralizedLinearModels.poisson import run_poisson
from src.MachineLearning.GeneralizedLinearModels.gamma import run_gamma
from src.TimeSeries.rolling_selection import run_rolling_selection
from src.Visualization.regression.model_comparison import plot_model_comparison
from src.Visualization.regression.glm_residuals import plot_glm_residuals

def main():
    rng=np.random.RandomState(2026);X=pd.DataFrame(rng.normal(size=(100,3)),columns=list('abc'));y=3+X.a+rng.normal(size=100)*.4
    results=[run_information_selection(X,y),run_weighted_regression(X,y),run_logistic(X,rng.binomial(1,.4,100)),run_poisson(X,rng.poisson(3,size=100)),run_gamma(X,np.exp(y/4)),run_rolling_selection(pd.Series(y.to_numpy(),index=pd.date_range('2020-01-01',periods=100)),overrides={'candidate_orders':[[0,0,0],[1,0,0]]})]
    fig,axes=plt.subplots(1,2);results.append(plot_model_comparison(pd.DataFrame({'name':['baseline','candidate'],'cv_r2':[.2,.4]}),ax=axes[0]));results.append(plot_glm_residuals(results[2],ax=axes[1]));fig.canvas.draw();plt.close(fig)
    for result in results:result.to_json()
    print('8 applied modules executed with strict DTO JSON')
    return results
if __name__=='__main__':main()
