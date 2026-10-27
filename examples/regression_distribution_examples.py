"""
================================================================================
regression_distribution_examples.py - 새44개 모듈의 오프라인 실행·수치/그림 예제. 실제 UCI는 각 skeleton을 사용합니다.
================================================================================
설명:
    - 새44개 모듈의 오프라인 실행·수치/그림 예제. 실제 UCI는 각 skeleton을 사용합니다.
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
import importlib,json
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
from src.output import _json_value


def main():
    """고정 원 자료로 추론·전처리·NB·QQ·회귀 pane을 실행합니다.

    Args:
        None: 네트워크 없는 예제 자료를 생성합니다.
    Returns:
        dict: 실행 결과; 객체/그림은 각 DTO 계약을 따름.
    Raises:
        ValueError/TypeError: 정의역·학습·검정·그림 오류.
    """
    from src.stats.regression.ols import run_ols
    from src.Preprocessing.regression.polynomial import run_polynomial
    from src.Preprocessing.regression.vif_selection import run_vif_selection
    from src.Preprocessing.regression.target_transform import run_target_transform
    from src.Preprocessing.regression.influence_filter import run_influence_filter
    from src.MachineLearning.LinearModels.regression_workflow import run_regression_workflow
    rng=np.random.RandomState(17);X=pd.DataFrame(rng.normal(size=(100,3)),columns=list('abc'));y=2*X.a-X.b+rng.normal(size=100)*.3;model=run_ols(X,y,X_new=X.iloc[:8]);outputs={'ols':model}
    for name in ('joint_f','linearity','normality','heteroscedasticity','independence','influence'):
        outputs[name]=getattr(importlib.import_module('src.stats.regression.'+name),'run_'+name)(model)
    from src.stats.regression.collinearity import run_collinearity
    from src.stats.regression.nested_f import run_nested_f
    outputs['collinearity']=run_collinearity(X);outputs['nested_f']=run_nested_f(run_ols(X[['a','b']],y),model)
    outputs['polynomial']=run_polynomial(X,X.iloc[:8]);outputs['vif_selection']=run_vif_selection(X.assign(duplicate=X.a),X.assign(duplicate=X.a).iloc[:8]);outputs['target_transform']=run_target_transform(np.exp(y),overrides={'method':'log'});outputs['influence_filter']=run_influence_filter(X,y)
    outputs['workflow']=run_regression_workflow(X,y,overrides={'candidates':[{'name':'baseline'},{'name':'poly','degree':2}]})
    counts=pd.DataFrame(rng.binomial(5,.4,size=(120,4)),columns=list('pqrs'));labels=np.array(['ham','spam']*60)
    for name in ('gaussian','bernoulli','multinomial','complement','poisson','binomial','categorical'):
        data=(counts>0).astype(int) if name=='bernoulli' else counts
        if name=='categorical':data=counts.astype(str);data.iloc[0,0]=None
        updates={'n_trials':5} if name=='binomial' else None
        outputs['nb_'+name]=getattr(importlib.import_module('src.MachineLearning.NaiveBayes.'+name),'run_'+name)(data,labels,overrides=updates)
    distributions=dict(normal=stats.norm(),student_t=stats.t(df=5),exponential=stats.expon(),gamma=stats.gamma(a=2),lognormal=stats.lognorm(s=1),weibull=stats.weibull_min(c=1.5),beta=stats.beta(a=2,b=2),uniform=stats.uniform(),chi_square=stats.chi2(df=5),logistic=stats.logistic(),poisson=stats.poisson(mu=1),binomial=stats.binom(n=10,p=.5),bernoulli=stats.bernoulli(p=.5))
    for name,distribution in distributions.items():
        figure,ax=plt.subplots(figsize=(6,4))
        try:
            result=getattr(importlib.import_module('src.Visualization.qq.'+name),'plot_'+name)(distribution.rvs(size=80,random_state=rng),ax=ax);figure.canvas.draw();outputs['qq_'+name]=result
        finally:plt.close(figure)
    arguments=dict(residual_fitted=(model,),scale_location=(model,),residual_order=(model,),cooks_distance=(model,),leverage=(model,),coefficient_intervals=(model,),observed_predicted=(y,model.fitted_values),prediction_interval=(model,X.iloc[:8],y.iloc[:8]),partial_residual=(model,'a'),vif=(outputs['collinearity'],))
    for name,args in arguments.items():
        figure,ax=plt.subplots(figsize=(6,4))
        try:
            result=getattr(importlib.import_module('src.Visualization.regression.'+name),'plot_'+name)(*args,ax=ax);figure.canvas.draw();outputs['plot_'+name]=result
        finally:plt.close(figure)
    for output in outputs.values():json.loads(output.to_json())
    print(str(len(outputs))+' regression/distribution module examples completed')
    return outputs


if __name__=='__main__':main()
