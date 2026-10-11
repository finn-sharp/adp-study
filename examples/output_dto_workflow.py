"""
================================================================================
output_dto_workflow.py - 결과 DTO의 필드·사전·JSON 활용
================================================================================
설명: python -m examples.output_dto_workflow; 외부 자료/결과 파일 자동 생성 없음.
작성 정보: 작성자 김재현(Finn), 작성일 2026-10-11, 버전 1.0.0
변경 이력: 2026-10-11 v1.0.0 결과 데이터와 실행 객체의 접근 예제 추가
================================================================================
"""
__version__ = "1.0.0"
import json
import numpy as np
import pandas as pd
from src.stats import run_descriptive, run_anova, ANOVAProfile
from src.MachineLearning.Regression import run_multiple_regression


def main():
    """분석 결과의 재사용·JSON 저장용 변환을 시연합니다.

    Args:
        없음. 고정 합성 데이터 사용.
    Returns:
        None: 결과 타입·필드 및 JSON 저장용 값을 출력.
    Raises:
        ValueError: DTO 변환/모듈 입력 오류.
    """
    descriptive=run_descriptive([1,2,3,4,100])
    print(type(descriptive).__name__,descriptive.mean,descriptive['mean'])
    print(descriptive.to_json())
    X=np.arange(80.).reshape(40,2);y=2*X[:,0]+3
    regression=run_multiple_regression(X,y)
    print(type(regression).__name__,regression.test_metrics)
    print('New input prediction:',regression.model.predict(X[:2]))
    stored=json.loads(regression.to_json())
    print('Stored fields:',list(stored),'runtime excluded:', 'model' not in stored)
    frame=pd.DataFrame({'y':[1.,2,4,6,7,8],'group':['a']*3+['b']*3})
    anova=run_anova(frame,ANOVAProfile('y',['group']))
    print(type(anova).__name__,anova.analysis.p_value)
    print('ANOVA stored fields:',list(json.loads(anova.to_json())))

if __name__=='__main__':main()
