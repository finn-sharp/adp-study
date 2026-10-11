"""
================================================================================
multivariate_workflow.py - ANOVA·MANOVA·사후검정 호출 예제
================================================================================
설명: 고정 시드 합성 자료로 일원/Welch/이원/반복측정·다변량·사후검정을 실행합니다.
작성 정보: 작성자 김재현(Finn), 작성일 2026-10-11, 버전 1.0.0
변경 이력: 2026-10-11 v1.0.0 Python 3.7 호환 실행 예제 추가
실행: 저장소 루트에서 python -m examples.multivariate_workflow
================================================================================
"""
__version__ = "1.0.0"
import numpy as np
import pandas as pd
from src.stats import ANOVAProfile, run_anova, MANOVAProfile, run_manova, PostHocProfile, run_posthoc


def main():
    """합성 자료로 모듈 실행 및 검정표 접근법을 보여줍니다.

    Args:
        없음. 고정 난수 시드를 사용합니다.
    Returns:
        None: 검정표/통계량을 출력합니다.
    Raises:
        ImportError: requirements-py37.txt 또는 requirements-review.txt 의존성 미설치.
    """
    rng=np.random.RandomState(7)
    group=np.repeat(['A','B','C'],20)
    y=rng.normal(size=60)+np.repeat([0,1,2],20)
    data=pd.DataFrame({'score':y,'second':.4*y+rng.normal(size=60),'group':group})
    for equal in (True,False):
        result=run_anova(data,ANOVAProfile('score',['group'],equal_var=equal))
        print(result.test_type,result.statistic,result.p_value)
    data['factor2']=np.tile(np.repeat(['low','high'],10),3)
    two=run_anova(data,ANOVAProfile('score',['group','factor2'],method='two_way',ss_type=3))
    print('Two-way ANOVA:',two.anova_table)
    multiple=run_manova(data,MANOVAProfile(['score','second'],['group']))
    print('MANOVA:',multiple.tables['group'])
    for p in (PostHocProfile(),PostHocProfile(method='games_howell',equal_var=False),
              PostHocProfile(method='pairwise_t',equal_var=False,correction='holm')):
        comparisons=run_posthoc(data.score,data.group,p)
        print(comparisons.method,comparisons.table)
    repeated=pd.DataFrame({'subject':np.repeat(range(10),3),'visit':np.tile(['A','B','C'],10),'score':rng.normal(size=30)})
    print('Repeated ANOVA:',run_anova(repeated,ANOVAProfile('score',['visit'],method='repeated',subject='subject')).anova_table)

if __name__=='__main__':main()
