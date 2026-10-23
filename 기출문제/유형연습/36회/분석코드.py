"""문제 순서대로 실제 계산·그림·수치 답안을 생성하는 실행 코드."""
from pathlib import Path
import sys
FOLDER = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
repo = next(p for p in [FOLDER]+list(FOLDER.parents) if (p/'tools/exams').is_dir())
sys.path.insert(0,str(repo/'기출문제/유형연습/공통'))
from analysis import Context
ctx=Context(FOLDER)


# 1-1: 연속 target 예측에 학습 내부 전진 변수 선택을 적용하시오. 3-fold CV RMSE가 0.01 이상 개선될 때만 변수를 추가하고 이력·선택 변수·보류 RMSE를 제시하시오.
ctx.run('1-1', 'cascade_select')

# 1-2: 선택한 회귀기에 대해 학습 out-of-fold 연속 예측값과 최종 시험 예측값을 생성하시오. 자기 정답이나 in-sample 적합값을 후속 분류에 넣는 것과 비교해 누출 위험을 설명하시오.
ctx.run('1-2', 'cascade_oof')

# 1-3: x1~x4와 예측 연속값으로 분류 파이프라인을 적합하고 최종 시험 F1·AUC·중요도와 전체 두 단계 오류 전파의 한계를 제시하시오.
ctx.run('1-3', 'cascade_classify')

# 2-1: 중첩 JSON을 행과 열로 평탄화하고 date·source·entity·temperature·load를 갖는 표를 저장하시오. 객체 수·행 수·키·타입을 검증하시오.
ctx.run('2-1', 'json_flatten')

# 2-2: entity별 평균 온도와 일자별 총 load를 요약하시오. 중복 또는 빈 records 배열이 있을 때 집계 의미·검사 방법을 설명하시오.
ctx.run('2-2', 'json_aggregate')

# 3-1: 변수 간 상관과 표본 적합성을 점검하고, 상관행렬이 단위행렬인지 검정하시오. 가설·통계량·자유도·p-value·기각 여부와 표본 적합성 지표를 제시하시오.
ctx.run('3-1', 'factor_suitability')

# 3-2: 상관행렬 고유값·scree 그림과 200회 독립 열 순열의 95% 기준(seed=36)을 비교해 유지할 차원 수를 제안하시오. 두 판단 기준의 차이와 탐색적 선택의 한계를 설명하시오.
ctx.run('3-2', 'factor_count')

# 3-3: 선택 차원으로 공통요인 모형을 적합해 적재량·고유분산·공통성을 제시하고 회전 후 해석하시오. 주성분 분석과의 목적·오차 모형 차이를 설명하시오.
ctx.run('3-3', 'factor_fit')
