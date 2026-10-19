"""문제 순서대로 실제 계산·그림·수치 답안을 생성하는 실행 코드."""
from pathlib import Path
import sys
FOLDER = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
repo = next(p for p in [FOLDER]+list(FOLDER.parents) if (p/'tools/exams').is_dir())
sys.path.insert(0,str(repo/'기출문제/유형연습/공통'))
from analysis import Context
ctx=Context(FOLDER)


# 1-1: 결측 처리 대안 두 가지를 비교하고 품질·분포·공선성을 조사하시오. 타깃을 이용하지 않는 주성분 표현의 필요성과 해석 손실을 설명하시오.
ctx.run('1-1', 'hepatitis_prepare')

# 1-2: 정상 대 질환의 이진 타깃을 만들고 원 학습과 무작위 오버/언더 표본 대안을 비교하시오. CV의 F1과 원 분포 보류 평가에서 precision·recall·F1·AUC를 보고하시오.
ctx.run('1-2', 'hepatitis_binary')

# 1-3: 질환군 1~3만 남겨 서로 다른 세 분류 모형을 같은 macro-F1 기준으로 비교하시오. 선택 모형의 보류 혼동행렬·정밀도·재현율과 특성 중요도 해석을 제시하시오.
ctx.run('1-3', 'hepatitis_multi')

# 2-1: 이용량 표를 변환·집계하고 날씨를 날짜로 일대일 결합하시오. 중복 키·미매칭 행을 검사하고 주말 여부를 생성하시오. 숫자 결측과 날짜 누락의 차이를 설명하시오.
ctx.run('2-1', 'subway_merge')

# 2-2: 날씨·주말 특성으로 이용량을 예측하는 선형 및 트리 모형을 시간순 CV로 비교하시오. 정해진 최종 연도의 RMSE·MAE·R²와 시간 의존성에 따른 한계를 보고하시오.
ctx.run('2-2', 'subway_models')

# 3-1: 두 공정의 평균 불량률 차이를 평가하시오. 평균·표준편차·차이, 적절한 검정·가설·통계량·p-value·기각 여부를 제시하시오. 일별 비율과 이항 성공/실패 원시 건수의 차이를 설명하시오.
ctx.run('3-1', 'defect')

# 3-2: 연착 건수 자료가 하나의 일정률 발생 모형에 맞는지 평가하시오. 모수를 추정하고 관측·기대 빈도를 만들며 기대도수가 5 이상이 되도록 꼬리를 합치시오. 모수 추정 자유도·p-value·판정을 보고하시오.
ctx.run('3-2', 'poisson_gof')

# 3-3: 노선과 연착 구간(≤10, 10초과~20, >20분)의 관련성을 평가하시오. 구간 경계·관측/기대도수·가설·통계량·자유도·p-value·기각 여부를 제시하시오.
ctx.run('3-3', 'delay')

# 3-4: ACF·PACF를 설명하고 자료의 정상성을 진단하시오. ARMA 후보 (1,0),(2,0),(1,1),(2,1)을 학습 80%에서 AIC로 고르고, 시험 20%의 RMSE 및 선택 모형 잔차 10시차 진단을 제시하시오.
ctx.run('3-4', 'arma')
