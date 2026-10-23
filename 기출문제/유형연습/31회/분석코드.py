"""문제 순서대로 실제 계산·그림·수치 답안을 생성하는 실행 코드."""
from pathlib import Path
import sys
FOLDER = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
repo = next(p for p in [FOLDER]+list(FOLDER.parents) if (p/'tools/exams').is_dir())
sys.path.insert(0,str(repo/'기출문제/유형연습/공통'))
from analysis import Context
ctx=Context(FOLDER)


# 1-1: 체형 자료의 품질을 확인하고 BMI를 파생하시오. BMI 산식·키 단위와 예측 시 사용 가능성을 설명한 뒤 seed=31로 7:3 층화 분할하시오.
ctx.run('1-1', 'obesity_prepare')

# 1-2: 학습 폴드 안에서 전처리하는 분류 파이프라인과 제한된 튜닝을 구성하시오. 학습 내부 CV의 macro-F1으로 설정을 선택하고 보류 macro-F1·혼동행렬을 보고하시오.
ctx.run('1-2', 'obesity_model')

# 1-3: checkup의 검사일-생년월일 차이를 365일로 나눈 몫을 연령으로 정의하고 BMI와 suitable을 생성하시오. 연령 16/17세·성별별 허용 BMI 구간은 기준표.json을 사용하며 경계·범위 밖을 검증하시오.
ctx.run('1-3', 'checkup_features')

# 1-4: 적절체중 여부와 exercise·sleep의 관계를 학습 구간에서 검정하시오. 가설·p-value·다중비교 판단을 제시하고, 보정 후 유의한 변수만 포함하는 후속 분석의 한계를 설명하시오.
ctx.run('1-4', 'checkup_select')

# 1-5: 두 관계 검정에서 보정 후 유의한 변수만 이용해 Logistic과 트리 앙상블을 비교하시오. 아무 변수도 남지 않으면 상수확률 기준모형으로 처리하시오. 같은 보류 자료에서 최종 ROC·AUC와 계수/오즈비를 보고하시오.
ctx.run('1-5', 'checkup_models')

# 2-1: 제품 무게 관측 후 theta의 사후분포 평균·분산·95% 신용구간을 계산하시오. 평균이 사전평균과 관측값 사이에 놓이는 이유와 신용구간의 의미를 설명하시오.
ctx.run('2-1', 'bayes')

# 2-2: 광고비 TV·radio·paper로 sales를 설명하시오. 전체 모형의 가설·F·p-value·기각 여부, 계수 및 잔차 진단·VIF를 제시하시오.
ctx.run('2-2', 'ads_full')

# 2-3: 광고 변수의 제거 기준을 5%로 미리 정하고 가장 큰 비유의 p-value 변수를 하나씩 제거해 재적합하시오. 과정·최종 식·VIF와 선택 후 추론의 한계를 설명하시오.
ctx.run('2-3', 'ads_backward')

# 2-4: 시간당 평균 접수 2.4건인 안정된 독립 도착 과정에서 0건 및 2건 이상 확률을 구하시오. 포아송 가정을 어기는 과산포·시간대 효과가 있으면 무엇을 재검토할지 설명하시오.
ctx.run('2-4', 'poisson_probability')
