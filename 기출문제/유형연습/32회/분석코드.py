"""문제 순서대로 실제 계산·그림·수치 답안을 생성하는 실행 코드."""
from pathlib import Path
import sys
FOLDER = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
repo = next(p for p in [FOLDER]+list(FOLDER.parents) if (p/'tools/exams').is_dir())
sys.path.insert(0,str(repo/'기출문제/유형연습/공통'))
from analysis import Context
ctx=Context(FOLDER)


# 1-1: 다중 선형모형의 전체 가설·계수와 잔차 그림을 제시하고, 정규성·등분산성·함수형태 진단을 각각 구분해 수행하시오. 전체 F의 유의성과 선형성 진단의 차이를 설명하시오.
ctx.run('1-1', 'housing_diagnostics')

# 1-2: 특성 공선성을 확인하고 표준화 후 모든 주성분으로 회귀한 모형과 원래 회귀의 적합값을 비교하시오. 왜 모든 주성분 유지가 기존 잔차 문제를 자동 해결하지 않는지 설명하시오.
ctx.run('1-2', 'housing_pca')

# 1-3: 원모형의 HC3 강건 표준오차와 통상 표준오차를 비교하시오. 모형 가정 위반에 대한 대안을 제시하고 강건 표준오차가 해결하는 것과 해결하지 못하는 것을 구분하시오.
ctx.run('1-3', 'housing_robust')

# 2-1: cyl×gear 관측/기대도수를 만들고 근사 검정의 적합성을 평가하시오. 희소 셀이 있을 때 원표의 전체 관계를 검정하는 대안을 설명하시오.
ctx.run('2-1', 'category_expected')

# 2-2: 행·열 주변합을 조건으로 한 무작위 표 4999개(seed=32)를 생성해 전체 독립성을 평가하시오. 관측 Pearson 통계량, (초과횟수+1)/(4999+1) 형태 p-value·Monte Carlo 표준오차·기각 여부를 제시하시오.
ctx.run('2-2', 'category_mc')

# 2-3: 두 집단 급여의 분포·이상치를 확인하고 순위를 이용해 분포의 차이를 평가하시오. 검정 선택·가정·H0/H1·통계량·p-value·기각 여부를 제시하고 자동으로 중앙값 차이라고 해석할 수 있는지 설명하시오.
ctx.run('2-3', 'salary')
