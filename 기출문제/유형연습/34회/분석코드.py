"""
================================================================================
분석코드.py - 문제 순서대로 실제 계산·그림·수치 답안을 생성하는 실행 코드.
================================================================================
설명:
    - 문제 순서대로 실제 계산·그림·수치 답안을 생성하는 실행 코드.
분석 가정:
    - 제공 자료·문항 명세·실행 환경을 따릅니다. 복원·창작 연습 자료의 수치를 공식 기출 정답으로 해석하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-27
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""
from pathlib import Path
import sys
FOLDER = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
repo = next(p for p in [FOLDER]+list(FOLDER.parents) if (p/'tools/exams').is_dir())
sys.path.insert(0,str(repo/'기출문제/유형연습/공통'))
from analysis import Context
ctx=Context(FOLDER)


# 1-1: 서열값·결측·ind 규칙을 검증해 bleed를 만들고 누락 범주가 보존된 빈도/교차표를 제시하시오. ind=2의 정답은 평가 외에 사용하지 마시오.
ctx.run('1-1', 'clinical_prepare')

# 1-2: 관측 학습 라벨로 선형 및 앙상블 분류기를 내부 CV로 비교하시오. 관측 학습 행에는 out-of-fold 확률을, 미관측 학습 및 최종 시험에는 학습 모형 예측을 생성하고 관측 시험 bleed의 F1·AUC를 보고하시오.
ctx.run('1-2', 'clinical_oof')

# 1-3: age·bp·signal과 앞 단계 예측확률을 사용해 dead를 예측하시오. bleed를 관측 여부에 따라 진짜/예측 라벨로 섞지 않고, 시험 데이터에서 앞 단계 오차가 전파되는 한계를 설명하시오.
ctx.run('1-3', 'clinical_dead')

# 2-1: 일별 발전량을 구성·결합하고 계절을 만들며 변수별 기술통계와 기상변수-발전량 관계를 제시하시오. 계절별 평균 차이를 단정하기 전에 시간 의존성을 설명하시오.
ctx.run('2-1', 'power_merge')

# 2-2: 선형 및 비선형 회귀기를 시간순 CV로 선택해 최종 연도의 성능을 비교하시오. 미래 날짜의 실측 날씨가 필요하다는 가정과 실제 예보 운영의 차이를 설명하시오.
ctx.run('2-2', 'power_models')

# 3-1: 성별·학습법·상호작용을 평가할 설계를 선택하고 효과별 H0/H1·F·자유도·p-value·기각 여부를 제시하시오. 반복측정을 반영하고 within 효과에는 Greenhouse-Geisser 보정을 적용하시오.
ctx.run('3-1', 'mixed_anova')

# 3-2: 학습법별 차이를 같은 학생끼리 후속 비교하고 세 쌍의 원/보정 p-value·가설·기각 여부를 제시하시오. 상호작용이 유의할 때 주효과 후속 비교만으로 충분한지 설명하시오.
ctx.run('3-2', 'mixed_posthoc')

# 3-3: 독립 학생의 score를 study·failures·absences·support·internet으로 설명하시오. 전체 검정·계수·잔차 진단과 설명력, 단위가 다른 계수를 크기만으로 비교하는 문제를 설명하시오.
ctx.run('3-3', 'score_ols')
