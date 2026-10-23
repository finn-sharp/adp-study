# 회귀·분류·시계열 응용 모의10회

**신규10회·150문항·각100점·160개 실행 코드 셀·50개 그림·40쪽 문제지**입니다. 기존1~5회·생존분석 모의와 별도 묶음으로 관리합니다. 회차별 회귀36점·분류40점·시계열24점, 권장180분입니다. 기존 공문서형 A4·나눔고딕·상황형 가정/가설/해석 계약을 유지합니다.

각 회차에 문제지PDF·실행 풀이ipynb·분석코드.py·해설.md·고정 합성CSV3개·실행설정·실제 수치 답안·검증 기록이 있습니다. 데이터와 상위 공통 분석 코드·저장소 src를 함께 사용하세요. 공식 기출/공식 정답/실제 기업 데이터가 아닙니다.

| 회차 | 주제 | 특별 회귀 조건 | 분류 보완 / 탐색 |
| --- | --- | --- | --- |
| [01회](01회/README.md) | 공동주택 가격과 계약 심사 | power | random_over / grid |
| [02회](02회/README.md) | 물류 배송비와 파손 위험 | lognormal | smote / optuna |
| [03회](03회/README.md) | 발전 출력과 장비 경보 | quadratic | random_under / grid |
| [04회](04회/README.md) | 공정 수율과 불량 검출 | heteroscedastic | smote / grid |
| [05회](05회/README.md) | 중고차 가격과 반품 위험 | influence | random_over / optuna |
| [06회](06회/README.md) | 에너지 사용량과 요금 미납 | ordered | random_under / grid |
| [07회](07회/README.md) | 촉매 반응량과 공정 실패 | exponential | smote / grid |
| [08회](08회/README.md) | 보험 지급액과 청구 위험 | gamma | random_over / optuna |
| [09회](09회/README.md) | 고객센터 문의 횟수와 이탈 | poisson | random_under / grid |
| [10회](10회/README.md) | 서비스 성장량과 전환 예측 | growth | smote / grid |

## 분석 흐름

1. 원행 학습/최종평가 분리 → 폴드 내부 Feature 전처리 → 기준 OLS·계수 t/전체F·R²/AIC/BIC.
2. 선형 사양·잔차 정규성·등분산·시간순 자기상관·VIF·Cook 진단 → 원반응 AIC/BIC 후진선택.
3. 조건별 변환·다항식·학습 분산 추정·명시 영향 행 처리·비선형/분포별 평균 모형을 CV R²로 비교 → 같은 보류 원단위 지표·ΔR²와 남은 가정 보고.
4. 자연 발생률 보류를 봉인하고 imblearn Pipeline 안에서만 샘플링·튜닝 → 별도 설명용 logit z/오즈비/LR·사양 → ROC/AUC·AP·Brier·고정 임계값 비용.
5. 과거 rolling 차수 선택 → AIC/BIC·자유도 보정 잔차 자기상관 → 동일 미래 선택/naive·RMSE/MAE/R²/MASE·조건부 구간.

새 공통 모듈8개와 비선형 CV factory는 [전체 명세](../../../docs/REGRESSION_DISTRIBUTION_MODULES.md)에 있습니다. [출처·생성 가정](출처및설계.md) 및 [검증 결과](검증결과.json)를 함께 읽으세요. 비기각·R² 증가·Cook flag·좋은 AUC만으로 가정 충족/인과/확률 보정/현장 성능을 증명하지 않습니다.

## 실행

```bash
python -m pip install -r 기출문제/모의고사/응용10회/requirements-py312.txt
python tools/exams/run_applied_mocks.py
python tools/exams/audit_applied_mocks.py
# 단일 회차의 독립 Python 코드
python 기출문제/모의고사/응용10회/01회/분석코드.py
# 문항/데이터/PDF 재생성 후 위 실행·검증을 다시 수행
python tools/exams/build_applied_mocks.py
```

콘텐츠 전체 수치 실행은 Python3.12입니다. 모듈3.7/3.12 호환검사와 콘텐츠 실행 환경을 구분합니다. 회차별 원단위 보류 ΔR²가 음수여도 선택을 바꾸지 않고 그대로 보고합니다.

## 실제 실행의 회귀 비교

아래는 이 합성 자료·고정 계획의 결과이며 개선이 보장된다는 뜻이 아닙니다. test로 후보를 다시 선택하지 않았습니다.

| 회차 | CV 선택 후보 | 기준 test R² | 선택 test R² |
| --- | --- | ---: | ---: |
| 01 | baseline | 0.93207 | 0.93207 |
| 02 | log_target | 0.53820 | 0.71281 |
| 03 | polynomial | -0.02063 | 0.97979 |
| 04 | fgls | 0.79228 | 0.78792 |
| 05 | baseline | 0.84278 | 0.84278 |
| 06 | aic | 0.72384 | 0.72723 |
| 07 | nonlinear | 0.94227 | 0.98897 |
| 08 | gamma | 0.32236 | 0.49054 |
| 09 | vif | 0.39745 | 0.37787 |
| 10 | nonlinear | 0.97507 | 0.99218 |
