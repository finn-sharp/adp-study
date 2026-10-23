# ADP 분석 모듈과 학습 자료

ADP 실기 학습을 위한 재사용 분석 모듈과 문제지·실행 풀이를 관리합니다. 현재 표준 모듈은 **266개**, 모델·목적별 JSON은 **293개**이며 학습 자료는 **41묶음·393문항·434개 코드 셀·145개 그림·118쪽 문제지**입니다.

- [전체 분석 파이프라인](docs/ANALYSIS_PIPELINE.md): Dataset→Feature→학습/CV 전처리·sampling→모형→예측 후 처리→평가, Recipe/Profile/DTO의 역할
- [전체 모듈 기능 카탈로그](docs/ANALYSIS_CATALOG.md): 266개 모듈의 단계별 기능·입력·대표 결과
- [문서 안내](docs/README.md): 분야별 상세 명세·API·검증·학습 자료의 기준 문서
- [전체 학습 자료](기출문제/전체목록.md), [응용 모의10회](기출문제/모의고사/응용10회/README.md): 회귀·분류·시간·sampling·튜닝을 적용한 문제와 실행 풀이

## 빠른 시작

아래 명령은 저장소 루트의 bash/zsh에서 실행합니다. 먼저 Python 3.12의 별도 환경에서 생존분석·시계열·비선형 회귀·불균형 처리 예제를 실행할 수 있습니다.

```bash
python3.12 -m venv venv/domain
source venv/domain/bin/activate
python -m pip install -r requirements-domain-py312.txt

# 고정 예제 데이터 → JSON 레시피 → 실행기 → 전용 DTO
python -m src.stats.skeleton.kaplan_meier
python -m src.TimeSeries.skeleton.arima
python -m src.MachineLearning.skeleton.nonlinear_regression

# 신규 도메인의 수치·입력·결과 계약 검증
python -m unittest discover -s tests -p 'test_domain_extension.py' -v
```

Python에서 직접 호출할 때도 수식을 다시 정의할 필요 없이 데이터와 설정을 전달합니다.

```python
from src.stats.survival.kaplan_meier import run_kaplan_meier

result = run_kaplan_meier(
    durations=[1, 2, 2, 3, 4],
    event_observed=[1, 1, 0, 0, 1],  # 사건=1, 우측 중도절단=0
    overrides={"label": "공급사 A", "evaluation_times": [0, 1, 2, 3, 4]},
)

print(result.survival_at_times)     # [1.0, 0.8, 0.6, 0.6, 0.0]
print(result.median_survival_time)  # 4.0
print(result.to_json(indent=2))
```

## 분석 도메인

모듈 수는 `src/Recipes/loader.py`의 등록 항목 기준입니다. 한 모듈이 여러 분석 목적을 지원할 수 있어 모듈 수와 레시피 수는 다릅니다.

| 도메인 | 모듈 | 주요 구현 | 경로 |
| --- | ---: | --- | --- |
| 통계 | 72 | 분포 16종, 평균·비율·분산·비모수 검정, ANOVA/MANOVA·사후검정, 진단·상관분석, 생존분석 | `src/stats/` |
| 머신러닝 | 37 | 선형·비선형 회귀, Naive Bayes, 부스팅·신경망, 군집·차원 축소, 변수 선택·최적화 | `src/MachineLearning/` |
| 전처리·표본 보완 | 23 | 수치·범주형 결측·스케일·변환·인코딩, Feature 파이프라인·RoPE, 리샘플링 12종 | `src/Preprocessing/` |
| 평가 | 56 | 분류·확률·Lift/Gain·회귀·군집·시계열 지표, 분야별 묶음·모델 비교·CV 요약 | `src/Evaluation/` |
| 시각화 | 61 | 분포·결측·범주·관계·군집·축소·분류·연관·연결망·시간 패턴 | `src/Visualization/` |
| 딥러닝 | 8 | TensorFlow Dense·CNN·SimpleRNN·LSTM·GRU·시계열 예측 | `src/DeepLearning/` |
| 시계열 | 3 | ARIMA·SARIMA, 학습 AIC 차수 선택·예측구간·외생변수 검증 | `src/TimeSeries/` |
| 연관규칙 | 3 | Apriori·FP-Growth·Eclat | `src/Association/` |
| 사회연결망 | 1 | 중심성·PageRank·커뮤니티 | `src/Networks/` |
| 최적화 | 2 | 정수·실수 GA, 직렬 생산라인 설비 배분·독립 전수검증 | `src/Optimization/` |
| **합계** | **266** | | |

최근 도메인 확장에는 다음 19개 모듈과 대응 JSON·스켈레톤을 각각 추가했습니다.

| 영역 | 제공 내용 |
| --- | --- |
| 생존분석 4개 | Kaplan–Meier·Greenwood 점별 신뢰구간·중앙수명, RMST, 다집단 로그순위·쌍별 Holm, Cox 회귀·HR/CI·생존예측·PH 진단 |
| 시계열 2개 | ARIMA·SARIMA, 학습에서만 차수 선택, 수렴·실패 기록, 미래 외생변수의 크기·열·날짜 정렬 확인 |
| 비선형 회귀 1개 | exponential·logistic·Michaelis–Menten 평균모형, 초기값·경계·근사 공분산, 학습과 보류 평가 분리 |
| 불균형 처리 12개 | RandomOver·SMOTE·ADASYN·BorderlineSMOTE·SVMSMOTE·SMOTENC·SMOTEN, RandomUnder·TomekLinks·ENN·NearMiss·ClusterCentroids |

수치 거리 기반 리샘플링을 범주 코드에 일괄 적용하지 않습니다. SMOTENC는 혼합 자료, SMOTEN은 전체 범주 자료에 사용합니다. 자료형별 지원 범위와 CV 사용법은 [도메인 명세](docs/DOMAIN_ARCHITECTURE.md)를 참고하세요.

## 아키텍처와 모듈 규약

공통 실행 계약은 **기법별 JSON → Profile → `run_*()` → 전용 Output DTO**입니다. Profile을 직접 전달하는 호출도 지원합니다.

| 구성 | 책임 |
| --- | --- |
| `src/Recipes/` | 기본·사용자 JSON을 읽고 Profile을 구성합니다. 설정 우선순위는 JSON → `overrides`입니다. |
| 각 분석 도메인 | 입력·가정·설정을 검증하고 계산·학습·예측을 수행합니다. |
| `src/output.py` | 결과 DTO의 필드 접근과 사전·JSON 변환 규약을 제공합니다. |
| 각 도메인의 `skeleton/` | `load_data`·`load_recipe`·`build_pipeline`·`main`으로 실행 흐름을 보여줍니다. |
| `기출문제/` | 문항별 데이터 결합·중간 상태·표·그림·가설 보고·업무 해석을 관리합니다. |
| `tools/exams/` | 문항·문제지·해설 생성, 실제 코드 실행, 수치·출처·레이아웃 대조와 통합 집계를 담당합니다. |
| `tests/`, `.github/workflows/` | 참조값·경계값·자료 분리·호환성·실제 실행을 검사합니다. |

모든 표준 분석 모듈은 작성 정보·변경 이력과 다음 7개 섹션을 유지합니다.

1. Imports
2. Input Profile
3. Output DTO
4. Input Validation
5. Individual Model Runners
6. Public Runner
7. Example Execution

Profile·DTO의 필드에는 타입·기본값 또는 반환 조건·배열 대응 관계를, 함수에는 `Args`·`Returns`·`Raises`를 명시합니다. 명시적 Profile과 `recipe_path`·`overrides`를 동시에 전달하면 오류입니다. 적합 모델·변환기·sampler 등 실행 객체는 DTO 속성으로 접근하며 기본 JSON에서 제외됩니다. JSON은 결과 저장용으로, 적합 모델의 복원 형식이 아닙니다.

기존 Z/t 검정의 import·CLI 호환 경로도 유지합니다. 상세 작성 규칙은 [모듈 컨벤션](docs/MODEL_MODULE_FORMAT.md)에 있습니다.

## 기출 유형과 모의고사

| 자료 | 구성 | 문항 | 안내 |
| --- | --- | ---: | --- |
| [기존 기출·모의](기출문제/README.md) | 15·17·18·19·20·21·22·23·24·25·26·28회, 보충 자료, 모의 1~4회 — 22묶음 | 162 | 제공 자료·출제 후기·보완 조건을 대조한 상황형 학습 자료 |
| [생존분석 통계 모의](기출문제/모의고사/생존분석_통계/README.md) | 3개 대문항, 권장 90분, 연습 배점 60점 | 14 | 기본 출제 유형과 RMST·Cox 등의 확장 연습을 구분 |
| [30~36회 유형 연습](기출문제/유형연습/README.md) | 공개 후기에서 확인한 유형을 정제한 7묶음 | 53 | 32회는 통계 영역, 35회는 공간 전처리 영역의 부분 연습 |
| [생산 모의5회](기출문제/모의고사/5회/README.md) | 100점 종합1묶음 | 14 | 생산 최적화·독립 검증·표본 추론 |
| [응용 모의10회](기출문제/모의고사/응용10회/README.md) | 각100점 종합10묶음 | 150 | 조건별 회귀·샘플링 분류·rolling 시계열 |
| **합계** | **41묶음** | **393** | **434개 코드 셀·145개 그림·118쪽 문제지** |

각 묶음에는 공문서형 A4 문제지 PDF, 별도 실행 풀이·해설 노트북, 데이터, 실제 답안 수치와 검증 기록이 있습니다. 30~36회에는 `분석코드.py`와 `해설.md`도 제공합니다.

검정 문항은 **방법 선택 근거·가정 → H0/H1 → 통계량·p-value → 기각 여부 → 상황에 맞는 결론**을 다룹니다. 확률 계산·추정·최적화에는 불필요한 p-value를 붙이지 않습니다.

자료는 공식 원문·실제 시험 데이터·공식 정답의 완전 복원을 의미하지 않습니다. 원 자료, 후기에서 확인한 유형, 새 조건과 합성 데이터를 출처 명세에서 구분합니다. Cox 등 확장 연습을 실제 기출 출제 확인으로 표시하지 않습니다.

SARIMA·KM, 최근 회차의 표본 보완·ARMA, 생존분석 모의의 주요 계산은 공통 API에 연결했습니다. 기존 풀이에는 sklearn·SciPy·statsmodels 직접 호출도 남아 있습니다. [문항별 참조표](기출문제/전체목록.json)는 정적 참조를 기록하며 모든 계산의 이관 완료나 실행 커버리지를 뜻하지 않습니다.

## 목적별 실행 환경

의존성은 용도와 Python 버전별로 분리합니다. TensorFlow 환경은 NumPy 핀이 달라 일반 분석 환경과 별도로 설치합니다.

| 목적 | Python 3.12 | Python 3.7 |
| --- | --- | --- |
| 일반 분석·모델 | `requirements-models-py312.txt` | `requirements-models-py37.txt` |
| Grid·Random·Optuna 최적화 추가 | `requirements-optimization-py312.txt` | `requirements-optimization-py37.txt` |
| 생존·시계열·비선형·표본 보완 | `requirements-domain-py312.txt` | `requirements-domain-py37.txt` |
| TensorFlow | `requirements-tensorflow-py312.txt` | `requirements-tensorflow-py37.txt` |
| 기존 기출·모의 실행 | `기출문제/requirements-py312.txt` | 별도 호환 안내: `기출문제/requirements-py37.txt` |
| 30~36회 / 생존분석 모의 | 각 폴더의 `requirements-py312.txt` | Python 3.7 런타임 호환을 주장하지 않음 |

일반 모델·최적화·신규 도메인을 함께 사용할 Python 3.12 환경의 설치와 검증 예시입니다.

```bash
python -m pip install -r requirements-models-py312.txt \
    -r requirements-optimization-py312.txt -r requirements-domain-py312.txt
python -m unittest discover -s tests -v
```

Python 3.7 일반 모델 환경은 HDBSCAN 빌드 전에 다음 의존성을 준비합니다.

```bash
python -m pip install numpy==1.21.6 Cython==0.29.37 wheel==0.38.4
python -m pip install --no-build-isolation -r requirements-models-py37.txt
```

Python 3.7은 별도 호환 환경으로 관리하며 실제 검증 버전은 3.7.17입니다. Apple Silicon에서 이 환경이 필요하면 x86_64용 별도 환경을 준비하세요. 저장소의 실행 검증과 실제 시험 서버의 Python 3.7.4·제공 패키지는 구분합니다. 루트의 `requirements.txt`는 기존 학습용 설정이며 전체 표준 모듈의 설치 명세가 아닙니다.

노트북은 `src`·공통 코드·데이터를 함께 사용하므로 저장소 전체를 내려받아 실행합니다. 회차별 설치·실행·재생성 명령은 각 README를 따릅니다.

## 검증과 재생성

2026-10-20 도메인 통합 기준으로 다음을 확인했습니다.

- Python 3.7·3.12에서 신규 모듈 19개와 스켈레톤 19개를 실제 실행했습니다.
- 로컬 Python 3.12 전체 테스트 146개를 통과했습니다. 선택 TensorFlow 관련 검사 등 7개는 건너뛰었으며 TensorFlow는 별도 CI에서 검사했습니다.
- 전체 30묶음·259개 코드 셀을 재실행하고 문항·문제지·가설 보고·독립 수치를 대조했습니다.
- 이전 main과 답안 JSON 30개·3,377개 수치를 비교해 허용오차 내 차이가 없음을 확인했습니다.
- 통합 PR의 최종 커밋에서 분석·TensorFlow·시험·도메인·집계 CI가 모두 통과했습니다. 이후 커밋의 상태는 [GitHub Actions](https://github.com/finn-sharp/adp-study/actions)에서 확인합니다.

등록부·문서 수치와 전체 목록의 일치 여부를 검사하거나 학습 목록을 재생성할 수 있습니다.

```bash
# Python 3.12 환경에서 집계 도구 의존성을 준비
python -m pip install nbformat==5.10.4
python tools/exams/audit_catalog.py
python tools/exams/build_catalog.py

# 신규 모듈 19개 + 스켈레톤 19개의 독립 실행
python tools/check_domain_examples.py

# 회귀 비교: 대상 git ref가 로컬에 있어야 함
python tools/exams/compare_answers.py --base 22d39e6e44074db666a528149a9e10120e58227a
```

생성기는 실행 출력을 초기화할 수 있습니다. 문제지·노트북을 재생성한 뒤에는 해당 묶음의 실제 실행과 검사를 다시 수행합니다. 기록은 [도메인 검증](docs/DOMAIN_VALIDATION.json), [답안 회귀 비교](docs/DOMAIN_ANSWER_REGRESSION.json), [기존 검증 명세](docs/VALIDATION.md)에 있습니다.

## 상세 문서

[문서 안내](docs/README.md)에 기능별 기준 문서를 모았습니다. 분석 흐름은 [파이프라인](docs/ANALYSIS_PIPELINE.md), 구현 기능은 [카탈로그](docs/ANALYSIS_CATALOG.md), 설정은 [Recipe/Profile](docs/RECIPES.md), 새 모듈 작성은 [개발 컨벤션](docs/MODEL_MODULE_FORMAT.md)을 기준으로 확인합니다.

## 분석·해석의 경계

지도 학습과 변수 선택의 기본 실행기는 IID 분할을 사용하며 시간·그룹 자료에는 적절한 별도 분할이 필요합니다. 전처리·변수 선택·리샘플링은 학습 구간 또는 CV 학습 폴드 안에서 적합합니다. CV 최고점과 최종 보류 성능을 구분합니다.

군집·차원 축소는 탐색 분석입니다. t-SNE/MDS와 현재 밀도 군집 실행기는 새 관측의 `predict`/`transform`을 제공하지 않습니다. L2 규제 자체는 희소화를 보장하지 않으며, SNA는 사회연결망 분석입니다.

생존분석은 독립 관측·비정보성 우측 중도절단을 가정합니다. KM 신뢰구간은 점별 구간이고, HR은 순간 위험비입니다. 검정 비기각은 동일성이나 가정의 증명이 아니며, 합성 자료의 결과를 실제 현장의 인과 효과로 해석하지 않습니다.
