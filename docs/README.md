# 문서 안내

[전체 분석 파이프라인](ANALYSIS_PIPELINE.md)에서 Dataset·Feature·Recipe·Profile·모형·DTO·평가의 관계를 먼저 확인하고, [전체 모듈 기능 카탈로그](ANALYSIS_CATALOG.md)에서 단계별 제공 기능을 찾으세요. 함수 내부 리뷰와 변경 이력보다 입력·제공 기능·산출물·사용 조건을 우선합니다.

## 기능별 상세 명세

| 목적 | 기준 문서 | 다루는 범위 |
| --- | --- | --- |
| 모듈·자료 읽기 | [가독성 기준](READABILITY.md) | 리팩터링 책임·도메인 사전·문제/풀이/해설 구성·생성/검증 |
| 전체 분석 설계 | [파이프라인](ANALYSIS_PIPELINE.md) | 분할·학습/CV·Feature·sampling·예측 후 처리·DTO 연결·목적별 분기 |
| 모듈 명세 기준 | [전수 명세 기준](MODULE_SPECIFICATIONS.md) | 전체 파일 분류·Python 헤더·생성 코드·CI 검사 |
| 파일 데이터 로딩 | [데이터 로딩](DATA_LOADING.md) | 지원 형식·입력/출력·스키마·예제·선택 의존성 |
| 전체 모듈 선택 | [기능 카탈로그](ANALYSIS_CATALOG.md) | 등록 모듈268개의 기능·입력·대표 결과, 로딩 및 기존 10단계 분류 |
| 설정 재현 | [레시피](RECIPES.md) | JSON·Profile·overrides·task·설정 오류·설정과 결과의 구분 |
| Feature·상관 | [전처리·상관](PREPROCESSING_CORRELATION_MODULES.md) | 수치/범주 전처리·RoPE·FeatureSpec·상관행렬 |
| 모형 후보 비교 | [튜닝](PARAMETER_OPTIMIZATION.md) | Grid·Random·Optuna·IID/시간/그룹 CV·재적합 |
| 회귀·분포 조건 | [회귀·분포](REGRESSION_DISTRIBUTION_MODULES.md) | OLS·가정·VIF/Cook·전처리·AIC/BIC·FGLS·GLM·비선형·분포별 NB/QQ·회귀 그림 |
| 통계 검정·확률 | [통계](STATISTICS_COMPLETE.md) | 평균/비율/분산·비모수·범주·다변량·ANOVA·사후검정·확률분포 |
| 생존·시간·표본 보완 | [도메인 상세](DOMAIN_ARCHITECTURE.md) | KM/RMST/logrank/Cox·ARIMA/SARIMA·비선형·12개 sampler의 API/자료형 |
| 딥러닝·연관규칙 | [딥러닝·연관](DEEP_ASSOCIATION_MODULES.md) | Dense·CNN·순환·시간 윈도, Apriori 계열·FP-Growth·Eclat의 계약 |
| 예측 평가 | [평가](EVALUATION_METRICS.md) | 분류·회귀·군집·시간 지표와 모형 비교·fold 요약 |
| EDA·의사결정 그림 | [Axes 시각화](EDA_VISUALIZATION_MODULES.md) | 원자료·지표를 호출자 ax로 조합하는 입력/표현 계약 |
| 목적함수·생산 배분 | [유전 알고리즘](GENETIC_OPTIMIZATION.md) | 혼합 정수 GA·제약·생산 배분·독립 검증·수렴 그림 |
| 실제 데이터 연결 | [UCI skeleton](UCI_SKELETONS.md) | 출처·학습 경계·분야별 예제·실행 환경 |
| 정확한 필드 조회 | [핵심 API 참조](reference/CORE_API.md) | 초기 핵심 모델·기초 검정·TF의 Profile/DTO 전체 필드; 그 밖의 필드는 각 분야 명세 |

## 개발·검증·학습 자료 관리

| 목적 | 문서 |
| --- | --- |
| 새 모듈 작성 규약 | [7개 섹션·메타데이터·필드·함수 명세](MODEL_MODULE_FORMAT.md) |
| 검증 근거·이력 | [실행 검증 기록](VALIDATION.md), [도메인 수치/API 검증](DOMAIN_VALIDATION.json), [기존 답안 비교](DOMAIN_ANSWER_REGRESSION.json) |
| 기법별 필드의 기계 참조 | [도메인19개 API 스냅샷](DOMAIN_EXTENSION_API.json); 전체 등록부를 대신하지 않음 |
| 전체 학습 자료 | [통합 목록](../기출문제/전체목록.md), [종합 모의1~20회](../기출문제/모의고사/README.md) |
| 문제 설계·출처·공문서 서식 | [상황형 설계](SCENARIO_EXAM_DESIGN.md), [자료 대조](EXAM_RECONSTRUCTION_AUDIT.md), [17~23회 출처](ADP_17_23_SOURCE_AUDIT.md), [서식](QUESTION_PAPER_STYLE.md) |
| 원 스켈레톤 근거 | [참고 XGBoost](reference-skeleton/xgboost.py); 실행 기능은 src의 현행 모듈 기준 |

## 문서 통합 원칙과 이번 정리

- 통계 선택·ANOVA 안내는 통계 명세에, TensorFlow 공통 안내는 딥러닝·연관 명세에 통합했습니다.
- 응용 회귀 확장의 추가 명세는 회귀·분포 명세에 통합했습니다. 개별 모듈의 필드·조건과 참조 근거는 유지합니다.
- 이전 분석 카탈로그의 긴 필드 나열은 핵심 API 참조로 분리하고, 전체 카탈로그는 268개 기능을 단계별로 한 번씩 싣습니다.
- 정식 구현으로 대체된 제안 데모와 구현 전 통계 사전분석은 삭제했습니다. 최종 지원 범위·원문 오류 교정·대응표·원 참조 skeleton·검증 증빙은 남깁니다.
- 특정 시점의 검증 JSON은 당시의 증빙입니다. 현재 전체 모듈 수와 기능 범위는 등록부 기반 카탈로그를 기준으로 합니다.

기능 추가 시 등록부와 소스를 반영한 뒤 `python tools/docs/build_module_catalog.py`로 목록을 갱신합니다. `python tools/docs/build_module_catalog.py --check`는 생성 결과와 등록부의 정합성을 검사합니다. 분야 명세에는 해당 기능만 설명하고 전체 공통 흐름을 복제하지 않습니다.
