# Recipe·Profile 설정 관리

실행 설정은 `src/Recipes/*.json`, 기법별 설정 계약은 해당 모듈의 Profile, 실행 결과는 전용 DTO에 둡니다. 현재 전체 JSON 294개이며, 등록 모듈의 기본 모델·목적 로드 조합은 282개입니다. 차이는 추가 RF/LightGBM 튜닝 레시피처럼 사용자 경로로 선택하는 파일 때문입니다. 모든 기능은 [전체 카탈로그](ANALYSIS_CATALOG.md), 각 단계의 연결은 [파이프라인](ANALYSIS_PIPELINE.md)을 참고하세요.

## 설정을 선택하는 두 방식

| 방식 | 처리 | 사용 상황 |
| --- | --- | --- |
| Profile 생략 | 기본 또는 사용자 JSON 읽기→overrides 적용→기법별 Profile 구성 | 반복 설정과 일회성 실험을 분리 |
| 명시적 Profile | 전달한 객체 그대로 기법별 runner 검증에 사용; JSON 읽지 않음 | 코드에서 타입 있는 설정을 직접 관리 |

```python
from src.Recipes import load_profile, load_recipe
from src.stats.regression.ols import run_ols

# 학습 자료 X_train/y_train은 별도로 준비합니다.
result = run_ols(X_train, y_train, overrides={'cov_type': 'HC3'})
profile = load_profile('regression-ols', overrides={'cov_type': 'HC3'})
same_result = run_ols(X_train, y_train, profile=profile)
raw_settings = load_recipe('regression-ols')
```

명시적 Profile과 `recipe_path`·`task`·`overrides`는 동시에 지정할 수 없습니다. 이미 만든 Profile의 출처 설정과 일회성 재정의를 호출 시 섞지 않습니다.

## JSON·기본값·우선순위

JSON은 Profile 필드 이름을 키로 갖는 평면 객체입니다. Recipe는 Pipeline 전체 구조나 관측값을 담는 전역 설정이 아닙니다. 전처리·모델·탐색·평가·시각화의 설정은 각 기법별로 로드하고 분석 코드에서 조합합니다.

1. 등록 키와 선택 task로 기본 파일을 정합니다. `recipe_path`가 있으면 그 파일을 대신 읽습니다.
2. JSON과 overrides 각각의 별칭·목록을 정규화하고 알 수 없는 키를 검사합니다.
3. overrides가 JSON의 같은 필드보다 우선합니다.
4. 둘 다 없는 필드만 dataclass의 문서화된 기본값을 사용합니다.
5. runner/factory가 Profile 타입·필드 값·입력 자료의 조건을 검사합니다.

기본 실행 설정은 JSON 값이며 dataclass 기본값과 항상 같다고 가정하지 않습니다. 로더는 파일을 캐시하거나 수정하지 않습니다. 기본 파일은 loader 위치 기준이고 사용자 recipe_path의 상대 경로는 호출 작업 디렉터리 기준입니다.

## task와 파일 선택

| 예 | 파일·계약 |
| --- | --- |
| `load_profile('xgboost', task='classification')` | `xgboost-classification.json` |
| `load_profile('xgboost', task='regression')` | `xgboost-regression.json` |
| `load_profile('regression-ols')` | `regression-ols.json`; task 인자 없음 |
| `load_profile('metric-regression-r2')` | `metric-regression-r2.json`; 평가 분야가 키에 포함됨 |
| `load_profile('optimization-grid-search', task='regression')` | 회귀 탐색용 기본 JSON; estimator에 맞는 search_space 필요 |

지원 task는 등록부 기준입니다. JSON의 task 또는 overrides로 호출에서 선택한 task를 바꿀 수 없습니다. 분류→회귀 변경은 `task='regression'`으로 명시합니다. 통계·그림·지표처럼 task를 받지 않는 키에는 인자를 추가하지 않습니다.

사용자 파일은 `load_profile('xgboost', task='regression', recipe_path='recipes/my-xgboost.json')`처럼 전달합니다. 기본 파일을 먼저 읽고 사용자 JSON을 덧씌우는 방식이 아니라 선택 파일 하나를 읽습니다.

## 호환 정규화와 오류

- XGBoost `alpha`→`reg_alpha`, `lambda`→`reg_lambda` 별칭을 지원합니다. 같은 계층에 별칭/정식 키가 동시에 있으면 오류입니다. overrides의 정식 키로 기본 JSON의 별칭 값을 변경할 수 있습니다.
- hidden_layer_sizes·시퀀스 target_columns·conv_filters 등 해당 설정의 JSON 목록은 Profile 계약에 맞춰 tuple로 정규화합니다. Feature 전처리의 target_columns는 List[str]를 유지합니다.
- 파일 누락, 중복 JSON 키, 비객체 JSON, 비유한 JSON 상수, 알 수 없는 필드, 지원하지 않는 task, task 불일치는 오류입니다. 잘못된 키를 overrides로 덮어 오류를 감추지 않습니다.
- Profile만 로드해도 모든 분석 가정을 검증하는 것은 아닙니다. 표본 수·값 정의역·열·기간·이웃 조건은 실행 시 검증합니다.

## 결과 기록과 환경

적용 설정은 기법별 DTO의 `parameters` 등 문서화된 필드와 필요시 별도 실행 명세에 남깁니다. 모든 DTO에 parameters가 있다고 가정하지 않습니다. 데이터 출처·분할·FeatureSpec·후보 공간·선택 이력도 함께 보관해야 실험을 재현할 수 있습니다.

`to_json()`은 결과 저장이며 학습 객체 복원이나 Recipe 생성 기능이 아닙니다. 선택 의존성과 Python별 설치 파일은 저장소 [실행 환경](../README.md#목적별-실행-환경)을 따릅니다. Optuna 실행에는 해당 환경의 optimization requirements가 필요하고, 전체 모의고사 Python3.12 requirements에도 Optuna가 포함됩니다.
