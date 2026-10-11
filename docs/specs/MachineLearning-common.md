# `src/MachineLearning/common.py`

[소스 코드](../../src/MachineLearning/common.py) · [반환 DTO 계약](../OUTPUT_DTOS.md)

## 파일 개요

================================================================================
common.py - 지도 학습의 데이터 분할, Pipeline, 교차검증 및 평가
================================================================================
설명:
    - 01장 데이터 분할·스케일링과 03/06/11/14장 Pipeline·모델 선택을 공통화합니다.
    - 테스트 데이터는 튜닝에 사용하지 않고, 스케일러는 학습 데이터에서만 적합합니다.
작성 정보:
    - 작성일: 2026-10-10
    - 버전: 1.0.0
변경 이력:
    2026-10-10  v1.0.0  Profile/Result와 공통 실행기 추가
================================================================================
Output DTO 갱신: 2026-10-11 v1.1.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가

## `SupervisedProfile`

공통 학습 조건.

Attributes:
    task: 'regression' 또는 'classification'. 단일 타깃만 지원합니다.
    test_size: 전체 표본 중 테스트 비율 (0 < test_size < 1).
    random_state: 분할·교차검증에 사용할 정수 시드.
    scaling: 'none', 'standard', 'minmax' 중 하나.
    stratify: 분류에서 클래스 비율을 유지할지 여부. 회귀에서는 무시합니다.
    param_grid: 선택적 탐색 공간. 모델 인자는 'alpha' 또는 'model__alpha'로 지정합니다.
    cv: param_grid가 있을 때 학습 데이터의 교차검증 폴드 수 (최소 2).
    scoring: sklearn 평가 이름. None이면 회귀는 neg_mean_squared_error, 분류는 accuracy.
    n_jobs: GridSearchCV 병렬 작업 수. 기본 1로 시험 환경 자원 사용을 제한합니다.

## `SupervisedOutputDTO`

지도 학습 결과.

Attributes:
    model_name: 실제 추정기 클래스 이름.
    task: 회귀 또는 분류.
    model: 적합된 Pipeline. 새 데이터의 predict에도 동일 전처리를 적용합니다.
    train_metrics: 학습 데이터 지표 (일반화 성능 지표가 아님).
    test_metrics: 보류한 테스트 데이터 지표.
    y_test: 테스트 정답, 원래 입력 순서가 아닌 test_indices 순서.
    predictions: 테스트 예측값, y_test와 같은 순서.
    train_indices: 입력 데이터의 학습 행 위치.
    test_indices: 입력 데이터의 테스트 행 위치.
    feature_names: DataFrame 열 이름 또는 자동 생성된 이름 (입력 순서).
    classes: 분류 클래스 순서. 회귀는 None.
    probabilities: predict_proba가 있는 분류 모델의 테스트 확률. 나머지는 None.
    confusion: classes 순서의 혼동행렬. 회귀는 None.
    best_params: 탐색을 수행한 경우 Pipeline 파라미터 이름의 최적 설정.
    best_cv_score: 학습 데이터 CV 최적 점수. 음의 MSE는 클수록 좋습니다.
    cv_results: 탐색 결과 사전. 탐색을 생략하면 None.

## `validate_features`

수치형 2차원 특성 행렬을 검증합니다.

Args:
    X: 배열 또는 DataFrame. 결측값은 호출 전에 처리해야 합니다.
    min_samples: 필요한 최소 행 수.
Returns:
    np.ndarray: 유한한 float 행렬.
Raises:
    ValueError: 비수치 데이터, NaN/무한대, 빈 열 또는 잘못된 차원·행 수.

## `make_scaler`

스케일러를 생성합니다.

Args:
    scaling: 'none', 'standard', 'minmax'.
Returns:
    Any: 미적합 스케일러 또는 'passthrough'.
Raises:
    ValueError: 지원하지 않는 스케일링 이름.

## `run_supervised`

분할 → Pipeline 적합/학습 데이터 CV → 테스트 평가를 수행합니다.

Args:
    X: 결측치 없는 수치형 2차원 특성. DataFrame 열 순서는 예측 시에도 유지합니다.
    y: 길이가 X와 같은 1차원 타깃. 분류는 일관된 문자열 또는 숫자 라벨.
    estimator: profile.task에 맞는 sklearn 추정기 (각 모델 실행기가 선택).
    profile: 공통 학습 조건 및 모델별 설정.
    extra_steps: 스케일러 뒤, 모델 앞에 배치할 미적합 변환기. 예: ('pca', PCA()).
Returns:
    SupervisedOutputDTO: 학습 Pipeline, 평가 지표, 분할 위치 및 선택적 CV 결과.
Raises:
    ValueError: 잘못된 입력·설정, 표본/클래스 수 부족 또는 유효하지 않은 모델 인자.

Notes:
    무작위 IID 분할입니다. 시계열·그룹 데이터는 이 실행기 대신 별도 분할이 필요합니다.
    분류 ROC-AUC는 이진 타깃과 predict_proba가 있고 테스트에 두 클래스가 있을 때만 반환합니다.
    양성 클래스는 classes[1]입니다. SVC의 probability=False에서는 AUC를 반환하지 않습니다.
