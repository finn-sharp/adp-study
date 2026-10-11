# `src/Preprocessing/sampling.py`

[소스 코드](../../src/Preprocessing/sampling.py) · [반환 DTO 계약](../OUTPUT_DTOS.md)

## 파일 개요

================================================================================
sampling.py - imbalanced-learn 증강 5종·감소 5종 및 데이터 유형 명세
================================================================================
설명:
    - classification 이산 타깃의 불균형 리샘플링을 제공합니다. annotation 라벨 생성은 하지 않습니다.
    - 수치형(numeric), 범주형(categorical), 혼합형(mixed)을 명시적으로 구분합니다.
    - 증강: RandomOverSampler, SMOTE, ADASYN, SMOTENC, SMOTEN.
    - 감소: RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours, ClusterCentroids.
    - 범주형/혼합형 감소는 RandomUnderSampler만 직접 지원합니다. 다른 기법은 수치 거리 기반입니다.
사용 예제:
    - python -m examples.extended_workflow
    - docs/EXTENSIONS.md의 import/Profile/run 호출 예제를 참조합니다.
작성 정보: 작성일 2026-10-11, 버전 1.1.0
변경 이력:
    2026-10-11 v1.0.0 유형 검증·분포 보고·학습 폴드 Pipeline 추가
    2026-10-11 v1.1.0 전체 10종 Python 3.7 실행 호환 환경 명세 갱신
================================================================================
Output DTO 갱신: 2026-10-11 v1.2.0 명시적 DTO·사전/JSON 변환 및 Result 호환 별칭 추가

## `SamplingProfile`

리샘플링 설정.

Attributes:
    method: OVER_METHODS 또는 UNDER_METHODS의 이름.
    data_kind: numeric, categorical, mixed. 타깃이 아닌 특성의 의미적 유형.
    categorical_features: mixed의 범주형 열 위치 목록. SMOTENC 필수, 문자열 열 이름은 미지원.
    sampling_strategy: imblearn의 str/dict/float/callable. float는 이진 분류만 지원.
    random_state: 재현용 정수 시드.
    neighbors: SMOTE/SMOTENC/SMOTEN k_neighbors, ADASYN/NearMiss/ENN n_neighbors.
    parameters: 기법별 추가 인자. 핵심 필드와 중복 지정하면 오류.

## `SamplingOutputDTO`

리샘플링 결과와 클래스별 분포.

Attributes:
    X: 증강/감소된 특성. imblearn이 가능한 경우 입력 DataFrame 유형 유지.
    y: 결과 이산 타깃.
    sampler: fit_resample 완료된 imblearn 객체.
    counts_before: 원래 클래스별 표본 수.
    counts_after: 결과 클래스별 표본 수. ADASYN/정제 기법은 정확한 균형을 보장하지 않음.
    sample_indices: 원본 행 선택 기법에서만 제공. 합성 표본의 출처 행을 뜻하지 않음.

## `make_sampler`

유형과 설정을 검증하여 미적합 sampler를 생성합니다.

Args:
    profile: 기법·유형·샘플링 목표·이웃 수·기법별 인자.
Returns:
    Any: fit_resample 가능한 imblearn 객체; 학습이나 데이터 변환은 하지 않음.
Raises:
    ValueError: 지원하지 않는 조합 또는 핵심 파라미터 중복.
    ImportError: imblearn 미설치 또는 SMOTEN 미지원 버전.
Notes:
    SMOTEN은 imblearn >=0.8 필요. 전체 10종은 requirements-py37.txt의
    Python 3.7/sklearn 1.0.2/imblearn 0.8.1에서 검증했습니다. 기존 시험용
    sklearn 0.23/imblearn 0.7 조합에는 SMOTEN이 없습니다.

## `resample_training_data`

이미 분리한 학습 데이터만 리샘플링합니다.

Args:
    X: 결측/무한대 없는 2차원 학습 특성. 범주형 숫자 코드는 data_kind로 명시.
    y: 학습 특성과 같은 행 수, 최소 두 클래스의 1차원 이산 라벨.
    profile: 기법·유형·이웃 수·샘플링 목표.
Returns:
    SamplingOutputDTO: 변환 데이터, sampler, 클래스별 전후 표본 수 및 선택적 원본 위치.
Raises:
    ValueError: 입력/범주형 위치/기법 조건 오류 또는 이웃/소수 클래스 표본 수 부족.
    ImportError: 선택 의존성 미설치/미지원 버전.
Notes:
    전체 데이터를 먼저 리샘플링하면 누수가 발생합니다. CV에는 build_sampling_pipeline을
    사용하세요. 합성 표본을 독립 관측으로 간주한 통계적 회귀 추론에는 사용하지 않습니다.
    거리 기법은 특성 단위의 영향을 받습니다. 스케일링은 학습 데이터/폴드에서만 적합하세요.

## `build_sampling_pipeline`

GridSearchCV에서 학습 폴드에만 리샘플링하는 미적합 Pipeline을 생성합니다.

Args:
    estimator: fit/predict를 지원하는 미적합 분류 추정기.
    profile: sampler 설정. categorical_features는 before_sampling 출력 열 위치 기준.
    before_sampling: 선택적 미적합 변환기. 수치형 scaler 또는 ColumnTransformer 등.
    after_sampling: 선택적 미적합 변환기. 범주형 결과를 분류 모델 입력으로 인코딩할 때 사용.
Returns:
    Any: imblearn Pipeline; predict 시 sampler를 건너뜀. CV에는 이 전체 객체 전달.
Raises:
    ValueError: sampler 설정 오류. 실제 데이터의 상세 검증은 fit 시 imblearn이 수행.
    ImportError: imblearn 미설치.
Notes:
    SMOTENC/SMOTEN 전에 원핫 인코딩하면 유형 의미가 달라집니다. after_sampling에
    인코더를 두세요. numeric/categorical은 특성 유형이며 모든 타깃은 분류 라벨입니다.
    Pipeline의 fit_transform은 fit 후 transform과 동일하지 않을 수 있으므로 사용하지 않습니다.
