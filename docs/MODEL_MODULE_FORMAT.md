# 모델 모듈 명세와 실행

여섯 모듈은 기존 통계 검정 파일의 작성 정보, 변경 이력, 메타데이터와
Attributes·Args·Returns·Raises 문서 형식을 따릅니다.
모델 파일 자체에 공통 입력과 출력 필드를 모두 명시합니다.

| 순서 | 섹션 | 내용 |
| --- | --- | --- |
| 파일 첫머리 | 모듈 명세 | 설명, 분석 가정, 작성 정보, 변경 이력 |
| 메타데이터 | 작성 정보 | `__author__`, `__email__`, `__version__`, `__status__` |
| 1 | Imports | 표준 라이브러리와 공통 계약 |
| 2 | Input Profile | 전체 입력 필드, 타입, 기본값, 제약 |
| 3 | Output DTO | 전체 출력 필드, 타입, 배열 순서와 조건 |
| 4 | Input Validation | 입력과 모델별 설정 검증 |
| 5 | Individual Model Runners | 추정기 생성 및 공통 학습 실행 |
| 6 | Public Runner | `run_모델명(X, y, profile)` |
| 7 | Example Execution | 실제 데이터 학습, 새 예측, DTO·JSON 변환 |

## 공개 API

| 파일 | Profile / Output DTO | 실행 함수 | 지원 목적 |
| --- | --- | --- | --- |
| `src/MachineLearning/Trees/xgboost.py` | `XGBoostProfile` / `XGBoostOutputDTO` | `run_xgboost` | 분류·회귀 |
| `src/MachineLearning/Trees/catboost.py` | `CatBoostProfile` / `CatBoostOutputDTO` | `run_catboost` | 분류·회귀 |
| `src/MachineLearning/Trees/lightgbm.py` | `LightGBMProfile` / `LightGBMOutputDTO` | `run_lightgbm` | 분류·회귀 |
| `src/MachineLearning/Trees/adaboost.py` | `AdaBoostProfile` / `AdaBoostOutputDTO` | `run_adaboost` | 분류·회귀 |
| `src/MachineLearning/NeuralNetworks/mlp.py` | `MLPProfile` / `MLPOutputDTO` | `run_mlp` | 분류·회귀 |
| `src/MachineLearning/NeuralNetworks/perceptron.py` | `PerceptronProfile` / `PerceptronOutputDTO` | `run_perceptron` | 분류 |

MCP 요청은 확인한 의미에 따라 MLP(다층 퍼셉트론)로 구현했습니다.

## 사용 예제

저장소 루트에서 실행합니다.

```python
from sklearn.datasets import load_iris
from src.MachineLearning.Trees.xgboost import XGBoostProfile, run_xgboost

X, y = load_iris(return_X_y=True)
profile = XGBoostProfile(
    task="classification",
    n_estimators=100,
    cv=3,
    param_grid={"max_depth": [2, 3]},
)
result = run_xgboost(X, y, profile)
print(result.test_metrics)
print(result.best_params)
print(result.model.predict(X[:3]))
print(result.to_json(indent=2))
```

학습/테스트는 무작위 IID 분할입니다. 시계열·그룹 자료에는 별도 분할이 필요합니다.
스케일러 적합과 GridSearchCV는 학습 구간에서만 수행합니다.
`model`은 전처리를 포함한 적합 Pipeline이며, JSON은 모델을 제외한 결과 저장용입니다.
`test_indices`는 입력 행의 위치이며 `y_test`, `predictions`, `probabilities`의 행 순서와 일치합니다.

입력은 결측값 없는 수치형 특성과 단일 타깃입니다. CatBoost에도 이 계약을 적용하므로
범주형 특성은 사전에 수치로 변환해야 합니다. CatBoost 원 모델은
`result.model.named_steps["model"].native_model_`로 접근합니다.
XGBoost 분류 타깃은 `0, 1, ..., K-1`이어야 합니다.
Perceptron은 확률 예측이 없어 `probabilities=None`입니다.
MLP의 수렴 경고는 숨기지 않으며 반복 수·스케일링 조건을 점검해야 합니다.
LightGBM 4.6.0과 scikit-learn 1.8.0 조합에서는 배열 입력에도 특성 이름 경고가 발생할 수 있습니다.
이 조합에서 예측·CV·결과 검증은 통과했으며 경고를 숨기지 않습니다.
AdaBoost 분류는 두 환경 모두 SAMME로 구성합니다.

## 검증 환경

| Python | 의존성 파일 | XGBoost | CatBoost | LightGBM | scikit-learn |
| --- | --- | --- | --- | --- | --- |
| 3.7.17 | `requirements-models-py37.txt` | 0.90 | 1.2.5 | 2.2.3 | 1.0.2 |
| 3.12.14 | `requirements-models-py312.txt` | CPU 3.0.5 | 1.2.8 | 4.6.0 | 1.8.0 |

```bash
python -m pip install -r requirements-models-py37.txt
# Python 3.12에서는 requirements-models-py312.txt 사용
python -m unittest discover -s tests -v
python -m src.MachineLearning.Trees.xgboost
python -m src.MachineLearning.Trees.catboost
python -m src.MachineLearning.Trees.lightgbm
python -m src.MachineLearning.Trees.adaboost
python -m src.MachineLearning.NeuralNetworks.mlp
python -m src.MachineLearning.NeuralNetworks.perceptron
```

테스트는 11개 분류·회귀 조합, 여섯 모델의 CV, CatBoost 문자열 라벨 및 회귀 CV,
학습 구간만 사용하는 스케일링, 입력 오류, DTO 직렬화와 명세 형식을 검증합니다.
Python 3.7.4 및 실제 시험 서버에서는 직접 실행하지 않았습니다.
