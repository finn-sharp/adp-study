# 모델 모듈 명세와 실행

여덟 모듈은 기존 통계 검정 파일의 작성 정보, 변경 이력, 메타데이터와
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
| `src/MachineLearning/LinearModels/linear_regression.py` | `LinearRegressionProfile` / `LinearRegressionOutputDTO` | `run_linear_regression` | 단순·다중 선형회귀 |
| `src/MachineLearning/NeuralNetworks/adaline.py` | `AdalineProfile` / `AdalineOutputDTO` | `run_adaline` | 이진 분류 |

MCP 요청은 확인한 의미에 따라 MLP(다층 퍼셉트론)로 구현했습니다.

## 사용 예제

저장소 루트에서 실행합니다.

```python
from sklearn.datasets import load_iris
from src.Recipes import load_profile
from src.MachineLearning.Trees.xgboost import run_xgboost

X, y = load_iris(return_X_y=True)
profile = load_profile("xgboost", overrides={
    "cv": 3,
    "param_grid": {"max_depth": [2, 3]},
})
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

선형회귀는 특성 한 열이면 단순회귀, 여러 열이면 다중회귀를 수행합니다.
기본값은 OLS이며 `positive=True`이면 비음수 계수 제약을 적용합니다.
DTO에 계수·절편·보류 잔차·설계 행렬 rank·특이값을 제공합니다.
`positive=True`인 모델의 rank·특이값은 `None`입니다.
계수와 절편은 스케일러 변환 좌표계 기준이며 p-value·신뢰구간은 제공하지 않습니다.

아달린은 정확히 두 클래스를 지원하는 배치 경사하강법 모델입니다.
원래 두 라벨을 `classes[0]→-1`, `classes[1]→+1`로 매핑하고 영벡터에서 학습합니다.
손실은 `J=0.5*mean((Xw+b-t)²)`이며 가중치 갱신은 `w += eta0*X.T@(t-z)/n`,
절편 갱신은 `b += eta0*mean(t-z)`입니다.
연속 출력 `z>=0`이면 두 번째 클래스를 예측합니다. 연속 출력은 확률이 아니므로
`probabilities=None`이며 DTO에 `activation_scores`, `loss_curve`, `n_iter`, `converged`를 제공합니다.
`tol=None`이면 최대 epoch까지 수행합니다. 허용오차를 지정한 경우 연속 epoch의 손실 감소가
0 이상 tol 이하이면 수렴으로 표시하며, 반복 한도까지 미수렴이면 경고합니다.
기본 `standard` 스케일링을 사용하고 발산 시 학습률을 낮춥니다.

```python
from src.MachineLearning.LinearModels.linear_regression import run_linear_regression
from src.MachineLearning.NeuralNetworks.adaline import run_adaline

# X_numeric: 유한한 수치형 2차원 특성, y_continuous: 1차원 회귀 타깃
regression = run_linear_regression(X_numeric, y_continuous)
print(regression.coefficients, regression.intercept, regression.test_metrics)

# X_binary: 수치형 특성, y_binary: 정확히 두 클래스의 1차원 타깃
classification = run_adaline(X_binary, y_binary)
print(classification.loss_curve, classification.converged, classification.test_metrics)
```

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
python -m src.MachineLearning.LinearModels.linear_regression
python -m src.MachineLearning.NeuralNetworks.adaline
```

테스트는 11개 분류·회귀 조합, 여섯 모델의 CV, CatBoost 문자열 라벨 및 회귀 CV,
학습 구간만 사용하는 스케일링, 입력 오류, DTO 직렬화와 명세 형식을 검증합니다.
추가로 선형회귀의 단순·다중 계수 복원과 비음수 제약·CV, 아달린의 손계산 배치 갱신,
문자열 라벨·CV·손실 감소·수렴·발산·결과 계약을 검증합니다. 현재 전체 검증 범위와 실행 근거는 [검증 기록](VALIDATION.md)을 확인합니다.
Python 3.7.4 및 실제 시험 서버에서는 직접 실행하지 않았습니다.

## 모델별 설정 관리

실행 설정은 [별도 JSON 레시피](RECIPES.md)에서 관리합니다. Profile 생략 시 기본 레시피를 사용하고,
명시적 Profile 전달 시에는 전달한 설정을 사용합니다. `.py`의 7개 섹션과 전체 Profile·DTO 명세는 유지합니다.

## 확장 모듈

추가 군집·차원 축소·변수 선택·연관규칙·연결망도 같은 7개 섹션과 전용 Profile·Output DTO를 사용합니다.
각 기법의 전체 명세는 [분석 기법 목록](ANALYSIS_CATALOG.md)에 정리했습니다.
