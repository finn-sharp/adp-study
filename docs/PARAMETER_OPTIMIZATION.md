# 교차검증 파라미터 최적화

사용자 첨부 `[ML] - (외부자료)Template Code.html`의 6절 GridSearchCV 및 Optuna-KFold 흐름을 재사용 가능한 모듈로 구현했습니다. 자료의 미완성 trial 설정·평가지표·최적값 재적합 부분을 채우고, 전처리는 CV 폴드 안에서 적합하도록 변경했습니다.

| 모듈 | Profile / Output DTO | 공개 함수 | 후보 표현 |
| --- | --- | --- | --- |
| `src/MachineLearning/Optimization/grid_search.py` | GridSearchProfile / GridSearchOutputDTO | run_grid_search | 파라미터별 후보 목록의 직적 |
| `src/MachineLearning/Optimization/random_search.py` | RandomSearchProfile / RandomSearchOutputDTO | run_random_search | 후보 목록에서 n_iter개 무작위 추출 |
| `src/MachineLearning/Optimization/optuna_search.py` | OptunaSearchProfile / OptunaSearchOutputDTO | run_optuna_search | categorical / int / float, seed 고정 TPE |

RandomSearch는 첨부의 Grid/Optuna를 보완하는 추가 방법입니다. 모든 모듈은 기존 7개 섹션, JSON → Profile → 실행기 → 전용 Output DTO 컨벤션을 따릅니다.

## 입력 및 평가 계약

- 입력 `X,y`는 **학습 자료만** 전달합니다. 자동 test 분할은 수행하지 않습니다. 보류 자료로 최적화하지 않도록 호출자가 먼저 분리합니다.
- clone 가능한 sklearn estimator 또는 전처리/변수 선택/리샘플링을 포함한 Pipeline을 받습니다. `Basepipeline`은 `.pipeline`을 전달합니다. 파라미터 이름은 `estimator.get_params(deep=True)`와 정확히 일치해야 하며 Pipeline은 `model__max_depth` 같은 이름을 사용합니다.
- 결측·범주형 입력은 Pipeline에 대응 전처리가 있어야 합니다. 이미 전체 자료로 적합한 인코더/스케일러/선택기의 누출은 본 모듈이 복구하지 못합니다. SMOTE는 `imblearn.pipeline.Pipeline` 안에 넣습니다.
- single-output classification/regression. IID 분류는 StratifiedKFold, 회귀는 KFold. time은 입력 행이 시간순이라는 가정의 TimeSeriesSplit; group은 GroupKFold와 행별 groups. time/group에는 `shuffle=False`가 필요합니다.
- 분류 기본 scorer=accuracy, 회귀=neg_mean_absolute_error. **모든 점수는 maximize**합니다. MAE/MSE 등 손실은 sklearn의 음수 scorer를 사용하므로 `best_score=-MAE`입니다. Optuna도 같은 scorer 및 maximize로 통일해 방향 혼동을 막습니다.
- model은 최고 후보를 전체 입력 학습 자료에 재적합한 clone이며 원 estimator를 변경하지 않습니다. refit=False면 None입니다. test를 재적합 자료에 넣지 않습니다.
- 모든 후보는 같은 CV 분할을 사용합니다. `folds`는 원 입력 X의 위치, 전체 원자료 위치가 아닙니다. skeleton의 train_indices로 전체 위치에 대응할 수 있습니다.
- 잘못된 후보·적합 실패·비유한 점수는 오류입니다. Optuna pruning은 완료 폴드 누적 평균을 보고하며 pruned 후보는 최적값에서 제외하고 history에 상태와 부분 점수를 보존합니다. timeout은 Optuna만 지원합니다.
- CV 최고값은 후보 선정 편향을 포함합니다. 일반화 성능은 보류 test 또는 nested CV로 별도 평가합니다. 최적화의 CV 표준편차를 신뢰구간으로 해석하지 않습니다.

## 설정 및 결과

Profile 필드: task, scoring, cv, split, shuffle, random_state, search_space, n_iter, n_jobs, refit, timeout, pruning. 상세 타입·기본값·예외는 각 모듈의 Attributes/Args/Returns/Raises에 있습니다.

공통 DTO 필드: method, task, scoring, best_params, best_score, cv_results, folds, n_candidates, elapsed_seconds, model, study. `cv_results`는 params/mean_score/std_score/state/splitN_score를 가진 DataFrame입니다. `model`, `study`는 `to_dict()`/`to_json()` 기본 출력에서 제외됩니다. 이 JSON은 결과 기록이며 모델 직렬화 형식이 아닙니다.

레시피 18개: 방법별·task별 기본 6개, RF/LightGBM × 3방법 × 2task 전용 12개. generic 기본은 bare tree의 max_depth 후보이며 다른 estimator/Pipeline은 search_space를 재정의합니다. 모델별 레시피는 skeleton에서 `recipe_path`로 로드합니다.

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from src.MachineLearning.Optimization.optuna_search import run_optuna_search

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', RandomForestRegressor(random_state=1234, n_jobs=1)),
])
result = run_optuna_search(X_train, y_train, pipeline, task='regression', overrides={
    'scoring': 'neg_mean_absolute_error',
    'search_space': {
        'model__n_estimators': {'type':'int', 'low':30, 'high':90, 'step':10},
        'model__max_depth': {'type':'int', 'low':2, 'high':8},
    },
    'n_iter': 20,
})
print(result.best_params, -result.best_score)  # CV MAE
prediction = result.model.predict(X_test)    # 보류 자료 예측
print(result.to_json(indent=2))
```

예측 후 별도 평가 모듈에 y_test와 prediction을 전달해 실제 보류 오차를 계산합니다.

## skeleton

`src/MachineLearning/skeleton/optimization_{grid_search,random_search,optuna_search}.py`

load_data → load_recipe → build_pipeline → train 내부 CV → 최적 estimator 재적합 → 보류 평가 순서입니다. 내장 Iris/Diabetes를 사용해 오프라인 실행 가능하며, 기존 UCI skeleton의 (X,y)를 main(data=(X,y))로 전달할 수 있습니다. RF/LightGBM 및 분류/회귀를 지원합니다.

```bash
python -m src.MachineLearning.skeleton.optimization_grid_search
python -m src.MachineLearning.skeleton.optimization_random_search
python -m src.MachineLearning.skeleton.optimization_optuna_search
```

```python
from src.MachineLearning.skeleton.optimization_optuna_search import main
result = main(task='regression', model='lightgbm', overrides={'n_iter': 8})
```

## Python 및 의존성

기본 환경은 기존 requirements-models-py37/py312를 사용합니다. Optuna는 별도 선택 의존성으로 일반 import에서는 필요하지 않습니다. Python 3.7은 Optuna 3.0.6, Python 3.12는 4.5.0을 지정합니다. requirements-optimization-py37.txt / requirements-optimization-py312.txt를 추가 설치하고 CI 일반 분석 matrix에서 실행합니다. Optuna 병렬 trial은 난수 재현성을 위해 n_jobs=1로 제한합니다. grid/random은 -1 또는 양의 정수를 지원합니다.

공식 명세: [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), [Optuna Trial](https://optuna.readthedocs.io/en/v3.0.6/reference/generated/optuna.trial.Trial.html). 첨부 LightGBM 후보가 모든 자료에 최적은 아니므로 별도 모델별 JSON으로 조정합니다.
