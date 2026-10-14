"""random_search skeleton: 로딩 → JSON → Pipeline → train CV → 보류 평가.

작성 정보: 김재현(Finn), 2026-10-14, Proprietary.
원 자료 ML Template Code의 RF/LightGBM tuning 흐름을 실행 가능한 코드로 연결.
"""
__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.Recipes import load_profile
from src.MachineLearning.Optimization.random_search import run_random_search
from src.Evaluation.adapters import evaluate_predictions


def load_data(task="classification"):
    """오프라인 sklearn 원자료를 로딩합니다.

    Args:
        task: classification/regression.
    Returns:
        tuple: 특성 X와 단일 타깃 y.
    Raises:
        ValueError: 지원하지 않는 목적.
    """
    if task not in ("classification", "regression"):
        raise ValueError("task 오류")
    return (load_iris if task == "classification" else load_diabetes)(return_X_y=True)


def load_recipe(task="classification", model="random-forest", overrides=None):
    """모델·목적별 탐색 JSON을 Profile로 읽습니다.

    Args:
        task: classification/regression.
        model: random-forest/lightgbm.
        overrides: 일회성 레시피 재정의.
    Returns:
        object: 방법별 전용 Profile.
    Raises:
        ValueError/TypeError/OSError: 모델·레시피 오류.
    """
    if model not in ("random-forest", "lightgbm"):
        raise ValueError("model은 random-forest/lightgbm입니다.")
    path = Path(__file__).resolve().parents[2] / "Recipes" / ("optimization-" + model + "-random-search-" + task + ".json")
    return load_profile("optimization-random-search", task=task, recipe_path=str(path), overrides=overrides)


def build_pipeline(task="classification", model="random-forest"):
    """대치·표준화·모델을 CV 폴드 안에서 적합할 Pipeline.

    Args:
        task: classification/regression.
        model: random-forest/lightgbm.
    Returns:
        Pipeline: 미적합 전처리 및 모델.
    Raises:
        ValueError/ImportError: 목적·모델·선택 LightGBM 미설치.
    """
    if task not in ("classification", "regression"):
        raise ValueError("task 오류")
    if model == "random-forest":
        cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
        estimator = cls(random_state=1234, n_jobs=1)
    elif model == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor
        cls = LGBMClassifier if task == "classification" else LGBMRegressor
        estimator = cls(random_state=1234, n_jobs=1, verbosity=-1)
    else:
        raise ValueError("model 오류")
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()), ("model", estimator)])


def main(task="classification", model="random-forest", overrides=None, data=None):
    """학습 CV로만 최적화한 뒤 보류 test를 한 번 평가합니다.

    Args:
        task, model: JSON 및 estimator 선택.
        overrides: 실행만 재정의할 Profile 값.
        data: 선택 (X,y). None이면 내장 sklearn 원자료.
    Returns:
        dict: optimization DTO, holdout_metrics, train/test 원 행 위치.
    Raises:
        ValueError/ImportError: 자료·설정 또는 선택 LightGBM/Optuna 의존성 오류.
    """
    import numpy as np
    from sklearn.utils import _safe_indexing
    X, y = load_data(task) if data is None else data
    ids = np.arange(len(y))
    train, test = train_test_split(ids, test_size=.3, random_state=1234,
                                  stratify=y if task == "classification" else None)
    profile = load_recipe(task, model, overrides)
    result = run_random_search(_safe_indexing(X, train), _safe_indexing(y, train),
                          build_pipeline(task, model), profile)
    if result.model is None:
        raise ValueError("보류 평가 skeleton에는 refit=True가 필요합니다.")
    pred = result.model.predict(_safe_indexing(X, test))
    prob = result.model.predict_proba(_safe_indexing(X, test)) if task == "classification" else None
    metrics = evaluate_predictions(_safe_indexing(y, test), pred, task, probabilities=prob,
                                   classes=result.model.classes_ if task == "classification" else None)
    print(result.best_params, result.best_score, metrics)
    return {"optimization": result, "holdout_metrics": metrics,
             "train_indices": train, "test_indices": test}


if __name__ == "__main__":
    main(overrides={"n_iter": 4})
