"""
================================================================================
loader.py - 모델별 JSON 레시피 읽기·검증·Profile 구성
================================================================================
설명:
    - src/Recipes의 평면 JSON 설정을 모델별 Profile로 변환합니다.
    - 레시피 → 명시적 overrides → Profile → 실행기 순서로 설정을 적용합니다.

분석 가정:
    - 레시피는 JSON 객체이며 입력 데이터는 실행기에서 별도로 검증합니다.
    - 파일 누락·알 수 없는 키·중복 키는 오류이며 설정을 조용히 무시하지 않습니다.

작성 정보:
    - 작성자: 김재현(Finn)
    - 이메일: penguin.klg@gmail.com
    - 작성일: 2026-10-11
    - 라이선스: Proprietary

변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    ----------------------------------------------------------------------------
    2026-10-11  김재현(Finn)     v1.0.0    모델별 레시피와 실행기 연결
================================================================================
"""
__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"

from dataclasses import fields
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


# 모델명: (모듈, Profile, 기본 task, 지원 task). 통계 검정에는 task 필드가 없습니다.
_REGISTRY = {
    'tf-classification': ('src.DeepLearning.TensorFlow.classification', 'TFClassificationProfile', None, (None,)),
    'tf-regression': ('src.DeepLearning.TensorFlow.regression', 'TFRegressionProfile', None, (None,)),
    'tf-rnn': ('src.DeepLearning.TensorFlow.rnn', 'TFRNNProfile', 'regression', ('classification', 'regression')),
    'tf-time-series': ('src.DeepLearning.TensorFlow.time_series', 'TFTimeSeriesProfile', None, (None,)),

    'chi-square': ('src.stats.tests.nonParametric.chi_square', 'ChiSquareProfile', None, (None,)),
    'independence': ('src.stats.tests.nonParametric.independence', 'IndependenceProfile', None, (None,)),
    'mcnemar': ('src.stats.tests.nonParametric.mcnemar', 'McNemarProfile', None, (None,)),
    'f-test': ('src.stats.tests.parametric.Variance.f_test', 'FTestProfile', None, (None,)),
    'normality': ('src.stats.tests.diagnostics.normality', 'NormalityProfile', None, (None,)),
    'homogeneity': ('src.stats.tests.diagnostics.homogeneity', 'HomogeneityProfile', None, (None,)),

    "xgboost": ("src.MachineLearning.Trees.xgboost", "XGBoostProfile", "classification", ("classification", "regression")),
    "catboost": ("src.MachineLearning.Trees.catboost", "CatBoostProfile", "classification", ("classification", "regression")),
    "lightgbm": ("src.MachineLearning.Trees.lightgbm", "LightGBMProfile", "classification", ("classification", "regression")),
    "adaboost": ("src.MachineLearning.Trees.adaboost", "AdaBoostProfile", "classification", ("classification", "regression")),
    "mlp": ("src.MachineLearning.NeuralNetworks.mlp", "MLPProfile", "classification", ("classification", "regression")),
    "perceptron": ("src.MachineLearning.NeuralNetworks.perceptron", "PerceptronProfile", "classification", ("classification",)),
    "adaline": ("src.MachineLearning.NeuralNetworks.adaline", "AdalineProfile", "classification", ("classification",)),
    "linear-regression": ("src.MachineLearning.LinearModels.linear_regression", "LinearRegressionProfile", "regression", ("regression",)),
    "oneway-anova": ("src.stats.tests.parametric.Average.Multivariate.OneWay", "OneWayANOVAProfile", None, (None,)),
    "twoway-anova": ("src.stats.tests.parametric.Average.Multivariate.TwoWay", "TwoWayANOVAProfile", None, (None,)),
    "manova": ("src.stats.tests.parametric.Average.Multivariate.MANOVA", "MANOVAProfile", None, (None,)),
    "posthoc": ("src.stats.tests.parametric.Average.Multivariate.PostHoc", "PostHocProfile", None, (None,)),
    'kmeans': ('src.MachineLearning.Clustering.kmeans', 'KMeansProfile', None, (None,)),
    'dbscan': ('src.MachineLearning.Clustering.dbscan', 'DBSCANProfile', None, (None,)),
    'hdbscan': ('src.MachineLearning.Clustering.hdbscan', 'HDBSCANProfile', None, (None,)),
    'k-distance': ('src.MachineLearning.Clustering.k_distance', 'KDistanceProfile', None, (None,)),
    'pca': ('src.MachineLearning.Decomposition.pca', 'PCAProfile', None, (None,)),
    'lda': ('src.MachineLearning.Decomposition.lda', 'LDAProfile', None, (None,)),
    'umap': ('src.MachineLearning.Decomposition.umap', 'UMAPProfile', None, (None,)),
    'tsne': ('src.MachineLearning.Decomposition.tsne', 'TSNEProfile', None, (None,)),
    'mds': ('src.MachineLearning.Decomposition.mds', 'MDSProfile', None, (None,)),
    'som': ('src.MachineLearning.Decomposition.som', 'SOMProfile', None, (None,)),
    'naive-bayes': ('src.MachineLearning.NaiveBayes.naive_bayes', 'NaiveBayesProfile', 'classification', ('classification',)),
    'wrapper-selection': ('src.MachineLearning.FeatureSelection.wrapper', 'WrapperSelectionProfile', 'classification', ('classification', 'regression')),
    'regularized-selection': ('src.MachineLearning.FeatureSelection.regularized', 'RegularizedSelectionProfile', 'classification', ('classification', 'regression')),
    'apriori': ('src.Association.apriori', 'AprioriProfile', None, (None,)),
    'sna': ('src.Networks.sna', 'SNAProfile', None, (None,)),
}


def _unique_object(pairs: Any) -> Dict[str, Any]:
    """JSON 중복 키를 거부합니다.

    Args:
        pairs: JSON 디코더의 키·값 목록.
    Returns:
        Dict[str, Any]: 중복 없는 JSON 객체.
    Raises:
        ValueError: 중복 키.
    """
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("레시피의 중복 키: " + key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    """JSON 표준 밖의 NaN/Infinity를 거부합니다.

    Args:
        value: 비표준 수치 상수 이름.
    Returns:
        None: 항상 오류를 반환.
    Raises:
        ValueError: 비표준 수치 상수.
    """
    raise ValueError("레시피에 비유한 JSON 상수를 사용할 수 없습니다: " + value)


def load_recipe(model_name: str, task: Optional[str] = None,
                recipe_path: Optional[str] = None) -> Dict[str, Any]:
    """모델·목적별 기본 파일 또는 사용자 JSON을 매번 읽습니다.

    Args:
        model_name: 등록 모델명. linear-regression, oneway-anova 등의 하이픈 이름 사용.
        task: 분류/회귀 선택. None이면 모델의 기본 목적. 통계 검정은 None만 허용.
        recipe_path: 사용자 JSON 경로. None이면 이 loader.py 옆의 모델-task.json 또는 검정명.json.
    Returns:
        Dict[str, Any]: 원본 JSON 설정. 캐시하거나 파일을 수정하지 않음.
    Raises:
        ValueError: 모델·task·JSON 형식·중복 키·비유한 값 오류.
        FileNotFoundError: 파일 누락. 내장 기본값으로 조용히 대체하지 않음.
        OSError: 파일 읽기 실패.
    """
    if model_name not in _REGISTRY:
        raise ValueError("지원하지 않는 모델명: " + str(model_name))
    _, _, default_task, supported = _REGISTRY[model_name]
    selected = default_task if task is None else task
    if selected not in supported:
        raise ValueError("해당 모델에서 지원하지 않는 task: " + str(selected))
    filename = model_name + ("-" + selected if selected is not None else "") + ".json"
    path = Path(recipe_path) if recipe_path is not None else Path(__file__).resolve().parent / filename
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(data, dict):
        raise ValueError("레시피 최상위는 JSON 객체여야 합니다.")
    return data


def _normalize(model_name: str, values: Dict[str, Any]) -> Dict[str, Any]:
    """기존 XGBoost 별칭과 JSON 목록을 Profile 형식으로 변환합니다.

    Args:
        model_name: 등록된 모델명.
        values: 원본 레시피 또는 overrides 사전.
    Returns:
        Dict[str, Any]: 원 사전을 변경하지 않은 정규화 설정.
    Raises:
        ValueError: 별칭과 정식 키를 동시에 지정한 경우.
    """
    result = dict(values)
    if model_name == "xgboost":
        for alias, canonical in (("alpha", "reg_alpha"), ("lambda", "reg_lambda")):
            if alias in result:
                if canonical in result:
                    raise ValueError("레시피 키 충돌: " + alias + " / " + canonical)
                result[canonical] = result.pop(alias)
    for key in ("hidden_layer_sizes", "target_columns"):
        if key in result and isinstance(result[key], list):
            result[key] = tuple(result[key])
    return result


def load_profile(model_name: str, task: Optional[str] = None,
                 recipe_path: Optional[str] = None,
                 overrides: Optional[Dict[str, Any]] = None) -> Any:
    """레시피를 읽고 Profile 필드·목적을 검증하여 구성합니다.

    Args:
        model_name: 등록 모델명.
        task: 선택할 분류/회귀 목적. 통계 검정은 None.
        recipe_path: 선택적 사용자 JSON 경로.
        overrides: 레시피 위에 적용할 명시적 Profile 설정. task 변경은 허용하지 않음.
    Returns:
        Any: 모델 전용 Profile. 설정 범위와 데이터 조건은 공개 실행기가 검증.
    Raises:
        ValueError: 알 수 없는 키·목적 불일치·별칭 충돌·JSON 오류.
        TypeError: overrides가 사전이 아닌 경우.
        FileNotFoundError/OSError: 레시피 파일 읽기 오류.
    Notes:
        누락된 필드는 Profile의 문서화된 기본값을 사용합니다.
        JSON에서 회귀를 선택하려면 task='regression'도 명시해야 기본 파일 선택과 일치합니다.
    """
    values = _normalize(model_name, load_recipe(model_name, task, recipe_path))
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise TypeError("overrides는 사전이어야 합니다.")
        updates = _normalize(model_name, overrides)
    else:
        updates = {}
    module_name, class_name, default_task, _ = _REGISTRY[model_name]
    # python -m에서 이미 실행 중인 모듈을 다시 import하면 Profile 클래스가 둘이 됩니다.
    main_module = sys.modules.get("__main__")
    if getattr(getattr(main_module, "__spec__", None), "name", None) == module_name:
        Profile = getattr(main_module, class_name)
    else:
        Profile = getattr(importlib.import_module(module_name), class_name)
    allowed = {field.name for field in fields(Profile)}
    # 오타가 overrides에 덮여 사라져도 원본 레시피의 오류를 알립니다.
    unknown = (set(values) | set(updates)) - allowed
    if unknown:
        raise ValueError("지원하지 않는 레시피 키: " + ", ".join(sorted(unknown)))
    selected = default_task if task is None else task
    if "task" in allowed:
        if any(layer.get("task", selected) != selected for layer in (values, updates)):
            raise ValueError("레시피/overrides의 task가 선택한 목적과 다릅니다.")
        values["task"] = selected
    values.update(updates)
    return Profile(**values)


def resolve_profile(model_name: str, profile: Any, recipe_path: Optional[str] = None,
                    task: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Any:
    """공개 실행기의 명시적 Profile 또는 레시피 경로를 선택합니다.

    Args:
        model_name: 등록 모델명.
        profile: 명시적 Profile 또는 None.
        recipe_path: Profile이 None일 때 사용할 사용자 레시피.
        task: Profile이 None일 때 선택할 목적.
        overrides: Profile이 None일 때 레시피에 적용할 설정.
    Returns:
        Any: 명시적 Profile을 그대로 반환하거나 JSON에서 구성한 Profile.
    Raises:
        ValueError: Profile과 레시피 옵션을 함께 지정한 경우 또는 레시피 오류.
        TypeError: 잘못된 overrides 타입.
        FileNotFoundError/OSError: 레시피 파일 읽기 오류.
    """
    if profile is not None:
        if recipe_path is not None or task is not None or overrides is not None:
            raise ValueError("명시적 Profile과 recipe_path/task/overrides는 함께 지정할 수 없습니다.")
        return profile
    return load_profile(model_name, task=task, recipe_path=recipe_path, overrides=overrides)
