"""
================================================================================
grid_search.py - GridSearchCV 격자 탐색
================================================================================
설명:
    - 사용자 ML Template Code의 CV 최적화를 재사용 가능한 모듈로 구현합니다.
분석 가정:
    - X, y는 학습 자료만 전달합니다. 보류 test는 최적화 후 별도 평가합니다.
    - 전처리·선택·리샘플링은 clone 가능한 Pipeline에 넣어 폴드 안에서 적합합니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-14
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR           VERSION   DESCRIPTION
    2026-10-14  김재현(Finn)       v1.0.0    Profile/DTO/JSON/skeleton 추가
================================================================================
"""
__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.output import OutputDTO
from src.Recipes import resolve_profile
from ._common import validate_inputs, execute_search

# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class GridSearchProfile:
    """탐색 설정. 후보는 estimator.get_params(deep=True)의 키를 사용합니다.

    Attributes:
        task (str): classification/regression. 기본 classification.
        scoring (Optional[str]): sklearn scorer 이름; None은 accuracy/neg_mean_absolute_error.
            모든 점수는 클수록 좋습니다. 음수 손실은 실제 loss와 부호가 반대입니다.
        cv (int): 2 이상 폴드 수. 기본 3.
        split (str): iid/time/group. 기본 iid; 그룹 자료는 groups도 전달.
        shuffle (bool): iid 폴드 섞기. time/group에서는 False만 허용.
        random_state (int): 0 이상 시드. 기본 1234.
        search_space (Dict): grid/random은 후보 목록, optuna는 type/choices 또는 low/high/step/log.
        n_iter (int): random 후보/optuna trial 예산. grid에서는 사용하지 않음. 기본 10.
        n_jobs (int): -1 또는 양수; Optuna는 재현성을 위해 1만 허용.
        refit (bool): 최고 설정을 X 전체에 재적합; False이면 model=None.
        timeout (Optional[float]): Optuna 시간 예산(초); grid/random은 None만 허용.
        pruning (bool): Optuna median fold pruning; grid/random은 False만 허용.
    """
    task: str = "classification"
    scoring: Optional[str] = None
    cv: int = 3
    split: str = "iid"
    shuffle: bool = True
    random_state: int = 1234
    search_space: Dict[str, Any] = field(default_factory=dict)
    n_iter: int = 10
    n_jobs: int = 1
    refit: bool = True
    timeout: Optional[float] = None
    pruning: bool = False

# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class GridSearchOutputDTO(OutputDTO):
    """탐색 결과. fold 행 위치는 입력 X 기준이며 test 결과는 포함하지 않습니다.

    Attributes:
        method (str): grid_search/random_search/optuna_search.
        task (str): classification/regression.
        scoring (str): 실제 사용한 sklearn scorer 이름.
        best_params (Dict): 최고 CV 후보 설정.
        best_score (float): 최고 CV 평균. 모두 maximize이며 음수 손실의 부호에 주의.
        cv_results (pd.DataFrame): 전체 후보 params·평균·표준편차·상태·fold 점수.
        folds (List[Dict]): train_indices/validation_indices, 원 입력 행 위치.
        n_candidates (int): 시도 후보 수. Optuna는 pruned 포함.
        elapsed_seconds (float): 탐색과 재적합에 걸린 초.
        model (Any): 재적합 clone. refit=False이면 None, JSON 제외.
        study (Any): Optuna Study 또는 None, JSON 제외.
    """
    method: str
    task: str
    scoring: str
    best_params: Dict[str, Any]
    best_score: float
    cv_results: pd.DataFrame
    folds: List[Dict[str, Any]]
    n_candidates: int
    elapsed_seconds: float
    model: Any
    study: Any = None
    runtime_fields = OutputDTO.runtime_fields | frozenset(("study",))

# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------
def _validate_inputs(X, y, estimator, profile, groups=None):
    """Profile 타입·행 수·후보·scorer·fold 제약을 검증합니다.

    Args:
        X, y: 학습 자료만 전달. Pipeline 안에서 결측·범주 처리 가능.
        estimator: clone/get_params/fit 가능한 sklearn estimator/Pipeline.
        profile: GridSearchProfile。
        groups: group 분할의 행별 식별자. 다른 분할에서는 지정 불가.
    Returns:
        tuple: folds 및 scorer 이름.
    Raises:
        TypeError/ValueError: 타입·설정·자료·분할 조건 위반.
    """
    if not isinstance(profile, GridSearchProfile):
        raise TypeError("GridSearchProfile이 필요합니다.")
    return validate_inputs(X, y, estimator, profile, "grid_search", groups)

# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _run_model(X, y, estimator, profile, folds, scoring):
    """동일 fold에서 후보를 평가하고 최적 clone을 재적합합니다.

    Args:
        X, y, estimator, profile: 검증된 입력. 원 estimator를 변경하지 않음.
        folds, scoring: 검증된 분할·scorer.
    Returns:
        GridSearchOutputDTO: CV 점수·후보 이력·선택적 재적합 모델.
    Raises:
        ValueError/ImportError: 적합 실패·비유한 점수·선택 의존성 미설치.
    """
    return GridSearchOutputDTO(**execute_search(X, y, estimator, profile, "grid_search", folds, scoring))

# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def run_grid_search(X, y, estimator, profile=None, *, groups=None, task=None,
               recipe_path=None, overrides=None):
    """Profile 또는 전용 JSON으로 학습 자료 파라미터를 최적화합니다.

    Args:
        X, y: test를 포함하지 않는 학습 자료. 동일 행 순서.
        estimator: 미적합 estimator 또는 전처리를 포함한 Pipeline.
        profile: 명시 GridSearchProfile. recipe/task/overrides와 배타적.
        groups: group split 식별자.
        task, recipe_path, overrides: Profile 생략 시 레시피 선택·재정의.
    Returns:
        GridSearchOutputDTO: model로 새 자료 predict 가능. DTO JSON 변환 가능.
    Raises:
        TypeError/ValueError/OSError/ImportError: 입력·설정·CV·학습·선택 의존성 오류.
    """
    profile = resolve_profile("optimization-grid-search", profile, recipe_path=recipe_path,
                              task=task, overrides=overrides)
    folds, scoring = _validate_inputs(X, y, estimator, profile, groups)
    return _run_model(X, y, estimator, profile, folds, scoring)

# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    X, y = load_iris(return_X_y=True)
    result = run_grid_search(X, y, DecisionTreeClassifier(random_state=1234),
                        overrides={"search_space": {'max_depth': [2, 3, 4]}, "n_iter": 3})
    print(result.to_json(indent=2)[:800])
