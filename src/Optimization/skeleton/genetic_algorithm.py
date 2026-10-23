"""UCI Iris 특성 부분집합: train CV 안에서 GA 선택, 최종 test 한 번 평가.

GA 목적은 CV 오차+0.001×특성 수, 빈 부분집합은 실행불가입니다.
"""

from functools import partial
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from src.datasets.uci import load_tabular
from src.Recipes import load_profile
from src.Optimization.genetic_algorithm import run_genetic_algorithm


def load_data(cache_dir=None):
    """실제 UCI Iris를 로딩합니다.

    Args:
        cache_dir: UCI 다운로드 캐시 경로.
    Returns:
        tuple: (X,y) 특성과 라벨.
    Raises:
        ValueError/OSError: 다운로드 또는 자료 오류.
    """
    data = load_tabular("classification", cache_dir=cache_dir)
    return data.drop(columns="target"), data.target.to_numpy()


def load_recipe(overrides=None):
    """GA 레시피를 읽습니다.

    Args:
        overrides: 일회성 재정의.
    Returns:
        GeneticAlgorithmProfile: 탐색 설정.
    Raises:
        ValueError/TypeError/OSError: 레시피 오류.
    """
    return load_profile("genetic-algorithm", overrides=overrides)


def build_pipeline():
    """각 CV 학습 폴드에서만 표준화를 적합합니다.

    Args:
        없음: 모델 후보는 사전 고정.
    Returns:
        Pipeline: 미적합 스케일러와 로지스틱 모형.
    Raises:
        None: 생성만 수행.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=500, random_state=1234)),
        ]
    )


def main(cache_dir=None, overrides=None, data=None):
    """학습/CV 특성 선택과 원행 보류 평가를 분리합니다.

    Args:
        cache_dir: UCI 자료 캐시.
        overrides: GA 레시피 재정의.
        data: 선택 (X,y); 없으면 실제 UCI 자료.
    Returns:
        dict: 탐색 DTO, 선택 열·행 위치·보류 정확도·모델.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·실행가능 해 미발견.
    """
    X, y = load_data(cache_dir) if data is None else data
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    train, test = train_test_split(
        np.arange(len(y)), test_size=0.3, stratify=y, random_state=1234
    )
    cv = StratifiedKFold(3, shuffle=True, random_state=1234)
    cache = {}

    def objective(mask):
        """train CV의 패널티 포함 오차를 반환합니다.

        Args:
            mask: 정수 0/1 유전자.
        Returns:
            float: 평균 검증 오차+특성 수 패널티.
        Raises:
            ValueError: 모델 적합 오류.
        """
        key = tuple(mask.astype(int))
        if key not in cache:
            selected = mask.astype(bool)
            cache[key] = (
                1.0
                if not selected.any()
                else 1
                - cross_val_score(
                    build_pipeline(),
                    X[train][:, selected],
                    y[train],
                    cv=cv,
                    scoring="accuracy",
                ).mean()
                + 0.001 * selected.sum()
            )
        return cache[key]

    def constraints(mask):
        """빈 부분집합을 불허합니다.

        Args:
            mask: 정수 0/1 유전자.
        Returns:
            tuple: 부등식·등식 잔차.
        Raises:
            None: 유한 유전자 입력.
        """
        return [1 - mask.sum()], []

    profile = load_recipe(overrides)
    if profile.maximize:
        raise ValueError("이 스켈레톤 목적은 최소화입니다.")
    result = run_genetic_algorithm(
        objective,
        np.tile([0, 1], (X.shape[1], 1)),
        integer_mask=np.ones(X.shape[1], dtype=bool),
        constraints=constraints,
        profile=profile,
    )
    if not result.feasible:
        raise ValueError("빈 부분집합 이외의 해를 찾지 못했습니다.")
    selected = result.best_solution.astype(bool)
    model = build_pipeline().fit(X[train][:, selected], y[train])
    return dict(
        search=result,
        selected_features=np.flatnonzero(selected),
        train_indices=train,
        test_indices=test,
        test_accuracy=float(
            accuracy_score(y[test], model.predict(X[test][:, selected]))
        ),
        model=model,
    )


if __name__ == "__main__":
    result = main()
    print(result["search"].to_json(indent=2))
    print("test_accuracy:", result["test_accuracy"])
