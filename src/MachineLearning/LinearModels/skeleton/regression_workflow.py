"""
================================================================================
regression_workflow.py - regression_workflow UCI 스켈레톤.
================================================================================
설명:
    - regression_workflow UCI 스켈레톤.
    - 원 자료 → JSON/Profile → 학습 전용 파이프라인 → 수치/진단/사용자 Axes.
    - 작성자: 김재현(Finn), 2026-10-12.
분석 가정:
    - 자료의 열·타깃·분할 조건을 확인하고, 학습 전처리는 학습 구간에서만 적합합니다. 실행은 명시한 진입점에서 수행합니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-27
    - 라이선스: Proprietary; UCI 원본은 CC BY 4.0. import에서 다운로드·학습·그림 생성 없음.
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
from src.output import OutputDTO
from src.Recipes import load_profile
from src.datasets.uci import load_tabular
from src.MachineLearning.LinearModels.regression_workflow import (
    run_regression_workflow,
    build_regression_estimator,
)
from src.stats.regression.joint_f import run_joint_f
from src.stats.regression.nested_f import run_nested_f
from src.stats.regression.ols import run_ols
from src.stats.regression.linearity import run_linearity
from src.stats.regression.normality import run_normality
from src.stats.regression.heteroscedasticity import run_heteroscedasticity
from src.stats.regression.independence import run_independence
from src.stats.regression.collinearity import run_collinearity
from src.stats.regression.influence import run_influence


@dataclass
class RegressionStudyOutputDTO(OutputDTO):
    """회귀 설계·진단·그림의 독립 결과.

    Attributes:
        workflow (Any): CV 선택·원 단위 보류·조건부 계수 추론 DTO.
        diagnostics (Dict[str,Any]): 개별 가정/공선성/영향도/추가 F DTO.
        observation_mapping (Any): 원 입력↔적합 행 위치·Cook flag·실제 제거 기록.
        feature_selection (Any): 선택된 변환 설계의 VIF 제거 이력.
        figure (Any): 사용자/예제 소유 Figure 또는 None; JSON 제외.
        axes (Any): 사용자/예제 Axes 배열 또는 None; JSON 제외.
        plots (Dict[str,Any]): 이름별 개별 Plot DTO; JSON은 수치만.
    """

    workflow: Any
    diagnostics: Dict[str, Any]
    observation_mapping: Any
    feature_selection: Any
    figure: Any
    axes: Any
    plots: Dict[str, Any]


def load_data(cache_dir=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        cache_dir: 전용 명세의 cache_dir 입력.
    Returns:
        tuple: Wine 원 Feature 11개, quality 타깃.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    data = load_tabular("regression", cache_dir=cache_dir)
    return data.drop(columns="target").copy(), data.target.copy()


def load_recipe(overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        overrides: 전용 명세의 overrides 입력.
    Returns:
        RegressionWorkflowProfile: CV/보류/전처리 후보 설정.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    return load_profile("regression-workflow", overrides=overrides)


def build_pipeline(preprocessor=None, **options):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        preprocessor: 전용 명세의 preprocessor 입력.
        **options: 전용 명세의 **options 입력.
    Returns:
        RegressionEstimator: clone 가능한 미적합 모형; fit은 CV 내부.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    return build_regression_estimator(preprocessor=preprocessor, **options)


def main(cache_dir=None, overrides=None, axes=None, draw=False):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        cache_dir: 전용 명세의 cache_dir 입력.
        overrides: 전용 명세의 overrides 입력.
        axes: 전용 명세의 axes 입력.
        draw: 전용 명세의 draw 입력.
    Returns:
        RegressionStudyOutputDTO: 수치·절차/행 이력·선택적 subplot.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    import pandas as pd

    X, y = load_data(cache_dir)
    profile = load_recipe(overrides)
    workflow = run_regression_workflow(X, y, profile=profile)
    inference = workflow.inference
    model = workflow.model
    final_design = inference.design.drop(columns="Intercept")
    diagnostics = {
        "joint_f": run_joint_f(inference),
        "linearity": run_linearity(inference),
        "normality": run_normality(inference),
        "heteroscedasticity": run_heteroscedasticity(inference),
        "independence": run_independence(inference),
        "collinearity": run_collinearity(final_design),
        "influence": run_influence(inference),
    }
    if final_design.shape[1] > 1 and inference.model.cov_type == "nonrobust":
        restricted = run_ols(final_design.iloc[:, :-1], inference.model.model.endog)
        diagnostics["nested_f"] = run_nested_f(restricted, inference)
    kept_original = workflow.train_indices[model.kept_positions_]
    mapping = diagnostics["influence"].table.copy()
    mapping["original_position"] = kept_original
    removed = workflow.train_indices[model.removed_positions_]
    if len(removed):
        extra = pd.DataFrame(
            {
                "position": np.nan,
                "original_position": removed,
                "removed_during_training": True,
            }
        )
        mapping["removed_during_training"] = False
        mapping = pd.concat([mapping, extra], ignore_index=True, sort=False)
    else:
        mapping["removed_during_training"] = False
    figure = None
    grid = None
    plots = {}
    if draw or axes is not None:
        from src.Visualization.skeleton._common import render_panels
        from src.Visualization.regression.coefficient_intervals import (
            plot_coefficient_intervals,
        )
        from src.Visualization.regression.residual_fitted import plot_residual_fitted
        from src.Visualization.regression.scale_location import plot_scale_location
        from src.Visualization.regression.cooks_distance import plot_cooks_distance
        from src.Visualization.regression.leverage import plot_leverage
        from src.Visualization.regression.observed_predicted import (
            plot_observed_predicted,
        )
        from src.Visualization.regression.vif import plot_vif
        from src.Visualization.regression.prediction_interval import (
            plot_prediction_interval,
        )
        from src.Visualization.regression.partial_residual import plot_partial_residual
        from src.Visualization.regression.residual_order import plot_residual_order
        from src.Visualization.qq.normal import plot_normal

        future = model.transform_features(X.iloc[workflow.test_indices])
        model_y = model.target_transformer_.transform(workflow.y_test)
        calls = [
            ("coefficients", lambda ax: plot_coefficient_intervals(inference, ax=ax)),
            ("residual_fitted", lambda ax: plot_residual_fitted(inference, ax=ax)),
            (
                "qq_normal",
                lambda ax: plot_normal(
                    inference.residuals, ax=ax, overrides={"fit_parameters": True}
                ),
            ),
            ("scale_location", lambda ax: plot_scale_location(inference, ax=ax)),
            ("cooks_distance", lambda ax: plot_cooks_distance(inference, ax=ax)),
            ("leverage", lambda ax: plot_leverage(inference, ax=ax)),
            (
                "observed_predicted",
                lambda ax: plot_observed_predicted(
                    workflow.y_test, workflow.predictions, ax=ax
                ),
            ),
            ("vif", lambda ax: plot_vif(diagnostics["collinearity"], ax=ax)),
            (
                "prediction_interval",
                lambda ax: plot_prediction_interval(inference, future, model_y, ax=ax),
            ),
            (
                "partial_residual",
                lambda ax: plot_partial_residual(
                    inference, final_design.columns[0], ax=ax
                ),
            ),
            ("residual_order", lambda ax: plot_residual_order(inference, ax=ax)),
        ]
        dashboard = render_panels(calls, axes=axes, ncols=3)
        figure = dashboard.figure
        grid = dashboard.axes
        plots = dashboard.panels
    return RegressionStudyOutputDTO(
        workflow, diagnostics, mapping, model.selector_.history_, figure, grid, plots
    )


if __name__ == "__main__":
    result = main()
    print(result.workflow.comparison)
    print(result.workflow.selected_metrics)
