"""
================================================================================
visualization_gallery_inputs.py - 전체 시각화 PNG의 고정 합성 입력과 판단 목적. 네트워크·파일 쓰기는 수행하지 않습니다.
================================================================================
설명:
    - 전체 시각화 PNG의 고정 합성 입력과 판단 목적. 네트워크·파일 쓰기는 수행하지 않습니다.
분석 가정:
    - 입력의 형상·타입·자료 순서를 호출자가 확인합니다. 계산과 파일 쓰기의 조건은 각 공개 함수의 계약을 따릅니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-27
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""

from functools import lru_cache
import numpy as np
import pandas as pd
from scipy import stats


@lru_cache(maxsize=None)
def example_cases(section=None):
    """61개 시각화의 명시 입력·설정·자료 설명을 구성합니다.

    Args:
        section: 선택 영역 이름. None이면 전체; 개별 실행은 필요한 영역까지만 구성.
    Returns:
        dict: 모듈 상대 경로별 args/overrides/note와 선택 variants.
    Raises:
        ValueError/ImportError: 예제 모형 적합·선택 의존성 오류.
    """
    if section == "optimization":
        return {"optimization.convergence": _optimization_case()}
    from src.stats.regression.ols import run_ols
    from src.stats.regression.collinearity import run_collinearity
    from src.MachineLearning.GeneralizedLinearModels.logistic import run_logistic
    from src.MachineLearning.GeneralizedLinearModels.poisson import run_poisson
    from src.MachineLearning.GeneralizedLinearModels.gamma import run_gamma
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.dummy import DummyRegressor
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import networkx as nx

    rng = np.random.RandomState(20261027)
    X = pd.DataFrame(
        rng.normal(size=(120, 4)), columns=["amount", "duration", "frequency", "score"]
    )
    X.duration = 0.65 * X.amount + 0.65 * X.duration
    values = np.concatenate([rng.normal(0, 0.6, 90), rng.normal(2.4, 0.45, 30)])
    categories = np.repeat(["A", "B", "C", "D"], [55, 32, 21, 12])
    outcome = rng.binomial(1, np.repeat([0.15, 0.3, 0.5, 0.75], [55, 32, 21, 12]))
    grouped = values + np.repeat([0.0, 0.5, 1.0, 1.5], [55, 32, 21, 12])
    missing = X.copy()
    missing.loc[::4, "duration"] = np.nan
    missing.loc[::4, "frequency"] = np.nan
    missing.loc[::7, "frequency"] = np.nan
    centers = np.array([[-2.0, -1.0], [0.0, 2.0], [2.0, -1.0]])
    cluster_labels = np.repeat(["cluster A", "cluster B", "cluster C"], 20)
    cluster_X = np.repeat(centers, 20, axis=0) + rng.normal(scale=0.45, size=(60, 2))
    probability = 1 / (1 + np.exp(-1.5 * X.amount + 0.5 * X.frequency))
    truth = rng.binomial(1, probability)
    scores = np.clip(probability * 0.9 + 0.05, 0.01, 0.99)
    dates = pd.date_range("2026-01-01", periods=112, freq="D")
    series = (
        20
        + 0.035 * np.arange(112)
        + 2 * np.sin(2 * np.pi * np.arange(112) / 7)
        + rng.normal(0, 0.6, 112)
    )
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [
            ("A", "B", 4),
            ("A", "C", 3),
            ("B", "C", 2),
            ("C", "D", 1),
            ("D", "E", 3),
            ("D", "F", 2),
            ("E", "F", 2),
        ]
    )
    graph.add_node("isolated")
    cases = {}

    def add(path, args, note, overrides=None):
        cases[path] = dict(args=args, overrides=overrides or {}, note=note)

    for name in ("histogram", "kde", "ecdf", "qq"):
        add("numeric." + name, (values,), "고정 합성 혼합분포: 중심·꼬리·다봉성 검토")
    for name, args in {
        "count": (categories,),
        "pareto": (categories,),
        "grouped_distribution": (grouped, categories),
        "contingency": (categories, outcome),
        "mosaic": (categories, outcome),
        "category_rate": (categories, outcome),
    }.items():
        add(
            "categorical." + name,
            args,
            "합성 범주별 표본 수·분포·발생률; 인과 해석 아님",
        )
    for name in ("missingness", "missing_matrix", "missing_association"):
        add(
            "quality." + name,
            (missing,),
            "합성 공동 결측: 빈도와 패턴 확인; 결측 원인 판정 아님",
        )
    add(
        "relationships.scatter",
        (X.amount, X.duration, categories),
        "대응 행을 유지한 합성 변수 관계",
    )
    add("relationships.hexbin", (X.amount, X.duration), "합성 관측의 밀집 구간 확인")
    add(
        "relationships.correlation",
        (X.corr(),),
        "사전 계산 Pearson 계수; 상관만으로 변수 제거하지 않음",
    )
    add(
        "clustering.dendrogram",
        (cluster_X,),
        "합성 세 집단: 절단 높이는 검토 후보",
        {"truncate_last": 18, "cut_height": 6.0},
    )
    add(
        "clustering.silhouette",
        (cluster_X, cluster_labels),
        "주어진 합성 군집 라벨과 같은 거리 공간의 분리도",
    )
    add(
        "clustering.k_distance",
        (cluster_X,),
        "자기 자신 포함 k번째 이웃 거리; eps 후보 검토",
    )
    add(
        "clustering.elbow",
        ([2, 3, 4, 5, 6], [240, 105, 92, 84, 79]),
        "설명용 사전 계산 inertia; 그림에서 모형을 재적합하지 않음",
    )
    add(
        "decomposition.explained_variance",
        ([0.58, 0.24, 0.12, 0.06],),
        "설명용 사전 계산 분산 비율과 누적 비율",
    )
    add(
        "decomposition.loadings",
        (
            np.array(
                [
                    [0.65, 0.6, 0.35, 0.25],
                    [-0.2, -0.2, 0.75, -0.6],
                    [0.3, -0.5, 0.1, 0.6],
                ]
            ),
            list(X.columns),
        ),
        "설명용 사전 계산 성분 계수; 전체 성분 부호는 임의",
    )
    for name in ("roc", "precision_recall", "threshold", "calibration"):
        add(
            "classification." + name,
            (truth, scores),
            "고정 합성 정답·확률; 실제 학습 모형의 보류 성능 주장 아님",
        )
    add(
        "classification.confusion",
        (truth, (scores >= 0.5).astype(int)),
        "합성 정답·확률에 임계값0.5 적용; 양성 라벨1",
    )
    add(
        "association.rules",
        (
            pd.DataFrame(
                {
                    "support": [0.08, 0.12, 0.18, 0.25, 0.32, 0.4],
                    "confidence": [0.35, 0.5, 0.68, 0.55, 0.8, 0.9],
                    "lift": [1.1, 1.8, 2.5, 1.3, 2.2, 1.6],
                }
            ),
        ),
        "설명용 사전 계산 규칙 지표; support와 lift를 함께 검토",
    )
    add(
        "networks.graph",
        (graph,),
        "합성 가중 무방향 그래프; 배치는 실제 공간 거리가 아님",
    )
    add(
        "selection.feature_importance",
        (list(X.columns), [0.18, 0.1, 0.04, -0.01], [0.025, 0.02, 0.015, 0.02]),
        "설명용 사전 계산 중요도·반복 표준편차; 신뢰구간 아님",
    )
    add(
        "selection.validation_curve",
        (
            [20, 40, 60, 80, 100],
            [0.95, 0.91, 0.88, 0.86, 0.85],
            [0.6, 0.7, 0.76, 0.79, 0.8],
            [0.02] * 5,
            [0.06, 0.045, 0.035, 0.03, 0.025],
        ),
        "설명용 사전 계산 학습량별 점수·표준편차; 실제 CV 실험 아님",
    )
    add(
        "time_series.timeline",
        (dates, series),
        "고정 합성 일별 추세·주기; 후행 요약만 사용",
    )
    for name in ("lag", "autocorrelation"):
        add("time_series." + name, (series,), "동일 간격·시간순 합성 자료의 지연 구조")
    add(
        "time_series.seasonal",
        (dates, series),
        "요일별 합성 관측 분포; 관측되지 않은 기간을 만들지 않음",
    )

    if section not in (None, "qq", "regression"):
        return _with_colorbars(cases)

    distributions = dict(
        normal=stats.norm(),
        student_t=stats.t(df=5),
        exponential=stats.expon(),
        gamma=stats.gamma(a=2),
        lognormal=stats.lognorm(s=1),
        weibull=stats.weibull_min(c=1.5),
        beta=stats.beta(a=2, b=2),
        uniform=stats.uniform(),
        chi_square=stats.chi2(df=5),
        logistic=stats.logistic(),
        poisson=stats.poisson(mu=1),
        binomial=stats.binom(n=10, p=0.5),
        bernoulli=stats.bernoulli(p=0.5),
    )
    for i, (name, distribution) in enumerate(distributions.items()):
        add(
            "qq." + name,
            (distribution.rvs(size=120, random_state=20261027 + i),),
            "기본 모수의 " + name + " 합성 표본; 그림만으로 분포 가정을 확정하지 않음",
        )

    if section == "qq":
        return _with_colorbars(cases)

    R = pd.DataFrame(rng.normal(size=(120, 3)), columns=["x1", "x2", "x3"])
    R.x2 = 0.96 * R.x1 + 0.18 * R.x2
    y = 2 * R.x1 - 0.5 * R.x2 + 0.8 * R.x3 + rng.normal(0, 0.4, 120)
    R.iloc[-1] = [4.0, 3.8, 3.0]
    y.iloc[-1] += 8
    ols = run_ols(R, y)
    future = pd.DataFrame(
        {
            "x1": np.linspace(-2, 2, 32),
            "x2": np.linspace(-1.92, 1.92, 32),
            "x3": np.zeros(32),
        }
    )
    future_y = 2 * future.x1 - 0.5 * future.x2 + rng.normal(0, 0.4, len(future))
    for name in (
        "residual_fitted",
        "scale_location",
        "residual_order",
        "cooks_distance",
        "leverage",
        "coefficient_intervals",
    ):
        add(
            "regression." + name,
            (ols,),
            "합성 학습 OLS: 공선성·영향 관측을 포함; 자동 행 제거 아님",
        )
    add(
        "regression.observed_predicted",
        (y, ols.fitted_values),
        "합성 학습 적합값; 보류 성능과 구분",
        {"sample_role": "train"},
    )
    add(
        "regression.prediction_interval",
        (ols, future, future_y),
        "새 합성 관측의 조건부 평균 구간과 관측 예측구간",
    )
    add(
        "regression.partial_residual",
        (ols, "x1"),
        "다른 선형 항을 보정한 부분 잔차; 공선성 조건 확인",
    )
    add(
        "regression.vif",
        (run_collinearity(R),),
        "합성 공선성 진단; VIF만으로 기계적인 변수 제거 금지",
    )
    cv = KFold(4, shuffle=True, random_state=20261027)
    comparisons = []
    for name, estimator in [
        ("mean baseline", DummyRegressor()),
        ("linear OLS", make_pipeline(StandardScaler(), LinearRegression())),
        ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
    ]:
        comparisons.append(
            dict(
                name=name,
                cv_r2=float(
                    cross_val_score(estimator, R, y, cv=cv, scoring="r2").mean()
                ),
            )
        )
    add(
        "regression.model_comparison",
        (pd.DataFrame(comparisons),),
        "동일 합성 자료·동일4-fold의 실제 CV R²; 최종 평가 자료 아님",
    )
    G = pd.DataFrame(rng.normal(size=(160, 2)), columns=["x1", "x2"])
    mean = np.exp(0.2 + 0.45 * G.x1 - 0.25 * G.x2)
    glm_models = {
        "logistic": run_logistic(
            G, rng.binomial(1, 1 / (1 + np.exp(-0.2 - 0.6 * G.x1 + 0.3 * G.x2)))
        ),
        "poisson": run_poisson(G, rng.poisson(mean)),
        "gamma": run_gamma(G, rng.gamma(shape=3.0, scale=mean / 3.0)),
    }
    add(
        "regression.glm_residuals",
        (glm_models["poisson"],),
        "합성 Poisson GLM의 Pearson 잔차; OLS 정규성 조건과 구분",
    )
    cases["regression.glm_residuals"]["variants"] = {}
    for family, result in glm_models.items():
        for residual in ("pearson", "deviance"):
            if family == "poisson" and residual == "pearson":
                continue
            cases["regression.glm_residuals"]["variants"][family + "_" + residual] = (
                dict(
                    args=(result,),
                    overrides={"residual": residual},
                    note="합성 "
                    + family
                    + " GLM의 "
                    + residual
                    + " 잔차; 분포별 조건부 분산 확인",
                )
            )
    if section is None:
        cases["optimization.convergence"] = _optimization_case()
    return _with_colorbars(cases)


def _optimization_case():
    """GA 시각화에 필요한 실제 탐색만 실행합니다."""
    from src.Optimization.genetic_algorithm import run_genetic_algorithm

    ga = run_genetic_algorithm(
        lambda v: (v[0] - 2) ** 2 + (v[1] + 1) ** 2,
        [(-5, 5), (-5, 5)],
        overrides={
            "population_size": 30,
            "generations": 35,
            "stall_generations": 20,
            "random_state": 20261027,
        },
    )
    return dict(
        args=(ga,),
        overrides={},
        note="실제 소규모 GA 실행 이력; 정체가 전역 최적성의 증거는 아님",
    )


def _with_colorbars(cases):
    """존재하는 예제에만 명시적 색상 범례를 연결합니다."""
    for path, label in {
        "association.rules": "Lift",
        "relationships.hexbin": "Count",
        "decomposition.loadings": "Component coefficient",
    }.items():
        if path in cases:
            cases[path]["colorbar_label"] = label
    return cases
