"""응용 모의 문항의 회귀 조합. 재사용 계산은 src 모듈을 호출합니다."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    TimeSeriesSplit,
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
from src.MachineLearning.LinearModels._regression_estimator import RegressionEstimator
from src.MachineLearning.LinearModels.weighted_regression import (
    build_weighted_regressor,
)
from src.MachineLearning.LinearModels.weighted_regression import run_weighted_regression
from src.MachineLearning.NonlinearModels.nonlinear_regression import (
    build_nonlinear_regressor,
)
from src.MachineLearning.GeneralizedLinearModels.gamma import run_gamma
from src.MachineLearning.GeneralizedLinearModels.poisson import run_poisson
from src.MachineLearning.Optimization.grid_search import run_grid_search
from src.stats.regression.information_selection import run_information_selection
from src.stats.regression.normality import run_normality
from src.stats.regression.linearity import run_linearity
from src.stats.regression.heteroscedasticity import run_heteroscedasticity
from src.stats.regression.independence import run_independence
from src.stats.regression.collinearity import run_collinearity
from src.stats.regression.influence import run_influence
from src.Visualization.qq.normal import plot_normal
from src.Visualization.regression.glm_residuals import plot_glm_residuals
from src.Visualization.regression.residual_fitted import plot_residual_fitted
from src.Visualization.regression.partial_residual import plot_partial_residual
from src.Visualization.regression.vif import plot_vif
from src.Visualization.regression.cooks_distance import plot_cooks_distance
from src.Visualization.regression.observed_predicted import plot_observed_predicted
from src.Visualization.regression.coefficient_intervals import (
    plot_coefficient_intervals,
)
from src.Visualization.regression.model_comparison import plot_model_comparison


def metrics(y, pred):
    return dict(
        r2=float(r2_score(y, pred)),
        rmse=float(np.sqrt(mean_squared_error(y, pred))),
        mae=float(mean_absolute_error(y, pred)),
    )


class RegressionTasks:
    """해당 분석의 학습 상태와 문항 결과를 연결합니다."""

    def reg_design(self):
        self.X = self.reg.drop(columns=["id", "target"])
        self.y = self.reg.target
        ids = np.arange(len(self.y))
        if self.plan["ordered_regression"]:
            cut = int(len(ids) * 0.7)
            self.rt, self.rv = ids[:cut], ids[cut:]
            self.cv = TimeSeriesSplit(3)
        else:
            self.rt, self.rv = train_test_split(
                ids, test_size=0.3, random_state=self.seed
            )
            self.rt = np.sort(self.rt)
            self.rv = np.sort(self.rv)
            self.cv = KFold(3, shuffle=True, random_state=self.seed)
        return dict(
            n=len(self.reg),
            feature_names=self.X.columns.tolist(),
            missing=self.X.isna().sum().to_dict(),
            train_indices=self.rt,
            test_indices=self.rv,
            split="chronological" if self.plan["ordered_regression"] else "iid",
            baseline="train-only median+scale+OLS; final test sealed",
        )

    def reg_baseline(self):
        self.base = RegressionEstimator().fit(
            self.X.iloc[self.rt], self.y.iloc[self.rt]
        )
        self.base_cv = float(
            cross_val_score(
                RegressionEstimator(),
                self.X.iloc[self.rt],
                self.y.iloc[self.rt],
                cv=self.cv,
                scoring="r2",
            ).mean()
        )
        inf = self.base.inference_
        return dict(
            train=metrics(
                self.y.iloc[self.rt], self.base.predict(self.X.iloc[self.rt])
            ),
            cv_r2=self.base_cv,
            coefficients=inf.coefficients,
            fit_metrics=inf.fit_metrics,
            final_test="not evaluated yet",
        )

    def reg_diagnostics(self):
        m = self.base.inference_
        self.base_influence = run_influence(m)
        self.base_diagnostics = dict(
            normality=run_normality(m),
            linearity=run_linearity(m),
            heteroscedasticity=run_heteroscedasticity(m),
            independence=run_independence(
                m, overrides={"ordered": self.plan["ordered_regression"], "nlags": 3}
            ),
            collinearity=run_collinearity(m.design.drop(columns="Intercept")),
        )
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        plot_residual_fitted(m, ax=axes[0, 0])
        plot_normal(m.residuals, ax=axes[0, 1], overrides={"fit_parameters": True})
        plot_cooks_distance(m, ax=axes[1, 0])
        plot_coefficient_intervals(m, ax=axes[1, 1])
        plot_vif(self.base_diagnostics["collinearity"], ax=axes[0, 2])
        for feature, ax in zip(
            m.design.columns[1:], [axes[1, 2], axes[2, 0], axes[2, 1], axes[2, 2]]
        ):
            plot_partial_residual(m, feature, ax=ax)
        plt.tight_layout()
        plt.show()
        return dict(
            tests=self.base_diagnostics,
            influence=self.base_influence.table,
            review_original_positions=(
                self.rt[self.base_influence.table.cook_flag.to_numpy()]
                if "cook_flag" in self.base_influence.table
                else []
            ),
            interpretation="tests are conditional on fitted training model; nonrejection does not prove assumptions; Cook flags are review only",
        )

    def reg_selection(self):
        design = self.base.transform_features(self.X.iloc[self.rt])
        self.ic_results = {
            criterion: run_information_selection(
                design,
                self.y.iloc[self.rt].to_numpy(),
                overrides={"criterion": criterion},
            )
            for criterion in ["aic", "bic"]
        }
        return dict(
            aic=self.ic_results["aic"],
            bic=self.ic_results["bic"],
            same_rows=self.rt,
            same_response="original y; never compare with log(y) AIC",
            note="variable selection is refitted inside each CV fold below; full-train table is descriptive",
        )

    def _reg_candidates(self):
        full = self.X.columns.tolist()
        candidates = [
            ("baseline", RegressionEstimator(), full, {"degree": [1]}),
            ("vif", RegressionEstimator(vif_threshold=5.0), full, {"degree": [1]}),
            (
                "aic",
                RegressionEstimator(vif_threshold=5.0, information_criterion="aic"),
                full,
                {"degree": [1]},
            ),
            (
                "bic",
                RegressionEstimator(vif_threshold=5.0, information_criterion="bic"),
                full,
                {"degree": [1]},
            ),
        ]
        kind = self.plan["regression_kind"]
        if kind == "power":
            candidates.append(
                (
                    "feature_power",
                    RegressionEstimator(vif_threshold=5.0, power="yeo-johnson"),
                    full,
                    {"degree": [1]},
                )
            )
        elif kind == "lognormal":
            candidates.append(
                (
                    "log_target",
                    RegressionEstimator(
                        target_method="log", smearing=True, vif_threshold=5.0
                    ),
                    full,
                    {"degree": [1]},
                )
            )
        elif kind == "quadratic":
            candidates.append(
                (
                    "polynomial",
                    RegressionEstimator(vif_threshold=10.0),
                    full,
                    {"degree": [1, 2]},
                )
            )
        elif kind == "heteroscedastic":
            candidates.append(
                ("fgls", build_weighted_regressor(), full, {"iterations": [1, 2]})
            )
        elif kind == "influence":
            candidates.append(
                (
                    "cook_drop",
                    RegressionEstimator(
                        vif_threshold=5.0,
                        influence_action="drop",
                        max_remove_fraction=0.2,
                    ),
                    full,
                    {"degree": [1]},
                )
            )
        elif kind == "ordered":
            candidates.append(
                (
                    "hac_inference",
                    RegressionEstimator(
                        vif_threshold=5.0, cov_type="HAC", hac_maxlags=3
                    ),
                    full,
                    {"degree": [1]},
                )
            )
        elif kind in ("exponential", "growth"):
            settings = (
                dict(
                    model_name="exponential",
                    initial_parameters=[2.0, 0.5, 1.0],
                    lower_bounds=[0.1, 0.01, -5.0],
                    upper_bounds=[8.0, 2.0, 8.0],
                )
                if kind == "exponential"
                else dict(
                    model_name="logistic",
                    initial_parameters=[10.0, 1.0, 1.5],
                    lower_bounds=[1.0, 0.01, -2.0],
                    upper_bounds=[20.0, 6.0, 6.0],
                )
            )
            candidates.append(
                (
                    "nonlinear",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median")),
                            ("model", build_nonlinear_regressor(overrides=settings)),
                        ]
                    ),
                    full[:1],
                    {"model__maxfev": [10000]},
                )
            )
        elif kind in ("gamma", "poisson"):
            cls = GammaRegressor if kind == "gamma" else PoissonRegressor
            candidates.append(
                (
                    kind,
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                            ("model", cls(max_iter=500)),
                        ]
                    ),
                    full,
                    {"model__alpha": [0.0, 0.1]},
                )
            )
        return candidates

    def reg_tuning(self):
        self.reg_search = {}
        self.reg_columns = {}
        rows = []
        for name, model, columns, space in self._reg_candidates():
            result = run_grid_search(
                self.X.iloc[self.rt][columns],
                self.y.iloc[self.rt],
                model,
                task="regression",
                overrides={
                    "search_space": space,
                    "scoring": "r2",
                    "split": "time" if self.plan["ordered_regression"] else "iid",
                    "shuffle": not self.plan["ordered_regression"],
                    "cv": 3,
                    "random_state": self.seed,
                    "n_jobs": 1,
                },
            )
            self.reg_search[name] = result
            self.reg_columns[name] = columns
            rows.append(
                dict(
                    name=name,
                    cv_r2=float(result.best_score),
                    best_params=result.best_params,
                )
            )
        self.reg_comparison = pd.DataFrame(rows)
        self.reg_name = max(
            self.reg_search, key=lambda n: self.reg_search[n].best_score
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_model_comparison(self.reg_comparison, ax=ax)
        plt.tight_layout()
        plt.show()
        return dict(
            comparison=self.reg_comparison,
            selected_name=self.reg_name,
            folds=self.reg_search[self.reg_name].folds,
            scope="train CV only; raw y R2 shared across all candidates; no final test used",
        )

    def reg_final(self):
        result = self.reg_search[self.reg_name]
        model = result.model
        cols = self.reg_columns[self.reg_name]
        prediction = model.predict(self.X.iloc[self.rv][cols])
        baseline = self.base.predict(self.X.iloc[self.rv])
        details = {}
        kind = self.plan["regression_kind"]
        if isinstance(model, RegressionEstimator):
            details.update(
                fitted_response_scale=model.target_method,
                model_inference=model.inference_,
                removed_original_positions=self.rt[model.removed_positions_],
                selected_features=model.output_features_,
            )
            details["after_tests"] = dict(
                normality=run_normality(model.inference_),
                linearity=run_linearity(model.inference_),
                heteroscedasticity=run_heteroscedasticity(model.inference_),
                independence=run_independence(
                    model.inference_,
                    overrides={"ordered": self.plan["ordered_regression"], "nlags": 3},
                ),
            )
        if kind == "heteroscedastic":
            details["fgls"] = run_weighted_regression(
                self.X.iloc[self.rt], self.y.iloc[self.rt], self.X.iloc[self.rv]
            )
            wres = details["fgls"].weighted_residuals
            stat, p = stats.shapiro(wres)
            details["weighted_residual_check"] = dict(
                statistic=float(stat),
                p_value=float(p),
                interpretation="descriptive conditional check after variance estimation; not an exact independent normality test",
            )
        if kind in ("exponential", "growth"):
            native = self.reg_search["nonlinear"].model.named_steps["model"].result_
            details["nonlinear_candidate"] = native
            details["nonlinear_covariance_condition_number"] = float(
                np.linalg.cond(native.covariance)
            )
        if kind in ("gamma", "poisson"):
            preprocessing = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]
            )
            train = pd.DataFrame(
                preprocessing.fit_transform(self.X.iloc[self.rt]), columns=cols
            )
            test = pd.DataFrame(
                preprocessing.transform(self.X.iloc[self.rv]), columns=cols
            )
            details["family_inference"] = (
                run_gamma if kind == "gamma" else run_poisson
            )(train, self.y.iloc[self.rt].to_numpy(), test)
        selected = metrics(self.y.iloc[self.rv], prediction)
        base = metrics(self.y.iloc[self.rv], baseline)
        if kind in ("gamma", "poisson", "heteroscedastic"):
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            plot_observed_predicted(self.y.iloc[self.rv], prediction, ax=axes[0])
            if kind == "heteroscedastic":
                plot_normal(
                    details["fgls"].weighted_residuals,
                    ax=axes[1],
                    overrides={
                        "fit_parameters": True,
                        "title": "Estimated-weight residuals (conditional)",
                    },
                )
            else:
                plot_glm_residuals(details["family_inference"], ax=axes[1])
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_observed_predicted(self.y.iloc[self.rv], prediction, ax=ax)
        plt.tight_layout()
        plt.show()
        return dict(
            selected_name=self.reg_name,
            baseline_test=base,
            selected_test=selected,
            delta_r2=selected["r2"] - base["r2"],
            test_indices=self.rv,
            truth=self.y.iloc[self.rv].to_numpy(),
            predictions=prediction,
            details=details,
            conclusion="CV selected before test. R2 improvement is empirical, not guaranteed; covariance correction alone does not change predictions; nonlinear/GLM do not inherit all OLS assumptions.",
        )
