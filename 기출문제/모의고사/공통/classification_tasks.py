"""
================================================================================
classification_tasks.py - 응용 모의 문항의 분류 조합. 재사용 계산은 src 모듈을 호출합니다.
================================================================================
설명:
    - 응용 모의 문항의 분류 조합. 재사용 계산은 src 모듈을 호출합니다.
분석 가정:
    - 제공 자료·문항 명세·실행 환경을 따릅니다. 복원·창작 연습 자료의 수치를 공식 기출 정답으로 해석하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-27
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)
from matplotlib import pyplot as plt
from src.MachineLearning.GeneralizedLinearModels.logistic import run_logistic
from src.MachineLearning.Optimization.grid_search import run_grid_search
from src.MachineLearning.Optimization.optuna_search import run_optuna_search
from src.Preprocessing.resampling._common import load_resampler
from src.Preprocessing.regression.vif_selection import build_vif_selector
from src.Visualization.classification.roc import plot_roc
from src.Visualization.classification.precision_recall import plot_precision_recall
from src.Visualization.classification.calibration import plot_calibration


class ClassificationTasks:
    """해당 분석의 학습 상태와 문항 결과를 연결합니다."""

    def cls_design(self):
        self.CX = self.cls.drop(columns=["id", "target"])
        self.Cy = self.cls.target.to_numpy()
        self.ct, self.ce = train_test_split(
            np.arange(len(self.Cy)),
            test_size=0.3,
            stratify=self.Cy,
            random_state=self.seed,
        )
        self.ct = np.sort(self.ct)
        self.ce = np.sort(self.ce)
        return dict(
            train_indices=self.ct,
            test_indices=self.ce,
            class_counts_train=pd.Series(self.Cy[self.ct]).value_counts().to_dict(),
            positive_label=1,
            event=self.plan["classification_event"],
            rule="natural prevalence test sealed; no resampling before split",
        )

    def cls_tuning(self):
        from imblearn.pipeline import Pipeline as ImbPipeline

        self.cls_search = {}
        sampler = self.plan["sampler"]
        method = self.plan["classification_search"]
        rows = []
        for name in ["none", sampler]:
            settings = (
                {"k_neighbors": 3, "random_state": self.seed}
                if name == "smote"
                else {"random_state": self.seed}
            )
            sampling = (
                "passthrough"
                if name == "none"
                else load_resampler(name, overrides=settings)
            )
            pipeline = ImbPipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("vif", build_vif_selector(overrides={"threshold": 5.0})),
                    ("sample", sampling),
                    (
                        "model",
                        LogisticRegression(max_iter=1000, random_state=self.seed),
                    ),
                ]
            )
            space = {"model__C": [0.1, 1.0, 10.0]}
            if method == "optuna":
                import optuna

                optuna.logging.set_verbosity(optuna.logging.WARNING)
                search = run_optuna_search(
                    self.CX.iloc[self.ct],
                    self.Cy[self.ct],
                    pipeline,
                    task="classification",
                    overrides={
                        "search_space": {
                            "model__C": {
                                "type": "categorical",
                                "choices": [0.1, 1.0, 10.0],
                            }
                        },
                        "scoring": "roc_auc",
                        "n_iter": 6,
                        "cv": 3,
                        "random_state": self.seed,
                    },
                )
            else:
                search = run_grid_search(
                    self.CX.iloc[self.ct],
                    self.Cy[self.ct],
                    pipeline,
                    task="classification",
                    overrides={
                        "search_space": space,
                        "scoring": "roc_auc",
                        "cv": 3,
                        "random_state": self.seed,
                    },
                )
            self.cls_search[name] = search
            rows.append(
                dict(
                    name=name, cv_auc=search.best_score, best_params=search.best_params
                )
            )
        self.cls_name = max(
            self.cls_search, key=lambda n: self.cls_search[n].best_score
        )
        return dict(
            method=method,
            comparison=rows,
            selected=self.cls_name,
            folds=self.cls_search[self.cls_name].folds,
            sampling="only in imblearn Pipeline.fit of CV training folds; validation/test never resampled",
        )

    def cls_logit(self):
        # Explanatory unpenalized GLM is distinct from tuned penalized predictive classifier.
        pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("vif", build_vif_selector(overrides={"threshold": 5.0})),
            ]
        )
        a = pipeline.fit_transform(self.CX.iloc[self.ct], self.Cy[self.ct])
        names = pipeline["vif"].get_feature_names_out().tolist()
        frame = pd.DataFrame(a, columns=names)
        self.logit = run_logistic(frame, self.Cy[self.ct])
        extra = frame.copy()
        col = names[0]
        extra[col + "_squared"] = extra[col] ** 2
        expanded = run_logistic(extra, self.Cy[self.ct])
        lr = 2 * (expanded.model.llf - self.logit.model.llf)
        p = float(stats.chi2.sf(max(lr, 0), 1))
        return dict(
            explanatory_model=self.logit,
            logit_specification=dict(
                H0="first feature quadratic log-odds term = 0",
                H1="quadratic term is not zero",
                lr_statistic=float(lr),
                df=1,
                p_value=p,
                reject_H0=p < 0.05,
            ),
            note="unpenalized natural prevalence explanation is separate from sampled/tuned L2 predictive model. No OLS residual normality requirement; a single quadratic test does not prove global logit linearity.",
        )

    def cls_final(self):
        self.prob = self.cls_search[self.cls_name].model.predict_proba(
            self.CX.iloc[self.ce]
        )[:, 1]
        truth = self.Cy[self.ce]
        self.cls_auc = float(roc_auc_score(truth, self.prob))
        self.cls_ap = float(average_precision_score(truth, self.prob))
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        plot_roc(truth, self.prob, ax=axes[0], overrides={"positive_label": 1})
        plot_precision_recall(
            truth, self.prob, ax=axes[1], overrides={"positive_label": 1}
        )
        plot_calibration(truth, self.prob, ax=axes[2], overrides={"positive_label": 1})
        plt.tight_layout()
        plt.show()
        return dict(
            selected=self.cls_name,
            test_indices=self.ce,
            truth=truth,
            probabilities=self.prob,
            roc_auc=self.cls_auc,
            average_precision=self.cls_ap,
            brier=float(brier_score_loss(truth, self.prob)),
            confusion_at_fixed_half=confusion_matrix(truth, self.prob >= 0.5),
            prevalence=float(truth.mean()),
            threshold=0.5,
        )

    def cls_decision(self):
        truth = self.Cy[self.ce]
        pred = self.prob >= 0.5
        tn, fp, fn, tp = confusion_matrix(truth, pred).ravel()
        cost = int(self.plan["false_negative_cost"] * fn + fp)
        return dict(
            false_negative_cost=self.plan["false_negative_cost"],
            false_positive_cost=1,
            holdout_cost_at_prespecified_half=cost,
            decision="ROC AUC measures ranking, AP reflects imbalance and Brier reflects probability quality. Resampling alters prevalence: calibrated probabilities or an operating threshold need separate training validation; do not optimize a threshold on final test.",
            logistic_not_continuous_regression="0/1 target uses logit mean; no ordinary R2/t/F/normal residual assumptions",
        )
