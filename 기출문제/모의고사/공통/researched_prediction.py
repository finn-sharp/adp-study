"""모의16~20회의 설명용 OLS와 학습 CV 기반 회귀·분류를 조합합니다."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    roc_curve,
    confusion_matrix,
    recall_score,
)
from imblearn.pipeline import Pipeline as SamplingPipeline
from src.Preprocessing.feature_pipeline import (
    make_feature_spec,
    build_feature_preprocessor,
)
from src.Preprocessing.numeric.imputation import build_numeric_imputation_transformer
from src.Preprocessing.numeric.scaling import build_numeric_scaling_transformer
from src.Preprocessing.resampling._common import load_resampler
from src.stats.regression.ols import run_ols
from src.stats.regression.nested_f import run_nested_f
from src.stats.regression.normality import run_normality
from src.stats.regression.linearity import run_linearity
from src.stats.regression.heteroscedasticity import run_heteroscedasticity
from src.stats.regression.collinearity import run_collinearity
from src.stats.regression.influence import run_influence
from src.Evaluation.classification.metrics import run_classification_metrics
from src.Visualization.qq.normal import plot_normal
from src.Visualization.regression.residual_fitted import plot_residual_fitted
from researched_core import regression_metrics, finish_plot


def preprocessor(drop=None):
    """매 CV clone에서 새로 적합하는 Feature 그룹 설정."""
    numeric = make_feature_spec(
        "numeric",
        ["x1", "x2", "x3"],
        [
            ("impute", build_numeric_imputation_transformer()),
            ("scale", build_numeric_scaling_transformer()),
        ],
    )
    category = make_feature_spec(
        "category",
        ["group"],
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", drop=drop, sparse_output=False),
            ),
        ],
    )
    return build_feature_preprocessor([numeric, category])


def execute(ctx, sec, step, state):
    if sec["family"] == "reg":
        return regression(ctx, sec, step, state)
    return classification(ctx, sec, step, state)


def regression(ctx, sec, step, s):
    if step == 1:
        d = pd.read_csv(ctx.folder / "data/regression.csv")
        X = d[["x1", "x2", "x3", "group"]]
        y = d.target
        train, test = train_test_split(
            np.arange(len(d)), test_size=0.3, random_state=ctx.seed
        )
        s.update(X=X, y=y, train=train, test=test)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        axes[0].hist(X.iloc[train].x1.dropna())
        axes[0].set_title("Training x1")
        axes[1].scatter(X.iloc[train].x1, y.iloc[train], s=12)
        axes[1].set_title("Training x1 vs target")
        finish_plot()
        return dict(
            n=len(d),
            train_indices=train,
            test_indices=test,
            missing=X.iloc[train].isna().sum().to_dict(),
            conclusion="학습 %d행·봉인 평가 %d행. 실제값으로 만든 파생변수는 사용하지 않는다."
            % (len(train), len(test)),
        )
    if step == 2:
        prep = preprocessor("first")
        a = prep.fit_transform(s["X"].iloc[s["train"]])
        names = list(prep.get_feature_names_out())
        X = pd.DataFrame(a, columns=names)
        # The interaction is a predeclared explanatory design, independent of model search.
        if sec["settings"]["interaction"]:
            X["x1:x2"] = X.iloc[:, 0] * X.iloc[:, 1]
        y = s["y"].iloc[s["train"]].to_numpy()
        full = run_ols(X, y)
        simple = run_ols(X.iloc[:, :1], y)
        s.update(ols=full, design=X)
        return dict(
            coefficient_scale="수치형은 학습 표준편차1단위, 범주는 B−A. 교호작용은 표준화 좌표의 곱.",
            coefficients=full.coefficients,
            fit_metrics=full.fit_metrics,
            simple_fit_metrics=simple.fit_metrics,
            nested_f=run_nested_f(simple.model, full.model).tests,
            conclusion="학습 R² %.4f는 봉인 평가 성능이 아니다. 계수 t·F는 사전 OLS 사양과 추론 가정에 조건부다."
            % full.fit_metrics["r2"],
        )
    if step == 3:
        model = s["ols"].model
        normal = run_normality(model)
        linear = run_linearity(model)
        hetero = run_heteroscedasticity(model)
        vif = run_collinearity(s["design"])
        influence = run_influence(model)
        influence_table = influence.table.copy()
        influence_table["original_position"] = s["train"][
            influence_table.position.to_numpy(dtype=int)
        ]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        plot_normal(s["ols"].residuals, ax=axes[0])
        plot_residual_fitted(model, ax=axes[1])
        finish_plot()
        return dict(
            normality=normal.tests,
            linearity=linear.tests,
            heteroscedasticity=hetero.tests,
            vif=vif.table,
            influence_top=influence_table.sort_values(
                "cooks_distance", ascending=False
            ).head(5),
            removed_rows=0,
            conclusion="본 풀이에서는 영향 관측을 삭제하지 않는다. 진단은 사양 점검이며 비기각을 가정의 증명으로 읽지 않는다.",
        )
    X = s["X"]
    y = s["y"]
    train = s["train"]
    test = s["test"]
    cv = KFold(3, shuffle=True, random_state=ctx.seed)
    folds = list(cv.split(X.iloc[train]))
    records = []
    fitted = {}
    candidates = [
        ("baseline", LinearRegression(), {}, False),
        ("polynomial", LinearRegression(), {}, True),
        (
            "forest",
            RandomForestRegressor(n_estimators=60, random_state=ctx.seed),
            {"model__max_depth": [3, 5]},
            False,
        ),
    ]
    for name, model, grid, poly in candidates:
        parts = [("features", preprocessor("first"))]
        if poly:
            parts.append(("poly", PolynomialFeatures(2, include_bias=False)))
        parts.append(("model", model))
        search = GridSearchCV(
            Pipeline(parts),
            grid,
            cv=folds,
            scoring="neg_root_mean_squared_error",
            error_score="raise",
            n_jobs=1,
        )
        search.fit(X.iloc[train], y.iloc[train])
        fitted[name] = search.best_estimator_
        records.append(
            dict(
                candidate=name,
                cv_rmse=-search.best_score_,
                parameters=str(search.best_params_),
            )
        )
    selected = min(records, key=lambda r: r["cv_rmse"])["candidate"]
    prediction = fitted[selected].predict(X.iloc[test])
    baseline = fitted["baseline"].predict(X.iloc[test])
    score = regression_metrics(y.iloc[test], prediction)
    base = regression_metrics(y.iloc[test], baseline)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].scatter(y.iloc[test], prediction)
    axes[0].set_title("Holdout: observed vs predicted")
    axes[1].bar([r["candidate"] for r in records], [r["cv_rmse"] for r in records])
    axes[1].set_title("Training CV RMSE")
    finish_plot()
    return dict(
        candidates=records,
        selected=selected,
        test_metrics=score,
        baseline_metrics=base,
        delta_r2=score["r2"] - base["r2"],
        truth=y.iloc[test].to_numpy(),
        predictions=prediction,
        baseline_predictions=baseline,
        folds=[
            dict(train_indices=train[a], validation_indices=train[b]) for a, b in folds
        ],
        conclusion="CV에서 %s 선택, 보류 ΔR²=%.4f. 예측모형의 개선은 OLS 잔차 가정 충족이나 인과의 근거가 아니다."
        % (selected, score["r2"] - base["r2"]),
    )


def classifier_candidates(seed):
    lr = LogisticRegression(max_iter=1500, random_state=seed)
    rf = RandomForestClassifier(n_estimators=60, max_depth=4, random_state=seed)
    gb = GradientBoostingClassifier(n_estimators=40, max_depth=2, random_state=seed)
    return [
        ("logit", lr, {"model__C": [0.3, 1.0]}),
        ("forest", rf, {"model__max_depth": [3, 5]}),
        ("boost", gb, {}),
        (
            "soft_voting",
            VotingClassifier(
                estimators=[("lr", lr), ("rf", rf), ("gb", gb)], voting="soft"
            ),
            {},
        ),
    ]


def classification(ctx, sec, step, s):
    if step == 1:
        d = pd.read_csv(ctx.folder / "data/classification.csv")
        X = d[["x1", "x2", "x3", "group"]]
        y = d.target
        train, test = train_test_split(
            np.arange(len(d)), test_size=0.3, random_state=ctx.seed, stratify=y
        )
        s.update(X=X, y=y, train=train, test=test)
        table = pd.crosstab(X.iloc[train].group, y.iloc[train], normalize="index")
        fig, ax = plt.subplots(figsize=(5, 3))
        table.plot.bar(stacked=True, ax=ax)
        ax.set_title("Training class distribution")
        finish_plot()
        return dict(
            classes=sorted(y.unique()),
            class_counts=y.value_counts().sort_index().to_dict(),
            train_indices=train,
            test_indices=test,
            conclusion="층화 분리 후 평가 발생률은 유지한다. 클래스 순서와 확률 열 순서를 함께 저장한다.",
        )
    if step == 2:
        X = s["X"].iloc[s["train"]]
        y = s["y"].iloc[s["train"]]
        folds = list(
            StratifiedKFold(3, shuffle=True, random_state=ctx.seed).split(X, y)
        )
        records = []
        fitted = {}
        for name, model, grid in classifier_candidates(ctx.seed):
            parts = [("features", preprocessor())]
            if sec["settings"]["sampling"]:
                parts.append(
                    (
                        "sample",
                        load_resampler(
                            "random_over", overrides={"random_state": ctx.seed}
                        ),
                    )
                )
            parts.append(("model", model))
            search = GridSearchCV(
                SamplingPipeline(parts),
                grid,
                cv=folds,
                scoring="f1_macro",
                error_score="raise",
                n_jobs=1,
            )
            search.fit(X, y)
            fitted[name] = search.best_estimator_
            records.append(
                dict(
                    candidate=name,
                    cv_macro_f1=search.best_score_,
                    parameters=str(search.best_params_),
                )
            )
        chosen = max(records, key=lambda r: r["cv_macro_f1"])["candidate"]
        s.update(model=fitted[chosen], selected=chosen)
        return dict(
            candidates=records,
            selected=chosen,
            folds=[
                dict(train_indices=s["train"][a], validation_indices=s["train"][b])
                for a, b in folds
            ],
            sampling=sec["settings"]["sampling"],
            conclusion="학습 CV macro F1로 %s 선택. 확률 평균 앙상블이 항상 개별 후보보다 좋지는 않다."
            % chosen,
        )
    if step == 3:
        model = s["model"]
        X = s["X"].iloc[s["test"]]
        y = s["y"].iloc[s["test"]].to_numpy()
        p = model.predict_proba(X)
        labels = model.classes_
        pred = labels[p.argmax(axis=1)]
        multi = len(labels) > 2
        evaluation = run_classification_metrics(
            y,
            pred,
            y_score=p,
            overrides=dict(labels=labels.tolist(), average="macro", positive_label=1),
        ).value
        auc = evaluation["roc_auc"]
        recall = recall_score(y, pred, labels=labels, average=None, zero_division=0)
        f1 = evaluation["f1"]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        cm = confusion_matrix(y, pred, labels=labels)
        axes[0].imshow(cm, cmap="Blues")
        axes[0].set_title("Holdout confusion matrix")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")
        axes[0].set_xticks(range(len(labels)), labels=labels)
        axes[0].set_yticks(range(len(labels)), labels=labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                axes[0].text(j, i, str(cm[i, j]), ha="center", va="center")
        for j, label in enumerate(labels):
            if not multi and j == 0:
                continue
            fpr, tpr, _ = roc_curve((y == label).astype(int), p[:, j])
            axes[1].plot(fpr, tpr, label=str(label))
        axes[1].legend(title="Class")
        axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
        axes[1].set_title("One-vs-rest ROC")
        axes[1].set_xlabel("False positive rate")
        axes[1].set_ylabel("True positive rate")
        finish_plot()
        return dict(
            labels=labels,
            confusion_matrix=cm,
            recall_by_class=recall,
            macro_f1=f1,
            roc_auc=auc,
            auc_definition="macro one-vs-rest" if multi else "positive class 1",
            truth=y,
            predictions=pred,
            probabilities=p,
            test_indices=s["test"],
            conclusion="보류 macro F1=%.4f, AUC=%.4f. 최저 재현율 클래스 %s=%.4f도 함께 검토해야 한다. 높은 AUC는 확률 보정·최적 비용 임계값을 보장하지 않는다."
            % (f1, auc, labels[np.argmin(recall)], min(recall)),
        )
    new = pd.read_csv(ctx.folder / "data/submission.csv")
    prediction = s["model"].predict(new[["x1", "x2", "x3", "group"]])
    table = pd.DataFrame(dict(id=new.id, prediction=prediction))
    assert table.id.is_unique and table.prediction.notna().all()
    table.to_csv(ctx.out / "submission.csv", index=False)
    return dict(
        submission_rows=len(table),
        unique_ids=bool(table.id.is_unique),
        predicted_counts=table.prediction.value_counts().sort_index().to_dict(),
        head=table.head(),
        conclusion="제출 %d행을 원 순서로 저장했다. 제출 자료의 정답이 없으므로 실제 제출 정확도는 알 수 없다."
        % len(table),
    )
