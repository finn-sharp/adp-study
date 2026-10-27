"""
================================================================================
researched_temporal.py - 시간 경계·집계 단위가 중요한 모의16~20회 분석 조합.
================================================================================
설명:
    - 시간 경계·집계 단위가 중요한 모의16~20회 분석 조합.
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
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from statsmodels.stats.diagnostic import acorr_ljungbox
from src.TimeSeries.sarima import run_sarima
from src.TimeSeries.arima import run_arima
from src.stats.tests.TimeSeries.adf import run_adf
from src.Evaluation.clustering.silhouette import run_silhouette
from src.Visualization.qq.normal import plot_normal
from researched_core import regression_metrics, finish_plot


def execute(ctx, sec, step, state):
    return {"ts": seasonal, "risk": risk, "lag": lag, "energy": energy}[sec["family"]](
        ctx, sec, step, state
    )


def sarima(train, order, period, steps):
    return run_sarima(
        np.asarray(train),
        overrides=dict(
            order=list(order),
            seasonal_order=[0, 1, 0, period],
            steps=steps,
            maxiter=200,
            require_convergence=True,
            trend=None,
        ),
    )


def seasonal(ctx, sec, step, s):
    if step == 1:
        d = pd.read_csv(ctx.folder / "data/series.csv", parse_dates=["date"])
        assert (
            d.date.is_unique
            and d.date.is_monotonic_increasing
            and d.target.notna().all()
        )
        p = sec["settings"]["period"]
        y = d.target.to_numpy()
        train = y[:-8]
        s.update(y=y, train=train, dates=d.date, p=p)
        adf = run_adf(train)
        diff = run_adf(train[p:] - train[:-p])
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        axes[0].plot(train)
        axes[0].set_title("Training level")
        axes[1].plot(train[p:] - train[:-p])
        axes[1].set_title("Training seasonal difference")
        finish_plot()
        return dict(
            train_end=87,
            test_start=88,
            period=p,
            adf_p=adf.p_value,
            seasonal_difference_adf_p=diff.p_value,
            conclusion="마지막8시점을 봉인했다. ADF는 단위근에 관한 검정이며 정규성 검정과 다르다.",
        )
    if step == 2:
        n = len(s["train"])
        folds = [
            (np.arange(n - 8), np.arange(n - 8, n - 4)),
            (np.arange(n - 4), np.arange(n - 4, n)),
        ]
        records = []
        for order in [(1, 0, 0), (0, 1, 1), (1, 1, 0)]:
            errors = []
            failed = []
            for a, b in folds:
                try:
                    fit = sarima(s["train"][a], order, s["p"], len(b))
                    errors.append(
                        float(np.sqrt(np.mean((s["train"][b] - fit.predictions) ** 2)))
                    )
                except (ValueError, RuntimeError) as error:
                    failed.append(str(error))
            records.append(
                dict(
                    order=list(order),
                    cv_rmse=float(np.mean(errors)) if not failed else None,
                    failures=failed,
                )
            )
        valid = [r for r in records if r["cv_rmse"] is not None]
        if not valid:
            raise ValueError("모든 SARIMA 후보가 실패했습니다.")
        selected = min(valid, key=lambda r: r["cv_rmse"])
        s["order"] = selected["order"]
        return dict(
            candidates=records,
            selected_order=s["order"],
            folds=[dict(train_indices=a, validation_indices=b) for a, b in folds],
            conclusion="rolling 평균 RMSE 최소 차수 %s를 최종평가 이전에 선택했다."
            % s["order"],
        )
    if step == 3:
        fit = sarima(s["train"], s["order"], s["p"], 8)
        s["fit"] = fit
        burn = s["p"] + s["order"][1] + 1
        res = fit.residuals[burn:]
        df = s["order"][0] + s["order"][2]
        lag_value = max(df + 2, min(12, len(res) // 4))
        lb = acorr_ljungbox(res, lags=[lag_value], model_df=df, return_df=True)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        axes[0].plot(res)
        axes[0].set_title("Residuals after burn-in")
        plot_normal(res, ax=axes[1])
        finish_plot()
        return dict(
            aic=fit.aic,
            bic=fit.bic,
            converged=fit.converged,
            burn_in=burn,
            ljung_box=lb,
            model_df=df,
            conclusion="수렴=%s, 초기%d시점 제외. Ljung-Box 비기각은 독립성 전체나 구간 보정을 증명하지 않는다."
            % (fit.converged, burn),
        )
    fit = s["fit"]
    truth = s["y"][-8:]
    pred = fit.predictions
    naive = np.resize(s["train"][-s["p"] :], 8)
    scale = np.mean(abs(s["train"][s["p"] :] - s["train"][: -s["p"]]))
    score = regression_metrics(truth, pred)
    base = regression_metrics(truth, naive)
    coverage = np.mean(
        (truth >= fit.prediction_lower) & (truth <= fit.prediction_upper)
    )
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(s["dates"].iloc[-8:], truth, label="Actual")
    ax.plot(s["dates"].iloc[-8:], pred, label="SARIMA")
    ax.plot(s["dates"].iloc[-8:], naive, label="Seasonal naive")
    ax.fill_between(
        s["dates"].iloc[-8:], fit.prediction_lower, fit.prediction_upper, alpha=0.2
    )
    ax.legend()
    ax.set_title("Sealed final horizon")
    finish_plot()
    return dict(
        test_metrics=score,
        baseline_metrics=base,
        mase=score["mae"] / scale,
        baseline_mase=base["mae"] / scale,
        mase_denominator=scale,
        interval_coverage=coverage,
        truth=truth,
        predictions=pred,
        baseline_predictions=naive,
        interval_lower=fit.prediction_lower,
        interval_upper=fit.prediction_upper,
        conclusion="원단위 RMSE %.4f, naive %.4f, MASE %.4f. 8개 시점 구간 포함률 %.3f를 보장으로 해석하지 않는다."
        % (score["rmse"], base["rmse"], score["mae"] / scale, coverage),
    )


def risk(ctx, sec, step, s):
    if step == 1:
        d = pd.read_csv(ctx.folder / "data/risk.csv", parse_dates=["date"]).sort_values(
            ["country", "date"]
        )
        assert not d.duplicated(["country", "date"]).any()
        assert (d.groupby("country").population.nunique() == 1).all()
        s["d"] = d
        fig, ax = plt.subplots(figsize=(6, 3))
        for c, g in d.groupby("country"):
            ax.plot(g.date, g.cases, label=c)
        ax.legend()
        ax.set_title("Synthetic cumulative cases")
        finish_plot()
        return dict(
            rows=len(d),
            countries=d.country.nunique(),
            duplicate_keys=0,
            conclusion="가상 국가별 누적 신고 자료다. 실제 개인 위험이나 방역 정책으로 일반화하지 않는다.",
        )
    if step == 2:
        d = s["d"].copy()
        for col in ["cases", "deaths"]:
            d["daily_" + col] = d.groupby("country")[col].diff().fillna(d[col])
        s["daily"] = d
        negative = d[d.daily_cases < 0]
        return dict(
            negative_corrections=len(negative),
            correction_rows=negative[["country", "date", "daily_cases"]],
            daily_rows=d[["country", "date", "daily_cases", "daily_deaths"]],
            conclusion="음수 정정%d건을0으로 자르지 않고 보존했다. 국가별 일별 합은 마지막 누적과 같다."
            % len(negative),
        )
    if step == 3:
        last = s["daily"].groupby("country", sort=True).tail(1).copy()
        last["case_rate_10000"] = last.cases / last.population * 10000
        last["death_rate_100000"] = last.deaths / last.population * 100000
        last["score"] = last.case_rate_10000 + last.death_rate_100000
        last = last.sort_values("score", ascending=False)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(last.country, last.score)
        ax.set_title("Defined training-only risk index")
        finish_plot()
        return dict(
            ranking=last[["country", "case_rate_10000", "death_rate_100000", "score"]],
            conclusion="최상위 가상 국가 %s. 이는 지문에서 정의한 두 스케일 합이며 공인 보건지수나 감염 확률이 아니다."
            % last.iloc[0].country,
        )
    y = s["daily"].loc[s["daily"].country == "A", "daily_cases"].to_numpy()
    train = y[:-7]
    truth = y[-7:]
    fit = run_arima(
        train, overrides=dict(order=[1, 0, 0], steps=7, trend="c", maxiter=200)
    )
    base = np.repeat(np.mean(train[-7:]), 7)
    score = regression_metrics(truth, fit.predictions)
    baseline = regression_metrics(truth, base)
    return dict(
        train_end=len(train) - 1,
        test_start=len(train),
        test_metrics=score,
        baseline_metrics=baseline,
        truth=truth,
        predictions=fit.predictions,
        baseline_predictions=base,
        conclusion="ARIMA RMSE %.4f, 과거7일 평균 %.4f. 연속 모형은 건수의 정수·비음수 제약을 자동 보장하지 않는다."
        % (score["rmse"], baseline["rmse"]),
    )


def lag(ctx, sec, step, s):
    if step == 1:
        d = pd.read_csv(ctx.folder / "data/temperature.csv", parse_dates=["date"])
        assert d.date.is_unique and d.date.is_monotonic_increasing
        mismatches = {}
        for k in (1, 2):
            expected = d.target.shift(k)
            valid = expected.notna()
            bad = valid & ~np.isclose(d["lag" + str(k)], expected, equal_nan=True)
            mismatches["lag" + str(k)] = d.index[bad].tolist()
            d["lag" + str(k)] = expected
        frame = d.iloc[2:].copy()
        day = frame.date.dt.dayofyear
        X = pd.DataFrame(
            dict(
                lag1=frame.lag1,
                lag2=frame.lag2,
                sin=np.sin(2 * np.pi * day / 365),
                cos=np.cos(2 * np.pi * day / 365),
            )
        )
        y = frame.target.to_numpy()
        cut = int(len(y) * 0.7)
        proper = int(cut * 0.8)
        s.update(X=X, y=y, cut=cut, proper=proper)
        return dict(
            mismatches=mismatches,
            excluded_features=["date", "target", "audit"],
            valid_rows=len(y),
            training_end=cut - 1,
            test_start=cut,
            proper_end=proper - 1,
            calibration_start=proper,
            conclusion="오류 지연값2건을 원 target의 과거값으로 재구성했다. calibration과 최종평가는 후보 선택에 사용하지 않는다.",
        )
    if step == 2:
        X = s["X"].iloc[: s["proper"]]
        y = s["y"][: s["proper"]]
        folds = list(TimeSeriesSplit(3, gap=2).split(X))
        records = []
        models = {}
        for name, model, grid in [
            (
                "forest",
                RandomForestRegressor(n_estimators=60, random_state=ctx.seed),
                {"model__max_depth": [3, 5]},
            ),
            ("svr", SVR(epsilon=0.1), {"model__C": [1, 10]}),
        ]:
            pipe = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("model", model),
                ]
            )
            search = GridSearchCV(
                pipe,
                grid,
                cv=folds,
                scoring="neg_root_mean_squared_error",
                error_score="raise",
                n_jobs=1,
            )
            search.fit(X, y)
            models[name] = search.best_estimator_
            records.append(
                dict(
                    candidate=name,
                    cv_rmse=-search.best_score_,
                    parameters=str(search.best_params_),
                )
            )
        selected = min(records, key=lambda r: r["cv_rmse"])["candidate"]
        s["model"] = models[selected]
        s["selected"] = selected
        return dict(
            candidates=records,
            selected=selected,
            folds=[dict(train_indices=a, validation_indices=b) for a, b in folds],
            conclusion="proper 학습 구간의 시간 CV에서 %s 선택. calibration과 최종 기간은 탐색에서 제외했다."
            % selected,
        )
    if step == 3:
        a = s["proper"]
        b = s["cut"]
        errors = np.abs(s["y"][a:b] - s["model"].predict(s["X"].iloc[a:b]))
        rank = min(len(errors), int(np.ceil((len(errors) + 1) * 0.9)))
        width = np.sort(errors)[rank - 1]
        s["width"] = width
        return dict(
            calibration_size=len(errors),
            quantile_rank=rank,
            half_width=width,
            calibration_residuals=errors,
            proper_end=a - 1,
            calibration_start=a,
            calibration_end=b - 1,
            conclusion="시간순 calibration 절대잔차 경험 반폭 %.4f. 시간 의존성 때문에 교환가능성 기반 보장을 주장하지 않는다."
            % width,
        )
    X = s["X"].iloc[s["cut"] :]
    truth = s["y"][s["cut"] :]
    prediction = s["model"].predict(X)
    lo = prediction - s["width"]
    hi = prediction + s["width"]
    coverage = np.mean((truth >= lo) & (truth <= hi))
    naive = X.lag1.to_numpy()
    score = regression_metrics(truth, prediction)
    base = regression_metrics(truth, naive)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(truth, label="Actual")
    ax.plot(prediction, label=s["selected"])
    ax.fill_between(np.arange(len(truth)), lo, hi, alpha=0.2)
    ax.legend()
    ax.set_title("Daily one-step evaluation")
    finish_plot()
    return dict(
        test_metrics=score,
        baseline_metrics=base,
        coverage=coverage,
        mean_width=2 * s["width"],
        truth=truth,
        predictions=prediction,
        baseline_predictions=naive,
        interval_lower=lo,
        interval_upper=hi,
        conclusion="매일 직전 실제값이 공개되는1-step RMSE %.4f(naive %.4f), 경험구간 포함률 %.3f. 기준보다 나쁘거나 포함률이 낮으면 그 실패를 보고한다. 여러 날 동시 예측이나 미래 보장으로 해석하지 않는다."
        % (score["rmse"], base["rmse"], coverage),
    )


def energy(ctx, sec, step, s):
    if step == 1:
        d = pd.read_csv(ctx.folder / "data/energy.csv", parse_dates=["timestamp"])
        assert not d.duplicated(["household", "timestamp"]).any()
        assert d.kwh.ge(0).all()
        d["hour"] = d.timestamp.dt.hour
        d["weekday"] = d.timestamp.dt.dayofweek
        d["date"] = d.timestamp.dt.date
        daily = d.groupby(["household", "date"]).kwh.sum().reset_index()
        profile = d.pivot_table(
            index="household", columns="hour", values="kwh", aggfunc="mean"
        )
        assert profile.shape == (12, 24)
        s.update(d=d, daily=daily, profile=profile)
        return dict(
            observations=len(d),
            households=d.household.nunique(),
            household_days=len(daily),
            total_kwh=d.kwh.sum(),
            daily_mean_kwh=daily.kwh.mean(),
            conclusion="시간 전력량을 일별로 합산했다. 1시간 평균 프로필은 같은 시간대14일의 평균이며 일별 합과 다르다.",
        )
    if step == 2:
        z = StandardScaler().fit_transform(s["profile"])
        records = []
        models = {}
        for k in (2, 3, 4):
            model = KMeans(n_clusters=k, n_init=10, random_state=ctx.seed).fit(z)
            value = run_silhouette(y_pred=model.labels_, X=z).value
            records.append(dict(k=k, silhouette=value, inertia=model.inertia_))
            models[k] = model
        best = max(records, key=lambda r: (r["silhouette"], -r["k"]))
        s["labels"] = pd.DataFrame(
            dict(household=s["profile"].index, cluster=models[best["k"]].labels_)
        )
        s["k"] = best["k"]
        return dict(
            candidates=records,
            selected_k=s["k"],
            assignments=s["labels"],
            conclusion="같은 표준화 공간의 silhouette로 k=%d를 골랐다. 전체 자료의 탐색이며 새 가구 예측 성능이 아니다."
            % s["k"],
        )
    if step == 3:
        d = s["d"].merge(
            s["labels"], on="household", how="left", validate="many_to_one"
        )
        assert len(d) == len(s["d"]) and d.cluster.notna().all()
        s["merged"] = d
        low = d.groupby(["cluster", "weekday", "hour"]).kwh.mean().min()
        high = d.groupby(["cluster", "weekday", "hour"]).kwh.mean().max()
        fig, axes = plt.subplots(1, s["k"], figsize=(4 * s["k"], 3), squeeze=False)
        for k, ax in enumerate(axes[0]):
            table = (
                d[d.cluster == k]
                .pivot_table(
                    index="weekday", columns="hour", values="kwh", aggfunc="mean"
                )
                .reindex(index=range(7), columns=range(24))
            )
            heat = ax.imshow(table, aspect="auto", vmin=low, vmax=high, cmap="viridis")
            fig.colorbar(heat, ax=ax, label="kWh", fraction=0.046, pad=0.04)
            ax.set_title("Cluster %d: mean kWh" % k)
            ax.set_xlabel("Hour")
            ax.set_ylabel("Weekday (Mon=0)")
        finish_plot()
        return dict(
            rows_before=len(s["d"]),
            rows_after=len(d),
            missing_labels=int(d.cluster.isna().sum()),
            common_vmin=low,
            common_vmax=high,
            conclusion="라벨 연결 전후%d행으로 같다. 모든 군집 히트맵에 같은 색 범위를 적용했다."
            % len(d),
        )
    d = s["merged"]
    rows = []
    for k, g in d.groupby("cluster"):
        h = g.groupby("hour").kwh.mean()
        rows.append(
            dict(
                cluster=int(k),
                households=g.household.nunique(),
                mean_daily_kwh=g.groupby(["household", "date"]).kwh.sum().mean(),
                peak_hour=int(h.idxmax()),
                night_fraction=g.loc[g.hour < 6, "kwh"].sum() / g.kwh.sum(),
            )
        )
    return dict(
        summary=rows,
        conclusion="피크·야간 비율은 집단 평균 패턴이다. 절감 효과는 가구 수준 실험과 미래 운영 자료에서 별도로 검증해야 한다.",
    )
