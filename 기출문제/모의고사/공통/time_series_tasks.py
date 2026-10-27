"""
================================================================================
time_series_tasks.py - 응용 모의 문항의 시계열 조합. 재사용 계산은 src 모듈을 호출합니다.
================================================================================
설명:
    - 응용 모의 문항의 시계열 조합. 재사용 계산은 src 모듈을 호출합니다.
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

from matplotlib import pyplot as plt
from src.TimeSeries.rolling_selection import run_rolling_selection
from src.Visualization.time_series.timeline import plot_timeline
from src.stats.tests.TimeSeries.adf import run_adf


class TimeSeriesTasks:
    """해당 분석의 학습 상태와 문항 결과를 연결합니다."""

    def ts_design(self):
        horizon = self.plan["horizon"]
        self.ts_train = self.ts.iloc[:-horizon]
        self.ts_test = self.ts.iloc[-horizon:]
        return dict(
            n=len(self.ts),
            frequency="daily",
            period=self.plan["period"],
            train_end=str(self.ts_train.index[-1]),
            test_start=str(self.ts_test.index[0]),
            horizon=horizon,
            training_adf=run_adf(self.ts_train.to_numpy()),
            rule="chronological only; transforms and lag selection must use past observations",
        )

    def ts_tuning(self):
        self.forecast = run_rolling_selection(
            self.ts,
            overrides={
                "horizon": self.plan["horizon"],
                "validation_size": 7,
                "folds": 2,
                "period": self.plan["period"],
                "candidate_orders": [[0, 0, 0], [1, 0, 0]],
                "maxiter": 100,
            },
        )
        return dict(
            comparison=self.forecast.comparison,
            folds=self.forecast.folds,
            selected_order=self.forecast.selected_order,
            information=self.forecast.residual_diagnostics,
        )

    def ts_diagnostics(self):
        return dict(
            diagnostics=self.forecast.residual_diagnostics,
            train_information=dict(
                aic=self.forecast.model.aic, bic=self.forecast.model.bic
            ),
            meaning="Ljung-Box subtracts p+q+P+Q degrees of freedom and removes initialization. Nonrejection does not prove independence. Compare differencing/families via rolling error; AIC/BIC are not predictive R2.",
        )

    def ts_final(self):
        f = self.forecast
        fig, ax = plt.subplots(figsize=(9, 4))
        plot_timeline(
            self.ts.index[-35:],
            self.ts.iloc[-35:],
            ax=ax,
            overrides={"rolling_window": 0},
        )
        ax.plot(
            self.ts_test.index, f.predictions, label="Selected forecast", color="C1"
        )
        ax.plot(
            self.ts_test.index,
            f.baseline_predictions,
            label="Naive forecast",
            color="C2",
        )
        ax.fill_between(
            self.ts_test.index,
            f.model.prediction_lower,
            f.model.prediction_upper,
            alpha=0.2,
            label="Conditional interval",
        )
        ax.legend()
        ax.set_title("Future holdout after rolling selection")
        plt.tight_layout()
        plt.show()
        return dict(
            test_metrics=f.test_metrics,
            test_indices=f.test_indices,
            truth=self.ts_test.to_numpy(),
            predictions=f.predictions,
            baseline_predictions=f.baseline_predictions,
            interval_lower=f.model.prediction_lower,
            interval_upper=f.model.prediction_upper,
            interval="conditional on fitted selected model; selection uncertainty not included",
        )
