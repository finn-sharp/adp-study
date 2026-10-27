"""
================================================================================
audit_researched_mocks.py - 종합16~20회: 지문·배점·원행·분모·직접 수계산·출처 대응을 독립 검증합니다.
================================================================================
설명:
    - 종합16~20회: 지문·배점·원행·분모·직접 수계산·출처 대응을 독립 검증합니다.
분석 가정:
    - 저장소 경로와 입력 명세를 기준으로 실행합니다. 생성·검사 대상과 외부 의존성은 각 실행 함수의 조건을 따릅니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-27
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import json, sys, re
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import nbformat
import fitz
from researched_mock_specs import specification, SOURCES
from researched_mock_data import generate

ROOT = Path(__file__).resolve().parents[2]
MOCK = ROOT / "기출문제/모의고사"


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-9)


def frame(v):
    return pd.DataFrame(v["data"], columns=v["columns"], index=v["index"])


def metrics(truth, prediction, reported):
    y = np.asarray(truth)
    p = np.asarray(prediction)
    e = y - p
    close(np.sqrt(np.mean(e**2)), reported["rmse"])
    close(np.mean(abs(e)), reported["mae"])
    close(1 - np.sum(e**2) / np.sum((y - y.mean()) ** 2), reported["r2"])


def check_regression(folder, sec, by):
    d = pd.read_csv(folder / "data/regression.csv")
    a = by["reg_1"]
    train = a["train_indices"]
    test = a["test_indices"]
    assert set(train).isdisjoint(test) and set(train) | set(test) == set(range(240))
    x = d.iloc[train][["x1", "x2", "x3"]]
    x = x.fillna(x.median())
    x = (x - x.mean()) / x.std(ddof=0)
    x["group_B"] = (d.group.iloc[train].to_numpy() == "B").astype(float)
    if sec["settings"]["interaction"]:
        x["x1:x2"] = x.x1 * x.x2
    fit = sm.OLS(d.target.iloc[train].to_numpy(), sm.add_constant(x.to_numpy())).fit()
    reported = by["reg_2"]
    close(fit.rsquared, reported["fit_metrics"]["r2"])
    close(fit.aic, reported["fit_metrics"]["aic"])
    close(fit.bic, reported["fit_metrics"]["bic"])
    close(fit.params, frame(reported["coefficients"]).coefficient)
    final = by["reg_4"]
    close(final["truth"], d.target.iloc[test])
    metrics(final["truth"], final["predictions"], final["test_metrics"])
    metrics(final["truth"], final["baseline_predictions"], final["baseline_metrics"])
    close(
        final["delta_r2"], final["test_metrics"]["r2"] - final["baseline_metrics"]["r2"]
    )
    selected = min(final["candidates"], key=lambda r: r["cv_rmse"])["candidate"]
    assert final["selected"] == selected
    for fold in final["folds"]:
        assert set(fold["train_indices"]).isdisjoint(
            fold["validation_indices"]
        ) and set(fold["train_indices"] + fold["validation_indices"]) <= set(train)
    assert by["reg_3"]["removed_rows"] == 0


def check_classifier(folder, by):
    d = pd.read_csv(folder / "data/classification.csv")
    a = by["cls_1"]
    train = a["train_indices"]
    test = a["test_indices"]
    assert set(train).isdisjoint(test) and set(train) | set(test) == set(range(360))
    final = by["cls_3"]
    labels = final["labels"]
    y = np.asarray(final["truth"])
    p = np.asarray(final["probabilities"])
    pred = np.asarray(final["predictions"])
    close(y, d.target.iloc[test])
    close(p.sum(axis=1), 1)
    np.testing.assert_array_equal(pred, np.array(labels)[p.argmax(axis=1)])
    cm = np.array([[np.sum((y == a) & (pred == b)) for b in labels] for a in labels])
    np.testing.assert_array_equal(cm, final["confusion_matrix"])
    f = []
    auc = []
    for j, label in enumerate(labels):
        tp = cm[j, j]
        den = cm[j, :].sum() + cm[:, j].sum()
        f.append(2 * tp / den if den else 0)
        truth = y == label
        ranks = stats.rankdata(p[:, j])
        n1 = truth.sum()
        n0 = len(y) - n1
        auc.append((ranks[truth].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    close(np.mean(f), final["macro_f1"])
    close(np.mean(auc) if len(labels) > 2 else auc[1], final["roc_auc"])
    for fold in by["cls_2"]["folds"]:
        assert set(fold["train_indices"]).isdisjoint(
            fold["validation_indices"]
        ) and set(fold["train_indices"] + fold["validation_indices"]) <= set(train)
    source = pd.read_csv(folder / "data/submission.csv")
    result = pd.read_csv(folder / "실행결과/submission.csv")
    assert (
        list(result) == ["id", "prediction"]
        and result.id.tolist() == source.id.tolist()
        and result.id.is_unique
        and set(result.prediction) <= set(labels)
        and "target" not in source
    )


def check_text(folder, sec, by):
    data = pd.read_csv(folder / "data/reviews.csv", keep_default_na=False)
    lex = pd.read_csv(folder / "data/lexicon.csv").set_index("token")
    vocab = set(lex.index[lex.pos == sec["settings"]["pos"]])
    tokens = [re.findall("[a-z]+", s.lower()) for s in data.text]
    table = frame(by["text_2"]["frequency"])
    for r in table.itertuples():
        assert r.frequency == sum(
            row.count(r.token) for row in tokens
        ) and r.document_frequency == sum(r.token in row for row in tokens)
    assert set(table.token) <= vocab
    scores = pd.read_csv(folder / "실행결과/document_scores.csv")
    assert scores.id.tolist() == data.id.tolist()
    for i, row in enumerate(tokens):
        match = [
            lex.loc[t, "polarity"]
            for t in row
            if t in lex.index and lex.loc[t, "pos"] == "ADJ"
        ]
        value = np.mean(match) if match else np.nan
        close(value, scores.sentiment.iloc[i])
        assert scores.adjective_count.iloc[i] == len(match)
    assert by["text_3"]["unscored_documents"] == int(scores.sentiment.isna().sum())


def check_survey(folder, by):
    data = pd.read_csv(folder / "data/survey.csv")
    mask = data[["q1", "q2", "q3", "q4"]].notna().all(axis=1)
    x = data.loc[mask, ["q1", "q2", "q3", "q4"]].copy()
    x[["q2", "q4"]] = 6 - x[["q2", "q4"]]
    k = 4
    raw = k / (k - 1) * (1 - x.var(ddof=1).sum() / x.sum(axis=1).var(ddof=1))
    corr = x.corr().to_numpy()
    rbar = corr[np.triu_indices(k, 1)].mean()
    close(raw, by["survey_2"]["after"]["alpha"])
    close(
        k * rbar / (1 + (k - 1) * rbar), by["survey_2"]["after"]["standardized_alpha"]
    )
    score = x.mean(axis=1)
    group = data.loc[mask, "group"]
    a = score[group == "A"]
    b = score[group == "B"]
    test = stats.ttest_ind(a, b, equal_var=False)
    close(test.statistic, by["survey_4"]["statistic"])
    close(test.pvalue, by["survey_4"]["p_value"])
    ci = test.confidence_interval()
    close([ci.low, ci.high], by["survey_4"]["confidence_interval"])
    assert by["survey_1"]["excluded_rows"] == int((~mask).sum())


def check_seasonal(folder, sec, by):
    y = pd.read_csv(folder / "data/series.csv").target.to_numpy()
    period = sec["settings"]["period"]
    final = by["ts_4"]
    close(final["truth"], y[-8:])
    metrics(final["truth"], final["predictions"], final["test_metrics"])
    metrics(final["truth"], final["baseline_predictions"], final["baseline_metrics"])
    close(final["baseline_predictions"], np.resize(y[:-8][-period:], 8))
    scale = np.mean(abs(y[period:-8] - y[: -8 - period]))
    close(scale, final["mase_denominator"])
    close(final["test_metrics"]["mae"] / scale, final["mase"])
    close(
        np.mean(
            (np.array(final["truth"]) >= np.array(final["interval_lower"]))
            & (np.array(final["truth"]) <= np.array(final["interval_upper"]))
        ),
        final["interval_coverage"],
    )
    for fold in by["ts_2"]["folds"]:
        assert (
            max(fold["train_indices"]) < min(fold["validation_indices"])
            and max(fold["validation_indices"]) < len(y) - 8
        )
    valid = [r for r in by["ts_2"]["candidates"] if r["cv_rmse"] is not None]
    assert (
        min(valid, key=lambda r: r["cv_rmse"])["order"] == by["ts_2"]["selected_order"]
    )


def check_risk(folder, by):
    d = pd.read_csv(folder / "data/risk.csv").sort_values(["country", "date"])
    daily = d.groupby("country").cases.diff().fillna(d.cases)
    assert (daily < 0).sum() == by["risk_2"]["negative_corrections"] == 1
    rows = frame(by["risk_2"]["daily_rows"])
    close(rows.daily_cases, daily)
    last = d.groupby("country").tail(1).copy()
    last["score"] = (
        last.cases / last.population * 10000 + last.deaths / last.population * 100000
    )
    rank = frame(by["risk_3"]["ranking"])
    assert (
        rank.country.tolist()
        == last.sort_values("score", ascending=False).country.tolist()
    )
    close(rank.score, last.sort_values("score", ascending=False).score)
    final = by["risk_4"]
    close(final["truth"], daily[d.country == "A"].to_numpy()[-7:])
    metrics(final["truth"], final["predictions"], final["test_metrics"])


def check_lag(folder, by):
    d = pd.read_csv(folder / "data/temperature.csv")
    design = by["lag_1"]
    assert design["mismatches"] == dict(lag1=[25], lag2=[97])
    assert design["calibration_start"] > design["proper_end"]
    cut = design["test_start"]
    proper = design["calibration_start"]
    final = by["lag_4"]
    close(final["truth"], d.target.to_numpy()[2:][cut:])
    close(final["baseline_predictions"], d.target.shift(1).to_numpy()[2:][cut:])
    metrics(final["truth"], final["predictions"], final["test_metrics"])
    for fold in by["lag_2"]["folds"]:
        assert (
            max(fold["train_indices"]) + 2 < min(fold["validation_indices"])
            and max(fold["validation_indices"]) < proper
        )
    cal = by["lag_3"]
    e = np.array(cal["calibration_residuals"])
    rank = min(len(e), int(np.ceil((len(e) + 1) * 0.9)))
    assert rank == cal["quantile_rank"]
    close(np.sort(e)[rank - 1], cal["half_width"])
    close(final["mean_width"], 2 * cal["half_width"])
    close(np.array(final["predictions"]) - cal["half_width"], final["interval_lower"])
    close(
        np.mean(
            (np.array(final["truth"]) >= np.array(final["interval_lower"]))
            & (np.array(final["truth"]) <= np.array(final["interval_upper"]))
        ),
        final["coverage"],
    )


def check_energy(folder, by):
    d = pd.read_csv(folder / "data/energy.csv", parse_dates=["timestamp"])
    close(d.kwh.sum(), by["energy_1"]["total_kwh"])
    assert len(d) == 12 * 14 * 24
    assert by["energy_3"]["rows_before"] == by["energy_3"]["rows_after"] == len(d)
    assert by["energy_3"]["missing_labels"] == 0
    assign = frame(by["energy_2"]["assignments"])
    assert assign.household.is_unique
    merged = d.merge(assign, on="household", validate="many_to_one")
    summary = by["energy_4"]["summary"]
    assert sum(r["households"] for r in summary) == 12
    for r in summary:
        g = merged[merged.cluster == r["cluster"]]
        close(
            g.groupby(["household", g.timestamp.dt.date]).kwh.sum().mean(),
            r["mean_daily_kwh"],
        )


def audit():
    reports = []
    for number in range(16, 21):
        folder = MOCK / ("%d회" % number)
        spec = json.loads((folder / "문항명세.json").read_text())
        assert spec == specification(number)
        questions = [q for sec in spec["sections"] for q in sec["questions"]]
        assert (
            len(questions) == len({q["id"] for q in questions}) == 12
            and sum(q["points"] for q in questions) == 100
        )
        assert all(
            ref in SOURCES for sec in spec["sections"] for ref in sec["source_ids"]
        )
        book = nbformat.read(folder / ("모의고사_%d회_풀이.ipynb" % number), 4)
        nbformat.validate(book)
        code = [c for c in book.cells if c.cell_type == "code"]
        assert len(code) == 13
        assert all(
            c.execution_count is not None
            and not any(o.output_type == "error" for o in c.outputs)
            for c in code
        )
        answers = json.loads((folder / "실행결과/답안수치.json").read_text())
        assert [r["question"] for r in answers] == [q["id"] for q in questions]
        pdf = fitz.open(folder / ("모의고사_%d회_문제지.pdf" % number))
        text = re.sub(r"\s+", "", "".join(p.get_text() for p in pdf))
        assert len(pdf) == 7
        for q in questions:
            assert re.sub(r"\s+", "", q["text"]) in text
        with TemporaryDirectory() as temporary:
            generate(Path(temporary), spec)
            for path in (folder / "data").glob("*.csv"):
                assert (
                    path.read_bytes()
                    == (Path(temporary) / "data" / path.name).read_bytes()
                ), path
        for i, sec in enumerate(spec["sections"], 1):
            by = {
                r["task"]: r["result"]
                for r in answers
                if r["question"].startswith(str(i) + "-")
            }
            family = sec["family"]
            if family == "reg":
                check_regression(folder, sec, by)
            elif family == "cls":
                check_classifier(folder, by)
            elif family == "text":
                check_text(folder, sec, by)
            elif family == "survey":
                check_survey(folder, by)
            elif family == "ts":
                check_seasonal(folder, sec, by)
            elif family == "risk":
                check_risk(folder, by)
            elif family == "lag":
                check_lag(folder, by)
            elif family == "energy":
                check_energy(folder, by)
            elif family == "assoc":
                r = by["assoc_3"]
                close(r["count_ab"] / r["n"], r["support"])
                close(r["count_ab"] / r["count_a"], r["confidence"])
                close(r["confidence"] / (r["count_b"] / r["n"]), r["lift"])
                close(r["support"], r["reported_support"])
                close(r["confidence"], r["reported_confidence"])
                close(r["lift"], r["reported_lift"])
        report = dict(
            number=number,
            status="passed",
            questions=12,
            points=100,
            code_cells=13,
            figures=sum(
                "image/png" in o.get("data", {}) for c in code for o in c.outputs
            ),
            pdf_pages=len(pdf),
            source_ids=spec["source_ids"],
            checks=[
                "지문·배점·출처 대응",
                "고정 합성 CSV 재생성",
                "직접 OLS·지표·순위 AUC·Welch·alpha 검산",
                "CV/최종평가/calibration 경계",
                "거래·문서·시간·가구 집계 분모",
            ],
        )
        (folder / "검증결과.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        reports.append(report)
    result = dict(
        status="passed",
        exams=5,
        questions=60,
        code_cells=65,
        figures=sum(r["figures"] for r in reports),
        pdf_pages=35,
        reports=reports,
    )
    (MOCK / "공통/조사모의_검증결과.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    return result


if __name__ == "__main__":
    result = audit()
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "reports"}, ensure_ascii=False
        )
    )
