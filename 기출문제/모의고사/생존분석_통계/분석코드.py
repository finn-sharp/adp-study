"""이 파일 위치를 기준으로 위에서 아래로 실행하는 모의 풀이."""
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)

from pathlib import Path
from itertools import combinations
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import sys
from pathlib import Path

repo = next(
    p for p in [Path.cwd()] + list(Path.cwd().parents) if (p / "src/Recipes").is_dir()
)
sys.path.insert(0, str(repo))
from src.stats.survival.kaplan_meier import run_kaplan_meier
from src.stats.survival.rmst import run_rmst
from src.stats.survival.logrank import run_logrank
from src.stats.survival.cox import run_cox


def km_at(km, t):
    return float(np.r_[1.0, km.survival][np.searchsorted(km.times, t, side="right")])


def plot_km(km, ax):
    times = np.r_[0.0, km.times]
    values = np.r_[1.0, km.survival]
    line = ax.step(times, values, where="post", label=km.parameters["label"])[0]
    ax.fill_between(
        times,
        np.r_[1.0, km.confidence_lower],
        np.r_[1.0, km.confidence_upper],
        step="post",
        alpha=0.15,
        color=line.get_color(),
    )
    mask = km.censored > 0
    ax.plot(km.times[mask], km.survival[mask], "+", color=line.get_color())
    ax.legend()


def add_risk_counts(kms, ax):
    ticks = [0, 10, 20, 30, 40]
    rows = []
    labels = []
    for g, km in kms.items():
        removed = km.events + km.censored
        rows.append(
            [str(int(km.sample_size - removed[km.times < t].sum())) for t in ticks]
        )
        labels.append(g)
    table = ax.table(
        cellText=rows,
        rowLabels=labels,
        colLabels=ticks,
        loc="bottom",
        bbox=[0, -0.5, 1, 0.23],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)


ALPHA = 0.05
results = []


def record(question, **kwargs):
    results.append(dict(question=question, **kwargs))


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    s = p[order]
    out = np.empty(len(p))
    out[order] = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * s))
    return out.tolist()


def report(question, method, H0, H1, stat, p, df, evidence, adjusted=None):
    used = float(p) if adjusted is None else float(adjusted)
    reject = used < ALPHA
    conclusion = evidence + ("가 있다." if reject else "가 충분하지 않다.")
    row = dict(
        method=method,
        H0=H0,
        H1=H1,
        statistic=float(stat),
        p_value=float(p),
        df=df,
        alpha=ALPHA,
        reject_H0=bool(reject),
        conclusion=conclusion,
    )
    if adjusted is not None:
        row["p_holm"] = float(adjusted)
    record(question, **row)
    display(pd.DataFrame([row]))


supplier = pd.read_csv("data/supplier.csv")
subscription = pd.read_csv("data/subscription.csv")
equipment = pd.read_csv("data/equipment.csv")

# 문항 1-1

display(supplier.groupby("group").agg(n=("event", "size"), failures=("event", "sum")))
print("우측 중도절단: 실제 수명은 기록된 종료 시점보다 길다. event=0은 고장 아님.")

# 문항 1-2

a = supplier[supplier.group == "A"]
km_a = run_kaplan_meier(a.time, a.event, overrides={"label": "A"})
display(
    pd.DataFrame(
        dict(
            time=km_a.times,
            at_risk=km_a.at_risk,
            observed=km_a.events,
            censored=km_a.censored,
        )
    )
)
hand12 = (15 / 16) * (14 / 15)
assert np.isclose(km_at(km_a, 12), hand12)
print("A: n(12)=15, d(12)=1, c(12)=1, S(12)=", hand12)

# 문항 1-3

kms = {}
rows = []
for g, d in supplier.groupby("group"):
    km = run_kaplan_meier(d.time, d.event, overrides={"label": str(g)})
    kms[g] = km
    for t in [25, 30, 35]:
        s = float(km_at(km, t))
        rows.append(dict(group=g, month=t, S=s, F=1 - s))
probabilities = pd.DataFrame(rows)
display(probabilities)
conditional = float(km_at(kms["A"], 35) / km_at(kms["A"], 25))
print("P(T>35 | T>25), A =", conditional)
record("1-3", probabilities=rows, conditional_A=conditional)

# 문항 1-4

fig, ax = plt.subplots(figsize=(7, 4))
for km in kms.values():
    plot_km(km, ax)
add_risk_counts(kms, ax)
ax.set(
    xlabel="Months",
    ylabel="Survival probability",
    title="Supplier survival (pointwise 95% CI)",
)
plt.tight_layout()
plt.show()
medians = {
    g: (float(k.median_survival_time) if k.median_survival_time is not None else None)
    for g, k in kms.items()
}
print("Median (None = not reached):", medians)
print("60개월: 관찰 범위 밖으로 추정 보고하지 않는다.")
record("1-4", median_months=medians, outside_followup_60_months="not estimated")

# 문항 1-5

b = supplier[supplier.group == "B"]
lr = run_logrank(supplier.time, supplier.event, supplier.group)
report(
    "1-5",
    "Log-rank",
    "S_A(t)=S_B(t) for all t",
    "different for some t",
    lr.statistic,
    lr.p_value,
    1,
    "공급사별 전체 생존분포가 다르다는 근거",
)
print("생존분포 차이는 원인 효과나 특정 보증기간의 경제적 최적성을 증명하지 않는다.")

# 문항 2-1

display(
    subscription.groupby("group").agg(
        n=("event", "size"), events=("event", "sum"), max_time=("time", "max")
    )
)
display(
    subscription.groupby("group")
    .event.apply(lambda x: 1 - x.mean())
    .rename("censor_fraction")
)
sub_kms = {}
fig, ax = plt.subplots(figsize=(7, 4))
for g, d in subscription.groupby("group"):
    k = run_kaplan_meier(d.time, d.event, overrides={"label": str(g)})
    sub_kms[g] = k
    plot_km(k, ax)
ax.set(xlabel="Months", ylabel="Retention probability")
plt.tight_layout()
plt.show()
retention24 = {g: km_at(k, 24) for g, k in sub_kms.items()}
print("S(24):", retention24)
record("2-1", retention24=retention24)

# 문항 2-2

overall = run_logrank(subscription.time, subscription.event, subscription.group)
report(
    "2-2",
    "Multivariate log-rank",
    "S_A=S_B=S_C for all t",
    "at least one differs",
    overall.statistic,
    overall.p_value,
    2,
    "정책별 전체 유지기간 분포가 다르다는 근거",
)

# 문항 2-3

if overall.p_value < ALPHA:
    pairs = []
    for g, h in combinations(sorted(subscription.group.unique()), 2):
        d = subscription[subscription.group == g]
        e = subscription[subscription.group == h]
        pair_data = subscription[subscription.group.isin([g, h])]
        r = run_logrank(pair_data.time, pair_data.event, pair_data.group)
        pairs.append(
            dict(
                pair=g + "-" + h, statistic=float(r.statistic), p_value=float(r.p_value)
            )
        )
    adj = holm([x["p_value"] for x in pairs])
    for row, padj in zip(pairs, adj):
        report(
            "2-3",
            "Pairwise log-rank " + row["pair"],
            "S_left=S_right for all t",
            "different for some t",
            row["statistic"],
            row["p_value"],
            1,
            "해당 두 정책의 생존분포가 다르다는 근거",
            padj,
        )
    display(pd.DataFrame(pairs).assign(p_holm=adj, reject=np.array(adj) < ALPHA))
else:
    print("전체 검정 비기각: 사전 규칙에 따라 후속 비교하지 않는다.")

# 문항 2-4

rmst = {
    g: run_rmst(d.time, d.event, overrides={"tau": 24}).estimate
    for g, d in subscription.groupby("group")
}
print("RMST(0,24), months:", rmst)
record("2-4", rmst_24=rmst, meaning="descriptive only; no test/CI requested")
print("비무작위 정책: 교란·비용·고객구성·후속 표본을 검토해야 한다.")

# 문항 3-1

xcols = ["temperature10", "load10", "new_part"]
model_data = equipment[["time", "event"] + xcols].copy()
prediction_profiles = pd.DataFrame(
    {"temperature10": [1.0, 1.0], "load10": [1.0, 1.0], "new_part": [0, 1]},
    index=["old", "new"],
)
cox_result = run_cox(
    model_data, new_data=prediction_profiles, overrides={"prediction_times": [24.0]}
)
cph = cox_result.model
print("장비 수:", len(model_data), "사건 수:", int(model_data.event.sum()))
display(cox_result.coefficients[["coef", "se(coef)", "exp(coef)", "p"]])

# 문항 3-2

coef = cox_result.coefficients[
    [
        "coef",
        "se(coef)",
        "exp(coef)",
        "exp(coef) lower 95%",
        "exp(coef) upper 95%",
        "z",
        "p",
    ]
]
display(coef)
for name, row in coef.iterrows():
    report(
        "3-2",
        "Cox Wald z: " + name,
        "beta_" + name + "=0",
        "beta_" + name + " != 0",
        row["z"],
        row["p"],
        None,
        name + "과 조건부 고장 위험의 연관성이 있다는 근거",
    )
record("3-2", coefficients=coef.reset_index().to_dict(orient="records"))

# 문항 3-3

profiles = pd.DataFrame(
    {"temperature10": [1.0, 1.0], "load10": [1.0, 1.0], "new_part": [0, 1]},
    index=["old", "new"],
)
pred = cox_result.predictions.iloc[0]
display(pd.DataFrame({"S24": pred, "F24": 1 - pred}))
hr = float(np.exp(cph.params_["new_part"]))
print("HR new/old:", hr, "F24 new/old:", float((1 - pred["new"]) / (1 - pred["old"])))
record("3-3", S24=pred.to_dict(), HR=hr)

# 문항 3-4

ph = cox_result.ph_diagnostics.copy()
ph["p_holm"] = ph.adjusted_p_value
display(ph)
for name, row in ph.iterrows():
    report(
        "3-4",
        "Schoenfeld rank diagnostic: " + name,
        "beta_" + name + " constant over time",
        "beta_" + name + " varies with time",
        row.test_statistic,
        row.p,
        1,
        name + "의 비례위험 가정 위반을 시사하는 근거",
        row.p_holm,
    )
res = cph.compute_residuals(model_data, kind="scaled_schoenfeld")
rank_time = model_data.loc[res.index, "time"].rank().to_numpy()
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for ax, name in zip(axes, xcols):
    ax.scatter(rank_time, res[name], s=8, alpha=0.5)
    line = np.polyfit(rank_time, res[name].to_numpy(), 1)
    xx = np.array([rank_time.min(), rank_time.max()])
    ax.plot(xx, np.polyval(line, xx), color="black")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set(title=name, xlabel="Event-time rank")
plt.tight_layout()
plt.show()
print(
    "잔차 직선은 시각적 참고선이며 독립적인 확정 검정이 아니다. 비기각은 가정의 증명이 아니다."
)

# 문항 3-5

r = cox_result.coefficients.loc["new_part"]
print(
    "신규 부품 HR={:.4f}, 95% CI=({:.4f},{:.4f}), individual p={:.6g}".format(
        r["exp(coef)"], r["exp(coef) lower 95%"], r["exp(coef) upper 95%"], r["p"]
    )
)
print(
    "위 수치는 합성 자료의 조건부 연관성이다. 부품 도입은 비용·안전·실험 설계·외부 검증을 추가해 판단한다."
)
out = Path("실행결과")
out.mkdir(exist_ok=True)
(out / "답안수치.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)
