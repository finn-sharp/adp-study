"""
================================================================================
분석코드.py - 이 파일 위치를 기준으로 위에서 아래로 실행하는 모의 풀이.
================================================================================
설명:
    - 이 파일 위치를 기준으로 위에서 아래로 실행하는 모의 풀이.
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
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)

from pathlib import Path
import sys, numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from IPython.display import display

# repository root 또는 회차 폴더에서 탐색; 경로를 하드코딩하지 않음.
ROOT = next(
    p
    for p in [Path.cwd()] + list(Path.cwd().parents)
    if (p / "기출문제/공통/reconstruction.py").exists()
)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "기출문제/공통"))
from reconstruction import *
from matplotlib import font_manager

font_path = ROOT / "기출문제/공통/NanumGothic-Regular.ttf"
font_manager.fontManager.addfont(str(font_path))
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = ROOT / "기출문제" / "모의고사/3회"
DATA = HERE / "data"
OUT = HERE / "실행결과"
OUT.mkdir(exist_ok=True)


def load(name):
    return pd.read_csv(DATA / name)


print("데이터 종류: 합성 복원 연습 자료, 원 시험 데이터가 아님")

# 상황형 답안 기록 초기화
_SCENARIO_ROOT = next(
    p
    for p in [Path.cwd()] + list(Path.cwd().parents)
    if (p / "기출문제/공통/scenario_inference.py").is_file()
)
sys.path.insert(0, str(_SCENARIO_ROOT / "기출문제/공통"))
from scenario_inference import (
    reset_reports,
    set_question,
    report_test,
    report_anova,
    report_correlations,
    report_adf,
    report_residuals,
    report_regression_diagnostics,
)

reset_reports(OUT)

# 문항 1-1

set_question("1-1")
car = load("Carseats.csv")
a = car.loc[car.Urban == "Yes", "Sales"]
b_ = car.loc[car.Urban == "No", "Sales"]
print("Bartlett:", stats.bartlett(a, b_))
print("Welch t-test:", stats.ttest_ind(a, b_, equal_var=False))

# 상황형 요구: 가설·실제 수치·기각 여부·맥락 결론
_w = stats.ttest_ind(a, b_, equal_var=False)
_va = a.var(ddof=1) / len(a)
_vb = b_.var(ddof=1) / len(b_)
_df = (_va + _vb) ** 2 / (_va**2 / (len(a) - 1) + _vb**2 / (len(b_) - 1))
report_test(
    "Welch 독립 두 표본 t",
    "도시/비도시 평균 매출이 같다",
    "도시/비도시 평균 매출이 다르다",
    _w.statistic,
    _w.pvalue,
    "도시 여부에 따라 평균 매출이 다르다",
    "독립 매장·집단별 정규성 가정·등분산 불필요. 지역의 인과 효과로 단정하지 않음.",
    df=_df,
)

# 문항 1-2

set_question("1-2")
numeric_correlations(
    car,
    "Sales",
    ["CompPrice", "Income", "Advertising", "Population", "Price", "Age", "Education"],
)

# 상황형 요구: 가설·실제 수치·기각 여부·맥락 결론
report_correlations(
    car,
    "Sales",
    ["CompPrice", "Income", "Advertising", "Population", "Price", "Age", "Education"],
)

# 문항 1-3

set_question("1-3")
car_model = select_ols(
    car,
    "Sales",
    ["CompPrice", "Income", "Advertising", "Population", "Price", "Age", "Education"],
    "backward",
)
print("회귀식 계수:", car_model.params.to_dict())

# 문항 2-1

set_question("2-1")
black = load("BlackFriday.csv")
black[["Product_Category_2", "Product_Category_3"]] = black[
    ["Product_Category_2", "Product_Category_3"]
].fillna(0)
black["Product_all"] = black[
    ["Product_Category_1", "Product_Category_2", "Product_Category_3"]
].sum(axis=1)
black["User_ID"] = black.User_ID.astype(str)
for c in [
    "Occupation",
    "Marital_Status",
    "Product_Category_1",
    "Product_Category_2",
    "Product_Category_3",
]:
    black[c] = black[c].astype("category")
black1 = pd.concat(
    [
        black,
        pd.get_dummies(
            black[["Gender", "Age", "City_Category", "Stay_In_Current_City_Years"]],
            prefix=["Gender", "Age", "City", "Stay"],
        ),
    ],
    axis=1,
)
print(black1.head())

# 문항 2-2

set_question("2-2")
exclude = [
    "User_ID",
    "Product_ID",
    "Gender",
    "Age",
    "City_Category",
    "Stay_In_Current_City_Years",
    "Product_all",
]
cluster_X = black1.drop(columns=exclude).copy()
for c in cluster_X.columns:
    cluster_X[c] = cluster_X[c].astype(float)
black_cluster, k_table = cluster(cluster_X, OUT, original=black)

# 문항 2-3

set_question("2-3")
print(pd.crosstab(black_cluster.clust, black_cluster.Gender))
print(pd.crosstab(black_cluster.clust, black_cluster.Age))
print(black_cluster.groupby("clust").Purchase.agg(["count", "mean", "median"]))

# 문항 3-1

set_question("3-1")
comments = load("comments.csv")
lexicon = load("lexicon.csv")
comments["clean"] = comments.text.map(clean_text)
print(comments.head())

# 문항 3-2

set_question("3-2")
adjectives = lexical_pos(comments.text, lexicon, "Adjective")
print(adjectives[:10])
word_plot(adjectives, 10)

# 문항 3-3

set_question("3-3")
feb = comments[pd.to_datetime(comments.date).dt.month == 2]
assert len(feb) > 0
word_plot(lexical_pos(feb.text, lexicon, "Noun"), 20)
