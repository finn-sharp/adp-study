"""이 파일 위치를 기준으로 위에서 아래로 실행하는 모의 풀이."""
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
HERE = ROOT / "기출문제" / "모의고사/1회"
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
lotto = load("lotto.csv")
transactions = [set(map(str, row)) for row in lotto.to_numpy()]
assert all(len(t) == 6 for t in transactions)
from collections import Counter

counts = Counter(x for t in transactions for x in t)
top = pd.Series(counts).sort_values(ascending=False).head(10)
print(top)
top.plot.bar()
plt.title("Top 10 synthetic lottery numbers")
plt.show()

# 문항 1-2

set_question("1-2")
from src.Association.apriori import run_apriori

association = run_apriori(
    [list(t) for t in transactions],
    overrides={
        "min_support": 0.002,
        "min_confidence": 0.8,
        "min_lift": 0.0,
        "max_length": 6,
        "max_rules": 300000,
        "max_candidates": 300000,
    },
)
rules_1 = association.rules.sort_values("lift", ascending=False).head(30)
print(rules_1)
rules_1.to_csv(OUT / "rules_1.csv", index=False)
print("전체 규칙 수:", len(association.rules))

# 문항 2-1

set_question("2-1")
fifa = load("FIFA.csv")
parts = fifa.Height.str.extract(r"(\d+)'(\d+)").astype(float)
fifa["Height_cm"] = parts[0] * 30 + parts[1] * 2.5
print(fifa[["Height", "Height_cm"]].head())

# 문항 2-2

set_question("2-2")
positions = {
    "ST": "Forward",
    "LW": "Forward",
    "CM": "Midfielder",
    "CDM": "Midfielder",
    "CB": "Defender",
    "LB": "Defender",
    "RB": "Defender",
    "GK": "GoalKeeper",
}
fifa["Position_Class"] = pd.Categorical(fifa.Position.map(positions))
assert not fifa.Position_Class.isna().any()
print(fifa.Position_Class.value_counts())

# 문항 2-3

set_question("2-3")
anova(fifa, "Value", "Position_Class")

# 상황형 요구: 가설·실제 수치·기각 여부·맥락 결론
_aov_model = anova(fifa, "Value", "Position_Class")
report_anova(_aov_model, "Value", ["Position_Class"])

# 문항 2-4

set_question("2-4")
anova(fifa, "Value", "Position_Class", "Preferred_Foot")

# 상황형 요구: 가설·실제 수치·기각 여부·맥락 결론
_aov_model = anova(fifa, "Value", "Position_Class", "Preferred_Foot")
report_anova(_aov_model, "Value", ["Position_Class", "Preferred_Foot"])

# 문항 2-5

set_question("2-5")
fifa_ols = select_ols(
    fifa, "Value", ["Age", "Overall", "Wage", "Height_cm", "Weight_lb"], "stepwise"
)

# 문항 3-1

set_question("3-1")
reviews = load("reviews.csv")
lexicon = load("lexicon.csv")
reviews["clean"] = reviews.text.map(clean_text)
print(reviews[["text", "clean"]].head())
reviews.to_csv(OUT / "clean_reviews.csv", index=False)

# 문항 3-2

set_question("3-2")
from sklearn.feature_extraction.text import CountVectorizer

vocab = lexicon.loc[lexicon.pos == "Noun", "word"].tolist()
vectorizer = CountVectorizer(vocabulary=vocab, token_pattern=r"(?u)\b\w+\b")
tdm = pd.DataFrame(vectorizer.fit_transform(reviews.clean).toarray(), columns=vocab)
print(tdm.head())
tdm.to_csv(OUT / "TDM.csv", index=False)
word_plot(lexical_pos(reviews.text, lexicon), 10)
