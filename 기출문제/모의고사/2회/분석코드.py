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
HERE = ROOT / "기출문제" / "모의고사/2회"
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
admit = load("Admission.csv")
numeric_correlations(
    admit, "Chance_of_Admit", ["GRE", "TOEFL", "Univ_Rating", "SOP", "LOR", "CGPA"]
)

# 상황형 요구: 가설·실제 수치·기각 여부·맥락 결론
report_correlations(
    admit, "Chance_of_Admit", ["GRE", "TOEFL", "Univ_Rating", "SOP", "LOR", "CGPA"]
)

# 문항 1-2

set_question("1-2")
admit_model = select_ols(
    admit,
    "Chance_of_Admit",
    ["GRE", "TOEFL", "Univ_Rating", "SOP", "LOR", "CGPA", "Research"],
    "stepwise",
)
residual_diagnostics(admit_model)

# 상황형 요구: 가설·실제 수치·기각 여부·맥락 결론
report_regression_diagnostics(admit_model)

# 문항 2-1

set_question("2-1")
titanic = load("titanic.csv").replace("", np.nan)
titanic["pclass"] = titanic.pclass.astype("category")
titanic["sex"] = titanic.sex.astype("category")
ids_train, ids_test = train_test_split(
    np.arange(len(titanic)),
    test_size=0.3,
    random_state=12345,
    stratify=titanic.survived,
)
median_age = titanic.iloc[ids_train].age.median()
titanic["age_imputed"] = titanic.age.fillna(median_age)
titanic["age_1"] = pd.cut(
    titanic.age_imputed, bins=list(range(0, 91, 10)), labels=list(range(9)), right=False
)
print(titanic[["age", "age_imputed", "age_1"]].head())
print("train age median:", median_age)

# 문항 2-2

set_question("2-2")
titanic_scores, titanic_models, train, test = classify(
    titanic,
    "survived",
    ["pclass", "sex", "sibsp", "parch", "fare", "embarked"],
    OUT,
    seed=12345,
)

# 문항 2-3

set_question("2-3")
print(titanic_scores)
print(
    "보류 최고 accuracy 모델:",
    titanic_scores.loc[titanic_scores.accuracy.idxmax(), "model"],
)

# 문항 3-1

set_question("3-1")
speech = load("speech.csv")
lexicon = load("lexicon.csv")
speech["clean"] = speech.text.map(clean_text)
print(speech[["text", "clean"]].head())

# 문항 3-2(보충)

set_question("3-2(보충)")
word_plot(lexical_pos(speech.text, lexicon), 10)
