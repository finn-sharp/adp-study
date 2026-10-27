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
HERE = ROOT / "기출문제" / "모의고사/4회"
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
weather = load("weatherAUS.csv")
print(weather.describe(include="all"))
missing = weather.isna().sum()
dropped = missing[missing >= 10000].index.tolist()
weather = weather.drop(columns=dropped).dropna().copy()
weather["Date"] = pd.to_datetime(weather.Date)
print("제외 열:", dropped, "남은 행:", len(weather))
train, test = train_test_split(
    np.arange(len(weather)),
    test_size=0.3,
    random_state=6789,
    stratify=weather.RainTomorrow,
)
print(len(train), len(test))

# 문항 1-2

set_question("1-2")
weather_scores, weather_models, train, test = classify(
    weather, "RainTomorrow", [c for c in weather if c != "RainTomorrow"], OUT, seed=6789
)

# 문항 1-3(보충)

set_question("1-3(보충)")
print(weather_scores)
print("시간 순 운영에서는 랜덤분할 지표가 낙관적일 수 있습니다.")

# 문항 2-1

set_question("2-1")
bike = load("bike_marketing.csv")
bike["pop_density"] = bike.pop_density.astype("category")
anova(bike, "revenues", "pop_density")

# 상황형 요구: 가설·실제 수치·기각 여부·맥락 결론
_aov_model = anova(bike, "revenues", "pop_density")
report_anova(_aov_model, "revenues", ["pop_density"])

# 문항 2-2

set_question("2-2")
bike_model = select_ols(
    bike,
    "revenues",
    ["google_adwords", "facebook", "twitter", "marketing_total", "employees"],
    "forward",
)

# 문항 2-3

set_question("2-3")
residual_diagnostics(bike_model)

# 상황형 요구: 가설·실제 수치·기각 여부·맥락 결론
report_regression_diagnostics(bike_model)

# 문항 3-1

set_question("3-1")
instagram = load("instagram.csv")
lexicon = load("lexicon.csv")
instagram["clean"] = instagram.text.map(clean_text)
print(instagram.head())

# 문항 3-2

set_question("3-2")
assert "태교여행" in lexicon.word.tolist()
word_plot(lexical_pos(instagram.text, lexicon), 10)
