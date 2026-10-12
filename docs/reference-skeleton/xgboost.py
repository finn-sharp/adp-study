"""
================================================================================
xgboost.py - xgboost 스캘레톤 코드
================================================================================
설명:
	- xgboost 기반 추론 과정에 대한 기초 구조를 확인해볼 수 있는 코드

작성 정보:
	- 작성자: 김재현(Finn) (penguin-klg@naver.com)
	- 작성일: 2026-09-10
	- 라이선스: Proprietary

변경 이력:
	DATE        AUTHOR           VERSION   DESCRIPTION
	----------------------------------------------------------------------------
	2026-09-10  김재현(Finn)       v1.0.0    최초 작성
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin-klg@naver.com"
__version__ = "1.0.0"
__status__ = "Development"

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..base.feature import FeatureSpec
from ..base.pipeline import Basepipeline
from ..base.preprocessor import BasePreprocessor
from ..trees.xgboost import XGBoostRegressionModel


DATA_URL = (
	"https://archive.ics.uci.edu/ml/machine-learning-databases/"
	"wine-quality/winequality-red.csv"
)


def load_data() -> pd.DataFrame:
	"""UCI Wine Quality red wine 데이터를 불러옵니다."""
	return pd.read_csv(DATA_URL, sep=";")


def load_recipe() -> dict:
	"""XGBoost recipe JSON을 읽습니다."""
	recipe_path = Path(__file__).parents[3] / "src" / "Recipes" / "xgboost-regression.json"
	with recipe_path.open(encoding="utf-8") as file:
		return json.load(file)


def build_pipeline(feature_columns):
	"""수치형 feature 전처리와 XGBoost 회귀 모델을 연결합니다."""
	features = [
		FeatureSpec(
			name="numeric",
			columns=list(feature_columns),
			transformer=StandardScaler(),
		)
	]
	preprocessor = BasePreprocessor(features)
	model = XGBoostRegressionModel(load_recipe())
	return Basepipeline(preprocessor, model.get_model())


def main():
	"""UCI Wine Quality 회귀 예제를 실행합니다."""
	data = load_data()
	target_column = "quality"
	feature_columns = [column for column in data.columns if column != target_column]

	X = data[feature_columns]
	y = data[target_column]
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=42,
	)

	pipeline = build_pipeline(feature_columns)
	pipeline.pipeline.fit(X_train, y_train)
	predictions = pipeline.pipeline.predict(X_test)

	print("=== Example : UCI Wine Quality XGBoost ===")
	print(f"rows={len(data)}, features={len(feature_columns)}")
	print(f"rmse={mean_squared_error(y_test, predictions) ** 0.5:.4f}")
	print(f"r2={r2_score(y_test, predictions):.4f}")
	print("preprocessor:", pipeline.get_preprocessor().describe())
	print("model:", type(pipeline.get_model()).__name__)


if __name__ == "__main__":
	main()
