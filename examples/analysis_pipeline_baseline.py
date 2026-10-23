"""Dataset → Feature → 학습 내 sampling/CV → 예측 → 평가의 오프라인 예제."""
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from src.Preprocessing.feature_pipeline import make_feature_spec, build_feature_preprocessor
from src.Preprocessing.numeric.imputation import build_numeric_imputation_transformer
from src.Preprocessing.numeric.scaling import build_numeric_scaling_transformer
from src.Preprocessing.resampling._common import load_resampler
from src.MachineLearning.Optimization.grid_search import run_grid_search
from src.Evaluation.classification.roc_auc import run_roc_auc


def main():
    """미적합 구성 요소를 학습 CV에 연결하고 봉인된 평가 자료에서 AUC를 계산합니다."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data[:, :6], columns=data.feature_names[:6])
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, random_state=1234, stratify=y)
    feature = make_feature_spec('numeric', list(X.columns), [
        ('impute', build_numeric_imputation_transformer()),
        ('scale', build_numeric_scaling_transformer()),
    ])
    estimator = Pipeline([
        ('features', build_feature_preprocessor([feature])),
        ('sample', load_resampler('random_over', overrides={'random_state': 1234})),
        ('model', LogisticRegression(max_iter=1000, random_state=1234)),
    ])
    search = run_grid_search(X_train, y_train, estimator, overrides={
        'scoring': 'roc_auc', 'cv': 3,
        'search_space': {'model__C': [.1, 1., 10.]},
    })
    model = search.model
    positive = list(model.classes_).index(1)
    scores = model.predict_proba(X_test)[:, positive]
    auc = run_roc_auc(y_true=y_test, y_score=scores,
                      overrides={'average': 'binary', 'positive_label': 1})
    return {'search': search, 'auc': auc}


if __name__ == '__main__':
    results = main()
    print(results['auc'].to_json(indent=2))
