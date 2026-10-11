"""
study_workflow.py - 모듈별 최소 실행 예제

실행: 저장소 루트에서 python -m examples.study_workflow
입력: sklearn 내장 데이터 및 고정 시드 합성 데이터.
출력: 지도 학습/군집화/통계/연관분석/시계열/텍스트 결과 요약.
예외: statsmodels/mlxtend 등 requirements-ml.txt 의존성이 필요합니다.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes
from src.MachineLearning.Linear.ridge import RidgeProfile, run_ridge
from src.MachineLearning.Linear.logistic import LogisticProfile, run_logistic
from src.MachineLearning.Clustering.clustering import ClusteringProfile, run_clustering
from src.MachineLearning.Decomposition.pca import PCAClassificationProfile, run_pca_classification
from src.stats.analysis import StatisticalProfile, run_statistical_test, run_descriptive
from src.stats.tests.parametric.Average.Univariate import TestProfile, run_one_sample_test
from src.stats.tests.parametric.Average.Bivariate import TwoSampleTestProfile, run_two_sample_test
from src.Association.apriori import AssociationProfile, run_association
from src.TimeSeries.forecasting import ForecastProfile, run_forecast
from src.TextMining.text import TextProfile, TopicProfile, run_vectorization, run_text_classification, run_topic_model


def main():
    """각 모듈을 독립적인 데이터에 실행합니다."""
    X, y = load_iris(return_X_y=True)
    R, t = load_diabetes(return_X_y=True)
    ridge = run_ridge(R, t, RidgeProfile(param_grid={'alpha': [0.1, 1, 10]}))
    print('Ridge', ridge.test_metrics, ridge.best_params)
    logistic = run_logistic(X, y, LogisticProfile())
    print('Logistic', logistic.test_metrics)
    pca = run_pca_classification(X, y, PCAClassificationProfile(param_grid={'pca__n_components': [2, 3], 'C': [0.1, 1]}))
    print('PCA + Logistic', pca.test_metrics, pca.best_params)
    print('KMeans', run_clustering(X, ClusteringProfile(), labels_true=y).metrics)
    a, b = [1, 2, 4, 5, 7, 8], [2, 3, 3, 6, 7, 10]
    print('One sample', run_one_sample_test(a, TestProfile(mu_0=3)))
    print('Welch', run_two_sample_test(a, b, TwoSampleTestProfile(equal_var=False)))
    print('Paired', run_statistical_test(a, StatisticalProfile(test='paired_t'), other=b))
    print('Descriptive', run_descriptive(a))
    transactions = [['Milk', 'Eggs', 'Bread'], ['Milk', 'Eggs'], ['Bread', 'Eggs'], ['Milk', 'Bread']]
    print('Association', run_association(transactions, AssociationProfile()).rules[['support', 'confidence', 'lift']])
    series = pd.Series(20+np.cumsum(np.random.RandomState(4).normal(size=80)),
                       index=pd.date_range('2020-01-01', periods=80, freq='D'))
    print('ARIMA', run_forecast(series, ForecastProfile(holdout=8)).metrics)
    documents = [('good excellent happy' if i%2 else 'bad awful sad')+' unique'+str(i) for i in range(40)]
    labels = np.arange(40)%2
    print('Text', run_text_classification(documents, labels, TextProfile()).test_metrics)
    vectors = run_vectorization(documents[:30], test_documents=documents[30:])
    print('TF-IDF', vectors.train_matrix.shape, vectors.test_matrix.shape)
    print('Topics', run_topic_model(documents, TopicProfile(max_iter=5)).top_words)


if __name__ == '__main__':
    main()
