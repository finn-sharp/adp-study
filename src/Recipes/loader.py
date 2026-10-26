"""
=========================================================================loader.py - 모델별 JSON 레시피 읽기·검증·Profile 구성
=========================================================================설명:
    - src/Recipes의 평면 JSON 설정을 모델별 Profile로 변환합니다.
    - 레시피 → 명시적 overrides → Profile → 실행기 순서로 설정을 적용합니다.

분석 가정:
    - 레시피는 JSON 객체이며 입력 데이터는 실행기에서 별도로 검증합니다.
    - 파일 누락·알 수 없는 키·중복 키는 오류이며 설정을 조용히 무시하지 않습니다.

작성 정보:
    - 작성자: 김재현(Finn)
    - 이메일: penguin.klg@gmail.com
    - 작성일: 2026-10-11
    - 라이선스: Proprietary

변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    ----------------------------------------------------------------------------
    2026-10-11  김재현(Finn)     v1.0.0    모델별 레시피와 실행기 연결
========================================================================="""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"

from dataclasses import fields
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


# 모델명: (모듈, Profile, 기본 task, 지원 task). 통계 검정에는 task 필드가 없습니다.
_REGISTRY = {
    "tabular-loader": ("src.datasets.tabular", "TabularLoaderProfile", None, (None,)),
    "plot-regression-glm-residuals": (
        "src.Visualization.regression.glm_residuals",
        "GLMResidualsProfile",
        None,
        (None,),
    ),
    "regression-information-selection": (
        "src.stats.regression.information_selection",
        "InformationSelectionProfile",
        None,
        (None,),
    ),
    "glm-logistic": (
        "src.MachineLearning.GeneralizedLinearModels.logistic",
        "LogisticGLMProfile",
        None,
        (None,),
    ),
    "glm-poisson": (
        "src.MachineLearning.GeneralizedLinearModels.poisson",
        "PoissonGLMProfile",
        None,
        (None,),
    ),
    "glm-gamma": (
        "src.MachineLearning.GeneralizedLinearModels.gamma",
        "GammaGLMProfile",
        None,
        (None,),
    ),
    "regression-weighted": (
        "src.MachineLearning.LinearModels.weighted_regression",
        "WeightedRegressionProfile",
        None,
        (None,),
    ),
    "timeseries-rolling-selection": (
        "src.TimeSeries.rolling_selection",
        "RollingSelectionProfile",
        None,
        (None,),
    ),
    "plot-regression-model-comparison": (
        "src.Visualization.regression.model_comparison",
        "RegressionModelComparisonProfile",
        None,
        (None,),
    ),
    "genetic-algorithm": (
        "src.Optimization.genetic_algorithm",
        "GeneticAlgorithmProfile",
        None,
        (None,),
    ),
    "production-allocation": (
        "src.Optimization.production_allocation",
        "ProductionAllocationProfile",
        None,
        (None,),
    ),
    "plot-optimization-convergence": (
        "src.Visualization.optimization.convergence",
        "OptimizationConvergenceProfile",
        None,
        (None,),
    ),
    "regression-ols": ("src.stats.regression.ols", "OLSProfile", None, (None,)),
    "regression-joint-f": (
        "src.stats.regression.joint_f",
        "JointFProfile",
        None,
        (None,),
    ),
    "regression-nested-f": (
        "src.stats.regression.nested_f",
        "NestedFProfile",
        None,
        (None,),
    ),
    "regression-linearity": (
        "src.stats.regression.linearity",
        "LinearityProfile",
        None,
        (None,),
    ),
    "regression-normality": (
        "src.stats.regression.normality",
        "RegressionNormalityProfile",
        None,
        (None,),
    ),
    "regression-heteroscedasticity": (
        "src.stats.regression.heteroscedasticity",
        "HeteroscedasticityProfile",
        None,
        (None,),
    ),
    "regression-independence": (
        "src.stats.regression.independence",
        "IndependenceProfile",
        None,
        (None,),
    ),
    "regression-collinearity": (
        "src.stats.regression.collinearity",
        "CollinearityProfile",
        None,
        (None,),
    ),
    "regression-influence": (
        "src.stats.regression.influence",
        "InfluenceProfile",
        None,
        (None,),
    ),
    "regression-polynomial": (
        "src.Preprocessing.regression.polynomial",
        "PolynomialProfile",
        None,
        (None,),
    ),
    "regression-target-transform": (
        "src.Preprocessing.regression.target_transform",
        "TargetTransformProfile",
        None,
        (None,),
    ),
    "regression-vif-selection": (
        "src.Preprocessing.regression.vif_selection",
        "VIFSelectionProfile",
        None,
        (None,),
    ),
    "regression-influence-filter": (
        "src.Preprocessing.regression.influence_filter",
        "InfluenceFilterProfile",
        None,
        (None,),
    ),
    "regression-workflow": (
        "src.MachineLearning.LinearModels.regression_workflow",
        "RegressionWorkflowProfile",
        None,
        (None,),
    ),
    "naive-bayes-gaussian": (
        "src.MachineLearning.NaiveBayes.gaussian",
        "GaussianProfile",
        None,
        (None,),
    ),
    "naive-bayes-bernoulli": (
        "src.MachineLearning.NaiveBayes.bernoulli",
        "BernoulliProfile",
        None,
        (None,),
    ),
    "naive-bayes-multinomial": (
        "src.MachineLearning.NaiveBayes.multinomial",
        "MultinomialProfile",
        None,
        (None,),
    ),
    "naive-bayes-complement": (
        "src.MachineLearning.NaiveBayes.complement",
        "ComplementProfile",
        None,
        (None,),
    ),
    "naive-bayes-poisson": (
        "src.MachineLearning.NaiveBayes.poisson",
        "PoissonProfile",
        None,
        (None,),
    ),
    "naive-bayes-binomial": (
        "src.MachineLearning.NaiveBayes.binomial",
        "BinomialProfile",
        None,
        (None,),
    ),
    "naive-bayes-categorical": (
        "src.MachineLearning.NaiveBayes.categorical",
        "CategoricalProfile",
        None,
        (None,),
    ),
    "plot-qq-normal": ("src.Visualization.qq.normal", "QQNormalProfile", None, (None,)),
    "plot-qq-student-t": (
        "src.Visualization.qq.student_t",
        "QQStudentTProfile",
        None,
        (None,),
    ),
    "plot-qq-exponential": (
        "src.Visualization.qq.exponential",
        "QQExponentialProfile",
        None,
        (None,),
    ),
    "plot-qq-gamma": ("src.Visualization.qq.gamma", "QQGammaProfile", None, (None,)),
    "plot-qq-lognormal": (
        "src.Visualization.qq.lognormal",
        "QQLognormalProfile",
        None,
        (None,),
    ),
    "plot-qq-weibull": (
        "src.Visualization.qq.weibull",
        "QQWeibullProfile",
        None,
        (None,),
    ),
    "plot-qq-beta": ("src.Visualization.qq.beta", "QQBetaProfile", None, (None,)),
    "plot-qq-uniform": (
        "src.Visualization.qq.uniform",
        "QQUniformProfile",
        None,
        (None,),
    ),
    "plot-qq-chi-square": (
        "src.Visualization.qq.chi_square",
        "QQChiSquareProfile",
        None,
        (None,),
    ),
    "plot-qq-logistic": (
        "src.Visualization.qq.logistic",
        "QQLogisticProfile",
        None,
        (None,),
    ),
    "plot-qq-poisson": (
        "src.Visualization.qq.poisson",
        "QQPoissonProfile",
        None,
        (None,),
    ),
    "plot-qq-binomial": (
        "src.Visualization.qq.binomial",
        "QQBinomialProfile",
        None,
        (None,),
    ),
    "plot-qq-bernoulli": (
        "src.Visualization.qq.bernoulli",
        "QQBernoulliProfile",
        None,
        (None,),
    ),
    "plot-regression-residual-fitted": (
        "src.Visualization.regression.residual_fitted",
        "ResidualFittedProfile",
        None,
        (None,),
    ),
    "plot-regression-scale-location": (
        "src.Visualization.regression.scale_location",
        "ScaleLocationProfile",
        None,
        (None,),
    ),
    "plot-regression-residual-order": (
        "src.Visualization.regression.residual_order",
        "ResidualOrderProfile",
        None,
        (None,),
    ),
    "plot-regression-cooks-distance": (
        "src.Visualization.regression.cooks_distance",
        "CooksDistanceProfile",
        None,
        (None,),
    ),
    "plot-regression-leverage": (
        "src.Visualization.regression.leverage",
        "LeverageProfile",
        None,
        (None,),
    ),
    "plot-regression-coefficient-intervals": (
        "src.Visualization.regression.coefficient_intervals",
        "CoefficientIntervalsProfile",
        None,
        (None,),
    ),
    "plot-regression-observed-predicted": (
        "src.Visualization.regression.observed_predicted",
        "ObservedPredictedProfile",
        None,
        (None,),
    ),
    "plot-regression-prediction-interval": (
        "src.Visualization.regression.prediction_interval",
        "PredictionIntervalProfile",
        None,
        (None,),
    ),
    "plot-regression-partial-residual": (
        "src.Visualization.regression.partial_residual",
        "PartialResidualProfile",
        None,
        (None,),
    ),
    "plot-regression-vif": (
        "src.Visualization.regression.vif",
        "VIFPlotProfile",
        None,
        (None,),
    ),
    # Domain architecture extension (2026-10-20)
    "survival-kaplan-meier": (
        "src.stats.survival.kaplan_meier",
        "KaplanMeierProfile",
        None,
        (None,),
    ),
    "survival-rmst": ("src.stats.survival.rmst", "RMSTProfile", None, (None,)),
    "survival-logrank": ("src.stats.survival.logrank", "LogrankProfile", None, (None,)),
    "survival-cox": ("src.stats.survival.cox", "CoxProfile", None, (None,)),
    "timeseries-arima": ("src.TimeSeries.arima", "ARIMAProfile", None, (None,)),
    "timeseries-sarima": ("src.TimeSeries.sarima", "SARIMAProfile", None, (None,)),
    "nonlinear-regression": (
        "src.MachineLearning.NonlinearModels.nonlinear_regression",
        "NonlinearRegressionProfile",
        None,
        (None,),
    ),
    "resampling-random-over": (
        "src.Preprocessing.resampling.random_over",
        "RandomOverProfile",
        None,
        (None,),
    ),
    "resampling-smote": (
        "src.Preprocessing.resampling.smote",
        "SMOTEProfile",
        None,
        (None,),
    ),
    "resampling-adasyn": (
        "src.Preprocessing.resampling.adasyn",
        "ADASYNProfile",
        None,
        (None,),
    ),
    "resampling-borderline-smote": (
        "src.Preprocessing.resampling.borderline_smote",
        "BorderlineSMOTEProfile",
        None,
        (None,),
    ),
    "resampling-svm-smote": (
        "src.Preprocessing.resampling.svm_smote",
        "SVMSMOTEProfile",
        None,
        (None,),
    ),
    "resampling-smotenc": (
        "src.Preprocessing.resampling.smotenc",
        "SMOTENCProfile",
        None,
        (None,),
    ),
    "resampling-smoten": (
        "src.Preprocessing.resampling.smoten",
        "SMOTENProfile",
        None,
        (None,),
    ),
    "resampling-random-under": (
        "src.Preprocessing.resampling.random_under",
        "RandomUnderProfile",
        None,
        (None,),
    ),
    "resampling-tomek-links": (
        "src.Preprocessing.resampling.tomek_links",
        "TomekLinksProfile",
        None,
        (None,),
    ),
    "resampling-edited-nearest-neighbours": (
        "src.Preprocessing.resampling.edited_nearest_neighbours",
        "EditedNearestNeighboursProfile",
        None,
        (None,),
    ),
    "resampling-near-miss": (
        "src.Preprocessing.resampling.near_miss",
        "NearMissProfile",
        None,
        (None,),
    ),
    "resampling-cluster-centroids": (
        "src.Preprocessing.resampling.cluster_centroids",
        "ClusterCentroidsProfile",
        None,
        (None,),
    ),
    "optimization-grid-search": (
        "src.MachineLearning.Optimization.grid_search",
        "GridSearchProfile",
        "classification",
        ("classification", "regression"),
    ),
    "optimization-random-search": (
        "src.MachineLearning.Optimization.random_search",
        "RandomSearchProfile",
        "classification",
        ("classification", "regression"),
    ),
    "optimization-optuna-search": (
        "src.MachineLearning.Optimization.optuna_search",
        "OptunaSearchProfile",
        "classification",
        ("classification", "regression"),
    ),
    "metric-comparison": (
        "src.Evaluation.comparison",
        "ComparisonProfile",
        None,
        (None,),
    ),
    "metric-cv-summary": (
        "src.Evaluation.cv_summary",
        "CVSummaryProfile",
        None,
        (None,),
    ),
    "metric-classification-accuracy": (
        "src.Evaluation.classification.accuracy",
        "ClassificationAccuracyProfile",
        None,
        (None,),
    ),
    "metric-classification-precision": (
        "src.Evaluation.classification.precision",
        "ClassificationPrecisionProfile",
        None,
        (None,),
    ),
    "metric-classification-recall": (
        "src.Evaluation.classification.recall",
        "ClassificationRecallProfile",
        None,
        (None,),
    ),
    "metric-classification-f1": (
        "src.Evaluation.classification.f1",
        "ClassificationF1Profile",
        None,
        (None,),
    ),
    "metric-classification-fbeta": (
        "src.Evaluation.classification.fbeta",
        "ClassificationFbetaProfile",
        None,
        (None,),
    ),
    "metric-classification-specificity": (
        "src.Evaluation.classification.specificity",
        "ClassificationSpecificityProfile",
        None,
        (None,),
    ),
    "metric-classification-balanced-accuracy": (
        "src.Evaluation.classification.balanced_accuracy",
        "ClassificationBalancedAccuracyProfile",
        None,
        (None,),
    ),
    "metric-classification-mcc": (
        "src.Evaluation.classification.mcc",
        "ClassificationMccProfile",
        None,
        (None,),
    ),
    "metric-classification-cohen-kappa": (
        "src.Evaluation.classification.cohen_kappa",
        "ClassificationCohenKappaProfile",
        None,
        (None,),
    ),
    "metric-classification-confusion-matrix": (
        "src.Evaluation.classification.confusion_matrix",
        "ClassificationConfusionMatrixProfile",
        None,
        (None,),
    ),
    "metric-classification-roc-auc": (
        "src.Evaluation.classification.roc_auc",
        "ClassificationRocAucProfile",
        None,
        (None,),
    ),
    "metric-classification-average-precision": (
        "src.Evaluation.classification.average_precision",
        "ClassificationAveragePrecisionProfile",
        None,
        (None,),
    ),
    "metric-classification-pr-auc": (
        "src.Evaluation.classification.pr_auc",
        "ClassificationPrAucProfile",
        None,
        (None,),
    ),
    "metric-classification-log-loss": (
        "src.Evaluation.classification.log_loss",
        "ClassificationLogLossProfile",
        None,
        (None,),
    ),
    "metric-classification-brier-score": (
        "src.Evaluation.classification.brier_score",
        "ClassificationBrierScoreProfile",
        None,
        (None,),
    ),
    "metric-classification-gini": (
        "src.Evaluation.classification.gini",
        "ClassificationGiniProfile",
        None,
        (None,),
    ),
    "metric-classification-ks": (
        "src.Evaluation.classification.ks",
        "ClassificationKsProfile",
        None,
        (None,),
    ),
    "metric-classification-lift": (
        "src.Evaluation.classification.lift",
        "ClassificationLiftProfile",
        None,
        (None,),
    ),
    "metric-classification-gain": (
        "src.Evaluation.classification.gain",
        "ClassificationGainProfile",
        None,
        (None,),
    ),
    "metric-classification-lift-curve": (
        "src.Evaluation.classification.lift_curve",
        "ClassificationLiftCurveProfile",
        None,
        (None,),
    ),
    "metric-classification-gain-curve": (
        "src.Evaluation.classification.gain_curve",
        "ClassificationGainCurveProfile",
        None,
        (None,),
    ),
    "metric-classification-roc-curve": (
        "src.Evaluation.classification.roc_curve",
        "ClassificationRocCurveProfile",
        None,
        (None,),
    ),
    "metric-classification-precision-recall-curve": (
        "src.Evaluation.classification.precision_recall_curve",
        "ClassificationPrecisionRecallCurveProfile",
        None,
        (None,),
    ),
    "metric-classification-calibration": (
        "src.Evaluation.classification.calibration",
        "ClassificationCalibrationProfile",
        None,
        (None,),
    ),
    "metric-classification-metrics": (
        "src.Evaluation.classification.metrics",
        "ClassificationMetricsProfile",
        None,
        (None,),
    ),
    "metric-regression-mae": (
        "src.Evaluation.regression.mae",
        "RegressionMaeProfile",
        None,
        (None,),
    ),
    "metric-regression-mse": (
        "src.Evaluation.regression.mse",
        "RegressionMseProfile",
        None,
        (None,),
    ),
    "metric-regression-rmse": (
        "src.Evaluation.regression.rmse",
        "RegressionRmseProfile",
        None,
        (None,),
    ),
    "metric-regression-r2": (
        "src.Evaluation.regression.r2",
        "RegressionR2Profile",
        None,
        (None,),
    ),
    "metric-regression-adjusted-r2": (
        "src.Evaluation.regression.adjusted_r2",
        "RegressionAdjustedR2Profile",
        None,
        (None,),
    ),
    "metric-regression-median-absolute-error": (
        "src.Evaluation.regression.median_absolute_error",
        "RegressionMedianAbsoluteErrorProfile",
        None,
        (None,),
    ),
    "metric-regression-mape": (
        "src.Evaluation.regression.mape",
        "RegressionMapeProfile",
        None,
        (None,),
    ),
    "metric-regression-smape": (
        "src.Evaluation.regression.smape",
        "RegressionSmapeProfile",
        None,
        (None,),
    ),
    "metric-regression-wape": (
        "src.Evaluation.regression.wape",
        "RegressionWapeProfile",
        None,
        (None,),
    ),
    "metric-regression-msle": (
        "src.Evaluation.regression.msle",
        "RegressionMsleProfile",
        None,
        (None,),
    ),
    "metric-regression-rmsle": (
        "src.Evaluation.regression.rmsle",
        "RegressionRmsleProfile",
        None,
        (None,),
    ),
    "metric-regression-explained-variance": (
        "src.Evaluation.regression.explained_variance",
        "RegressionExplainedVarianceProfile",
        None,
        (None,),
    ),
    "metric-regression-max-error": (
        "src.Evaluation.regression.max_error",
        "RegressionMaxErrorProfile",
        None,
        (None,),
    ),
    "metric-regression-pinball-loss": (
        "src.Evaluation.regression.pinball_loss",
        "RegressionPinballLossProfile",
        None,
        (None,),
    ),
    "metric-regression-metrics": (
        "src.Evaluation.regression.metrics",
        "RegressionMetricsProfile",
        None,
        (None,),
    ),
    "metric-clustering-silhouette": (
        "src.Evaluation.clustering.silhouette",
        "ClusteringSilhouetteProfile",
        None,
        (None,),
    ),
    "metric-clustering-davies-bouldin": (
        "src.Evaluation.clustering.davies_bouldin",
        "ClusteringDaviesBouldinProfile",
        None,
        (None,),
    ),
    "metric-clustering-calinski-harabasz": (
        "src.Evaluation.clustering.calinski_harabasz",
        "ClusteringCalinskiHarabaszProfile",
        None,
        (None,),
    ),
    "metric-clustering-ari": (
        "src.Evaluation.clustering.ari",
        "ClusteringAriProfile",
        None,
        (None,),
    ),
    "metric-clustering-ami": (
        "src.Evaluation.clustering.ami",
        "ClusteringAmiProfile",
        None,
        (None,),
    ),
    "metric-clustering-nmi": (
        "src.Evaluation.clustering.nmi",
        "ClusteringNmiProfile",
        None,
        (None,),
    ),
    "metric-clustering-homogeneity": (
        "src.Evaluation.clustering.homogeneity",
        "ClusteringHomogeneityProfile",
        None,
        (None,),
    ),
    "metric-clustering-completeness": (
        "src.Evaluation.clustering.completeness",
        "ClusteringCompletenessProfile",
        None,
        (None,),
    ),
    "metric-clustering-v-measure": (
        "src.Evaluation.clustering.v_measure",
        "ClusteringVMeasureProfile",
        None,
        (None,),
    ),
    "metric-clustering-metrics": (
        "src.Evaluation.clustering.metrics",
        "ClusteringMetricsProfile",
        None,
        (None,),
    ),
    "metric-time-series-mase": (
        "src.Evaluation.time_series.mase",
        "TimeSeriesMaseProfile",
        None,
        (None,),
    ),
    "metric-time-series-rmsse": (
        "src.Evaluation.time_series.rmsse",
        "TimeSeriesRmsseProfile",
        None,
        (None,),
    ),
    "metric-time-series-horizon-errors": (
        "src.Evaluation.time_series.horizon_errors",
        "TimeSeriesHorizonErrorsProfile",
        None,
        (None,),
    ),
    "metric-time-series-metrics": (
        "src.Evaluation.time_series.metrics",
        "TimeSeriesMetricsProfile",
        None,
        (None,),
    ),
    "plot-selection-feature-importance": (
        "src.Visualization.selection.feature_importance",
        "SelectionFeatureImportanceProfile",
        None,
        (None,),
    ),
    "plot-selection-validation-curve": (
        "src.Visualization.selection.validation_curve",
        "SelectionValidationCurveProfile",
        None,
        (None,),
    ),
    "plot-numeric-histogram": (
        "src.Visualization.numeric.histogram",
        "NumericHistogramProfile",
        None,
        (None,),
    ),
    "plot-numeric-kde": (
        "src.Visualization.numeric.kde",
        "NumericKdeProfile",
        None,
        (None,),
    ),
    "plot-numeric-ecdf": (
        "src.Visualization.numeric.ecdf",
        "NumericEcdfProfile",
        None,
        (None,),
    ),
    "plot-numeric-qq": (
        "src.Visualization.numeric.qq",
        "NumericQqProfile",
        None,
        (None,),
    ),
    "plot-categorical-count": (
        "src.Visualization.categorical.count",
        "CategoricalCountProfile",
        None,
        (None,),
    ),
    "plot-categorical-pareto": (
        "src.Visualization.categorical.pareto",
        "CategoricalParetoProfile",
        None,
        (None,),
    ),
    "plot-categorical-grouped-distribution": (
        "src.Visualization.categorical.grouped_distribution",
        "CategoricalGroupedDistributionProfile",
        None,
        (None,),
    ),
    "plot-categorical-contingency": (
        "src.Visualization.categorical.contingency",
        "CategoricalContingencyProfile",
        None,
        (None,),
    ),
    "plot-categorical-mosaic": (
        "src.Visualization.categorical.mosaic",
        "CategoricalMosaicProfile",
        None,
        (None,),
    ),
    "plot-categorical-category-rate": (
        "src.Visualization.categorical.category_rate",
        "CategoricalCategoryRateProfile",
        None,
        (None,),
    ),
    "plot-quality-missingness": (
        "src.Visualization.quality.missingness",
        "QualityMissingnessProfile",
        None,
        (None,),
    ),
    "plot-quality-missing-matrix": (
        "src.Visualization.quality.missing_matrix",
        "QualityMissingMatrixProfile",
        None,
        (None,),
    ),
    "plot-quality-missing-association": (
        "src.Visualization.quality.missing_association",
        "QualityMissingAssociationProfile",
        None,
        (None,),
    ),
    "plot-relationships-scatter": (
        "src.Visualization.relationships.scatter",
        "RelationshipsScatterProfile",
        None,
        (None,),
    ),
    "plot-relationships-hexbin": (
        "src.Visualization.relationships.hexbin",
        "RelationshipsHexbinProfile",
        None,
        (None,),
    ),
    "plot-relationships-correlation": (
        "src.Visualization.relationships.correlation",
        "RelationshipsCorrelationProfile",
        None,
        (None,),
    ),
    "plot-clustering-dendrogram": (
        "src.Visualization.clustering.dendrogram",
        "ClusteringDendrogramProfile",
        None,
        (None,),
    ),
    "plot-clustering-silhouette": (
        "src.Visualization.clustering.silhouette",
        "ClusteringSilhouetteProfile",
        None,
        (None,),
    ),
    "plot-clustering-k-distance": (
        "src.Visualization.clustering.k_distance",
        "ClusteringKDistanceProfile",
        None,
        (None,),
    ),
    "plot-clustering-elbow": (
        "src.Visualization.clustering.elbow",
        "ClusteringElbowProfile",
        None,
        (None,),
    ),
    "plot-decomposition-explained-variance": (
        "src.Visualization.decomposition.explained_variance",
        "DecompositionExplainedVarianceProfile",
        None,
        (None,),
    ),
    "plot-decomposition-loadings": (
        "src.Visualization.decomposition.loadings",
        "DecompositionLoadingsProfile",
        None,
        (None,),
    ),
    "plot-classification-confusion": (
        "src.Visualization.classification.confusion",
        "ClassificationConfusionProfile",
        None,
        (None,),
    ),
    "plot-classification-roc": (
        "src.Visualization.classification.roc",
        "ClassificationRocProfile",
        None,
        (None,),
    ),
    "plot-classification-precision-recall": (
        "src.Visualization.classification.precision_recall",
        "ClassificationPrecisionRecallProfile",
        None,
        (None,),
    ),
    "plot-classification-threshold": (
        "src.Visualization.classification.threshold",
        "ClassificationThresholdProfile",
        None,
        (None,),
    ),
    "plot-classification-calibration": (
        "src.Visualization.classification.calibration",
        "ClassificationCalibrationProfile",
        None,
        (None,),
    ),
    "plot-association-rules": (
        "src.Visualization.association.rules",
        "AssociationRulesProfile",
        None,
        (None,),
    ),
    "plot-networks-graph": (
        "src.Visualization.networks.graph",
        "NetworksGraphProfile",
        None,
        (None,),
    ),
    "plot-time-series-timeline": (
        "src.Visualization.time_series.timeline",
        "TimeSeriesTimelineProfile",
        None,
        (None,),
    ),
    "plot-time-series-lag": (
        "src.Visualization.time_series.lag",
        "TimeSeriesLagProfile",
        None,
        (None,),
    ),
    "plot-time-series-autocorrelation": (
        "src.Visualization.time_series.autocorrelation",
        "TimeSeriesAutocorrelationProfile",
        None,
        (None,),
    ),
    "plot-time-series-seasonal": (
        "src.Visualization.time_series.seasonal",
        "TimeSeriesSeasonalProfile",
        None,
        (None,),
    ),
    "feature-preprocessing": (
        "src.Preprocessing.feature_pipeline",
        "FeaturePreprocessingProfile",
        None,
        (None,),
    ),
    "numeric-scaling": (
        "src.Preprocessing.numeric.scaling",
        "NumericScalingProfile",
        None,
        (None,),
    ),
    "numeric-imputation": (
        "src.Preprocessing.numeric.imputation",
        "NumericImputationProfile",
        None,
        (None,),
    ),
    "numeric-power": (
        "src.Preprocessing.numeric.power",
        "NumericPowerProfile",
        None,
        (None,),
    ),
    "categorical-imputation": (
        "src.Preprocessing.categorical.imputation",
        "CategoricalImputationProfile",
        None,
        (None,),
    ),
    "categorical-encoding": (
        "src.Preprocessing.categorical.encoding",
        "CategoricalEncodingProfile",
        None,
        (None,),
    ),
    "correlation-pearson": (
        "src.stats.correlation.numeric.pearson",
        "PearsonProfile",
        None,
        (None,),
    ),
    "correlation-spearman": (
        "src.stats.correlation.numeric.spearman",
        "SpearmanProfile",
        None,
        (None,),
    ),
    "correlation-kendall": (
        "src.stats.correlation.numeric.kendall",
        "KendallProfile",
        None,
        (None,),
    ),
    "correlation-cramers-v": (
        "src.stats.correlation.categorical.cramers_v",
        "CramersVProfile",
        None,
        (None,),
    ),
    "correlation-phi": (
        "src.stats.correlation.categorical.phi",
        "PhiProfile",
        None,
        (None,),
    ),
    "rope": ("src.Preprocessing.positional.rope", "RoPEProfile", None, (None,)),
    "correlation-matrix-transform": (
        "src.stats.correlation.matrix_transform",
        "CorrelationMatrixTransformProfile",
        None,
        (None,),
    ),
    "tf-cnn": (
        "src.DeepLearning.TensorFlow.cnn",
        "TFCNNProfile",
        "classification",
        ("classification", "regression"),
    ),
    "tf-simple-rnn": (
        "src.DeepLearning.TensorFlow.simple_rnn",
        "TFSimpleRNNProfile",
        "classification",
        ("classification", "regression"),
    ),
    "tf-lstm": (
        "src.DeepLearning.TensorFlow.lstm",
        "TFLSTMProfile",
        "classification",
        ("classification", "regression"),
    ),
    "tf-gru": (
        "src.DeepLearning.TensorFlow.gru",
        "TFGRUProfile",
        "classification",
        ("classification", "regression"),
    ),
    "fp-growth": ("src.Association.fp_growth", "FPGrowthProfile", None, (None,)),
    "eclat": ("src.Association.eclat", "EclatProfile", None, (None,)),
    "distribution-bernoulli": (
        "src.stats.distributions.bernoulli",
        "BernoulliProfile",
        None,
        (None,),
    ),
    "distribution-binomial": (
        "src.stats.distributions.binomial",
        "BinomialProfile",
        None,
        (None,),
    ),
    "distribution-normal": (
        "src.stats.distributions.normal",
        "NormalProfile",
        None,
        (None,),
    ),
    "distribution-hypergeometric": (
        "src.stats.distributions.hypergeometric",
        "HypergeometricProfile",
        None,
        (None,),
    ),
    "distribution-geometric": (
        "src.stats.distributions.geometric",
        "GeometricProfile",
        None,
        (None,),
    ),
    "distribution-negative-binomial": (
        "src.stats.distributions.negative_binomial",
        "NegativeBinomialProfile",
        None,
        (None,),
    ),
    "distribution-poisson": (
        "src.stats.distributions.poisson",
        "PoissonProfile",
        None,
        (None,),
    ),
    "distribution-exponential": (
        "src.stats.distributions.exponential",
        "ExponentialProfile",
        None,
        (None,),
    ),
    "distribution-weibull": (
        "src.stats.distributions.weibull",
        "WeibullProfile",
        None,
        (None,),
    ),
    "distribution-chi-square": (
        "src.stats.distributions.chi_square",
        "ChiSquareDistributionProfile",
        None,
        (None,),
    ),
    "distribution-gamma": (
        "src.stats.distributions.gamma",
        "GammaProfile",
        None,
        (None,),
    ),
    "distribution-beta": ("src.stats.distributions.beta", "BetaProfile", None, (None,)),
    "distribution-student-t": (
        "src.stats.distributions.student_t",
        "StudentTProfile",
        None,
        (None,),
    ),
    "distribution-cauchy": (
        "src.stats.distributions.cauchy",
        "CauchyProfile",
        None,
        (None,),
    ),
    "distribution-lognormal": (
        "src.stats.distributions.lognormal",
        "LognormalProfile",
        None,
        (None,),
    ),
    "distribution-f": (
        "src.stats.distributions.f",
        "FDistributionProfile",
        None,
        (None,),
    ),
    "one-sample-mean": (
        "src.stats.tests.parametric.Average.one_sample",
        "OneSampleMeanProfile",
        None,
        (None,),
    ),
    "two-sample-mean": (
        "src.stats.tests.parametric.Average.two_sample",
        "TwoSampleMeanProfile",
        None,
        (None,),
    ),
    "paired-t": (
        "src.stats.tests.parametric.Average.paired",
        "PairedTProfile",
        None,
        (None,),
    ),
    "one-sample-variance": (
        "src.stats.tests.parametric.Variance.one_sample",
        "OneSampleVarianceProfile",
        None,
        (None,),
    ),
    "one-sample-proportion": (
        "src.stats.tests.parametric.Proportion.one_sample",
        "OneSampleProportionProfile",
        None,
        (None,),
    ),
    "two-sample-proportion": (
        "src.stats.tests.parametric.Proportion.two_sample",
        "TwoSampleProportionProfile",
        None,
        (None,),
    ),
    "sign": (
        "src.stats.tests.nonParametric.Location.sign",
        "SignProfile",
        None,
        (None,),
    ),
    "wilcoxon": (
        "src.stats.tests.nonParametric.Location.wilcoxon",
        "WilcoxonProfile",
        None,
        (None,),
    ),
    "runs": (
        "src.stats.tests.nonParametric.Sequence.runs",
        "RunsProfile",
        None,
        (None,),
    ),
    "mann-whitney": (
        "src.stats.tests.nonParametric.Location.mann_whitney",
        "MannWhitneyProfile",
        None,
        (None,),
    ),
    "kruskal-wallis": (
        "src.stats.tests.nonParametric.Location.kruskal_wallis",
        "KruskalWallisProfile",
        None,
        (None,),
    ),
    "mood-median": (
        "src.stats.tests.nonParametric.Location.mood_median",
        "MoodMedianProfile",
        None,
        (None,),
    ),
    "friedman": (
        "src.stats.tests.nonParametric.Location.friedman",
        "FriedmanProfile",
        None,
        (None,),
    ),
    "repeated-measures-anova": (
        "src.stats.tests.parametric.Average.anova.repeated_measures",
        "RepeatedMeasuresANOVAProfile",
        None,
        (None,),
    ),
    "anderson-darling": (
        "src.stats.tests.diagnostics.anderson_darling",
        "AndersonDarlingProfile",
        None,
        (None,),
    ),
    "kolmogorov-smirnov": (
        "src.stats.tests.diagnostics.kolmogorov_smirnov",
        "KolmogorovSmirnovProfile",
        None,
        (None,),
    ),
    "hotelling": (
        "src.stats.tests.parametric.Multivariate.hotelling",
        "HotellingProfile",
        None,
        (None,),
    ),
    "box-m": ("src.stats.tests.diagnostics.box_m", "BoxMProfile", None, (None,)),
    "odds-ratio": (
        "src.stats.tests.Association.odds_ratio",
        "OddsRatioProfile",
        None,
        (None,),
    ),
    "risk-ratio": (
        "src.stats.tests.Association.risk_ratio",
        "RiskRatioProfile",
        None,
        (None,),
    ),
    "fisher-exact": (
        "src.stats.tests.nonParametric.Categorical.fisher_exact",
        "FisherExactProfile",
        None,
        (None,),
    ),
    "cmh": ("src.stats.tests.Association.cmh", "CMHProfile", None, (None,)),
    "bowker": (
        "src.stats.tests.nonParametric.Categorical.bowker",
        "BowkerProfile",
        None,
        (None,),
    ),
    "cochran-q": (
        "src.stats.tests.nonParametric.Categorical.cochran_q",
        "CochranQProfile",
        None,
        (None,),
    ),
    "adf": ("src.stats.tests.TimeSeries.adf", "ADFProfile", None, (None,)),
    "chi-square-homogeneity": (
        "src.stats.tests.nonParametric.Categorical.homogeneity",
        "ChiSquareHomogeneityProfile",
        None,
        (None,),
    ),
    "tf-classification": (
        "src.DeepLearning.TensorFlow.classification",
        "TFClassificationProfile",
        None,
        (None,),
    ),
    "tf-regression": (
        "src.DeepLearning.TensorFlow.regression",
        "TFRegressionProfile",
        None,
        (None,),
    ),
    "tf-rnn": (
        "src.DeepLearning.TensorFlow.rnn",
        "TFRNNProfile",
        "regression",
        ("classification", "regression"),
    ),
    "tf-time-series": (
        "src.DeepLearning.TensorFlow.time_series",
        "TFTimeSeriesProfile",
        None,
        (None,),
    ),
    "chi-square": (
        "src.stats.tests.nonParametric.Categorical.chi_square",
        "ChiSquareProfile",
        None,
        (None,),
    ),
    "independence": (
        "src.stats.tests.nonParametric.Categorical.independence",
        "IndependenceProfile",
        None,
        (None,),
    ),
    "mcnemar": (
        "src.stats.tests.nonParametric.Categorical.mcnemar",
        "McNemarProfile",
        None,
        (None,),
    ),
    "f-test": (
        "src.stats.tests.parametric.Variance.f_test",
        "FTestProfile",
        None,
        (None,),
    ),
    "normality": (
        "src.stats.tests.diagnostics.normality",
        "NormalityProfile",
        None,
        (None,),
    ),
    "homogeneity": (
        "src.stats.tests.diagnostics.homogeneity",
        "HomogeneityProfile",
        None,
        (None,),
    ),
    "xgboost": (
        "src.MachineLearning.Trees.xgboost",
        "XGBoostProfile",
        "classification",
        ("classification", "regression"),
    ),
    "catboost": (
        "src.MachineLearning.Trees.catboost",
        "CatBoostProfile",
        "classification",
        ("classification", "regression"),
    ),
    "lightgbm": (
        "src.MachineLearning.Trees.lightgbm",
        "LightGBMProfile",
        "classification",
        ("classification", "regression"),
    ),
    "adaboost": (
        "src.MachineLearning.Trees.adaboost",
        "AdaBoostProfile",
        "classification",
        ("classification", "regression"),
    ),
    "mlp": (
        "src.MachineLearning.NeuralNetworks.mlp",
        "MLPProfile",
        "classification",
        ("classification", "regression"),
    ),
    "perceptron": (
        "src.MachineLearning.NeuralNetworks.perceptron",
        "PerceptronProfile",
        "classification",
        ("classification",),
    ),
    "adaline": (
        "src.MachineLearning.NeuralNetworks.adaline",
        "AdalineProfile",
        "classification",
        ("classification",),
    ),
    "linear-regression": (
        "src.MachineLearning.LinearModels.linear_regression",
        "LinearRegressionProfile",
        "regression",
        ("regression",),
    ),
    "oneway-anova": (
        "src.stats.tests.parametric.Average.anova.one_way",
        "OneWayANOVAProfile",
        None,
        (None,),
    ),
    "twoway-anova": (
        "src.stats.tests.parametric.Average.anova.two_way",
        "TwoWayANOVAProfile",
        None,
        (None,),
    ),
    "manova": (
        "src.stats.tests.parametric.Multivariate.manova",
        "MANOVAProfile",
        None,
        (None,),
    ),
    "posthoc": (
        "src.stats.tests.parametric.Average.anova.posthoc",
        "PostHocProfile",
        None,
        (None,),
    ),
    "kmeans": ("src.MachineLearning.Clustering.kmeans", "KMeansProfile", None, (None,)),
    "dbscan": ("src.MachineLearning.Clustering.dbscan", "DBSCANProfile", None, (None,)),
    "hdbscan": (
        "src.MachineLearning.Clustering.hdbscan",
        "HDBSCANProfile",
        None,
        (None,),
    ),
    "k-distance": (
        "src.MachineLearning.Clustering.k_distance",
        "KDistanceProfile",
        None,
        (None,),
    ),
    "pca": ("src.MachineLearning.Decomposition.pca", "PCAProfile", None, (None,)),
    "lda": ("src.MachineLearning.Decomposition.lda", "LDAProfile", None, (None,)),
    "umap": ("src.MachineLearning.Decomposition.umap", "UMAPProfile", None, (None,)),
    "tsne": ("src.MachineLearning.Decomposition.tsne", "TSNEProfile", None, (None,)),
    "mds": ("src.MachineLearning.Decomposition.mds", "MDSProfile", None, (None,)),
    "som": ("src.MachineLearning.Decomposition.som", "SOMProfile", None, (None,)),
    "naive-bayes": (
        "src.MachineLearning.NaiveBayes.naive_bayes",
        "NaiveBayesProfile",
        "classification",
        ("classification",),
    ),
    "wrapper-selection": (
        "src.MachineLearning.FeatureSelection.wrapper",
        "WrapperSelectionProfile",
        "classification",
        ("classification", "regression"),
    ),
    "regularized-selection": (
        "src.MachineLearning.FeatureSelection.regularized",
        "RegularizedSelectionProfile",
        "classification",
        ("classification", "regression"),
    ),
    "apriori": ("src.Association.apriori", "AprioriProfile", None, (None,)),
    "sna": ("src.Networks.sna", "SNAProfile", None, (None,)),
}


def _unique_object(pairs: Any) -> Dict[str, Any]:
    """JSON 중복 키를 거부합니다.

    Args:
        pairs: JSON 디코더의 키·값 목록.
    Returns:
        Dict[str, Any]: 중복 없는 JSON 객체.
    Raises:
        ValueError: 중복 키.
    """
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("레시피의 중복 키: " + key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    """JSON 표준 밖의 NaN/Infinity를 거부합니다.

    Args:
        value: 비표준 수치 상수 이름.
    Returns:
        None: 항상 오류를 반환.
    Raises:
        ValueError: 비표준 수치 상수.
    """
    raise ValueError("레시피에 비유한 JSON 상수를 사용할 수 없습니다: " + value)


def load_recipe(
    model_name: str, task: Optional[str] = None, recipe_path: Optional[str] = None
) -> Dict[str, Any]:
    """모델·목적별 기본 파일 또는 사용자 JSON을 매번 읽습니다.

    Args:
        model_name: 등록 모델명. linear-regression, oneway-anova 등의 하이픈 이름 사용.
        task: 분류/회귀 선택. None이면 모델의 기본 목적. 통계 검정은 None만 허용.
        recipe_path: 사용자 JSON 경로. None이면 이 loader.py 옆의 모델-task.json 또는 검정명.json.
    Returns:
        Dict[str, Any]: 원본 JSON 설정. 캐시하거나 파일을 수정하지 않음.
    Raises:
        ValueError: 모델·task·JSON 형식·중복 키·비유한 값 오류.
        FileNotFoundError: 파일 누락. 내장 기본값으로 조용히 대체하지 않음.
        OSError: 파일 읽기 실패.
    """
    if model_name not in _REGISTRY:
        raise ValueError("지원하지 않는 모델명: " + str(model_name))
    _, _, default_task, supported = _REGISTRY[model_name]
    selected = default_task if task is None else task
    if selected not in supported:
        raise ValueError("해당 모델에서 지원하지 않는 task: " + str(selected))
    filename = model_name + ("-" + selected if selected is not None else "") + ".json"
    path = (
        Path(recipe_path)
        if recipe_path is not None
        else Path(__file__).resolve().parent / filename
    )
    with path.open(encoding="utf-8") as handle:
        data = json.load(
            handle, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    if not isinstance(data, dict):
        raise ValueError("레시피 최상위는 JSON 객체여야 합니다.")
    return data


def _normalize(model_name: str, values: Dict[str, Any]) -> Dict[str, Any]:
    """기존 XGBoost 별칭과 JSON 목록을 Profile 형식으로 변환합니다.

    Args:
        model_name: 등록된 모델명.
        values: 원본 레시피 또는 overrides 사전.
    Returns:
        Dict[str, Any]: 원 사전을 변경하지 않은 정규화 설정.
    Raises:
        ValueError: 별칭과 정식 키를 동시에 지정한 경우.
    """
    result = dict(values)
    if model_name == "xgboost":
        for alias, canonical in (("alpha", "reg_alpha"), ("lambda", "reg_lambda")):
            if alias in result:
                if canonical in result:
                    raise ValueError("레시피 키 충돌: " + alias + " / " + canonical)
                result[canonical] = result.pop(alias)
    for key in ("hidden_layer_sizes", "target_columns", "conv_filters"):
        if model_name == "feature-preprocessing" and key == "target_columns":
            continue  # FeaturePreprocessingProfile declares List[str]; sequence models use tuple.
        if key in result and isinstance(result[key], list):
            result[key] = tuple(result[key])
    return result


def load_profile(
    model_name: str,
    task: Optional[str] = None,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    """레시피를 읽고 Profile 필드·목적을 검증하여 구성합니다.

    Args:
        model_name: 등록 모델명.
        task: 선택할 분류/회귀 목적. 통계 검정은 None.
        recipe_path: 선택적 사용자 JSON 경로.
        overrides: 레시피 위에 적용할 명시적 Profile 설정. task 변경은 허용하지 않음.
    Returns:
        Any: 모델 전용 Profile. 설정 범위와 데이터 조건은 공개 실행기가 검증.
    Raises:
        ValueError: 알 수 없는 키·목적 불일치·별칭 충돌·JSON 오류.
        TypeError: overrides가 사전이 아닌 경우.
        FileNotFoundError/OSError: 레시피 파일 읽기 오류.
    Notes:
        누락된 필드는 Profile의 문서화된 기본값을 사용합니다.
        JSON에서 회귀를 선택하려면 task='regression'도 명시해야 기본 파일 선택과 일치합니다.
    """
    values = _normalize(model_name, load_recipe(model_name, task, recipe_path))
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise TypeError("overrides는 사전이어야 합니다.")
        updates = _normalize(model_name, overrides)
    else:
        updates = {}
    module_name, class_name, default_task, _ = _REGISTRY[model_name]
    # python -m에서 이미 실행 중인 모듈을 다시 import하면 Profile 클래스가 둘이 됩니다.
    main_module = sys.modules.get("__main__")
    if getattr(getattr(main_module, "__spec__", None), "name", None) == module_name:
        Profile = getattr(main_module, class_name)
    else:
        Profile = getattr(importlib.import_module(module_name), class_name)
    allowed = {field.name for field in fields(Profile)}
    # 오타가 overrides에 덮여 사라져도 원본 레시피의 오류를 알립니다.
    unknown = (set(values) | set(updates)) - allowed
    if unknown:
        raise ValueError("지원하지 않는 레시피 키: " + ", ".join(sorted(unknown)))
    selected = default_task if task is None else task
    if "task" in allowed:
        if any(layer.get("task", selected) != selected for layer in (values, updates)):
            raise ValueError("레시피/overrides의 task가 선택한 목적과 다릅니다.")
        values["task"] = selected
    values.update(updates)
    return Profile(**values)


def resolve_profile(
    model_name: str,
    profile: Any,
    recipe_path: Optional[str] = None,
    task: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    """공개 실행기의 명시적 Profile 또는 레시피 경로를 선택합니다.

    Args:
        model_name: 등록 모델명.
        profile: 명시적 Profile 또는 None.
        recipe_path: Profile이 None일 때 사용할 사용자 레시피.
        task: Profile이 None일 때 선택할 목적.
        overrides: Profile이 None일 때 레시피에 적용할 설정.
    Returns:
        Any: 명시적 Profile을 그대로 반환하거나 JSON에서 구성한 Profile.
    Raises:
        ValueError: Profile과 레시피 옵션을 함께 지정한 경우 또는 레시피 오류.
        TypeError: 잘못된 overrides 타입.
        FileNotFoundError/OSError: 레시피 파일 읽기 오류.
    """
    if profile is not None:
        if recipe_path is not None or task is not None or overrides is not None:
            raise ValueError(
                "명시적 Profile과 recipe_path/task/overrides는 함께 지정할 수 없습니다."
            )
        return profile
    return load_profile(
        model_name, task=task, recipe_path=recipe_path, overrides=overrides
    )
