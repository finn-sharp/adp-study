# 독립 평가 메트릭 전체 명세

표준 평가 모듈 **56개**, 전용 JSON **56개**, 평가 skeleton **56개**를 제공합니다. 개별 지표 50개·분야별 묶음 4개·모델 비교/CV 요약 2개입니다. 추가 의존성 없이 기존 NumPy/pandas/scikit-learn을 사용합니다.

## 공통 실행 계약

- 학습·분할·전처리 적합·그림 생성 없이 입력 예측만 평가합니다. 기존 7개 섹션과 전용 Profile/Output DTO를 유지합니다.
- 기본 JSON → overrides → Profile. 명시적 Profile과 recipe_path/overrides 동시 전달은 오류입니다.
- 공통 공개 인자: `y_true=None, y_pred=None, profile=None, *, y_score=None, X=None, y_train=None, recipe_path=None, overrides=None`.
- 분류/확률 입력은 1D 단일 타깃입니다. multilabel·sample_weight·추론용 유의성 검정은 이 구현의 지원 범위가 아닙니다.
- 정답·예측·점수는 같은 행 순서입니다. shape·유한성·클래스 포함·확률 범위를 검증하며 결측 행을 자동 제거하지 않습니다.
- 개별 지표가 수학적으로 정의 불가이면 UndefinedMetric(ValueError). 묶음 평가만 해당 지표를 NaN(JSON null)과 warnings에 보존합니다. 잘못된 입력/설정은 묶음에서도 오류입니다.
- 입력이 없는 메트릭 계열은 묶음에서 생략합니다: 점수 미제공 시 확률/순위 계열, 군집 참조 라벨 미제공 시 외부 지표, X 미제공 시 내부 지표, y_train 미제공 시 scaled forecast 계열.

### 입력 배열 계약

| 인자 | 형태·의미 |
| --- | --- |
| y_true/y_pred 분류 | (n,) 이산 숫자 또는 문자열 라벨 |
| y_score | (n,) 양성 점수 또는 (n,classes) 확률. 행렬은 Profile.labels로 열 순서 필수 지정 |
| y_true/y_pred 회귀 | 같은 (n,) 또는 (n,targets) |
| y_pred 군집 / X | (n,) 군집 라벨 / 같은 전처리 좌표 (n,p); noise 마스크를 X·정답에도 함께 적용 |
| y_true/y_pred 시계열 | (n,), (창,horizon), (창,horizon,targets). 2D는 horizon 축으로 해석 |
| y_train | 시간 순 (시간,) 또는 (시간,targets); 평가 자료를 분모에 사용하지 않음 |

### 확률·순위·군집 경계

- 1D score는 positive_label의 점수입니다. ROC/AP/KS/Lift는 유한 decision score도 허용합니다. Log Loss/Brier/calibration은 [0,1] 확률만 허용합니다.
- 2D 확률은 [0,1]·행 합 1(절대 오차 1e-7)·명시 클래스 순서가 필요합니다. sklearn 내부 정렬과 달라도 정답을 그 순서로 인코딩합니다.
- ROC/AP/Lift/Gain 등 이진 순위 평가에는 실제 양성·음성이 모두 있어야 합니다. 다중 클래스 ROC/AP는 모든 클래스가 평가 자료에 있어야 합니다.
- top-k는 ceil(n*f), score 동률은 원 입력 행 순서를 유지합니다. tie가 많은 경우 입력 순서에 결과가 의존하므로 details.tie_policy를 보존합니다.
- Lift/Gain curves의 details.curve는 origin을 포함하고 value는 origin을 제외합니다. Gain origin은 0, Lift origin은 정의 불가 null입니다. Lift는 연관규칙의 confidence/support와 별도입니다.
- 군집은 기본 exclude_noise=True, noise_label=-1. 잡음 제외 후 내부 지표에 필요한 군집 수를 확인합니다. 내부 지표의 X는 실제 적합 좌표를 사용해야 합니다.
- Davies–Bouldin/Calinski–Harabasz는 유클리드 좌표 기준입니다. distance 설정은 silhouette에만 적용합니다.
- MASE/RMSSE는 학습 구간 seasonal_period lag 차분으로 분모를 계산합니다. 학습 구간이 lag보다 짧거나 naive 오차가 0이면 정의 불가입니다.

## Profile 전체 필드

### classification

같은 분야의 개별 지표와 묶음은 동일 설정 필드를 직접 선언합니다. 해당 지표에 적용되는 설정만 계산에 사용합니다.

| 필드 | 타입 | 기본 JSON | 의미·제약 |
| --- | --- | --- | --- |
| `average` | `str` | `"weighted"` | binary/micro/macro/weighted/None 집계; None은 클래스별 배열. |
| `positive_label` | `Any` | `1` | binary 지표·순위 확률의 양성 클래스. |
| `labels` | `Any` | `null` | 확률 열의 클래스 순서. 행렬 확률은 반드시 지정. |
| `zero_division` | `int` | `0` | precision/recall/F의 0 분모 대체값: 0 또는 1. |
| `beta` | `float` | `1.0` | F-beta의 유한 양수 beta. |
| `top_fractions` | `Any` | `[0.1, 0.2, 0.3]` | Lift/Gain 선별 비율 (0,1]; ceil(n*f), 동률은 원 행 순서. |
| `n_bins` | `int` | `10` | calibration 동일 너비 bin 수, 양의 정수. |

### regression

같은 분야의 개별 지표와 묶음은 동일 설정 필드를 직접 선언합니다. 해당 지표에 적용되는 설정만 계산에 사용합니다.

| 필드 | 타입 | 기본 JSON | 의미·제약 |
| --- | --- | --- | --- |
| `multioutput` | `str` | `"uniform_average"` | uniform_average/raw_values; 다중 타깃 집계. |
| `zero_policy` | `str` | `"raise"` | MAPE: raise/omit; 0 실제값의 명시 처리. |
| `n_features` | `int` | `0` | adjusted R² 설명변수 수; n > p+1. |
| `quantile` | `float` | `0.5` | pinball loss의 분위수 (0,1). |

### clustering

같은 분야의 개별 지표와 묶음은 동일 설정 필드를 직접 선언합니다. 해당 지표에 적용되는 설정만 계산에 사용합니다.

| 필드 | 타입 | 기본 JSON | 의미·제약 |
| --- | --- | --- | --- |
| `exclude_noise` | `bool` | `true` | True이면 noise_label 행을 모든 군집 평가에서 함께 제외. |
| `noise_label` | `Any` | `-1` | 제외할 예측 군집 라벨. |
| `distance` | `str` | `"euclidean"` | silhouette 거리: euclidean/manhattan/cosine. |

### time_series

같은 분야의 개별 지표와 묶음은 동일 설정 필드를 직접 선언합니다. 해당 지표에 적용되는 설정만 계산에 사용합니다.

| 필드 | 타입 | 기본 JSON | 의미·제약 |
| --- | --- | --- | --- |
| `seasonal_period` | `int` | `1` | MASE/RMSSE 학습 naive 차분 lag; 양의 정수. |

## 개별 모듈·레시피·계산 정의

| 분야 | 모듈 / runner | 레시피 | 정의 |
| --- | --- | --- | --- |
| classification | [accuracy / `run_accuracy`](../src/Evaluation/classification/accuracy.py) | `metric-classification-accuracy.json` | 맞은 예측 비율. |
| classification | [precision / `run_precision`](../src/Evaluation/classification/precision.py) | `metric-classification-precision.json` | TP/(TP+FP); average에 따른 OvR 집계. |
| classification | [recall / `run_recall`](../src/Evaluation/classification/recall.py) | `metric-classification-recall.json` | TP/(TP+FN); average에 따른 OvR 집계. |
| classification | [f1 / `run_f1`](../src/Evaluation/classification/f1.py) | `metric-classification-f1.json` | precision·recall의 조화평균. |
| classification | [fbeta / `run_fbeta`](../src/Evaluation/classification/fbeta.py) | `metric-classification-fbeta.json` | (1+β²)PR/(β²P+R). |
| classification | [specificity / `run_specificity`](../src/Evaluation/classification/specificity.py) | `metric-classification-specificity.json` | TN/(TN+FP); 다중 클래스는 OvR, weighted는 실제 클래스 빈도. |
| classification | [balanced_accuracy / `run_balanced_accuracy`](../src/Evaluation/classification/balanced_accuracy.py) | `metric-classification-balanced-accuracy.json` | 평가 자료에 존재하는 클래스 recall의 평균. |
| classification | [mcc / `run_mcc`](../src/Evaluation/classification/mcc.py) | `metric-classification-mcc.json` | Matthews correlation; 다중 클래스 지원. |
| classification | [cohen_kappa / `run_cohen_kappa`](../src/Evaluation/classification/cohen_kappa.py) | `metric-classification-cohen-kappa.json` | 우연 일치 보정 일치도. 한 클래스만 존재하면 정의 불가. |
| classification | [confusion_matrix / `run_confusion_matrix`](../src/Evaluation/classification/confusion_matrix.py) | `metric-classification-confusion-matrix.json` | 행=정답, 열=예측, details.class_order 순서. |
| classification | [roc_auc / `run_roc_auc`](../src/Evaluation/classification/roc_auc.py) | `metric-classification-roc-auc.json` | 이진 ROC 면적; 다중 클래스 OvR macro/weighted/None. |
| classification | [average_precision / `run_average_precision`](../src/Evaluation/classification/average_precision.py) | `metric-classification-average-precision.json` | recall 증가량으로 가중한 precision; 다중 클래스 OvR 지원. |
| classification | [pr_auc / `run_pr_auc`](../src/Evaluation/classification/pr_auc.py) | `metric-classification-pr-auc.json` | 이진 precision-recall 곡선의 사다리꼴 면적; AP와 구분. |
| classification | [log_loss / `run_log_loss`](../src/Evaluation/classification/log_loss.py) | `metric-classification-log-loss.json` | 올바른 클래스 확률의 음의 로그 평균; 1e-15 clipping 후 행 정규화. |
| classification | [brier_score / `run_brier_score`](../src/Evaluation/classification/brier_score.py) | `metric-classification-brier-score.json` | 이진 양성 확률 MSE [0,1]; 다중 클래스 one-hot 제곱오차 합 평균 [0,2]. |
| classification | [gini / `run_gini`](../src/Evaluation/classification/gini.py) | `metric-classification-gini.json` | 이진 2*ROC-AUC-1; 경제학의 분포 Gini와 구분. |
| classification | [ks / `run_ks`](../src/Evaluation/classification/ks.py) | `metric-classification-ks.json` | 이진 score의 양성/음성 누적 분포 최대 절대 차이; max(abs(TPR-FPR)), 방향/threshold 함께 제공. |
| classification | [lift / `run_lift`](../src/Evaluation/classification/lift.py) | `metric-classification-lift.json` | 상위 ceil(n*f) 실제 양성률 / 전체 실제 양성률. 요청 비율이 아닌 실제 선택 비율로 계산. |
| classification | [gain / `run_gain`](../src/Evaluation/classification/gain.py) | `metric-classification-gain.json` | 상위 ceil(n*f)의 양성 수 / 전체 양성 수. |
| classification | [lift_curve / `run_lift_curve`](../src/Evaluation/classification/lift_curve.py) | `metric-classification-lift-curve.json` | score 내림차순 순위별 cumulative lift; origin의 lift는 NaN. |
| classification | [gain_curve / `run_gain_curve`](../src/Evaluation/classification/gain_curve.py) | `metric-classification-gain-curve.json` | score 내림차순 누적 gain; origin=0, 최종=1. |
| classification | [roc_curve / `run_roc_curve`](../src/Evaluation/classification/roc_curve.py) | `metric-classification-roc-curve.json` | drop_intermediate=False의 FPR/TPR/threshold 전체. |
| classification | [precision_recall_curve / `run_precision_recall_curve`](../src/Evaluation/classification/precision_recall_curve.py) | `metric-classification-precision-recall-curve.json` | precision/recall/threshold; terminal point에는 threshold 없음. |
| classification | [calibration / `run_calibration`](../src/Evaluation/classification/calibration.py) | `metric-classification-calibration.json` | 동일 너비 bin별 평균 확률·실제 양성률·표본 수 및 ECE; 빈 bin 제외. |
| classification | [metrics / `run_classification_metrics`](../src/Evaluation/classification/metrics.py) | `metric-classification-metrics.json` | 같은 분야 모든 가용 지표를 계산; 상세 값은 value 사전. |
| regression | [mae / `run_mae`](../src/Evaluation/regression/mae.py) | `metric-regression-mae.json` | mean(abs(actual-prediction)). |
| regression | [mse / `run_mse`](../src/Evaluation/regression/mse.py) | `metric-regression-mse.json` | mean((actual-prediction)^2). |
| regression | [rmse / `run_rmse`](../src/Evaluation/regression/rmse.py) | `metric-regression-rmse.json` | 타깃별 sqrt(MSE) 이후 타깃 집계. |
| regression | [r2 / `run_r2`](../src/Evaluation/regression/r2.py) | `metric-regression-r2.json` | 1-SSE/SST; 최소 2행, 비상수 실제값 필요; sklearn의 상수 타깃 보정값을 사용하지 않음. |
| regression | [adjusted_r2 / `run_adjusted_r2`](../src/Evaluation/regression/adjusted_r2.py) | `metric-regression-adjusted-r2.json` | 1-(1-R²)*(n-1)/(n-p-1); n_features=p, n>p+1. 설명변수 수와 선형모형 해석 책임은 호출자. |
| regression | [median_absolute_error / `run_median_absolute_error`](../src/Evaluation/regression/median_absolute_error.py) | `metric-regression-median-absolute-error.json` | 타깃별 절대오차 중앙값. |
| regression | [mape / `run_mape`](../src/Evaluation/regression/mape.py) | `metric-regression-mape.json` | 100*mean(abs(error/actual)); 0 실제값 raise 또는 타깃별 omit. |
| regression | [smape / `run_smape`](../src/Evaluation/regression/smape.py) | `metric-regression-smape.json` | 200*mean(abs(error)/(abs(actual)+abs(prediction))); 0/0=0. |
| regression | [wape / `run_wape`](../src/Evaluation/regression/wape.py) | `metric-regression-wape.json` | 100*sum(abs(error))/sum(abs(actual)); 타깃별 분모 0은 정의 불가. |
| regression | [msle / `run_msle`](../src/Evaluation/regression/msle.py) | `metric-regression-msle.json` | mean((log1p(actual)-log1p(prediction))²); 실제/예측 모두 비음수. |
| regression | [rmsle / `run_rmsle`](../src/Evaluation/regression/rmsle.py) | `metric-regression-rmsle.json` | sqrt(MSLE), 타깃별 계산 후 집계. |
| regression | [explained_variance / `run_explained_variance`](../src/Evaluation/regression/explained_variance.py) | `metric-regression-explained-variance.json` | 1-Var(error)/Var(actual); 비상수 실제값 필요. |
| regression | [max_error / `run_max_error`](../src/Evaluation/regression/max_error.py) | `metric-regression-max-error.json` | 타깃별 최대 절대오차; 이후 설정대로 집계. |
| regression | [pinball_loss / `run_pinball_loss`](../src/Evaluation/regression/pinball_loss.py) | `metric-regression-pinball-loss.json` | mean(max(q*(actual-pred),(q-1)*(actual-pred))). |
| regression | [metrics / `run_regression_metrics`](../src/Evaluation/regression/metrics.py) | `metric-regression-metrics.json` | 같은 분야 모든 가용 지표를 계산; 상세 값은 value 사전. |
| clustering | [silhouette / `run_silhouette`](../src/Evaluation/clustering/silhouette.py) | `metric-clustering-silhouette.json` | 표본별 (b-a)/max(a,b) 평균; 지정 거리, 2<=군집수<n. |
| clustering | [davies_bouldin / `run_davies_bouldin`](../src/Evaluation/clustering/davies_bouldin.py) | `metric-clustering-davies-bouldin.json` | 유클리드 중심 기반 DB index; 작을수록 좋음. |
| clustering | [calinski_harabasz / `run_calinski_harabasz`](../src/Evaluation/clustering/calinski_harabasz.py) | `metric-clustering-calinski-harabasz.json` | 유클리드 군집간/군집내 분산비 CH index; 클수록 좋음. |
| clustering | [ari / `run_ari`](../src/Evaluation/clustering/ari.py) | `metric-clustering-ari.json` | 우연 보정 Rand Index; 라벨 숫자/이름 자체의 대응 불필요. |
| clustering | [ami / `run_ami`](../src/Evaluation/clustering/ami.py) | `metric-clustering-ami.json` | 우연 보정 mutual information; arithmetic 평균 정규화. |
| clustering | [nmi / `run_nmi`](../src/Evaluation/clustering/nmi.py) | `metric-clustering-nmi.json` | mutual information arithmetic 평균 정규화. |
| clustering | [homogeneity / `run_homogeneity`](../src/Evaluation/clustering/homogeneity.py) | `metric-clustering-homogeneity.json` | 각 예측 군집의 참조 클래스 순도 entropy 지표. |
| clustering | [completeness / `run_completeness`](../src/Evaluation/clustering/completeness.py) | `metric-clustering-completeness.json` | 각 참조 클래스가 같은 군집에 모이는 entropy 지표. |
| clustering | [v_measure / `run_v_measure`](../src/Evaluation/clustering/v_measure.py) | `metric-clustering-v-measure.json` | homogeneity와 completeness의 조화평균. |
| clustering | [metrics / `run_clustering_metrics`](../src/Evaluation/clustering/metrics.py) | `metric-clustering-metrics.json` | 같은 분야 모든 가용 지표를 계산; 상세 값은 value 사전. |
| time_series | [mase / `run_mase`](../src/Evaluation/time_series/mase.py) | `metric-time-series-mase.json` | 타깃별 평가 MAE / 학습 seasonal naive MAE, 이후 타깃 평균. |
| time_series | [rmsse / `run_rmsse`](../src/Evaluation/time_series/rmsse.py) | `metric-time-series-rmsse.json` | 타깃별 sqrt(평가 MSE / 학습 seasonal naive MSE), 이후 타깃 평균. |
| time_series | [horizon_errors / `run_horizon_errors`](../src/Evaluation/time_series/horizon_errors.py) | `metric-time-series-horizon-errors.json` | 창별 미래 horizon/target 축을 유지하는 MAE/RMSE 표와 전체 MAE/RMSE. |
| time_series | [metrics / `run_time_series_metrics`](../src/Evaluation/time_series/metrics.py) | `metric-time-series-metrics.json` | 같은 분야 모든 가용 지표를 계산; 상세 값은 value 사전. |

## Output DTO 전체 필드

56개 중 개별 메트릭/분야별 묶음 54개는 아래 필드를 각 전용 클래스에 직접 선언합니다. 클래스명은 분야+메트릭의 PascalCase (`ClassificationLiftProfile`, `ClassificationLiftOutputDTO` 등)입니다.

| 필드 | 타입 | 의미 |
| --- | --- | --- |
| metric_name | str | 분야-메트릭 이름 |
| value | Any | 스칼라, 클래스별/타깃별 배열, 곡선 배열, 또는 묶음 지표 사전 |
| n_samples | int | 평가 행 수; 군집은 잡음 제외 후 행 수, 시계열은 창 수 |
| details | dict | parameters와 순서·정의·표. 묶음은 metrics[메트릭명]에 개별 details 보존 |
| warnings | list[str] | 묶음의 정의 불가 지표별 사유 |

details의 추가 필드: 분류 class_order; confusion matrix의 matrix; ROC/PR/Lift/Gain의 curve; Lift/Gain의 top, baseline_rate, tie_policy; KS threshold/signed_difference; calibration bins; 회귀 per_target 및 MAPE omitted_per_target; 군집 excluded_rows/evaluation_indices; MASE/RMSSE training_scale/per_target; horizon_errors의 mae_by_horizon_target/rmse_by_horizon_target/table. 모든 모듈의 parameters는 적용 Profile 전체 필드를 보존합니다.

## 모델 비교와 CV 요약

`run_comparison(y_true, predictions, profile=None, *, scores=None, classes=None, recipe_path=None, overrides=None)`는 한 정답에 모델명→예측 사전을 평가합니다. scores/classes도 모델명별 사전입니다. 모델마다 확률 열 순서를 별도로 검증합니다.

| ComparisonProfile 필드 | 타입 | 기본값 | 제약 |
| --- | --- | --- | --- |
| evaluation_task | str | classification | classification/regression |

| ComparisonOutputDTO 필드 | 타입 | 의미 |
| --- | --- | --- |
| table | DataFrame | 행=모델, 열=유한 스칼라 지표. 모델에서 사용할 수 없는 지표는 NaN |
| directions | dict | 각 지표 higher/lower; calibration ECE는 lower |
| n_samples | int | 동일 평가 정답 행 수 |
| parameters | dict | evaluation_task |
| warnings | list | 같은 행 순서 책임, 통계적 유의성/최종 test 모델 선택 제한 |

`run_cv_summary(scores, profile=None, *, recipe_path=None, overrides=None)`는 지표명→1D 폴드 점수 사전을 받습니다. 지표 계산·학습·CV 자체를 수행하지 않습니다.

| CVSummaryProfile 필드 | 타입 | 기본값 | 제약 |
| --- | --- | --- | --- |
| ddof | int | 1 | 0 또는 1; std 자유도 보정 |
| nan_policy | str | raise | raise/omit; Infinity는 항상 오류 |

| CVSummaryOutputDTO 필드 | 타입 | 의미 |
| --- | --- | --- |
| table | DataFrame | metric/count/mean/std/min/max/omitted |
| parameters | dict | ddof/nan_policy |
| warnings | list | 폴드 수 부족 시 std=null; 표준편차는 신뢰구간 아님 |

전용 레시피는 metric-comparison.json / metric-cv-summary.json입니다. 예측 행 순서는 자동으로 검증할 식별자가 없으므로 호출자가 보장해야 합니다.

## 기존 실행기·skeleton 연결

- 기존 지도학습 common과 ML 지도/변수선택 skeleton은 독립 평가 어댑터를 사용하며 기존 accuracy/precision_weighted/recall_weighted/f1_weighted/mae/mse/rmse/r2 키를 유지합니다.
- 기존 scalar metrics 사전에는 유한 스칼라만 추가합니다. 정의 불가 지표의 사유·null 및 곡선/표/클래스별 값은 독립 평가 DTO로 조회합니다. adjusted R²는 모형과 설명변수 수의 의도적 지정이 필요하므로 호환 어댑터에서 자동 보고하지 않습니다.
- TensorFlow 공통 학습 결과와 모든 DL skeleton에도 연결합니다. 시계열 skeleton은 원 train_indices·선택 target_columns의 자료로 MASE/RMSSE를 계산하고 horizon 표를 출력합니다.
- KMeans/DBSCAN/HDBSCAN skeleton은 실제 전처리 좌표의 내부 지표와 Iris 참조 라벨의 외부 지표를 평가합니다. 전체 자료 탐색이라는 범위를 유지합니다.
- 평가 skeleton 56개는 src/Evaluation/skeleton에 있습니다. 각 지표의 같은 분야/파일명 경로이며, comparison.py/cv_summary.py는 루트에 있습니다. 모든 skeleton은 load_data/load_recipe/build_pipeline/main을 제공합니다.
- UCI 이진 Iris: 학습 구간 StandardScaler+LogisticRegression → 보류 라벨/양성 확률; Wine Quality: 학습 구간 scaler+OLS → 보류 회귀; 군집 Iris: scaler+KMeans 전체 자료 탐색; Bike: 시간 순 80/20 분할 후 직전 관측의 2-step naive 예측.
- CV skeleton은 각 폴드 내부에서 scaler와 모델을 재적합하고 독립 macro F1을 계산합니다. 공유 모델/분할 상태를 메트릭 모듈 안에 저장하지 않습니다.

## 사용 예제

```python
from src.Evaluation.classification.lift import run_lift
from src.Evaluation.classification.metrics import run_classification_metrics
from src.Evaluation.regression.rmse import run_rmse
from src.Evaluation.time_series.mase import run_mase

lift = run_lift(y_true, y_score=positive_scores,
                overrides={"positive_label": 1, "top_fractions": [0.1, 0.2]})
print(lift.value, lift.details["top"])
evaluation = run_classification_metrics(y_true, predictions, y_score=probabilities,
                overrides={"labels": model.classes_.tolist(), "positive_label": 1})
print(evaluation.value, evaluation.warnings)
print(evaluation.to_json(indent=2))
rmse = run_rmse(actual, predicted)
mase = run_mase(future_actual, future_predicted, y_train=training_series,
                overrides={"seasonal_period": 7})
```

```bash
python -m examples.evaluation_metric_examples
python -m src.Evaluation.skeleton.classification.lift
python -m src.Evaluation.skeleton.regression.rmse
python -m src.Evaluation.skeleton.clustering.silhouette
python -m src.Evaluation.skeleton.time_series.mase
python -m src.Evaluation.skeleton.comparison
python -m src.Evaluation.skeleton.cv_summary
```

## 수치 기준

표준 지표는 [scikit-learn 공식 평가 문서](https://scikit-learn.org/stable/modules/model_evaluation.html)를 기준으로 참조값을 검증합니다. 별도 정의인 Lift/Gain·백분율·MASE/RMSSE는 위 수식과 손계산 테스트로 검증합니다. Python 3.7은 sklearn 1.0.2에서 제공하는 API만 사용하며 실제 실행은 CI의 별도 환경으로 확인합니다.
