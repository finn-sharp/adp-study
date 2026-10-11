# 참고 자료와 구현 대응

첨부 자료의 학습 절차를 재구성했다. 원본 자료를 저장소에 재배포하지 않는다.
모델 기본값을 조정했으므로 원본 출력과 동일한 수치를 보장하지 않는다.

| 자료 | 구현 |
|---|---|
| 01장 머신 러닝과 파이썬.ipynb | common.py (분할·스케일링) |
| 02장 최소 제곱법 모델.ipynb | Linear/least_squares.py, numerical.py (normal/SVD/TSVD) |
| 03장 로지스틱 회귀 모델.ipynb | Linear/logistic.py (스케일링 Pipeline) |
| 04장 라쏘 모델.ipynb | Linear/lasso.py, feature_selection.py (필터·CV) |
| 05장 릿지 회귀 모델_20230902.ipynb | Linear/ridge.py, numerical.py (폐형식 릿지) |
| 06장 결정 트리 모델.ipynb | Trees/decision_tree.py (회귀·분류·CV) |
| 07장 랜덤 포레스트 모델.ipynb | Trees/random_forest.py, feature_selection.py (퍼뮤테이션 중요도) |
| 08장 그레이디언트 부스팅 트리 모델.ipynb | Trees/gradient_boosting.py, adaboost.py, feature_selection.py, TextMining/text.py (LIME) |
| 09장 K-최근접 이웃 모델.ipynb | Neighbors/knn.py (회귀·분류) |
| 10장 서포트 벡터 머신 모델.ipynb | SVM/svm.py (분류·회귀 커널) |
| 11장 다층 퍼셉트론 모델.ipynb | NeuralNetworks/mlp.py, common.py (동일 분할 모델 비교) |
| 12장 K-평균 군집화 모델.ipynb | Clustering/clustering.py (KMeans·실루엣·ARI) |
| 13장 계층적 군집화 모델.ipynb | Clustering/clustering.py (계층적·연결 행렬) |
| 14장 주성분 분석 모델.ipynb | Decomposition/pca.py, manifold.py (PCA Pipeline·임베딩) |
| 1. 지도학습.pdf | 지도 학습 모델·선택적 XGBoost/LightGBM·Gaussian/Multinomial/Bernoulli NB·평가 |
| 2. 통계분석.pdf | 기술통계·정규성·등분산·평균/비모수/반복측정 검정·회귀 추론/진단·변수 선택·요인분석 |
| 2.5 카이제곱 분포와 F분포 - Must Learning with R.pdf | 분산 카이제곱 검정·분산비 F 검정 |
| 3. 연관분석.pdf | Association/apriori.py |
| 4. 군집화.pdf | 군집화·거리 행렬·계층 연결·DBSCAN·GaussianMixture |
| 5. 데이터시각화.pdf | Visualization/plots.py (기본 차트·히트맵·pairplot) |
| 7. 시계열분석.pdf | TimeSeries/forecasting.py (ADF/반복 차분/변환/분해·ARIMA/SARIMAX/VARMAX) |
| 8. 텍스트마이닝.pdf | TextMining/text.py (정제·Count/TF-IDF·토픽·선택적 NLTK/KoNLPy) |

## 재구성한 부분

- 교육용 직접 구현 셀은 sklearn 추정기로 재구성한다. 수치 선형 회귀 풀이만 별도 비교 구현한다.
- sklearn 0.23.2에 없는 as_frame=True, positive=True, rand_score를 기본 예제에서 사용하지 않는다.
- Lasso 모델 선택은 학습 폴드 내부 Pipeline + GridSearchCV다. LassoLarsIC AIC/BIC 모델은 별도로 제공하지 않는다.
- elasticnet 탐색은 solver=saga와 l1_ratio를 함께 지정한다.
- 계층 군집·MDS·t-SNE에 새 표본 predict/transform을 가정하지 않는다.
- 제거된 구 ARIMA API는 지원 ARIMA/SARIMAX/VARMAX API로 대체한다.
- LDA는 Count 입력의 sklearn 구현이며 gensim과 수치 결과가 같지는 않다.
- NLTK 코퍼스/한국어 JVM/LIME 기능은 선택적 의존성이다. 자동 다운로드하지 않으며 기본 테스트에는 포함하지 않는다.

## 원본 확인용 SHA-256

| 파일 | SHA-256 |
|---|---|
| 01장 머신 러닝과 파이썬.ipynb | `468cf25d32df7763e119631dcd583be04a494f0945082c4d895e25e542466090` |
| 02장 최소 제곱법 모델.ipynb | `56d1781dfb6d21e5059aa23fd6f65acee1518e4ebc37e176075e43308f5c05de` |
| 03장 로지스틱 회귀 모델.ipynb | `e66c0e25a7906eddcd5837e7596f0f6cf7de8311d1f2475ad15b76049137b228` |
| 04장 라쏘 모델.ipynb | `24496d83a0001a5137ac01c263117afaf18efc3cc744f8c37b3e3ee68b50aff2` |
| 05장 릿지 회귀 모델_20230902.ipynb | `2d833f6dac9dca735ecde6e39bb91b7d6f7387cdb146692a34ac6767bacc9107` |
| 06장 결정 트리 모델.ipynb | `a49625743ca7f63a52d908087fef0b23f05351e675dfd9a65371dd24042cd8c4` |
| 07장 랜덤 포레스트 모델.ipynb | `5cb2c17ee22344061dd3a04bae5b8b1a36afaed8def65cd290993fb31413a207` |
| 08장 그레이디언트 부스팅 트리 모델.ipynb | `f785276ebac89f6077e03fd18284be6f76514a09c645219daee62f81918923c3` |
| 09장 K-최근접 이웃 모델.ipynb | `addc3204ee3d4e01b6a31f5805e9ba9da10700da97fbdd814c1d85f684fd7fed` |
| 1. 지도학습.pdf | `b8cf2bccf2a554a9b8f2824ec9fa42663e2525bfa791cb6c5df2baa02d8fe3ed` |
| 10장 서포트 벡터 머신 모델.ipynb | `7e318d026ee88fd52db3781e3ecfb322d3f82f4ae62c379fbc3e19240ab49a83` |
| 11장 다층 퍼셉트론 모델.ipynb | `bc720ce6e05536e7e4a5d7d95c5ed48751609ddacbf653c79221a93aae057536` |
| 12장 K-평균 군집화 모델.ipynb | `31b6aee0de872f1d355bcd20067999c59ec53897097f3bac63f93af2ac65f8e4` |
| 13장 계층적 군집화 모델.ipynb | `5afec6a93e81ed1dea42fb6f1bec89b037875cd276dda6e2767e1f32b59da568` |
| 14장 주성분 분석 모델.ipynb | `cebeaa3ce30fafcbc9597a97699cc9dded5e4effd9b243b72ced43549230872c` |
| 2. 통계분석.pdf | `44f3de93ccd95d38fc02c219e8cb342e8552d03fce8a892e6ed540979f16d330` |
| 2.5 카이제곱 분포와 F분포 - Must Learning with R.pdf | `db3f77e0134b03d513eb0ef45459ccba1b3852e1af95cecd1ffbd3ef29bc5a21` |
| 3. 연관분석.pdf | `366971801afbf01d8515545b59194024565c0f33ca0c3bf9327d226938f9e88f` |
| 4. 군집화.pdf | `0c6cd078e6593eedd5693c3db812add893d619041159e496995e83a23fb97e86` |
| 5. 데이터시각화.pdf | `6c1ba36a2f32a60623692f6fb695fac4f051e82d6e2af405c73b38bf3cd1c893` |
| 7. 시계열분석.pdf | `b308c8cc024519a28ee16c1ea9e31c138aa4019847e59b64d6000b167f770193` |
| 8. 텍스트마이닝.pdf | `efb7dae09340a03cce9936bf241a41994e7c5fb7cf5d96e0d95cefc546ecc4c3` |
