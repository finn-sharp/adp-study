# ADP 모듈 구조 및 작성 규칙

기존 `src/MachineLearning`, `src/stats/tests`, `src/Recipes`, `codebook` 구조를 유지하고
학습 절차를 재사용 가능한 함수로 확장한다. 기존 EDA·전처리 노트북은 이번 추가 모듈의
실행 의존성이 아니다.

| 경로 | 책임 |
|---|---|
| `src/MachineLearning/common.py` | 지도 학습 Profile/Result, 입력 검증, 분할, Pipeline, 학습 데이터 CV, 평가 |
| `src/MachineLearning/Linear` | 최소제곱, 로지스틱, Lasso, Ridge, 교육용 수치 풀이 |
| `src/MachineLearning/Trees` | 결정 트리, RF, GB, AdaBoost, 선택적 XGBoost/LightGBM |
| `src/MachineLearning/Neighbors`, `SVM`, `NeuralNetworks`, `NaiveBayes` | 알고리즘별 Profile와 실행기 |
| `src/MachineLearning/Clustering` | 군집화, 거리 행렬, 연결 행렬 |
| `src/MachineLearning/Decomposition` | PCA, 요인분석, PCA-분류 Pipeline, 탐색적 임베딩 |
| `src/stats` | 기술통계·가설 검정·회귀 추론·반복측정 ANOVA |
| `src/Association`, `TimeSeries`, `TextMining`, `Visualization` | 분야별 Profile/Result와 실행기 |
| `src/Recipes` | JSON 하이퍼파라미터 설정. 모델 Profile 생성에 전달 |
| `codebook` | 모듈을 호출하는 설명·실행 노트북 |
| `examples` | 외부 데이터 없이 실행 가능한 통합 예제 |
| `docs/specs` | 코드 파일별 명세, Profile 필드, 함수 계약 |
| `tests` | 수치 기준 비교, 누수 방지, 입력 검증, 모델 실행 회귀검사 |

## 코드별 문서 명세

1. 파일 상단: 파일 이름, 목적, 참고 자료, 작성 정보, 변경 이력.
2. Profile: 각 설정의 의미·기본값·유효 범위. 공통 필드는 상속 관계를 명시.
3. Result: 모든 반환 필드의 의미, 행/열/클래스 순서, 지표 단위, 선택적 값의 조건.
4. 공개 함수: `Args`, `Returns`, `Raises`, 필요하면 가정·제한을 `Notes`에 설명.
5. 실행 예제: 저장소 루트의 `python -m ...` 또는 codebook에서 모듈 호출.
6. 구현 변경 시 docstring, `docs/specs`, Recipes, 테스트를 함께 갱신.

코드 내 docstring이 입력·출력 계약의 기준이며 `docs/specs`는 코드별 탐색용 명세다.
통계 검정에서 `Literal` 등의 타입 힌트만으로 값 검증이 실행되는 것은 아니다.

## 데이터 흐름과 평가 경계

지도 학습은 행 위치를 분할한 뒤 `scaler → optional transformer → model` Pipeline을
학습한다. `GridSearchCV`는 학습 구간만 받고 스케일러·PCA도 각 폴드 안에서 적합한다.
보류 테스트는 최종 평가에만 사용한다. `train_indices`, `test_indices`는 원본 입력의
행 위치이며 DataFrame의 라벨 인덱스가 아니다. 새 입력의 열 순서는 학습과 같아야 한다.

회귀 지표는 MAE/MSE/RMSE/R², 분류 지표는 accuracy와 가중 precision/recall/F1이다.
이진 확률 모델의 ROC-AUC 양성 클래스는 `result.classes[1]`이다. 테스트에 두 클래스가
있어야 AUC를 반환한다. 지표 값은 비율이며 백분율이 아니다.

군집화·차원 축소·변수 선택·회귀 추론은 기본적으로 전체 전달 데이터의 탐색적 분석이다.
학습 데이터에서 얻은 설명력이나 정확도를 일반화 성능으로 취급하지 않는다.
특성 선택을 CV와 결합하려면 미적합 selector를 학습 Pipeline 안에 넣는다.

시계열은 마지막 `holdout`을 보류하고 앞부분으로만 적합·AIC 탐색한다. ARIMA의 `d`와
SARIMAX의 `D`는 모형 내부 차분이므로 원 시계열을 입력한다. 진단용 변환/차분 결과와
예측 입력을 구분한다. 반복 차분은 `.diff()`를 여러 번 적용하며 `.diff(2)`와 다르다.

텍스트 분류는 분할 후 어휘/IDF를 적합한다. 외부 형태소 분석기는 tokenizer 콜백으로
연결하며 기본 실행 중에는 코퍼스 다운로드를 수행하지 않는다. 기본 토큰 패턴은
두 글자 이상 단어를 사용한다. `clean_text`는 소문자화를 자동 수행하지 않는다.

## 지원 환경

- 기준 시험 재현 환경: Python 3.7, sklearn 0.23.2, scipy 1.7.3 등 `requirements-ml.txt`.
- 개발 검증 환경: Python 3.12 및 `requirements-review.txt`. 시험 재현 환경과 별도 가상환경 사용.
- 버전 차이는 feature-name API와 AgglomerativeClustering의 affinity/metric 이름에 한정해 처리.
- XGBoost/LightGBM/KoNLPy/NLTK는 선택적 사용. 해당 모듈을 사용하지 않는 기본 import를 막지 않는다.
- 무작위 지도 학습 분할은 IID 자료용이다. 시계열·그룹 자료에는 별도 분할이 필요하다.
- 희소 수치 행렬·다중 타깃·범주형 자동 인코딩은 공통 지도 학습 실행기의 범위 밖이다.
- 기존 `xgboost-regression.json`은 학습 라이브러리용 레시피다. 모든 키를 `XGBoostProfile`에
  그대로 넣을 수 있는 형식으로 변경하지 않았으며 신규 Profile 레시피는 `xgboost.json`이다.

## 검증

```bash
python -m unittest discover -s tests -v
python -m examples.study_workflow
```

테스트는 scipy/statsmodels 기준 결과, Type I/II/III ANOVA 행 추출, 동일 시드 분할,
학습 전용 스케일러/PCA/어휘, 시간 보류 예측, 데이터 오류와 선택적 모듈을 확인한다.
선택적 부스팅 패키지가 없으면 해당 테스트만 skip한다. 외부 원본 자료의 모든 셀을
그대로 실행하거나 벤치마크 결과를 재현하는 테스트는 아니다.

## 회귀·시계열·리샘플링 확장 (2026-10-11)

- `src/MachineLearning/Regression`: 다중 선형·다항·임의 비선형 곡선 회귀.
- `src/TimeSeries/arima.py`: 기존 forecast 엔진을 재사용한 ARIMA/SARIMA 전용 API.
- `src/Preprocessing/sampling.py`: 수치형·범주형·혼합형 지원 범위를 명시한 증강 5종·감소 5종.
- 리샘플링 CV는 imblearn Pipeline을 사용하며 학습 폴드만 변환한다.

[지원 표·호출 예제·환경 제한](EXTENSIONS.md)을 참조한다.

## Python 3.7 및 분산분석 확장

전체 모듈 환경은 `requirements-py37.txt`의 Python 3.7/sklearn 1.0.2/imblearn 0.8.1 조합이다.
기존 시험용 고정 파일은 별도 환경으로 유지한다.
`src/stats/anova.py`는 기존 ANOVA 계산을 재사용한 long-form 통합 API,
`manova.py`는 여러 반응의 요인별 다변량 검정, `posthoc.py`는 독립 집단 가족 단위 비교다.
[구체적 명세와 가정](MULTIVARIATE.md)을 참조한다.

## Output DTO 공통 계약

모든 공개 분석 실행 API는 `src/output.py`의 OutputDTO를 상속한 명시적 dataclass를 반환한다.
모델별 DTO는 공통 지도학습 결과 필드를 상속하고, 기존 Result 이름은 호환 별칭으로 유지한다.
DTO는 필드/키 접근과 native 사전·JSON 변환을 제공하며 실행 객체와 분석 데이터의 저장 규칙을
구분한다. 구체적 반환 목록과 호환 경계는 [Output DTO 명세](OUTPUT_DTOS.md)를 참고한다.
