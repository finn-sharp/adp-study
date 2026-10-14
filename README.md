# ADP 분석 모듈과 학습 자료

ADP 학습 노트북과 재사용 가능한 분석 실행기를 관리합니다. 구현된 코드는
**Profile → 모델별 JSON 레시피 → 실행 함수 → 전용 Output DTO**를 사용합니다.
현재 표준 분석·시각화 모듈은 **192개**, 모델·목적별 레시피는 **219개**입니다.
기존 단일·독립 두 표본 Z/t도 표준 공개 모듈로 정비하고 이전 호출을 호환 경로로 유지합니다.

## 기출·모의 복원 및 파라미터 최적화

- [기출문제 학습 묶음](기출문제/README.md): 기존 15·17·20·22·23회, 모의고사 1~4회, 18·19·21·24·25·26·28회 및 22회 보충. 문제지 PDF와 별도 실행 완료 풀이를 회차별로 제공합니다.
- [최적화 명세](docs/PARAMETER_OPTIMIZATION.md): GridSearchCV·RandomizedSearchCV·Optuna TPE, Profile/DTO·모델별 JSON 및 RF/LightGBM skeleton. CV는 학습 자료만 사용합니다.
- [자료 대조·복원 가정](docs/EXAM_RECONSTRUCTION_AUDIT.md): 원 자료가 없는 문항은 합성 연습 자료이며 공식 원문/정답과 구분합니다.

## 구현 범위

| 영역 | 구현 내용 |
| --- | --- |
| Feature 전처리 | 수치형 결측·스케일·power, 범주형 결측·onehot/ordinal/frequency, FeatureSpec 순서 조합, 별도 RoPE |
| 상관분석 | Pearson/Spearman/Kendall, Cramér V·signed Phi, 행렬 정리·삼각·거리 변환 |
| 지도 학습 | XGBoost, CatBoost, LightGBM, AdaBoost, MLP, Perceptron, Adaline, 단순·다중 선형회귀, Gaussian/Multinomial/Bernoulli/Complement Naive Bayes |
| 군집·거리 | K-means, DBSCAN, HDBSCAN, 자기 포함 k-distance 진단 |
| 차원 축소 | PCA, 지도 LDA, UMAP, t-SNE, SOM, metric/nonmetric MDS |
| 변수 선택 | Forward/Backward/CV Stepwise, L1/L2 계수 기반 선택 |
| 연관·연결망 | Apriori와 support/confidence/lift 등 규칙 지표, SNA 중심성·PageRank·커뮤니티 |
| 평균·다변량 검정 | 기존 Z/t, 고전/Welch 일원 ANOVA, 이원 ANOVA Type I·II·III, 일원 MANOVA, Tukey 및 Student/Welch 쌍별 t+Holm/Bonferroni |
| 통계 PDF 확장 | 분포16종, 모비율·대응t·단일분산χ², 순위·반복측정, OR/RR·RxC Fisher·CMH·Bowker·Cochran Q, AD·KS/Lilliefors·Hotelling·Box M·ADF |
| 추가 검정·진단 | 카이제곱 적합도, 독립성, 정확/근사 McNemar, 두 표본 분산비 F, Shapiro/D’Agostino/Jarque–Bera, Levene/Bartlett/Fligner |
| TensorFlow | Dense 분류·회귀, SimpleRNN/LSTM/GRU sequence-to-one, 시간 순 분할과 다중 스텝 시계열 예측 |

빈 비모수 onesample/twosample 파일은 구현에 포함하지 않습니다. ARIMA/SARIMA,
비선형 회귀 및 imbalanced-learn 실행 모듈은 현재 main에 포함되지 않았습니다.
SNA는 사회연결망 분석이며 신경망이 아닙니다. L2 자체는 희소화를 보장하지 않습니다.

## 빠른 실행

저장소 루트에서 실행하세요. Python 3.7/3.12별 의존성을 분리합니다.

```bash
# 일반 분석 환경 (Python 3.12)
python -m pip install -r requirements-models-py312.txt
python -m unittest discover -s tests -v
python -m src.stats.tests.nonParametric.independence

# 별도 TensorFlow 환경 (Python 3.12; NumPy 핀이 달라 위 환경과 혼합하지 않음)
python -m pip install -r requirements-tensorflow-py312.txt
python -m unittest discover -s tests -p 'test_tensorflow_modules.py' -v
python -m src.DeepLearning.TensorFlow.time_series
```

Python 3.7 일반 환경은 `requirements-models-py37.txt`를 사용합니다. HDBSCAN 빌드 전
`numpy==1.21.6 Cython==0.29.37 wheel==0.38.4`를 설치하고
`python -m pip install --no-build-isolation -r requirements-models-py37.txt`로 설치합니다.
TensorFlow 3.7 환경은 `requirements-tensorflow-py37.txt`로 별도 설치합니다.
`requirements.txt`는 기존 학습용 설정이며 새 표준 모듈 전체의 설치 명세가 아닙니다.

```python
from src.MachineLearning.Clustering.dbscan import run_dbscan
from src.stats.tests.nonParametric.independence import run_independence

result = run_independence([[30, 20], [10, 40]], overrides={"correction": False})
print(result.p_value, result.reject_h0)
print(result.to_json(indent=2))
# 군집: run_dbscan(X, overrides={"eps": 0.5, "min_samples": 5})
```

Profile 생략 시 `src/Recipes`의 모델별 JSON을 매번 읽습니다. 설정 우선순위는
JSON → overrides이며 명시적 Profile과 레시피 옵션을 동시에 지정하면 오류입니다.
모델 객체는 DTO 속성으로 접근하며 기본 사전/JSON에서는 제외합니다.

## 문서와 저장소 구조

| 경로·문서 | 내용 |
| --- | --- |
| `src/MachineLearning` | 지도 학습·군집·차원 축소·변수 선택 |
| `src/stats/tests` | 실제 통계 검정·가정 진단 코드 |
| `src/DeepLearning/TensorFlow` | 선택 의존성 TensorFlow 실행기 |
| `src/Association`, `src/Networks` | 연관규칙·사회연결망 |
| `src/Recipes` | 모델·목적별 JSON과 공통 loader |
| `tests` | 참조값·자료 분리·DTO·레시피·실제 학습 검증 |
| [전체 명세](docs/ANALYSIS_CATALOG.md) | 모든 구현 목록과 Profile/DTO 필드 |
| [모듈 컨벤션](docs/MODEL_MODULE_FORMAT.md) | 파일 메타데이터·7개 섹션·공통 실행 계약 |
| [레시피 사용법](docs/RECIPES.md) | 기본·사용자 JSON과 overrides |
| [ANOVA 계열](docs/ANOVA_MODULES.md) | 가정과 지원 범위 |
| [추가 통계 검정](docs/STATISTICAL_TESTS.md) | 검정 선택·귀무가설·근사 조건 |
| [TensorFlow 정리](docs/TENSORFLOW_MODULES.md) | 첨부 4개 자료의 핵심·입력 shape·학습/평가·윈도 |
| [통계 전체 구현](docs/STATISTICS_COMPLETE.md) | PDF 대응·전체 입력/출력·분포·정확/근사 경계 |
| [PDF 사전분석](docs/STATISTICS_PDF_AUDIT.md) | 구현 전 전수 조사·이관 전략 |
| [검증 기록](docs/VALIDATION.md) | 환경과 실제 실행 범위 |

## 실행·해석 범위

지도 학습과 변수 선택은 IID 분할이며 전처리는 학습 자료에 적합합니다.
군집·차원 축소는 전체 입력의 탐색 분석입니다. t-SNE/MDS와 현재 밀도 군집 실행기는
새 관측 predict/transform을 제공하지 않습니다. TensorFlow RNN 표본은 독립 시퀀스이며,
연속 시계열은 시간 순 윈도 실행기를 사용합니다. 상세 범위는 각 문서에 있습니다.
Python 3.7.4 시험 서버와 동일한 환경·패키지 제공을 보장하지 않습니다.

## 기존 로컬 학습 환경 설정

아래는 기존 Python 3.7 학습 환경 참고입니다. 시험 환경 정보는 현재 공식 안내를
별도로 확인하고, 새 분석 모듈은 위의 전용 의존성 파일을 사용하세요.

## Mac (Apple Silicon) 환경 설정: Python 3.7 활성화

> Apple Silicon(M1/M2/M3 등) Mac에서는 Python 3.7 버전을 공식 지원하지 않으므로, **Rosetta 2 (x86_64) 기반의 Miniconda**를 통해 설치를 진행해야 합니다.

```bash
# 1. x86_64 아키텍처로 터미널 실행 및 확인
arch -x86_64 zsh
uname -m  # 출력 결과가 x86_64인지 확인

# 2. x86_64용 Miniconda 설치 파일 다운로드 및 실행
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh

# 3. 환경 설정 반영 및 플랫폼 확인
source ~/.zshrc
conda info | grep -E "platform|subdir"

```

---

## 가상환경(Virtual Environment) 구축

> 충돌 방지를 위해 ADP 시험 환경과 동일한 **Python 3.7** 버전의 가상환경을 생성합니다.

```bash
# Conda 설치 경로 확인
type -a conda

# Python 3.7 가상환경 생성 및 활성화
conda create -n adp python=3.7
# 가상환경 Python 버전 확인
conda list -n adp python
conda activate adp

# 버전 및 환경 확인
python --version   # Python 3.7.x 확인
conda env list     # 가상환경 목록 확인

```

---

## 필수 패키지 설치

> 최신 `pip` 버전 사용 시 발생하는 패키지 의존성 충돌 및 설치 오류를 방지하기 위해 구버전을 기준으로 설치합니다.

```bash
# 1. pip 및 빌드 도구 버전 고정 설치
python -m pip install \
"pip<24" \
"setuptools==57.5.0" \
"wheel<0.38"

# 2. 형태소 분석기 및 필수 라이브러리 설치
conda install -c conda-forge mecab -y
conda install -c conda-forge swig -y

# 3. requirements.txt를 통한 추가 패키지 일괄 설치 및 jupyter에 가상환경 커널 추가
python -m pip install -r requirements.txt
python -m ipykernel install --user --name adp  \
--display-name "Python (adp)"  
jupyter kernelspec list # 등록 커널 확인
```
- 이후 환경 재시작하여 커널 지정하기 :
 > `command` + `shift` + `p` / `developer:reload window` 클릭하기

## 통계 전체 범위와 경로 이관

PDF의 신규 검정/효과크기23개·분포16종, 범주동질성 API와 기존평균 표준화를 제공합니다. 폴더는 검정계열·목적, 입력은 표본설계를 구분합니다. 기존 import와 python -m 경로는 유지합니다.
[통계 전체 명세](docs/STATISTICS_COMPLETE.md)를 참고하세요.

## UCI 모델 스켈레톤

UCI 스켈레톤 46개는 구현 분야별 폴더에 둡니다: MachineLearning 21개, DeepLearning 8개, Association 3개, Networks 1개, Preprocessing 7개, stats 상관/행렬 6개.
확률분포 예제는 `src/stats/skeleton/`, 공유 UCI 로더는 `src/datasets/`에 있습니다.
참고 XGBoost의 데이터 로딩 → 레시피 → FeatureSpec 전처리 → 모델 → 수치 평가 흐름을 유지합니다.
군집·차원 축소·변수 선택·TensorFlow·연관규칙·네트워크도 목적에 맞는 파이프라인을 제공합니다.
[참고 코드 분석·데이터 출처·실행 방법](docs/UCI_SKELETONS.md)을 확인하세요. 시각화는 포함하지 않습니다.

## Feature 전처리·상관분석 확장

수치/범주형 Feature의 순서가 있는 전처리, 별도 RoPE 임베딩 회전, 수치/범주 상관분석과
상관행렬 변환을 추가했습니다. [전체 명세](docs/PREPROCESSING_CORRELATION_MODULES.md)를 참고하세요.

## Axes 기반 EDA 시각화

`src/Visualization`의 35개 모듈은 단일 ax에 그려 subplot으로 조합합니다. 회귀 시각화는 제외하고
분포·결측·범주·군집·차원 축소·분류 결정·연관/연결망·시간 패턴을 포함합니다.
[전체 Profile/DTO·의사결정 근거·UCI 조합 예제](docs/EDA_VISUALIZATION_MODULES.md)를 참고하세요.

## 독립 평가 메트릭

분류·확률·Lift/Gain·회귀·군집·시계열의 개별 지표 50개, 분야별 묶음 4개, 모델 비교/CV 요약 2개와 JSON·평가 skeleton 56개를 추가했습니다. 기존 ML/DL·군집 skeleton과 연결합니다. [전체 입력·출력·수식·경계 명세](docs/EVALUATION_METRICS.md)
