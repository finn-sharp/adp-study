# ADP 필기 및 실기 학습 가이드

첨부 학습 자료를 기존 아키텍처에 맞춰 `Profile → 실행 함수 → Result` 모듈로 재구성했습니다.
회귀·분류·군집화·통계·연관분석·시계열·텍스트·시각화를 같은 문서 명세로 사용합니다.

## 학습 모듈 및 문서

| 내용 | 위치 |
|---|---|
| 공통 구조·입력/평가 경계 | [아키텍처](docs/ARCHITECTURE.md) |
| 코드별 Profile/Result/함수 명세 | [명세 목록](docs/MODULE_SPECS.md) |
| 첨부 자료별 구현 대응·재구성 범위 | [참고 자료 대응표](docs/SOURCE_MAPPING.md) |
| 실행 예제 | [통합 예제](examples/study_workflow.py), `codebook/3장` 이후 노트북 |
| 하이퍼파라미터 설정 | [Recipes](src/Recipes) |
| 검증 | [테스트](tests/test_modules.py) |
| 실제 환경별 검증 결과·제한 | [검증 기록](docs/VALIDATION.md) |

### 전체 모듈 실행 환경

Python 3.7의 별도 가상환경에서 전체 분석 모듈과 테스트를 실행하려면:

```bash
python -m pip install -r requirements-py37.txt
python -m examples.study_workflow
python -m unittest discover -s tests -v
```

Python 3.12 개발 환경은 **별도의 가상환경**에서 `requirements-review.txt`를 설치합니다.
`requirements-ml.txt`는 기존 sklearn 0.23.2 기반의 기본 모듈용이며 전체 확장 테스트 환경이 아닙니다.
기존 `requirements.txt`는 전체 학습 환경용이며 새 모듈에 필요한 `typing_extensions`를 추가했습니다.
XGBoost/LightGBM, NLTK/KoNLPy, LIME는 사용 기능에 따라 추가 설치하는 선택적 의존성입니다.
시험 환경의 부스팅 패키지 버전은 기존 `requirements.txt`를 따릅니다.

### 모델 호출

```python
from sklearn.datasets import load_diabetes
from src.MachineLearning.Linear.ridge import RidgeProfile, run_ridge

X, y = load_diabetes(return_X_y=True)
profile = RidgeProfile(param_grid={"alpha": [0.1, 1, 10]}, cv=3)
result = run_ridge(X, y, profile)
print(result.test_metrics)
print(result.best_params)
# 새 입력에는 같은 특성 열 순서로 전달
predictions = result.model.predict(X[:3])
```

JSON 설정은 해당 모델 Profile 생성에 전달합니다.

```python
import json
with open("src/Recipes/ridge.json", encoding="utf-8") as file:
    profile = RidgeProfile(**json.load(file))
```

스케일러·PCA·문서 어휘는 학습 데이터에서만 적합하고 CV도 학습 데이터에서만 수행합니다.
군집화·차원 축소·회귀 추론은 탐색적 분석이며 시계열은 마지막 구간을 테스트로 보류합니다.
기존 EDA/전처리 노트북의 저장된 오류를 이번 변경에서 일괄 수정한 것은 아닙니다.

---

## 시험 환경 안내

> 아래 값은 기존 README의 시험 환경 기록입니다. 실제 응시 환경은 해당 회차의 공식 안내를 확인하세요.

* **방식**: 클라우드 기반 CBT (Computer Based Test), 크롬(Chrome) 브라우저 사용
* **서버 사양**: Ubuntu (64-bit) OS, 2 CPU, 4GB 메모리 (컨테이너별 제공)
* **분석 도구 및 버전**
* **R**: 버전 4.1.3, RStudio 제공
* **Python**: 버전 3.7.4, Jupyter Notebook 제공


---

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

### 회귀·시계열·불균형 데이터 확장

다중 선형/다항/비선형 회귀, ARIMA/SARIMA 전용 실행기와 imblearn 증강 5종·감소 5종을
추가했습니다. 각 Python 파일 안에 명세를 포함하며 수치형/범주형 지원은
[확장 명세](docs/EXTENSIONS.md)를 참고하세요. `python -m examples.extended_workflow`로 실행합니다.
전체 10종은 Python 3.7의 `requirements-py37.txt` 또는 Python 3.12의 `requirements-review.txt` 환경을 사용합니다.

### Python 3.7 전체 모듈 실행 및 ANOVA/MANOVA/사후검정

```bash
python3.7 -m venv .venv-py37
source .venv-py37/bin/activate
python -m pip install -r requirements-py37.txt
python -m unittest discover -s tests -v
python -m examples.multivariate_workflow
```

전체 모듈 환경은 기존 시험용 패키지 고정 파일과 별도입니다. Python 3.7.17과 3.12에서 전체
60개 테스트, 실행 예제 4개와 새 노트북 10개를 검증했습니다. 일원/Welch/이원/반복측정 ANOVA, 요인별 MANOVA,
Tukey HSD/Games–Howell/보정 쌍별 t 사후검정을 제공합니다.
각 `.py` 내부의 명세와 [분산분석 사용 안내](docs/MULTIVARIATE.md)를 참고하세요.

### 분석 Output DTO

모든 분석 API는 명시적인 dataclass Output DTO를 반환합니다. `result.field`로 결과를
재사용하고 `to_dict()`/`to_json()`으로 데이터 저장용 구조로 변환합니다. 모델·그림 객체는
필드에 유지하며 기본 JSON에서는 제외합니다. 기존 Result 이름과 사전 키 접근도 지원합니다.
[모듈별 반환 DTO·필드 계약](docs/OUTPUT_DTOS.md)과
`python -m examples.output_dto_workflow`를 참고하세요.
