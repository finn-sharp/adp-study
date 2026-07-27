# ADP 필기 및 실기 학습 가이드

---

## 시험 환경 안내

> K-Data 공식 안내 기준 시험 환경입니다.

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
pip install -r requirements.txt
python -m ipykernel install --user --name adp  \
--display-name "Python (adp)"  
jupyter kernelspec list # 등록 커널 확인
```
- 이후 환경 재시작하여 커널 지정하기 :
 > `command` + `shift` + `p` / `developer:reload window` 클릭하기