# 구현 검증 기록

## 최신 상태: Output DTO까지 포함한 전체 실행 검증

2026-10-11 작업 복구와 DTO 직렬화 수정 후 Python 3.7.17과 Python 3.12.14에서 각각
**60개 unittest가 skip 없이 통과**했습니다. 두 환경에서 실행 예제 4개와 사용 노트북
3~12장 10개도 완료했습니다. 아래 기록은 단계별 이전 결과이며 최종 추가 검증을 끝에 기록합니다.

## 환경과 결과

| 검증 환경 | 라이브러리 | 결과 |
|---|---|---|
| Python 3.7.12 | sklearn 0.23.2, numpy 1.21.6, scipy 1.7.3, pandas 1.1.2, statsmodels 0.13.2, mlxtend 0.17.0 | 29개 unittest 통과 |
| Python 3.12 | sklearn 1.8.0, numpy 2.3.5, scipy 1.17.0, pandas 2.2.3, statsmodels 0.14.6, mlxtend 0.23.4 | 29개 unittest 통과 |
| 선택적 부스팅 (3.7) | XGBoost 0.90, LightGBM 2.2.3 | 회귀·분류 호출 포함 통과 |
| 선택적 부스팅 (3.12) | XGBoost CPU 3.0.5, LightGBM 4.6.0 | 회귀·분류 호출 포함 통과 |
| 통합 실행 예제 | `python -m examples.study_workflow` | 양쪽 환경 완료 |
| 새 사용 노트북 7개 | 3장~9장, 새 네임스페이스에서 셀 순서 실행 | 양쪽 환경 완료 |
| 문서·형식 | 로컬 Markdown 링크, Python 3.7 문법, `git diff --check` | 통과 |

Python 3.7.4 자체를 실행한 결과는 아니다. 동일한 기존 패키지 버전 조합을 Python 3.7.12에서
실제로 설치해 확인했다. Python 3.12는 개발 검증 환경이며 시험 환경을 대체하지 않는다.

## 검증 범위

- 지도 학습 모델별 회귀/분류, 동일 시드 분할, 예측/타깃 정렬.
- 학습 데이터 전용 StandardScaler, PCA, 문서 어휘 및 CV.
- SVD·폐형식 Ridge와 numpy/sklearn 기준 수치 비교.
- 단일/두 표본 Z/T 검정과 scipy 비교, Classic/Welch ANOVA와 statsmodels 비교.
- Type I/II/III 이원 ANOVA의 이름 기반 결과 추출, 교호작용 포함/제외.
- 비모수·정규성·등분산·범주형·상관·분산 검정, 퇴화 입력 거부.
- 회귀계수·진단·전진/후진/단계적 변수 선택, 반복측정 ANOVA.
- 군집 라벨·잡음 처리·거리·연결 행렬, PCA/요인분석, MDS/Isomap/LLE/t-SNE.
- 중복 거래 항목 처리와 규칙이 없는 연관분석.
- 시간 순서 ARIMA/SARIMAX/VARMAX 보류 예측, 반복 차분·ADF 진단.
- 텍스트 분류·LDA 토픽 분포, 기본 그래프·pairplot, rcParams 복원.

## 제한

- 업로드한 원본 노트북/PDF의 모든 예제 수치를 동일하게 재현하지는 않는다.
- 기존 EDA/전처리 노트북은 이번 실행 검증에 포함하지 않았다.
- NLTK 코퍼스/POS/WordNet, KoNLPy JVM 및 LIME 선택 기능은 기본 테스트에서 검증하지 않았다.
- 새 노트북은 코드 셀을 새 상태에서 순서대로 실행해 확인했다. Jupyter 브라우저 UI 검증은 아니다.
- 버전에서 발생하는 upstream 폐기 예정 경고를 전역에서 숨기지 않는다.
- GitHub Actions 결과는 실제 실행 완료 후 확인해야 하며, 위 표는 로컬 실행 기록이다.

## 2026-10-11 추가 확장 검증

Python 3.12, sklearn 1.8.0, imbalanced-learn 0.14.2, sklearn-compat 0.1.6,
scipy 1.17.0, statsmodels 0.14.6에서 전체 unittest **39개 통과**(기존 29개 + 확장 10개).
XGBoost/LightGBM을 포함하며 skip 없이 실행했다.

- 다중 회귀의 알려진 계수 복원, 다항 교호항 재현 및 학습 CV 차수 선택.
- 비선형 지수곡선의 알려진 계수 복원, 추정기 clone 및 잘못된 함수 출력 거부.
- 증강 5종/감소 5종 실제 fit_resample, 클래스별 수 보고, 범주형/혼합형 값 유지.
- CV 폴드별 스케일러 평균·sampler 선택 위치 검증, 예측 시 행 수 보존.
- SMOTEN 뒤 원핫 인코딩 Pipeline 실행.
- ARIMA/SARIMA 보류 구간 인덱스·구간 검증 및 보류 값 변경 시 예측 불변성.
- `python -m examples.extended_workflow` 실행 완료.
- 새 10장 노트북 코드 셀 순서 실행, 모든 src Python 3.7 구문 검사 완료.

이번 확장은 Python 3.7에서 실행 검증하지 않았다. 위쪽의 Python 3.7 29개 통과는
이전 구현의 기록이다. 기존 sklearn 0.23.2에는 imblearn 0.7이 호환되며 SMOTEN이 없다.
전체 10종은 requirements-review.txt의 별도 현대 환경에서 실행한다.
실제 GitHub Actions 실행과 PR 게시 여부는 이 로컬 검증과 별개다.

## Python 3.7 실제 실행 재검증 및 ANOVA/MANOVA/사후검정

| 환경 | 주요 패키지 | 결과 |
|---|---|---|
| Python 3.7.12 | numpy 1.21.6, scipy 1.7.3, pandas 1.3.5, sklearn 1.0.2, imblearn 0.8.1, statsmodels 0.13.2 | 51개 테스트 통과, skip 없음 |
| Python 3.12 | numpy 2.3.5, scipy 1.17.0, pandas 2.2.3, sklearn 1.8.0, imblearn 0.14.2, statsmodels 0.14.6 | 같은 51개 통과, skip 없음 |

- 기존 29개 + 회귀/시계열/리샘플링 10개 + 분산분석 12개 테스트.
- Python 3.7의 SMOTEN/SMOTENC를 포함한 증강 5종·감소 5종 실행 검증.
- Classic/Welch ANOVA 및 이원 Type III를 scipy/statsmodels와 수치 비교.
- 반복측정 ANOVA, MANOVA 요인/교호작용 검정표와 rank/공분산 특이 입력 거부.
- Tukey p·구간, Games–Howell 두 집단과 Welch t의 일치, Holm/Bonferroni p 및 구간 의미.
- 통합/회귀·시계열·샘플링/분산분석 실행 예제 3개를 두 환경에서 완료.
- 새 모듈 사용 노트북 3~11장 9개의 코드 셀을 두 환경에서 새 네임스페이스로 순서 실행.
- Python 3.7 전체 src 구문 검사, Markdown 로컬 링크 및 실제 원격 기준 패치 적용 검사.

`requirements-py37.txt`는 전체 모듈용 독립 가상환경이다. 기존 시험용 sklearn 0.23.2
고정 파일과 함께 설치하지 않는다. Python 3.7.4 자체나 모든 3.7 패치 버전, 원본 EDA/전처리
노트북, 선택 NLP/LIME 기능은 이번 검증에 포함하지 않았다. GitHub Actions는 실제 게시 후
결과를 확인해야 한다.

## 모든 분석 출력 DTO 검증

- 분석 API 55개가 명시적인 dataclass Output DTO를 반환하도록 선언. 반환 DTO 53종.
- 원래 51개 테스트와 DTO 계약/호환/직렬화 8개를 Python 3.7.12·3.12에서 실행: 총59개 통과.
- 기존 Result 별칭·모델별 전용 DTO·기술통계 필드/키 접근·없는 키 처리.
- 배열·날짜 인덱스·DataFrame/Series·희소 COO·집합·NumPy scalar·비유한 수치 JSON 변환.
- JSON 키 충돌·미지원 객체·runtime 포함 요청을 명시적으로 거부.
- ANOVA의 중첩 DTO, 거리/연결행렬 NumPy 호환, Pairplot 그림/메타데이터 유지.
- 실행 예제 4개와 새 사용 노트북 10개를 양쪽 환경에서 fresh namespace로 완료.
- 영어 NLTK 자원/POS/WordNet 및 LIME의 DTO 반환 선언은 검사했지만 실제 선택 엔진 실행은
  이번 기본 테스트에 포함하지 않았다. 코퍼스/JVM 같은 기존 선택 기능의 미검증 제한은 유지한다.
- JSON은 실행 객체나 원 pandas dtype을 복원하는 형식이 아니며 저장 규칙은 OUTPUT_DTOS.md에 명시.

## 2026-10-11 작업 복구 후 재검증

최신 원격 main 위에 이전 변경 패키지를 복구하여 기존 EDA/전처리 노트북과 참고 PDF를 보존했습니다.
적합 TransactionEncoder와 PolynomialFeatures가 기본 JSON에 포함되던 문제를 수정했습니다.
두 실행 객체는 DTO 필드 및 `to_dict(include_runtime=True)`에서 유지하며 기본 사전/JSON에서는 제외합니다.
실제 연관분석과 다항 회귀 결과의 직렬화를 확인하는 회귀 테스트를 추가했습니다.

| 환경 | 결과 |
|---|---|
| Python 3.7.17, requirements-py37.txt | 60개 unittest 통과, skip 없음; 예제 4개·새 노트북 10개 완료 |
| Python 3.12.14, requirements-review.txt + XGBoost CPU 3.0.5/LightGBM 4.6.0 | 같은 60개 통과, skip 없음; 예제 4개·새 노트북 10개 완료 |
| 정적 검증 | Python 3.7 전체 src 구문, Markdown 로컬 링크, git diff --check 통과 |

Python 3.7.4 자체의 실행 검증과 선택 NLP/LIME 엔진 검증은 포함하지 않았습니다.
GitHub Actions 결과는 로컬 실행 기록과 구분하여 실제 게시 후 확인합니다.
