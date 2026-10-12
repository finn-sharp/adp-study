# 검증 기록

## 2026-10-12: 통계 PDF 전체 모듈 복구·확장

Apple Silicon macOS의 Python 3.12.14에서 저장소의 일반 분석용 고정 버전을 사용했습니다.
XGBoost는 같은 3.0.5 버전의 macOS 패키지를 사용했습니다.

- 전체 unittest 72개 중 68개 통과, TensorFlow 실제 학습 4개는 별도 환경이 필요해 skip.
- 신규 통계 검사 14개: SciPy/statsmodels 참조값, Fisher RxC 미시상태 열거와 Monte Carlo,
  대응 결측·동률·0차이, 공분산 변환 불변성, 반복측정 ANOVA의 큰 공통 위치 이동,
  기존 import 및 결과 계약 검증.
- 분포 검사 7개: 16종의 확률·모수화·모멘트·난수, 꼬리와 구간의 수치 안정성,
  PDF 예제 32개에서 교정한 계산과 명시한 보충 가정 검증.
- 기법별 Profile·DTO·7개 섹션·독립 JSON 레시피 42개와 전체 레시피 87개 검증.
- 실행 예제 85개 통과. 이전 `python -m` 경로 9개를 포함하며 PDF 분포 예제 32개도 실행했습니다.
- Python 3.7 구문과 문서 링크를 검사하며, 복구 후 실제 3.7 실행과 별도 TensorFlow 학습은
  GitHub Actions의 기존 네 환경에서 확인합니다. 로컬 결과로 해당 실행을 대체하지 않습니다.

일반 모델 검사는 TensorFlow의 NumPy 핀과 분리했습니다.

## 2026-10-12: 추가 통계 검정·TensorFlow·README

일반 분석 환경과 TensorFlow 환경은 NumPy 핀 차이 때문에 따로 설치합니다.

| 환경 | 실행 범위 |
| --- | --- |
| Python 3.7.17 / requirements-models-py37.txt | unittest 51개 중 일반·계약 검증 47개 통과; TensorFlow 실제 학습 4개 skip. 일반 모듈 예제 33개 통과 |
| Python 3.12.14 / requirements-models-py312.txt | unittest 51개 중 일반·계약 검증 47개 통과; TensorFlow 실제 학습 4개는 의도적으로 skip. 일반 모듈 예제 33개 통과 |
| Python 3.12.14 / requirements-tensorflow-py312.txt | TensorFlow 테스트 6개와 실행 예제 4개 통과 |
| Python 3.7.17 / requirements-tensorflow-py37.txt | TensorFlow 테스트 6개와 실행 예제 4개 통과 |

검증 내용:

- 카이제곱 적합도와 독립성의 통계량·p 값·기대도수·자유도, Yates 보정 및 Cramér’s V.
- McNemar의 정확/근사·연속성 보정, 불일치쌍 0 처리, 잘못된 3×3 대응 표 거부.
- 분산비 F의 세 방향과 귀무 분산비, 표본을 바꿔도 동일한 양측 p 값.
- 정규성 3종과 등분산 3종의 SciPy 참조값, 상수 표본/부족한 표본/결측/잘못된 빈도 거부.
- 새 모듈의 7개 섹션, Profile/DTO 전체 필드 설명, 함수 Args/Returns/Raises, 레시피 실제 로딩.
- TensorFlow Dense 분류의 원 문자열 라벨 복원·확률 합, Dense 단일/다중 타깃 회귀, RNN 3종 셀 및 분류.
- 새 입력 예측 adapter, 학습 자료만 적합한 scaler, 학습/검증/test 분리, 최저 검증 손실의 가중치 복원.
- 시계열의 원 시점 분할, 입력/타깃 창 정렬, 다중 스텝·타깃 열 선택, 원 시점·원 단위 예측 대응.
- Python 3.7 구문과 문서 상대 링크, git diff 공백 검증.

테스트는 작은 합성 자료를 사용하며 모델의 우수한 예측 성능을 주장하지 않습니다.
TensorFlow 미설치 일반 환경에서는 모듈 import/Profile/입력 검증이 실행되고 실제 학습만 skip합니다.
GitHub Actions는 Python 3.7/3.12 일반 분석과 두 버전 TensorFlow의 네 작업을 따로 실행합니다.

Python 3.7.4 자체, 실제 ADP 시험 서버, GPU/Apple Silicon 네이티브 환경은 직접 검증하지 않았습니다.
