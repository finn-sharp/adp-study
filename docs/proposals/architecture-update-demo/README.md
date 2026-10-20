> 2026-10-20 검토 승인 후 정식 구현은 [도메인 명세](../../DOMAIN_ARCHITECTURE.md)와 src로 이관했습니다. 아래 내용은 승인 당시 데모 기록입니다.

# 전체 업데이트 검토 데모

상태: 사용자 검토 전. 이 폴더의 후보만 작성했으며 src, 기존 문제지·풀이, CI는 변경하지 않았습니다.

## 먼저 검토할 내용

대표 후보는 `kaplan_meier.py`입니다. 수식과 계산은 모듈 내부에 두고 사용자는 입력 자료와 레시피만 전달합니다. 다음 기준을 이후 모든 신규·수정 모듈에도 적용합니다.

1. 기존 작성 정보·변경 이력·메타데이터를 유지합니다.
2. Imports / Input Profile / Output DTO / Input Validation / Individual Model Runners / Public Runner / Example Execution의 7개 섹션을 유지합니다.
3. 각 Profile·DTO의 모든 필드에 타입·기본값 또는 반환 조건·배열 대응 관계를 명시합니다. 함수에는 Args·Returns·Raises와 분석 가정을 기록합니다.
4. 설정은 기법별 JSON으로 분리하고 기존 `load_profile`·`resolve_profile`을 사용합니다. JSON → overrides 우선순위와 명시적 Profile 혼합 금지 규칙을 유지합니다.
5. 전용 결과 dataclass는 기존 `OutputDTO`를 상속합니다. 적합 객체와 수치 결과의 직렬화를 구분합니다.
6. 추정·예측·확률 계산과 가설검정을 구분합니다. 추정 모듈에는 불필요한 p-value나 H0/H1을 붙이지 않습니다.

## 데모 사용 모습

정식 이관 후 사용자가 작성하는 코드는 다음 정도입니다.

```python
from src.stats.survival.kaplan_meier import run_kaplan_meier

result = run_kaplan_meier(
    durations=[1, 2, 2, 3, 4],
    event_observed=[1, 1, 0, 0, 1],
    overrides={"label": "공급사 A", "evaluation_times": [0, 1, 2, 3, 4]},
)
print(result.survival_at_times)       # [1.0, 0.8, 0.6, 0.6, 0.0]
print(result.median_survival_time)    # 4.0
print(result.to_json(indent=2))
```

아직 위의 정식 import 경로는 생성하지 않았습니다. 검토 후보는 아래 명령으로 실행합니다.

```bash
python docs/proposals/architecture-update-demo/run_demo.py
```

검토 실행기는 기존 레시피 로더에 후보를 **현재 프로세스에서만 임시 등록**하고 별도 JSON을 전달합니다. 종료 시 등록을 복원하며 운영 파일을 수정하지 않습니다. 검토용 별도 로더를 만들지 않았습니다.

수계산: t=2에서 사건 1명과 절단 1명은 모두 직전 위험집합 4명에 포함됩니다. 따라서 생존확률은 (4/5)×(3/4)=0.6입니다. `demo_result.json`은 이 실행의 실제 DTO 출력, `validation.json`은 검증 결과입니다.

## 전체 업데이트 범위 제안

| 작업 | 이관·신설 위치 | 완료 기준 |
| --- | --- | --- |
| 생존분석 공통화 | `src/stats/survival/`, 대응 JSON·skeleton | KM과 신뢰구간, RMST, 로그순위·쌍별 보정, Cox HR·CI·생존예측·PH 진단을 추정/검정 계약에 맞게 분리 |
| 시계열 공통화 | `src/TimeSeries/`, 대응 JSON·skeleton | ARIMA/SARIMA, 수렴·예측구간·시간 인덱스·차분 조건 명시, 시간순 검증과 외생변수 미래 가용성 확인 |
| 비선형 회귀 공통화 | `src/MachineLearning/NonlinearModels/` | 모형 종류·파라미터·초깃값·경계·실패 조건 명세, 추정 공분산의 해석 조건, 보류 평가 분리 |
| 불균형 처리 공통화 | `src/Preprocessing/resampling/` | 오버/언더 기법별 수치·범주·혼합 지원표, 기법별 JSON, CV 학습 폴드 안에서만 적합, 합성 샘플의 원행 대응 불가 명시 |
| 시험 풀이 연결 | 회차 공통 실행기와 해설 생성 도구 | 재사용 가능한 계산은 src를 호출하고, 문항별 데이터 결합·정답 문구·보고는 시험 코드에서 처리 |
| 전체 통합 목록 | 기출문제 통합 명세와 docs | 30묶음의 회차·유형·문항·데이터·src 경로·출처 수준·검증 환경을 한 목록으로 관리 |
| 문서 수치 정합성 | README·RECIPES·카탈로그 | 등록부·JSON·통합 명세에서 집계하여 수작업 숫자 불일치 검사 |
| 환경·회귀 검증 | requirements·tests·Actions | 핵심 모듈 Python 3.7/3.12 실제 실행, 선택 의존성 분리, 시험 확장은 검증된 환경만 표시 |

불균형 처리에서 요청된 오버/언더 각 5종의 구체적인 기법 선정은 자료형·설치 버전의 지원 여부를 먼저 조사해 명세합니다. 숫자를 맞추기 위해 범주형에 수치 전용 기법을 적용하지 않습니다. 이외에도 기존 기출에서 사용하는 계산을 문항-모듈 대응표로 먼저 전수 점검하고, 누락을 별도 목록으로 남깁니다.

## 이관 설계

| 후보 파일 | 검토 후 정식 위치·연결 |
| --- | --- |
| `kaplan_meier.py` | `src/stats/survival/kaplan_meier.py` |
| `survival-kaplan-meier.json` | `src/Recipes/survival-kaplan-meier.json` |
| 등록 키 | `survival-kaplan-meier` → 모듈·KaplanMeierProfile·통계 task=None |
| 사용 예제 | `src/stats/skeleton/kaplan_meier.py` |
| 시험 연결 | 생존분석 해설 생성기가 src를 호출하고 업무 맥락의 설명은 해설에 유지 |
| 검증 | 손계산·공식 구현 대조·경계값·DTO/레시피 계약, 시험 노트북 재실행 |

데모는 기본 KM 추정의 계약을 검토하기 위한 최소 후보입니다. 신뢰구간·RMST·검정·Cox는 아직 구현하지 않았습니다. 표의 전체 범위가 완료되었다는 의미가 아닙니다.

## 변경 후 검증 원칙

- 이관 전 답안과 이관 후 답안을 문항별로 비교합니다. 수치·결론이 바뀌면 원인과 수정된 가정을 기록합니다.
- 검정은 통계량·p값·가설·유의수준·기각 여부의 대응을 확인하고, 확률·추정 문제는 별도 규칙으로 검사합니다.
- 시간·그룹·IID 분할을 구분하고 전처리·변수 선택·표본 보완이 평가 자료를 보지 않는지 검사합니다.
- 데이터·문항·PDF·실행 결과의 해시와 명세 대응을 다시 생성·검증합니다.
- Python 3.7 문법 검사와 Python 3.7 실제 실행을 구분해 기록합니다. 지원할 수 없는 의존성은 별도 환경으로 명시합니다.

KM 입력·사건표·중앙수명 비교의 기술 참고: https://lifelines.readthedocs.io/en/stable/fitters/univariate/KaplanMeierFitter.html
미도달 중앙수명은 참고 구현의 inf 대신 기존 저장 규약에 맞춰 DTO에서 None, JSON에서 null로 명시합니다.

## 검토 요청

검토할 핵심은 **이 정도의 모듈 명세와 입력·출력 구조**, **시험별 계산을 공통 모듈로 연결하는 방식**, **위 전체 업데이트 범위**입니다. 검토가 끝나면 정식 경로 등록·전체 이관·문서 갱신·검증·PR을 진행합니다.
