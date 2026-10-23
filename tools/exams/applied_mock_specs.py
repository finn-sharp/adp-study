"""응용 종합 모의10회: 독립 시나리오·사전 분석 계획·문항별 채점 관점."""

CASES = [
    (
        "공동주택 가격과 계약 심사",
        "power",
        ["area", "building_age", "station_distance", "rooms"],
        "가격(천만원)",
        "연체 발생",
        "random_over",
        1,
        "grid",
        False,
        "면적·연식·거리·방 수가 가격에 주는 관계를 설명한다. 입력 간 중복 정보와 결측 가능성을 점검한다. 변환이 설명력을 높인다는 전제 없이 검증한다.",
    ),
    (
        "물류 배송비와 파손 위험",
        "lognormal",
        ["weight", "distance", "volume", "handling"],
        "배송비(천원)",
        "파손 발생",
        "smote",
        1,
        "optuna",
        False,
        "배송비는 양수이며 큰 비용에서 변동 폭이 크다. 원 비용과 변환 반응의 설명력을 혼동하지 말고 원 비용으로 역변환한 예측을 비교한다.",
    ),
    (
        "발전 출력과 장비 경보",
        "quadratic",
        ["irradiance", "temperature", "voltage", "cloud"],
        "출력(kW)",
        "고장 경보",
        "random_under",
        7,
        "grid",
        False,
        "출력과 주요 입력의 관계에 곡률이 의심된다. 다항식의 학습 설명력 증가를 그대로 일반화 개선으로 간주하지 않는다.",
    ),
    (
        "공정 수율과 불량 검출",
        "heteroscedastic",
        ["load", "pressure", "temperature", "speed"],
        "수율 점수",
        "불량 발생",
        "smote",
        1,
        "grid",
        False,
        "부하가 큰 배치에서 수율 오차 변동 폭이 커질 수 있다. 알려진 분산은 없으며 학습 잔차에서 추정하는 분산 모형은 불확실성을 가진다.",
    ),
    (
        "중고차 가격과 반품 위험",
        "influence",
        ["mileage", "age", "engine", "maintenance"],
        "거래 가격 점수",
        "반품 발생",
        "random_over",
        7,
        "optuna",
        False,
        "극단적인 거래가 영향력이 큰 관측인지 점검한다. 모든 극단값을 오류로 가정하지 않는다. 사전 허용한 제거 후보는 학습 행에서만 비교하며 제거 이력을 남긴다.",
    ),
    (
        "에너지 사용량과 요금 미납",
        "ordered",
        ["outdoor_temp", "occupancy", "insulation", "equipment"],
        "전력 사용량 점수",
        "미납 발생",
        "random_under",
        7,
        "grid",
        True,
        "회귀 행은 날짜순 등간격이며 가까운 날의 오차가 관련될 수 있다. 무작위 분할을 사용하지 않고 공분산 보정과 평균 예측 개선을 구분한다.",
    ),
    (
        "촉매 반응량과 공정 실패",
        "exponential",
        ["concentration", "temperature", "stirring", "impurity"],
        "반응량(mg)",
        "실험 실패",
        "smote",
        1,
        "grid",
        False,
        "주요 농도에 대한 평균 반응이 지수 형태일 수 있다. 사전에 정한 양수 계수 경계에서 평균 함수와 초기값의 영향을 검토하고 선형 기준과 비교한다.",
    ),
    (
        "보험 지급액과 청구 위험",
        "gamma",
        ["severity", "delay", "exposure", "complexity"],
        "지급액(백만원)",
        "청구 이상",
        "random_over",
        7,
        "optuna",
        False,
        "지급액은 0보다 큰 연속량이며 평균이 커질수록 분산이 커질 수 있다. 적합한 분포·평균 링크·분산 구조를 고려하고 정규 오차를 일괄 강요하지 않는다.",
    ),
    (
        "고객센터 문의 횟수와 이탈",
        "poisson",
        ["users", "release_change", "usage", "complexity"],
        "일별 문의 횟수",
        "고객 이탈",
        "random_under",
        1,
        "grid",
        False,
        "같은 하루 노출의 문의 횟수는 0을 포함하는 비음수 정수다. 평균-분산 관계와 과산포를 검토하고 연속 반응 모형의 한계를 설명한다.",
    ),
    (
        "서비스 성장량과 전환 예측",
        "growth",
        ["promotion", "price", "visits", "engagement"],
        "누적 가입량 점수",
        "유료 전환",
        "smote",
        7,
        "grid",
        False,
        "연속 가입량의 평균은 포화될 수 있다. 연속 반응의 S자 성장곡선과 이진 전환 확률을 예측하는 모형이 같은 분석이 아님을 구분한다.",
    ),
]
REG_TASKS = [
    "reg_design",
    "reg_baseline",
    "reg_diagnostics",
    "reg_selection",
    "reg_tuning",
    "reg_final",
]
CLS_TASKS = ["cls_design", "cls_tuning", "cls_logit", "cls_final", "cls_decision"]
TS_TASKS = ["ts_design", "ts_tuning", "ts_diagnostics", "ts_final"]
SPECIAL = {
    "power": (
        "특성의 Yeo-Johnson 변환 후보",
        "변환은 학습 폴드에서만 적합한다. 이미 같은 열공간인 OLS를 표준화하는 것만으로 R²가 증가하지 않는다. 변환 필요성은 CV와 진단을 함께 해석한다.",
    ),
    "lognormal": (
        "양수 목표의 log 변환과 학습 잔차 smearing 후보",
        "log(y)의 잔차 진단과 원 y 단위 예측 성능은 서로 다른 좌표계다. 단순 exp(예측)은 원 반응 조건부 평균이 아니며 학습 잔차 보정도 오차 사양에 의존한다.",
    ),
    "quadratic": (
        "차수1·2의 다항식 확장 후보",
        "곱항·제곱항 생성 뒤 공선성을 다시 점검한다. 학습 R²의 단조 증가가 CV/보류 개선을 보장하지 않으며 선택 후 t/F 해석은 조건부다.",
    ),
    "heteroscedastic": (
        "학습 분산 추정의 가중 회귀1·2회 후보",
        "가중치는 참 분산을 알고 지정한 값이 아니다. 추정 역분산·가중 잔차·원단위 R²를 구분하고 조건부 표준오차가 분산 추정 불확실성을 반영하지 않음을 밝힌다.",
    ),
    "influence": (
        "명시적 학습 Cook 제거 후보",
        "기본 영향도는 검토 flag다. 제거 후보만 학습에서4/n·최대20% 가드를 적용하며 유지 원행에서 대치·변환·변수선택을 다시 적합한다. 최종 평가 행은 제거하지 않는다.",
    ),
    "ordered": (
        "학습 시계열 분할 및 HAC 공분산 후보",
        "HAC는 오차 자기상관에 대한 공분산 보정이며 평균 예측이나 R²를 바꾸는 전처리가 아니다. 자기상관이 남으면 후속 시계열/동적 평균 사양을 검토하고 모든 가정이 해결됐다고 주장하지 않는다.",
    ),
    "exponential": (
        "단일 농도의 지수 평균함수 후보",
        "비선형 최소제곱은 초기값·계수 경계·식별성에 의존한다. 계수 공분산은 국소 근사이며 OLS의 계수 t/F나 VIF를 파라미터에 그대로 적용하지 않는다.",
    ),
    "gamma": (
        "양수 연속 반응의 Gamma log-mean 후보",
        "log 평균 링크와 Var(Y|X)=phi mu²를 확인한다. Pearson/deviance 잔차·분산 사양을 사용하며 반응 또는 잔차 정규성·보통 R²를 분포 조건으로 강요하지 않는다. 예측 비교는 원단위 R²다.",
    ),
    "poisson": (
        "정수 횟수 반응의 Poisson log-mean 후보",
        "동일 노출의 횟수와 평균=분산 조건을 점검한다. 과산포·과도한0을 검토하며 서로 다른 노출에는 알려진 log exposure가 필요하다. pseudo R²와 원 단위 보류 R²는 별개다.",
    ),
    "growth": (
        "연속 반응의 포화 S자 평균함수 후보",
        "성장곡선의 L,k,x0는 연속 평균함수 파라미터다. 이진 타깃의 로지스틱 분류 계수와 구분하며 국소 공분산·초기값·경계·정의역·외삽 한계를 보고한다.",
    ),
}


def specification(number):
    title, kind, features, unit, event, sampler, period, method, ordered, scenario = (
        CASES[number - 1]
    )
    seed = 20270000 + number
    special, interpretation = SPECIAL[kind]
    plan = dict(
        number=number,
        seed=seed,
        regression_kind=kind,
        features=features,
        unit=unit,
        classification_event=event,
        sampler=sampler,
        classification_search=method,
        ordered_regression=ordered,
        period=period,
        horizon=7,
        false_negative_cost=3 + number,
    )
    reg_context = (
        scenario
        + " regression.csv의240행은 "
        + ("시간순 일별 관측" if ordered else "서로 독립인 개체/배치")
        + "을 가정한 고정 합성 자료다. 목표 target의 단위는 "
        + unit
        + "이며 id는 식별자다. 입력은 "
        + ", ".join(features)
        + "다. 입력은 고정 업무 기준 대비 편차 지수이므로 음수가 가능하며 표본 전체에서 미리 추정한 표준화 값이 아니다. "
        + (
            "앞70%/뒤30%의 시간순 분할과 학습 TimeSeriesSplit3-fold를 사용한다."
            if ordered
            else "시드"
            + str(seed)
            + "로 학습70%/최종평가30% 분할하고 학습 shuffled3-fold를 사용한다."
        )
        + " 후보 선택 기준은 동일 원반응 단위 CV R² 최대다. 기본 대치·표준화·선형 모형, VIF5, AIC/BIC 후진 선택과 "
        + special
        + "를 사전 지정한다. 모든 학습 통계·변수선택·행 처리는 폴드 안에서 적합한다. 유의수준은5%이며 반복 진단은 탐색적 개별검정으로 보고한다. 최종평가로 후보나 가정을 골라서는 안 된다."
    )
    reg_texts = [
        "관측 단위·목표 척도·예측 시점과 입력의 역할을 정리하고 결측·분포·공선성 후보를 탐색하시오. 지정 분할의 원행 위치를 남기고 전체 자료 대치나 식별자 입력이 왜 부적절한지 설명하시오.",
        "기본 전처리와 선형 기준 모형을 학습에 적합하고 학습/CV R², 계수 t 검정·전체 F 검정·AIC·BIC를 제시하시오. 계수·설명력·일반화 성능의 의미와 기본 추론 가정을 구분하시오. 최종평가는 아직 사용하지 마시오.",
        "기준 모형의 선형 사양·잔차 정규성·등분산성과 관측 설계에 맞는 독립성/자기상관을 진단하시오. 가설·통계량·p값·가정·판정과 잔차/분위수/영향도 그림을 보고하시오. VIF와 Cook 영향도의 대상을 구분하고 이상 관측을 자동으로 오류라고 단정하지 마시오.",
        "동일 학습 행·동일 원반응에서 AIC와 BIC로 후진 변수선택을 수행하고 제거 경로·남은 특성을 비교하시오. VIF 제거와의 차이, 서로 다른 y변환·관측 행의 정보량을 직접 비교하는 오류, 변수선택 후 추론의 한계를 설명하시오.",
        "사전 지정 전처리·변수선택·대안 모형을 학습 CV 내부에서 튜닝하시오. 후보별 원단위 CV R²·선택 파라미터·실패 여부와 분할을 보고하고 하나를 선택하시오. 특히 "
        + special
        + "의 적용 조건과 처리 단위를 설명하시오.",
        "선택 이후에만 기준/선택 모형을 동일 최종평가 행으로 평가하여 R²·RMSE·MAE와 R² 차이를 보고하시오. 진단 전후 또는 대안 분포의 적절한 잔차·분산·영향도 근거를 해석하고, 개선이 없거나 가정 위반이 남아도 그대로 보고하시오. "
        + interpretation,
    ]
    cls_context = (
        "classification.csv는 별도 독립360관측의 합성 자료이며 target=1은 "
        + event
        + "이다. c1~c4는 사건 발생 전의 수치 위험 신호, id는 식별자다. 층화 학습70%/최종평가30% 분할(시드"
        + str(seed)
        + "), 학습3-fold ROC AUC를 사용한다. 무보완과 "
        + sampler
        + "를 비교하되 학습 폴드 안의 대치→표준화→VIF5→샘플링→L2 로지스틱 예측모형을 사용한다. C 후보는0.1·1·10이고 "
        + (
            "Optuna6 trials, 시드고정"
            if method == "optuna"
            else "GridSearchCV 전수비교"
        )
        + "로 튜닝한다. 운영 임계값0.5는 사전 고정이며 false-negative 비용"
        + str(plan["false_negative_cost"])
        + "·false-positive 비용1이다. 별도로 원 학습 분포의 비벌점 이진 GLM을 설명용 적합하되 예측모형과 같은 추정량이라 주장하지 않는다."
    )
    cls_texts = [
        "목표 사건·양성 라벨·클래스 불균형과 층화 분할을 정리하시오. 데이터 분리 전에 샘플을 보완하면 발생하는 누출과 최종평가의 자연 발생률을 유지해야 하는 이유를 설명하시오.",
        "지정 탐색과 샘플링 모듈을 연결하고 동일 학습 폴드에서 무보완/보완 후보를 비교하시오. 파라미터·CV ROC AUC·선택 후보·폴드 위치를 제시하고 검증/최종평가에는 샘플링을 수행하지 않았음을 확인하시오.",
        "원 학습 분포의 설명용 이진 모형에서 계수·오즈비·구간·z 검정·우도 비교·AIC/BIC를 보고하시오. 첫 선택 특성의 제곱항 추가에 대한 log-odds 사양 검토를 수행하고 가설·자유도·p값·한계를 제시하시오. OLS 잔차 정규성·보통 R²·t/F를 일괄 적용하지 마시오.",
        "CV 선택 뒤 최종평가의 ROC 곡선·ROC AUC·precision-recall·평균정밀도·확률 보정 그림과 Brier 점수, 고정0.5 혼동행렬을 제시하시오. 불균형에서 정확도 하나만으로 판단하는 한계를 설명하시오.",
        "사전 지정 비용과 임계값으로 최종평가 오분류 비용을 계산하고 운영 의사결정을 제안하시오. AUC·확률 보정·분류 임계값의 차이, 샘플링에 따른 발생률 변화와 별도 학습 검증 필요성을 설명하시오. 최종평가를 보고 새 임계값을 최적화하지 마시오.",
    ]
    ts_context = (
        "timeseries.csv는 "
        + title.split("과")[0]
        + "의 별도140일 합성 수요/사용 기록이다. date는 중복 없는 등간격 일자, target은 별도의 연속 수요 지표이며 회귀/분류의 target과 대상·단위가 다르다. 공변량 예측은 사용하지 않는다. 마지막7일은 최종 미래 평가로 봉인한다. 학습의 마지막14일을 두 구간으로 나눠 각각 직전까지 확장 학습하고7일씩 예측한다. 차수 후보(0,0,0)·(1,0,0)를 rolling 평균 RMSE로 선택한다. "
        + (
            "계절 주기는7이며 각 후보에 계절(1,0,0,7)을 고정한다. 기준은 직전7일 반복이다."
            if period == 7
            else "비계절 ARIMA를 사용하며 기준은 직전 값 반복이다."
        )
        + " 예측구간 신뢰수준95%이고 최적화 최대100회다. 회귀/분류 자료와는 다른 과제다."
    )
    ts_texts = [
        "시간 간격·순서·목표·예측 horizon을 확인하고 과거/최종 미래를 나누시오. 학습의 정상성 진단 가설·통계량·p값과 추세·계절성의 한계를 설명하시오. 무작위 CV가 부적절한 이유를 서술하시오.",
        "지정 expanding rolling 검증을 구현하여 후보별 RMSE·MAE·수렴/실패, 각 과거/검증 구간을 제시하고 차수를 선택하시오. 최종 미래값을 차수·전처리 선택에 사용하지 마시오.",
        "선택 모형의 학습 AIC·BIC와 초기화 구간을 제외한 잔차 자기상관 진단을 보고하시오. 추정한 AR/MA 자유도를 반영하고 가설·통계량·p값·판정·해석을 제시하시오. 차분이나 우도 정의가 다른 모형의 정보량 단순 비교를 경계하시오.",
        "선택 모형과 지정 naive의 동일 미래7일 RMSE·MAE·R²·MASE를 비교하고 예측·관측·조건부 예측구간을 시각화하시오. MASE의 학습 분모, 음의 보류 R² 가능성, 차수 선택 불확실성을 구간이 모두 반영하지 않는 한계를 설명하시오.",
    ]
    sections = []
    answers_reg = [
        "원행 위치를 먼저 봉인하고 결측 대치·스케일·선택은 Pipeline.fit 안에 둔다. id는 의미 없는 식별자이며 test는 최종 보고만 한다.",
        "R²와 수정 R²는 적합 설명력, CV R²는 학습 밖 예측 평가다. t는 개별 계수, F는 지정 전체 기울기 가설이다. 고전적 추론은 올바른 사양·독립·등분산·정규 오차에 조건부이며 y 자체 정규성을 요구하지 않는다.",
        "정규성은 적합 잔차의 JB, 사양은 RESET, 등분산은 Koenker/BP를 사용한다. IID의 DW는 기술값만, 명시 시간순은 BG를 사용한다. 가설을 각각 분리하고 탐색적 개별 p값을 해석한다. VIF는 설계 열, Cook은 관측의 추정 영향이며 비기각은 가정 증명이 아니다.",
        "AIC=-2ll+2k, BIC=-2ll+log(n)k의 감소를 따라 탐욕적 후진 선택한다. 같은 행/반응의 비교다. 최적 부분집합 보장이나 선택 후 추론 보정은 없으며 CV 폴드에서 다시 선택해야 한다.",
        "원단위 CV R² 최대를 선택하며 후보별 clone은 각 학습 폴드에서 모든 처리를 새로 적합한다. 기준 대비 진단과 점수를 함께 보고 test로 모델을 바꾸지 않는다. "
        + interpretation,
        "같은 보류 원행에서 기준/선택 점수를 계산하고 ΔR²=선택-기준을 그대로 보고한다. 개선이나 모든 가정 만족을 보장하지 않는다. "
        + interpretation,
    ]
    answers_cls = [
        "층화 분할은 사건 비율을 보존한다. 샘플링은 imblearn Pipeline의 CV fit 내부이며 검증·test는 원 발생률 그대로다. 합성 표본은 원행을 갖는 실제 관측으로 보고하지 않는다.",
        "동일 폴드 ROC AUC 최대 후보를 선택한다. 대치·스케일·VIF·샘플링을 폴드 안에 둔다. Optuna는 시드·trial 예산·후보 공간을 고정하며 Grid는 지정 후보를 전수 비교한다.",
        "logit는 성공확률이 아니라 log odds의 선형 사양이다. 자연 발생률의 비벌점 GLM z/우도 추론과 표본보완·L2 예측모형은 별개다. 추가 제곱항의 우도비는 df1 점근 비교이며 비기각이 전체 사양의 증명은 아니다. 오즈비는 확률비·인과효과가 아니다.",
        "같은 최종행과 양성1 확률로 ROC AUC·AP·Brier를 계산한다. AP는 기준 발생률과 함께, 보정 그림은 bin 표본수와 함께 해석한다. 샘플링이 후험확률 보정을 보장하지 않는다.",
        "비용=FN×지정비용+FP×1이다. 임계값0.5를 사전 고정했으므로 test에서 유리한 값으로 바꾸지 않는다. 운영 임계값·확률 재보정은 별도 학습 검증/OOF에서 사전 결정해야 한다.",
    ]
    answers_ts = [
        "과거에서만 정상성(단위근 귀무) 진단을 수행한다. ADF 비기각은 단위근 확정이 아니며 추세·계절 사양과 작은 표본의 한계를 함께 확인한다.",
        "각 rolling train은 validation보다 앞서고 모든 validation은 최종test보다 앞선다. 후보 적합 실패·미수렴은 기록하고 적격 후보의 평균 RMSE로 선택한다.",
        "Ljung-Box는 지정 지연까지 자기상관 없음이 귀무이며 AR/MA p+q+P+Q를 자유도에서 차감한다. 초기화 구간을 제외한다. 비기각은 모든 독립성의 증명이 아니며 AIC/BIC는 학습 적합 기준이다.",
        "동일 미래 horizon에서 선택/naive를 비교한다. MASE 분모는 train의 주기별 차이 절대평균이며0이면 정의하지 않는다. 예측구간은 선택 모형에 조건부이며 차수 선택·모형 오사양 불확실성을 전부 포함하지 않는다.",
    ]
    for prefix, title0, data, context, texts, tasks, answers, points in [
        (
            "1",
            "회귀 설계와 조건별 개선",
            "regression.csv",
            reg_context,
            reg_texts,
            REG_TASKS,
            answers_reg,
            6,
        ),
        (
            "2",
            "이진 사건 예측과 샘플링",
            "classification.csv",
            cls_context,
            cls_texts,
            CLS_TASKS,
            answers_cls,
            8,
        ),
        (
            "3",
            "미래 수요와 시간순 검증",
            "timeseries.csv",
            ts_context,
            ts_texts,
            TS_TASKS,
            answers_ts,
            6,
        ),
    ]:
        sections.append(
            dict(
                title=prefix + ". " + title0,
                data=data,
                context=context,
                questions=[
                    dict(
                        id=prefix + "-" + str(i),
                        text=t,
                        task=task,
                        answer=a,
                        points=points,
                    )
                    for i, (t, task, a) in enumerate(zip(texts, tasks, answers), 1)
                ],
            )
        )
    return dict(
        exam_number=number + 5,
        number=number,
        title=title,
        kind="original_applied_mock",
        questions=15,
        points=100,
        plan=plan,
        sections=sections,
        scope="창작 종합 모의; 공식 기출 원문/정답 아님",
    )
