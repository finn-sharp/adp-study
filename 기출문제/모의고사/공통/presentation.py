"""수치 답안과 읽기용 보고를 분리합니다. 계산·모형 선택은 수행하지 않습니다."""

import pandas as pd
from IPython.display import display

TITLES = {
    "reg_design": "회귀 자료와 평가 설계",
    "reg_baseline": "기준 회귀모형의 적합 결과",
    "reg_diagnostics": "기준 모형의 가정과 영향도",
    "reg_selection": "AIC·BIC 변수 선택 비교",
    "reg_tuning": "학습 CV에서 처리 후보 선택",
    "reg_final": "같은 최종 평가 자료의 회귀 성능",
    "cls_design": "예측 사건과 학습·평가 분리",
    "cls_tuning": "표본 보완과 분류 설정 비교",
    "cls_logit": "설명용 로지스틱 회귀의 계수 해석",
    "cls_final": "최종 분류·확률 예측 성능",
    "cls_decision": "고정 임계값의 오분류 비용",
    "ts_design": "시간 순서와 예측 구간 설계",
    "ts_tuning": "과거 구간에서 차수 비교",
    "ts_diagnostics": "선택 시계열 모형의 잔차 진단",
    "ts_final": "미래 관측과 단순 기준 예측 비교",
}
READING = {
    "reg_design": "입력은 실제 면적·거리 등의 원 측정값이 아니라 고정 기준 대비 편차 지수입니다. id는 관측 식별자이고 target만 예측할 반응입니다.",
    "reg_baseline": "계수 검정은 학습 적합의 조건부 추론입니다. 학습 R²와 CV R²를 구분하고 최종 평가 자료는 아직 사용하지 않습니다.",
    "reg_diagnostics": "가정별 귀무가설을 따로 읽습니다. 기각하지 못해도 가정이 참임을 증명하지 않습니다. VIF는 변수 중복, Cook은 관측 영향입니다.",
    "reg_selection": "AIC/BIC는 같은 행·같은 반응 좌표계에서 비교합니다. 여기에 표시한 전체 학습 선택을 다음 CV의 각 학습 fold에서 다시 수행합니다.",
    "reg_tuning": "CV R²가 가장 큰 후보를 선택했습니다. 선택에 최종 평가 점수를 사용하지 않습니다.",
    "reg_final": "ΔR²는 선택 모형−기준 모형입니다. 값이 음수이면 최종 평가에서 개선되지 않았다는 뜻이며 그대로 보고합니다.",
    "cls_design": "target=1의 실제 사건 뜻과 양성 라벨을 먼저 확인합니다. 보완한 학습 발생률과 최종 평가의 자연 발생률을 혼동하지 않습니다.",
    "cls_tuning": "두 후보는 같은 CV fold로 비교합니다. 표본 보완은 fit 과정에서만 실행하고 검증·최종 평가 행은 변경하지 않습니다.",
    "cls_logit": "아래는 원 학습 발생률의 비벌점 설명 모형입니다. 샘플링·L2 규제를 사용한 예측모형과 계수가 같다고 해석하지 않습니다. 오즈비는 확률비가 아닙니다.",
    "cls_final": "ROC AUC는 순위, AP는 양성 검출의 precision/recall 요약, Brier는 확률의 제곱오차입니다. 임계값은 사전에 정한 0.5입니다.",
    "cls_decision": "비용 = 놓친 실제 사건(FN)×누락 비용 + 잘못 경보한 사건(FP)×오경보 비용입니다. 이 자료로 유리한 새 임계값을 고르지 않습니다.",
    "ts_design": "한 행은 하루입니다. 마지막 7일은 미래 평가이며, 차수 선택이나 정상성 진단에 포함하지 않습니다.",
    "ts_tuning": "각 검증 구간보다 앞선 날만 학습에 사용합니다. 선택 기준은 과거 rolling 평균 RMSE입니다.",
    "ts_diagnostics": "Ljung–Box는 지정 지연까지의 자기상관을 진단합니다. 비기각은 모든 형태의 독립성 증명이 아니며 AIC/BIC는 예측 오차가 아닙니다.",
    "ts_final": "naive는 직전 값 또는 직전 7일을 반복하는 단순 기준입니다. 같은 미래 날짜로 비교하며 MASE의 분모는 과거 학습 자료로만 계산합니다.",
}


def frame(value):
    """JSON 표 또는 행 목록을 읽기용 DataFrame으로 복원합니다."""
    if isinstance(value, dict) and {"data", "columns", "index"} <= set(value):
        return pd.DataFrame(
            value["data"], columns=value["columns"], index=value["index"]
        )
    if isinstance(value, dict):
        return pd.DataFrame([{"항목": k, "값": v} for k, v in value.items()])
    return pd.DataFrame(value)


def scalar_items(values):
    return {
        k: v
        for k, v in values.items()
        if v is None or isinstance(v, (str, bool, int, float))
    }


def tables(task, value):
    """문항별 핵심 수치·계수·비교표를 반환합니다. 전체 원 결과는 변경하지 않습니다."""
    v = value
    if task == "reg_design":
        return [
            (
                "자료·분할",
                frame(
                    {
                        "전체 행": v["n"],
                        "학습 행": len(v["train_indices"]),
                        "최종 평가 행": len(v["test_indices"]),
                        "분리 방식": v["split"],
                    }
                ),
            ),
            ("열별 결측 수", frame(v["missing"])),
        ]
    if task == "reg_baseline":
        return [
            ("학습 성능", frame(dict(v["train"], cv_r2=v["cv_r2"]))),
            ("계수·검정·구간", frame(v["coefficients"])),
            ("학습 적합·정보량", frame(v["fit_metrics"])),
        ]
    if task == "reg_diagnostics":
        items = []
        for name, dto in v["tests"].items():
            key = "tests" if "tests" in dto else "table"
            items.append((name, frame(dto[key])))
        influence = frame(v["influence"])
        cook = next(
            (
                c
                for c in influence.columns
                if "cook" in c and influence[c].dtype.kind in "fiu"
            ),
            None,
        )
        items.append(
            (
                "영향 관측 검토: Cook 값 상위 8행(전체는 JSON)",
                (
                    influence.sort_values(cook, ascending=False).head(8)
                    if cook
                    else influence.head(8)
                ),
            )
        )
        return items
    if task == "reg_selection":
        items = []
        for criterion in ["aic", "bic"]:
            dto = v[criterion]
            items.append((criterion.upper() + " 선택 요약", frame(scalar_items(dto))))
            for key in ["trace", "selection_trace", "history"]:
                if key in dto:
                    items.append((criterion.upper() + " 선택 경로", frame(dto[key])))
            if "selected_features" in dto:
                items.append(
                    (
                        criterion.upper() + " 선택 변수",
                        pd.DataFrame({"변수": dto["selected_features"]}),
                    )
                )
        return items
    if task == "reg_tuning":
        return [
            ("동일 fold의 CV 비교", frame(v["comparison"])),
            (
                "선택 후보",
                frame({"선택": v["selected_name"], "평가 자료 사용": "없음"}),
            ),
        ]
    if task == "reg_final":
        result = [
            (
                "같은 최종 평가 행의 성능",
                pd.DataFrame(
                    [
                        dict(모형="기준", **v["baseline_test"]),
                        dict(모형=v["selected_name"], **v["selected_test"]),
                    ]
                ),
            ),
            ("R² 변화", frame({"선택−기준": v["delta_r2"]})),
        ]
        for name, dto in v.get("details", {}).get("after_tests", {}).items():
            result.append(("선택 모형 " + name, frame(dto["tests"])))
        for key in ["fgls", "family_inference", "nonlinear_candidate"]:
            dto = v.get("details", {}).get(key)
            if dto:
                for field in ["coefficients", "fit_metrics", "diagnostics"]:
                    if field in dto:
                        result.append(
                            (
                                key + " " + field,
                                (
                                    frame(dto[field])
                                    if isinstance(dto[field], (dict, list))
                                    else str(dto[field])
                                ),
                            )
                        )
        return result
    if task == "cls_design":
        return [
            (
                "사건·분리",
                frame(
                    {
                        "양성 사건": v["event"],
                        "양성 라벨": v["positive_label"],
                        "학습 행": len(v["train_indices"]),
                        "최종 평가 행": len(v["test_indices"]),
                    }
                ),
            ),
            ("학습 클래스 수", frame(v["class_counts_train"])),
        ]
    if task == "cls_tuning":
        return [
            ("학습 CV 비교", frame(v["comparison"])),
            ("선택", frame({"탐색": v["method"], "선택 후보": v["selected"]})),
        ]
    if task == "cls_logit":
        dto = v["explanatory_model"]
        return [
            ("설명용 계수·오즈비", frame(dto["coefficients"])),
            ("추가 제곱항의 우도비 검정", frame(v["logit_specification"])),
        ]
    if task == "cls_final":
        confusion = pd.DataFrame(
            v["confusion_at_fixed_half"],
            index=["실제 0", "실제 1"],
            columns=["예측 0", "예측 1"],
        )
        return [
            (
                "최종 성능",
                frame(
                    {
                        k: v[k]
                        for k in [
                            "roc_auc",
                            "average_precision",
                            "brier",
                            "prevalence",
                            "threshold",
                        ]
                    }
                ),
            ),
            ("고정 0.5의 혼동행렬", confusion),
        ]
    if task == "cls_decision":
        return [
            (
                "사전 비용의 실제 결과",
                frame(
                    {
                        k: v[k]
                        for k in [
                            "false_negative_cost",
                            "false_positive_cost",
                            "holdout_cost_at_prespecified_half",
                        ]
                    }
                ),
            )
        ]
    if task == "ts_design":
        return [
            (
                "시간 분리",
                frame(
                    {
                        k: v[k]
                        for k in [
                            "n",
                            "frequency",
                            "period",
                            "train_end",
                            "test_start",
                            "horizon",
                        ]
                    }
                ),
            ),
            ("과거 자료의 ADF", frame(scalar_items(v["training_adf"]))),
        ]
    if task == "ts_tuning":
        return [
            ("rolling 후보 비교", frame(v["comparison"])),
            ("선택 차수", frame({"order": str(v["selected_order"])})),
        ]
    if task == "ts_diagnostics":
        return [
            ("학습 정보량", frame(v["train_information"])),
            ("잔차 진단", frame(scalar_items(v["diagnostics"]))),
        ]
    if task == "ts_final":
        scores = v["test_metrics"]
        return [
            (
                "동일 미래 성능",
                pd.DataFrame(
                    [
                        dict(모형=name, **scores[name])
                        for name in ["selected", "baseline"]
                    ]
                ),
            ),
            (
                "날짜 순서의 예측·관측·구간",
                pd.DataFrame(
                    {
                        k: v[k]
                        for k in [
                            "truth",
                            "predictions",
                            "baseline_predictions",
                            "interval_lower",
                            "interval_upper",
                        ]
                    }
                ),
            ),
            (
                "MASE",
                frame(
                    {
                        k: val
                        for k, val in scores.items()
                        if k not in ["selected", "baseline"]
                    }
                ),
            ),
        ]
    raise ValueError("표시 규칙이 없는 단계: " + task)


def present_result(task, value):
    """전체 행/확률 배열을 쏟지 않고 핵심 표와 해석 순서를 표시합니다."""
    print(TITLES[task])
    print(READING[task])
    if task in ("reg_final", "cls_final", "cls_decision", "ts_final"):
        print(conclusion(task, value))
    with pd.option_context(
        "display.max_rows",
        16,
        "display.max_columns",
        12,
        "display.float_format",
        lambda x: "%.6g" % x,
    ):
        for title, data in tables(task, value):
            print("\n" + title)
            display(data)
    print("\n전체 정밀 수치·원행 위치·경고: 실행결과/답안수치.json")


def conclusion(task, value):
    """표시할 실제 수치를 이용한 비교 결론. 선택이나 수치 결과를 변경하지 않습니다."""
    if task == "reg_final":
        before = value["baseline_test"]["r2"]
        after = value["selected_test"]["r2"]
        delta = value["delta_r2"]
        direction = (
            "높아졌습니다"
            if delta > 1e-10
            else ("낮아졌습니다" if delta < -1e-10 else "같습니다")
        )
        return (
            "이 자료에서 %s의 최종 R²는 %.4f로 기준 %.4f보다 %s(차이 %+.4f). 최종 평가를 보고 다른 후보로 바꾸지는 않습니다."
            % (value["selected_name"], after, before, direction, delta)
        )
    if task == "cls_final":
        return (
            "평가 자료의 사건 비율은 %.1f%%, ROC AUC는 %.4f, AP는 %.4f, Brier는 %.4f입니다. 높은 순위 성능만으로 확률 보정이나 운영 비용의 적절성을 증명하지 않습니다."
            % (
                100 * value["prevalence"],
                value["roc_auc"],
                value["average_precision"],
                value["brier"],
            )
        )
    if task == "cls_decision":
        return (
            "지정한 누락 비용 %s와 오경보 비용 %s를 적용한 최종 총비용은 %s입니다. 이 비용을 줄일 새 임계값은 별도 학습 검증에서 정해야 합니다."
            % (
                value["false_negative_cost"],
                value["false_positive_cost"],
                value["holdout_cost_at_prespecified_half"],
            )
        )
    if task == "ts_final":
        a = value["test_metrics"]["selected"]["rmse"]
        b = value["test_metrics"]["baseline"]["rmse"]
        return (
            "같은 미래 7일의 RMSE는 선택 모형 %.4f, 단순 기준 %.4f입니다. 이번 미래 구간에서는 %s의 오차가 더 작습니다. 이 비교와 학습 기준으로 표준화한 MASE를 구분합니다."
            % (a, b, "선택 모형" if a < b else "단순 기준")
        )
    return READING[task]
