"""
================================================================================
cv_summary.py - UCI 자료·예측·독립 평가 skeleton
================================================================================
설명:
    - 원 자료 → JSON/Profile → 학습 구간 전처리·모델 → 보류 예측 → 독립 메트릭.
분석 가정:
    - 보류 자료는 학습 적합에 사용하지 않습니다. 군집은 전체 자료 탐색입니다.
    - 시간 순 예제는 직전 관측 naive 예측이며 import 시 다운로드·학습하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-13
    - 라이선스: Proprietary (UCI 자료의 라이선스는 별도)
변경 이력 (Revision History):
    DATE        AUTHOR           VERSION   DESCRIPTION
    ----------------------------------------------------------------------------
    2026-10-13  김재현(Finn)      v1.0.0    메트릭별 데이터·레시피·평가 흐름
================================================================================
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from src.Recipes import load_profile
from src.datasets.uci import load_tabular
from src.Evaluation.classification.f1 import run_f1
from src.Evaluation.cv_summary import run_cv_summary


def load_data(cache_dir=None):
    """실행 계약.

    Args:
        선택 UCI 캐시.
    Returns:
        Iris DataFrame.
    Raises:
        OSError/ValueError.
    """
    return load_tabular("classification", cache_dir=cache_dir)


def load_recipe(overrides=None):
    """실행 계약.

    Args:
        설정 재정의.
    Returns:
        CVSummaryProfile.
    Raises:
        ValueError/OSError.
    """
    return load_profile("metric-cv-summary", overrides=overrides)


def build_pipeline(profile=None, overrides=None):
    """실행 계약.

    Args:
        profile 또는 overrides.
    Returns:
        폴드별 평가 함수.
    Raises:
        ValueError.
    """
    if profile is not None and overrides is not None:
        raise ValueError("Profile/overrides 동시 지정 불가")
    p = load_recipe(overrides) if profile is None else profile

    def execute(data):
        """실행 계약.

        Args:
            원 Iris 표.
        Returns:
            CV 요약 DTO.
        Raises:
            ValueError.
        """
        X = data.drop(columns="target")
        y = LabelEncoder().fit_transform(data.target)
        scores = []
        for train, test in StratifiedKFold(3, shuffle=True, random_state=1234).split(
            X, y
        ):
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", LogisticRegression(max_iter=500)),
                ]
            )
            model.fit(X.iloc[train], y[train])
            scores.append(
                float(
                    run_f1(
                        y[test],
                        model.predict(X.iloc[test]),
                        overrides={"average": "macro"},
                    ).value
                )
            )
        return run_cv_summary({"f1_macro": scores}, profile=p)

    return execute


def main(cache_dir=None, overrides=None, data=None):
    """실행 계약.

    Args:
        cache_dir/overrides와 선택 data.
    Returns:
        CV 요약 DTO.
    Raises:
        OSError/ValueError.
    """
    result = build_pipeline(overrides=overrides)(
        load_data(cache_dir) if data is None else data
    )
    print(result.to_json(indent=2))
    return result


if __name__ == "__main__":
    main()
