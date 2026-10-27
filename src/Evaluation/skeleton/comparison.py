"""
================================================================================
comparison.py - UCI 자료·예측·독립 평가 skeleton
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
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-13  김재현(Finn)      v1.0.0    메트릭별 데이터·레시피·평가 흐름
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from src.Recipes import load_profile
from src.datasets.uci import load_tabular
from src.Evaluation.comparison import run_comparison


def load_data(cache_dir=None):
    """실행 계약.

    Args:
        cache_dir는 선택 UCI 캐시.
    Returns:
        이진 Iris DataFrame.
    Raises:
        OSError/ValueError.
    """
    return load_tabular("classification", binary=True, cache_dir=cache_dir)


def load_recipe(overrides=None):
    """실행 계약.

    Args:
        overrides는 설정 재정의.
    Returns:
        ComparisonProfile.
    Raises:
        ValueError/OSError.
    """
    return load_profile("metric-comparison", overrides=overrides)


def build_pipeline(profile=None, overrides=None):
    """실행 계약.

    Args:
        profile 또는 overrides.
    Returns:
        동일 보류 비교 함수.
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
            비교 DTO.
        Raises:
            ValueError.
        """
        if p.evaluation_task != "classification":
            raise ValueError("이 예제는 분류 비교입니다.")
        X = data.drop(columns="target")
        y = LabelEncoder().fit_transform(data.target)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=1234
        )
        models = {
            "logistic": Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", LogisticRegression(max_iter=500)),
                ]
            ),
            "dummy": DummyClassifier(strategy="prior"),
        }
        for model in models.values():
            model.fit(Xtr, ytr)
        return run_comparison(
            yte,
            {k: m.predict(Xte) for k, m in models.items()},
            scores={k: m.predict_proba(Xte) for k, m in models.items()},
            classes={k: m.classes_ for k, m in models.items()},
            profile=p,
        )

    return execute


def main(cache_dir=None, overrides=None, data=None):
    """실행 계약.

    Args:
        cache_dir/overrides와 선택 data.
    Returns:
        비교 DTO.
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
