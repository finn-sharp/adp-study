"""
================================================================================
_common.py - UCI 평가 skeleton: 자료 → 학습 구간 전처리/모델 → 보류 예측 → 독립 메트릭.
================================================================================
설명:
    - UCI 평가 skeleton: 자료 → 학습 구간 전처리/모델 → 보류 예측 → 독립 메트릭.
분석 가정:
    - 자료의 열·타깃·분할 조건을 확인하고, 학습 전처리는 학습 구간에서만 적합합니다. 실행은 명시한 진입점에서 수행합니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-27
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from src.datasets.uci import load_tabular, load_bike


def load_data(group, cache_dir=None):
    """분야별 공식 UCI 자료를 명시 호출 때만 읽습니다.

    Args:
        group: 평가 분야.
        cache_dir: UCI 캐시 경로 또는 None.
    Returns:
        DataFrame: Iris/Wine Quality/Bike 원 자료.
    Raises:
        ValueError/OSError: 자료·다운로드 오류.
    """
    if group == "time_series":
        return load_bike(cache_dir)
    return load_tabular(
        "regression" if group == "regression" else "classification",
        binary=group == "classification",
        cache_dir=cache_dir,
    )


def prepare(group, data):
    """동일 보류 자료를 개별 메트릭·묶음 평가에 재사용합니다.

    Args:
        group: 평가 분야.
        data: 원 DataFrame; time_series는 시간 순 수치 한 열.
    Returns:
        dict: 공개 메트릭의 정답·예측·점수·좌표·학습 시계열 입력.
    Raises:
        ValueError: 자료·분할·클래스 조건 오류.
    """
    if group == "time_series":
        a = data.to_numpy(dtype=float).reshape(-1)
        cutoff = int(len(a) * 0.8)
        if cutoff < 3 or len(a) - cutoff < 3:
            raise ValueError("시간 순 예제는 최소 충분한 학습/보류 구간 필요")
        starts = np.arange(cutoff, len(a) - 1)
        actual = np.array([a[i : i + 2] for i in starts])
        predicted = np.repeat(a[starts - 1, None], 2, axis=1)
        return dict(y_true=actual, y_pred=predicted, y_train=a[:cutoff])
    X = data.drop(columns="target")
    y = data.target.to_numpy()
    if group == "clustering":
        transformed = StandardScaler().fit_transform(X)
        pred = KMeans(n_clusters=3, n_init=10, random_state=1234).fit_predict(
            transformed
        )
        return dict(y_true=y, y_pred=pred, X=transformed)
    train, test = train_test_split(
        np.arange(len(data)),
        test_size=0.3,
        random_state=1234,
        stratify=y if group == "classification" else None,
    )
    if group == "classification":
        encoder = LabelEncoder().fit(y[train])
        encoded = encoder.transform(y)
        if len(encoder.classes_) != 2:
            raise ValueError("순위 메트릭 skeleton은 이진 자료를 사용합니다.")
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(random_state=1234, max_iter=500)),
            ]
        )
        model.fit(X.iloc[train], encoded[train])
        return dict(
            y_true=encoded[test],
            y_pred=model.predict(X.iloc[test]),
            y_score=model.predict_proba(X.iloc[test])[:, 1],
        )
    model = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
    model.fit(X.iloc[train], y[train])
    return dict(y_true=y[test], y_pred=model.predict(X.iloc[test]))
