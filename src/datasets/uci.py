"""
================================================================================
uci.py - UCI 스켈레톤 공식 자료 로딩·정제
================================================================================
설명:
    - 데이터 로딩 → JSON/Profile → 전처리/모델 연결 → 학습/분석 → 수치 평가.
분석 가정:
    - 보류 자료는 전처리 적합·변수 선택·모델 선택에 사용하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin-klg@naver.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary (UCI 자료의 라이선스는 별도 출처 명시)
변경 이력 (Revision History):
    DATE        AUTHOR           VERSION   DESCRIPTION
    ----------------------------------------------------------------------------
    2026-10-12  김재현(Finn)       v1.0.0    UCI 스켈레톤의 공통 계약과 수치 자료 변환
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin-klg@naver.com"
__version__ = "1.0.0"
__status__ = "Development"

from io import BytesIO
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile
import numpy as np
import pandas as pd

DATASETS = {
    "wine": ("186/wine+quality.zip", "winequality-red.csv"),
    "iris": ("53/iris.zip", "iris.data"),
    "bike": ("275/bike+sharing+dataset.zip", "day.csv"),
    "libras": ("181/libras+movement.zip", "movement_libras.data"),
    "retail": ("352/online+retail.zip", "Online Retail.xlsx"),
}


def dataset_bytes(name, cache_dir=None):
    """공식 ZIP의 지정 파일만 읽으며 import 시 다운로드하지 않습니다.

    Args:
        name: DATASETS의 키.
        cache_dir: 캐시 위치; 기본 UCI_SKELETON_CACHE 또는 저장소 .uci-cache.
    Returns:
        bytes: UCI 원본 파일의 내용.
    Raises:
        OSError/ValueError: 네트워크·캐시·ZIP 파일 오류.
    """
    relative, member = DATASETS[name]
    root = Path(
        cache_dir
        or os.environ.get(
            "UCI_SKELETON_CACHE", Path(__file__).parents[2] / ".uci-cache"
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / (name + ".zip")
    if not path.exists():
        with urlopen(
            "https://archive.ics.uci.edu/static/public/" + relative, timeout=60
        ) as response:
            payload = response.read()
        with ZipFile(BytesIO(payload)) as archive:
            if archive.testzip() is not None:
                raise ValueError("UCI ZIP 검증 실패")
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as temporary:
            temporary.write(payload)
            temp_path = Path(temporary.name)
        temp_path.replace(path)
    with ZipFile(path) as archive:
        matches = [n for n in archive.namelist() if Path(n).name == member]
        if len(matches) != 1:
            raise ValueError("UCI 원본 파일을 하나로 식별할 수 없습니다: " + member)
        return archive.read(matches[0])


def load_tabular(task="regression", binary=False, cache_dir=None):
    """Wine Quality 회귀 또는 Iris 분류 자료를 로딩합니다.

    Args:
        task: regression/classification.
        binary: Iris에서 두 종류만 선택; Adaline 전용.
        cache_dir: 선택 캐시 위치.
    Returns:
        pd.DataFrame: 수치 특성 및 마지막 target 열.
    Raises:
        ValueError/OSError: 목적·스키마·다운로드 오류.
    """
    if task == "regression":
        frame = pd.read_csv(BytesIO(dataset_bytes("wine", cache_dir)), sep=";").rename(
            columns={"quality": "target"}
        )
    elif task == "classification":
        names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]
        frame = pd.read_csv(
            BytesIO(dataset_bytes("iris", cache_dir)), header=None, names=names
        ).dropna(how="all")
        if binary:
            frame = frame[frame.target.isin(["Iris-setosa", "Iris-versicolor"])]
    else:
        raise ValueError("task는 classification/regression입니다.")
    if frame.empty or frame.isna().any().any() or "target" not in frame:
        raise ValueError("UCI 자료 스키마/결측 오류")
    if not np.isfinite(frame.drop(columns="target").to_numpy(dtype=float)).all():
        raise ValueError("유한한 수치 특성이 필요합니다.")
    return frame.reset_index(drop=True)


def load_bike(cache_dir=None):
    """누락 없는 일별 cnt 시계열을 원 날짜 순으로 반환합니다.

    Args:
        cache_dir: 선택 캐시 위치.
    Returns:
        pd.DataFrame: DatetimeIndex와 cnt 열; casual/registered 누수 특성 제외.
    Raises:
        ValueError/OSError: 누락 날짜·중복·결측 또는 다운로드 오류.
    """
    frame = pd.read_csv(
        BytesIO(dataset_bytes("bike", cache_dir)), parse_dates=["dteday"]
    )
    frame = frame.set_index("dteday").sort_index()[["cnt"]]
    expected = pd.date_range(frame.index.min(), frame.index.max(), freq="D")
    if (
        frame.index.has_duplicates
        or not frame.index.equals(expected)
        or frame.isna().any().any()
    ):
        raise ValueError("일별 시계열의 날짜 간격/결측 오류")
    return frame


def load_libras(cache_dir=None):
    """독립 손동작 45프레임·2좌표 시퀀스와 원 분류 라벨을 반환합니다.

    Args:
        cache_dir: 선택 캐시 위치.
    Returns:
        tuple: X (표본,45,2), y (표본,).
    Raises:
        ValueError/OSError: 스키마·결측 또는 다운로드 오류.
    """
    frame = pd.read_csv(BytesIO(dataset_bytes("libras", cache_dir)), header=None)
    if frame.shape[1] != 91 or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("LIBRAS는 90 좌표 특성과 1 클래스 열이 필요합니다.")
    return frame.iloc[:, :90].to_numpy(dtype=float).reshape(-1, 45, 2), frame.iloc[
        :, 90
    ].to_numpy(dtype=int)


def load_baskets(cache_dir=None, max_transactions=1000, max_items=30):
    """Retail의 완료된 영수증별 고유 상품 집합을 만듭니다.

    Args:
        cache_dir: 선택 캐시 위치.
        max_transactions: 원 시간 순 완전한 영수증 수 한도.
        max_items: 선택한 영수증에서 거래 빈도가 높은 상품 수 한도.
    Returns:
        list: 취소/음수 수량/가격/우편요금 제외 후 2상품 이상 영수증 집합.
    Raises:
        ValueError/OSError/ImportError: 한도·유효 자료·다운로드·Excel 엔진 오류.
    """
    if min(max_transactions, max_items) < 2:
        raise ValueError("영수증·상품 한도는 각각 2 이상이어야 합니다.")
    frame = pd.read_excel(
        BytesIO(dataset_bytes("retail", cache_dir)),
        engine="openpyxl",
        dtype={"InvoiceNo": str, "StockCode": str},
    )
    frame = frame.dropna(
        subset=["InvoiceNo", "StockCode", "Quantity", "UnitPrice", "InvoiceDate"]
    )
    frame = frame[
        (~frame.InvoiceNo.str.upper().str.startswith("C"))
        & (frame.Quantity > 0)
        & (frame.UnitPrice > 0)
    ]
    frame = frame[frame.StockCode.str.match(r"^\d{5}[A-Z]?$", na=False)].sort_values(
        "InvoiceDate", kind="stable"
    )
    # Select complete invoice groups, rather than cutting raw rows in the middle of an invoice.
    invoices = frame.InvoiceNo.drop_duplicates().iloc[:max_transactions]
    frame = frame[frame.InvoiceNo.isin(invoices)]
    pairs = frame[["InvoiceNo", "StockCode"]].drop_duplicates()
    counts = (
        pairs.StockCode.value_counts().rename_axis("item").reset_index(name="count")
    )
    selected = counts.sort_values(["count", "item"], ascending=[False, True]).item.iloc[
        :max_items
    ]
    baskets = [
        set(g.StockCode)
        for _, g in pairs[pairs.StockCode.isin(selected)].groupby(
            "InvoiceNo", sort=False
        )
    ]
    baskets = [items for items in baskets if len(items) >= 2]
    if len(baskets) < 2:
        raise ValueError("유효한 다상품 영수증이 부족합니다.")
    return baskets
