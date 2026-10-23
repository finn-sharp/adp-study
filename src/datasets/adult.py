"""UCI Adult 공식 train/test를 Feature 기반 전처리·범주 분석에 제공합니다.

출처: Becker & Kohavi (1996), https://archive.ics.uci.edu/dataset/2/adult, CC BY 4.0.
"""

from io import BytesIO
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile
import numpy as np
import pandas as pd

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


def load_adult(cache_dir=None, max_rows=1000):
    """타깃 열을 분리할 수 있도록 공식 train/test DataFrame을 반환합니다.

    Args:
        cache_dir: 선택 캐시; 기본 UCI_SKELETON_CACHE 또는 저장소 .uci-cache.
        max_rows: 각 공식 파일의 첫 행 수 한도; None이면 전체.
    Returns:
        tuple: train DataFrame,test DataFrame; ?는 NaN, income 라벨은 통일.
    Raises:
        ValueError/OSError: 행 수·ZIP·스키마·네트워크 오류.
    """
    if max_rows is not None and (
        isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1
    ):
        raise ValueError("max_rows는 양수 정수/None입니다.")
    root = Path(
        cache_dir
        or os.environ.get(
            "UCI_SKELETON_CACHE", Path(__file__).parents[2] / ".uci-cache"
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / "adult.zip"
    if not path.exists():
        with urlopen(
            "https://archive.ics.uci.edu/static/public/2/adult.zip", timeout=60
        ) as response:
            payload = response.read()
        with ZipFile(BytesIO(payload)) as archive:
            if archive.testzip() is not None:
                raise ValueError("Adult ZIP 오류")
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as temporary:
            temporary.write(payload)
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
    frames = []
    with ZipFile(path) as archive:
        for filename in ("adult.data", "adult.test"):
            matches = [
                name for name in archive.namelist() if Path(name).name == filename
            ]
            if len(matches) != 1:
                raise ValueError("공식 Adult 파일 식별 오류")
            frame = pd.read_csv(
                BytesIO(archive.read(matches[0])),
                header=None,
                names=ADULT_COLUMNS,
                skipinitialspace=True,
                na_values=["?"],
                comment="|",
                nrows=max_rows,
            )
            if frame.empty or frame.shape[1] != 15:
                raise ValueError("Adult 스키마 오류")
            frame["income"] = frame.income.str.rstrip(".")
            if (
                frame.income.isna().any()
                or not frame.income.isin(["<=50K", ">50K"]).all()
            ):
                raise ValueError("Adult 타깃 라벨 오류")
            frames.append(frame)
    return tuple(frames)
