"""
================================================================================
_deep_datasets.py - 새 딥러닝 스켈레톤용 UCI Optdigits 공식 train/test 로딩.
================================================================================
설명:
    - 새 알고리즘 파일의 공통 검증·학습·자료 로딩 계약을 제공합니다.
분석 가정:
    - 입력 정의역·학습 전용 전처리·거래 분모와 알고리즘별 한도를 유지합니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    별도 알고리즘의 공통 계약 추가
================================================================================
"""
__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
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


def load_optdigits(cache_dir=None):
    """8×8 픽셀 입력과 공식 train/test 분리 자료를 반환합니다.

    Args:
        cache_dir: 선택 캐시 경로; 기본 UCI_SKELETON_CACHE 또는 .uci-cache.
    Returns:
        tuple: X_train (3823,8,8,1), y_train, X_test (1797,8,8,1), y_test.
    Raises:
        ValueError/OSError: UCI 스키마·픽셀 범위·ZIP·네트워크 오류.
    """
    root=Path(cache_dir or os.environ.get('UCI_SKELETON_CACHE',Path(__file__).parents[3]/'.uci-cache'))
    root.mkdir(parents=True,exist_ok=True);path=root/'optdigits.zip'
    if not path.exists():
        with urlopen('https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip',timeout=60) as response:payload=response.read()
        with ZipFile(BytesIO(payload)) as archive:
            if archive.testzip() is not None:raise ValueError('UCI ZIP 오류')
        with tempfile.NamedTemporaryFile(dir=root,delete=False) as temporary:
            temporary.write(payload);temp_path=Path(temporary.name)
        temp_path.replace(path)
    result=[]
    with ZipFile(path) as archive:
        for name in ('optdigits.tra','optdigits.tes'):
            matches=[n for n in archive.namelist() if Path(n).name==name]
            if len(matches)!=1:raise ValueError('공식 파일 식별 오류: '+name)
            frame=pd.read_csv(BytesIO(archive.read(matches[0])),header=None)
            values=frame.to_numpy(dtype=float)
            if values.shape[1]!=65 or not np.isfinite(values).all() or (values!=np.floor(values)).any():raise ValueError('Optdigits의 64 정수 픽셀·클래스 스키마 오류')
            if (values[:,:64]<0).any() or (values[:,:64]>16).any() or not np.isin(values[:,64],np.arange(10)).all():raise ValueError('픽셀/클래스 범위 오류')
            result.extend([values[:,:64].astype(np.float32).reshape(-1,8,8,1),values[:,64].astype(int)])
    return tuple(result)
