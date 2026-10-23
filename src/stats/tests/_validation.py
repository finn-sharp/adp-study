"""
================================================================================
_validation.py - 통계 검정 공통 입력 검증
================================================================================
설명:
    - Profile → 모델별 JSON 레시피 → 실행기 → Output DTO 계약을 제공합니다.
분석 가정:
    - 입력 자료의 독립성·정규성 등 설계 가정은 호출자가 보장.
작성 정보:
    - 작성자: 김재현(Finn)
    - 이메일: penguin.klg@gmail.com
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    입력 검증·전용 DTO·레시피·예제 추가
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"
import numpy as np


def alpha(value):
    """

    Args:
        value: 검증할 설정/자료.
    Returns:
        None: 유효하면 반환.
    Raises:
        ValueError/TypeError: 입력 shape·타입·조건 위반.
    """

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or not 0 < value < 1
    ):
        raise ValueError("alpha는 0과 1 사이 유한한 수여야 합니다.")


def vector(value, minimum=2):
    """

    Args:
        value: 검증할 설정/자료.
        minimum: 최소 관측 수.
    Returns:
        np.ndarray: 유한한 수치 표본.
    Raises:
        ValueError/TypeError: 입력 shape·타입·조건 위반.
    """
    a = np.asarray(value, dtype=float)
    if a.ndim != 1 or len(a) < minimum or not np.isfinite(a).all():
        raise ValueError("유한한 1차원 표본과 충분한 관측이 필요합니다.")
    return a


def counts(value, table=False):
    """

    Args:
        value: 검증할 설정/자료.
        table: 2차원 도수표 여부.
    Returns:
        np.ndarray: 유효한 빈도 배열.
    Raises:
        ValueError/TypeError: 입력 shape·타입·조건 위반.
    """
    a = np.asarray(value, dtype=float)
    if (
        a.ndim != (2 if table else 1)
        or a.size == 0
        or not np.isfinite(a).all()
        or (a < 0).any()
        or (a != np.floor(a)).any()
    ):
        raise ValueError("빈도가 아닌 확률/음수/결측은 사용할 수 없습니다.")
    return a
