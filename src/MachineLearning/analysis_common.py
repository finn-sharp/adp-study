"""
================================================================================
analysis_common.py - 탐색 분석 입력 검증과 군집 요약
================================================================================
설명:
    - 탐색 분석 입력 검증과 군집 요약을 Profile·전용 Output DTO·공개 실행기로 제공합니다.
    - 실행 설정은 src/Recipes의 모델별 JSON에서 별도로 관리합니다.

분석 가정:
    - 검증은 통계적 가정 충족을 보장하지 않습니다.
    - 결측값과 범주형 인코딩은 실행 전에 처리합니다.

작성 정보:
    - 작성자: 김재현(Finn)
    - 이메일: penguin.klg@gmail.com
    - 작성일: 2026-10-11
    - 라이선스: Proprietary

변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    ----------------------------------------------------------------------------
    2026-10-11  김재현(Finn)     v1.0.0    모델별 레시피·전체 명세·실행 예제 작성
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"

from typing import Any, List, Optional
import numpy as np
from sklearn.metrics import silhouette_score
from src.MachineLearning.common import make_scaler


def positive_int(value: Any, name: str, minimum: int = 1) -> None:
    """정수 범위를 검증합니다.

    Args:
        value: 설정값. bool 제외.
        name: 설정 이름.
        minimum: 포함 하한.
    Returns:
        None: 유효하면 반환.
    Raises:
        ValueError: 정수 또는 하한 위반.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(name + "는 " + str(minimum) + " 이상 정수여야 합니다.")


def finite_number(value: Any, name: str, positive: bool = False) -> None:
    """유한한 수치 범위를 검증합니다.

    Args:
        value: int/float 설정. bool 제외.
        name: 설정 이름.
        positive: True이면 0 거부.
    Returns:
        None: 유효하면 반환.
    Raises:
        ValueError: 자료형·유한성·하한 위반.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
    ):
        raise ValueError(name + "는 유한한 수여야 합니다.")
    if value < 0 or (positive and value == 0):
        raise ValueError(name + "의 하한을 확인하세요.")


def validate_scaling(scaling: str, random_state: Optional[int] = None) -> None:
    """스케일링 이름과 선택적 난수 시드를 검증합니다.

    Args:
        scaling: none/standard/minmax.
        random_state: 0 이상 정수 또는 검증 생략 None.
    Returns:
        None: 유효하면 반환.
    Raises:
        ValueError: 스케일링 또는 시드 오류.
    """
    make_scaler(scaling)
    if random_state is not None:
        positive_int(random_state, "random_state", 0)


def feature_names(X: Any) -> List[str]:
    """입력 열 이름을 보존하고 중복 이름을 거부합니다.

    Args:
        X: 검증된 2차원 특성 또는 DataFrame.
    Returns:
        List[str]: 원 열 순서의 이름.
    Raises:
        ValueError: 문자열 변환 후 열 이름 중복.
    """
    names = (
        [str(c) for c in X.columns]
        if hasattr(X, "columns")
        else ["feature_" + str(i) for i in range(np.asarray(X).shape[1])]
    )
    if len(set(names)) != len(names):
        raise ValueError("입력 열 이름은 중복될 수 없습니다.")
    return names


def cluster_summary(X: np.ndarray, labels: np.ndarray) -> dict:
    """잡음을 제외한 군집 수와 silhouette을 계산합니다.

    Args:
        X: 거리 계산에 사용한 전처리 좌표.
        labels: 원 행 순서의 군집 라벨. -1은 잡음.
    Returns:
        dict: n_clusters, noise_mask, silhouette. 지표 미정의 시 None.
    Raises:
        ValueError: 좌표 또는 라벨 길이 오류.
    """
    mask = labels == -1
    clean = labels[~mask]
    count = len(np.unique(clean))
    score = None
    if 1 < count < len(clean):
        score = float(silhouette_score(X[~mask], clean))
    return dict(n_clusters=count, noise_mask=mask, silhouette=score)
