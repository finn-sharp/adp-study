"""
================================================================================
hypergeometric.py - 초기하 분포 확률·분위수·모멘트
================================================================================
설명:
    - 기법별 Profile·Output DTO·공개 실행기와 독립 JSON 레시피를 제공합니다.
분석 가정:
    - 이산 PMF/연속 PDF와 누적·꼬리·구간 확률을 구분합니다. 모수 추론 검정과 별도입니다.
작성 정보:
    - 작성자: 김재현(Finn)
    - 이메일: penguin.klg@gmail.com
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)     v1.0.0    통계 PDF 전체 범위 모듈화·입력 검증·명세
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"


# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from src.output import OutputDTO
from src.Recipes import resolve_profile
from scipy import stats
from src.stats.tests._common import integer, number
from ._common import evaluate_distribution


# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------


@dataclass
class HypergeometricProfile:
    """HypergeometricProfile 전체 필드 계약.

    Attributes:
        N (int): 모집단 개체 수 ≥1 기본 50.
        K (int): 성공 개체 수 0≤K≤N 기본 10.
        n (int): 비복원 추출 개체 수 0≤n≤N 기본 5.
        random_state (int): 난수 시드, 0 이상 기본 1234.
    """

    N: int = 50
    K: int = 10
    n: int = 5
    random_state: int = 1234


# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------


@dataclass
class HypergeometricOutputDTO(OutputDTO):
    """HypergeometricOutputDTO 전체 필드 계약.

    Attributes:
        distribution_name (str): 분포 이름.
        distribution_kind (str): discrete/continuous.
        parameters (Dict[str, Any]): 실제 적용 Profile 전체.
        evaluation_points (Optional[np.ndarray]): 조회한 x 배열; 미요청 None.
        pmf (Optional[np.ndarray]): 이산 점확률; 연속/미요청 None.
        pdf (Optional[np.ndarray]): 연속 밀도; 이산/미요청 None.
        cdf (Optional[np.ndarray]): P(X≤x); 미요청 None.
        survival (Optional[np.ndarray]): P(X>x); ≥는 cutoff를 한 칸 조정. 미요청 None.
        interval (Optional[Tuple[float, float]]): 조회 구간 [low,high], 중심 구간은 모수 신뢰구간 아님.
        interval_probability (Optional[float]): 양끝 포함 구간 확률. 이산은 ceil(low)~floor(high).
        quantile_probabilities (Optional[np.ndarray]): 조회 q∈[0,1].
        quantiles (Optional[np.ndarray]): CDF 역함수 PPF.
        inverse_survival (Optional[np.ndarray]): SF 역함수 ISF.
        mean (Optional[float]): 존재하고 유한한 기댓값; 미정의 None.
        variance (Optional[float]): 존재하고 유한한 분산; 미정의 None.
        moment_defined (Dict[str, bool]): 평균/분산의 유한 존재 여부.
        samples (Optional[np.ndarray]): 선택 난수 표본; 미요청 None.
        distribution (Any): SciPy frozen distribution. 기본 사전/JSON에서 제외.
        warnings (List[str]): 정의역/모멘트/예제 해석 경고.
    """

    distribution_name: str
    distribution_kind: str
    parameters: Dict[str, Any]
    evaluation_points: Optional[np.ndarray]
    pmf: Optional[np.ndarray]
    pdf: Optional[np.ndarray]
    cdf: Optional[np.ndarray]
    survival: Optional[np.ndarray]
    interval: Optional[Tuple[float, float]]
    interval_probability: Optional[float]
    quantile_probabilities: Optional[np.ndarray]
    quantiles: Optional[np.ndarray]
    inverse_survival: Optional[np.ndarray]
    mean: Optional[float]
    variance: Optional[float]
    moment_defined: Dict[str, bool]
    samples: Optional[np.ndarray]
    distribution: Any
    warnings: List[str]

    runtime_fields = OutputDTO.runtime_fields | frozenset(("distribution",))


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------


def _validate_parameters(p):
    """해당 분포 모수의 타입과 정의역을 확인합니다.

    Args:
        p: 해당 분포의 Profile.
    Returns:
        None: 모든 모수가 유효한 경우.
    Raises:
        ValueError/TypeError: 정수·유한값·모수 정의역 오류.
    """
    integer(p.random_state, "random_state")
    integer(p.N, "N", 1)
    integer(p.K, "K", 0)
    integer(p.n, "n", 0)
    if p.K > p.N or p.n > p.N:
        raise ValueError("K/n은 N 이하여야 합니다.")


def _validate_inputs(profile: HypergeometricProfile) -> None:
    """분포 모수 검증.

    Args:
        profile: HypergeometricProfile.
    Returns:
        None: 유효하면 반환.
    Raises:
        TypeError/ValueError: 모수 타입·범위 오류.
    """
    if not isinstance(profile, HypergeometricProfile):
        raise TypeError("HypergeometricProfile이 필요합니다.")
    _validate_parameters(profile)


# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------


def _make_distribution(p):
    """해당 분포의 모수를 SciPy 객체로 변환합니다.

    Args:
        p: 검증한 분포 Profile.
    Returns:
        tuple: frozen distribution과 이산 여부.
    Raises:
        ValueError: 분포 모수 정의역 오류.
    """
    return (stats.hypergeom(p.N, p.K, p.n), True)


def _run_distribution(
    profile: HypergeometricProfile, queries: Dict[str, Any]
) -> HypergeometricOutputDTO:
    """검증한 분포와 조회 연산을 실행합니다.

    Args:
        profile: 분포 모수.
        queries: x/interval/q/n_samples 요청 사전.
    Returns:
        HypergeometricOutputDTO: 밀도/질량·확률·분위수·모멘트.
    Raises:
        ValueError: 조회 구간·확률·난수 크기 오류.
    """
    return HypergeometricOutputDTO(
        **evaluate_distribution(
            "hypergeometric", profile, queries, *_make_distribution(profile)
        )
    )


# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------


def run_hypergeometric(
    profile: Optional[HypergeometricProfile] = None,
    *,
    x: Any = None,
    interval: Any = None,
    q: Any = None,
    n_samples: int = 0,
    recipe_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> HypergeometricOutputDTO:
    """초기하 분포 조회.

    Args:
        profile: 명시적 HypergeometricProfile; None이면 기법별 JSON.
        x: 점 조회, 실수 scalar/1차원 배열. NaN 거부; ±inf 허용.
        interval: [low,high] 양끝 포함 구간 또는 None.
        q: PPF/ISF에 사용할 0≤q≤1 scalar/배열.
        n_samples: 난수 표본수, 0은 생성 안함.
        recipe_path: 사용자 JSON 경로.
        overrides: 레시피 위 적용 조건.
    Returns:
        HypergeometricOutputDTO: runtime distribution은 기본 JSON 제외.
    Raises:
        TypeError/ValueError: 모수·조회·동시 Profile/레시피 오류.
        FileNotFoundError/OSError: 레시피 읽기 실패.
    """
    profile = resolve_profile(
        "distribution-hypergeometric",
        profile,
        recipe_path=recipe_path,
        overrides=overrides,
    )
    _validate_inputs(profile)
    return _run_distribution(
        profile, {"x": x, "interval": interval, "q": q, "n_samples": n_samples}
    )


# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(
        run_hypergeometric(
            x=[0, 1, 2], interval=(0, 2), q=[0.25, 0.5, 0.75], n_samples=3
        ).to_json(indent=2)
    )
