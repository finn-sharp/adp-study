"""
================================================================================
tomek_links.py - TomekLinks 학습 표본 보완
================================================================================
설명:
    - 기법별 Profile·JSON 레시피·공개 실행기·전용 Output DTO를 제공합니다.
분석 가정:
    - 학습 폴드에만 적합합니다. 지원 자료형=numeric. 수치 거리 기법의 범주 코드 직접 적용을 금지합니다.
작성 정보:
    - 작성자: 김재현(Finn)
    - 이메일: penguin.klg@gmail.com
    - 작성일: 2026-10-20
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-20  김재현(Finn)     v1.0.0    도메인 공통화·전용 명세·실행 예제 추가
================================================================================
"""
__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"


# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
from src.output import OutputDTO
from src.Recipes import resolve_profile
from ._common import validate, resample

# ---------------------------------------------------------------------------
# 2. Input Profile
# ---------------------------------------------------------------------------
@dataclass
class TomekLinksProfile:
    """Attributes:
        sampling_strategy (str/float/Dict: imblearn 목표 클래스 규칙. 기본 'auto'. float는 이진만, cleaning은 균형 개수를 보장하지 않음.)
    """
    sampling_strategy: Any = 'auto'

# ---------------------------------------------------------------------------
# 3. Output DTO
# ---------------------------------------------------------------------------
@dataclass
class TomekLinksOutputDTO(OutputDTO):
    """Attributes:
        method (str): 실행 기법 이름.
        X_resampled (Any): 보완한 학습 특성. 행 순서는 y_resampled와 대응.
        y_resampled (Any): 보완한 학습 타깃.
        class_counts_before (Dict[Any,int]): 원래 클래스별 수.
        class_counts_after (Dict[Any,int]): 보완 후 클래스별 수.
        sample_indices (Optional[np.ndarray]): 제공되는 경우 원 입력 행 위치.
            복제·선택은 대응 가능, 합성·중심점은 None. 인덱스 라벨과 다름.
        sampler (Any): 적합 네이티브 imblearn sampler. 기본 JSON에서 제외.
        warnings (List[str]): 평가 누출·원행 대응 제한.
    """
    method: str
    X_resampled: Any
    y_resampled: Any
    class_counts_before: Dict[Any, int]
    class_counts_after: Dict[Any, int]
    sample_indices: Optional[np.ndarray]
    sampler: Any
    warnings: List[str]

# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------
def _validate_inputs(X,y,p):
    """Args: X: 학습 특성; y: 학습 타깃; p: 전용 Profile.
    Returns: None.
    Raises: TypeError/ValueError: Profile·자료형·shape 오류. 이웃·클래스 조건은 estimator도 검사.
    """
    validate(X,y,p,TomekLinksProfile,'numeric')

# ---------------------------------------------------------------------------
# 5. Individual Model Runners
# ---------------------------------------------------------------------------
def _run(X,y,p):
    """Args: X/y: 검증된 학습 입력; p: Profile.
    Returns: TomekLinksOutputDTO: 학습 표본과 클래스 수.
    Raises: ImportError: imbalanced-learn 누락. ValueError: 보완 불가능한 입력.
    """
    return resample(X,y,p,'tomek_links',TomekLinksOutputDTO)

# ---------------------------------------------------------------------------
# 6. Public Runner
# ---------------------------------------------------------------------------
def run_tomek_links(X,y,profile: Optional[TomekLinksProfile] = None,
                recipe_path: Optional[str] = None, overrides: Optional[Dict[str,Any]] = None) -> TomekLinksOutputDTO:
    """Args:
        X: 학습 자료만. shape=(n,p), 결측 없음. 지원 자료형: numeric.
        y: 같은 행 순서의 이진/다중 분류 타깃, shape=(n,).
        profile: 전용 Profile. JSON 옵션과 혼용 불가.
        recipe_path: 사용자 JSON. 기본 resampling-tomek-links.json.
        overrides: 일회성 설정.
    Returns: TomekLinksOutputDTO: 실제 보완 자료·전후 클래스 수·적합 sampler.
    Raises: TypeError/ValueError/OSError: 입력·설정·표본 부족. ImportError: 의존성 누락.
    Notes: CV에서는 load_resampler로 네이티브 sampler를 imblearn Pipeline 안에 넣으세요.
    """
    p=resolve_profile('resampling-tomek-links',profile,recipe_path=recipe_path,overrides=overrides)
    _validate_inputs(X,y,p)
    return _run(X,y,p)

# ---------------------------------------------------------------------------
# 7. Example Execution
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from sklearn.datasets import make_classification
    X,y=make_classification(n_samples=160,n_features=4,n_redundant=0,weights=[.75,.25],random_state=1234)
    result=run_tomek_links(X,y)
    print(result.class_counts_before,result.class_counts_after)
