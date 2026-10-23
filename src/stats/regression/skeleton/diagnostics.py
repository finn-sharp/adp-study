"""diagnostics UCI 스켈레톤.

원 자료 → JSON/Profile → 학습 전용 파이프라인 → 수치/진단/사용자 Axes.
작성자: 김재현(Finn), 2026-10-12. 변경 이력: v1.0.0 최초 추가.
라이선스: Proprietary; UCI 원본은 CC BY 4.0. import에서 다운로드·학습·그림 생성 없음.
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"
from src.MachineLearning.LinearModels.skeleton.regression_workflow import (
    load_data,
    load_recipe,
    build_pipeline,
)
from src.MachineLearning.LinearModels.skeleton.regression_workflow import main as _main


def main(cache_dir=None, overrides=None):
    """전용 입력/설정을 검증하고 해당 계산을 수행합니다.

    Args:
        cache_dir: 전용 명세의 cache_dir 입력.
        overrides: 전용 명세의 overrides 입력.
    Returns:
        RegressionStudyOutputDTO: 각자의 파일에서 계산한 회귀 진단.
    Raises:
        ValueError/TypeError/OSError: 입력·정의역·설정·레시피 오류.
    """
    return _main(cache_dir=cache_dir, overrides=overrides, draw=False)


if __name__ == "__main__":
    result = main()
    print(result.diagnostics["joint_f"].tests)
    print(result.workflow.inference.coefficients)
