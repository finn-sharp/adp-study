"""
================================================================================
tf_time_series.py - tf-time-series UCI 스켈레톤
================================================================================
설명:
    - 데이터 로딩 → JSON/Profile → 전처리/모델 연결 → 학습/분석 → 수치 평가.
분석 가정:
    - Dense/독립 손동작은 표본 분할, Bike Sharing 예측은 원 날짜를 먼저 분할하고 각 구간 안에서만 창을 만듭니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary (UCI 자료의 라이선스는 별도 출처 명시)
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)       v1.0.0    기존 파이프라인 기반 UCI 수치 예제
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""

__author__ = "김재현(Finn)"
__email__ = "penguin.klg@gmail.com"
__version__ = "1.0.0"
__status__ = "Development"

from src.Recipes import load_profile
from src.DeepLearning.TensorFlow.time_series import run_tf_time_series
from src.datasets.uci import load_tabular, load_libras, load_bike


def load_data(cache_dir=None):
    """UCI bike 원본을 모델의 표본/시퀀스 계약으로 변환합니다.

    Args:
        cache_dir: 선택 UCI 캐시.
    Returns:
        object: Dense 표, 독립 시퀀스 또는 날짜 순 시계열.
    Raises:
        ValueError/OSError: 자료·다운로드 오류.
    """
    return load_bike(cache_dir)


def load_recipe(overrides=None):
    """기존 TensorFlow JSON에서 전용 Profile을 읽습니다.

    Args:
        overrides: epochs/cell 등 명시적인 설정.
    Returns:
        object: TensorFlow 전용 Profile.
    Raises:
        ValueError/TypeError/OSError: 레시피 오류.
    """
    return load_profile("tf-time-series", task=None, overrides=overrides)


def build_pipeline(profile=None, overrides=None):
    """학습기 실행 함수를 구성하고 기존 train/validation/test 전처리를 유지합니다.

    Args:
        profile: TensorFlow Profile 또는 None.
        overrides: Profile 생략 시 명시적 재정의.
    Returns:
        callable: 원 자료에서 학습·예측·평가까지 실행하는 함수.
    Raises:
        ValueError: Profile과 overrides 동시 지정.
    """
    if profile is not None and overrides is not None:
        raise ValueError("profile과 overrides를 동시에 지정하지 않습니다.")
    profile = load_recipe(overrides) if profile is None else profile

    def execute(data):
        """기존 runner의 내부 분할 이후에만 scaler를 적합합니다.

        Args:
            data: load_data가 반환한 원 자료.
        Returns:
            OutputDTO: 원 단위 예측과 수치 성능.
        Raises:
            ValueError/ImportError: 자료 조건 또는 TensorFlow 미설치.
        """
        return run_tf_time_series(data.to_numpy(dtype=float), profile=profile)

    return execute


def main(cache_dir=None, overrides=None):
    """UCI 자료 → JSON → 학습 파이프라인 → 보류 수치 결과를 실행합니다.

    Args:
        cache_dir: 선택 UCI 캐시.
        overrides: 명시적 레시피 재정의.
    Returns:
        OutputDTO: 기존 TensorFlow 전용 결과.
    Raises:
        ValueError/OSError/ImportError: 자료·레시피·라이브러리 오류.
    """
    data = load_data(cache_dir)
    profile = load_recipe(overrides)
    pipeline = build_pipeline(profile=profile)
    result = pipeline(data)
    from src.Evaluation.time_series.metrics import run_time_series_metrics

    columns = (
        list(profile.target_columns)
        if profile.target_columns is not None
        else list(range(data.shape[1]))
    )
    training = data.to_numpy(dtype=float)[result.train_indices][:, columns]
    evaluation = run_time_series_metrics(
        result.y_test, result.predictions, y_train=training
    )
    result.test_metrics.update(
        {k: float(v) for k, v in evaluation.value.items() if k != "horizon_errors"}
    )
    print("horizon_metrics:", evaluation.details["metrics"]["horizon_errors"]["table"])
    print("=== Example : UCI bike tf_time_series ===")
    print("input_shape:", result.input_shape)
    print("metrics:", result.test_metrics)
    print("epochs_trained:", result.epochs_trained)
    return result


if __name__ == "__main__":
    main()
