"""rope UCI 스켈레톤.

데이터 로딩 → JSON/Profile → Feature/분석 연결 → 실행 → 수치 확인.
전처리는 학습 Feature에서만 적합하며 타깃을 제외합니다. 시각화를 포함하지 않습니다.
작성자: 김재현(Finn), 2026-10-12. 변경 이력: v1.0.0 최초 추가.
라이선스: Proprietary; UCI 자료는 별도 CC BY 4.0.
"""
__author__ = "김재현(Finn)"
__email__ = "penguin-klg@naver.com"
__version__ = "1.0.0"
__status__ = "Development"

import numpy as np
from src.Recipes import load_profile
from src.datasets.uci import load_libras
from src.Preprocessing.positional.rope import build_rope_transformer, run_rope

def load_data(cache_dir=None):
    """Libras 원 프레임을 고정 투영해 위치 회전용 임베딩 예제를 만듭니다.

    Args:
        cache_dir: 선택 UCI 원본 캐시 경로 또는 None.
    Returns:
        ndarray: (360,45,16); 투영은 학습된 모델이 아닙니다.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·스키마·다운로드 오류.
    """
    coordinates, labels = load_libras(cache_dir)
    projection = np.random.RandomState(1234).normal(size=(2, 16))
    return coordinates @ projection

def load_recipe(overrides=None):
    """RoPE JSON을 읽습니다.

    Args:
        overrides: 기본 JSON의 명시적 필드 재정의 또는 None.
    Returns:
        RoPEProfile: 회전 설정.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·스키마·다운로드 오류.
    """
    return load_profile('rope', overrides=overrides)

def build_pipeline(profile=None, overrides=None):
    """표 Feature와 분리한 임베딩 단계의 미적합 위치 변환기를 만듭니다.

    Args:
        profile: 전용 Profile 또는 None; overrides와 동시 지정 불가.
        overrides: 기본 JSON의 명시적 필드 재정의 또는 None.
    Returns:
        RotaryPositionTransformer: 통계 학습 없는 변환기.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·스키마·다운로드 오류.
    """
    if profile is not None and overrides is not None:
        raise ValueError('profile과 overrides는 함께 지정하지 않습니다.')
    return build_rope_transformer(load_recipe(overrides) if profile is None else profile)

def main(cache_dir=None, overrides=None):
    """시퀀스 축 위치를 회전하고 shape·노름 보존을 확인합니다.

    Args:
        cache_dir: 선택 UCI 원본 캐시 경로 또는 None.
        overrides: 기본 JSON의 명시적 필드 재정의 또는 None.
    Returns:
        RoPEOutputDTO: 회전 임베딩·위치·주파수.
    Raises:
        ValueError/TypeError/OSError: 자료·설정·스키마·다운로드 오류.
    """
    embeddings = load_data(cache_dir)
    profile = load_recipe(overrides)
    transformer = build_pipeline(profile=profile)
    # RoPE는 미적합 Feature 통계가 없으며 임베딩 단계에만 연결합니다.
    result = run_rope(embeddings, profile=profile)
    print('embedding shape:', result.preprocessed_feature.shape)
    print('maximum norm error:', float(np.max(np.abs(np.linalg.norm(embeddings, axis=-1) - np.linalg.norm(result.preprocessed_feature, axis=-1)))))
    return result


if __name__ == '__main__':
    main()
