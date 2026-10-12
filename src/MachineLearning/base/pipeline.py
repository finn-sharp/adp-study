"""
================================================================================
pipeline.py - 전처리·모델의 Basepipeline 연결
================================================================================
설명:
    - 사용자 참고 스켈레톤의 sklearn Pipeline 구성과 교체/조회 API를 제공합니다.
분석 가정:
    - 전처리와 모델은 같은 Pipeline에서 학습하고 새 입력을 처리합니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin-klg@naver.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR           VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)       v1.0.0    UCI 스켈레톤 파이프라인 연결 추가
================================================================================
"""
__author__ = "김재현(Finn)"
__email__ = "penguin-klg@naver.com"
__version__ = "1.0.0"
__status__ = "Development"

from sklearn.pipeline import Pipeline


class Basepipeline:
    """원 참고 스켈레톤의 전처리·모델 연결 객체.

    Attributes:
        preprocessor (object): sklearn 호환 전처리기.
        model (object): 학습/예측/분석 추정기.
        pipeline (Pipeline): preprocessor → model 순서의 sklearn Pipeline.
    """
    def __init__(self, preprocessor, model):
        """전처리와 모델을 같은 학습 파이프라인으로 연결합니다.

        Args:
            preprocessor: fit/transform을 지원하는 변환기.
            model: fit을 지원하는 마지막 추정기.
        Returns:
            None: 미적합 연결 객체.
        Raises:
            ValueError: sklearn 단계 설정 오류.
        """
        self.preprocessor = preprocessor
        self.model = model
        self.pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])

    def get_model(self):
        """현재 모델을 조회합니다.

        Args:
            None: 현재 연결.
        Returns:
            object: 현재 마지막 추정기.
        Raises:
            None: 현재 참조를 반환합니다.
        """
        return self.model

    def set_model(self, newmodel):
        """연결된 모델을 교체합니다.

        Args:
            newmodel: 새 미적합 또는 적합 모델.
        Returns:
            None: Pipeline과 조회 참조를 함께 갱신.
        Raises:
            ValueError: sklearn 설정 오류.
        """
        self.pipeline.set_params(model=newmodel)
        self.model = newmodel

    def get_preprocessor(self):
        """현재 전처리를 조회합니다.

        Args:
            None: 현재 연결.
        Returns:
            object: 현재 전처리기.
        Raises:
            None: 현재 참조를 반환합니다.
        """
        return self.preprocessor

    def set_preprocessor(self, newpreprocessor):
        """연결된 전처리를 교체합니다.

        Args:
            newpreprocessor: 새 미적합 또는 적합 변환기.
        Returns:
            None: Pipeline과 조회 참조를 함께 갱신.
        Raises:
            ValueError: sklearn 설정 오류.
        """
        self.pipeline.set_params(preprocessor=newpreprocessor)
        self.preprocessor = newpreprocessor
