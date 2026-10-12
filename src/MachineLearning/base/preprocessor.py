"""
================================================================================
preprocessor.py - FeatureSpec과 ColumnTransformer 연결
================================================================================
설명:
    - 사용자 참고 스켈레톤의 특성 그룹 전처리와 설정 조회를 제공합니다.
분석 가정:
    - CV에는 get_preprocessor의 sklearn 변환기를 연결해 폴드마다 적합합니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin-klg@naver.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR           VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)       v1.0.0    UCI 스켈레톤 전처리 연결 추가
================================================================================
"""
__author__ = "김재현(Finn)"
__email__ = "penguin-klg@naver.com"
__version__ = "1.0.0"
__status__ = "Development"

from sklearn.compose import ColumnTransformer
from .feature import FeatureSpec


class BasePreprocessor:
    """FeatureSpec 목록을 조합하는 전처리 설정 객체.

    Attributes:
        features (list): 등록한 FeatureSpec 목록.
        preprocessor (object): 활성 특성의 ColumnTransformer 또는 None.
    """
    def __init__(self, features=None):
        """특성 그룹을 등록하고 미적합 전처리를 생성합니다.

        Args:
            features: FeatureSpec iterable 또는 None.
        Returns:
            None: 미적합 설정 객체.
        Raises:
            ValueError/TypeError: 잘못된 특성 또는 중복 식별자.
        """
        self.features = []
        self.preprocessor = None
        for feature in features or []:
            self.add_feature(feature)

    def _build_preprocessor(self):
        """활성 그룹을 ColumnTransformer 설정으로 변환합니다.

        Args:
            None: 등록한 그룹을 사용합니다.
        Returns:
            None: 미적합 전처리로 갱신합니다.
        Raises:
            ValueError: sklearn 전처리 설정 오류.
        """
        transformers = [(f.name, f.transformer, f.columns) for f in self.features if f.enabled]
        self.preprocessor = ColumnTransformer(transformers, remainder='drop') if transformers else None

    def add_feature(self, feature):
        """고유 이름의 특성 그룹을 등록합니다.

        Args:
            feature: FeatureSpec.
        Returns:
            BasePreprocessor: 갱신한 자신.
        Raises:
            TypeError/ValueError: 자료형 또는 중복 이름 오류.
        """
        if not isinstance(feature, FeatureSpec):
            raise TypeError('FeatureSpec이 필요합니다.')
        if any(f.name == feature.name for f in self.features):
            raise ValueError('feature 이름이 중복되었습니다: ' + feature.name)
        self.features.append(feature)
        self._build_preprocessor()
        return self

    def remove_feature(self, name):
        """이름으로 특성 그룹을 제거합니다.

        Args:
            name: 등록한 그룹 이름.
        Returns:
            BasePreprocessor: 갱신한 자신.
        Raises:
            KeyError: 등록하지 않은 이름.
        """
        self.get_feature(name)
        self.features = [f for f in self.features if f.name != name]
        self._build_preprocessor()
        return self

    def get_feature(self, name):
        """등록한 특성 그룹을 조회합니다.

        Args:
            name: 그룹 이름.
        Returns:
            FeatureSpec: 등록한 그룹.
        Raises:
            KeyError: 등록하지 않은 이름.
        """
        for feature in self.features:
            if feature.name == name:
                return feature
        raise KeyError(name)

    def describe(self):
        """등록한 그룹들의 수치·이름 설정을 조회합니다.

        Args:
            None: 등록한 그룹을 사용합니다.
        Returns:
            list: FeatureSpec 설명 목록.
        Raises:
            None: 검증한 설정만 반환합니다.
        """
        return [f.describe() for f in self.features]

    def get_preprocessor(self):
        """sklearn Pipeline/CV에서 사용할 변환기를 반환합니다.

        Args:
            None: 현재 전처리 설정.
        Returns:
            ColumnTransformer: 미적합 또는 이미 적합한 변환기.
        Raises:
            ValueError: 활성 그룹 없음.
        """
        if self.preprocessor is None:
            raise ValueError('활성 feature 전처리가 없습니다.')
        return self.preprocessor

    def fit(self, X, y=None):
        """전처리를 적합합니다.

        Args:
            X: 원 특성 자료.
            y: 선택 타깃.
        Returns:
            BasePreprocessor: 적합한 자신.
        Raises:
            ValueError: 활성 그룹·열·변환 오류.
        """
        self.get_preprocessor().fit(X, y)
        return self

    def transform(self, X):
        """적합한 전처리로 새 특성을 변환합니다.

        Args:
            X: 같은 열 이름의 새 자료.
        Returns:
            object: 변환한 수치 행렬.
        Raises:
            ValueError/NotFittedError: 미적합 또는 입력 오류.
        """
        return self.get_preprocessor().transform(X)

    def fit_transform(self, X, y=None):
        """전처리를 적합하고 같은 자료를 변환합니다.

        Args:
            X: 학습 특성.
            y: 선택 타깃.
        Returns:
            object: 변환한 학습 행렬.
        Raises:
            ValueError: 입력·변환 오류.
        """
        return self.get_preprocessor().fit_transform(X, y)
