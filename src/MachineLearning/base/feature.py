"""
================================================================================
feature.py - FeatureSpec 전처리 스키마
================================================================================
설명:
    - 사용자 참고 스켈레톤의 특성 그룹·열·전처리 계약을 제공합니다.
분석 가정:
    - 전처리 적합은 데이터 분할 이후 학습 자료에서 수행합니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin-klg@naver.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR           VERSION   DESCRIPTION
    2026-10-12  김재현(Finn)       v1.0.0    UCI 스켈레톤의 특성 계약 추가
================================================================================
"""
__author__ = "김재현(Finn)"
__email__ = "penguin-klg@naver.com"
__version__ = "1.0.0"
__status__ = "Development"

from dataclasses import dataclass
from typing import Any, List


@dataclass
class FeatureSpec:
    """한 특성 그룹의 전처리 조건.

    Attributes:
        name (str): 비어 있지 않은 그룹 식별자.
        columns (List[str]): 같은 순서로 처리할 원 입력 열 이름.
        transformer (Any): sklearn 변환기 또는 passthrough/drop 문자열.
        enabled (bool): False이면 그룹을 전처리에서 제외; 기본 True.
    """
    name: str
    columns: List[str]
    transformer: Any
    enabled: bool = True

    def __post_init__(self):
        """식별자·열·변환기를 확인하고 원 열 목록을 복사합니다.

        Args:
            None: dataclass 생성자 필드를 사용합니다.
        Returns:
            None: 유효한 설정을 보존합니다.
        Raises:
            ValueError/TypeError: 이름·열·활성 상태·변환기 오류.
        """
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('feature 이름은 비어 있지 않은 문자열이어야 합니다.')
        if isinstance(self.columns, str) or not self.columns or any(not isinstance(c, str) or not c for c in self.columns):
            raise ValueError('feature에는 비어 있지 않은 열 이름 목록이 필요합니다.')
        if len(set(self.columns)) != len(self.columns):
            raise ValueError('같은 그룹의 열 이름이 중복되었습니다.')
        if not isinstance(self.enabled, bool):
            raise TypeError('enabled는 bool이어야 합니다.')
        if isinstance(self.transformer, str):
            if self.transformer not in ('passthrough', 'drop'):
                raise ValueError('문자열 변환기는 passthrough/drop만 지원합니다.')
        elif not hasattr(self.transformer, 'fit') or not hasattr(self.transformer, 'transform'):
            raise TypeError('sklearn 호환 fit/transform 변환기가 필요합니다.')
        self.columns = list(self.columns)

    def describe(self):
        """메모리 주소 없는 전처리 설정을 반환합니다.

        Args:
            None: 현재 특성 그룹을 사용합니다.
        Returns:
            dict: 이름·원 열 순서·변환기 종류·활성 상태.
        Raises:
            None: 검증한 필드를 반환합니다.
        """
        kind = self.transformer if isinstance(self.transformer, str) else type(self.transformer).__name__
        return {'name': self.name, 'columns': self.columns.copy(), 'transformer': kind, 'enabled': self.enabled}
