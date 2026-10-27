"""
================================================================================
_common.py - 최적화 Profile의 기본 타입·유한성 검증. 목적함수와 자료는 JSON과 분리합니다.
================================================================================
설명:
    - 최적화 Profile의 기본 타입·유한성 검증. 목적함수와 자료는 JSON과 분리합니다.
분석 가정:
    - 입력의 형상·타입·자료 순서를 호출자가 확인합니다. 계산과 파일 쓰기의 조건은 각 공개 함수의 계약을 따릅니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-27
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""

from dataclasses import fields
import numpy as np


def validate_profile(profile, cls, check_alpha=False):
    """전용 설정 기본 타입을 확인합니다.

    Args:
        profile: 검증할 설정 객체.
        cls: 전용 dataclass 타입.
        check_alpha: 공통 검증 인터페이스 호환용, 최적화에서는 False.
    Returns:
        None: 설정이 유효함.
    Raises:
        TypeError/ValueError: 필드 타입 또는 유한성 오류.
    """
    if not isinstance(profile, cls):
        raise TypeError("전용 Profile이 필요합니다.")
    for field in fields(profile):
        value = getattr(profile, field.name)
        if field.type is bool and not isinstance(value, bool):
            raise TypeError(field.name + " bool 오류")
        if field.type is int and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise TypeError(field.name + " int 오류")
        if field.type is float and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise ValueError(field.name + " 유한 실수 오류")
