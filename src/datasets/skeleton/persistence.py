"""
================================================================================
persistence.py - 원 자료/분석 객체 → 저장 Profile → save → load → 복원 DTO 스켈레톤.
================================================================================
설명:
    - 원 자료/분석 객체 → 저장 Profile → save → load → 복원 DTO 스켈레톤.
분석 가정:
    - 기존 객체를 저장하고 복원합니다. pickle/joblib 입력은 신뢰한 파일이어야 하며, 전처리 적합·재학습을 수행하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-27
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
from src.datasets import save, load_artifact


def main(value, destination, *, save_overrides=None, load_overrides=None):
    """동일 환경에서 저장·복원한 객체와 출처 DTO를 확인합니다.

    Args:
        value: Parquet용 DataFrame 또는 pickle/joblib 분석 객체.
        destination: 로컬 파일 경로. 부모 폴더는 존재해야 함.
        save_overrides: PersistenceProfile의 저장 설정.
        load_overrides: PersistenceProfile의 복원 설정. pickle/joblib은 trusted=True 필수.
    Returns:
        PersistenceOutputDTO: value에 복원 객체를 담은 DTO.
    Raises:
        TypeError/ValueError/OSError/ImportError: 형식·입력·파일·의존성 오류.
    """
    saved = save(value, destination, overrides=save_overrides)
    restored = load_artifact(destination, overrides=load_overrides)
    print(saved.to_json(indent=2))
    print(restored.to_json(indent=2))
    return restored


if __name__ == "__main__":
    with TemporaryDirectory() as folder:
        result = main(
            pd.DataFrame({"feature": [1, 2], "target": [3, 5]}),
            Path(folder) / "table.pkl",
            load_overrides={"trusted": True},
        )
        print(result.value)
