"""파일/스트림 → 로딩 Profile → DataFrame DTO → Feature 입력 스켈레톤.

실행: python -m src.datasets.skeleton.tabular
사용자 자료: main("data.xlsx", overrides={"sheet_name": "자료"})
학습용 fit·결측 대치·sampling은 이 로딩 단계에서 수행하지 않습니다.
"""

from io import StringIO
from src.datasets.tabular import run_tabular_loader


def load_data(source, *, recipe_path=None, overrides=None):
    """형식별 설정을 적용하고 출처·경고를 포함하는 DTO를 반환합니다.

    Args:
        source: 로컬 경로·bytes·text/binary 스트림.
        recipe_path: 선택 데이터 로딩 JSON 레시피.
        overrides: 모델 설정과 분리한 로딩 설정 재정의.
    Returns:
        TabularLoaderOutputDTO: data·스키마·출처·설정·경고.
    Raises:
        TypeError/ValueError/OSError/ImportError: 공통 로더 입력·파싱·의존성 오류.
    """
    return run_tabular_loader(source, recipe_path=recipe_path, overrides=overrides)


def main(source=None, *, recipe_path=None, overrides=None):
    """외부 파일 또는 내장 JSON 예제를 읽습니다. 다운로드하지 않습니다.

    Args:
        source: 사용자 입력; None이면 익명 JSON 예제 스트림.
        recipe_path: 선택 로딩 레시피 경로.
        overrides: 명시적 로딩 설정. 내장 예제에서는 format=json이 필요.
    Returns:
        TabularLoaderOutputDTO: 후속 Feature/Model에 전달할 data를 포함.
    Raises:
        TypeError/ValueError/OSError/ImportError: 로더 입력·파싱·의존성 오류.
    """
    if source is None:
        source = StringIO('[{"feature":1,"target":2},{"feature":3,"target":4}]')
        if recipe_path is None:
            overrides = dict({"format": "json"}, **(overrides or {}))
    result = load_data(source, recipe_path=recipe_path, overrides=overrides)
    print(result.to_json(indent=2))
    return result


if __name__ == "__main__":
    main()
