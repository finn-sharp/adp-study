"""레시피 → 실행기 → 전용 DTO의 재현 가능한 사용 예제."""
import numpy as np
import pandas as pd
from src.Recipes import load_profile
from src.stats.survival.rmst import run_rmst


def main():
    """기본 JSON을 읽고 고정 자료로 실행합니다."""
    profile=load_profile('survival-rmst')
    result=run_rmst([10,20,30],[1,0,1],profile)
    print(result.to_json(indent=2))
    return result


if __name__=="__main__":
    main()
