"""레시피 → 실행기 → 전용 DTO의 재현 가능한 사용 예제."""
import numpy as np
import pandas as pd
from src.Recipes import load_profile
from src.Preprocessing.resampling.svm_smote import run_svm_smote


def main():
    """기본 JSON을 읽고 고정 자료로 실행합니다."""
    profile=load_profile('resampling-svm-smote')
    from sklearn.datasets import make_classification
    X,y=make_classification(n_samples=160,n_features=4,n_redundant=0,weights=[.75,.25],random_state=1234)
    result=run_svm_smote(X,y,profile)
    print(result.to_json(indent=2))
    return result


if __name__=="__main__":
    main()
