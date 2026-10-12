"""selection UCI subplot 스켈레톤.

원 자료 → pane별 Profile → 분석 결과/그리기 함수 연결 → 명시적 subplot 조합.
각 pane의 수치와 가정을 반환합니다. 작성자: 김재현(Finn), 2026-10-12.
변경 이력: v1.0.0 최초 추가. 라이선스: Proprietary; UCI 원 자료는 CC BY 4.0.
"""
__author__='김재현(Finn)'
__email__='penguin.klg@gmail.com'
__version__='1.0.0'
__status__='Development'
import numpy as np
from src.Recipes import load_profile
from ._common import render_panels
from src.datasets.uci import load_tabular
from src.Visualization.selection.feature_importance import plot_feature_importance
from src.Visualization.selection.validation_curve import plot_validation_curve

RECIPES={'feature_importance': 'plot-selection-feature-importance', 'validation_curve': 'plot-selection-validation-curve'}

def load_data(cache_dir=None):
    """학습·그림 생성 없이 공식 UCI 원 자료를 읽습니다.

    Args:
        cache_dir: 선택 UCI 원본 캐시.
    Returns:
        object: 분야별 원 관측 자료.
    Raises:
        ValueError/TypeError/OSError: 입력·설정·자료·다운로드·축 오류.
    """
    return load_tabular('classification',cache_dir=cache_dir)

def load_recipe(overrides=None):
    """pane별 기존 JSON/Profile을 읽습니다.

    Args:
        overrides: pane 이름별 레시피 overrides 사전 또는 None.
    Returns:
        dict: pane 이름 → 전용 Profile.
    Raises:
        ValueError/TypeError/OSError: 입력·설정·자료·다운로드·축 오류.
    """
    overrides={} if overrides is None else overrides
    if not isinstance(overrides,dict) or set(overrides)-set(RECIPES):raise ValueError('pane override 이름 오류')
    return {name:load_profile(key,overrides=overrides.get(name)) for name,key in RECIPES.items()}

def build_pipeline(data, profiles=None):
    """원 자료에서 필요한 분석 수치를 준비하고 단일 ax 호출을 연결합니다.

    Args:
        data: load_data의 원 자료.
        profiles: pane 이름별 전용 Profile 또는 None.
    Returns:
        list: (pane 이름, ax callable) 목록.
    Raises:
        ValueError/TypeError/OSError: 입력·설정·자료·다운로드·축 오류.
    """
    profiles=load_recipe() if profiles is None else profiles
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split,learning_curve,StratifiedKFold
    from sklearn.inspection import permutation_importance
    X=data.drop(columns='target');y=data.target
    train,holdout=train_test_split(np.arange(len(data)),test_size=.4,random_state=1234,stratify=y)
    validation,test=train_test_split(holdout,test_size=.5,random_state=1234,stratify=y.iloc[holdout])
    # 최종 test는 그림·임계값·Feature 검토에 사용하지 않습니다.
    model=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=1000,random_state=1234))]).fit(X.iloc[train],y.iloc[train])
    importance=permutation_importance(model,X.iloc[validation],y.iloc[validation],n_repeats=10,random_state=1234,scoring='accuracy')
    sizes,train_scores,validation_scores=learning_curve(model,X.iloc[train],y.iloc[train],cv=StratifiedKFold(3,shuffle=True,random_state=1234),train_sizes=[.5,.75,1.],scoring='accuracy',shuffle=True,random_state=1234)
    return [
        ('feature_importance',lambda ax:plot_feature_importance(X.columns.tolist(),importance.importances_mean,importance.importances_std,ax=ax,profile=profiles['feature_importance'])),
        ('validation_curve',lambda ax:plot_validation_curve(sizes,train_scores.mean(axis=1),validation_scores.mean(axis=1),train_scores.std(axis=1),validation_scores.std(axis=1),ax=ax,profile=profiles['validation_curve'])),
    ]

def main(cache_dir=None, axes=None, overrides=None):
    """사용자의 subplot 축에 연결하거나 예제 레이아웃을 생성합니다.

    Args:
        cache_dir: 선택 UCI 원본 캐시.
        axes: 기존 subplot Axes 배열 또는 None; None에서만 예제 레이아웃 생성.
        overrides: pane 이름별 레시피 overrides 사전 또는 None.
    Returns:
        EDADashboardOutputDTO: 같은 축과 pane별 수치.
    Raises:
        ValueError/TypeError/OSError: 입력·설정·자료·다운로드·축 오류.
    """
    data=load_data(cache_dir)
    profiles=load_recipe(overrides)
    calls=build_pipeline(data,profiles)
    return render_panels(calls,axes=axes,ncols=2)

if __name__=='__main__':
    result=main()
    print({name:list(panel.diagnostics) for name,panel in result.panels.items()})
