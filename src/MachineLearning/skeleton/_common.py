"""
================================================================================
_common.py - UCI 스켈레톤 파이프라인·결과 공통 계약
================================================================================
설명:
    - 데이터 로딩 → JSON/Profile → 전처리/모델 연결 → 학습/분석 → 수치 평가.
분석 가정:
    - 보류 자료는 전처리 적합·변수 선택·모델 선택에 사용하지 않습니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin-klg@naver.com)
    - 작성일: 2026-10-12
    - 라이선스: Proprietary (UCI 자료의 라이선스는 별도 출처 명시)
변경 이력 (Revision History):
    DATE        AUTHOR           VERSION   DESCRIPTION
    ----------------------------------------------------------------------------
    2026-10-12  김재현(Finn)       v1.0.0    UCI 스켈레톤의 공통 계약과 수치 자료 변환
================================================================================
"""
__author__ = "김재현(Finn)"
__email__ = "penguin-klg@naver.com"
__version__ = "1.0.0"
__status__ = "Development"

from dataclasses import asdict, dataclass, replace
import importlib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from src.output import OutputDTO
from src.MachineLearning.base.feature import FeatureSpec
from src.MachineLearning.base.preprocessor import BasePreprocessor
from src.MachineLearning.base.pipeline import Basepipeline
from src.MachineLearning.common import make_scaler


@dataclass
class SkeletonOutputDTO(OutputDTO):
    """스켈레톤의 수치 결과 및 재사용 가능한 학습 파이프라인.

    Attributes:
        model_name (str): 스켈레톤 모델명.
        dataset_name (str): 사용한 UCI 자료명.
        task (str): 분류·회귀·분석 목적.
        n_rows (int): 실제 사용한 표본 수.
        n_features (int): 실제 입력 특성 수.
        metrics (dict): 보류 성능 또는 전체 자료의 기술 지표.
        details (dict): 수치/shape·선택 특성·실제 레시피 등 진단.
        model (object): 학습 파이프라인; JSON에서 제외.
        encoder (object): 학습 라벨 인코더; 분류 이외 None, JSON에서 제외.
        predictions (object): 보류 예측 또는 전체 embedding/labels; 미해당 None.
        train_indices (object): 원 입력의 학습 행 위치; 분석은 None.
        test_indices (object): 원 입력의 보류 행 위치; 분석은 None.
    """
    model_name: str
    dataset_name: str
    task: str
    n_rows: int
    n_features: int
    metrics: dict
    details: dict
    model: object
    encoder: object = None
    predictions: object = None
    train_indices: object = None
    test_indices: object = None

    def predict(self, X):
        """지도 모델의 전처리와 라벨 복원을 새 입력에도 재사용합니다.

        Args:
            X: 학습과 같은 이름/순서의 원 특성 DataFrame.
        Returns:
            np.ndarray: 회귀값 또는 원 분류 라벨.
        Raises:
            ValueError/AttributeError: 비지도 분석 또는 입력/미적합 오류.
        """
        if self.task not in ('classification', 'regression'):
            raise ValueError('이 분석 결과에는 지도 학습 predict를 적용하지 않습니다.')
        values = self.model.pipeline.predict(X)
        return self.encoder.inverse_transform(np.asarray(values, dtype=int)) if self.encoder is not None else values


def connect_pipeline(columns, estimator, profile, scaling=None):
    """FeatureSpec을 sklearn ColumnTransformer로 변환해 모델과 연결합니다.

    Args:
        columns: 입력 특성 열 순서.
        estimator: 미적합 sklearn 추정기 또는 분석 adapter.
        profile: 모델별 Profile.
        scaling: 특성 선택 내부 CV 등에서 필요한 명시적 스케일링 재정의.
    Returns:
        Basepipeline: FeatureSpec 설명과 Profile을 보존한 연결 객체.
    Raises:
        ValueError/TypeError: 열·전처리 설정 오류.
    """
    features = [FeatureSpec(name='numeric', columns=list(columns), transformer=make_scaler(profile.scaling if scaling is None else scaling))]
    preprocessor = BasePreprocessor(features)
    # The supplied BasePreprocessor is not cloneable; use its sklearn transformer for CV.
    pipeline = Basepipeline(preprocessor.get_preprocessor(), estimator)
    pipeline.feature_specification = preprocessor
    pipeline.profile = profile
    return pipeline


def fit_evaluate(data, pipeline, model_name):
    """분할 → 학습 구간 CV/적합 → 보류 예측 → 수치 평가를 수행합니다.

    Args:
        data: 수치 특성 및 target 열의 DataFrame.
        pipeline: build_pipeline에서 만든 연결 객체.
        model_name: 출력에 사용할 모델명.
    Returns:
        SkeletonOutputDTO: 원 라벨 예측·행 위치·전처리를 포함한 모델.
    Raises:
        ValueError/TypeError: 자료·분할·CV·모델 적합 오류.
    """
    p = pipeline.profile
    X, y = data.drop(columns='target'), data.target
    classification = p.task == 'classification'
    train, test = train_test_split(np.arange(len(data)), test_size=p.test_size, random_state=p.random_state,
                                  stratify=y if classification and p.stratify else None)
    encoder = LabelEncoder().fit(y.iloc[train]) if classification else None
    train_y = encoder.transform(y.iloc[train]) if classification else y.iloc[train].to_numpy()
    best_params = {}
    if p.param_grid is not None:
        splitter = StratifiedKFold(p.cv, shuffle=True, random_state=p.random_state) if classification else KFold(p.cv, shuffle=True, random_state=p.random_state)
        grid = {}
        nested = getattr(pipeline.get_model(), 'named_steps', None)
        for key, values in p.param_grid.items():
            if nested is None:
                name = key if '__' in key else 'model__' + key
            elif key.startswith('preprocessor__'):
                name = key
            elif key.startswith('model__') and key.split('__')[1] in nested:
                name = key
            elif key.startswith('model__'):
                name = 'model__model__' + key[len('model__'):]
            else:
                name = 'model__' + key if '__' in key else 'model__model__' + key
            grid[name] = values
        search = GridSearchCV(pipeline.pipeline, grid, cv=splitter, scoring=p.scoring or ('accuracy' if classification else 'neg_mean_squared_error'), n_jobs=p.n_jobs, error_score='raise')
        search.fit(X.iloc[train], train_y)
        pipeline.pipeline = search.best_estimator_
        pipeline.preprocessor = pipeline.pipeline.named_steps['preprocessor']
        pipeline.model = pipeline.pipeline.named_steps['model']
        best_params = search.best_params_
    else:
        pipeline.pipeline.fit(X.iloc[train], train_y)
    prediction = pipeline.pipeline.predict(X.iloc[test])
    if classification:
        prediction = encoder.inverse_transform(np.asarray(prediction, dtype=int))
        metrics = {'accuracy': float(accuracy_score(y.iloc[test], prediction)), 'f1_weighted': float(f1_score(y.iloc[test], prediction, average='weighted'))}
    else:
        metrics = {'rmse': float(np.sqrt(mean_squared_error(y.iloc[test], prediction))), 'mae': float(mean_absolute_error(y.iloc[test], prediction)), 'r2': float(r2_score(y.iloc[test], prediction))}
    dataset = 'UCI Iris' if classification else 'UCI Wine Quality (red)'
    details = {'feature_names': X.columns.tolist(), 'preprocessor': pipeline.feature_specification.describe(), 'best_params': best_params, 'recipe': asdict(p)}
    final = pipeline.get_model()
    if hasattr(final, 'named_steps') and 'selector' in final.named_steps:
        details['selected_features'] = X.columns[final.named_steps['selector'].get_support()].tolist()
    return SkeletonOutputDTO(model_name, dataset, p.task, len(data), X.shape[1], metrics, details, pipeline, encoder, prediction, train, test)


class AnalysisEstimator(TransformerMixin, BaseEstimator):
    """기존 분석 runner를 Pipeline의 마지막 추정기로 연결합니다.

    Attributes:
        module_name (str): 기법 구현 모듈 경로.
        runner_name (str): 구현의 공개 실행 함수.
        profile (object): 외부 전처리와 중복 적합하지 않을 Profile.
        result_ (object): fit 후 기법별 Output DTO.
    """
    def __init__(self, module_name, runner_name, profile):
        """adapter 생성.

        Args:
            module_name: 기법 모듈.
            runner_name: 공개 runner.
            profile: 모델 Profile.
        Returns:
            None: 미적합 adapter.
        Raises:
            None: 생성 시 학습하지 않음.
        """
        self.module_name = module_name
        self.runner_name = runner_name
        self.profile = profile

    def fit(self, X, y=None):
        """이미 전처리한 자료에 기존 분석을 적합합니다.

        Args:
            X: 전처리한 특성.
            y: LDA의 분류 라벨; 다른 분석에서는 사용하지 않음.
        Returns:
            AnalysisEstimator: 적합한 자신.
        Raises:
            ValueError/ImportError: 기법 조건·라이브러리 오류.
        """
        runner = getattr(importlib.import_module(self.module_name), self.runner_name)
        profile = replace(self.profile, scaling='none')
        self.result_ = runner(X, y, profile=profile) if self.runner_name == 'run_lda' else runner(X, profile=profile)
        return self

    def fit_transform(self, X, y=None, **kwargs):
        """학습 embedding을 반환합니다. t-SNE/MDS의 새 입력 좌표는 만들지 않습니다.

        Args:
            X: 학습 특성.
            y: LDA 라벨 또는 None.
            kwargs: sklearn 호환 추가 인자.
        Returns:
            np.ndarray: 학습 embedding.
        Raises:
            ValueError/AttributeError: 변환을 지원하지 않는 분석.
        """
        self.fit(X, y)
        return self.result_.embedding

    def transform(self, X):
        """새 입력 변환을 지원하는 기법에 한해 재사용합니다.

        Args:
            X: 같은 특성 순서의 새 입력.
        Returns:
            np.ndarray: 새 입력 좌표.
        Raises:
            ValueError/AttributeError: 미지원 또는 미적합.
        """
        if not getattr(self.result_, 'supports_transform', False):
            raise ValueError('이 기법은 새 입력 transform을 지원하지 않습니다.')
        return self.result_.model.transform(X)


def report(result):
    """원 참고 코드처럼 수치 요약만 출력합니다.

    Args:
        result: SkeletonOutputDTO.
    Returns:
        SkeletonOutputDTO: 같은 결과 객체.
    Raises:
        TypeError: 잘못된 결과 구조.
    """
    print('=== Example : {} {} ==='.format(result.dataset_name, result.model_name))
    print('rows={}, features={}'.format(result.n_rows, result.n_features))
    print('metrics:', result.metrics)
    print('details:', result.details)
    return result
