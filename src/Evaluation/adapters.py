"""기존 학습 실행기와 독립 평가 DTO를 연결하는 호환 어댑터."""
import numpy as np
from src.Evaluation.classification.metrics import run_classification_metrics
from src.Evaluation.regression.metrics import run_regression_metrics


def evaluate_predictions(y_true, y_pred, task, probabilities=None, classes=None):
    """기존 scalar metrics 계약을 유지하면서 독립 평가 결과를 추가합니다.

    Args:
        y_true: 보류 정답.
        y_pred: 같은 행 순서 예측.
        task: classification/regression.
        probabilities: classes 열 순서의 예측 확률 또는 None.
        classes: 모델이 제공한 클래스 순서 또는 None.
    Returns:
        dict: 스칼라 지표와 기존 weighted 별칭; curves/행렬은 개별 DTO로 조회.
    Raises:
        ValueError: 정답/예측/확률 정렬 및 shape 오류.
    """
    if task == 'regression':
        result = run_regression_metrics(y_true, y_pred)
    elif task == 'classification':
        options = {'average':'weighted'}
        if classes is not None:
            options.update(labels=np.asarray(classes).tolist(), positive_label=np.asarray(classes)[-1].item() if isinstance(np.asarray(classes)[-1],np.generic) else np.asarray(classes)[-1])
        result = run_classification_metrics(y_true, y_pred, y_score=probabilities, overrides=options)
    else:
        raise ValueError('classification/regression만 지원합니다.')
    values={name:float(value) for name,value in result.value.items() if np.isscalar(value) and np.isfinite(value)}
    # Adjusted R² requires a deliberate explanatory-variable count and model interpretation.
    values.pop('adjusted_r2', None)
    if task == 'classification':
        for name in ('precision','recall','f1'):
            if name in values: values[name+'_weighted']=values[name]
        if probabilities is not None:
            for name in ('lift','gain'):
                table=result.details['metrics'].get(name,{}).get('top')
                if table is not None:
                    for _, row in table.iterrows():
                        values['{}@{:g}%'.format(name,100*row.requested_fraction)]=float(row[name])
    return values
